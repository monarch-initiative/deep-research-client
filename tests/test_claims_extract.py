"""Splitting sources into units and extracting claims from them (issue #43).

Curated structured sources need no model, so they are tested end to end. The
LLM path is tested here up to the request -- units, prompt, the refusal to run
without a client -- and end to end against a real model under the ``llm``
marker.
"""

import os
from pathlib import Path

import pytest

from deep_research_client.claims import AnchorStatus, SourceType
from deep_research_client.claims.extract import (
    SourceFormat,
    aextract_claims,
    detect_format,
    extract_claims,
)
from deep_research_client.claims.llm import build_prompt
from deep_research_client.claims.models import content_sha256
from deep_research_client.claims.units import markdown_units, structured_units

INPUT = Path(__file__).parent / "input" / "claims"
REPORT = INPUT / "marfan_report.md"


def test_report_units_cover_the_answer_and_skip_what_the_client_added():
    """The question, bibliography and generated sections are not the source's claims."""
    text = REPORT.read_text(encoding="utf-8")

    units = markdown_units(text)

    assert [u.section for u in units] == [
        "Marfan syndrome > Genetics",
        "Marfan syndrome > Clinical features",
        "Marfan syndrome > Management",
    ]
    bodies = "".join(u.body for u in units)
    assert "What causes Marfan syndrome" not in bodies, "the question is the user's"
    assert "Dietz HC" not in bodies, "the bibliography is not a claim"
    assert all(u.text is text for u in units), "offsets index the whole file"
    assert units[0].bibliography[3].startswith("Shores J")


def test_a_long_section_is_split_at_paragraph_breaks_keeping_its_heading():
    """Every piece stays within budget and knows which section it came from."""
    paragraphs = "\n\n".join(f"Paragraph {n} says something specific." for n in range(40))
    text = f"# Long\n\n{paragraphs}\n"

    units = markdown_units(text, max_chars=300)

    assert len(units) > 1
    assert {u.section for u in units} == {"Long"}
    assert all(len(u.body) <= 300 for u in units)
    assert "".join(u.body for u in units).strip() == paragraphs


def test_a_comment_in_a_code_block_is_not_a_heading():
    """A "#" line inside a fence is code, so the section is not split there."""
    text = "# Setup\n\nInstall it:\n\n```bash\n# install the package\npip install x\n```\n\nThen run it.\n"

    units = markdown_units(text)

    assert [u.section for u in units] == ["Setup"]
    assert "Then run it." in units[0].body


@pytest.mark.parametrize("heading", ["## References", "### Sources", "## Bibliography:"])
def test_a_reference_list_the_provider_wrote_is_not_decomposed(heading):
    """A provider's own reference list, and anything under it, names sources rather than claims."""
    level = heading.split()[0]
    text = (
        f"# Marfan syndrome\n\nFBN1 variants cause it [1].\n\n{heading}\n\n"
        f"1. Dietz HC. Nature. 1991.\n\n{level}# Reviews\n\n2. Judge DP. Lancet. 2005.\n\n"
        f"# Outlook\n\nLosartan is under study.\n"
    )

    units = markdown_units(text)

    assert [u.section for u in units] == ["Marfan syndrome", "Outlook"]


def test_structured_units_are_the_prose_fields_by_path():
    """Identifiers and short labels are not prose, so they are not sent."""
    import yaml

    data = yaml.safe_load((INPUT / "generic_notes.yaml").read_text(encoding="utf-8"))

    units = structured_units(data)

    assert [u.source_path for u in units] == ["sections[0].text", "sections[1].text"]
    assert units[0].body.startswith("Marfan syndrome is caused by")


def test_the_prompt_carries_the_unit_verbatim_and_its_place():
    """The model sees exactly the text it must quote from."""
    unit = markdown_units(REPORT.read_text(encoding="utf-8"))[0]

    system, user = build_prompt(unit)

    assert '"quote"' in system["content"]
    assert user["content"].startswith("Section: Marfan syndrome > Genetics")
    assert unit.body in user["content"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("marfan_report.md", SourceFormat.MARKDOWN),
        ("marfan_dismech.yaml", SourceFormat.DISMECH),
        ("fbn1_gene_review.yaml", SourceFormat.GENE_REVIEW),
        ("generic_notes.yaml", SourceFormat.STRUCTURED),
    ],
)
def test_the_format_is_detected_from_the_file(name, expected):
    """Each fixture is read the way its content calls for."""
    import yaml

    path = INPUT / name
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix == ".yaml" else None

    assert detect_format(path, data) == expected


@pytest.mark.parametrize("name", ["marfan_report.md", "generic_notes.yaml"])
def test_prose_sources_refuse_to_run_without_a_model(name):
    """There is no model-free decomposition to fall back to, so it is an error."""
    with pytest.raises(ValueError, match="needs an LLM client"):
        extract_claims(INPUT / name)


def test_a_dismech_file_maps_record_by_record_with_paths():
    """Curated records are already claims; each keeps its place in the file."""
    path = INPUT / "marfan_dismech.yaml"

    claims = extract_claims(path)

    assert claims.extractor.name == "dismech"
    assert claims.source.source_type == SourceType.STRUCTURED_DOCUMENT
    assert claims.source.content_sha256 == content_sha256(path.read_text(encoding="utf-8"))
    assert claims.source.title == "Marfan syndrome"
    assert [c.source_path for c in claims.claim_list] == [
        "pathophysiology[0]", "phenotypes[1]", "phenotypes[2]", "treatments[0]", "inheritance[0]",
    ], "phenotypes[0] is not a record, and the index still counts it"
    assert all(c.anchor_status == AnchorStatus.NOT_APPLICABLE for c in claims.claim_list)
    assert {c.subject.id for c in claims.claim_list} == {"MONDO:0007947"}

    by_path = {c.source_path: c for c in claims.claim_list}
    ectopia = by_path["phenotypes[2]"]
    assert ectopia.claim_text == "Marfan syndrome has phenotype Ectopia lentis."
    assert (ectopia.predicate.label, ectopia.object.id) == ("has phenotype", "HP:0001083")
    mechanism = by_path["pathophysiology[0]"]
    assert {e.id for e in mechanism.entities} == {"GO:0048251", "HGNC:3603"}
    assert [(c.marker, c.reference_id) for c in mechanism.citations] == [
        ("PMID:1852208", "PMID:1852208"),
    ]


def test_a_gene_review_file_skips_removed_annotations_and_file_references():
    """The same records the evaluation loader reads, with paths."""
    claims = extract_claims(INPUT / "fbn1_gene_review.yaml")

    assert claims.extractor.name == "gene-review"
    assert [c.source_path for c in claims.claim_list] == [
        "existing_annotations[0]", "core_functions[0]",
    ]
    accepted = claims.claim_list[0]
    assert accepted.subject.label == "FBN1"
    assert [c.marker for c in accepted.citations] == ["PMID:1852208"]


def test_a_claim_set_serialises_for_the_next_step():
    """Alignment reads these back from disk, so the file must round-trip."""
    from deep_research_client.claims import ClaimSet

    claims = extract_claims(INPUT / "marfan_dismech.yaml")

    assert ClaimSet.model_validate_json(claims.model_dump_json()) == claims


@pytest.mark.llm
async def test_a_real_model_decomposes_the_report_into_anchored_claims():
    """End to end against a real OpenAI-compatible model.

    Needs OPENAI_API_KEY (and optionally OPENAI_BASE_URL, CLAIMS_MODEL).
    Asserts structure, not wording: most claims anchor, every span reads as
    the source does, and the negated TGFBR2 statement is marked negated.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("needs OPENAI_API_KEY")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    text = REPORT.read_text(encoding="utf-8")

    claims = await aextract_claims(REPORT, llm_client=client, model=os.getenv("CLAIMS_MODEL", "gpt-4o-mini"))

    assert len(claims.claim_list) >= 5
    anchored = [c for c in claims.claim_list if c.source_span is not None]
    assert len(anchored) >= 0.8 * len(claims.claim_list)
    assert claims.mismatched_spans(text) == []
    assert any(c.negated for c in claims.claim_list if "TGFBR2" in c.claim_text)
