"""Splitting sources into units and extracting claims from them (issue #43).

Curated structured sources need no model, so they are tested end to end. The
LLM path is tested here up to the request -- units, prompt, the refusal to run
without a client -- and end to end against a real model under the ``llm``
marker.
"""

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from deep_research_client.claims import (
    AnchorStatus,
    CitationStatus,
    ClaimBasis,
    ClaimTopic,
    SourceType,
)
from deep_research_client.claims.extract import (
    SourceFormat,
    aextract_claims,
    detect_format,
    extract_claims,
)
from deep_research_client.claims.llm import build_prompt
from deep_research_client.claims.models import content_sha256
from deep_research_client.claims.parsing import claims_from_reply
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


def test_a_markdown_file_read_as_structured_is_refused_not_read_as_empty():
    """Forcing a format the file is not must fail, not return a claim set with no claims."""
    with pytest.raises(ValueError, match="not a YAML or JSON document"):
        extract_claims(REPORT, source_format=SourceFormat.STRUCTURED)


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
    assert all(c.about == ClaimTopic.DOMAIN for c in claims.claim_list)
    assert (mechanism.citation_status, mechanism.basis) == (
        CitationStatus.CITED, ClaimBasis.SECONDARY_SOURCE,
    ), "evidence is a cited work"
    assert (ectopia.citation_status, ectopia.basis) == (CitationStatus.UNCITED, None), (
        "a record with no evidence says nothing about what it rests on"
    )


def test_a_gene_review_file_skips_removed_annotations_and_file_references():
    """The same records the evaluation loader reads, with paths."""
    claims = extract_claims(INPUT / "fbn1_gene_review.yaml")

    assert claims.extractor.name == "gene-review"
    assert [c.source_path for c in claims.claim_list] == [
        "existing_annotations[0]", "core_functions[0]",
    ]
    accepted = claims.claim_list[0]
    assert (accepted.subject.label, accepted.subject.id) == ("FBN1", "UniProtKB:P35555")
    assert [c.marker for c in accepted.citations] == ["PMID:1852208"]


def test_a_claim_set_serialises_for_the_next_step():
    """Alignment reads these back from disk, so the file must round-trip."""
    from deep_research_client.claims import ClaimSet

    claims = extract_claims(INPUT / "marfan_dismech.yaml")

    assert ClaimSet.model_validate_json(claims.model_dump_json()) == claims


def _real_client(backend: str) -> tuple[Any, str]:
    """A client and model for one backend, or a skip when it cannot run here."""
    if backend == "claude-code":
        if shutil.which("claude") is None:
            pytest.skip("needs the `claude` CLI on PATH")
        from deep_research_client.claude_code_chat import (
            DEFAULT_CLAUDE_CODE_MODEL,
            ClaudeCodeChatClient,
        )

        return ClaudeCodeChatClient(), os.getenv("CLAIMS_CLAUDE_MODEL", DEFAULT_CLAUDE_CODE_MODEL)
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("needs OPENAI_API_KEY")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    return client, os.getenv("CLAIMS_MODEL", "gpt-4o-mini")


@pytest.mark.llm
@pytest.mark.parametrize("backend", ["openai", "claude-code"])
async def test_a_real_model_decomposes_the_report_into_anchored_claims(backend):
    """End to end against a real model.

    The openai backend needs OPENAI_API_KEY (and optionally OPENAI_BASE_URL,
    CLAIMS_MODEL). The claude-code backend needs a logged-in `claude` CLI
    (and optionally CLAIMS_CLAUDE_MODEL). Asserts structure, not wording:
    most claims anchor, every span reads as the source does, and the negated
    TGFBR2 statement is marked negated.
    """
    client, model = _real_client(backend)
    text = REPORT.read_text(encoding="utf-8")

    claims = await aextract_claims(REPORT, llm_client=client, model=model)

    # The model that answered, not the name asked for: "sonnet" is an alias.
    assert claims.extractor.model and claims.extractor.model != "sonnet"

    assert len(claims.claim_list) >= 5
    anchored = [c for c in claims.claim_list if c.source_span is not None]
    assert len(anchored) >= 0.8 * len(claims.claim_list)
    assert claims.mismatched_spans(text) == []
    assert any(c.negated for c in claims.claim_list if "TGFBR2" in c.claim_text)


def test_a_paper_s_identifier_comes_from_its_own_lines_not_a_subsection_s():
    """A "Related work" list under a paper names other papers; it does not identify this one."""
    listing = (
        "# Papers\n\n### [1] A study\n\n- Year: 2025\n\n"
        "#### Related work\n\n- PMID: 12345678\n\nMice live longer.\n"
    )

    units = markdown_units(listing)

    assert [u.section for u in units] == ["Papers > [1] A study", "Papers > [1] A study > Related work"]
    assert all(u.section_citation.marker == "[1]" for u in units), "the subsection is still paper 1's"
    assert all(u.section_citation.reference_id is None for u in units)


def test_a_provider_s_own_reference_list_resolves_its_markers():
    """With no client Citations section, [n] resolves through the provider's References."""
    report = (
        "## Output\n\n# Marfan\n\nMarfan syndrome is caused by FBN1 variants [1].\n\n"
        "### References\n\n1. Dietz HC et al. Nature. 1991. PMID: 1852208\n"
    )
    (unit,) = markdown_units(report)
    reply = ('{"claims": [{"claim": "FBN1 variants cause Marfan syndrome.",'
             ' "quote": "Marfan syndrome is caused by FBN1 variants", "citations": ["[1]"]}]}')

    (claim,) = claims_from_reply(reply, unit)

    assert [(c.marker, c.reference_id) for c in claim.citations] == [("[1]", "PMID:1852208")]


def test_a_long_table_is_split_between_rows_and_each_piece_keeps_its_header():
    """A table has no paragraph breaks; sent whole, a 12 KB one outran every reply budget."""
    rows = "".join(f"| Phenotype {i} | {i}% | Review {i} |\n" for i in range(300))
    report = "## Output\n\n# Report\n\n## Phenotypes\n\n| Phenotype | Frequency | Source |\n|---|---|---|\n" + rows

    units = markdown_units(report, max_chars=2000)

    assert len(units) > 1 and all(len(u.body) <= 2000 for u in units)
    assert units[0].context is None, "the first piece holds its own header"
    assert {u.context for u in units[1:]} == {"| Phenotype | Frequency | Source |\n|---|---|---|"}
    user = build_prompt(units[1])[1]["content"]
    header_at, text_at = user.index("| Phenotype | Frequency"), user.index("TEXT:")
    assert header_at < text_at and "| Phenotype | Frequency" not in user[text_at:]

    reply = ('{"claims": [{"claim": "The table has a Frequency column.",'
             ' "quote": "| Phenotype | Frequency | Source |"}]}')
    (claim,) = claims_from_reply(reply, units[1])
    assert claim.anchor_status == AnchorStatus.UNANCHORED, "the header is context, not the unit's text"


def test_a_reference_list_is_known_by_its_entries_not_only_its_name():
    """Falcon's "Key references" list is skipped; its "Evidence sources" prose is not."""
    report = (
        "## Output\n\n# Report\n\n### 1.4 Evidence sources (patient-level vs aggregated)\n\n"
        "Most evidence comes from aggregated cohort studies.\n\n"
        "## Key references (URLs in evidence)\n"
        "- Mustillo et al., 2023. https://doi.org/10.1007/s10875-022-01418-y\n"
        "- Biggs et al., 2023. https://doi.org/10.1007/s11882-023-01071-4\n"
    )

    assert [u.section for u in markdown_units(report)] == [
        "Report > 1.4 Evidence sources (patient-level vs aggregated)",
    ]
