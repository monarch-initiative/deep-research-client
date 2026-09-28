"""The claims datamodel and its generated code (issue #43)."""

from pathlib import Path

import pytest

from deep_research_client.claims import (
    AnchorStatus,
    CitationHandle,
    CitationScope,
    CitationStatus,
    Claim,
    ClaimBasis,
    ClaimSet,
    ClaimTopic,
    ExtractorInfo,
    NearestPassage,
    SourceDocument,
    SourceType,
    TextSpan,
)


def test_datamodel_matches_linkml_schema() -> None:
    """claims/datamodel.py is generated; regenerate it with `just gen-datamodel`."""
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    schema = Path("src/deep_research_client/claims/claims.yaml")
    generated = repo_root / "src/deep_research_client/claims/datamodel.py"

    gen_pydantic = shutil.which("gen-pydantic", path=str(Path(sys.executable).parent))
    if not gen_pydantic:
        pytest.skip("linkml is not installed; install the dev dependency group to check drift")

    completed = subprocess.run(
        [gen_pydantic, str(schema)], cwd=repo_root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, f"gen-pydantic failed:\n{completed.stderr}"
    assert completed.stdout == generated.read_text(encoding="utf-8"), (
        "claims/datamodel.py does not match claims.yaml. Either the schema changed "
        "or linkml was upgraded; run `just gen-datamodel` and review the diff."
    )


def test_a_claim_set_round_trips_through_json() -> None:
    """The set is the unit written to disk and read back by alignment later."""
    claims = ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="llm-atomic", model="m", prompt_version="1"),
        claims=[Claim(
            id="c1", claim_text="A causes B.", anchor_status=AnchorStatus.EXACT,
            source_span=TextSpan(start=4, end=15, text="A causes B."),
            citations=[CitationHandle(marker="[1]", scope=CitationScope.SENTENCE)], citation_status=CitationStatus.CITED,
            about=ClaimTopic.DOMAIN, basis=ClaimBasis.SECONDARY_SOURCE,
        )],
    )

    again = ClaimSet.model_validate_json(claims.model_dump_json())

    assert again == claims
    assert again.claim_list[0].anchor_status == AnchorStatus.EXACT
    assert again.claim_list[0].basis == ClaimBasis.SECONDARY_SOURCE


def test_a_moved_span_is_reported() -> None:
    """A span whose recorded text no longer matches the source is caught."""
    claims = ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="x"),
        claims=[Claim(
            id="c1", claim_text="A causes B.", anchor_status=AnchorStatus.EXACT,
            source_span=TextSpan(start=0, end=11, text="A causes B."),
            citation_status=CitationStatus.UNCITED,
        )],
    )

    assert claims.mismatched_spans("A causes B.") == []
    assert [c.id for c in claims.mismatched_spans("Now: A causes B.")] == ["c1"]


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"anchor_status": AnchorStatus.EXACT, "citation_status": CitationStatus.UNCITED,
          "citations": [CitationHandle(marker="[1]", scope=CitationScope.SENTENCE)]}, "c1 is UNCITED but has citations"),
        ({"anchor_status": AnchorStatus.EXACT, "citation_status": CitationStatus.CITED},
         "c1 is CITED but has no citations"),
        ({"anchor_status": AnchorStatus.UNANCHORED, "citation_status": CitationStatus.UNCITED},
         "c1 is UNCITED but is UNANCHORED"),
        ({"anchor_status": AnchorStatus.EXACT, "citation_status": CitationStatus.UNKNOWN},
         "c1 is UNKNOWN but has no citations"),
        ({"anchor_status": AnchorStatus.EXACT, "citation_status": CitationStatus.UNCITED,
          "about": ClaimTopic.WORK, "basis": ClaimBasis.OBSERVATION},
         "c1 has a basis, which only a DOMAIN claim has"),
        ({"anchor_status": AnchorStatus.EXACT, "citation_status": CitationStatus.UNCITED,
          "basis": ClaimBasis.BACKGROUND_KNOWLEDGE},
         "c1 has a basis, which only a DOMAIN claim has"),
        ({"anchor_status": AnchorStatus.EXACT, "citation_status": CitationStatus.UNCITED,
          "nearest_passage": NearestPassage(text="A causes B", score=90.0)},
         "c1 has a nearest passage, which only an UNANCHORED claim has"),
    ],
)
def test_a_claim_whose_slots_contradict_each_other_is_refused(fields, problem) -> None:
    """An uncited claim with citations, or a basis on a work claim, cannot be written."""
    with pytest.raises(ValueError, match=problem):
        ClaimSet(
            source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
            extractor=ExtractorInfo(name="x"),
            claims=[Claim(id="c1", claim_text="A causes B.", **fields)],
        )


def test_a_saved_set_that_contradicts_itself_does_not_load() -> None:
    """The rules hold when reading back as well as when building."""
    saved = (
        '{"source": {"id": "r.md", "source_type": "MARKDOWN_REPORT"},'
        ' "extractor": {"name": "x"},'
        ' "claims": [{"id": "c1", "claim_text": "A.", "anchor_status": "EXACT",'
        ' "citation_status": "CITED"}]}'
    )

    with pytest.raises(ValueError, match="c1 is CITED but has no citations"):
        ClaimSet.model_validate_json(saved)


def test_a_cited_claim_called_background_knowledge_is_listed_not_refused() -> None:
    """The two answers contradict each other; which one is wrong is the model's, not the code's."""
    cited = [CitationHandle(marker="[2]", scope=CitationScope.SENTENCE)]
    claims = ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="x"),
        claims=[
            Claim(id="c1", claim_text="A.", anchor_status=AnchorStatus.EXACT, citations=cited,
                  citation_status=CitationStatus.CITED, about=ClaimTopic.DOMAIN,
                  basis=ClaimBasis.BACKGROUND_KNOWLEDGE),
            Claim(id="c2", claim_text="B.", anchor_status=AnchorStatus.EXACT, citations=cited,
                  citation_status=CitationStatus.CITED, about=ClaimTopic.DOMAIN,
                  basis=ClaimBasis.SECONDARY_SOURCE),
        ],
    )

    assert [c.id for c in claims.cited_background_claims] == ["c1"]


def _claims_schema_view():
    """The claims schema, as linkml-reference-validator's plugin reads it."""
    from linkml_runtime.utils.schemaview import SchemaView

    return SchemaView(str(Path(__file__).resolve().parent.parent
                          / "src/deep_research_client/claims/claims.yaml"))


@pytest.mark.parametrize(
    ("class_name", "slot_name", "kind"),
    [
        ("TextSpan", "text", "excerpt"),
        ("CitationHandle", "reference_id", "reference"),
        ("SourceDocument", "title", "title"),
    ],
)
def test_linkml_reference_validator_finds_the_fields_it_knows(class_name, slot_name, kind) -> None:
    """The slot URIs are the ones its field detection looks for."""
    from linkml_reference_validator.field_detection import (
        is_excerpt_slot,
        is_reference_slot,
        is_title_slot,
    )

    view = _claims_schema_view()
    detect = {"excerpt": is_excerpt_slot, "reference": is_reference_slot, "title": is_title_slot}

    assert detect[kind](view.induced_slot(slot_name, class_name))
    others = [k for k in detect if k != kind]
    assert not any(detect[k](view.induced_slot(slot_name, class_name)) for k in others)


def test_no_class_pairs_a_source_quote_with_a_cited_reference() -> None:
    """The validator checks an excerpt against a reference only within one class.

    A span is the source's own words and a citation is the work cited for
    them, so no class may hold both: that would ask the validator whether a
    report's sentence is a quote from the paper it cites.
    """
    from linkml_reference_validator.field_detection import is_excerpt_slot, is_reference_slot

    view = _claims_schema_view()
    for class_name in view.all_classes():
        slots = [view.induced_slot(s, class_name) for s in view.class_slots(class_name)]
        assert not (any(is_excerpt_slot(s) for s in slots) and any(is_reference_slot(s) for s in slots)), class_name
