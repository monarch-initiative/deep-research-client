"""The OAK backend's search, against a tiny local ontology (no network, no model).

A three-term ``.obo`` read through OAK's ``simpleobo:`` adapter exercises candidate
gathering, labels, optional definitions, dedup, the empty-adapter skip and the
failed-search counters, without reaching a service or a model. Skipped when the
optional ``terms`` extra is not installed.
"""

from pathlib import Path

import pytest

pytest.importorskip("linkml_term_validator")  # the `terms` extra (OAK)

from deep_research_client.claims import (  # noqa: E402 - after importorskip
    AnchorStatus,
    CitationStatus,
    Claim,
    ClaimSet,
    EntityMention,
    ExtractorInfo,
    OakGrounder,
    SourceDocument,
    SourceType,
    ground_claims,
)
from deep_research_client.claims.grounding_oak import GroundingError  # noqa: E402
from deep_research_client.ontologies import OntologyConfig  # noqa: E402

OBO = f"simpleobo:{Path(__file__).parent / 'input' / 'claims' / 'grounding' / 'tiny.obo'}"


def _grounder(adapters, **kw) -> OakGrounder:
    return OakGrounder(OntologyConfig(ontology_adapters=adapters), llm_client=None, model="x", **kw)


def test_search_gathers_candidates_from_a_local_ontology() -> None:
    grounder = _grounder({"MONDO": OBO}, include_definitions=True)
    found = {c.id: c for c in grounder._candidates("Marfan syndrome")}
    assert "MONDO:0007947" in found
    assert found["MONDO:0007947"].label == "Marfan syndrome"
    assert found["MONDO:0007947"].definition  # fetched because include_definitions
    assert (grounder.searches_ok, grounder.searches_failed) == (1, 0)


def test_definitions_are_off_by_default() -> None:
    grounder = _grounder({"MONDO": OBO})
    candidates = grounder._candidates("Marfan syndrome")
    assert candidates and all(c.definition is None for c in candidates)


def test_candidates_are_deduped_across_ontologies() -> None:
    """Two prefixes over the same file find the same term; it is kept once."""
    grounder = _grounder({"MONDO": OBO, "HP": OBO})
    ids = [c.id for c in grounder._candidates("Marfan syndrome")]
    assert ids.count("MONDO:0007947") == 1


def test_an_empty_adapter_string_is_not_searched() -> None:
    grounder = _grounder({"MONDO": OBO, "NCIT": ""})
    grounder._candidates("Marfan syndrome")
    assert grounder.searches_failed == 0  # NCIT was skipped, not searched and failed


#: Three ontologies whose adapters all point at missing files, so every search fails.
_ALL_BAD = {"A": "simpleobo:/no/a.obo", "B": "simpleobo:/no/b.obo", "C": "simpleobo:/no/c.obo"}


def test_a_single_transient_failure_does_not_abort_the_run() -> None:
    """One failed search on a one-ontology config is tolerated (returns nothing), not fatal."""
    grounder = _grounder({"X": "simpleobo:/no/such/file.obo"})
    assert grounder._candidates("anything") == []  # no raise
    assert (grounder.searches_ok, grounder.searches_failed) == (0, 1)


def test_a_total_outage_raises_rather_than_returning_nothing() -> None:
    """Once enough searches have failed with none succeeding, it is an outage, not empty."""
    grounder = _grounder(_ALL_BAD)
    with pytest.raises(GroundingError, match="Every ontology search"):
        grounder._candidates("anything")
    assert (grounder.searches_ok, grounder.searches_failed) == (0, 3)  # exactly the floor


def test_a_total_outage_fails_fast_without_calling_the_model() -> None:
    """ground_claims raises before any candidate, so the model (llm_client=None) is never reached."""
    claim_set = ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="x"),
        claims=[Claim(id="c1", claim_text="X.", anchor_status=AnchorStatus.EXACT,
                      citation_status=CitationStatus.UNCITED, entities=[EntityMention(label="acne")])],
    )
    grounder = _grounder(_ALL_BAD)
    with pytest.raises(GroundingError):
        ground_claims(claim_set, grounder)
