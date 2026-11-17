"""Deterministic publication-id normalization (issue #41)."""

import pytest

from deep_research_client.claims import (
    PublicationIdType,
    normalize_publication_id,
)


@pytest.mark.parametrize(
    ("raw", "canonical_id", "id_type", "confidence"),
    [
        ("PMID:7913883", "PMID:7913883", PublicationIdType.PMID, 1.0),
        ("PMID: 7913883", "PMID:7913883", PublicationIdType.PMID, 1.0),
        ("See PMC11000121.", "PMC:PMC11000121", PublicationIdType.PMCID, 1.0),
        ("doi:10.1038/ng1234", "DOI:10.1038/ng1234", PublicationIdType.DOI, 1.0),
        ("Deposited under GSE68086.", "GEO:GSE68086", PublicationIdType.GEO, 1.0),
        ("Registered as NCT00000419.", "NCT:NCT00000419", PublicationIdType.NCT, 1.0),
        ("nct04280705", "NCT:NCT04280705", PublicationIdType.NCT, 1.0),
    ],
)
def test_clean_identifiers_resolve_with_full_confidence(
    raw: str, canonical_id: str, id_type: PublicationIdType, confidence: float
) -> None:
    result = normalize_publication_id(raw)
    assert result.canonical_id == canonical_id
    assert result.id_type == id_type
    assert result.confidence == confidence
    assert result.resolved is True
    assert result.raw == raw  # the raw string is always kept


def test_identifier_recovered_from_url_is_less_confident() -> None:
    result = normalize_publication_id("https://pubmed.ncbi.nlm.nih.gov/7913883")
    assert result.canonical_id == "PMID:7913883"
    assert result.id_type == PublicationIdType.PMID
    assert result.confidence == 0.7  # one indirection from the written citation
    assert result.source_url == "https://pubmed.ncbi.nlm.nih.gov/7913883"


def test_equivalent_forms_resolve_to_one_canonical_id() -> None:
    """A written PMID and the matching PubMed URL normalise to the same id."""
    written = normalize_publication_id("PMID:7913883")
    from_url = normalize_publication_id("https://pubmed.ncbi.nlm.nih.gov/7913883")
    assert written.canonical_id == from_url.canonical_id == "PMID:7913883"


def test_a_bare_url_resolves_weakly_to_the_url() -> None:
    result = normalize_publication_id("see https://example.org/paper for details")
    assert result.id_type == PublicationIdType.URL
    assert result.canonical_id == "https://example.org/paper"
    assert result.confidence == 0.5
    assert result.source_url == "https://example.org/paper"


@pytest.mark.parametrize(
    "raw",
    [
        "(see https://example.org/paper).",
        "https://example.org/paper,",
        "available at https://example.org/paper;",
        "[https://example.org/paper]",
    ],
)
def test_a_bare_url_drops_trailing_prose_punctuation(raw: str) -> None:
    """Punctuation abutting a URL in prose is not part of the URL."""
    result = normalize_publication_id(raw)
    assert result.canonical_id == "https://example.org/paper"


def test_a_doi_url_normalises_to_the_doi_not_a_bare_url() -> None:
    """A DOI expressed as a URL resolves to the DOI id, the same as the written form."""
    result = normalize_publication_id("https://doi.org/10.1038/ng1234")
    assert result.canonical_id == "DOI:10.1038/ng1234"
    assert result.id_type == PublicationIdType.DOI


@pytest.mark.parametrize(
    "raw",
    ["Smith et al. 2020, Nature", "[3]", "personal communication", ""],
)
def test_unresolvable_strings_are_kept_not_dropped(raw: str) -> None:
    """#41: preserve unresolved raw strings with a confidence rather than dropping them."""
    result = normalize_publication_id(raw)
    assert result.resolved is False
    assert result.canonical_id is None
    assert result.id_type == PublicationIdType.UNKNOWN
    assert result.confidence == 0.0
    assert result.raw == raw


def test_the_first_identifier_wins_when_a_string_has_several() -> None:
    result = normalize_publication_id("PMID:7913883 (see also DOI:10.1038/ng1234)")
    assert result.canonical_id == "PMID:7913883"
