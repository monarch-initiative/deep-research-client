"""The citation verifier's verdict mapping and its empty-claim fast path.

The pure mapping is exercised over real ReferenceCheck instances (the generated
model, not a mock): a check is plain data, so building one and asserting the
verdict needs no network and no ``validation`` extra. The live resolution path is
covered by test_verification_integration.py.
"""

import pytest

from deep_research_client.claims import (
    AnchorStatus,
    CitationHandle,
    CitationScope,
    CitationStatus,
    Claim,
    ClaimBasis,
    ClaimTopic,
    VerificationStatus,
    verdict_from_check,
)
from deep_research_client.validation.datamodel import (
    ReferenceCheck,
    ReferenceStatus,
    TopicalRelevance,
)


def test_not_found_identifier_is_unsupported() -> None:
    check = ReferenceCheck(reference_id="PMID:999999999", status=ReferenceStatus.NOT_FOUND)
    verdict = verdict_from_check(check, "c1")
    assert verdict.status == VerificationStatus.UNSUPPORTED.value
    assert verdict.reference_id == "PMID:999999999"
    assert verdict.confidence == 0.9
    assert verdict.verifier == "citation"


def test_unverifiable_identifier_is_unverifiable() -> None:
    check = ReferenceCheck(reference_id="ISBN:123", status=ReferenceStatus.UNVERIFIABLE,
                           message="no resolver for prefix")
    verdict = verdict_from_check(check, "c1")
    assert verdict.status == VerificationStatus.UNVERIFIABLE.value
    assert verdict.note == "no resolver for prefix"


@pytest.mark.parametrize(
    "relevance",
    [TopicalRelevance.ON_TOPIC, TopicalRelevance.UNCERTAIN, TopicalRelevance.NOT_ASSESSED],
)
def test_verified_and_on_or_unweighed_topic_is_supported(relevance: TopicalRelevance) -> None:
    check = ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED, relevance=relevance)
    verdict = verdict_from_check(check, "c1")
    assert verdict.status == VerificationStatus.SUPPORTED.value
    # On topic is more confident than merely resolved.
    expected = 0.8 if relevance == TopicalRelevance.ON_TOPIC else 0.6
    assert verdict.confidence == expected


def test_verified_but_off_topic_needs_review() -> None:
    check = ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED,
                           relevance=TopicalRelevance.OFF_TOPIC, relevance_score=0.05)
    verdict = verdict_from_check(check, "c1")
    assert verdict.status == VerificationStatus.NEEDS_REVIEW.value
    assert verdict.method == "citation:relevance"


def test_averify_returns_nothing_for_an_uncited_claim_without_building_the_validator() -> None:
    """An uncited claim needs no resolver, so the optional extra is never touched."""
    from deep_research_client.claims import CitationVerifier

    verifier = CitationVerifier()
    uncited = Claim(id="c1", claim_text="X drives Y.", anchor_status=AnchorStatus.EXACT,
                    about=ClaimTopic.DOMAIN, basis=ClaimBasis.OBSERVATION,
                    citation_status=CitationStatus.UNCITED)
    import asyncio

    assert asyncio.run(verifier.averify(uncited)) == []
    assert verifier._validator is None  # never built: no network, no extra required


def test_averify_skips_citations_that_resolve_to_no_identifier() -> None:
    """A cited claim whose markers carry no reference_id gives the verifier nothing."""
    from deep_research_client.claims import CitationVerifier

    verifier = CitationVerifier()
    cited_unresolved = Claim(
        id="c1", claim_text="X.", anchor_status=AnchorStatus.EXACT, about=ClaimTopic.DOMAIN,
        basis=ClaimBasis.SECONDARY_SOURCE, citation_status=CitationStatus.CITED,
        citations=[CitationHandle(marker="[1]", scope=CitationScope.SENTENCE)],  # no reference_id
    )
    import asyncio

    assert asyncio.run(verifier.averify(cited_unresolved)) == []
    assert verifier._validator is None
