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


def test_relevance_needs_report_level_text_not_one_claim() -> None:
    """Why the verifier weighs relevance against the report, not a single claim.

    The keyword heuristic keeps only terms a text repeats (MIN_TERM_OCCURRENCES),
    so one claim sentence yields no keywords and relevance can never be assessed --
    the regression that passing claim.claim_text as topic_text caused. This runs
    the real extract_keywords/assess_relevance path (no mocks) so it cannot quietly
    regress again.
    """
    from deep_research_client.validation.relevance import assess_relevance, extract_keywords

    one_sentence = "Marfan syndrome is caused by FBN1 variants."
    assert extract_keywords(one_sentence) == []  # no term repeats -> nothing to weigh

    report = (
        "Marfan syndrome is a connective tissue disorder. "
        "Marfan syndrome is caused by FBN1 variants. "
        "FBN1 encodes fibrillin. Fibrillin fibres weaken in Marfan syndrome. "
        "FBN1 variants disrupt fibrillin assembly in connective tissue."
    )
    keywords = extract_keywords(report)
    assert keywords  # a report's repeated vocabulary gives the heuristic something

    on_topic = "FBN1 variants cause Marfan syndrome by disrupting fibrillin. " * 8
    off_topic = "Pollen development in Arabidopsis thaliana anthers. " * 8
    assert assess_relevance(keywords, on_topic, body=on_topic).relevance == TopicalRelevance.ON_TOPIC
    assert assess_relevance(keywords, off_topic, body=off_topic).relevance == TopicalRelevance.OFF_TOPIC


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
