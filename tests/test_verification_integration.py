"""Live citation verification against real identifiers (network, `validation` extra).

Excluded from the default suite (``-m "not integration"``): it resolves real
identifiers over the NCBI/CrossRef APIs. Run with ``just test-full`` or
``pytest -m integration``.
"""

import asyncio

import pytest

pytest.importorskip("linkml_reference_validator")  # the `validation` extra

from deep_research_client.claims import (  # noqa: E402 - after importorskip
    AnchorStatus,
    CitationHandle,
    CitationScope,
    CitationStatus,
    CitationVerifier,
    Claim,
    ClaimBasis,
    ClaimTopic,
    VerificationStatus,
)


def _cited(cid: str, reference_id: str) -> Claim:
    return Claim(
        id=cid, claim_text="Marfan syndrome is caused by FBN1 variants.",
        anchor_status=AnchorStatus.EXACT, about=ClaimTopic.DOMAIN,
        basis=ClaimBasis.SECONDARY_SOURCE, citation_status=CitationStatus.CITED,
        citations=[CitationHandle(marker=reference_id, reference_id=reference_id,
                                  scope=CitationScope.SENTENCE)],
    )


@pytest.mark.integration
def test_a_real_pmid_resolves_to_a_supported_verdict() -> None:
    """A genuine PMID resolves; the verdict is SUPPORTED or NEEDS_REVIEW, never UNSUPPORTED."""
    verifier = CitationVerifier()
    verdicts = asyncio.run(verifier.averify(_cited("c1", "PMID:7981675")))  # a real FBN1/Marfan paper
    assert len(verdicts) == 1
    assert verdicts[0].reference_id == "PMID:7981675"
    assert verdicts[0].status in {
        VerificationStatus.SUPPORTED.value,
        VerificationStatus.NEEDS_REVIEW.value,
    }


@pytest.mark.integration
def test_a_confabulated_pmid_is_unsupported() -> None:
    """A syntactically valid but non-existent PMID resolves to nothing: UNSUPPORTED."""
    verifier = CitationVerifier()
    verdicts = asyncio.run(verifier.averify(_cited("c1", "PMID:999999999")))
    assert len(verdicts) == 1
    assert verdicts[0].status == VerificationStatus.UNSUPPORTED.value
