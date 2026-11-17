"""Verification orchestration: triage plus pluggable verifiers (issue #43, item 5).

The verifier seam is exercised with an injected fake verifier -- a real
implementation of the Verifier protocol, as grounding's tests inject a real fake
Grounder -- not a mock. Nothing here reaches a network or a model.
"""

import json
from pathlib import Path

from deep_research_client.claims import (
    AnchorStatus,
    CitationHandle,
    CitationScope,
    CitationStatus,
    Claim,
    ClaimAlignment,
    ClaimBasis,
    ClaimSet,
    ClaimTopic,
    ClaimVerification,
    FlagKind,
    VerificationReport,
    VerificationStatus,
    verify_claims,
)


class FakeVerifier:
    """A Verifier that returns a fixed verdict for named claims, nothing for the rest."""

    def __init__(self, verdicts: dict[str, VerificationStatus], name: str = "fake") -> None:
        self.name = name
        self._verdicts = verdicts

    async def averify(self, claim: Claim) -> list[ClaimVerification]:
        status = self._verdicts.get(claim.id)
        if status is None:
            return []
        return [ClaimVerification(claim_id=claim.id, status=status, method="fake:test",
                                  verifier=self.name)]


def _claim(cid, text="X drives Y.", *, about=ClaimTopic.DOMAIN, basis=ClaimBasis.OBSERVATION,
           citation_status=CitationStatus.UNCITED, citations=None) -> Claim:
    return Claim(id=cid, claim_text=text, anchor_status=AnchorStatus.EXACT, about=about,
                 basis=basis, citation_status=citation_status, citations=citations)


def _cited(cid, reference_id="PMID:1") -> Claim:
    return _claim(cid, citation_status=CitationStatus.CITED, basis=ClaimBasis.SECONDARY_SOURCE,
                  citations=[CitationHandle(marker="[1]", reference_id=reference_id,
                                            scope=CitationScope.SENTENCE)])


def _set(*claims: Claim) -> ClaimSet:
    from deep_research_client.claims import ExtractorInfo, SourceDocument, SourceType
    return ClaimSet(source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
                    extractor=ExtractorInfo(name="x"), claims=list(claims))


def test_triage_only_run_does_not_raise_and_carries_flags() -> None:
    """The old stub raised NotImplementedError; verify_claims now returns a report."""
    report = verify_claims(_set(_claim("c1")))
    assert isinstance(report, VerificationReport)
    assert report.verifier.name == "triage"
    assert report.verifier.methods == ["triage"]
    assert report.verification_list == []
    assert [f.kind for f in report.flags_for("c1")] == [FlagKind.UNSUPPORTED_CLAIM.value]
    assert report.verifier.verified == 0


def test_a_verifier_adds_verdicts_and_names_itself() -> None:
    report = verify_claims(
        _set(_cited("c1"), _cited("c2")),
        [FakeVerifier({"c1": VerificationStatus.SUPPORTED})],
    )
    assert report.verifier.name == "triage+fake"
    assert report.verifier.methods == ["triage", "fake"]
    verdict = report.verification_for("c1")
    assert verdict is not None and verdict.status == VerificationStatus.SUPPORTED.value
    assert report.verifier.verified == 1  # only c1 got a verdict


def test_unchecked_citation_flag_for_a_resolvable_claim_no_verifier_settled() -> None:
    """A cited, resolvable claim a verifier ran over but did not settle is flagged."""
    report = verify_claims(
        _set(_cited("c1"), _cited("c2")),
        [FakeVerifier({"c1": VerificationStatus.SUPPORTED})],  # says nothing about c2
    )
    assert [f.kind for f in report.flags_for("c2")] == [FlagKind.UNCHECKED_CITATION.value]
    assert report.flags_for("c1") == []  # c1 was settled SUPPORTED


def test_no_unchecked_flag_without_verifiers() -> None:
    """Triage-only must not flag every cited claim as unchecked -- it never checked."""
    report = verify_claims(_set(_cited("c1")))
    assert report.flags_of_kind(FlagKind.UNCHECKED_CITATION) == []


def test_an_unverifiable_only_claim_is_flagged_unchecked() -> None:
    """A citation that came back only UNVERIFIABLE was never settled, so it needs a look."""
    report = verify_claims(
        _set(_cited("c1")),
        [FakeVerifier({"c1": VerificationStatus.UNVERIFIABLE})],
    )
    assert [f.kind for f in report.flags_for("c1")] == [FlagKind.UNCHECKED_CITATION.value]
    verdict = report.verification_for("c1")
    assert verdict is not None and verdict.status == VerificationStatus.UNVERIFIABLE.value


def test_a_non_citation_unsupported_verdict_is_not_called_confabulated() -> None:
    """Only a citation:* UNSUPPORTED verdict means a fabricated citation (issue-#5 fix)."""
    report = verify_claims(
        _set(_cited("c1")),
        [FakeVerifier({"c1": VerificationStatus.UNSUPPORTED})],  # method is "fake:test"
    )
    kinds = {f.kind for f in report.flags_for("c1")}
    assert FlagKind.CONFABULATED_CITATION.value not in kinds
    # It also was not supported, so it stands unchecked for an agent.
    assert FlagKind.UNCHECKED_CITATION.value in kinds


class MultiVerifier:
    """A Verifier returning several verdicts for one claim, for the mixed-verdict case."""

    name = "multi"

    def __init__(self, verdicts: dict[str, list[VerificationStatus]]) -> None:
        self._verdicts = verdicts

    async def averify(self, claim: Claim) -> list[ClaimVerification]:
        return [
            ClaimVerification(claim_id=claim.id, status=s, method="citation:resolution",
                              reference_id=f"PMID:{i}", verifier=self.name)
            for i, s in enumerate(self._verdicts.get(claim.id, []), start=1)
        ]


def test_one_supported_work_addresses_a_claim_citing_several() -> None:
    """A claim with one SUPPORTED and one UNVERIFIABLE citation is not flagged unchecked."""
    report = verify_claims(
        _set(_cited("c1")),
        [MultiVerifier({"c1": [VerificationStatus.SUPPORTED, VerificationStatus.UNVERIFIABLE]})],
    )
    assert report.flags_of_kind(FlagKind.UNCHECKED_CITATION) == []
    assert len(report.verifications_for("c1")) == 2


def test_max_claims_leaves_uncovered_claims_flagged_unchecked() -> None:
    """Claims past the cap get no verdict, so their resolvable citations read as unchecked."""
    report = verify_claims(
        _set(_cited("c1"), _cited("c2")),
        [FakeVerifier({"c1": VerificationStatus.SUPPORTED, "c2": VerificationStatus.SUPPORTED})],
        max_claims=1,
    )
    assert report.flags_for("c1") == []  # c1 was checked and supported
    assert [f.kind for f in report.flags_for("c2")] == [FlagKind.UNCHECKED_CITATION.value]


def test_max_claims_bounds_verifier_coverage() -> None:
    report = verify_claims(
        _set(_cited("c1"), _cited("c2"), _cited("c3")),
        [FakeVerifier({"c1": VerificationStatus.SUPPORTED, "c2": VerificationStatus.SUPPORTED,
                       "c3": VerificationStatus.SUPPORTED})],
        max_claims=2,
    )
    assert {v.claim_id for v in report.verification_list} == {"c1", "c2"}  # c3 not reached


def test_alignment_contradictions_reach_the_report() -> None:
    from deep_research_client.claims import (AlignedClaimPair, AlignedSource, AlignedSourceType,
                                             AlignerInfo, EvidentialRelation, SemanticRelation)
    claim_set = _set(_claim("c1", basis=ClaimBasis.BACKGROUND_KNOWLEDGE),
                     _claim("c2", basis=ClaimBasis.BACKGROUND_KNOWLEDGE))
    alignment = ClaimAlignment(
        source_a=AlignedSource(id="r.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id="r.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        aligner=AlignerInfo(name="llm-pairwise"), within_source=True,
        pairs=[AlignedClaimPair(id="p1", claim_a_id="c1", claim_b_id="c2",
                                claim_a_text="A.", claim_b_text="not A.",
                                semantic_relation=SemanticRelation.CONTRADICTORY,
                                evidential_relation=EvidentialRelation.BOTH_BACKGROUND)],
    )
    report = verify_claims(claim_set, alignment=alignment)
    assert [f.kind for f in report.needs_human] == [FlagKind.SELF_CONTRADICTION.value]


def test_the_committed_verification_artifact_still_validates() -> None:
    """The docs example data must keep matching the schema and the triage output."""
    examples = Path(__file__).resolve().parent.parent / "examples" / "within-report-alignment"
    report = VerificationReport.model_validate(
        json.loads((examples / "acne_vulgaris.verification.json").read_text(encoding="utf-8"))
    )
    assert report.source_id == "Acne_Vulgaris-deep-research-falcon.md"
    assert report.flag_count == 193
    assert all(f.kind == FlagKind.UNSUPPORTED_CLAIM.value for f in report.flag_list)
    assert report.verifier.name == "triage"

    # Regenerating from the committed claims must reproduce the artifact's counts.
    claims = ClaimSet.model_validate(
        json.loads((examples / "acne_vulgaris.claims.json").read_text(encoding="utf-8"))
    )
    fresh = verify_claims(claims)
    assert fresh.flag_count == report.flag_count
