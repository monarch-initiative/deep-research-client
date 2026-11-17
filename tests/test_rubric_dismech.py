"""The folded rubric on real and report-shaped deep-research data, offline.

Two kinds of real-world coverage, both without a model or the network:

- A genuine DisMech report: the committed ``acne_vulgaris`` falcon report (267
  claims). Triage reproduces its verification profile, and the folded
  ``review_burden`` / ``grounding`` metrics are checked on that real report.
- Report-shaped fixtures with a recorded alignment: ``report_a`` (candidate)
  against ``gold_b`` (reference). These are small synthetic fixtures (so the
  alignment could be recorded), used to exercise the enriched
  ``summarize_alignment`` through its public entry point. The cross-source
  alignment is rebuilt from the recorded model replies.

The live extract -> ground -> verify path on real data is covered under the
integration marker in test_rubric_integration.py.
"""

import json
from functools import cache
from pathlib import Path

import yaml

from deep_research_client.claims import (
    AlignedClaimPair,
    AlignedSource,
    AlignedSourceType,
    AlignerInfo,
    AnchorStatus,
    CitationStatus,
    Claim,
    ClaimAlignment,
    ClaimSet,
    ClaimTopic,
    EvidentialRelation,
    ExtractorInfo,
    FlagKind,
    SemanticRelation,
    SourceDocument,
    SourceType,
    VerificationReport,
    candidate_pairs,
    evidential_relation,
    shares_citation,
    summarize_alignment,
    verify_claims,
)
from deep_research_client.claims.align_parsing import relation_from_reply

ALIGN = Path(__file__).parent / "input" / "claims" / "alignment"
MANIFEST = yaml.safe_load((ALIGN / "manifest.yaml").read_text(encoding="utf-8"))
MODEL = sorted(MANIFEST["recordings"])[0]
EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "within-report-alignment"


# ---- A genuine DisMech report: acne_vulgaris (falcon), 267 real claims ----

@cache
def _acne() -> tuple[ClaimSet, VerificationReport]:
    claims = ClaimSet.model_validate(
        json.loads((EXAMPLES / "acne_vulgaris.claims.json").read_text("utf-8")))
    verification = VerificationReport.model_validate(
        json.loads((EXAMPLES / "acne_vulgaris.verification.json").read_text("utf-8")))
    return claims, verification


def test_triage_reproduces_the_committed_profile_on_the_real_report() -> None:
    """verify_claims on the real acne report gives the committed flag profile."""
    claims, committed = _acne()
    fresh = verify_claims(claims)  # pure triage, no model, no network
    # Reproduces the committed count exactly (the literal lives in the fixture),
    assert fresh.flag_count == committed.flag_count
    # and the shape: all unsupported-claim flags, none needing a human.
    assert all(f.kind == FlagKind.UNSUPPORTED_CLAIM.value for f in fresh.flag_list)
    assert fresh.needs_human == []  # every issue here is agent-decidable, none human-only


def _trivial_reference(candidate: ClaimSet) -> tuple[ClaimAlignment, ClaimSet]:
    """A one-claim reference and a minimal cross-source alignment with candidate as A.

    The folded review_burden/grounding metrics are candidate-only, so the reference
    content does not affect them; this is just enough of a valid cross-source
    alignment to reach them through the public summarize_alignment.
    """
    reference = ClaimSet(
        source=SourceDocument(id="ref.yaml", source_type=SourceType.STRUCTURED_DOCUMENT),
        extractor=ExtractorInfo(name="x"),
        claims=[Claim(id="r1", claim_text="A curated reference claim.",
                      anchor_status=AnchorStatus.EXACT, about=ClaimTopic.DOMAIN,
                      citation_status=CitationStatus.UNCITED)],
    )
    alignment = ClaimAlignment(
        source_a=AlignedSource(id=candidate.source.id, source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id="ref.yaml", source_type=AlignedSourceType.STRUCTURED_DOCUMENT),
        aligner=AlignerInfo(name="llm-pairwise"),
        pairs=[AlignedClaimPair(id="p1", claim_a_id=candidate.claim_list[0].id, claim_b_id="r1",
                                claim_a_text=".", claim_b_text=".",
                                semantic_relation=SemanticRelation.CLOSE,
                                evidential_relation=EvidentialRelation.BOTH_SECONDARY)],
    )
    return alignment, reference


def test_folded_metrics_on_the_real_report_through_the_public_api() -> None:
    """review_burden and grounding read correctly off the real report + its verification."""
    claims, committed = _acne()
    alignment, reference = _trivial_reference(claims)
    summary = summarize_alignment(alignment, claims, reference, reference="b", verification=committed)
    # Nothing is flagged for a human, so the whole report is "clean" on review burden.
    burden = summary.metric("review_burden")
    assert burden is not None and burden.value == 1.0 and burden.denominator == 267
    assert burden.supporting_claim_ids in (None, [])
    # This report was never run through grounding (no grounder stamp), so there is no
    # coverage to score, even though extraction left a few entity ids behind.
    assert claims.grounder is None
    assert summary.metric("grounding") is None


# ---- Report-shaped fixtures with a recorded alignment: report_a vs gold_b ----

@cache
def _sources() -> tuple[ClaimSet, ClaimSet]:
    set_a = ClaimSet.model_validate(json.loads((ALIGN / MANIFEST["sources"]["a"]).read_text("utf-8")))
    set_b = ClaimSet.model_validate(json.loads((ALIGN / MANIFEST["sources"]["b"]).read_text("utf-8")))
    return set_a, set_b


def _aligned_source(claim_set: ClaimSet) -> AlignedSource:
    return AlignedSource(id=claim_set.source.id,
                         source_type=AlignedSourceType(claim_set.source.source_type))


@cache
def _alignment() -> ClaimAlignment:
    """Rebuild the report_a-vs-gold_b alignment from the recorded model replies."""
    set_a, set_b = _sources()
    pairs = candidate_pairs(set_a.claim_list, set_b.claim_list)
    aligned = []
    for i, candidate in enumerate(pairs, 1):
        verdict = relation_from_reply((ALIGN / MODEL / f"{i:02d}.txt").read_text("utf-8"))
        aligned.append(AlignedClaimPair(
            id=f"p{i}", claim_a_id=candidate.claim_a.id, claim_b_id=candidate.claim_b.id,
            claim_a_text=candidate.claim_a.claim_text, claim_b_text=candidate.claim_b.claim_text,
            semantic_relation=verdict.semantic_relation,
            evidential_relation=evidential_relation(candidate.claim_a, candidate.claim_b),
            shares_citation=shares_citation(candidate.claim_a, candidate.claim_b),
            candidate_signals=candidate.signal_list or None,
            confidence=verdict.confidence, rationale=verdict.rationale,
        ))
    return ClaimAlignment(
        source_a=_aligned_source(set_a), source_b=_aligned_source(set_b),
        aligner=AlignerInfo(name="llm-pairwise", model="recorded", prompt_version="1"),
        pairs=aligned,
    )


def test_enriched_rubric_through_the_public_api_on_a_recorded_alignment() -> None:
    """The five alignment metrics plus review_burden, via summarize_alignment."""
    set_a, set_b = _sources()
    verification = verify_claims(set_a)  # triage-only: pure
    summary = summarize_alignment(_alignment(), set_a, set_b, reference="b", verification=verification)

    names = {m.name for m in (summary.metrics or [])}
    assert {"relevance", "informativeness", "support", "completeness", "citation_quality"} <= names
    # Triage ran -> review_burden folds in; no citation verdicts and not grounded ->
    # those two folded metrics stay absent.
    assert "review_burden" in names
    assert "citation_accuracy" not in names and "grounding" not in names

    burden = summary.metric("review_burden")
    assert burden is not None and burden.value is not None and 0.0 <= burden.value <= 1.0
    candidate_ids = {c.id for c in set_a.claim_list}
    assert set(burden.supporting_claim_ids or []) <= candidate_ids
    assert burden.denominator == len(candidate_ids)
