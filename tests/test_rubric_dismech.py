"""The folded rubric on a real deep-research report, offline.

Uses the committed DisMech-style fixtures: a real report (``report_a``, the
candidate) against a curated reference (``gold_b``). The cross-source alignment is
rebuilt from the recorded model replies (as test_alignment_recorded does), the
candidate's verification is computed purely by triage (no model, no network), and
the enriched ``summarize_alignment`` is checked on that real data. The live
extract -> ground -> verify path is covered under the integration marker in
test_rubric_integration.py.
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
    ClaimAlignment,
    ClaimSet,
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
    """Rebuild the real report_a-vs-gold_b alignment from the recorded replies."""
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


def test_enriched_rubric_on_a_real_report_against_a_curated_reference() -> None:
    """The five alignment metrics plus review_burden, computed on real report data."""
    set_a, set_b = _sources()
    alignment = _alignment()
    verification = verify_claims(set_a)  # triage-only: pure, no model, no network
    summary = summarize_alignment(alignment, set_a, set_b, reference="b", verification=verification)

    names = {m.name for m in summary.metrics}
    # The five alignment metrics are always there.
    assert {"relevance", "informativeness", "support", "completeness", "citation_quality"} <= names
    # Triage ran, so review_burden folds in; triage produces no citation verdicts and
    # report_a is not grounded, so those two folded metrics stay absent.
    assert "review_burden" in names
    assert "citation_accuracy" not in names
    assert "grounding" not in names

    burden = summary.metric("review_burden")
    assert burden is not None
    # review_burden counts candidate claims with no human-only flag; it is a real
    # fraction of report_a's claims, and its shortfall ids are a subset of them.
    assert 0.0 <= burden.value <= 1.0
    candidate_ids = {c.id for c in set_a.claim_list}
    assert set(burden.supporting_claim_ids or []) <= candidate_ids
    assert burden.denominator == len(candidate_ids)


def test_the_reused_fixtures_are_a_real_report_and_a_reference() -> None:
    """Guard that these stay real multi-claim fixtures, not toy stubs."""
    set_a, set_b = _sources()
    assert len(set_a.claim_list) >= 3 and len(set_b.claim_list) >= 3
    assert _alignment().pair_list  # at least one candidate pair was classified
