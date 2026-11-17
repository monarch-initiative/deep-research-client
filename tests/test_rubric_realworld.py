"""The enriched rubric on a real DisMech report vs its curated record, offline.

The committed example in ``examples/report-vs-curated/`` is a real openai
deep-research report on Atelosteogenesis Type II against the curated DisMech
disorder record (``MONDO:0009727``), carried through extract -> ground -> verify ->
align and summarised with the folded rubric. This replays the committed artifacts
(no model, no network) and checks that ``summarize_alignment`` reproduces the
committed rubric, with all three folded metrics present. It is a representative
slice (see the example's README); the live full pipeline is in
test_rubric_integration.py.
"""

import json
from functools import cache
from pathlib import Path

from deep_research_client.claims import (
    ClaimAlignment,
    ClaimSet,
    VerificationReport,
    summarize_alignment,
)

EX = Path(__file__).resolve().parent.parent / "examples" / "report-vs-curated"
STEM = "atelosteogenesis_type_ii"


@cache
def _artifacts() -> tuple[ClaimSet, ClaimSet, ClaimAlignment, VerificationReport, dict]:
    report = ClaimSet.model_validate_json((EX / f"{STEM}.report.grounded.json").read_text("utf-8"))
    gold = ClaimSet.model_validate_json((EX / f"{STEM}.gold.claims.json").read_text("utf-8"))
    alignment = ClaimAlignment.model_validate_json((EX / f"{STEM}.align.json").read_text("utf-8"))
    verification = VerificationReport.model_validate_json(
        (EX / f"{STEM}.verification.json").read_text("utf-8"))
    rubric = json.loads((EX / f"{STEM}.rubric.json").read_text("utf-8"))
    return report, gold, alignment, verification, rubric


def test_the_committed_real_report_artifacts_are_consistent() -> None:
    """The ids line up: the alignment's A side is the report, its B side the record."""
    report, gold, alignment, verification, _ = _artifacts()
    assert report.source.id == "Atelosteogenesis_Type_II-deep-research-openai.md"
    assert gold.source.id == "Atelosteogenesis_Type_II.yaml"
    assert alignment.source_a.id == report.source.id
    assert alignment.source_b.id == gold.source.id
    assert verification.source_id == report.source.id
    assert report.grounder is not None  # the report was grounded


def test_summarize_reproduces_the_committed_enriched_rubric() -> None:
    """Recomputing from the committed pieces gives the committed rubric exactly."""
    report, gold, alignment, verification, committed = _artifacts()
    summary = summarize_alignment(alignment, report, gold, reference="b", verification=verification)
    fresh = {m.name: (m.numerator, m.denominator) for m in (summary.metrics or [])}
    committed_counts = {m["name"]: (m["numerator"], m["denominator"]) for m in committed["metrics"]}
    assert fresh == committed_counts


def test_all_three_folded_metrics_are_present_on_the_real_report() -> None:
    """Grounding, citation accuracy and review burden all fold in on real data."""
    report, gold, alignment, verification, _ = _artifacts()
    summary = summarize_alignment(alignment, report, gold, reference="b", verification=verification)
    names = {m.name for m in (summary.metrics or [])}
    assert {"grounding", "citation_accuracy", "review_burden"} <= names

    grounding = summary.metric("grounding")
    assert grounding is not None and grounding.value is not None and 0.0 < grounding.value < 1.0

    # Hand-verified anchor: the report cites real works on exactly these seven claims,
    # and every one resolves, so citation_accuracy is 7/7 with nothing fabricated.
    checked = {v.claim_id for v in verification.verification_list
               if (v.method or "").startswith("citation:")}
    assert checked == {"c45", "c46", "c47", "c49", "c50", "c51", "c52"}
    citation = summary.metric("citation_accuracy")
    assert citation is not None and citation.numerator == 7 and citation.denominator == 7
    assert citation.supporting_claim_ids in (None, [])  # none fabricated

    burden = summary.metric("review_burden")
    # The contradictory pairs' candidate-side claims are folded in as needing a human.
    assert burden is not None and burden.supporting_claim_ids
