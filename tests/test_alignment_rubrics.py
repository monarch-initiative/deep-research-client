"""Report-level rubric aggregation (issue #43, item 4): deterministic, no model."""

from typing import Optional

import pytest

from deep_research_client.claims import (
    AlignedClaimPair,
    AlignedSource,
    AlignedSourceType,
    AlignerInfo,
    AnchorStatus,
    CitationHandle,
    CitationScope,
    CitationStatus,
    Claim,
    ClaimAlignment,
    ClaimSet,
    ClaimTopic,
    ClaimVerification,
    Decidability,
    EntityMention,
    EvidentialRelation,
    ExtractorInfo,
    FlagKind,
    GrounderInfo,
    ReviewFlag,
    SemanticRelation,
    SourceDocument,
    SourceType,
    VerificationReport,
    VerificationStatus,
    VerifierInfo,
    summarize_alignment,
)


def _claim(cid, *, cited=False) -> Claim:
    return Claim(
        id=cid, claim_text=f"Claim {cid}.", anchor_status=AnchorStatus.EXACT, about=ClaimTopic.DOMAIN,
        citations=[CitationHandle(marker="[1]", reference_id="PMID:1", scope=CitationScope.SENTENCE)] if cited else None,
        citation_status=CitationStatus.CITED if cited else CitationStatus.UNCITED,
    )


def _set(source_id, source_type, claims) -> ClaimSet:
    return ClaimSet(
        source=SourceDocument(id=source_id, source_type=source_type),
        extractor=ExtractorInfo(name="x"), claims=claims,
    )


def _pair(pid, a, b, relation) -> AlignedClaimPair:
    return AlignedClaimPair(
        id=pid, claim_a_id=a, claim_b_id=b, claim_a_text=".", claim_b_text=".",
        semantic_relation=relation, evidential_relation=EvidentialRelation.BOTH_SECONDARY,
    )


def _m(summary, name):
    """The named metric, asserted present (narrows Optional for the type checker)."""
    metric = summary.metric(name)
    assert metric is not None, name
    return metric


@pytest.fixture
def scenario():
    """A candidate (A) of three claims against a reference (B) of three."""
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT,
                 [_claim("c1", cited=True), _claim("c2"), _claim("c3", cited=True)])
    set_b = _set("b.yaml", SourceType.STRUCTURED_DOCUMENT,
                 [_claim("r1"), _claim("r2"), _claim("r3")])
    alignment = ClaimAlignment(
        source_a=AlignedSource(id="a.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id="b.yaml", source_type=AlignedSourceType.STRUCTURED_DOCUMENT),
        aligner=AlignerInfo(name="llm-pairwise"),
        pairs=[
            _pair("p1", "c1", "r1", SemanticRelation.IDENTICAL),
            _pair("p2", "c2", "r2", SemanticRelation.CONTRADICTORY),
        ],
    )
    return alignment, set_a, set_b


def test_each_metric_is_counts_with_the_shortfall_listed(scenario) -> None:
    """c3 is novel and r3 is uncovered; p2 contradicts; c2 is the uncited domain claim."""
    alignment, set_a, set_b = scenario
    summary = summarize_alignment(alignment, set_a, set_b, reference="b")

    assert _m(summary, "relevance").value == pytest.approx(2 / 3)  # c1, c2 align
    assert _m(summary, "relevance").supporting_claim_ids == ["c3"]  # the shortfall
    assert _m(summary, "informativeness").value == pytest.approx(1 / 3)  # c3 is new
    assert _m(summary, "informativeness").supporting_claim_ids == ["c3"]
    assert _m(summary, "support").value == pytest.approx(1 / 2)  # 1 of 2 not contradictory
    assert _m(summary, "support").supporting_pair_ids == ["p2"]  # the contradiction
    assert _m(summary, "completeness").value == pytest.approx(2 / 3)  # r1, r2 covered
    assert _m(summary, "completeness").supporting_claim_ids == ["r3"]  # the gap
    assert _m(summary, "citation_quality").value == pytest.approx(2 / 3)  # c1, c3 cited
    assert _m(summary, "citation_quality").supporting_claim_ids == ["c2"]  # the uncited one


def test_the_reference_choice_swaps_which_source_is_judged(scenario) -> None:
    """With A as reference, completeness is about A's claims being covered by B."""
    alignment, set_a, set_b = scenario
    summary = summarize_alignment(alignment, set_a, set_b, reference="a")
    # Candidate is now B (3 claims, 2 aligned); reference is A (3 claims, 2 covered).
    assert summary.reference == "a"
    assert _m(summary, "completeness").value == pytest.approx(2 / 3)
    assert _m(summary, "informativeness").supporting_claim_ids == ["r3"]  # B's novel claim


def test_an_empty_denominator_leaves_value_null(scenario) -> None:
    """A metric over no claims did not apply; its value is null, not zero."""
    alignment, _, set_b = scenario
    empty_a = _set("a.md", SourceType.MARKDOWN_REPORT, [])
    # No A claims: relevance/informativeness/citation_quality are over zero claims.
    summary = summarize_alignment(alignment, empty_a, set_b, reference="b")
    assert _m(summary, "relevance").value is None
    assert _m(summary, "relevance").denominator == 0


def test_an_unknown_reference_is_refused(scenario) -> None:
    alignment, set_a, set_b = scenario
    with pytest.raises(ValueError, match='reference must be "a" or "b"'):
        summarize_alignment(alignment, set_a, set_b, reference="gold")


def test_claim_sets_that_do_not_match_the_alignment_are_refused(scenario) -> None:
    """Passing the sets swapped would otherwise give plausible but wrong metrics."""
    alignment, set_a, set_b = scenario
    with pytest.raises(ValueError, match="do not match the alignment"):
        summarize_alignment(alignment, set_b, set_a)  # A and B swapped


def test_a_within_source_alignment_has_no_rubric() -> None:
    """The rubric compares two sources; it is refused in the library, not only the CLI."""
    report = _set("r.md", SourceType.MARKDOWN_REPORT, [_claim("c1"), _claim("c2")])
    alignment = ClaimAlignment(
        source_a=AlignedSource(id="r.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id="r.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        aligner=AlignerInfo(name="llm-pairwise-within"), within_source=True,
        pairs=[_pair("p1", "c1", "c2", SemanticRelation.IDENTICAL)],
    )
    with pytest.raises(ValueError, match="within one source"):
        summarize_alignment(alignment, report, report)


def _grounded(cid: str, mentions: list[tuple[str, Optional[str]]]) -> Claim:
    """A candidate claim with entity mentions, each (label, id-or-None)."""
    return Claim(
        id=cid, claim_text=f"Claim {cid}.", anchor_status=AnchorStatus.EXACT,
        about=ClaimTopic.DOMAIN, citation_status=CitationStatus.UNCITED,
        entities=[EntityMention(label=label, id=cid_) for label, cid_ in mentions],
    )


def _candidate_alignment(set_a: ClaimSet) -> tuple[ClaimAlignment, ClaimSet]:
    """A minimal cross-source alignment with set_a (a.md) as the candidate (A)."""
    set_b = _set("b.yaml", SourceType.STRUCTURED_DOCUMENT, [_claim("r1")])
    alignment = ClaimAlignment(
        source_a=AlignedSource(id="a.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id="b.yaml", source_type=AlignedSourceType.STRUCTURED_DOCUMENT),
        aligner=AlignerInfo(name="llm-pairwise"),
        pairs=[_pair("p1", set_a.claim_list[0].id, "r1", SemanticRelation.IDENTICAL)],
    )
    return alignment, set_b


def test_grounding_metric_counts_mention_coverage() -> None:
    """With grounding run, the metric is grounded mentions over all mentions."""
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT,
                 [_grounded("c1", [("acne", "MONDO:1"), ("TLR2", None)]),
                  _grounded("c2", [("sebum", "CHEBI:1")])])
    set_a = set_a.model_copy(update={"grounder": GrounderInfo(name="oak-llm", considered=3, grounded=2)})
    alignment, set_b = _candidate_alignment(set_a)
    summary = summarize_alignment(alignment, set_a, set_b, reference="b", verification=None)
    grounding = _m(summary, "grounding")
    assert grounding.value == pytest.approx(2 / 3)  # 2 of 3 mentions grounded
    assert grounding.supporting_claim_ids == ["c1"]  # c1 has an ungrounded mention


def test_grounding_metric_absent_when_the_set_was_not_grounded() -> None:
    """No grounder stamp means grounding did not run, so there is nothing to score."""
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT,
                 [_grounded("c1", [("acne", "MONDO:1")])])  # entities, but grounder is None
    alignment, set_b = _candidate_alignment(set_a)
    summary = summarize_alignment(alignment, set_a, set_b, reference="b")
    assert summary.metric("grounding") is None


def test_citation_accuracy_reads_fabricated_citations_from_verdicts() -> None:
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT,
                 [_claim("c1", cited=True), _claim("c2", cited=True), _claim("c3", cited=True)])
    alignment, set_b = _candidate_alignment(set_a)
    verification = VerificationReport(
        source_id="a.md", verifier=VerifierInfo(name="triage+citation"),
        verifications=[
            ClaimVerification(claim_id="c1", status=VerificationStatus.SUPPORTED,
                              method="citation:resolution", reference_id="PMID:1"),
            ClaimVerification(claim_id="c2", status=VerificationStatus.UNSUPPORTED,
                              method="citation:resolution", reference_id="PMID:999999999"),
            # c3 got no citation verdict: not counted as checked.
        ],
    )
    summary = summarize_alignment(alignment, set_a, set_b, reference="b", verification=verification)
    accuracy = _m(summary, "citation_accuracy")
    assert accuracy.value == pytest.approx(1 / 2)  # c1 resolves, c2 is fabricated
    assert accuracy.supporting_claim_ids == ["c2"]  # the fabricated one
    assert accuracy.denominator == 2  # only c1 and c2 were checked


def test_unverifiable_only_citations_drop_out_of_citation_accuracy() -> None:
    """A citation that could not be resolved either way is not 'accurate' -- it is unchecked."""
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT,
                 [_claim("c1", cited=True), _claim("c2", cited=True)])
    alignment, set_b = _candidate_alignment(set_a)
    verification = VerificationReport(
        source_id="a.md", verifier=VerifierInfo(name="triage+citation"),
        verifications=[
            ClaimVerification(claim_id="c1", status=VerificationStatus.SUPPORTED,
                              method="citation:resolution", reference_id="PMID:1"),
            ClaimVerification(claim_id="c2", status=VerificationStatus.UNVERIFIABLE,
                              method="citation:resolution", reference_id="ISBN:x"),
        ],
    )
    summary = summarize_alignment(alignment, set_a, set_b, reference="b", verification=verification)
    accuracy = _m(summary, "citation_accuracy")
    assert accuracy.denominator == 1  # only c1 was decisively checked; c2 is excluded
    assert accuracy.value == 1.0


def test_an_off_topic_citation_still_counts_as_resolving() -> None:
    """NEEDS_REVIEW (citation:relevance) means the work resolved, so it is not fabricated."""
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT, [_claim("c1", cited=True)])
    alignment, set_b = _candidate_alignment(set_a)
    verification = VerificationReport(
        source_id="a.md", verifier=VerifierInfo(name="triage+citation"),
        verifications=[ClaimVerification(claim_id="c1", status=VerificationStatus.NEEDS_REVIEW,
                                         method="citation:relevance", reference_id="PMID:1")],
    )
    summary = summarize_alignment(alignment, set_a, set_b, reference="b", verification=verification)
    accuracy = _m(summary, "citation_accuracy")
    assert accuracy.denominator == 1 and accuracy.value == 1.0  # resolved, just off topic


def test_grounding_metric_is_none_when_a_grounded_set_has_no_mentions() -> None:
    """A grounded set with no entity mentions anywhere has no coverage to score."""
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT, [_claim("c1")])  # no entities
    set_a = set_a.model_copy(update={"grounder": GrounderInfo(name="oak-llm", considered=0, grounded=0)})
    alignment, set_b = _candidate_alignment(set_a)
    summary = summarize_alignment(alignment, set_a, set_b, reference="b")
    assert summary.metric("grounding") is None


def test_review_burden_folds_in_alignment_contradictions() -> None:
    """A contradicted candidate claim needs a human even when the verification is triage-only."""
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT, [_claim("c1"), _claim("c2")])
    set_b = _set("b.yaml", SourceType.STRUCTURED_DOCUMENT, [_claim("r1")])
    alignment = ClaimAlignment(
        source_a=AlignedSource(id="a.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id="b.yaml", source_type=AlignedSourceType.STRUCTURED_DOCUMENT),
        aligner=AlignerInfo(name="llm-pairwise"),
        pairs=[_pair("p1", "c1", "r1", SemanticRelation.CONTRADICTORY)],
    )
    # A triage-only verification carries no CONTRADICTION flag, yet c1 contradicts r1.
    verification = VerificationReport(source_id="a.md", verifier=VerifierInfo(name="triage"))
    summary = summarize_alignment(alignment, set_a, set_b, reference="b", verification=verification)
    burden = _m(summary, "review_burden")
    assert burden.supporting_claim_ids == ["c1"]  # folded in from the alignment
    assert burden.value == pytest.approx(1 / 2)


def test_review_burden_counts_a_claim_once_even_in_a_multi_id_flag() -> None:
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT, [_claim("c1"), _claim("c2"), _claim("c3")])
    alignment, set_b = _candidate_alignment(set_a)
    verification = VerificationReport(
        source_id="a.md", verifier=VerifierInfo(name="triage"),
        flags=[ReviewFlag(kind=FlagKind.SELF_CONTRADICTION, decidability=Decidability.HUMAN,
                          claim_ids=["c1", "c2"], message="c1 vs c2")],
    )
    summary = summarize_alignment(alignment, set_a, set_b, reference="b", verification=verification)
    burden = _m(summary, "review_burden")
    assert set(burden.supporting_claim_ids or []) == {"c1", "c2"}
    assert burden.value == pytest.approx(1 / 3)  # c3 is the only clean one


def test_review_burden_counts_claims_with_a_human_flag() -> None:
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT, [_claim("c1"), _claim("c2")])
    alignment, set_b = _candidate_alignment(set_a)
    verification = VerificationReport(
        source_id="a.md", verifier=VerifierInfo(name="triage"),
        flags=[ReviewFlag(kind=FlagKind.CONTRADICTION, decidability=Decidability.HUMAN,
                          claim_ids=["c1"], message="c1 contradicts something")],
    )
    summary = summarize_alignment(alignment, set_a, set_b, reference="b", verification=verification)
    burden = _m(summary, "review_burden")
    assert burden.value == pytest.approx(1 / 2)  # 1 of 2 claims is clean
    assert burden.supporting_claim_ids == ["c1"]  # the one needing a human


def test_verification_for_a_different_source_is_refused() -> None:
    set_a = _set("a.md", SourceType.MARKDOWN_REPORT, [_claim("c1")])
    alignment, set_b = _candidate_alignment(set_a)
    wrong = VerificationReport(source_id="other.md", verifier=VerifierInfo(name="triage"))
    with pytest.raises(ValueError, match="different source than the candidate"):
        summarize_alignment(alignment, set_a, set_b, reference="b", verification=wrong)


def test_without_verification_only_the_five_alignment_metrics_appear(scenario) -> None:
    """Backward compatible: the folded metrics appear only when their inputs are given."""
    alignment, set_a, set_b = scenario
    names = {m.name for m in (summarize_alignment(alignment, set_a, set_b, reference="b").metrics or [])}
    assert names == {"relevance", "informativeness", "support", "completeness", "citation_quality"}


def test_the_substrate_is_source_agnostic_report_vs_report() -> None:
    """Alignment and rubrics read claims, not source types: two reports align too.

    The issue asks for report-vs-report as well as report-vs-structured. Nothing
    in alignment or rubrics branches on source type, so the same code handles two
    markdown reports; this pins that.
    """
    set_a = _set("report_a.md", SourceType.MARKDOWN_REPORT, [_claim("c1"), _claim("c2")])
    set_b = _set("report_b.md", SourceType.MARKDOWN_REPORT, [_claim("r1")])
    alignment = ClaimAlignment(
        source_a=AlignedSource(id="report_a.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id="report_b.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        aligner=AlignerInfo(name="llm-pairwise"),
        pairs=[_pair("p1", "c1", "r1", SemanticRelation.CLOSE)],
    )
    summary = summarize_alignment(alignment, set_a, set_b, reference="b")
    assert _m(summary, "completeness").value == pytest.approx(1.0)  # r1 covered by c1
    assert _m(summary, "informativeness").supporting_claim_ids == ["c2"]  # c2 is novel
