"""Report-level rubric aggregation (issue #43, item 4): deterministic, no model."""

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
    EvidentialRelation,
    ExtractorInfo,
    SemanticRelation,
    SourceDocument,
    SourceType,
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
