"""Aggregate an alignment into report-level rubric metrics (issue #43, item 4).

Once two sources' claims are aligned, several qualitative judgements fall out of
the pairs and the two claim sets, with no further model call. One source is the
*reference* (say a curated gold file); the other is the *candidate* (say a report)
judged against it. The metrics are deliberately count-based and carry the ids
behind them, so a person can check any number and a downstream project can
recompute or extend the set rather than trust an opaque score.

Standard metrics, all as ``numerator/denominator`` with the shortfall listed:

- ``relevance``: candidate claims that align to anything in the reference.
- ``informativeness``: candidate claims with no reference partner (what the
  candidate adds).
- ``support``: aligned pairs that are not contradictory.
- ``completeness``: reference claims the candidate covers.
- ``citation_quality``: candidate domain claims that are cited.

When the candidate's grounding and verification are available, three more metrics
fold those later steps in, so accuracy is read from what was checked, not only from
contradiction counts:

- ``grounding``: candidate entity mentions resolved to an ontology id (coverage).
- ``citation_accuracy``: checked cited claims whose citations resolve to a real
  work (not fabricated). Driven by citation verdicts, unlike ``citation_quality``,
  which only asks whether a claim is cited at all.
- ``review_burden``: candidate claims that raise no flag only a human can settle.

Everything here is a pure function of the alignment, the two claim sets, and (for
the folded metrics) the candidate's verification report.
"""

from typing import Optional

from .alignment_models import ClaimAlignment, RubricMetric, RubricSummary
from .models import CitationStatus, Claim, ClaimSet, ClaimTopic, EntityMention
from .verification_models import VerificationReport, VerificationStatus

__all__ = ["summarize_alignment"]


def _mentions(claim: Claim) -> list[EntityMention]:
    """Every entity mention on a claim: its entities, subject and object."""
    mentions = list(claim.entities or [])
    mentions += [m for m in (claim.subject, claim.object) if m is not None]
    return mentions


def _grounding_metric(candidate: ClaimSet) -> Optional[RubricMetric]:
    """Coverage of the candidate's entity grounding, or None when it was not grounded.

    Only meaningful once grounding ran (``candidate.grounder`` set) and the claims
    carry entity mentions; otherwise there is nothing to measure.
    """
    if candidate.grounder is None:
        return None
    total = 0
    grounded = 0
    ungrounded_claim_ids: list[str] = []
    for claim in candidate.claim_list:
        mentions = _mentions(claim)
        if not mentions:
            continue
        total += len(mentions)
        here = sum(1 for m in mentions if m.id)
        grounded += here
        if here < len(mentions):
            ungrounded_claim_ids.append(claim.id)
    if total == 0:
        return None
    return _metric(
        "grounding",
        numerator=grounded,
        denominator=total,
        note=f"{grounded} of {total} candidate entity mentions are grounded to an ontology id",
        claim_ids=ungrounded_claim_ids,
    )


def _citation_accuracy_metric(
    candidate: ClaimSet, verification: VerificationReport
) -> Optional[RubricMetric]:
    """Checked cited claims whose citations resolve to a real work, or None if none checked.

    A claim is *checked* when a citation verifier returned a verdict for it; it is
    sound when none of those verdicts is a fabricated citation (an ``UNSUPPORTED``
    ``citation:`` verdict). This reads accuracy from the verdicts, where
    ``citation_quality`` only asks whether a claim is cited at all.
    """
    checked: list[str] = []
    fabricated: list[str] = []
    for claim in candidate.claim_list:
        verdicts = [
            v for v in verification.verifications_for(claim.id)
            if (v.method or "").startswith("citation:")
        ]
        if not verdicts:
            continue
        checked.append(claim.id)
        if any(VerificationStatus(v.status) == VerificationStatus.UNSUPPORTED for v in verdicts):
            fabricated.append(claim.id)
    if not checked:
        return None
    return _metric(
        "citation_accuracy",
        numerator=len(checked) - len(fabricated),
        denominator=len(checked),
        note=f"{len(fabricated)} of {len(checked)} checked claims cite a work that does not resolve",
        claim_ids=fabricated,
    )


def _review_burden_metric(
    candidate: ClaimSet, verification: VerificationReport
) -> RubricMetric:
    """Candidate claims that raise no human-only review flag (higher is cleaner)."""
    needs_human = {
        cid for flag in verification.needs_human for cid in (flag.claim_ids or [])
    }
    candidate_ids = [c.id for c in candidate.claim_list]
    flagged = [cid for cid in candidate_ids if cid in needs_human]
    return _metric(
        "review_burden",
        numerator=len(candidate_ids) - len(flagged),
        denominator=len(candidate_ids),
        note=f"{len(flagged)} of {len(candidate_ids)} candidate claims need human review",
        claim_ids=flagged,
    )


def _metric(
    name: str,
    numerator: int,
    denominator: int,
    note: str,
    *,
    pair_ids: Optional[list[str]] = None,
    claim_ids: Optional[list[str]] = None,
) -> RubricMetric:
    """One metric, with value computed as numerator/denominator (None if empty)."""
    return RubricMetric(
        name=name,
        numerator=numerator,
        denominator=denominator,
        value=(numerator / denominator) if denominator else None,
        supporting_pair_ids=pair_ids or None,
        supporting_claim_ids=claim_ids or None,
        note=note,
    )


def summarize_alignment(
    alignment: ClaimAlignment,
    set_a: ClaimSet,
    set_b: ClaimSet,
    *,
    reference: str = "b",
    verification: Optional[VerificationReport] = None,
) -> RubricSummary:
    """Aggregate an alignment into report-level rubric metrics.

    The five alignment metrics are always produced. When ``verification`` (the
    candidate's report) is given, and when the candidate was grounded, the folded
    metrics ``grounding``, ``citation_accuracy`` and ``review_burden`` are appended,
    so the summary reads accuracy from what grounding and verification found, not
    only from the alignment.

    Args:
        alignment: The alignment of set A against set B.
        set_a: Source A's claims (the A side of every pair).
        set_b: Source B's claims (the B side).
        reference: Which source is the reference the other is judged against,
            ``"a"`` or ``"b"``. The other source is the candidate.
        verification: The candidate's verification report (from
            :func:`~deep_research_client.claims.verify_claims`), to fold citation
            and review metrics in. Its ``source_id`` must be the candidate's.

    Returns:
        A summary with the standard metrics, each carrying the ids behind it, plus
        the folded metrics when their inputs are present.

    Raises:
        ValueError: If ``reference`` is not ``"a"`` or ``"b"``, or if a
            ``verification`` is given whose source is not the candidate.

    >>> from deep_research_client.claims import (AlignedClaimPair, AlignedSource,
    ...     AlignedSourceType, AlignerInfo, CitationHandle, CitationScope, Claim,
    ...     ExtractorInfo, SemanticRelation, SourceDocument)
    >>> def claim(i, cited=False):
    ...     return Claim(id=i, claim_text="X.", anchor_status="EXACT", about=ClaimTopic.DOMAIN,
    ...                  citations=[CitationHandle(marker="[1]", scope=CitationScope.SENTENCE)] if cited else None,
    ...                  citation_status=CitationStatus.CITED if cited else CitationStatus.UNCITED)
    >>> set_a = ClaimSet(
    ...     source=SourceDocument(id="a.md", source_type="MARKDOWN_REPORT"),
    ...     extractor=ExtractorInfo(name="x"),
    ...     claims=[claim("c1", cited=True), claim("c2")])
    >>> set_b = ClaimSet(
    ...     source=SourceDocument(id="b.yaml", source_type="STRUCTURED_DOCUMENT"),
    ...     extractor=ExtractorInfo(name="y"), claims=[claim("r1")])
    >>> pair = AlignedClaimPair(id="p1", claim_a_id="c1", claim_b_id="r1",
    ...     claim_a_text="X.", claim_b_text="X.", semantic_relation=SemanticRelation.IDENTICAL,
    ...     evidential_relation="BOTH_OBSERVATION")
    >>> alignment = ClaimAlignment(
    ...     source_a=AlignedSource(id="a.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
    ...     source_b=AlignedSource(id="b.yaml", source_type=AlignedSourceType.STRUCTURED_DOCUMENT),
    ...     aligner=AlignerInfo(name="llm-pairwise"), pairs=[pair])
    >>> summary = summarize_alignment(alignment, set_a, set_b, reference="b")
    >>> summary.metric("completeness").value  # 1 of 1 reference claim covered
    1.0
    >>> summary.metric("informativeness").value  # c2 is novel: 1 of 2 candidate claims
    0.5

    Fold in the candidate's verification to read citation accuracy from verdicts:

    >>> from deep_research_client.claims import (ClaimVerification, VerificationReport,
    ...     VerificationStatus, VerifierInfo)
    >>> vr = VerificationReport(source_id="a.md", verifier=VerifierInfo(name="triage+citation"),
    ...     verifications=[ClaimVerification(claim_id="c1", status=VerificationStatus.SUPPORTED,
    ...         method="citation:resolution", reference_id="PMID:1")])
    >>> enriched = summarize_alignment(alignment, set_a, set_b, reference="b", verification=vr)
    >>> enriched.metric("citation_accuracy").value  # c1's one checked citation resolves
    1.0
    """
    if reference not in ("a", "b"):
        raise ValueError(f'reference must be "a" or "b", not {reference!r}')
    if alignment.within_source:
        raise ValueError(
            "the rubric compares two different sources; this alignment is within one "
            "source (within_source is set). Its metrics would be meaningless."
        )
    if alignment.source_a.id != set_a.source.id or alignment.source_b.id != set_b.source.id:
        raise ValueError(
            "the claim sets do not match the alignment: it aligns "
            f"{alignment.source_a.id!r} (A) against {alignment.source_b.id!r} (B), "
            f"but was given {set_a.source.id!r} and {set_b.source.id!r} "
            "(are set_a and set_b in the wrong order?)"
        )

    if reference == "b":
        candidate, ref_set = set_a, set_b
        aligned_candidate, aligned_reference = alignment.aligned_a_ids, alignment.aligned_b_ids
    else:
        candidate, ref_set = set_b, set_a
        aligned_candidate, aligned_reference = alignment.aligned_b_ids, alignment.aligned_a_ids

    if verification is not None and verification.source_id != candidate.source.id:
        raise ValueError(
            "the verification is for a different source than the candidate: it "
            f"reports on {verification.source_id!r}, but the candidate is "
            f"{candidate.source.id!r} (pass the candidate's own verification)."
        )

    candidate_ids = [c.id for c in candidate.claim_list]
    reference_ids = [c.id for c in ref_set.claim_list]

    unaligned_candidate = [cid for cid in candidate_ids if cid not in aligned_candidate]
    uncovered_reference = [rid for rid in reference_ids if rid not in aligned_reference]

    aligned = alignment.aligned_pairs
    # A contradiction is an alignment (it is not UNRELATED), so it is in `aligned`.
    contradictions = [p.id for p in alignment.contradictions]

    domain = [c for c in candidate.claim_list if c.about == ClaimTopic.DOMAIN]
    uncited_domain = [c.id for c in domain if c.citation_status != CitationStatus.CITED]

    metrics = [
        _metric(
            "relevance",
            numerator=len(candidate_ids) - len(unaligned_candidate),
            denominator=len(candidate_ids),
            note=f"{len(unaligned_candidate)} candidate claims align to nothing in the reference",
            claim_ids=unaligned_candidate,
        ),
        _metric(
            "informativeness",
            numerator=len(unaligned_candidate),
            denominator=len(candidate_ids),
            note=f"{len(unaligned_candidate)} candidate claims add information the reference lacks",
            claim_ids=unaligned_candidate,
        ),
        _metric(
            "support",
            numerator=len(aligned) - len(contradictions),
            denominator=len(aligned),
            note=f"{len(contradictions)} of {len(aligned)} aligned pairs are contradictory",
            pair_ids=contradictions,
        ),
        _metric(
            "completeness",
            numerator=len(reference_ids) - len(uncovered_reference),
            denominator=len(reference_ids),
            note=f"{len(uncovered_reference)} reference claims the candidate does not cover",
            claim_ids=uncovered_reference,
        ),
        _metric(
            "citation_quality",
            numerator=len(domain) - len(uncited_domain),
            denominator=len(domain),
            note=f"{len(uncited_domain)} of {len(domain)} candidate domain claims are uncited",
            claim_ids=uncited_domain,
        ),
    ]

    # Fold in the later steps when their inputs are present. Grounding needs only the
    # candidate (it carries its own grounder stamp and entity ids); the citation and
    # review metrics need the candidate's verification report.
    grounding = _grounding_metric(candidate)
    if grounding is not None:
        metrics.append(grounding)
    if verification is not None:
        citation_accuracy = _citation_accuracy_metric(candidate, verification)
        if citation_accuracy is not None:
            metrics.append(citation_accuracy)
        metrics.append(_review_burden_metric(candidate, verification))

    return RubricSummary(reference=reference, metrics=metrics)
