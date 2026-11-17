"""Alignment models: the generated datamodel plus derived views and the two axes.

The fields come from ``alignment.yaml`` via ``alignment_datamodel.py``.
``ClaimAlignment`` and ``RubricSummary`` are subclassed here to add properties
and load-time checks computed from those fields; they add no fields of their own,
so their JSON schema stays exactly what LinkML would emit.

The semantic axis of a pair is judged by a model (see ``align_llm``). The
evidential axis is not judged: :func:`evidential_relation` derives it from the two
claims' ``about`` and ``basis`` (see ``claims.yaml``), and :func:`shares_citation`
reads their citations. Both are pure functions of two claims, so they are tested
without a model.

The generated base sets ``use_enum_values=True``, so an enum slot reads back as
its string value; compare with ``==`` against the enum member, which is a ``str``
subclass.
"""

from typing import Optional

from pydantic import model_validator

from .alignment_datamodel import (
    AlignedClaimPair,
    AlignedSource,
    AlignedSourceType,
    AlignerInfo,
    CandidateSignal,
    EvidentialRelation,
    RubricMetric,
    SemanticRelation,
)
from .alignment_datamodel import ClaimAlignment as GeneratedClaimAlignment
from .alignment_datamodel import RubricSummary as GeneratedRubricSummary
from .models import Claim, ClaimBasis, ClaimTopic

__all__ = [
    "AlignedClaimPair",
    "AlignedSource",
    "AlignedSourceType",
    "AlignerInfo",
    "CandidateSignal",
    "ClaimAlignment",
    "EvidentialRelation",
    "RubricMetric",
    "RubricSummary",
    "SemanticRelation",
    "evidential_relation",
    "shares_citation",
]


#: The evidential relation for each ordered pair of bases. Keyed (basis_a, basis_b).
_EVIDENTIAL_BY_BASIS: dict[tuple[str, str], EvidentialRelation] = {
    (ClaimBasis.OBSERVATION, ClaimBasis.OBSERVATION): EvidentialRelation.BOTH_OBSERVATION,
    (ClaimBasis.SECONDARY_SOURCE, ClaimBasis.SECONDARY_SOURCE): EvidentialRelation.BOTH_SECONDARY,
    (ClaimBasis.BACKGROUND_KNOWLEDGE, ClaimBasis.BACKGROUND_KNOWLEDGE): EvidentialRelation.BOTH_BACKGROUND,
    (ClaimBasis.OBSERVATION, ClaimBasis.SECONDARY_SOURCE): EvidentialRelation.A_OBSERVATION_B_SECONDARY,
    (ClaimBasis.SECONDARY_SOURCE, ClaimBasis.OBSERVATION): EvidentialRelation.A_SECONDARY_B_OBSERVATION,
    (ClaimBasis.OBSERVATION, ClaimBasis.BACKGROUND_KNOWLEDGE): EvidentialRelation.A_OBSERVATION_B_BACKGROUND,
    (ClaimBasis.BACKGROUND_KNOWLEDGE, ClaimBasis.OBSERVATION): EvidentialRelation.A_BACKGROUND_B_OBSERVATION,
    (ClaimBasis.SECONDARY_SOURCE, ClaimBasis.BACKGROUND_KNOWLEDGE): EvidentialRelation.A_SECONDARY_B_BACKGROUND,
    (ClaimBasis.BACKGROUND_KNOWLEDGE, ClaimBasis.SECONDARY_SOURCE): EvidentialRelation.A_BACKGROUND_B_SECONDARY,
}


def evidential_relation(claim_a: Claim, claim_b: Claim) -> EvidentialRelation:
    """How two claims' evidential standing relates, read from A to B.

    Derived from each claim's ``about`` and ``basis``; nothing is judged. A claim
    about a work rather than the domain has no basis, so the distinction does not
    apply; a domain claim whose basis the extractor did not give leaves the
    relation undetermined.

    Args:
        claim_a: The A-side claim.
        claim_b: The B-side claim.

    Returns:
        The evidential relation of A to B.

    The same conclusion can be a firsthand observation in one source and
    background knowledge in the other:

    >>> a = Claim(id="a", claim_text="X.", anchor_status="EXACT", citation_status="UNCITED",
    ...           about=ClaimTopic.DOMAIN, basis=ClaimBasis.OBSERVATION)
    >>> b = Claim(id="b", claim_text="X.", anchor_status="EXACT", citation_status="UNCITED",
    ...           about=ClaimTopic.DOMAIN, basis=ClaimBasis.BACKGROUND_KNOWLEDGE)
    >>> evidential_relation(a, b).value
    'A_OBSERVATION_B_BACKGROUND'
    >>> evidential_relation(b, a).value
    'A_BACKGROUND_B_OBSERVATION'

    A claim about a work does not take the distinction; a missing basis leaves it
    undetermined:

    >>> work = Claim(id="w", claim_text="A paper reports X.", anchor_status="EXACT",
    ...              citation_status="UNCITED", about=ClaimTopic.WORK)
    >>> evidential_relation(a, work).value
    'NOT_APPLICABLE'
    >>> no_basis = Claim(id="n", claim_text="X.", anchor_status="EXACT",
    ...                  citation_status="UNCITED", about=ClaimTopic.DOMAIN)
    >>> evidential_relation(a, no_basis).value
    'UNDETERMINED'
    """
    if claim_a.about == ClaimTopic.WORK or claim_b.about == ClaimTopic.WORK:
        return EvidentialRelation.NOT_APPLICABLE
    if claim_a.basis is None or claim_b.basis is None:
        return EvidentialRelation.UNDETERMINED
    return _EVIDENTIAL_BY_BASIS[(claim_a.basis, claim_b.basis)]


def shares_citation(claim_a: Claim, claim_b: Claim) -> bool:
    """Whether both claims cite a work with the same resolved reference_id.

    When true, agreement between the two claims is the same source echoed rather
    than independent corroboration. Only resolved ids count: an unresolved marker
    (``reference_id`` None) is not matched, because two reports' ``[1]`` markers
    are not the same work.

    Args:
        claim_a: The A-side claim.
        claim_b: The B-side claim.

    Returns:
        True if a resolved reference_id appears in both claims' citations.

    >>> from deep_research_client.claims import CitationHandle, CitationScope
    >>> def cite(ref):
    ...     return CitationHandle(marker=ref or "[1]", reference_id=ref, scope=CitationScope.SENTENCE)
    >>> a = Claim(id="a", claim_text="X.", anchor_status="EXACT", citation_status="CITED",
    ...           citations=[cite("PMID:1")])
    >>> b = Claim(id="b", claim_text="X.", anchor_status="EXACT", citation_status="CITED",
    ...           citations=[cite("PMID:1"), cite("PMID:2")])
    >>> shares_citation(a, b)
    True
    >>> c = Claim(id="c", claim_text="X.", anchor_status="EXACT", citation_status="CITED",
    ...           citations=[cite(None)])
    >>> shares_citation(a, c)
    False
    """
    ids_a = {c.reference_id for c in (claim_a.citations or []) if c.reference_id}
    ids_b = {c.reference_id for c in (claim_b.citations or []) if c.reference_id}
    return bool(ids_a & ids_b)


class ClaimAlignment(GeneratedClaimAlignment):
    """One source's claims aligned against another's, with derived views.

    A pair judged :data:`SemanticRelation.UNRELATED` is kept so the judgement is
    on record, but it does not count as an alignment: the aligned views and the
    coverage methods exclude it.

    >>> pair = AlignedClaimPair(
    ...     id="p1", claim_a_id="c1", claim_b_id="phenotypes[0]",
    ...     claim_a_text="A.", claim_b_text="A.",
    ...     semantic_relation=SemanticRelation.IDENTICAL,
    ...     evidential_relation=EvidentialRelation.A_OBSERVATION_B_SECONDARY,
    ...     candidate_signals=[CandidateSignal.ENTITY], confidence=0.9)
    >>> alignment = ClaimAlignment(
    ...     source_a=AlignedSource(id="a.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
    ...     source_b=AlignedSource(id="b.yaml", source_type=AlignedSourceType.STRUCTURED_DOCUMENT),
    ...     aligner=AlignerInfo(name="llm-pairwise"), pairs=[pair])
    >>> sorted(alignment.aligned_a_ids)
    ['c1']
    >>> alignment.unaligned_a_ids(["c1", "c2"])
    ['c2']
    """

    @model_validator(mode="after")
    def _confidence_in_range(self) -> "ClaimAlignment":
        """Refuse a pair whose stated confidence is outside 0..1."""
        for pair in self.pair_list:
            if pair.confidence is not None and not 0.0 <= pair.confidence <= 1.0:
                raise ValueError(
                    f"{pair.id} has confidence {pair.confidence}, outside 0..1"
                )
        return self

    @property
    def pair_list(self) -> list[AlignedClaimPair]:
        """Every pair, as a list even when the slot is unset."""
        return self.pairs or []

    @property
    def aligned_pairs(self) -> list[AlignedClaimPair]:
        """Pairs that align the two claims (semantic relation is not UNRELATED)."""
        return [p for p in self.pair_list if p.semantic_relation != SemanticRelation.UNRELATED]

    @property
    def aligned_a_ids(self) -> set[str]:
        """A-side claim ids that align to at least one B-side claim."""
        return {p.claim_a_id for p in self.aligned_pairs}

    @property
    def aligned_b_ids(self) -> set[str]:
        """B-side claim ids that align to at least one A-side claim."""
        return {p.claim_b_id for p in self.aligned_pairs}

    @property
    def contradictions(self) -> list[AlignedClaimPair]:
        """Pairs whose claims cannot both be true."""
        return [p for p in self.pair_list if p.semantic_relation == SemanticRelation.CONTRADICTORY]

    def unaligned_a_ids(self, a_claim_ids: list[str]) -> list[str]:
        """A-side claim ids, in the given order, that align to nothing in B.

        Args:
            a_claim_ids: Every claim id in source A's ClaimSet, in order.

        Returns:
            Those with no aligning pair: what source A contributes that B does not.
        """
        aligned = self.aligned_a_ids
        return [cid for cid in a_claim_ids if cid not in aligned]

    def unaligned_b_ids(self, b_claim_ids: list[str]) -> list[str]:
        """B-side claim ids, in the given order, that align to nothing in A.

        Args:
            b_claim_ids: Every claim id in source B's ClaimSet, in order.

        Returns:
            Those with no aligning pair: what source B holds that A does not cover.
        """
        aligned = self.aligned_b_ids
        return [cid for cid in b_claim_ids if cid not in aligned]


class RubricSummary(GeneratedRubricSummary):
    """A report-level summary of an alignment, with its metrics checked.

    >>> summary = RubricSummary(reference="b", metrics=[
    ...     RubricMetric(name="completeness", numerator=1, denominator=2, value=0.5)])
    >>> summary.metric("completeness").value
    0.5
    >>> summary.metric("missing") is None
    True
    """

    @model_validator(mode="after")
    def _metrics_are_consistent(self) -> "RubricSummary":
        """Refuse a metric whose value does not follow from its counts."""
        for metric in self.metrics or []:
            if metric.numerator < 0 or metric.numerator > metric.denominator:
                raise ValueError(
                    f"{metric.name}: numerator {metric.numerator} is not in 0..{metric.denominator}"
                )
            if metric.denominator == 0:
                if metric.value is not None:
                    raise ValueError(f"{metric.name}: denominator is 0 but value is {metric.value}")
            elif metric.value is None or abs(metric.value - metric.numerator / metric.denominator) > 1e-9:
                raise ValueError(
                    f"{metric.name}: value {metric.value} is not {metric.numerator}/{metric.denominator}"
                )
        return self

    def metric(self, name: str) -> Optional[RubricMetric]:
        """The metric with this name, or None.

        Args:
            name: The metric name to look up.

        Returns:
            The first metric so named, or None if there is none.
        """
        for metric in self.metrics or []:
            if metric.name == name:
                return metric
        return None
