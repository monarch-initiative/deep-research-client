"""Claim models: the generated datamodel plus derived views.

The fields come from ``claims.yaml`` via ``datamodel.py``. ``ClaimSet`` is
subclassed here to add properties computed from those fields, and adds no
fields of its own, so its JSON schema stays exactly what LinkML would emit.

``ClaimSet`` also checks, on construction and on loading, the rules that tie a
claim's slots together (see :func:`inconsistencies`), so a set that says a
claim is uncited while listing its citations cannot be written or read back.

The generated base sets ``use_enum_values=True``, so an enum slot reads back as
its string value; compare with ``==`` against the enum member, which is a
``str`` subclass.
"""

import hashlib
from typing import Optional

from pydantic import model_validator

from .datamodel import (
    AnchorStatus,
    CitationHandle,
    CitationScope,
    CitationStatus,
    Claim,
    ClaimBasis,
    ClaimTopic,
    EntityMention,
    ExtractorInfo,
    GrounderInfo,
    NearestPassage,
    SourceDocument,
    SourceType,
    TextSpan,
)
from .datamodel import ClaimSet as GeneratedClaimSet

__all__ = [
    "AnchorStatus",
    "CitationHandle",
    "CitationScope",
    "CitationStatus",
    "Claim",
    "ClaimBasis",
    "ClaimSet",
    "ClaimTopic",
    "EntityMention",
    "ExtractorInfo",
    "GrounderInfo",
    "NearestPassage",
    "SourceDocument",
    "SourceType",
    "TextSpan",
    "citation_status_for",
    "content_sha256",
    "entity_mentions",
    "ids_by_section",
    "inconsistencies",
    "resolved_reference_ids",
]


def content_sha256(text: str) -> str:
    """Hash the exact text that claim offsets refer to.

    Args:
        text: The source text, as extracted from.

    Returns:
        Hex SHA-256 of its UTF-8 encoding.

    >>> content_sha256("abc")[:12]
    'ba7816bf8f01'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def citation_status_for(citations: list[CitationHandle], anchor_status: str) -> CitationStatus:
    """The citation status that citations found for a claim imply.

    Args:
        citations: The citations kept for the claim.
        anchor_status: How the claim was located.

    Returns:
        UNKNOWN for a claim that could not be located, else CITED or UNCITED.

    >>> citation_status_for([], AnchorStatus.UNANCHORED).value
    'UNKNOWN'
    >>> citation_status_for([CitationHandle(marker="[1]", scope=CitationScope.SENTENCE)], AnchorStatus.EXACT).value
    'CITED'
    >>> citation_status_for([], AnchorStatus.NOT_APPLICABLE).value
    'UNCITED'
    """
    if anchor_status == AnchorStatus.UNANCHORED:
        return CitationStatus.UNKNOWN
    return CitationStatus.CITED if citations else CitationStatus.UNCITED


def entity_mentions(claim: Claim) -> list[EntityMention]:
    """Every entity mention on a claim: its entities, plus its subject and object.

    One definition of "a mention", shared by grounding (which fills their ids) and
    the rubric (which scores how many were filled), so the two cannot drift.

    Args:
        claim: The claim to read.

    Returns:
        Its :class:`EntityMention`s: the ``entities`` list, then ``subject`` and
        ``object`` when present.

    >>> claim = Claim(id="c", claim_text="X.", anchor_status=AnchorStatus.EXACT,
    ...               citation_status=CitationStatus.UNCITED,
    ...               entities=[EntityMention(label="acne")],
    ...               subject=EntityMention(label="TLR2"))
    >>> [m.label for m in entity_mentions(claim)]
    ['acne', 'TLR2']
    """
    mentions = list(claim.entities or [])
    mentions += [m for m in (claim.subject, claim.object) if m is not None]
    return mentions


def resolved_reference_ids(claim: Claim) -> set[str]:
    """The resolved reference_ids a claim cites, ignoring unresolved markers.

    Two reports' bare ``[1]`` markers are not the same work, so only citations
    that resolved to an identifier count. Shared by candidate generation (the
    citation signal) and :func:`alignment_models.shares_citation`.

    Args:
        claim: The claim to read.

    Returns:
        Every non-empty ``reference_id`` on the claim's citations.

    >>> claim = Claim(id="c", claim_text="X.", anchor_status=AnchorStatus.EXACT,
    ...               citation_status=CitationStatus.CITED,
    ...               citations=[CitationHandle(marker="[1]", reference_id="PMID:1", scope=CitationScope.SENTENCE),
    ...                          CitationHandle(marker="[2]", scope=CitationScope.SENTENCE)])
    >>> sorted(resolved_reference_ids(claim))
    ['PMID:1']
    """
    return {c.reference_id for c in (claim.citations or []) if c.reference_id}


def inconsistencies(claim: Claim) -> list[str]:
    """What is wrong with how a claim's slots fit together, if anything.

    Only rules the code can check are here. Whether ``about`` and ``basis``
    are right is the extractor's judgement, and nothing checks it.

    Args:
        claim: The claim to check.

    Returns:
        One sentence per broken rule; empty when the claim is consistent.

    >>> claim = Claim(id="c1", claim_text="A.", anchor_status=AnchorStatus.EXACT,
    ...               citation_status=CitationStatus.CITED,
    ...               about=ClaimTopic.WORK, basis=ClaimBasis.OBSERVATION)
    >>> for problem in inconsistencies(claim):
    ...     print(problem)
    c1 is CITED but has no citations
    c1 has a basis, which only a DOMAIN claim has
    """
    problems = []
    expected = citation_status_for(claim.citations or [], claim.anchor_status)
    if claim.citation_status != expected:
        if claim.citations and claim.anchor_status != AnchorStatus.UNANCHORED:
            reason = "has citations"
        elif claim.anchor_status == AnchorStatus.UNANCHORED:
            reason = "is UNANCHORED" + (" and has citations" if claim.citations else "")
        else:
            reason = "has no citations"
        problems.append(f"{claim.id} is {claim.citation_status} but {reason}")
    if claim.basis is not None and claim.about != ClaimTopic.DOMAIN:
        problems.append(f"{claim.id} has a basis, which only a DOMAIN claim has")
    if claim.nearest_passage is not None and claim.anchor_status != AnchorStatus.UNANCHORED:
        problems.append(f"{claim.id} has a nearest passage, which only an UNANCHORED claim has")
    return problems


def ids_by_section(claims: list[Claim], max_sections: int = 5) -> str:
    """Claim ids grouped by section, for a message a person will act on.

    A claim with no section is listed by its id alone: a curated record,
    whose id is already its path in the file, or a claim from a report's text
    before its first heading. All such claims form one group, placed where the
    first of them appears.

    Args:
        claims: The claims to list, in extraction order.
        max_sections: Groups to name before the rest are only counted.

    Returns:
        "c1, c2 in <section>; c9", in order of first appearance.

    >>> claims = [Claim(id=i, claim_text="A.", anchor_status=AnchorStatus.EXACT,
    ...                 citation_status=CitationStatus.UNCITED, section=section)
    ...           for i, section in [("c1", "Papers > [3] X"), ("c2", "Papers > [3] X"), ("phenotypes[2]", None)]]
    >>> ids_by_section(claims)
    'c1, c2 in Papers > [3] X; phenotypes[2]'
    >>> ids_by_section(claims, max_sections=1)
    'c1, c2 in Papers > [3] X; and 1 more section'
    """
    grouped: dict[Optional[str], list[str]] = {}
    for claim in claims:
        grouped.setdefault(claim.section, []).append(claim.id)
    shown = [
        f"{', '.join(ids)} in {section}" if section else ", ".join(ids)
        for section, ids in list(grouped.items())[:max_sections]
    ]
    hidden = len(grouped) - max_sections
    if hidden > 0:
        shown.append(f"and {hidden} more section{'s' if hidden > 1 else ''}")
    return "; ".join(shown)


class ClaimSet(GeneratedClaimSet):
    """The claims extracted from one source, with derived views.

    >>> claims = ClaimSet(
    ...     source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
    ...     extractor=ExtractorInfo(name="example"),
    ...     claims=[
    ...         Claim(id="c1", claim_text="A causes B.", anchor_status=AnchorStatus.EXACT,
    ...               citation_status=CitationStatus.UNCITED,
    ...               source_span=TextSpan(start=0, end=11, text="A causes B.")),
    ...         Claim(id="c2", claim_text="C.", anchor_status=AnchorStatus.UNANCHORED,
    ...               citation_status=CitationStatus.UNKNOWN),
    ...     ],
    ... )
    >>> [c.id for c in claims.unanchored_claims]
    ['c2']
    >>> claims.mismatched_spans("A causes B. More.")
    []
    """

    @model_validator(mode="after")
    def _claims_are_consistent(self) -> "ClaimSet":
        """Refuse a set with a claim whose slots contradict each other."""
        problems = [problem for claim in self.claim_list for problem in inconsistencies(claim)]
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def claim_list(self) -> list[Claim]:
        """The claims, as a list even when the slot is unset.

        Returns:
            Every claim, in extraction order.
        """
        return self.claims or []

    @property
    def unanchored_claims(self) -> list[Claim]:
        """Claims whose quote could not be found in the source.

        Kept rather than dropped so the gap is visible; nothing about them
        should be taken as coming from the source.

        Returns:
            Claims with anchor status UNANCHORED.
        """
        return [c for c in self.claim_list if c.anchor_status == AnchorStatus.UNANCHORED]

    @property
    def cited_background_claims(self) -> list[Claim]:
        """Claims the extractor called background knowledge that carry a citation.

        Background knowledge is by definition stated without a citation, so
        one of the extractor's two answers is wrong: the basis, or which
        claims the sentence's marker supports. Nothing here decides which.

        Returns:
            Claims with basis BACKGROUND_KNOWLEDGE and citation status CITED.
        """
        return [
            c for c in self.claim_list
            if c.basis == ClaimBasis.BACKGROUND_KNOWLEDGE and c.citation_status == CitationStatus.CITED
        ]

    def mismatched_spans(self, source_text: str) -> list[Claim]:
        """Claims whose span no longer points at the text it records.

        Only meaningful for a markdown report, whose spans are offsets into the
        whole file; prose inside a structured document is offset into its own
        field, found by ``source_path``. Check ``source.content_sha256`` first
        when the file may have changed: a mismatch there explains every one
        of these.

        Args:
            source_text: The text the claims were extracted from.

        Returns:
            Claims with a span whose recorded text differs from
            ``source_text[start:end]``.
        """
        return [
            c
            for c in self.claim_list
            if c.source_span is not None
            and source_text[c.source_span.start:c.source_span.end] != c.source_span.text
        ]
