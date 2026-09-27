"""Claim models: the generated datamodel plus derived views.

The fields come from ``claims.yaml`` via ``datamodel.py``. ``ClaimSet`` is
subclassed here to add properties computed from those fields, and adds no
fields of its own, so its JSON schema stays exactly what LinkML would emit.

The generated base sets ``use_enum_values=True``, so an enum slot reads back as
its string value; compare with ``==`` against the enum member, which is a
``str`` subclass.
"""

import hashlib

from .datamodel import (
    AnchorStatus,
    CitationHandle,
    Claim,
    EntityMention,
    ExtractorInfo,
    SourceDocument,
    SourceType,
    TextSpan,
)
from .datamodel import ClaimSet as GeneratedClaimSet

__all__ = [
    "AnchorStatus",
    "CitationHandle",
    "Claim",
    "ClaimSet",
    "EntityMention",
    "ExtractorInfo",
    "SourceDocument",
    "SourceType",
    "TextSpan",
    "content_sha256",
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


class ClaimSet(GeneratedClaimSet):
    """The claims extracted from one source, with derived views.

    >>> claims = ClaimSet(
    ...     source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
    ...     extractor=ExtractorInfo(name="example"),
    ...     claims=[
    ...         Claim(id="c1", claim_text="A causes B.", anchor_status=AnchorStatus.EXACT,
    ...               source_span=TextSpan(start=0, end=11, text="A causes B.")),
    ...         Claim(id="c2", claim_text="C.", anchor_status=AnchorStatus.UNANCHORED),
    ...     ],
    ... )
    >>> [c.id for c in claims.unanchored_claims]
    ['c2']
    >>> claims.mismatched_spans("A causes B. More.")
    []
    """

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
