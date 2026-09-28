"""Claim extraction: the specific assertions a source makes (issue #43).

A claim is one atomic assertion with enough provenance for a person to find it
in the source again. Extraction does not judge whether a claim is true;
alignment across sources and verification consume these records later.
"""

from .extract import (
    SourceFormat,
    aextract_claims,
    detect_format,
    extract_claims,
    needs_llm,
    resolve_format,
)
from .models import (
    AnchorStatus,
    CitationHandle,
    CitationStatus,
    Claim,
    ClaimBasis,
    ClaimSet,
    ClaimTopic,
    EntityMention,
    ExtractorInfo,
    SourceDocument,
    SourceType,
    TextSpan,
    content_sha256,
)

__all__ = [
    "SourceFormat",
    "aextract_claims",
    "detect_format",
    "extract_claims",
    "needs_llm",
    "resolve_format",
    "AnchorStatus",
    "CitationHandle",
    "CitationStatus",
    "Claim",
    "ClaimBasis",
    "ClaimSet",
    "ClaimTopic",
    "EntityMention",
    "ExtractorInfo",
    "SourceDocument",
    "SourceType",
    "TextSpan",
    "content_sha256",
]
