"""Claim extraction: the specific assertions a source makes (issue #43).

A claim is one atomic assertion with enough provenance for a person to find it
in the source again. Extraction does not judge whether a claim is true;
alignment across sources and verification consume these records later.
"""

from .extract import SourceFormat, aextract_claims, detect_format, extract_claims
from .models import (
    AnchorStatus,
    CitationHandle,
    Claim,
    ClaimSet,
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
