"""Claim extraction: the specific assertions a source makes (issue #43).

A claim is one atomic assertion with enough provenance for a person to find it
in the source again. Extraction does not judge whether a claim is true;
alignment across sources and verification consume these records later.
"""

from .alignment_models import (
    AlignedClaimPair,
    AlignedSource,
    AlignedSourceType,
    AlignerInfo,
    CandidateSignal,
    ClaimAlignment,
    EvidentialRelation,
    RubricMetric,
    RubricSummary,
    SemanticRelation,
    evidential_relation,
    shares_citation,
)
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
    CitationScope,
    CitationStatus,
    Claim,
    ClaimBasis,
    ClaimSet,
    ClaimTopic,
    EntityMention,
    ExtractorInfo,
    NearestPassage,
    SourceDocument,
    SourceType,
    TextSpan,
    content_sha256,
)

__all__ = [
    # Extraction (issue #43, item 1)
    "SourceFormat",
    "aextract_claims",
    "detect_format",
    "extract_claims",
    "needs_llm",
    "resolve_format",
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
    "NearestPassage",
    "SourceDocument",
    "SourceType",
    "TextSpan",
    "content_sha256",
    # Alignment data model and derived axes (issue #43, item 2)
    "evidential_relation",
    "shares_citation",
    "AlignedClaimPair",
    "AlignedSource",
    "AlignedSourceType",
    "AlignerInfo",
    "CandidateSignal",
    "ClaimAlignment",
    "EvidentialRelation",
    "SemanticRelation",
    "RubricMetric",
    "RubricSummary",
]
