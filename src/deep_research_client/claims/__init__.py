"""Claim extraction: the specific assertions a source makes (issue #43).

A claim is one atomic assertion with enough provenance for a person to find it
in the source again. Extraction does not judge whether a claim is true;
alignment across sources and verification consume these records later.
"""

from .align import (
    aalign_claim_sets,
    aalign_within,
    acompare_claims,
    align_claim_sets,
    align_within,
    compare_claims,
)
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
from .candidates import CandidatePair, candidate_pairs, candidate_pairs_within
from .extract import (
    SourceFormat,
    aextract_claims,
    detect_format,
    extract_claims,
    needs_llm,
    resolve_format,
)
from .grounding import (
    Grounder,
    GroundedTerm,
    GroundingCandidate,
    aground_claims,
    biomedical_ontology_config,
    ground_claims,
)
from .grounding_oak import OakGrounder
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
    GrounderInfo,
    NearestPassage,
    SourceDocument,
    SourceType,
    TextSpan,
    content_sha256,
)
from .rubrics import summarize_alignment
from .verification_models import (
    ClaimVerification,
    Decidability,
    FlagKind,
    ReviewFlag,
    VerificationReport,
    VerificationStatus,
    VerifiedSourceType,
    VerifierInfo,
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
    "aalign_claim_sets",
    "aalign_within",
    "acompare_claims",
    "align_claim_sets",
    "align_within",
    "compare_claims",
    "candidate_pairs",
    "candidate_pairs_within",
    "evidential_relation",
    "shares_citation",
    "AlignedClaimPair",
    "CandidatePair",
    "AlignedSource",
    "AlignedSourceType",
    "AlignerInfo",
    "CandidateSignal",
    "ClaimAlignment",
    "EvidentialRelation",
    "SemanticRelation",
    # Rubrics (issue #43, item 4)
    "summarize_alignment",
    "RubricMetric",
    "RubricSummary",
    # Grounding (issue #43, item 3)
    "aground_claims",
    "ground_claims",
    "biomedical_ontology_config",
    "Grounder",
    "GroundedTerm",
    "GroundingCandidate",
    "GrounderInfo",
    "OakGrounder",
    # Verification data model (issue #43, item 5)
    "ClaimVerification",
    "Decidability",
    "FlagKind",
    "ReviewFlag",
    "VerificationReport",
    "VerificationStatus",
    "VerifiedSourceType",
    "VerifierInfo",
]
