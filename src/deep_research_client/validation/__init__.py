"""Reference validation for deep research reports.

Extraction of reference identifiers and the report models describing what was
checked. Both have no third-party requirements.
"""

from .extraction import (
    ExtractedEvidence,
    FoundReference,
    QuotedClaim,
    extract_evidence,
    extract_quoted_claims,
    extract_references,
    find_reference_ids,
)
from .models import (
    ReferenceCheck,
    ReferenceStatus,
    ReferenceValidationReport,
    SupportingTextCheck,
)

__all__ = [
    "ExtractedEvidence",
    "FoundReference",
    "QuotedClaim",
    "ReferenceCheck",
    "ReferenceStatus",
    "ReferenceValidationReport",
    "SupportingTextCheck",
    "extract_evidence",
    "extract_quoted_claims",
    "extract_references",
    "find_reference_ids",
]
