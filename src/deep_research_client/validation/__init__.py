"""Reference validation for deep research reports.

Checks that the PMIDs and DOIs a report cites actually resolve, and that quotes
attributed to a reference really appear in it, using
`linkml-reference-validator <https://pypi.org/project/linkml-reference-validator/>`_.
Each resolved record is additionally weighed against the report's own vocabulary,
so a citation that exists but is about something else does not pass unremarked.

The extraction and report models here have no third-party requirements; only
:class:`~deep_research_client.validation.validator.ReferenceValidator` needs the
optional ``validation`` extra.
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
    VALIDATION_SECTION_HEADING,
    ReferenceCheck,
    ReferenceStatus,
    ReferenceValidationReport,
    SupportingTextCheck,
    TopicalRelevance,
    strip_validation_section,
)
from .relevance import (
    RelevanceAssessment,
    ScoredTerm,
    assess_relevance,
    extract_keywords,
    reference_text,
)
from .validator import INSTALL_HINT, ReferenceValidator, validator_is_available

__all__ = [
    "ExtractedEvidence",
    "FoundReference",
    "INSTALL_HINT",
    "QuotedClaim",
    "ReferenceCheck",
    "ReferenceStatus",
    "ReferenceValidationReport",
    "ReferenceValidator",
    "RelevanceAssessment",
    "ScoredTerm",
    "SupportingTextCheck",
    "TopicalRelevance",
    "VALIDATION_SECTION_HEADING",
    "assess_relevance",
    "extract_evidence",
    "extract_keywords",
    "extract_quoted_claims",
    "extract_references",
    "find_reference_ids",
    "reference_text",
    "strip_validation_section",
    "validator_is_available",
]
