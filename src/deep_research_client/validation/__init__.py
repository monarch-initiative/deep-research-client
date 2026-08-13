"""Reference validation for deep research reports.

Checks that the PMIDs and DOIs a report cites actually resolve, and that quotes
attributed to a reference really appear in it, using
`linkml-reference-validator <https://pypi.org/project/linkml-reference-validator/>`_.

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
    ReferenceCheck,
    ReferenceStatus,
    ReferenceValidationReport,
    SupportingTextCheck,
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
    "SupportingTextCheck",
    "extract_evidence",
    "extract_quoted_claims",
    "extract_references",
    "find_reference_ids",
    "validator_is_available",
]
