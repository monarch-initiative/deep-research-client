"""Reference and ontology term validation for deep research reports.

Checks that the PMIDs and DOIs a report cites actually resolve, and that quotes
attributed to a reference really appear in it, using
`linkml-reference-validator <https://pypi.org/project/linkml-reference-validator/>`_.
Each resolved record is additionally weighed against the report's own vocabulary,
so a citation that exists but is about something else does not pass unremarked.

It also checks the ontology terms a report cites, which fail in a way citations
do not: ``NCIT:C16814`` is a real term, so every existence check passes it, and
it means Malaysia rather than the echocardiography the report claimed. So a
term's identifier is resolved *and* compared with the name the report gave it,
using `linkml-term-validator <https://pypi.org/project/linkml-term-validator/>`_.

The extraction and report models here have no third-party requirements; only
:class:`~deep_research_client.validation.validator.ReferenceValidator` needs the
optional ``validation`` extra, and only
:class:`~deep_research_client.validation.term_validator.TermValidator` needs the
optional ``terms`` extra.
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
from .label_matching import compare_labels, label_similarity
from .sections import (
    TERM_VALIDATION_SECTION_HEADING,
    render_with_sections,
    split_validation_sections,
)
from .term_extraction import FoundTerm, extract_terms, find_term_ids
from .term_models import LabelAgreement, TermCheck, TermStatus, TermValidationReport
from .term_validator import (
    DEFAULT_ADAPTER,
    TERM_INSTALL_HINT,
    TermValidator,
    lookup_error_types,
    term_validator_is_available,
)
from .validator import INSTALL_HINT, ReferenceValidator, validator_is_available

__all__ = [
    "assess_relevance",
    "compare_labels",
    "DEFAULT_ADAPTER",
    "extract_evidence",
    "extract_keywords",
    "extract_quoted_claims",
    "extract_references",
    "extract_terms",
    "ExtractedEvidence",
    "find_reference_ids",
    "find_term_ids",
    "FoundReference",
    "FoundTerm",
    "INSTALL_HINT",
    "label_similarity",
    "LabelAgreement",
    "lookup_error_types",
    "QuotedClaim",
    "reference_text",
    "ReferenceCheck",
    "ReferenceStatus",
    "ReferenceValidationReport",
    "ReferenceValidator",
    "RelevanceAssessment",
    "render_with_sections",
    "ScoredTerm",
    "split_validation_sections",
    "strip_validation_section",
    "SupportingTextCheck",
    "TERM_INSTALL_HINT",
    "TERM_VALIDATION_SECTION_HEADING",
    "term_validator_is_available",
    "TermCheck",
    "TermStatus",
    "TermValidationReport",
    "TermValidator",
    "TopicalRelevance",
    "VALIDATION_SECTION_HEADING",
    "validator_is_available",
]
