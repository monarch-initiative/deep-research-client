"""Reference validation backed by ``linkml-reference-validator``.

Deep research providers routinely emit PMIDs and DOIs that look plausible but do
not resolve, and quotes that do not appear in the paper they are attributed to.
:class:`ReferenceValidator` resolves every identifier a report cites and checks
every attributed quote against the source text, so a report can be shipped with
its references already checked rather than with an unverified identifier list.

``linkml-reference-validator`` is an optional dependency; install it with::

    pip install "deep_research_client[validation]"
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

from .extraction import FoundReference, QuotedClaim, extract_quoted_claims, extract_references
from .models import (
    ReferenceCheck,
    ReferenceStatus,
    ReferenceValidationReport,
    SupportingTextCheck,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from ..models import ResearchResult

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "linkml-reference-validator is required for reference validation. "
    'Install it with: pip install "deep_research_client[validation]"'
)


@dataclass
class ReferenceValidator:
    """Resolve and check the references cited by a research report.

    Attributes:
        cache_dir: Directory where fetched references are cached. Reusing one
            directory across runs avoids re-fetching the same papers.
        email: Contact address sent to the NCBI Entrez API, which requires one.
        fetch_full_text: Whether to try to retrieve full text as well as
            abstracts. Slower, but quote checking is far more accurate with it.
        rate_limit_delay: Seconds to wait between upstream API requests.
        skip_prefixes: Identifier prefixes to report as unverifiable instead of
            attempting to resolve.
        max_references: Optional cap on the number of references resolved, for
            reports with very long bibliographies.

    Examples:
        >>> validator = ReferenceValidator(email="me@example.org")
        >>> validator.rate_limit_delay
        0.5
    """

    cache_dir: Optional[Union[str, Path]] = None
    email: Optional[str] = None
    fetch_full_text: bool = False
    rate_limit_delay: float = 0.5
    skip_prefixes: list[str] = field(default_factory=list)
    max_references: Optional[int] = None

    def validate_result(
        self,
        result: "ResearchResult",
        check_quotes: bool = True,
    ) -> ReferenceValidationReport:
        """Validate the references cited by a :class:`ResearchResult`.

        Args:
            result: The research result to check.
            check_quotes: Whether to also check quoted claims against sources.

        Returns:
            The validation report.
        """
        return self.validate_markdown(
            result.markdown,
            citations=result.citations,
            check_quotes=check_quotes,
        )

    def validate_markdown(
        self,
        markdown: str,
        citations: Optional[Iterable[str]] = None,
        check_quotes: bool = True,
        quote_pattern: Optional[re.Pattern[str]] = None,
    ) -> ReferenceValidationReport:
        """Validate the references cited in a block of report markdown.

        Args:
            markdown: The report body.
            citations: Optional citation strings scanned for identifiers.
            check_quotes: Whether to also check quoted claims against sources.
            quote_pattern: Optional override for the quoted-claim pattern.

        Returns:
            The validation report.
        """
        references = extract_references(markdown, citations)
        quoted_claims = extract_quoted_claims(markdown, quote_pattern) if check_quotes else []
        return self.validate_references(references, quoted_claims)

    def validate_references(
        self,
        references: Iterable[FoundReference],
        quoted_claims: Iterable[QuotedClaim] = (),
    ) -> ReferenceValidationReport:
        """Validate an explicit set of references and quoted claims.

        Args:
            references: References to resolve.
            quoted_claims: Quotes to check against their cited reference.

        Returns:
            The validation report.

        Raises:
            ImportError: If the optional ``validation`` extra is not installed.
        """
        if not validator_is_available():
            raise ImportError(INSTALL_HINT)

        references = list(references)
        quoted_claims = list(quoted_claims)

        truncated = False
        if self.max_references is not None and len(references) > self.max_references:
            logger.warning(
                "Checking only the first %d of %d references",
                self.max_references,
                len(references),
            )
            references = references[: self.max_references]
            truncated = True

        text_validator = self._build_text_validator()

        report = ReferenceValidationReport(
            validator_version=validator_version(),
            truncated=truncated,
        )

        for index, reference in enumerate(references, start=1):
            logger.info(
                "Resolving reference %d/%d: %s", index, len(references), reference.normalized_id
            )
            report.references.append(self._check_reference(text_validator, reference))

        status_by_id = {check.reference_id: check.status for check in report.references}

        for claim in quoted_claims:
            status = status_by_id.get(claim.reference_id)
            if status is None:
                # The reference was dropped by max_references; checking its quote
                # would re-open the network work the cap exists to avoid.
                continue
            if status == ReferenceStatus.UNVERIFIABLE:
                continue
            if status == ReferenceStatus.NOT_FOUND:
                # The reference itself is already reported as unresolved; re-fetching
                # it to confirm the quote cannot be checked would just repeat the miss.
                report.supporting_text.append(
                    SupportingTextCheck(
                        reference_id=claim.reference_id,
                        quote=claim.quote,
                        is_valid=False,
                        message="Reference did not resolve, so the quote cannot be checked",
                    )
                )
                continue
            logger.info("Checking quoted claim attributed to %s", claim.reference_id)
            report.supporting_text.append(self._check_quote(text_validator, claim))

        return report

    def _build_text_validator(self):
        """Construct the underlying ``SupportingTextValidator``.

        Returns:
            A configured ``linkml_reference_validator.SupportingTextValidator``.
        """
        from linkml_reference_validator.models import ReferenceValidationConfig
        from linkml_reference_validator.validation.supporting_text_validator import (
            SupportingTextValidator,
        )

        config_kwargs: dict = {
            "fetch_full_text": self.fetch_full_text,
            "rate_limit_delay": self.rate_limit_delay,
            "skip_prefixes": list(self.skip_prefixes),
        }
        if self.cache_dir is not None:
            config_kwargs["cache_dir"] = Path(self.cache_dir)
        if self.email is not None:
            config_kwargs["email"] = self.email

        return SupportingTextValidator(ReferenceValidationConfig(**config_kwargs))

    def _check_reference(self, text_validator, reference: FoundReference) -> ReferenceCheck:
        """Resolve a single reference identifier.

        Args:
            text_validator: The underlying supporting text validator.
            reference: The reference to resolve.

        Returns:
            The per-reference result.
        """
        from linkml_reference_validator.etl.sources.base import ReferenceSourceRegistry

        reference_id = reference.normalized_id
        prefix = reference_id.split(":", 1)[0]

        if prefix.upper() in {p.upper() for p in self.skip_prefixes}:
            return ReferenceCheck(
                reference_id=reference_id,
                status=ReferenceStatus.UNVERIFIABLE,
                occurrences=reference.count,
                message=f"Prefix '{prefix}' is configured to be skipped",
            )

        fetcher = text_validator.fetcher
        if ReferenceSourceRegistry.get_source(fetcher.normalize_reference_id(reference_id)) is None:
            return ReferenceCheck(
                reference_id=reference_id,
                status=ReferenceStatus.UNVERIFIABLE,
                occurrences=reference.count,
                message=f"No resolver is available for prefix '{prefix}'",
            )

        content = fetcher.fetch(reference_id)

        if content is None:
            return ReferenceCheck(
                reference_id=reference_id,
                status=ReferenceStatus.NOT_FOUND,
                occurrences=reference.count,
                message="Identifier did not resolve to a record",
            )

        message = None
        if not content.content:
            message = "Record resolved but no abstract or full text is available to check quotes against"

        return ReferenceCheck(
            reference_id=reference_id,
            status=ReferenceStatus.VERIFIED,
            occurrences=reference.count,
            title=content.title,
            year=content.year,
            journal=content.journal,
            doi=content.doi,
            content_type=content.content_type,
            message=message,
        )

    def _check_quote(self, text_validator, claim: QuotedClaim) -> SupportingTextCheck:
        """Check one quoted claim against the text of its reference.

        Args:
            text_validator: The underlying supporting text validator.
            claim: The quote and the reference it is attributed to.

        Returns:
            The per-quote result.
        """
        result = text_validator.validate(claim.quote, claim.reference_id)
        match = result.match_result

        return SupportingTextCheck(
            reference_id=claim.reference_id,
            quote=claim.quote,
            is_valid=result.is_valid,
            similarity_score=match.similarity_score if match else 0.0,
            matched_text=match.matched_text if match else None,
            best_match=match.best_match if match else None,
            suggested_fix=match.suggested_fix if match else None,
            message=result.message,
        )


def validator_is_available() -> bool:
    """Return whether the optional ``validation`` extra is installed.

    Examples:
        >>> isinstance(validator_is_available(), bool)
        True
    """
    from importlib.util import find_spec

    return find_spec("linkml_reference_validator") is not None


def validator_version() -> Optional[str]:
    """Return the installed ``linkml-reference-validator`` version, if any.

    The distribution metadata is authoritative; the package's own
    ``__version__`` attribute is not populated in released wheels.

    Examples:
        >>> version = validator_version()
        >>> version is None or isinstance(version, str)
        True
    """
    from importlib.metadata import PackageNotFoundError, version

    if not validator_is_available():
        return None
    try:
        return version("linkml-reference-validator")
    except PackageNotFoundError:  # pragma: no cover - installed without metadata
        return None
