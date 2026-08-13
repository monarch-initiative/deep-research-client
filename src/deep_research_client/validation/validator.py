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

if TYPE_CHECKING:  # pragma: no cover - imports only for type checking
    from linkml_reference_validator.validation.supporting_text_validator import (
        SupportingTextValidator,
    )

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

        reference_checks: list[ReferenceCheck] = []
        # Records whether each reference offered any text to match a quote
        # against, so a quote is never called "not found" when in truth there was
        # nothing to look in.
        has_content: dict[str, bool] = {}

        for index, reference in enumerate(references, start=1):
            logger.info(
                "Resolving reference %d/%d: %s", index, len(references), reference.normalized_id
            )
            check, checkable = self._check_reference(text_validator, reference)
            reference_checks.append(check)
            has_content[check.reference_id] = checkable

        reference_checks = _reclassify_truncated_dois(reference_checks)
        status_by_id = {check.reference_id: check.status for check in reference_checks}

        quote_checks = [
            self._resolve_quote(text_validator, claim, status_by_id, has_content, truncated)
            for claim in quoted_claims
        ]

        return ReferenceValidationReport(
            references=reference_checks,
            supporting_text=quote_checks,
            validator_version=validator_version(),
            truncated=truncated,
        )

    def _resolve_quote(
        self,
        text_validator: "SupportingTextValidator",
        claim: QuotedClaim,
        status_by_id: dict[str, ReferenceStatus],
        has_content: dict[str, bool],
        truncated: bool,
    ) -> SupportingTextCheck:
        """Check one quoted claim, or record why it could not be checked.

        Every extracted quote produces a result, so a report never quietly omits
        one. Cases with nothing to compare against are marked
        ``was_checkable=False`` rather than being reported as unsupported.

        Args:
            text_validator: The underlying supporting text validator.
            claim: The quote and the reference it is attributed to.
            status_by_id: Resolution status of each reference already checked.
            has_content: Whether each reference exposed text to search.
            truncated: Whether a reference limit dropped part of the bibliography.

        Returns:
            The per-quote result.
        """
        status = status_by_id.get(claim.reference_id)

        if status is None:
            # Usually dropped by max_references, in which case checking it would
            # re-open exactly the network work the cap exists to avoid. Without a
            # limit in play it means the quote's citation and the body scan
            # disagreed about the identifier, which is worth saying plainly
            # rather than blaming a limit that was never set.
            reason = (
                "Reference was not checked because the reference limit was reached"
                if truncated
                else "Reference was not among those extracted from the report body"
            )
            return self._unchecked_quote(claim, reason)
        if status == ReferenceStatus.UNVERIFIABLE:
            return self._unchecked_quote(
                claim, "Reference was skipped, so the quote was not checked"
            )
        if status == ReferenceStatus.NOT_FOUND:
            # The reference is already reported as unresolved; re-fetching it to
            # confirm the quote cannot be checked would just repeat the miss.
            return self._unchecked_quote(
                claim, "Reference did not resolve, so the quote could not be checked"
            )
        if not has_content.get(claim.reference_id, False):
            return self._unchecked_quote(
                claim,
                "Reference resolved but exposes no abstract or full text to search",
            )

        logger.info("Checking quoted claim attributed to %s", claim.reference_id)
        return self._check_quote(text_validator, claim)

    @staticmethod
    def _unchecked_quote(claim: QuotedClaim, message: str) -> SupportingTextCheck:
        """Build a result for a quote there was nothing to check against."""
        return SupportingTextCheck(
            reference_id=claim.reference_id,
            quote=claim.quote,
            was_checkable=False,
            is_valid=False,
            message=message,
        )

    def _build_text_validator(self) -> "SupportingTextValidator":
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

    def _check_reference(
        self,
        text_validator: "SupportingTextValidator",
        reference: FoundReference,
    ) -> tuple[ReferenceCheck, bool]:
        """Resolve a single reference identifier.

        Args:
            text_validator: The underlying supporting text validator.
            reference: The reference to resolve.

        Returns:
            The per-reference result, and whether the record exposed any text
            that a quote could be searched for.
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
            ), False

        fetcher = text_validator.fetcher
        if ReferenceSourceRegistry.get_source(fetcher.normalize_reference_id(reference_id)) is None:
            return ReferenceCheck(
                reference_id=reference_id,
                status=ReferenceStatus.UNVERIFIABLE,
                occurrences=reference.count,
                message=f"No resolver is available for prefix '{prefix}'",
            ), False

        content = fetcher.fetch(reference_id)

        if content is None:
            return ReferenceCheck(
                reference_id=reference_id,
                status=ReferenceStatus.NOT_FOUND,
                occurrences=reference.count,
                # Deliberately non-committal: the upstream fetcher returns None
                # both for "no such record" and for a transport failure it has
                # already swallowed, and this message is quoted verbatim in a
                # section that accuses citations of being fabricated.
                message="Identifier did not resolve to a record",
            ), False

        has_content = bool(content.content)
        message = None
        if not has_content:
            message = (
                "Record resolved but no abstract or full text is available to "
                "check quotes against"
            )

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
        ), has_content

    def _check_quote(
        self,
        text_validator: "SupportingTextValidator",
        claim: QuotedClaim,
    ) -> SupportingTextCheck:
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
            was_checkable=True,
            is_valid=result.is_valid,
            similarity_score=_clamp_similarity(match.similarity_score if match else 0.0),
            matched_text=match.matched_text if match else None,
            best_match=match.best_match if match else None,
            suggested_fix=match.suggested_fix if match else None,
            message=result.message,
        )


def _reclassify_truncated_dois(checks: list[ReferenceCheck]) -> list[ReferenceCheck]:
    """Demote a DOI that is a cut-short copy of another DOI in the same report.

    Deep research tools mangle their own citation lists: an Edison report was
    observed citing ``https://doi.org/10.1016/0092-8674(94)90302-6`` correctly in
    its body and ``https://doi.org/10.1016/0092-8674(94`` in its reference list.
    The truncated form resolves to nothing, so it would be listed as a possible
    fabrication - a false accusation against a citation the report got right
    elsewhere.

    When an unresolved DOI is a strict prefix of a DOI that did resolve, it is
    almost certainly the same identifier, mangled. Such a reference becomes
    UNVERIFIABLE: reported, but not counted as a fabrication and not enough to
    fail a build.

    Args:
        checks: Per-reference results, before quotes are resolved.

    Returns:
        The same results, with truncated duplicates demoted.

    Examples:
        >>> resolved = ReferenceCheck(
        ...     reference_id="DOI:10.1016/0092-8674(94)90302-6",
        ...     status=ReferenceStatus.VERIFIED,
        ... )
        >>> truncated = ReferenceCheck(
        ...     reference_id="DOI:10.1016/0092-8674(94",
        ...     status=ReferenceStatus.NOT_FOUND,
        ... )
        >>> out = _reclassify_truncated_dois([resolved, truncated])
        >>> out[1].status == ReferenceStatus.UNVERIFIABLE
        True
        >>> "truncated" in out[1].message
        True
    """
    verified_dois = [
        check.reference_id
        for check in checks
        if check.status == ReferenceStatus.VERIFIED
        and check.reference_id.upper().startswith("DOI:")
    ]
    if not verified_dois:
        return checks

    result: list[ReferenceCheck] = []
    for check in checks:
        longer = None
        if check.status == ReferenceStatus.NOT_FOUND and check.reference_id.upper().startswith(
            "DOI:"
        ):
            longer = next(
                (
                    candidate
                    for candidate in verified_dois
                    if candidate != check.reference_id
                    and candidate.startswith(check.reference_id)
                ),
                None,
            )
        if longer is None:
            result.append(check)
            continue

        logger.info(
            "Treating %s as a truncated copy of %s rather than a fabrication",
            check.reference_id,
            longer,
        )
        result.append(
            check.model_copy(
                update={
                    "status": ReferenceStatus.UNVERIFIABLE,
                    "message": (
                        f"Looks like a truncated copy of {longer}, which resolved; "
                        "not counted as a possible fabrication"
                    ),
                }
            )
        )
    return result


def _clamp_similarity(score: float) -> float:
    """Clamp a similarity score into the 0-1 range the schema enforces.

    The score comes from ``linkml-reference-validator``, so the range is not
    ours to guarantee. Without this, one out-of-range float would raise a
    ``ValidationError`` mid-loop and discard a report that may have cost minutes
    of network fetches.

    Args:
        score: Similarity score reported by the underlying validator.

    Returns:
        The score, constrained to 0.0-1.0.

    Examples:
        >>> _clamp_similarity(0.5)
        0.5
        >>> _clamp_similarity(1.0000000002)
        1.0
        >>> _clamp_similarity(-0.1)
        0.0
    """
    return max(0.0, min(1.0, score))


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
