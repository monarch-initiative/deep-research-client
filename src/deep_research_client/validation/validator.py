"""Reference validation backed by ``linkml-reference-validator``.

Deep research providers routinely emit PMIDs and DOIs that look plausible but do
not resolve, and quotes that do not appear in the paper they are attributed to.
:class:`ReferenceValidator` resolves every identifier a report cites and checks
every attributed quote against the source text, so a report can be shipped with
its references already checked rather than with an unverified identifier list.

A resolved identifier is still only a proof of existence, so each resolved record
is additionally weighed against the report's own vocabulary
(:mod:`deep_research_client.validation.relevance`). That catches the citation
that is real, quotable and about something else entirely.

``linkml-reference-validator`` is an optional dependency; install it with::

    pip install "deep_research_client[validation]"
"""

import logging
import re
import time
from itertools import batched
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

import httpx

from .extraction import FoundReference, QuotedClaim, extract_quoted_claims, extract_references
from .models import (
    ReferenceCheck,
    ReferenceStatus,
    ReferenceValidationReport,
    SupportingTextCheck,
)
from .relevance import (
    DEFAULT_KEYWORD_COUNT,
    ScoredTerm,
    TopicalRelevance,
    assess_relevance,
    extract_keywords,
    reference_text,
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
        check_relevance: Whether to weigh each resolved record against the
            report's own vocabulary. Costs no extra requests: the record has
            already been fetched to check that it resolves.
        keyword_count: How many of the report's terms relevance is judged
            against.

    Examples:
        >>> validator = ReferenceValidator(email="me@example.org")
        >>> validator.rate_limit_delay
        0.5
        >>> validator.check_relevance
        True
    """

    cache_dir: Optional[Union[str, Path]] = None
    email: Optional[str] = None
    fetch_full_text: bool = False
    rate_limit_delay: float = 0.5
    skip_prefixes: list[str] = field(default_factory=list)
    max_references: Optional[int] = None
    check_relevance: bool = True
    keyword_count: int = DEFAULT_KEYWORD_COUNT

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
        return self.validate_references(references, quoted_claims, topic_text=markdown)

    def validate_references(
        self,
        references: Iterable[FoundReference],
        quoted_claims: Iterable[QuotedClaim] = (),
        topic_text: Optional[str] = None,
    ) -> ReferenceValidationReport:
        """Validate an explicit set of references and quoted claims.

        Args:
            references: References to resolve.
            quoted_claims: Quotes to check against their cited reference.
            topic_text: The report the references were cited by, used to judge
                whether each resolved record is about the same subject. Without
                it, relevance is reported as ``NOT_ASSESSED``.

        Returns:
            The validation report.

        Raises:
            ImportError: If the optional ``validation`` extra is not installed.
        """
        if not validator_is_available():
            raise ImportError(INSTALL_HINT)

        references = list(references)
        quoted_claims = list(quoted_claims)
        keywords = (
            extract_keywords(topic_text, self.keyword_count)
            if self.check_relevance and topic_text
            else []
        )

        truncated = False
        if self.max_references is not None and len(references) > self.max_references:
            logger.warning(
                "Checking only the first %d of %d references",
                self.max_references,
                len(references),
            )
            references = references[: self.max_references]
            truncated = True

        # Full text exists to be searched for quotes. Fetching it when there is
        # no quote to check costs roughly twenty times the runtime and cannot
        # change a single verdict, so --full-text --no-check-quotes does not pay
        # for what it cannot use.
        want_full_text = self.fetch_full_text and bool(quoted_claims)
        if self.fetch_full_text and not quoted_claims:
            logger.info("No quoted claims to check; skipping full-text retrieval")
        text_validator = self._build_text_validator(fetch_full_text=want_full_text)

        # PMC accessions are resolved to a PMID (or DOI) in batched calls, so the
        # rest of the run uses identifiers the library already handles. Prefixes
        # the caller asked to skip are left out: --skip-prefix PMC means "do not
        # look these up", which has to include this lookup.
        skipped = {p.upper() for p in self.skip_prefixes}
        pmc_skipped = "PMC" in skipped
        pmc_aliases = resolve_pmc_accessions(
            []
            if pmc_skipped
            else [
                r.normalized_id
                for r in references
                if r.normalized_id.upper().startswith("PMC:")
            ],
            email=self.email,
            rate_limit_delay=self.rate_limit_delay,
        )

        reference_checks: list[ReferenceCheck] = []
        # Records whether each reference offered any text to match a quote
        # against, so a quote is never called "not found" when in truth there was
        # nothing to look in.
        has_content: dict[str, bool] = {}

        for index, reference in enumerate(references, start=1):
            logger.info(
                "Resolving reference %d/%d: %s", index, len(references), reference.normalized_id
            )
            check, checkable = self._check_reference(
                text_validator, reference, pmc_aliases, keywords
            )
            reference_checks.append(check)
            has_content[check.reference_id] = checkable

        reference_checks = _reclassify_truncated_dois(reference_checks)
        reference_checks = _withhold_off_topic_when_nothing_matched(reference_checks)
        status_by_id = {check.reference_id: check.status for check in reference_checks}
        content_type_by_id = {
            check.reference_id: check.content_type for check in reference_checks
        }

        quote_checks = [
            self._resolve_quote(
                text_validator,
                claim,
                status_by_id,
                has_content,
                truncated,
                pmc_aliases,
                pmc_skipped,
                content_type_by_id,
            )
            for claim in quoted_claims
        ]

        return ReferenceValidationReport(
            references=reference_checks,
            supporting_text=quote_checks,
            report_keywords=[keyword.term for keyword in keywords],
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
        pmc_aliases: dict[str, Optional[str]],
        pmc_skipped: bool,
        content_type_by_id: dict[str, Optional[str]],
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
            pmc_skipped: Whether the caller asked for PMC to be skipped, which is
                why its accessions are absent from ``pmc_aliases``.
            content_type_by_id: What kind of text each reference exposed, so a
                failed quote can say how much of the paper was actually searched.

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
            # A PMC accession missing from the alias map means the converter
            # could not be reached - unless the caller asked to skip PMC, in
            # which case no request was made and claiming an outage would be a
            # false statement in a user-facing report.
            unreachable = (
                not pmc_skipped
                and claim.reference_id.upper().startswith("PMC:")
                and claim.reference_id not in pmc_aliases
            )
            unverifiable_reason = (
                "The PMC ID service was unreachable, so the quote was not checked"
                if unreachable
                else "Reference was skipped, so the quote was not checked"
            )
            return self._unchecked_quote(claim, unverifiable_reason)
        if status == ReferenceStatus.NOT_FOUND:
            # The reference is already reported as unresolved; re-fetching it to
            # confirm the quote cannot be checked would just repeat the miss.
            return self._unchecked_quote(
                claim, "Reference did not resolve, so the quote could not be checked"
            )

        lookup_id = pmc_aliases.get(claim.reference_id) or claim.reference_id

        if not has_content.get(claim.reference_id, False):
            # No body text to search - but the title is still evidence, and a
            # record that resolved with a title and no abstract is ordinary for
            # book chapters, conference items and many DataCite DOIs. Confirming
            # against a title is safe here; contradicting a quote on the strength
            # of a title alone would not be, hence the fall-through.
            if self._quotes_the_title(text_validator, claim, lookup_id):
                return self._title_quote(claim)
            return self._unchecked_quote(
                claim,
                "Reference resolved but exposes no abstract or full text to search",
            )

        logger.info("Checking quoted claim attributed to %s", claim.reference_id)
        return self._check_quote(
            text_validator,
            claim,
            lookup_id,
            content_type_by_id.get(claim.reference_id),
        )

    @staticmethod
    def _title_quote(claim: QuotedClaim) -> SupportingTextCheck:
        """Build a result for a quote that is the cited reference's title."""
        return SupportingTextCheck(
            reference_id=claim.reference_id,
            quote=claim.quote,
            was_checkable=True,
            is_valid=True,
            similarity_score=1.0,
            matched_text=claim.quote,
            message="Quoted text is the title of the cited reference",
        )

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

    def _build_text_validator(
        self, fetch_full_text: Optional[bool] = None
    ) -> "SupportingTextValidator":
        """Construct the underlying ``SupportingTextValidator``.

        Args:
            fetch_full_text: Override for the configured setting, so a run with
                nothing to search can skip the expensive retrieval.

        Returns:
            A configured ``linkml_reference_validator.SupportingTextValidator``.
        """
        from linkml_reference_validator.models import ReferenceValidationConfig
        from linkml_reference_validator.validation.supporting_text_validator import (
            SupportingTextValidator,
        )

        config_kwargs: dict = {
            "fetch_full_text": (
                self.fetch_full_text if fetch_full_text is None else fetch_full_text
            ),
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
        pmc_aliases: dict[str, Optional[str]],
        keywords: Sequence[ScoredTerm] = (),
    ) -> tuple[ReferenceCheck, bool]:
        """Resolve a single reference identifier.

        Args:
            text_validator: The underlying supporting text validator.
            reference: The reference to resolve.
            pmc_aliases: PMC accessions mapped to a fetchable identifier.
            keywords: The citing report's keywords, for the relevance check.
                Empty when relevance is not being assessed.

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

        lookup_id = reference_id
        if prefix.upper() == "PMC":
            if reference_id not in pmc_aliases:
                return ReferenceCheck(
                    reference_id=reference_id,
                    status=ReferenceStatus.UNVERIFIABLE,
                    occurrences=reference.count,
                    message=(
                        "Could not reach the PMC ID service, so this accession was "
                        "neither confirmed nor ruled out"
                    ),
                ), False
            alias = pmc_aliases[reference_id]
            if alias is None:
                return ReferenceCheck(
                    reference_id=reference_id,
                    status=ReferenceStatus.NOT_FOUND,
                    occurrences=reference.count,
                    message="NCBI reports no such accession in PMC",
                ), False
            lookup_id = alias

        fetcher = text_validator.fetcher
        if ReferenceSourceRegistry.get_source(fetcher.normalize_reference_id(lookup_id)) is None:
            return ReferenceCheck(
                reference_id=reference_id,
                status=ReferenceStatus.UNVERIFIABLE,
                occurrences=reference.count,
                message=f"No resolver is available for prefix '{prefix}'",
            ), False

        content = fetcher.fetch(lookup_id)

        # A search-backed source answers a miss with an empty record rather than
        # None, so a record carrying neither a title nor any text is a failed
        # lookup, not a resolved reference with nothing to show for it. Without
        # this, a fabricated PMC accession would be reported as verified.
        if content is not None and not content.title and not content.content:
            content = None

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

        assessment = assess_relevance(
            keywords,
            reference_text(
                title=content.title,
                content=content.content,
                journal=content.journal,
                keywords=content.keywords,
            ),
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
            relevance=assessment.relevance,
            relevance_score=assessment.score,
            matched_keywords=list(assessment.matched_terms),
        ), has_content

    @staticmethod
    def _quotes_the_title(
        text_validator: "SupportingTextValidator",
        claim: QuotedClaim,
        lookup_id: str,
    ) -> bool:
        """Return whether a quote is really the cited paper's title.

        Reports routinely quote a title before citing it, and a title does not
        appear in the abstract, so the substring check fails and an ordinary
        citation is presented as an unsupported quotation. This happened four
        times in a single live report.

        A leading-substring comparison rather than equality, because quoting a
        title without its subtitle is standard practice: one of those four
        citations gave "…aortic disease in a middle-income country" for a paper
        registered as "…in a middle-income country: a case series study." The
        20-character floor on quote extraction keeps short generic prefixes out.

        Args:
            text_validator: The underlying supporting text validator.
            claim: The quote and the reference it is attributed to.
            lookup_id: Identifier to fetch the record under.

        Returns:
            True if the quote is the title, or the start of it.
        """
        # Free: ReferenceFetcher memoises in a process-local dict keyed on the
        # normalised id (reference_fetcher.py:126), and the reference pass has
        # already fetched this one, so no request is issued here.
        reference = text_validator.fetcher.fetch(lookup_id)
        if reference is None or not reference.title:
            return False

        title = text_validator.normalize_text(reference.title)
        quote = text_validator.normalize_text(claim.quote)
        return bool(quote) and title.startswith(quote)

    def _check_quote(
        self,
        text_validator: "SupportingTextValidator",
        claim: QuotedClaim,
        lookup_id: str,
        source_content_type: Optional[str] = None,
    ) -> SupportingTextCheck:
        """Check one quoted claim against the text of its reference.

        Args:
            text_validator: The underlying supporting text validator.
            claim: The quote and the reference it is attributed to.
            lookup_id: Identifier to fetch, which differs from the cited one for
                a PMC accession resolved to a PMID.
            source_content_type: What kind of text was searched, recorded so that
                a failure against an abstract alone can be read for what it is.

        Returns:
            The per-quote result.
        """
        result = text_validator.validate(claim.quote, lookup_id)
        match = result.match_result

        if not result.is_valid and self._quotes_the_title(text_validator, claim, lookup_id):
            return self._title_quote(claim)

        return SupportingTextCheck(
            reference_id=claim.reference_id,
            quote=claim.quote,
            was_checkable=True,
            is_valid=result.is_valid,
            similarity_score=_clamp_similarity(match.similarity_score if match else 0.0),
            matched_text=match.matched_text if match else None,
            best_match=match.best_match if match else None,
            suggested_fix=match.suggested_fix if match else None,
            source_content_type=source_content_type,
            message=result.message,
        )


PMC_IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"

# NCBI caps the ID converter at 200 identifiers per request. Past that the
# service errors, which would otherwise turn a large bibliography into a
# report-wide "could not reach the service" that is not true.
PMC_IDCONV_BATCH_SIZE = 200


def resolve_pmc_accessions(
    pmc_ids: Iterable[str],
    email: Optional[str] = None,
    rate_limit_delay: float = 0.5,
) -> dict[str, Optional[str]]:
    """Map PMC accessions onto identifiers the underlying validator can fetch.

    ``linkml-reference-validator`` can already *use* a PMC ID - its full-text
    provider fetches article bodies from one, and ``build_identifiers``
    understands a ``PMCID:`` prefix - but it registers no metadata source for the
    prefix, so ``ReferenceFetcher.fetch`` gives up before any of that runs.

    The obvious fix, registering Europe PMC as a JSON API source, was measured
    and rejected: it returned no hit for 1 of 17 real accessions taken from a
    live report, and a lookup gap in a tool that accuses citations of being
    fabricated is worse than the coverage gap it closes. NCBI's own ID converter
    resolved all 17, and says so explicitly when an accession does not exist, so
    it is used instead. Everything downstream then runs through the well-worn
    PMID path.

    Args:
        pmc_ids: Accessions in ``PMC:PMC12345678`` form.
        email: Contact address, which NCBI asks callers to send.
        rate_limit_delay: Seconds to pause between batches, for the rare report
            that needs more than one. Defaults to match
            :attr:`ReferenceValidator.rate_limit_delay`, so a direct caller does
            not get an unthrottled loop against NCBI.

    Returns:
        A mapping from each accession to the identifier to fetch instead
        (``PMID:...``, or ``DOI:...`` when the record has no PMID), or to
        ``None`` when NCBI states the accession does not exist. Accessions the
        service could not be asked about are absent from the mapping, so a
        network failure is never mistaken for a missing record.

    Examples:
        >>> resolve_pmc_accessions([])
        {}
    """
    accessions = [pmc_id.split(":", 1)[-1] for pmc_id in pmc_ids]
    if not accessions:
        return {}

    resolved: dict[str, Optional[str]] = {}
    for index, batch in enumerate(batched(accessions, PMC_IDCONV_BATCH_SIZE)):
        if index and rate_limit_delay:
            # The rest of the validator honours this delay; a run long enough to
            # need several batches should not machine-gun NCBI either.
            time.sleep(rate_limit_delay)
        params = {"ids": ",".join(batch), "format": "json", "tool": "deep-research-client"}
        if email:
            params["email"] = email

        try:
            response = httpx.get(
                PMC_IDCONV_URL, params=params, timeout=30, follow_redirects=True
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Deliberately non-fatal and deliberately not a verdict: without an
            # answer these references become unverifiable, never fabricated.
            logger.warning("Could not reach the PMC ID converter: %s", exc)
            continue

        batch_result = _parse_idconv_records(payload.get("records") or [])
        unanswered = {f"PMC:{a.upper()}" for a in batch} - set(batch_result)
        if unanswered:
            logger.warning(
                "The PMC ID converter returned nothing for %d accession(s): %s",
                len(unanswered),
                ", ".join(sorted(unanswered)),
            )
        resolved.update(batch_result)

    return resolved


def _parse_idconv_records(records: list[dict]) -> dict[str, Optional[str]]:
    """Turn ID converter records into a lookup mapping.

    Args:
        records: The ``records`` array from an ID converter response.

    Returns:
        Accession to replacement identifier, or to ``None`` when the service
        states the accession does not exist.

    Examples:
        >>> _parse_idconv_records([
        ...     {"pmcid": "PMC11157853", "pmid": 38849906, "doi": "10.1186/s13019-024-02793-w"},
        ... ])
        {'PMC:PMC11157853': 'PMID:38849906'}
        >>> _parse_idconv_records([
        ...     {"pmcid": "PMC99999999", "status": "error",
        ...      "errmsg": "Identifier not found in PMC"},
        ... ])
        {'PMC:PMC99999999': None}
        >>> _parse_idconv_records([{"pmcid": "PMC1", "doi": "10.1/only-a-doi"}])
        {'PMC:PMC1': 'DOI:10.1/only-a-doi'}
        >>> _parse_idconv_records([{"requested-id": "PMC2", "pmid": 7}])
        {'PMC:PMC2': 'PMID:7'}
        >>> _parse_idconv_records([])
        {}
    """
    resolved: dict[str, Optional[str]] = {}
    for record in records:
        accession = record.get("pmcid") or record.get("requested-id")
        if not accession:
            continue
        key = f"PMC:{accession.upper()}"
        if record.get("status") == "error":
            resolved[key] = None
        elif record.get("pmid"):
            resolved[key] = f"PMID:{record['pmid']}"
        elif record.get("doi"):
            resolved[key] = f"DOI:{record['doi']}"
    return resolved


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


# Below this many references weighed for relevance, a bibliography matching
# nothing is too little to reason about either way. Mirrors the floor on the
# outage hint in models.py, and for the same reason.
RELEVANCE_SANITY_MIN_REFERENCES = 3


def _withhold_off_topic_when_nothing_matched(
    checks: list[ReferenceCheck],
) -> list[ReferenceCheck]:
    """Stop a bad keyword set from accusing a whole bibliography.

    If not one reference in a report matches the vocabulary read off that report,
    the likelier explanation is that the vocabulary is wrong - a provider that
    echoed its prompt, a report too short to have a subject, a template whose
    boilerplate crowded out the findings - rather than that a researcher cited
    nothing relevant. This was measured: a Falcon report on homocystinuria
    yielded ``cellular, mechanism, provide, comprehensive, claim`` as its
    keywords, matched none of its eight references, and flagged a paper titled
    "Hyperhomocysteinemia in Adult Patients" as off topic.

    So an off-topic verdict has to be earned against a keyword set that demonstrably
    works somewhere. Off-topic references become UNCERTAIN when nothing at all
    came out on topic; the reasoning is recorded rather than silently dropped.

    Args:
        checks: Per-reference results, after relevance has been assessed.

    Returns:
        The same results, with unsupportable accusations withdrawn.

    Examples:
        >>> from .relevance import TopicalRelevance
        >>> flagged = [
        ...     ReferenceCheck(
        ...         reference_id=f"PMID:{n}",
        ...         status=ReferenceStatus.VERIFIED,
        ...         relevance=TopicalRelevance.OFF_TOPIC,
        ...     )
        ...     for n in (1, 2, 3)
        ... ]
        >>> withheld = _withhold_off_topic_when_nothing_matched(flagged)
        >>> all(c.relevance == TopicalRelevance.UNCERTAIN for c in withheld)
        True
        >>> "points at the vocabulary" in withheld[0].message
        True

        One reference matching is enough to show the keywords work:

        >>> mixed = flagged[:2] + [
        ...     ReferenceCheck(
        ...         reference_id="PMID:4",
        ...         status=ReferenceStatus.VERIFIED,
        ...         relevance=TopicalRelevance.ON_TOPIC,
        ...     )
        ... ]
        >>> sum(c.relevance == TopicalRelevance.OFF_TOPIC for c in
        ...     _withhold_off_topic_when_nothing_matched(mixed))
        2
    """
    assessed = [c for c in checks if c.relevance != TopicalRelevance.NOT_ASSESSED]
    if (
        len(assessed) < RELEVANCE_SANITY_MIN_REFERENCES
        or any(c.relevance == TopicalRelevance.ON_TOPIC for c in assessed)
        or not any(c.relevance == TopicalRelevance.OFF_TOPIC for c in assessed)
    ):
        return checks

    logger.info(
        "No reference matched this report's keywords, so the keywords are the "
        "likelier problem; withholding %d off-topic verdict(s)",
        sum(1 for c in assessed if c.relevance == TopicalRelevance.OFF_TOPIC),
    )
    return [
        check.model_copy(
            update={
                "relevance": TopicalRelevance.UNCERTAIN,
                "message": check.message
                or (
                    "Shares little of the report's vocabulary, but so did every "
                    "other reference, which points at the vocabulary rather than "
                    "at this citation"
                ),
            }
        )
        if check.relevance == TopicalRelevance.OFF_TOPIC
        else check
        for check in checks
    ]


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
