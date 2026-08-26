"""Ontology term validation backed by ``linkml-term-validator``.

Deep research providers cite ontology terms with the same confidence they cite
papers, and with the same reliability. An identifier that does not exist is the
easy case. The hard case is the identifier that does: ``NCIT:C16814`` is a real
NCIT term, so it survives every check that asks only whether a term exists, and
it denotes Malaysia. A report that puts it beside the words "Echocardiography
Test" is wrong in a way nothing but a label lookup will show.

:class:`TermValidator` resolves every CURIE a report cites, compares it with the
name the report gave it, and reports terms that have since been obsoleted, so a
report can be shipped with its terms already checked rather than with an
unverified list of identifiers.

Resolution goes through OAK, so any ontology OAK can reach is available. The
default adapter is ``ols:``, which looks terms up over the network one at a
time: right for the handful of terms in a report, wrong for bulk work, where
``sqlite:obo:`` downloads each ontology once and answers locally afterwards.

There is deliberately no rate-limit delay, unlike the reference side. The
distinction that makes one necessary there is drawn upstream here: a definitive
404 comes back as "no such term", while a connectivity failure, a 5xx, a 408 or
a 429 raises ``OntologyServiceUnavailableError`` rather than returning nothing.
So a throttled lookup fails the run instead of quietly becoming a
``NOT_FOUND`` accusation, and the cost of running into a rate limit is having to
run again rather than a confidently wrong report. Verified against
``linkml-term-validator`` 0.4.5.

``linkml-term-validator`` is an optional dependency; install it with::

    pip install "deep_research_client[terms]"
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

from .label_matching import compare_labels
from .sections import strip_validation_section
from .term_datamodel import LabelAgreement, TermCheck, TermStatus
from .term_extraction import FoundTerm, extract_terms
from .term_models import TermValidationReport

if TYPE_CHECKING:  # pragma: no cover - imports only for type checking
    from linkml_term_validator.utils import OntologyAccess

    from ..models import ResearchResult

logger = logging.getLogger(__name__)

TERM_INSTALL_HINT = (
    "linkml-term-validator is required for ontology term validation. "
    'Install it with: pip install "deep_research_client[terms]"'
)

#: OAK adapter used when the caller names none. Resolves one term per request
#: against the EBI's Ontology Lookup Service, which suits a report's worth of
#: terms; ``sqlite:obo:`` is the choice for bulk work.
DEFAULT_ADAPTER = "ols:"

#: Where resolved labels are cached between runs when the caller names no
#: directory. Mirrors the reference validator's ``./references_cache``.
DEFAULT_CACHE_DIR = Path("terms_cache")


@dataclass
class TermValidator:
    """Resolve and check the ontology terms cited by a research report.

    Attributes:
        adapter: OAK adapter string terms are resolved through. ``ols:`` and
            ``sqlite:obo:`` are expanded per prefix, so one setting covers every
            ontology a report cites.
        cache_dir: Directory where resolved labels are cached. Reusing one
            directory across runs avoids re-fetching the same terms.
        cache_labels: Whether to write resolved labels to that directory.
        oak_config: Path to an ``oak_config.yaml`` mapping prefixes to adapters,
            for ontologies the default adapter does not serve. When given, it is
            authoritative: a prefix absent from it gets no adapter at all.
        offline: Resolve only from the label cache, never building an adapter
            and so never reaching the network. Every uncached term comes back
            unverifiable.
        skip_prefixes: Prefixes to report as unverifiable instead of resolving.
        max_terms: Optional cap on the number of terms resolved, for reports
            with very long term lists.
        check_labels: Whether to compare the labels a report writes beside its
            CURIEs with the terms' own labels. Costs no extra requests: the
            label has already been fetched to check that the term resolves.

    Examples:
        >>> validator = TermValidator()
        >>> validator.adapter
        'ols:'
        >>> validator.check_labels
        True
    """

    adapter: str = DEFAULT_ADAPTER
    cache_dir: Optional[Union[str, Path]] = None
    cache_labels: bool = True
    oak_config: Optional[Union[str, Path]] = None
    offline: bool = False
    skip_prefixes: list[str] = field(default_factory=list)
    max_terms: Optional[int] = None
    check_labels: bool = True

    def validate_result(self, result: "ResearchResult") -> TermValidationReport:
        """Validate the ontology terms cited by a :class:`ResearchResult`.

        Args:
            result: The research result to check.

        Returns:
            The validation report.
        """
        return self.validate_markdown(result.markdown, citations=result.citations)

    def validate_markdown(
        self,
        markdown: str,
        citations: Optional[Iterable[str]] = None,
    ) -> TermValidationReport:
        """Validate the ontology terms cited in a block of report markdown.

        A validation section left by an earlier run is removed first. The CLI
        already strips one before calling, but a library caller re-validating a
        report annotated with ``--in-place`` would otherwise re-extract the
        identifiers that section lists - including, for a mislabelled term, both
        the name the report gave it and the name the ontology gave it.

        Args:
            markdown: The report body.
            citations: Optional citation strings scanned for identifiers.

        Returns:
            The validation report.
        """
        return self.validate_terms(extract_terms(strip_validation_section(markdown), citations))

    def validate_terms(self, terms: Iterable[FoundTerm]) -> TermValidationReport:
        """Validate an explicit set of extracted terms.

        Resolution happens in two passes, and the second is why. Whether a
        failed lookup means "this identifier was invented" or "nothing here
        could answer" depends on the prefix, and the strongest evidence that a
        prefix can be resolved is that one of its terms just was. So every term
        is looked up first, and only then classified against what the run
        learned about each prefix.

        Args:
            terms: Terms to resolve.

        Returns:
            The validation report.

        Raises:
            ImportError: If the optional ``terms`` extra is not installed.
        """
        if not term_validator_is_available():
            raise ImportError(TERM_INSTALL_HINT)

        terms = list(terms)
        truncated = False
        if self.max_terms is not None and len(terms) > self.max_terms:
            logger.warning("Checking only the first %d of %d terms", self.max_terms, len(terms))
            terms = terms[: self.max_terms]
            truncated = True

        ontology = self._build_ontology_access()
        skipped = {prefix.upper() for prefix in self.skip_prefixes}

        # Pass one: look everything up, recording nothing but what came back.
        resolved: dict[str, Optional[str]] = {}
        obsolete: dict[str, bool] = {}
        for index, term in enumerate(terms, start=1):
            if term.prefix.upper() in skipped:
                continue
            logger.info("Resolving term %d/%d: %s", index, len(terms), term.term_id)
            label = ontology.get_label(term.term_id)
            resolved[term.term_id] = label
            if label is not None:
                obsolete[term.term_id] = ontology.is_obsolete(term.term_id) is True

        resolvable_prefixes = {
            term.prefix for term in terms if resolved.get(term.term_id) is not None
        }
        configured_prefixes = set(getattr(ontology, "oak_config", {}) or {})
        expected = _known_ontology_prefixes() | resolvable_prefixes | configured_prefixes

        # Pass two: classify, now that each prefix's resolvability is known.
        checks = [
            self._check_term(term, ontology, resolved, obsolete, skipped, expected)
            for term in terms
        ]
        unresolvable_prefixes = list(
            dict.fromkeys(
                check.prefix
                for check in checks
                if check.status == TermStatus.UNVERIFIABLE
                and check.prefix.upper() not in skipped
            )
        )

        return TermValidationReport(
            terms=checks,
            unresolvable_prefixes=unresolvable_prefixes,
            adapter=self.adapter,
            validator_version=term_validator_version(),
            truncated=truncated,
        )

    def _build_ontology_access(self) -> "OntologyAccess":
        """Construct the OAK-backed label resolver this run will use."""
        from linkml_term_validator.utils import OntologyAccess

        return OntologyAccess(
            oak_adapter_string=self.adapter,
            cache_labels=self.cache_labels,
            cache_dir=Path(self.cache_dir) if self.cache_dir else DEFAULT_CACHE_DIR,
            oak_config_path=Path(self.oak_config) if self.oak_config else None,
            offline=self.offline,
        )

    def _check_term(
        self,
        term: FoundTerm,
        ontology: "OntologyAccess",
        resolved: dict[str, Optional[str]],
        obsolete: dict[str, bool],
        skipped: set[str],
        expected: frozenset[str],
    ) -> TermCheck:
        """Turn one looked-up term into its result row."""
        if term.prefix.upper() in skipped:
            return self._unverifiable(term, f"prefix {term.prefix} was skipped")

        label = resolved.get(term.term_id)
        if label is None:
            # Skipping is matched case-insensitively above, membership here
            # exactly. The asymmetry is deliberate: a prefix written in an
            # unexpected case falls out as UNVERIFIABLE, never as an accusation,
            # so erring here costs coverage rather than correctness. Matching
            # loosely would be the unsafe direction.
            if term.prefix not in expected:
                return self._unverifiable(
                    term, f"no configured ontology resolves the prefix {term.prefix}"
                )
            if self.offline:
                return self._unverifiable(term, "not in the label cache, and offline")
            return TermCheck(
                term_id=term.term_id,
                prefix=term.prefix,
                status=TermStatus.NOT_FOUND,
                occurrences=term.count,
                reported_labels=list(term.labels),
                message=f"{term.prefix} does not contain this term",
            )

        is_obsolete = obsolete.get(term.term_id, False)
        agreement, similarity = self._compare(term, label)
        return TermCheck(
            term_id=term.term_id,
            prefix=term.prefix,
            status=TermStatus.OBSOLETE if is_obsolete else TermStatus.VERIFIED,
            occurrences=term.count,
            ontology_label=label,
            reported_labels=list(term.labels),
            agreement=agreement,
            label_similarity=similarity,
            replaced_by=_replacement_for(ontology, term.term_id) if is_obsolete else None,
            message="the term is obsolete" if is_obsolete else None,
        )

    def _compare(self, term: FoundTerm, label: str) -> tuple[LabelAgreement, float]:
        """Judge every name the report gave a term, keeping the worst verdict.

        A report that names one identifier twice, once correctly, has still
        named it incorrectly once, and reporting the better of the two would
        hide exactly the mistake worth finding.
        """
        if not self.check_labels or not term.labels:
            return LabelAgreement.NOT_ASSESSED, 0.0

        # Ordered worst to best, so `min` picks the verdict to report.
        severity = {
            LabelAgreement.MISMATCH: 0,
            LabelAgreement.VARIANT: 1,
            LabelAgreement.MATCH: 2,
            LabelAgreement.NOT_ASSESSED: 3,
        }
        judged = [compare_labels(reported, label) for reported in term.labels]
        return min(judged, key=lambda pair: (severity[pair[0]], pair[1]))

    @staticmethod
    def _unverifiable(term: FoundTerm, message: str) -> TermCheck:
        """Build the result row for a term nothing was learned about."""
        return TermCheck(
            term_id=term.term_id,
            prefix=term.prefix,
            status=TermStatus.UNVERIFIABLE,
            occurrences=term.count,
            reported_labels=list(term.labels),
            message=message,
        )


def _replacement_for(ontology: "OntologyAccess", curie: str) -> Optional[str]:
    """Return the term an obsolete term was replaced by, when one is known.

    Read from the OLS payload the label lookup already fetched and cached, so it
    costs no extra request. OAK's generic metadata interface would be the right
    door, but the OLS adapter does not implement it, and adapters that do vary in
    what they expose - so an unavailable replacement is reported as no
    replacement rather than as an error. The term is still flagged as obsolete
    either way; only the suggested fix is missing.

    These are private attributes of ``OntologyAccess``, verified against
    ``linkml-term-validator`` 0.4.5. A rename upstream degrades this to "no
    replacement known" silently, which is why the integration test asserts a
    known replacement against the live service rather than trusting the path.

    Args:
        ontology: The resolver the term was looked up through.
        curie: The obsolete term.

    Returns:
        The replacement CURIE, or None if the ontology states none or the
        resolver cannot say.
    """
    term_dict = getattr(ontology, "_ols_term_dict", None)
    is_ols = getattr(ontology, "_is_ols_adapter", None)
    prefix = curie.split(":", 1)[0]
    adapter = ontology.get_adapter(prefix)
    if term_dict is None or is_ols is None or adapter is None or not is_ols(adapter):
        return None

    payload = term_dict(adapter, curie)
    if not payload:
        return None
    replacement = payload.get("term_replaced_by")
    if not replacement:
        return None
    # OLS reports replacements in the underscore form used by OBO PURLs.
    return str(replacement).replace("_", ":", 1)


@lru_cache(maxsize=1)
def _known_ontology_prefixes() -> frozenset[str]:
    """Prefixes an ontology resolver is expected to cover.

    The OBO context of ``prefixmaps`` - every ontology in the OBO library -
    rather than a hand-kept list, so a report citing an ontology nobody here
    thought of is still checked. Membership decides how a failed lookup is
    read: a term that does not resolve under a prefix on this list did not
    exist, while one under a prefix that is not is simply beyond what this run
    could check.

    Examples:
        >>> "HP" in _known_ontology_prefixes()
        True
        >>> "PMID" in _known_ontology_prefixes()
        False
    """
    from prefixmaps import load_converter

    return frozenset(load_converter("obo").prefix_map)


def lookup_error_types() -> tuple:
    """Exception types raised when a lookup could determine nothing at all.

    Separate from an ordinary "no such term", which is an answer.
    ``linkml-term-validator`` raises this for a connectivity failure, a 5xx, a
    408 or a 429 - cases where treating the term as absent would turn an outage
    into an accusation. The CLI reports them as an unreachable service, the same
    exit code the reference side uses, rather than as a traceback.

    Returns an empty tuple when the extra is not installed, so a caller can
    always splice it into an ``except`` clause.

    Examples:
        >>> isinstance(lookup_error_types(), tuple)
        True
    """
    if not term_validator_is_available():
        return ()

    from linkml_term_validator.utils.oak_utils import OntologyServiceUnavailableError

    return (OntologyServiceUnavailableError,)


def term_validator_is_available() -> bool:
    """Return whether the optional ``terms`` extra is installed.

    Examples:
        >>> isinstance(term_validator_is_available(), bool)
        True
    """
    from importlib.util import find_spec

    return find_spec("linkml_term_validator") is not None


def term_validator_version() -> Optional[str]:
    """Return the installed ``linkml-term-validator`` version, if any.

    The distribution metadata is authoritative; the package's own
    ``__version__`` attribute is not populated in released wheels.

    Examples:
        >>> version = term_validator_version()
        >>> version is None or isinstance(version, str)
        True
    """
    from importlib.metadata import PackageNotFoundError, version

    if not term_validator_is_available():
        return None
    try:
        return version("linkml-term-validator")
    except PackageNotFoundError:  # pragma: no cover - installed without metadata
        return None
