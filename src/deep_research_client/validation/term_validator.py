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

from .label_matching import LabelComparison, compare_labels, label_similarity
from .sections import strip_validation_section
from .term_datamodel import LabelAgreement, TermCheck, TermStatus
from .term_extraction import FoundTerm, extract_terms
from .term_models import (
    OBSOLESCENCE_UNCHECKED_MESSAGE,
    SERVICE_UNREACHABLE_MESSAGE,
    TermValidationReport,
)

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

#: After this many terms fail to resolve in a row, the ontology service is
#: treated as down and the remaining terms are marked unreachable without
#: another doomed, timeout-length call (issue #111). It bounds a total outage to
#: a handful of waits instead of one per term, while still giving a genuinely
#: intermittent service several chances to answer.
_CIRCUIT_BREAKER_THRESHOLD = 5


@dataclass
class _LookupResults:
    """What pass one learned about each term, handed to pass two.

    Bundled into one object so :meth:`TermValidator._check_term` takes the
    pass-one outcome as a single argument rather than four parallel collections.

    Attributes:
        resolved: Term id to its ontology label, or None when the lookup
            returned "no such term".
        obsolete: Term id to whether it is deprecated (only for resolved terms).
        unreachable: Term ids the service could not be reached for (issue #111).
        obsolete_unknown: Resolved term ids whose obsolescence check failed, so
            "not obsolete" is an assumption rather than a finding.
        from_cache: Term ids resolved from the label cache after the circuit
            breaker tripped. Pass two judges these through the cache-only
            accessor and does not claim a label mismatch it cannot substantiate.
    """

    resolved: dict[str, Optional[str]]
    obsolete: dict[str, bool]
    unreachable: set[str]
    obsolete_unknown: set[str]
    from_cache: set[str]


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
        lookup_errors = _per_term_lookup_errors()

        # Pass one: look everything up, recording nothing but what came back. A
        # lookup the service cannot answer for one term is tolerated, not fatal
        # (issue #111): the term is marked unreachable and pass two reports it
        # UNVERIFIABLE, never NOT_FOUND (which would turn an outage into an
        # accusation). Once a run of lookups fails in a row the live service is
        # taken to be down and the rest are resolved from the label cache only --
        # which answers for free, so a re-run really does resume from the cache --
        # with the genuinely uncached terms marked unreachable, no more
        # timeout-length calls.
        resolved: dict[str, Optional[str]] = {}
        obsolete: dict[str, bool] = {}
        unreachable: set[str] = set()
        obsolete_unknown: set[str] = set()
        from_cache: set[str] = set()
        consecutive_failures = 0
        tripped = False
        offline_fallback: Optional["OntologyAccess"] = None
        for index, term in enumerate(terms, start=1):
            if term.prefix.upper() in skipped:
                continue
            if tripped:
                # Live service down, but the cache still answers without the
                # network. Resolve what it holds; only uncached terms are
                # unreachable, so the "re-run resumes from the cache" promise holds
                # even when the breaker tripped.
                if offline_fallback is None:
                    offline_fallback = self._build_ontology_access(offline=True)
                try:
                    cached = offline_fallback.get_label(term.term_id)
                except lookup_errors as exc:
                    # Offline resolution is cache-only and should return None for
                    # an uncached term, not raise; but if a resolver instead
                    # reports "offline, not cached" as unavailable, treat it as
                    # unreachable rather than letting it abort the run post-trip.
                    logger.warning(
                        "Offline cache lookup failed for %s: %s", term.term_id, exc
                    )
                    unreachable.add(term.term_id)
                    continue
                if cached is not None:
                    resolved[term.term_id] = cached
                    from_cache.add(term.term_id)
                    try:
                        obsolete[term.term_id] = (
                            offline_fallback.is_obsolete(term.term_id) is True
                        )
                    except lookup_errors:
                        obsolete_unknown.add(term.term_id)
                else:
                    # A cache miss reads as unreachable, not NOT_FOUND: None here
                    # means "not cached", not "known absent", so erring to
                    # unreachable avoids a false fabrication claim during an outage.
                    unreachable.add(term.term_id)
                continue
            logger.info("Resolving term %d/%d: %s", index, len(terms), term.term_id)
            try:
                label = ontology.get_label(term.term_id)
            except lookup_errors as exc:
                logger.warning(
                    "Could not reach the ontology service for %s: %s", term.term_id, exc
                )
                unreachable.add(term.term_id)
                consecutive_failures += 1
                if consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                    logger.warning(
                        "%d lookups failed in a row; treating the live ontology service "
                        "as down and resolving the remaining terms from the cache only",
                        consecutive_failures,
                    )
                    tripped = True
                continue
            consecutive_failures = 0
            resolved[term.term_id] = label
            if label is not None:
                try:
                    obsolete[term.term_id] = ontology.is_obsolete(term.term_id) is True
                except lookup_errors as exc:
                    # The label resolved; only the obsolescence check failed. Keep
                    # the resolved term, but record that obsolescence is unknown so
                    # an obsolete term is not quietly reported as clean.
                    logger.warning(
                        "Could not check obsolescence for %s: %s", term.term_id, exc
                    )
                    obsolete_unknown.add(term.term_id)

        resolvable_prefixes = {
            term.prefix for term in terms if resolved.get(term.term_id) is not None
        }
        configured_prefixes = set(getattr(ontology, "oak_config", {}) or {})
        expected = _known_ontology_prefixes() | resolvable_prefixes | configured_prefixes

        # Pass two: classify, now that each prefix's resolvability is known. The
        # accessor is chosen PER TERM. A term resolved from the cache after the
        # breaker tripped is judged through the cache-only accessor, since the
        # live service is down; a term resolved live keeps the live accessor,
        # whose OLS payload the obsolescence check already memoised, so its
        # synonyms and replacement cost no request even now. This keeps a late
        # outage from discarding data earlier terms already fetched, and
        # ``_compare`` declines to judge a cache-resolved label it cannot check
        # synonyms for rather than risk a false mismatch.
        results = _LookupResults(
            resolved, obsolete, unreachable, obsolete_unknown, from_cache
        )
        checks = [
            self._check_term(
                term,
                offline_fallback
                if (offline_fallback is not None and term.term_id in from_cache)
                else ontology,
                results,
                skipped,
                expected,
            )
            for term in terms
        ]
        unresolvable_prefixes = list(
            dict.fromkeys(
                check.prefix
                for check in checks
                if check.status == TermStatus.UNVERIFIABLE
                and check.prefix.upper() not in skipped
                # A prefix we merely could not reach is not a prefix with no
                # resolver (issue #111); those terms are reported separately.
                and check.term_id not in unreachable
            )
        )

        return TermValidationReport(
            terms=checks,
            unresolvable_prefixes=unresolvable_prefixes,
            adapter=self.adapter,
            validator_version=term_validator_version(),
            truncated=truncated,
        )

    def _build_ontology_access(self, offline: Optional[bool] = None) -> "OntologyAccess":
        """Construct the OAK-backed label resolver this run will use.

        Args:
            offline: Override ``self.offline``. The circuit breaker builds a
                cache-only (``offline=True``) resolver so it can keep answering
                from the label cache after the live service is taken to be down.
        """
        from linkml_term_validator.utils import OntologyAccess

        return OntologyAccess(
            oak_adapter_string=self.adapter,
            cache_labels=self.cache_labels,
            cache_dir=Path(self.cache_dir) if self.cache_dir else DEFAULT_CACHE_DIR,
            oak_config_path=Path(self.oak_config) if self.oak_config else None,
            offline=self.offline if offline is None else offline,
        )

    def _check_term(
        self,
        term: FoundTerm,
        ontology: "OntologyAccess",
        results: _LookupResults,
        skipped: set[str],
        expected: frozenset[str],
    ) -> TermCheck:
        """Turn one looked-up term into its result row."""
        if term.prefix.upper() in skipped:
            return self._unverifiable(term, f"prefix {term.prefix} was skipped")

        # The service could not be reached for this term (issue #111). It is
        # UNVERIFIABLE, not NOT_FOUND: a connectivity failure is no evidence the
        # identifier is wrong. Checked before the resolved-label branch so it is
        # never mistaken for a term the ontology does not contain.
        if term.term_id in results.unreachable:
            return self._unverifiable(term, SERVICE_UNREACHABLE_MESSAGE)

        label = results.resolved.get(term.term_id)
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

        is_obsolete = results.obsolete.get(term.term_id, False)
        comparison = self._compare(
            term, label, ontology,
            synonyms_reachable=term.term_id not in results.from_cache,
        )
        # The label resolved, so the identifier exists and is VERIFIED; but if the
        # obsolescence check could not be reached, say so, so an obsolete term is
        # not read as clean just because the check was skipped (issue #111).
        if is_obsolete:
            message: Optional[str] = "the term is obsolete"
        elif term.term_id in results.obsolete_unknown:
            message = OBSOLESCENCE_UNCHECKED_MESSAGE
        else:
            message = None
        return TermCheck(
            term_id=term.term_id,
            prefix=term.prefix,
            status=TermStatus.OBSOLETE if is_obsolete else TermStatus.VERIFIED,
            occurrences=term.count,
            ontology_label=label,
            reported_labels=list(term.labels),
            agreement=comparison.agreement,
            label_similarity=comparison.similarity,
            matched_synonym=comparison.matched_synonym,
            replaced_by=_replacement_for(ontology, term.term_id) if is_obsolete else None,
            message=message,
        )

    def _compare(
        self,
        term: FoundTerm,
        label: str,
        ontology: "OntologyAccess",
        synonyms_reachable: bool = True,
    ) -> LabelComparison:
        """Judge every name the report gave a term, keeping the worst verdict.

        A report that names one identifier twice, once correctly, has still
        named it incorrectly once, and reporting the better of the two would
        hide exactly the mistake worth finding.

        Synonyms are fetched once per term rather than once per reported label,
        and only when there is a label to compare, so a report that names
        nothing costs nothing.

        ``synonyms_reachable`` is False for a term resolved from the cache after
        the circuit breaker tripped (issue #111): its synonyms live on the live
        service, which is down, and the cache-only accessor has no adapter to
        supply them. Judging the name against the bare label then risks a false
        MISMATCH on a reported synonym, so a non-exact name is left NOT_ASSESSED
        rather than guessed. (Plain ``--offline`` keeps its existing behaviour:
        the cache is its intended source, so a bare-label comparison there is a
        deliberate, tested verdict, not a degraded one.)
        """
        if not self.check_labels or not term.labels:
            return LabelComparison(LabelAgreement.NOT_ASSESSED, 0.0)

        # Synonyms cannot change a verdict when every name the report gave is
        # already the term's own label, and the worst verdict is what gets
        # reported, so a term named perfectly throughout needs no lookup. That is
        # free against the memoised OLS payload and a query per term against a
        # local adapter - and it is the reports with nothing wrong with them that
        # would otherwise pay most.
        if all(label_similarity(reported, label) == 1.0 for reported in term.labels):
            names = TermNames()
        elif not synonyms_reachable:
            # Cannot consult synonyms (service down, cache has no adapter), and
            # the name is not an exact label match -- decline to judge rather
            # than risk a false MISMATCH on a synonym we could not look up.
            return LabelComparison(LabelAgreement.NOT_ASSESSED, 0.0)
        else:
            try:
                names = _synonyms_for(ontology, term.term_id)
            except _per_term_lookup_errors() as exc:
                # The service went unreachable while fetching synonyms (issue
                # #111). Without them we cannot tell a synonym match from a
                # mismatch, so we decline to judge the name rather than raise a
                # false MISMATCH -- and, crucially, rather than abort the whole
                # run one step after the label resolved. The term itself still
                # resolved and stays VERIFIED.
                logger.warning(
                    "Could not fetch synonyms for %s; leaving its label unassessed: %s",
                    term.term_id,
                    exc,
                )
                return LabelComparison(LabelAgreement.NOT_ASSESSED, 0.0)

        # Ordered worst to best, so `min` picks the verdict to report.
        severity = {
            LabelAgreement.MISMATCH: 0,
            LabelAgreement.VARIANT: 1,
            LabelAgreement.MATCH: 2,
            LabelAgreement.NOT_ASSESSED: 3,
        }
        judged = [
            compare_labels(reported, label, names.exact, names.related)
            for reported in term.labels
        ]
        return min(judged, key=lambda c: (severity[c.agreement], c.similarity))

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


#: How the two sources spell "this synonym names the same thing as the label":
#: ``oio:hasExactSynonym`` is the predicate in an OAK alias map, and
#: ``hasExactSynonym`` is the bare scope in an OLS payload. Both spellings live
#: here because both are matched against this one set. Every other synonym
#: predicate or scope - broad, narrow, related - names something adjacent, which
#: is a different claim and earns a different verdict.
#:
#: An unrecognised spelling demotes a synonym to related, so a mistake here costs
#: a MATCH that reads as a VARIANT rather than the other way round.
_EXACT_SYNONYM_PREDICATES = frozenset({"oio:hasExactSynonym", "hasExactSynonym"})

#: The label's own predicate in an OAK alias map, which is not a synonym.
_LABEL_PREDICATE = "rdfs:label"


@dataclass(frozen=True)
class TermNames:
    """The names a term carries, beyond its label.

    Attributes:
        exact: Synonyms the ontology marks as exact - other names for the same
            thing.
        related: Synonyms of any other scope, or of unrecorded scope.

    Examples:
        >>> TermNames().exact
        ()
    """

    exact: tuple[str, ...] = ()
    related: tuple[str, ...] = ()


def _synonyms_for(ontology: "OntologyAccess", curie: str) -> TermNames:
    """Collect a term's synonyms, keeping their scope where the source records it.

    Two routes, because the adapters differ in what they implement:

    * OAK's ``entity_alias_map``, which is what a local adapter such as
      ``sqlite:obo:`` serves and what ontogpt reads. It is keyed by predicate, so
      scope survives.
    * The OLS payload, for the default ``ols:`` adapter, whose OAK
      implementation raises ``NotImplementedError`` for alias lookups. Its
      ``obo_synonym`` entries carry a scope each; the flat ``synonyms`` list does
      not, so anything only found there is treated as unscoped and therefore not
      exact.

    Either way this costs no request the run was not already making: the OLS
    payload has been fetched and cached by the obsolescence check, and a local
    adapter answers from its own database.

    Args:
        ontology: The resolver the term was looked up through.
        curie: The term to collect names for.

    Returns:
        The term's synonyms, split by whether the ontology calls them exact.
    """
    prefix = curie.split(":", 1)[0]
    adapter = ontology.get_adapter(prefix)
    if adapter is None:
        return TermNames()

    alias_map = getattr(adapter, "entity_alias_map", None)
    if callable(alias_map):
        try:
            mapping = alias_map(curie)
        except NotImplementedError:
            # "This route is not available" -- fall through to the OLS payload.
            # A connectivity error is NOT caught here: it propagates to the
            # caller (``_compare``), which turns it into an unassessed label
            # rather than a false mismatch, and keeps one unreachable synonym
            # lookup from aborting the whole run (issue #111).
            mapping = None
        if mapping:
            names = _names_from_alias_map(mapping)
            # An adapter that implements the call shallowly returns the label and
            # nothing else. That is a non-empty mapping carrying no synonyms, so
            # testing the mapping alone would skip the payload route and lose
            # them silently.
            if names.exact or names.related:
                return names

    return _names_from_ols_payload(ontology, adapter, curie)


def _names_from_alias_map(mapping: dict) -> TermNames:
    """Split an OAK alias map into exact and other synonyms.

    The label's own entry is dropped: it is not a synonym, and it is already
    compared against directly.

    Examples:
        >>> names = _names_from_alias_map({
        ...     "rdfs:label": ["Seizure"],
        ...     "oio:hasExactSynonym": ["Seizures"],
        ...     "oio:hasRelatedSynonym": ["Epilepsy"],
        ... })
        >>> names.exact, names.related
        (('Seizures',), ('Epilepsy',))
    """
    exact: list[str] = []
    related: list[str] = []
    for predicate, values in mapping.items():
        if predicate == _LABEL_PREDICATE:
            continue
        target = exact if predicate in _EXACT_SYNONYM_PREDICATES else related
        target.extend(value for value in values or [] if value)
    return TermNames(exact=tuple(dict.fromkeys(exact)), related=tuple(dict.fromkeys(related)))


def _names_from_ols_payload(
    ontology: "OntologyAccess", adapter: object, curie: str
) -> TermNames:
    """Read a term's synonyms out of the cached OLS payload.

    Reached through the same guarded private accessor as ``_replacement_for``,
    and degrading the same way: no payload means no synonyms, which costs
    coverage rather than correctness.
    """
    term_dict = getattr(ontology, "_ols_term_dict", None)
    is_ols = getattr(ontology, "_is_ols_adapter", None)
    if term_dict is None or is_ols is None or not is_ols(adapter):
        return TermNames()

    payload = term_dict(adapter, curie)
    if not payload:
        return TermNames()

    exact: list[str] = []
    related: list[str] = []
    for entry in payload.get("obo_synonym") or []:
        name = (entry or {}).get("name")
        if not name:
            continue
        scope = (entry or {}).get("scope") or ""
        (exact if scope in _EXACT_SYNONYM_PREDICATES else related).append(name)

    # The flat list carries no scope, so whatever it adds beyond the scoped
    # entries is treated as merely related. Better a variant than a mismatch;
    # calling an unscoped name exact would be a claim the payload never made.
    scoped = set(exact) | set(related)
    related.extend(name for name in payload.get("synonyms") or [] if name and name not in scoped)

    return TermNames(exact=tuple(dict.fromkeys(exact)), related=tuple(dict.fromkeys(related)))


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

    try:
        payload = term_dict(adapter, curie)
    except _per_term_lookup_errors() as exc:
        # The obsolescence check populates this payload, but if its own lookup
        # failed (issue #111) this call may go back to an unreachable service.
        # A missing replacement is already reported as "no replacement known",
        # so degrade to that rather than aborting a run after the label resolved.
        logger.warning("Could not fetch the replacement for %s: %s", curie, exc)
        return None
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


def _per_term_lookup_errors() -> tuple:
    """Exception types to tolerate for a *single* term's lookup (issue #111).

    The resolver's own outage type (see :func:`lookup_error_types`) plus the
    network-failure subclasses: a raw ``urllib`` failure (``URLError``), a
    dropped connection (``ConnectionError``) and a timeout (``TimeoutError``,
    which ``socket.timeout`` aliases). Deliberately *not* bare ``OSError``: that
    also covers ``PermissionError``, ``FileNotFoundError`` and a full disk raised
    while writing the label cache, which are local faults that should fail fast,
    not be mistaken for an unreachable service. ``ValueError`` (a malformed
    cached record) is kept out for the same reason.

    Examples:
        >>> TimeoutError in _per_term_lookup_errors()
        True
        >>> FileNotFoundError in _per_term_lookup_errors()
        False
    """
    import urllib.error

    return (ConnectionError, TimeoutError, urllib.error.URLError) + lookup_error_types()


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
