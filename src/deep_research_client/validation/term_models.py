"""Derived views over the term validation data model.

The data shape lives in ``term_validation.yaml`` and is generated into
:mod:`deep_research_client.validation.term_datamodel` with
``just gen-term-datamodel``. This module adds what a schema cannot express:
quantities computed from those slots, and rendering of a report as markdown or
as frontmatter.

:class:`TermCheck` is re-exported from the generated model unchanged; only the
report gains behaviour, and it adds no fields of its own so that its JSON schema
stays exactly what LinkML would emit.

The same ``use_enum_values = True`` quirk applies as to the reference model: an
enum-ranged slot stores the enum's *value* rather than the member, so compare
against members rather than reaching for ``.value`` or ``.name`` on them::

    >>> check = TermCheck(term_id="HP:1", prefix="HP", status=TermStatus.VERIFIED)
    >>> check.status
    'VERIFIED'
    >>> check.status == TermStatus.VERIFIED
    True
    >>> isinstance(check.status, TermStatus)
    False
"""

from typing import Any, Dict, List

from .sections import TERM_VALIDATION_SECTION_HEADING
from .term_datamodel import LabelAgreement, TermCheck, TermStatus
from .term_datamodel import TermValidationReport as GeneratedTermValidationReport

__all__ = [
    "OUTAGE_HINT_MIN_TERMS",
    "TERM_VALIDATION_SECTION_HEADING",
    "LabelAgreement",
    "TermCheck",
    "TermStatus",
    "TermValidationReport",
]

# Below this many resolvable terms, a clean sweep of failures is not evidence of
# an ontology service being down - it is as likely to be a short list of terms
# that really are wrong. The same floor, for the same reason, as the reference
# side applies to bibliographies.
OUTAGE_HINT_MIN_TERMS = 3


def _mislabelled_entry(check: TermCheck) -> Dict[str, Any]:
    """Summarise a mislabelled term as both of the names it goes by.

    The CURIE alone does not carry the finding. Reading
    ``mislabelled_terms: [NCIT:C16814]`` off a report's frontmatter tells you
    something is wrong and leaves you to go and look up what; the pair of names
    *is* the finding, and it fits on one line.

    ``reported_labels`` stays a list rather than collapsing to the single name
    that produced the verdict, because a report that calls one identifier two
    things has not decided what it cites, and labels contain commas often
    enough that joining them would be ambiguous.

    Examples:
        >>> _mislabelled_entry(
        ...     TermCheck(
        ...         term_id="NCIT:C16814",
        ...         prefix="NCIT",
        ...         status=TermStatus.VERIFIED,
        ...         ontology_label="Malaysia",
        ...         reported_labels=["Echocardiography Test"],
        ...         agreement=LabelAgreement.MISMATCH,
        ...     )
        ... )
        {'term_id': 'NCIT:C16814', 'reported_labels': ['Echocardiography Test'], 'ontology_label': 'Malaysia'}
    """
    return {
        "term_id": check.term_id,
        "reported_labels": list(check.reported_labels or []),
        "ontology_label": check.ontology_label,
    }


def _obsolete_entry(check: TermCheck) -> Dict[str, Any]:
    """Summarise an obsolete term as its name, and its replacement where known.

    The replacement is the actionable half - it is the edit the report needs -
    so it is written out whenever the ontology states one, and omitted rather
    than left null when it does not.

    Examples:
        >>> _obsolete_entry(
        ...     TermCheck(
        ...         term_id="GO:0008022",
        ...         prefix="GO",
        ...         status=TermStatus.OBSOLETE,
        ...         ontology_label="obsolete protein C-terminus binding",
        ...         replaced_by="GO:0005515",
        ...     )
        ... )
        {'term_id': 'GO:0008022', 'ontology_label': 'obsolete protein C-terminus binding', 'replaced_by': 'GO:0005515'}

        An ontology that names no replacement leaves the key out:

        >>> _obsolete_entry(
        ...     TermCheck(
        ...         term_id="GO:0000005",
        ...         prefix="GO",
        ...         status=TermStatus.OBSOLETE,
        ...         ontology_label="obsolete ribosomal chaperone activity",
        ...     )
        ... )
        {'term_id': 'GO:0000005', 'ontology_label': 'obsolete ribosomal chaperone activity'}
    """
    entry: Dict[str, Any] = {
        "term_id": check.term_id,
        "ontology_label": check.ontology_label,
    }
    if check.replaced_by:
        entry["replaced_by"] = check.replaced_by
    return entry


class TermValidationReport(GeneratedTermValidationReport):
    """Aggregate result of validating every ontology term in a report.

    Examples:
        >>> report = TermValidationReport(
        ...     terms=[
        ...         TermCheck(term_id="HP:0001250", prefix="HP", status=TermStatus.VERIFIED),
        ...         TermCheck(term_id="HP:9999999", prefix="HP", status=TermStatus.NOT_FOUND),
        ...     ]
        ... )
        >>> report.total_terms
        2
        >>> report.verified_count
        1
        >>> report.not_found_count
        1
        >>> report.has_problems
        True
    """

    @property
    def checked_terms(self) -> List[TermCheck]:
        """The per-term results, normalised to a list.

        Examples:
            >>> TermValidationReport().checked_terms
            []
        """
        return self.terms or []

    @property
    def total_terms(self) -> int:
        """Number of unique terms checked.

        Examples:
            >>> TermValidationReport().total_terms
            0
        """
        return len(self.checked_terms)

    @property
    def verified_count(self) -> int:
        """Number of terms that resolved to a current term."""
        return sum(1 for t in self.checked_terms if t.status == TermStatus.VERIFIED)

    @property
    def confabulated_terms(self) -> List[TermCheck]:
        """Terms that did not resolve in an ontology that resolves other terms.

        Examples:
            >>> report = TermValidationReport(
            ...     terms=[
            ...         TermCheck(term_id="HP:9999999", prefix="HP", status=TermStatus.NOT_FOUND)
            ...     ]
            ... )
            >>> [t.term_id for t in report.confabulated_terms]
            ['HP:9999999']
        """
        return [t for t in self.checked_terms if t.status == TermStatus.NOT_FOUND]

    @property
    def not_found_count(self) -> int:
        """Number of terms that failed to resolve."""
        return len(self.confabulated_terms)

    @property
    def obsolete_terms(self) -> List[TermCheck]:
        """Terms that resolve but have been deprecated."""
        return [t for t in self.checked_terms if t.status == TermStatus.OBSOLETE]

    @property
    def obsolete_count(self) -> int:
        """Number of terms that resolve but have been deprecated."""
        return len(self.obsolete_terms)

    @property
    def unverifiable_count(self) -> int:
        """Number of terms no resolver covered, or that were skipped."""
        return sum(1 for t in self.checked_terms if t.status == TermStatus.UNVERIFIABLE)

    @property
    def resolvable_count(self) -> int:
        """Number of terms a lookup actually returned an answer about.

        Excludes unverifiable terms, about which nothing was learned. Counting
        them as successes would let ``--skip-prefix`` dilute the failure rate,
        so that skipping the half of a term list that is invented would halve
        the reported figure.
        """
        return self.verified_count + self.not_found_count + self.obsolete_count

    @property
    def confabulation_rate(self) -> float:
        """Fraction of the terms we got an answer about that failed to resolve.

        Returns ``0.0`` when nothing was resolvable.

        Examples:
            >>> TermValidationReport().confabulation_rate
            0.0
            >>> TermValidationReport(
            ...     terms=[
            ...         TermCheck(term_id="A:11", prefix="A", status=TermStatus.NOT_FOUND),
            ...         TermCheck(term_id="A:22", prefix="A", status=TermStatus.VERIFIED),
            ...         TermCheck(term_id="A:33", prefix="A", status=TermStatus.UNVERIFIABLE),
            ...     ]
            ... ).confabulation_rate
            0.5
        """
        if not self.resolvable_count:
            return 0.0
        return self.not_found_count / self.resolvable_count

    @property
    def all_terms_failed(self) -> bool:
        """Whether enough terms failed at once to suggest a service problem.

        Every term in a report failing at once is far more often an unreachable
        ontology service than a report in which every identifier is invented, so
        the rendered report hedges when this is true.

        Examples:
            >>> TermValidationReport().all_terms_failed
            False
            >>> TermValidationReport(
            ...     terms=[
            ...         TermCheck(term_id="HP:9999991", prefix="HP", status=TermStatus.NOT_FOUND),
            ...         TermCheck(term_id="HP:9999992", prefix="HP", status=TermStatus.NOT_FOUND),
            ...         TermCheck(term_id="HP:9999993", prefix="HP", status=TermStatus.NOT_FOUND),
            ...     ]
            ... ).all_terms_failed
            True
        """
        return (
            self.resolvable_count >= OUTAGE_HINT_MIN_TERMS
            and self.not_found_count == self.resolvable_count
        )

    @property
    def label_checked_terms(self) -> List[TermCheck]:
        """Terms whose reported label was actually compared with the ontology's."""
        return [
            t for t in self.checked_terms if t.agreement != LabelAgreement.NOT_ASSESSED
        ]

    @property
    def labels_checked(self) -> int:
        """Number of terms label agreement was judged for."""
        return len(self.label_checked_terms)

    @property
    def labels_matching(self) -> int:
        """Number of terms the report named exactly as the ontology does."""
        return sum(1 for t in self.checked_terms if t.agreement == LabelAgreement.MATCH)

    @property
    def mislabelled_terms(self) -> List[TermCheck]:
        """Terms the report calls something the ontology does not call them.

        This is the outcome the whole check exists for: the identifier resolves,
        so nothing about it looks wrong, and it denotes something else entirely.

        Examples:
            >>> report = TermValidationReport(
            ...     terms=[
            ...         TermCheck(
            ...             term_id="NCIT:C16814",
            ...             prefix="NCIT",
            ...             status=TermStatus.VERIFIED,
            ...             ontology_label="Malaysia",
            ...             reported_labels=["Echocardiography Test"],
            ...             agreement=LabelAgreement.MISMATCH,
            ...         )
            ...     ]
            ... )
            >>> [t.term_id for t in report.mislabelled_terms]
            ['NCIT:C16814']
        """
        return [t for t in self.checked_terms if t.agreement == LabelAgreement.MISMATCH]

    @property
    def variant_label_terms(self) -> List[TermCheck]:
        """Terms whose reported label is related to the ontology's, but not it.

        Not an error and not counted as one. "Long QT syndrome" for
        ``MONDO:0100316`` ("Long QT syndrome 1") is how a report cites a subtype
        loosely, and it is also how a report cites the wrong subtype.
        """
        return [t for t in self.checked_terms if t.agreement == LabelAgreement.VARIANT]

    @property
    def inconsistently_named_terms(self) -> List[TermCheck]:
        """Terms the report itself names in more than one way.

        Worth seeing even when every name passes: a report that calls one
        identifier two things has not decided what it cites.

        Examples:
            >>> TermValidationReport().inconsistently_named_terms
            []
        """
        return [t for t in self.checked_terms if len(t.reported_labels or []) > 1]

    @property
    def has_problems(self) -> bool:
        """Whether any term failed to resolve or was named something else.

        Obsolete terms and variant labels are deliberately excluded: both are
        real findings, but neither is a fabrication, and neither should fail a
        build. They reach the reader through :meth:`summary` instead.

        Examples:
            >>> TermValidationReport().has_problems
            False
        """
        return bool(self.confabulated_terms) or bool(self.mislabelled_terms)

    def summary(self) -> Dict[str, Any]:
        """Return a compact, YAML-friendly summary suitable for frontmatter.

        ``confabulation_rate`` measures identifier resolution and nothing else,
        which is a trap for anyone skimming: a report whose every CURIE resolves
        but whose labels name different terms still shows ``0.0``. So the counts
        that would contradict a reassuring rate are stated outright rather than
        left to be derived, and ``needs_review`` is set whenever any of them is
        non-empty.

        Examples:
            >>> report = TermValidationReport(
            ...     terms=[
            ...         TermCheck(
            ...             term_id="NCIT:C16814",
            ...             prefix="NCIT",
            ...             status=TermStatus.VERIFIED,
            ...             ontology_label="Malaysia",
            ...             reported_labels=["Echocardiography Test"],
            ...             agreement=LabelAgreement.MISMATCH,
            ...         )
            ...     ]
            ... )
            >>> summary = report.summary()
            >>> summary["confabulation_rate"], summary["labels_mismatched"]
            (0.0, 1)
            >>> summary["mislabelled_terms"]
            [{'term_id': 'NCIT:C16814', 'reported_labels': ['Echocardiography Test'], 'ontology_label': 'Malaysia'}]
            >>> summary["needs_review"]
            True
        """
        summary: Dict[str, Any] = {
            "total_terms": self.total_terms,
            "verified": self.verified_count,
            "not_found": self.not_found_count,
            "obsolete": self.obsolete_count,
            "unverifiable": self.unverifiable_count,
            "confabulation_rate": round(self.confabulation_rate, 3),
        }
        if self.labels_checked:
            summary["labels_checked"] = self.labels_checked
            summary["labels_matching"] = self.labels_matching
            if self.mislabelled_terms:
                summary["labels_mismatched"] = len(self.mislabelled_terms)
                summary["mislabelled_terms"] = [
                    _mislabelled_entry(check) for check in self.mislabelled_terms
                ]
            if self.variant_label_terms:
                summary["labels_variant"] = len(self.variant_label_terms)
        if self.confabulated_terms:
            summary["unresolved_terms"] = [t.term_id for t in self.confabulated_terms]
        if self.obsolete_terms:
            summary["obsolete_terms"] = [
                _obsolete_entry(check) for check in self.obsolete_terms
            ]
        if self.unresolvable_prefixes:
            summary["unresolvable_prefixes"] = list(self.unresolvable_prefixes)
        if self.has_problems or self.obsolete_terms or self.variant_label_terms:
            # Stated rather than implied, so that reading one number off the
            # frontmatter cannot produce a false all-clear.
            summary["needs_review"] = True
        if self.adapter:
            summary["adapter"] = self.adapter
        if self.validator_version:
            summary["validator_version"] = self.validator_version
        if self.truncated:
            summary["truncated"] = True
        return summary

    def to_markdown(self, heading: str = TERM_VALIDATION_SECTION_HEADING) -> str:
        """Render the report as a markdown section.

        Args:
            heading: Heading line to place above the section body.

        Returns:
            Markdown text describing the validation outcome.

        Examples:
            >>> report = TermValidationReport(
            ...     terms=[
            ...         TermCheck(
            ...             term_id="NCIT:C16814",
            ...             prefix="NCIT",
            ...             status=TermStatus.VERIFIED,
            ...             ontology_label="Malaysia",
            ...             reported_labels=["Echocardiography Test"],
            ...             agreement=LabelAgreement.MISMATCH,
            ...         )
            ...     ]
            ... )
            >>> md = report.to_markdown()
            >>> "Malaysia" in md and "Echocardiography Test" in md
            True
            >>> TermValidationReport().to_markdown().splitlines()[-1]
            'No ontology term identifiers were found in this report.'
            >>> "stopped early" in TermValidationReport(truncated=True).to_markdown()
            True
        """
        lines = [heading, ""]

        if self.truncated:
            lines.append(
                "Validation stopped early because the term limit was reached; the "
                "counts below cover only the terms that were checked."
            )
            lines.append("")

        if not self.checked_terms:
            lines.append("No ontology term identifiers were found in this report.")
            return "\n".join(lines)

        provenance = "Checked with `linkml-term-validator`"
        if self.validator_version:
            provenance += f" {self.validator_version}"
        if self.adapter:
            provenance += f", through the `{self.adapter}` adapter"
        lines.append(f"{provenance}.")
        lines.append("")

        lines.append("| Outcome | Count |")
        lines.append("| --- | --- |")
        lines.append(f"| Terms checked | {self.total_terms} |")
        lines.append(f"| Resolved | {self.verified_count} |")
        lines.append(f"| Unresolved (possible confabulation) | {self.not_found_count} |")
        lines.append(f"| Obsolete | {self.obsolete_count} |")
        lines.append(f"| Unverifiable | {self.unverifiable_count} |")
        if self.labels_checked:
            # Worded as terms, not labels, because that is what they count: a
            # report naming one identifier three ways contributes one to each.
            # The "Terms named inconsistently" section below tells a reader the
            # two are not the same thing, so these rows must not imply otherwise.
            lines.append(f"| Terms whose name was checked | {self.labels_checked} |")
            lines.append(f"| Terms named correctly | {self.labels_matching} |")
            # Spelled out rather than left as the difference of the two rows
            # above: a reader who subtracts is a reader who might not.
            lines.append(
                f"| Terms named as a **different** term | {len(self.mislabelled_terms)} |"
            )
            if self.variant_label_terms:
                lines.append(
                    f"| Terms whose name is worth a second look | {len(self.variant_label_terms)} |"
                )
        lines.append("")

        if self.mislabelled_terms:
            lines.append("### Terms the report names something else")
            lines.append("")
            lines.append(
                "These identifiers resolve, so nothing about them looks wrong, and "
                "the ontology calls them something unrelated to what the report "
                "calls them. That usually means the identifier is not the one the "
                "sentence needs:"
            )
            lines.append("")
            lines.extend(self._label_lines(self.mislabelled_terms))
            lines.append("")

        if self.confabulated_terms:
            lines.append("### Unresolved terms")
            lines.append("")
            if self.all_terms_failed:
                scope = (
                    "**Every** term failed to resolve"
                    if self.not_found_count == self.total_terms
                    else (
                        f"**Every** term that could be looked up failed to resolve "
                        f"({self.not_found_count} of {self.total_terms}; the rest had "
                        "no resolver)"
                    )
                )
                lines.append(
                    f"{scope}. That is far more often an unreachable ontology service "
                    "than a report in which every identifier is invented - check "
                    "connectivity and re-run before treating these as fabrications."
                )
            else:
                lines.append(
                    "These identifiers do not exist in an ontology that resolved other "
                    "terms from the same prefix, so they were most likely invented:"
                )
            lines.append("")
            for check in self.confabulated_terms:
                named = self._reported_as(check)
                detail = check.message or "did not resolve"
                lines.append(
                    f"- `{check.term_id}` ({self._mentions(check)}){named} - {detail}"
                )
            lines.append("")

        if self.obsolete_terms:
            lines.append("### Obsolete terms")
            lines.append("")
            lines.append(
                "These terms are real but deprecated. Citing one is not a fabrication; "
                "it does mean the report is naming something the ontology has retired:"
            )
            lines.append("")
            for check in self.obsolete_terms:
                replacement = (
                    f" - replaced by `{check.replaced_by}`" if check.replaced_by else ""
                )
                label = f" ({check.ontology_label})" if check.ontology_label else ""
                lines.append(
                    f"- `{check.term_id}`{label} ({self._mentions(check)}){replacement}"
                )
            lines.append("")

        if self.variant_label_terms:
            lines.append("### Terms whose name is worth a second look")
            lines.append("")
            lines.append(
                "The report's name for these is recognisably related to the term's own "
                "name without being one of them. A loose paraphrase reads the same way "
                "as a citation of the wrong sibling term - and so does a *related* "
                "synonym, which the ontology records precisely because it names "
                "something adjacent rather than the same thing - so these are listed "
                "rather than judged:"
            )
            lines.append("")
            lines.extend(self._label_lines(self.variant_label_terms))
            lines.append("")

        if self.inconsistently_named_terms:
            lines.append("### Terms named inconsistently")
            lines.append("")
            lines.append(
                "The report gives these identifiers more than one name of its own:"
            )
            lines.append("")
            for check in self.inconsistently_named_terms:
                names = ", ".join(f'"{name}"' for name in check.reported_labels or [])
                lines.append(f"- `{check.term_id}` - called {names}")
            lines.append("")

        if self.unresolvable_prefixes:
            lines.append("### Prefixes with no resolver")
            lines.append("")
            lines.append(
                "Terms carrying these prefixes were not checked either way, because no "
                "configured ontology covers them. An unrecognised prefix may name an "
                "ontology this run could not reach as easily as one that does not "
                "exist, so nothing here is evidence of fabrication: "
                + ", ".join(f"`{prefix}`" for prefix in self.unresolvable_prefixes)
                + "."
            )
            lines.append("")

        if not self.has_problems:
            if self.verified_count == self.total_terms:
                lines.append("Every term resolved, and every label the report gave matched.")
            elif self.verified_count:
                lines.append(
                    f"{self.verified_count} of {self.total_terms} terms resolved to a "
                    "current term; the rest could not be looked up either way."
                )
            else:
                lines.append(
                    "No term could be looked up either way, so nothing here was "
                    "confirmed or contradicted."
                )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _mentions(check: TermCheck) -> str:
        """Render an occurrence count as "1 mention" or "3 mentions".

        Examples:
            >>> TermValidationReport._mentions(
            ...     TermCheck(term_id="HP:11", prefix="HP", status=TermStatus.VERIFIED)
            ... )
            '1 mention'
        """
        return f"{check.occurrences} mention{'' if check.occurrences == 1 else 's'}"

    @staticmethod
    def _reported_as(check: TermCheck) -> str:
        """Render the names a report gave a term, for use mid-sentence.

        Examples:
            >>> TermValidationReport._reported_as(
            ...     TermCheck(term_id="HP:11", prefix="HP", status=TermStatus.NOT_FOUND)
            ... )
            ''
        """
        labels = check.reported_labels or []
        if not labels:
            return ""
        return ", reported as " + ", ".join(f'"{label}"' for label in labels)

    @staticmethod
    def _label_lines(checks: List[TermCheck]) -> List[str]:
        """Render one bullet per term contrasting its two names.

        Examples:
            >>> TermValidationReport._label_lines([
            ...     TermCheck(
            ...         term_id="NCIT:C16814",
            ...         prefix="NCIT",
            ...         status=TermStatus.VERIFIED,
            ...         ontology_label="Malaysia",
            ...         reported_labels=["Echocardiography Test"],
            ...         agreement=LabelAgreement.MISMATCH,
            ...     )
            ... ])
            ['- `NCIT:C16814` (1 mention) - the report calls it "Echocardiography Test"; NCIT calls it **Malaysia**']

            A name recognised as a synonym says which one, so a reader can see
            what the report was reaching for rather than guessing:

            >>> TermValidationReport._label_lines([
            ...     TermCheck(
            ...         term_id="HP:0001250",
            ...         prefix="HP",
            ...         status=TermStatus.VERIFIED,
            ...         ontology_label="Seizure",
            ...         reported_labels=["Epilepsy"],
            ...         agreement=LabelAgreement.VARIANT,
            ...         matched_synonym="Epilepsy",
            ...     )
            ... ])
            ['- `HP:0001250` (1 mention) - the report calls it "Epilepsy"; HP calls it **Seizure**, and lists "Epilepsy" among its other names']
        """
        lines = []
        for check in checks:
            reported = ", ".join(f'"{label}"' for label in check.reported_labels or [])
            synonym = (
                f', and lists "{check.matched_synonym}" among its other names'
                if check.matched_synonym
                else ""
            )
            lines.append(
                f"- `{check.term_id}` ({TermValidationReport._mentions(check)}) - "
                f"the report calls it {reported}; {check.prefix} calls it "
                f"**{check.ontology_label}**{synonym}"
            )
        return lines
