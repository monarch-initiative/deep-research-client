"""Derived views over the reference validation data model.

The data shape lives in ``reference_validation.yaml`` and is generated into
:mod:`deep_research_client.validation.datamodel` with ``just gen-datamodel``.
This module adds what a schema cannot express: quantities computed from those
slots, and rendering of a report as markdown or as frontmatter.

:class:`ReferenceCheck` and :class:`SupportingTextCheck` are re-exported from the
generated model unchanged; only the report gains behaviour, and it adds no fields
of its own so that its JSON schema stays exactly what LinkML would emit.

One inherited quirk is worth knowing about. The generated ``ConfiguredBaseModel``
sets ``use_enum_values = True``, so an enum-ranged slot stores the enum's *value*
rather than the member::

    >>> check = ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED)
    >>> check.status
    'VERIFIED'
    >>> isinstance(check.status, ReferenceStatus)
    False

Comparisons still work, because :class:`ReferenceStatus` is a ``str`` enum::

    >>> check.status == ReferenceStatus.VERIFIED
    True

But ``check.status.value`` and ``check.status.name`` raise ``AttributeError`` at
runtime even though the annotation says otherwise, so compare against members
rather than reaching for attributes on them.
"""

import re
from typing import Any, Dict, List

from .datamodel import ReferenceCheck, ReferenceStatus, SupportingTextCheck
from .datamodel import ReferenceValidationReport as GeneratedReferenceValidationReport

__all__ = [
    "VALIDATION_SECTION_HEADING",
    "ReferenceCheck",
    "ReferenceStatus",
    "ReferenceValidationReport",
    "SupportingTextCheck",
    "strip_validation_section",
]

VALIDATION_SECTION_HEADING = "## Reference Validation"

_H2_HEADING_RE = re.compile(r"^##[ \t]+\S.*$", re.MULTILINE)


def strip_validation_section(markdown: str) -> str:
    """Remove a previously written validation section from a report.

    Validating a report that already carries a section would otherwise re-extract
    the identifiers that section lists, re-fetching flagged references and
    inflating the counts, so a second run must start from the original text.

    Only a *trailing* section is removed: a generated section is always appended
    last and contains no further level-two headings, so it is safe to strip
    exactly when the final ``##`` heading in the document is the validation
    heading. A report that discusses reference validation in its body and then
    continues with another section keeps everything, which matters because the
    caller writes this result back over the file.

    The one case it cannot see through is a validation heading inside a fenced
    code block that happens to be the last ``##`` in the file. Recognising that
    would mean parsing markdown rather than scanning it.

    Args:
        markdown: Report text, possibly ending in a validation section.

    Returns:
        The report without its trailing validation section, with trailing blank
        lines normalised to a single newline.

    Examples:
        >>> strip_validation_section("# Report\\n\\nBody text.\\n")
        '# Report\\n\\nBody text.\\n'
        >>> strip_validation_section(
        ...     "# Report\\n\\nBody text.\\n\\n## Reference Validation\\n\\nSomething.\\n"
        ... )
        '# Report\\n\\nBody text.\\n'
        >>> strip_validation_section("## Reference Validation\\n\\nOnly a section.\\n")
        ''

        A validation heading that is not the last section is left alone, along
        with everything after it:

        >>> strip_validation_section(
        ...     "# Report\\n\\n## Reference Validation\\n\\nWe discuss it.\\n"
        ...     "\\n## Conclusions\\n\\nImportant text.\\n"
        ... )
        '# Report\\n\\n## Reference Validation\\n\\nWe discuss it.\\n\\n## Conclusions\\n\\nImportant text.\\n'
    """
    text = markdown
    while True:
        headings = _H2_HEADING_RE.findall(text)
        if not headings or headings[-1].strip() != VALIDATION_SECTION_HEADING:
            break
        # Repeat, so a file left with stacked sections by an older run is cleaned
        # up rather than losing only the last of them.
        last_start = text.rindex(headings[-1])
        text = text[:last_start]
    return text.rstrip() + "\n" if text.strip() else ""


class ReferenceValidationReport(GeneratedReferenceValidationReport):
    """Aggregate result of validating every reference in a report.

    Examples:
        >>> report = ReferenceValidationReport(
        ...     references=[
        ...         ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED),
        ...         ReferenceCheck(reference_id="PMID:2", status=ReferenceStatus.NOT_FOUND),
        ...     ]
        ... )
        >>> report.total_references
        2
        >>> report.verified_count
        1
        >>> report.not_found_count
        1
        >>> report.confabulation_rate
        0.5
        >>> report.has_confabulations
        True
    """

    @property
    def checked_references(self) -> List[ReferenceCheck]:
        """The per-reference results, normalised to a list.

        The generated slot is ``Optional[list]``, because LinkML has no way to
        give a multivalued slot a default without also making it mandatory.
        Leaving the annotation as generated keeps the inherited serializer
        working (it turns empty lists into nulls under ``exclude_none``), so the
        normalisation happens here instead.

        Examples:
            >>> ReferenceValidationReport().checked_references
            []
        """
        return self.references or []

    @property
    def quote_checks(self) -> List[SupportingTextCheck]:
        """The per-quote results, normalised to a list.

        Examples:
            >>> ReferenceValidationReport().quote_checks
            []
        """
        return self.supporting_text or []

    @property
    def total_references(self) -> int:
        """Number of unique references checked.

        Examples:
            >>> ReferenceValidationReport().total_references
            0
        """
        return len(self.checked_references)

    @property
    def verified_count(self) -> int:
        """Number of references that resolved successfully."""
        return sum(1 for r in self.checked_references if r.status == ReferenceStatus.VERIFIED)

    @property
    def not_found_count(self) -> int:
        """Number of references that failed to resolve."""
        return len(self.confabulated_references)

    @property
    def unverifiable_count(self) -> int:
        """Number of references that were skipped or have no resolver."""
        return sum(
            1 for r in self.checked_references if r.status == ReferenceStatus.UNVERIFIABLE
        )

    @property
    def resolvable_count(self) -> int:
        """Number of references a lookup actually returned an answer about.

        Excludes unverifiable references, about which nothing was learned.
        """
        return self.verified_count + self.not_found_count

    @property
    def confabulation_rate(self) -> float:
        """Fraction of the references we got an answer about that failed to resolve.

        Unverifiable references are excluded from the denominator. Counting them
        as successes would let ``--skip-prefix`` silently dilute the rate, so
        that skipping the half of a bibliography that is fabricated would halve
        the reported figure.

        Returns ``0.0`` when nothing was resolvable.

        Examples:
            >>> ReferenceValidationReport().confabulation_rate
            0.0
            >>> ReferenceValidationReport(
            ...     references=[
            ...         ReferenceCheck(reference_id="A:1", status=ReferenceStatus.NOT_FOUND),
            ...         ReferenceCheck(reference_id="A:2", status=ReferenceStatus.VERIFIED),
            ...         ReferenceCheck(reference_id="A:3", status=ReferenceStatus.UNVERIFIABLE),
            ...     ]
            ... ).confabulation_rate
            0.5
        """
        if not self.resolvable_count:
            return 0.0
        return self.not_found_count / self.resolvable_count

    @property
    def all_references_failed(self) -> bool:
        """Whether every reference a lookup answered about failed to resolve.

        A whole bibliography failing at once is far more often a network or
        rate-limit problem than a report in which every citation is invented, so
        the rendered report hedges when this is true.

        Measured against :attr:`resolvable_count`, not the total. During an
        outage the identifier types that fail closed become unverifiable while
        the rest become not-found, so comparing against the total would suppress
        this banner in exactly the situation it was written for.

        Examples:
            >>> ReferenceValidationReport().all_references_failed
            False
            >>> ReferenceValidationReport(
            ...     references=[
            ...         ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.NOT_FOUND),
            ...         ReferenceCheck(reference_id="PMC:PMC99999", status=ReferenceStatus.UNVERIFIABLE),
            ...     ]
            ... ).all_references_failed
            True
        """
        return bool(self.resolvable_count) and self.not_found_count == self.resolvable_count

    @property
    def quotes_checked(self) -> int:
        """Number of quoted claims that could actually be checked."""
        return sum(1 for q in self.quote_checks if q.was_checkable)

    @property
    def quotes_valid_count(self) -> int:
        """Number of quoted claims found in their reference."""
        return sum(1 for q in self.quote_checks if q.was_checkable and q.is_valid)

    @property
    def unsupported_quotes(self) -> List[SupportingTextCheck]:
        """Quoted claims that were checked and not found in their reference."""
        return [q for q in self.quote_checks if q.was_checkable and not q.is_valid]

    @property
    def unchecked_quotes(self) -> List[SupportingTextCheck]:
        """Quoted claims there was nothing to check against.

        These are not evidence of confabulation: the source was unavailable, not
        contradictory.

        Examples:
            >>> ReferenceValidationReport().unchecked_quotes
            []
        """
        return [q for q in self.quote_checks if not q.was_checkable]

    @property
    def confabulated_references(self) -> List[ReferenceCheck]:
        """References that failed to resolve, and so may have been fabricated.

        Examples:
            >>> report = ReferenceValidationReport(
            ...     references=[
            ...         ReferenceCheck(reference_id="PMID:2", status=ReferenceStatus.NOT_FOUND)
            ...     ]
            ... )
            >>> [r.reference_id for r in report.confabulated_references]
            ['PMID:2']
        """
        return [r for r in self.checked_references if r.status == ReferenceStatus.NOT_FOUND]

    @property
    def has_confabulations(self) -> bool:
        """Whether any reference failed to resolve or any quote failed to match.

        Examples:
            >>> ReferenceValidationReport().has_confabulations
            False
        """
        return bool(self.confabulated_references) or bool(self.unsupported_quotes)

    def summary(self) -> Dict[str, Any]:
        """Return a compact, YAML-friendly summary suitable for frontmatter.

        Examples:
            >>> report = ReferenceValidationReport(
            ...     references=[ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED)]
            ... )
            >>> report.summary()["verified"]
            1
            >>> report.summary()["confabulation_rate"]
            0.0
        """
        summary: Dict[str, Any] = {
            "total_references": self.total_references,
            "verified": self.verified_count,
            "not_found": self.not_found_count,
            "unverifiable": self.unverifiable_count,
            "confabulation_rate": round(self.confabulation_rate, 3),
        }
        if self.quote_checks:
            summary["quotes_checked"] = self.quotes_checked
            summary["quotes_valid"] = self.quotes_valid_count
            if self.unchecked_quotes:
                summary["quotes_not_checkable"] = len(self.unchecked_quotes)
        if self.confabulated_references:
            summary["unresolved_references"] = [r.reference_id for r in self.confabulated_references]
        if self.validator_version:
            summary["validator_version"] = self.validator_version
        if self.truncated:
            summary["truncated"] = True
        return summary

    def to_markdown(self, heading: str = VALIDATION_SECTION_HEADING) -> str:
        """Render the report as a markdown section.

        Args:
            heading: Heading line to place above the section body.

        Returns:
            Markdown text describing the validation outcome.

        Examples:
            >>> report = ReferenceValidationReport(
            ...     references=[
            ...         ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED),
            ...         ReferenceCheck(
            ...             reference_id="PMID:2",
            ...             status=ReferenceStatus.NOT_FOUND,
            ...             message="Could not fetch reference",
            ...         ),
            ...     ]
            ... )
            >>> md = report.to_markdown()
            >>> "Unresolved references" in md
            True
            >>> "PMID:2" in md
            True
            >>> ReferenceValidationReport().to_markdown().splitlines()[-1]
            'No PMID or DOI references were found in this report.'
            >>> "stopped early" in ReferenceValidationReport(truncated=True).to_markdown()
            True
        """
        lines = [heading, ""]

        if self.truncated:
            lines.append(
                "Validation stopped early because the reference limit was reached; "
                "the counts below cover only the references that were checked, and "
                "quotes attributed to the rest were left unchecked."
            )
            lines.append("")

        if not self.checked_references:
            lines.append("No PMID or DOI references were found in this report.")
            return "\n".join(lines)

        provenance = "Checked with `linkml-reference-validator`"
        if self.validator_version:
            provenance += f" {self.validator_version}"
        lines.append(f"{provenance}.")
        lines.append("")

        lines.append("| Outcome | Count |")
        lines.append("| --- | --- |")
        lines.append(f"| References checked | {self.total_references} |")
        lines.append(f"| Resolved | {self.verified_count} |")
        lines.append(f"| Unresolved (possible confabulation) | {self.not_found_count} |")
        lines.append(f"| Unverifiable | {self.unverifiable_count} |")
        if self.quote_checks:
            lines.append(f"| Quoted claims checked | {self.quotes_checked} |")
            lines.append(f"| Quoted claims found in source | {self.quotes_valid_count} |")
            if self.unchecked_quotes:
                lines.append(
                    f"| Quoted claims with nothing to check against | {len(self.unchecked_quotes)} |"
                )
        lines.append("")

        if self.confabulated_references:
            lines.append("### Unresolved references")
            lines.append("")
            if self.all_references_failed:
                # Worded from resolvable_count, which is what the property
                # measures: saying "every reference" when most were unverifiable
                # would misdescribe the report to the person deciding whether to
                # investigate.
                scope = (
                    "**Every** reference failed to resolve"
                    if self.not_found_count == self.total_references
                    else (
                        f"**Every** reference that could be looked up failed to resolve "
                        f"({self.not_found_count} of {self.total_references}; the rest "
                        "were unverifiable)"
                    )
                )
                lines.append(
                    f"{scope}. That is far more often a network outage or an API rate "
                    "limit than a report in which every citation is invented - check "
                    "connectivity and re-run before treating these as fabrications."
                )
            else:
                lines.append(
                    "These identifiers did not resolve to a record and may be fabricated. "
                    "A lookup that failed for transport reasons is indistinguishable from "
                    "one that failed because the record does not exist, so spot-check "
                    "before acting on them:"
                )
            lines.append("")
            for check in self.confabulated_references:
                detail = check.message or "could not be resolved"
                mentions = check.occurrences
                plural = "" if mentions == 1 else "s"
                lines.append(
                    f"- `{check.reference_id}` ({mentions} mention{plural}) - {detail}"
                )
            lines.append("")

        if self.unsupported_quotes:
            lines.append("### Quotes not found in the cited source")
            lines.append("")
            lines.append(
                "Searched the abstract, any retrieved full text, and the title. A "
                "quote drawn from a part of the paper that was not retrieved will "
                "appear here too, so check before treating one as invented:"
            )
            lines.append("")
            for quote_check in self.unsupported_quotes:
                lines.append(f"- `{quote_check.reference_id}`: \"{quote_check.quote}\"")
                if quote_check.best_match:
                    lines.append(f"  - closest text in source: \"{quote_check.best_match}\"")
                elif quote_check.message:
                    lines.append(f"  - {quote_check.message}")
            lines.append("")

        if self.unchecked_quotes:
            lines.append("### Quotes that could not be checked")
            lines.append("")
            lines.append(
                "There was no text to compare these against, so they are neither "
                "confirmed nor contradicted:"
            )
            lines.append("")
            for quote_check in self.unchecked_quotes:
                lines.append(f"- `{quote_check.reference_id}`: \"{quote_check.quote}\"")
                if quote_check.message:
                    lines.append(f"  - {quote_check.message}")
            lines.append("")

        if not self.has_confabulations:
            if self.verified_count == self.total_references:
                lines.append("All extracted references resolved successfully.")
            elif self.verified_count:
                lines.append(
                    f"{self.verified_count} of {self.total_references} references "
                    "resolved; the rest could not be looked up either way."
                )
            else:
                lines.append(
                    "No reference could be looked up either way, so nothing here "
                    "was confirmed or contradicted."
                )
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
