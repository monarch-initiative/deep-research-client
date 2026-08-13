"""Derived views over the reference validation data model.

The data shape lives in ``reference_validation.yaml`` and is generated into
:mod:`deep_research_client.validation.datamodel` with ``just gen-datamodel``.
This module adds what a schema cannot express: quantities computed from those
slots, and rendering of a report as markdown or as frontmatter.

:class:`ReferenceCheck` and :class:`SupportingTextCheck` are re-exported from the
generated model unchanged; only the report gains behaviour.
"""

from typing import Any, Dict, List

from pydantic import Field

from .datamodel import ReferenceCheck, ReferenceStatus, SupportingTextCheck
from .datamodel import ReferenceValidationReport as GeneratedReferenceValidationReport

__all__ = [
    "ReferenceCheck",
    "ReferenceStatus",
    "ReferenceValidationReport",
    "SupportingTextCheck",
]


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

    # The generated slots are Optional[list] with a [] default, which would force
    # every accessor below to test for None. Narrow them here, reusing the
    # generated descriptions so the schema stays the single source of that prose.
    references: List[ReferenceCheck] = Field(
        default_factory=list,
        description=GeneratedReferenceValidationReport.model_fields["references"].description,
    )
    supporting_text: List[SupportingTextCheck] = Field(
        default_factory=list,
        description=GeneratedReferenceValidationReport.model_fields[
            "supporting_text"
        ].description,
    )

    @property
    def total_references(self) -> int:
        """Number of unique references checked.

        Examples:
            >>> ReferenceValidationReport().total_references
            0
        """
        return len(self.references)

    @property
    def verified_count(self) -> int:
        """Number of references that resolved successfully."""
        return sum(1 for r in self.references if r.status == ReferenceStatus.VERIFIED)

    @property
    def not_found_count(self) -> int:
        """Number of references that failed to resolve."""
        return len(self.confabulated_references)

    @property
    def unverifiable_count(self) -> int:
        """Number of references that were skipped or have no resolver."""
        return sum(1 for r in self.references if r.status == ReferenceStatus.UNVERIFIABLE)

    @property
    def confabulation_rate(self) -> float:
        """Fraction of references that failed to resolve.

        Returns ``0.0`` when there are no references to check.

        Examples:
            >>> ReferenceValidationReport().confabulation_rate
            0.0
        """
        if not self.references:
            return 0.0
        return self.not_found_count / len(self.references)

    @property
    def quotes_checked(self) -> int:
        """Number of quoted claims checked."""
        return len(self.supporting_text)

    @property
    def quotes_valid_count(self) -> int:
        """Number of quoted claims found in their reference."""
        return sum(1 for q in self.supporting_text if q.is_valid)

    @property
    def unsupported_quotes(self) -> List[SupportingTextCheck]:
        """Quoted claims that could not be found in their reference."""
        return [q for q in self.supporting_text if not q.is_valid]

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
        return [r for r in self.references if r.status == ReferenceStatus.NOT_FOUND]

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
        if self.supporting_text:
            summary["quotes_checked"] = self.quotes_checked
            summary["quotes_valid"] = self.quotes_valid_count
        if self.confabulated_references:
            summary["unresolved_references"] = [r.reference_id for r in self.confabulated_references]
        if self.validator_version:
            summary["validator_version"] = self.validator_version
        if self.truncated:
            summary["truncated"] = True
        return summary

    def to_markdown(self, heading: str = "## Reference Validation") -> str:
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
                "the counts below cover only the references that were checked."
            )
            lines.append("")

        if not self.references:
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
        if self.supporting_text:
            lines.append(f"| Quoted claims checked | {self.quotes_checked} |")
            lines.append(f"| Quoted claims found in source | {self.quotes_valid_count} |")
        lines.append("")

        if self.confabulated_references:
            lines.append("### Unresolved references")
            lines.append("")
            lines.append(
                "These identifiers did not resolve to a record and may be fabricated:"
            )
            lines.append("")
            for check in self.confabulated_references:
                detail = check.message or "could not be resolved"
                lines.append(f"- `{check.reference_id}` (cited {check.occurrences}x) - {detail}")
            lines.append("")

        if self.unsupported_quotes:
            lines.append("### Quotes not found in the cited source")
            lines.append("")
            for quote_check in self.unsupported_quotes:
                lines.append(f"- `{quote_check.reference_id}`: \"{quote_check.quote}\"")
                if quote_check.best_match:
                    lines.append(f"  - closest text in source: \"{quote_check.best_match}\"")
                elif quote_check.message:
                    lines.append(f"  - {quote_check.message}")
            lines.append("")

        if not self.has_confabulations:
            lines.append("All extracted references resolved successfully.")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
