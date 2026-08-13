"""Data models for reference validation reports.

These models are plain Pydantic and carry no dependency on the optional
``linkml-reference-validator`` package, so a report can be loaded, serialised and
rendered anywhere.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ReferenceStatus(str, Enum):
    """Outcome of resolving a single reference identifier.

    Examples:
        >>> ReferenceStatus.NOT_FOUND.value
        'not_found'
    """

    VERIFIED = "verified"
    """The identifier resolves to a real record."""

    NOT_FOUND = "not_found"
    """The identifier could not be resolved; likely confabulated."""

    UNVERIFIABLE = "unverifiable"
    """The identifier was skipped, or resolves but exposes no checkable content."""


class ReferenceCheck(BaseModel):
    """Result of resolving one reference cited by a report.

    Examples:
        >>> check = ReferenceCheck(
        ...     reference_id="PMID:7913883",
        ...     status=ReferenceStatus.VERIFIED,
        ...     title="Mutations in the transmembrane domain of FGFR3",
        ...     year="1994",
        ... )
        >>> check.status
        <ReferenceStatus.VERIFIED: 'verified'>
        >>> check.is_confabulated
        False
    """

    reference_id: str = Field(..., description="Normalized identifier, e.g. PMID:7913883")
    status: ReferenceStatus = Field(..., description="Resolution outcome")
    occurrences: int = Field(default=1, description="Times the identifier is cited in the report")
    title: Optional[str] = Field(default=None, description="Title of the resolved record")
    year: Optional[str] = Field(default=None, description="Publication year of the resolved record")
    journal: Optional[str] = Field(default=None, description="Journal or venue of the resolved record")
    doi: Optional[str] = Field(default=None, description="DOI of the resolved record")
    content_type: Optional[str] = Field(
        default=None, description="Kind of content retrieved (abstract_only, full_text_xml, ...)"
    )
    message: Optional[str] = Field(default=None, description="Explanation, present when not verified")

    @property
    def is_confabulated(self) -> bool:
        """Whether this identifier failed to resolve at all.

        Examples:
            >>> ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.NOT_FOUND).is_confabulated
            True
        """
        return self.status == ReferenceStatus.NOT_FOUND


class SupportingTextCheck(BaseModel):
    """Result of checking a quoted claim against the text of its reference.

    Examples:
        >>> check = SupportingTextCheck(
        ...     reference_id="PMID:7913883",
        ...     quote="widgets are blue",
        ...     is_valid=False,
        ...     message="Text part not found as substring",
        ... )
        >>> check.is_valid
        False
    """

    reference_id: str = Field(..., description="Normalized identifier the quote is attributed to")
    quote: str = Field(..., description="Quoted text as it appears in the report")
    is_valid: bool = Field(..., description="Whether the quote was found in the reference")
    similarity_score: float = Field(default=0.0, description="Similarity of the closest match (0-1)")
    matched_text: Optional[str] = Field(default=None, description="Matching span in the reference")
    best_match: Optional[str] = Field(default=None, description="Closest non-matching span, if any")
    suggested_fix: Optional[str] = Field(default=None, description="Suggested correction from the validator")
    message: Optional[str] = Field(default=None, description="Validator explanation")


class ReferenceValidationReport(BaseModel):
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

    references: List[ReferenceCheck] = Field(
        default_factory=list, description="Per-reference resolution results"
    )
    supporting_text: List[SupportingTextCheck] = Field(
        default_factory=list, description="Per-quote supporting text results"
    )
    validator_version: Optional[str] = Field(
        default=None, description="Version of linkml-reference-validator used"
    )
    truncated: bool = Field(
        default=False, description="Whether validation stopped early due to a reference limit"
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
        return sum(1 for r in self.references if r.status == ReferenceStatus.NOT_FOUND)

    @property
    def unverifiable_count(self) -> int:
        """Number of references that were skipped or exposed no content."""
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
        """References that failed to resolve."""
        return [r for r in self.references if r.is_confabulated]

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
