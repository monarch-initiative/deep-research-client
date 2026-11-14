"""Evidence files saved beside a report, and a local check of the report against them.

An evidence file is the text of one source as an agent retrieved it, with YAML
frontmatter naming the source::

    ---
    key: hodges-1971-clinical-manifestations
    identifiers: [PMID:5568449]
    source_url: https://pubmed.ncbi.nlm.nih.gov/5568449/
    content: abstract
    ---
    <the retrieved text>

:func:`check_against_evidence` checks a report against its evidence files with
no network access. It finds identifiers the report cites with no evidence file
(recalled rather than retrieved), and quotations attributed to a source,
written ``"..." (PMID:5568449)``, that do not occur in that source's evidence
text. It checks traceability, not truth: whether the evidence is relevant to,
and supports, the sentence it is attached to is left to a reader or grader.
Resolving the identifiers themselves is ``validate-references``' job.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from .extraction import extract_quoted_claims, find_reference_ids

_FRONTMATTER_DELIMITER = "---"


class EvidenceFile(BaseModel):
    """One retrieved source, as saved by an agent."""

    filename: str = Field(description="Name of the evidence file")
    key: str = Field(description="Citation key, the file's stem when the frontmatter has none")
    identifiers: List[str] = Field(
        default_factory=list,
        description="Normalized identifiers (PMID:, DOI:, ...) the source carries",
    )
    title: Optional[str] = Field(default=None, description="Title of the source")
    source_url: Optional[str] = Field(default=None, description="URL the text was fetched from")
    content: Optional[str] = Field(default=None, description="abstract, fulltext or excerpt")
    text: str = Field(description="The retrieved text, after the frontmatter")

    @property
    def citation(self) -> str:
        """How the source is cited: its first identifier, else its URL, else its key."""
        return (self.identifiers[0] if self.identifiers else None) or self.source_url or self.key


class UnanchoredQuote(BaseModel):
    """A quotation in the report that its cited evidence does not contain."""

    quote: str = Field(description="The quoted text")
    reference_id: str = Field(description="The identifier the quote is attributed to")
    reason: str = Field(description="Why it was not found")


class EvidenceCheck(BaseModel):
    """Result of checking a report against its evidence files."""

    evidence_files: int = Field(description="Evidence files read")
    cited_identifiers: int = Field(description="Distinct identifiers the report cites")
    cited_without_evidence: List[str] = Field(
        default_factory=list,
        description="Identifiers the report cites that no evidence file carries",
    )
    quotes_checked: int = Field(default=0, description="Attributed quotations in the report")
    quotes_anchored: int = Field(default=0, description="Quotations found in their cited evidence")
    unanchored_quotes: List[UnanchoredQuote] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every cited identifier has evidence and every quotation was found.

        >>> EvidenceCheck(evidence_files=0, cited_identifiers=0).passed
        True
        >>> EvidenceCheck(evidence_files=0, cited_identifiers=1, cited_without_evidence=["PMID:1"]).passed
        False
        """
        return not self.cited_without_evidence and not self.unanchored_quotes

    def failure_summary(self) -> Optional[str]:
        """One line saying what failed, or None if the check passed.

        >>> print(EvidenceCheck(evidence_files=1, cited_identifiers=2, cited_without_evidence=["PMID:1"],
        ...     unanchored_quotes=[UnanchoredQuote(quote="q", reference_id="PMID:2", reason="r")]).failure_summary())
        1 quotation not found in its cited evidence; 1 cited identifier with no evidence file
        """
        parts = []
        if self.unanchored_quotes:
            count = len(self.unanchored_quotes)
            parts.append(f"{count} quotation{'s' if count != 1 else ''} not found in {'their' if count != 1 else 'its'} cited evidence")
        if self.cited_without_evidence:
            count = len(self.cited_without_evidence)
            parts.append(f"{count} cited identifier{'s' if count != 1 else ''} with no evidence file")
        return "; ".join(parts) or None


def frontmatter_fields(lines: List[str]) -> dict[str, str]:
    """Read top-level ``name: value`` fields, each with any indented lines after it.

    Deliberately not YAML. Agents write free-text values -- a title such as
    ``Scurvy: a narrative review`` -- that strict YAML rejects, and one such
    file would otherwise hide a source's identifiers from the check. Splitting
    at the first colon reads every field, whatever its value contains.

    Args:
        lines: The lines between the frontmatter delimiters.

    Returns:
        Field names mapped to their raw text, continuation lines included.

    >>> frontmatter_fields(["title: Scurvy: a review", "identifiers:", "  - PMID:1"])
    {'title': 'Scurvy: a review', 'identifiers': '\\n  - PMID:1'}
    """
    fields: dict[str, str] = {}
    current: Optional[str] = None
    for line in lines:
        name, colon, value = line.partition(":")
        if colon and line[:1].strip() and not line.startswith("-"):
            current = name.strip()
            fields[current] = value.strip()
        elif current is not None:
            fields[current] += "\n" + line
    return fields


def _scalar(fields: dict[str, str], name: str) -> Optional[str]:
    """A field's value as one line, without surrounding quotes, or None if empty."""
    value = fields.get(name, "").strip().strip("\"'").strip()
    return value or None


def parse_evidence_file(filename: str, text: str) -> Optional[EvidenceFile]:
    """Read an evidence file, or return None for a file that is not one.

    The frontmatter is read with :func:`frontmatter_fields`, not as YAML, so a
    malformed free-text field cannot lose the source. Identifiers are found in
    the ``identifiers`` field's text and normalized the way the report's are,
    so ``PMID: 5568449`` in the frontmatter matches ``(PMID:5568449)`` in the
    report, in a block list or an inline one.

    Args:
        filename: The file's name.
        text: The file's contents.

    Returns:
        The evidence file, or None when there is no frontmatter.

    >>> evidence = parse_evidence_file("hodges-1971.md",
    ...     "---\\ntitle: Scurvy: a review\\nidentifiers: ['PMID: 5568449']\\n---\\nScurvy appeared.\\n")
    >>> (evidence.key, evidence.title, evidence.identifiers, evidence.text)
    ('hodges-1971', 'Scurvy: a review', ['PMID:5568449'], 'Scurvy appeared.\\n')
    >>> parse_evidence_file("notes.md", "Just notes.") is None
    True
    """
    lines = text.split("\n")
    if lines[0].strip() != _FRONTMATTER_DELIMITER:
        return None
    for end, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIMITER:
            break
    else:
        return None
    fields = frontmatter_fields(lines[1:end])
    if not fields:
        return None
    return EvidenceFile(
        filename=filename,
        key=_scalar(fields, "key") or filename.rsplit(".", 1)[0],
        identifiers=[found.normalized_id for found in find_reference_ids(fields.get("identifiers", ""))],
        title=_scalar(fields, "title"),
        source_url=_scalar(fields, "source_url"),
        content=_scalar(fields, "content"),
        text="\n".join(lines[end + 1:]),
    )


def check_against_evidence(markdown: str, evidence: List[EvidenceFile]) -> EvidenceCheck:
    """Check a report's citations and quotations against its evidence files.

    A quotation counts as found when the claims-anchoring matcher locates it in
    the text of any evidence file carrying the cited identifier, verbatim or
    after normalising whitespace, markdown and quotation marks.

    Args:
        markdown: The report.
        evidence: The evidence files saved with it.

    Returns:
        What was checked and what was not found.

    >>> source = parse_evidence_file("h.md",
    ...     "---\\nidentifiers: [PMID:5568449]\\n---\\nThe first signs of scurvy appeared after 29 days.\\n")
    >>> report = ('Signs came early: "the first signs of scurvy appeared after 29 days" (PMID:5568449). '
    ...     'Others disagree (PMID:12345678).')
    >>> check = check_against_evidence(report, [source])
    >>> (check.quotes_checked, check.quotes_anchored, check.cited_without_evidence)
    (1, 1, ['PMID:12345678'])
    """
    from ..claims.anchoring import locate_quote
    from ..claims.models import AnchorStatus

    by_identifier: dict[str, list[EvidenceFile]] = {}
    for source in evidence:
        for identifier in source.identifiers:
            by_identifier.setdefault(identifier, []).append(source)

    cited = sorted({found.normalized_id for found in find_reference_ids(markdown)})
    check = EvidenceCheck(
        evidence_files=len(evidence),
        cited_identifiers=len(cited),
        cited_without_evidence=[identifier for identifier in cited if identifier not in by_identifier],
    )
    for claim in extract_quoted_claims(markdown):
        check.quotes_checked += 1
        sources = by_identifier.get(claim.reference_id, [])
        if not sources:
            reason = "no evidence file carries this identifier"
        elif any(locate_quote(claim.quote, source.text)[1] is not AnchorStatus.UNANCHORED
                 for source in sources):
            check.quotes_anchored += 1
            continue
        else:
            reason = "not found in the cited evidence"
        check.unanchored_quotes.append(
            UnanchoredQuote(quote=claim.quote, reference_id=claim.reference_id, reason=reason)
        )
    return check
