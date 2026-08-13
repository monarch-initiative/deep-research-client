"""Extraction of reference identifiers and quoted claims from report markdown.

This module is deliberately dependency-free (standard library only) so that it
can be used without installing the optional ``validation`` extra. The actual
resolution of the extracted identifiers lives in
:mod:`deep_research_client.validation.validator`.
"""

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# Patterns for PMID and DOI references in markdown text. The negative lookahead on
# PMID stops a longer number (an accession, a phone number) from being truncated
# into a plausible-looking nine-digit PMID.
_PMID_PATTERN = re.compile(r"PMID[:\s]*(\d{6,9})(?!\d)", re.IGNORECASE)
# The capture stops at characters that never occur inside a DOI, so a DOI in a
# tight table cell (|doi:10.1/a|b|) does not swallow the next cell. Parentheses
# are deliberately allowed through, since DOIs such as
# 10.1016/0092-8674(94)90302-6 contain them; trailing ones are stripped below.
#
# The third alternative matches publisher landing pages - pnas.org/doi/10.1073/x,
# onlinelibrary.wiley.com/doi/10.1002/x, tandfonline.com/doi/full/10.1080/x - which
# is how deep research tools most often render a DOI. Without it those citations
# are silently left unchecked, so the report understates its own coverage.
_DOI_PATTERN = re.compile(
    r"(?:"
    r"doi[:\s]*"
    r"|https?://(?:dx\.)?doi\.org/"
    r"|/doi/(?:abs/|full/|pdf/|epdf/|epub/)?"
    r")(10\.\d{4,}/[^\s|`\"<>]+)",
    re.IGNORECASE,
)
# Both the current pubmed.ncbi.nlm.nih.gov host and the older
# www.ncbi.nlm.nih.gov/pubmed path, which providers still emit.
_URL_PATTERN = re.compile(
    r"https?://(?:pubmed\.ncbi\.nlm\.nih\.gov|(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed)/(\d+)"
)

# A markdown escape: a backslash before an ASCII punctuation character. Providers
# emit these inside DOIs (10.1007/978-3-030-80614-9\_8), and a DOI never contains
# a backslash of its own, so unescaping is safe.
_MARKDOWN_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")

# Characters that markdown routinely glues onto the end of a DOI and that a DOI
# will never legitimately end with. The emphasis and code characters matter most:
# a DOI written as **doi:10.1234/abc** or `doi:10.1234/abc` would otherwise keep
# its wrapper, fail to resolve, and be reported as a possible fabrication.
_DOI_TRAILING_CHARS = ".,;:)]}>\"'*_`|~^\\"

# A double-quoted span (straight or typographic quotes) immediately followed by a
# parenthesised or bracketed citation, e.g.::
#
#     "widgets are blue in most cases" (PMID:12345678)
#
# The 20-character minimum keeps single quoted terms ("gene X") out of the results.
#
# The citation group admits one level of nested parentheses so that a DOI
# containing them - 10.1016/0092-8674(94)90302-6 - is not cut short at its first
# closing bracket. It stays bracket-delimited rather than running to end of line,
# which would let one quote absorb the citation of the next sentence.
_QUOTED_CLAIM_PATTERN = re.compile(
    r"[\"“”]([^\"“”\n]{20,600})[\"“”]"
    r"[\s,;:—-]*"
    r"[(\[]((?:[^()\[\]\n]|\([^()\n]*\)){3,200})[)\]]"
)


@dataclass(frozen=True)
class FoundReference:
    """A reference identifier discovered in report text.

    Attributes:
        normalized_id: Canonical identifier such as ``PMID:12345678`` or
            ``DOI:10.1038/ng1234``, in the form understood by
            ``linkml-reference-validator``.
        raw: The raw substring the identifier was extracted from.
        count: Number of times the identifier occurs in the scanned text.
        url: Source URL, when the identifier was recovered from a link.

    Examples:
        >>> FoundReference(normalized_id="PMID:1", raw="PMID:1").count
        1
    """

    normalized_id: str
    raw: str
    count: int = 1
    url: Optional[str] = None


@dataclass(frozen=True)
class QuotedClaim:
    """A quoted span of text paired with the reference it is attributed to.

    Attributes:
        quote: The quoted text, without the surrounding quotation marks.
        reference_id: Normalized identifier of the cited reference.

    Examples:
        >>> QuotedClaim(quote="widgets are blue", reference_id="PMID:1").reference_id
        'PMID:1'
    """

    quote: str
    reference_id: str


def normalize_doi(doi: str) -> str:
    """Undo markdown escaping and strip punctuation that trails a DOI.

    Args:
        doi: A raw DOI string, possibly with trailing punctuation or markup.

    Returns:
        The DOI as the publisher registered it.

    Examples:
        >>> normalize_doi("10.1038/ng1234).")
        '10.1038/ng1234'
        >>> normalize_doi("10.1038/ng1234**")
        '10.1038/ng1234'
        >>> normalize_doi("10.1038/ng1234`")
        '10.1038/ng1234'
        >>> normalize_doi("10.1038/ng1234|")
        '10.1038/ng1234'
        >>> normalize_doi("10.1038/ng1234")
        '10.1038/ng1234'

        An escaped underscore, as providers write DOIs for book chapters:

        >>> normalize_doi(r"10.1007/978-3-030-80614-9\\_8")
        '10.1007/978-3-030-80614-9_8'
    """
    return _MARKDOWN_ESCAPE.sub(r"\1", doi).rstrip(_DOI_TRAILING_CHARS)


def find_reference_ids(text: str) -> list[FoundReference]:
    """Find every PMID and DOI reference in a block of text.

    Identifiers are de-duplicated while preserving first-appearance order, and
    each result records how many times the identifier occurs.

    ``count`` is a count of textual mentions, not of distinct citations. A
    markdown link such as ``[PMID:1](https://pubmed.ncbi.nlm.nih.gov/1)`` spells
    the identifier twice and counts twice, which is why the rendered report says
    "mentions" rather than "cited".

    Args:
        text: Markdown or plain text to scan.

    Returns:
        Ordered list of unique references found.

    Examples:
        >>> refs = find_reference_ids("Shown (PMID:7913883) and again in PMID: 7913883.")
        >>> [(r.normalized_id, r.count) for r in refs]
        [('PMID:7913883', 2)]
        >>> [r.normalized_id for r in find_reference_ids("See DOI:10.1038/ng1234.")]
        ['DOI:10.1038/ng1234']
        >>> [r.normalized_id for r in find_reference_ids("https://pubmed.ncbi.nlm.nih.gov/12345678")]
        ['PMID:12345678']
        >>> find_reference_ids("no references here")
        []
    """
    found: dict[str, FoundReference] = {}

    def _add(normalized_id: str, raw: str, url: Optional[str] = None) -> None:
        existing = found.get(normalized_id)
        if existing is None:
            found[normalized_id] = FoundReference(
                normalized_id=normalized_id, raw=raw, count=1, url=url
            )
        else:
            found[normalized_id] = FoundReference(
                normalized_id=existing.normalized_id,
                raw=existing.raw,
                count=existing.count + 1,
                url=existing.url or url,
            )

    for match in _PMID_PATTERN.finditer(text):
        _add(f"PMID:{match.group(1)}", match.group(0))

    for match in _URL_PATTERN.finditer(text):
        _add(f"PMID:{match.group(1)}", match.group(0), url=match.group(0))

    for match in _DOI_PATTERN.finditer(text):
        _add(f"DOI:{normalize_doi(match.group(1))}", match.group(0))

    return list(found.values())


def extract_references(
    markdown: str,
    citations: Optional[Iterable[str]] = None,
) -> list[FoundReference]:
    """Extract references from a report body and its citation list.

    Args:
        markdown: The report body.
        citations: Optional citation strings (e.g. ``ResearchResult.citations``)
            scanned in addition to the body.

    Returns:
        Ordered list of unique references, with occurrence counts summed across
        the body and the citation list.

    Examples:
        >>> refs = extract_references("Body cites PMID:7913883.", ["Paper. PMID:99999999"])
        >>> [r.normalized_id for r in refs]
        ['PMID:7913883', 'PMID:99999999']
        >>> extract_references("")
        []
    """
    combined = markdown
    if citations:
        combined = "\n".join([markdown, *citations])
    return find_reference_ids(combined)


def extract_quoted_claims(
    markdown: str,
    pattern: Optional[re.Pattern[str]] = None,
) -> list[QuotedClaim]:
    """Extract quoted spans that are directly attributed to a reference.

    Only quotes followed by a parenthesised or bracketed citation are returned,
    because those are the ones whose wording can be checked against the source.

    Args:
        markdown: The report body.
        pattern: Optional override pattern. It must expose the quote as capture
            group 1 and the citation as capture group 2.

    Returns:
        List of quote/reference pairs, in document order. A quote attributed to
        several references yields one entry per reference.

    Examples:
        >>> claims = extract_quoted_claims(
        ...     'The authors report "widgets are blue in most cases" (PMID:12345678).'
        ... )
        >>> claims[0].quote
        'widgets are blue in most cases'
        >>> claims[0].reference_id
        'PMID:12345678'
        >>> extract_quoted_claims('A short "quote" (PMID:12345678).')
        []
        >>> extract_quoted_claims('"an unattributed quotation of ample length"')
        []
    """
    regex = pattern or _QUOTED_CLAIM_PATTERN
    claims: list[QuotedClaim] = []

    for match in regex.finditer(markdown):
        quote = match.group(1).strip()
        for reference in find_reference_ids(match.group(2)):
            claims.append(QuotedClaim(quote=quote, reference_id=reference.normalized_id))

    return claims


@dataclass(frozen=True)
class ExtractedEvidence:
    """Everything extractable from a report that reference validation acts on.

    Attributes:
        references: Unique reference identifiers cited by the report.
        quoted_claims: Quotes attributed to a specific reference.

    Examples:
        >>> ExtractedEvidence().references
        []
    """

    references: list[FoundReference] = field(default_factory=list)
    quoted_claims: list[QuotedClaim] = field(default_factory=list)


def extract_evidence(
    markdown: str,
    citations: Optional[Iterable[str]] = None,
    quote_pattern: Optional[re.Pattern[str]] = None,
) -> ExtractedEvidence:
    """Extract both references and quoted claims from a report.

    Args:
        markdown: The report body.
        citations: Optional citation strings scanned for identifiers.
        quote_pattern: Optional override for the quoted-claim pattern.

    Returns:
        The extracted references and quoted claims.

    Examples:
        >>> evidence = extract_evidence(
        ...     'They note "widgets are blue in most cases" (PMID:12345678).'
        ... )
        >>> [r.normalized_id for r in evidence.references]
        ['PMID:12345678']
        >>> len(evidence.quoted_claims)
        1
    """
    return ExtractedEvidence(
        references=extract_references(markdown, citations),
        quoted_claims=extract_quoted_claims(markdown, quote_pattern),
    )
