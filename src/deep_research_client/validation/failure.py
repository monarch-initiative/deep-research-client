"""A record that a requested validation did not complete, for the saved report.

When a validator raises, the run still exits 3, but stderr and the exit code are
gone once the report is committed or handed on. Without a trace in the file, a
report whose validation failed reads exactly like one that never asked for it,
and a curator told to read ``needs_review`` first finds no such key (#124).

:class:`ValidationFailure` stands in for the report the validator would have
produced. It has the same ``summary()`` and ``to_markdown()`` the formatter calls
on a real report, so the failure lands where the result would have: a
frontmatter block under the usual key, and a section under the usual heading.
That also means re-validating the file later replaces both, through the same
helpers that replace a stale result.
"""

import re
import shlex
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel

from .sections import TERM_VALIDATION_SECTION_HEADING, VALIDATION_SECTION_HEADING

__all__ = ["ValidationFailure", "redact_url_secrets"]

ValidationKind = Literal["reference", "term"]

_HEADINGS: Dict[str, str] = {
    "reference": VALIDATION_SECTION_HEADING,
    "term": TERM_VALIDATION_SECTION_HEADING,
}
_LABELS: Dict[str, str] = {"reference": "Reference validation", "term": "Term validation"}
_CHECKED: Dict[str, str] = {
    "reference": "The identifiers and quotes this report cites have not been checked.",
    "term": "The ontology terms this report cites have not been checked.",
}
_RETRY_COMMAND: Dict[str, str] = {
    "reference": "validate-references",
    "term": "validate-terms",
}
#: Stands in for the report's path when the run printed it rather than saving it.
_REPORT_PLACEHOLDER = "REPORT.md"
#: What the retry must repeat from the original run to check the same way: a
#: different cache directory, or a missing --trust-cache or --ontology-config,
#: gives a different result than the run that failed would have.
_RETRY_OPTIONS: Dict[str, str] = {
    "reference": "the cache options the run used",
    "term": "the cache and ontology options the run used",
}

#: A query string in an error message: ``?`` followed by at least one
#: ``key=value``, running to the next whitespace. HTTP client errors quote the
#: request URL, and a query can carry a contact email or an API key; stderr is
#: not kept, but the report is. It runs to whitespace rather than stopping at
#: punctuation because a query may contain ``(``, ``)`` or ``'`` unencoded, and
#: stopping there would leave the rest of it, key included, in the file.
_QUERY_STRING_RE = re.compile(r"\?\S*=\S*")

#: Closing punctuation that may follow a quoted URL rather than belong to it,
#: given back after redaction so the surrounding message still reads correctly.
_TRAILING_DELIMITERS = "'\")]>,.;"


#: Login details written into a URL, ``scheme://user:password@host``. Matched
#: up to the last ``@`` before the host, so a password containing ``@`` is
#: removed whole.
_USERINFO_RE = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/?#]*@")


def _redact_query(match: "re.Match[str]") -> str:
    """The replacement for one query string, keeping any closing punctuation."""
    query = match.group()
    kept = query[len(query.rstrip(_TRAILING_DELIMITERS)):]
    return "?<query removed>" + kept


def redact_url_secrets(text: str) -> str:
    """Remove the parts of any quoted URL that can carry a secret.

    That is the query string, which can carry a contact email or an API key,
    and login details written into the URL (``user:password@``). Nothing else
    is changed: a local file path in an error is kept, since it says where the
    failure was.

    Args:
        text: An error message, possibly quoting request URLs.

    Returns:
        The message with each query string and each URL login replaced.

    Examples:
        >>> redact_url_secrets(
        ...     "503 Server Error for url: https://example.org/api?apikey=s3cret&q=GO"
        ... )
        '503 Server Error for url: https://example.org/api?<query removed>'
        >>> redact_url_secrets("Max retries exceeded with url: /terms?id=GO:1 (Caused by x)")
        'Max retries exceeded with url: /terms?<query removed> (Caused by x)'

        Punctuation inside the query does not end it early:

        >>> redact_url_secrets("url: https://x.org/s?q=label(GO)&apikey=s3cret")
        'url: https://x.org/s?<query removed>'
        >>> redact_url_secrets("for url 'https://x.org/s?q=it's&key=k'.")
        "for url 'https://x.org/s?<query removed>'."

        A question mark in prose is left alone:

        >>> redact_url_secrets("Is this a question? No = answer here.")
        'Is this a question? No = answer here.'

        Login details in a URL are removed, host kept; an email in prose is not
        a URL login and is left as it is:

        >>> redact_url_secrets("could not reach https://curator:p@ss@sparql.example.org/q")
        'could not reach https://<login removed>@sparql.example.org/q'
        >>> redact_url_secrets("contact someone@example.org")
        'contact someone@example.org'
    """
    text = _USERINFO_RE.sub(r"\1<login removed>@", text)
    return _QUERY_STRING_RE.sub(_redact_query, text)


class ValidationFailure(BaseModel):
    """A requested validation that raised instead of producing a report.

    Attributes:
        kind: Which validator failed: ``"reference"`` or ``"term"``.
        error: What went wrong, as one line, with its label as prefix, the
            same text the run printed to stderr.
        report_path: Where the report was saved, so the retry command can be
            copied as it is. None when the report was printed instead.

    Examples:
        >>> failure = ValidationFailure.from_exception("reference", TimeoutError("timed out"))
        >>> failure.error
        'Reference validation failed: timed out'
        >>> failure.summary()
        {'status': 'failed', 'error': 'Reference validation failed: timed out', 'needs_review': True}

        An exception with no message is named by its type rather than left blank:

        >>> ValidationFailure.from_exception("term", TimeoutError()).error
        'Term validation failed: TimeoutError'
    """

    kind: ValidationKind
    error: str
    report_path: Optional[str] = None

    @classmethod
    def from_exception(
        cls,
        kind: ValidationKind,
        exc: BaseException,
        report_path: Optional[str] = None,
    ) -> "ValidationFailure":
        """Record ``exc`` as the reason the ``kind`` validator did not complete.

        The message is collapsed onto one line, which keeps it from being read
        as a heading when the section is split off again, and query strings and
        login details are removed from any URL it quotes.

        Args:
            kind: Which validator failed.
            exc: What it raised.
            report_path: Where the report was saved, if it was.

        Returns:
            The failure record.

        Examples:
            >>> ValidationFailure.from_exception(
            ...     "term", OSError("line one\\n## line two")
            ... ).error
            'Term validation failed: line one ## line two'
        """
        detail = " ".join(str(exc).split()) or type(exc).__name__
        return cls(
            kind=kind,
            error=f"{_LABELS[kind]} failed: {redact_url_secrets(detail)}",
            report_path=report_path,
        )

    @property
    def heading(self) -> str:
        """The heading this validator's section is written under.

        Examples:
            >>> ValidationFailure(kind="term", error="x").heading
            '## Term Validation'
        """
        return _HEADINGS[self.kind]

    def summary(self) -> Dict[str, Any]:
        """The frontmatter block recording the failure.

        No counts and no ``confabulation_rate``: there are none, and a ``0.0``
        would read as a clean result. ``needs_review`` is set because the docs
        tell readers that its absence means nothing needed attention.

        Examples:
            >>> ValidationFailure(kind="term", error="Term validation failed: x").summary()
            {'status': 'failed', 'error': 'Term validation failed: x', 'needs_review': True}
        """
        return {"status": "failed", "error": self.error, "needs_review": True}

    def to_markdown(self) -> str:
        """The section recording the failure, under the validator's usual heading.

        Examples:
            >>> print(ValidationFailure(kind="reference", error="Reference validation failed: x").to_markdown())
            ## Reference Validation
            <BLANKLINE>
            **Not completed.** Reference validation was requested but did not complete. The identifiers and quotes this report cites have not been checked.
            <BLANKLINE>
            Reference validation failed: x
            <BLANKLINE>
            To check them now: `deep-research-client validate-references REPORT.md --in-place`, with the cache options the run used.
            <BLANKLINE>

            With the saved path, the command can be copied as it is:

            >>> failure = ValidationFailure(
            ...     kind="term", error="Term validation failed: x", report_path="out/my report.md"
            ... )
            >>> failure.to_markdown().splitlines()[-1]
            "To check them now: `deep-research-client validate-terms 'out/my report.md' --in-place`, with the cache and ontology options the run used."
        """
        label = _LABELS[self.kind]
        report = shlex.quote(self.report_path) if self.report_path else _REPORT_PLACEHOLDER
        retry = f"deep-research-client {_RETRY_COMMAND[self.kind]} {report} --in-place"
        return (
            f"{self.heading}\n\n"
            f"**Not completed.** {label} was requested but did not complete. "
            f"{_CHECKED[self.kind]}\n\n"
            f"{self.error}\n\n"
            f"To check them now: `{retry}`, with {_RETRY_OPTIONS[self.kind]}.\n"
        )
