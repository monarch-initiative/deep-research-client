"""The validation sections written into a report, and how to take them off again.

Both kinds of validation append a section to the report they checked, and both
have to remove whatever an earlier run left before checking again: re-validating
an annotated report would otherwise re-extract the identifiers its own section
lists, re-fetch the flagged ones, inflate the counts, and - for reference
validation - feed the validator's prose about fabrication into the report's
keywords.

Removing is only half of it. A command that strips every generated section and
then writes back only the one it produced deletes the other, and leaves that
other section's frontmatter summary behind describing a section no longer in the
file. So a caller that writes its result back to disk splits the report with
:func:`split_validation_sections` and reassembles it with
:func:`render_with_sections`, which puts the untouched section back where it was.

The heading constants live here rather than in either model module because each
needs to know about the other's section. A report carrying both is stripped of
both, in whatever order they were written.
"""

import re
from typing import Iterable, List, Optional, Tuple

__all__ = [
    "GENERATED_SECTION_HEADINGS",
    "TERM_VALIDATION_SECTION_HEADING",
    "VALIDATION_SECTION_HEADING",
    "render_with_sections",
    "split_validation_sections",
    "strip_validation_section",
]

VALIDATION_SECTION_HEADING = "## Reference Validation"
TERM_VALIDATION_SECTION_HEADING = "## Term Validation"

#: Every heading this package appends to a report, and therefore every heading
#: that may be removed from the end of one.
GENERATED_SECTION_HEADINGS = frozenset(
    {VALIDATION_SECTION_HEADING, TERM_VALIDATION_SECTION_HEADING}
)

_H2_HEADING_RE = re.compile(r"^##[ \t]+\S.*$", re.MULTILINE)


def split_validation_sections(
    markdown: str,
    headings: Optional[Iterable[str]] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """Separate a report from the validation sections appended to it.

    Only *trailing* sections are removed: a generated section is always appended
    last and contains no further level-two headings, so it is safe to strip
    exactly while the final ``##`` heading in the document is a generated one. A
    report that discusses reference validation in its body and then continues
    with another section keeps everything, which matters because the caller
    writes this result back over the file.

    The one case it cannot see through is a validation heading inside a fenced
    code block that happens to be the last ``##`` in the file. Recognising that
    would mean parsing markdown rather than scanning it.

    Args:
        markdown: Report text, possibly ending in one or more validation
            sections.
        headings: Headings to treat as generated. Defaults to every heading this
            package writes, so a report carrying both a reference and a term
            section is split from both.

    Returns:
        The report body, with trailing blank lines normalised to a single
        newline, and its trailing generated sections in document order as
        ``(heading, text)`` pairs.

    Examples:
        >>> body, sections = split_validation_sections(
        ...     "# Report\\n\\nBody.\\n\\n## Reference Validation\\n\\nRefs.\\n"
        ...     "\\n## Term Validation\\n\\nTerms.\\n"
        ... )
        >>> body
        '# Report\\n\\nBody.\\n'
        >>> [heading for heading, _ in sections]
        ['## Reference Validation', '## Term Validation']
        >>> sections[0][1]
        '## Reference Validation\\n\\nRefs.'

        A report with nothing appended comes back whole:

        >>> split_validation_sections("# Report\\n\\nBody.\\n")
        ('# Report\\n\\nBody.\\n', [])

        A section is cut at its heading, not wherever the heading's text last
        occurs, so prose mentioning it does not split the section in two:

        >>> _, sections = split_validation_sections(
        ...     "# Report\\n\\nBody.\\n\\n## Term Validation\\n"
        ...     "\\nSuperseded; see ## Term Validation in the archive.\\n"
        ... )
        >>> len(sections)
        1
    """
    generated = set(headings) if headings is not None else set(GENERATED_SECTION_HEADINGS)
    text = markdown
    sections: List[Tuple[str, str]] = []
    while True:
        # The match's own offset, not a search for its text. Searching would find
        # the last place the heading's characters occur, which is not necessarily
        # where the heading is: a section whose prose mentions "## Term
        # Validation" mid-sentence would be cut there instead, splitting it in
        # two and writing both halves back over the file.
        matches = list(_H2_HEADING_RE.finditer(text))
        if not matches or matches[-1].group().strip() not in generated:
            break
        # Repeat, so a file left with stacked sections by an older run is cleaned
        # up rather than losing only the last of them.
        start = matches[-1].start()
        sections.append((matches[-1].group().strip(), text[start:].rstrip()))
        text = text[:start]
    sections.reverse()
    body = text.rstrip() + "\n" if text.strip() else ""
    return body, sections


def render_with_sections(
    body: str,
    sections: Iterable[Tuple[str, str]],
    heading: str,
    rendered: str,
) -> str:
    """Reassemble a report, putting one freshly written section back in place.

    A section that was already there is replaced where it stood, so re-running
    one command neither moves nor deletes the other command's section. A section
    that was not there is appended. Stacked duplicates of one heading, which an
    older run could leave behind, collapse into the single fresh one.

    Args:
        body: The report without its generated sections.
        sections: The sections that were removed, in document order.
        heading: Heading of the section being written.
        rendered: The freshly rendered section, including its heading.

    Returns:
        The full report text.

    Examples:
        A re-run of term validation leaves the reference section alone, and in
        front of it, where it was:

        >>> body, sections = split_validation_sections(
        ...     "# Report\\n\\nBody.\\n\\n## Reference Validation\\n\\nRefs.\\n"
        ...     "\\n## Term Validation\\n\\nOld terms.\\n"
        ... )
        >>> render_with_sections(
        ...     body, sections, "## Term Validation", "## Term Validation\\n\\nNew terms."
        ... )
        '# Report\\n\\nBody.\\n\\n## Reference Validation\\n\\nRefs.\\n\\n## Term Validation\\n\\nNew terms.\\n'

        A section the report did not carry is appended:

        >>> body, sections = split_validation_sections("# Report\\n\\nBody.\\n")
        >>> render_with_sections(
        ...     body, sections, "## Term Validation", "## Term Validation\\n\\nTerms."
        ... )
        '# Report\\n\\nBody.\\n\\n## Term Validation\\n\\nTerms.\\n'
    """
    ordered = list(sections)
    kept = [(name, text) for name, text in ordered if name != heading]
    position = next(
        (index for index, (name, _) in enumerate(ordered) if name == heading), None
    )
    if position is None:
        kept.append((heading, rendered))
    else:
        insert_at = sum(1 for name, _ in ordered[:position] if name != heading)
        kept.insert(insert_at, (heading, rendered))

    parts = [body.rstrip()] + [text.rstrip() for _, text in kept]
    return "\n\n".join(part for part in parts if part) + "\n"


def strip_validation_section(
    markdown: str,
    headings: Optional[Iterable[str]] = None,
) -> str:
    """Remove previously written validation sections from a report.

    For a caller that only needs to read the report - extracting identifiers,
    scoring keywords - the sections are simply in the way. A caller that writes
    the result back to disk wants :func:`split_validation_sections` instead, so
    that the section it is not rewriting survives.

    Args:
        markdown: Report text, possibly ending in one or more validation
            sections.
        headings: Headings to treat as generated. Defaults to every heading this
            package writes.

    Returns:
        The report without its trailing validation sections, with trailing blank
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

        Both kinds of section come off, in whichever order they were appended:

        >>> strip_validation_section(
        ...     "# Report\\n\\nBody.\\n\\n## Reference Validation\\n\\nRefs.\\n"
        ...     "\\n## Term Validation\\n\\nTerms.\\n"
        ... )
        '# Report\\n\\nBody.\\n'

        A validation heading that is not the last section is left alone, along
        with everything after it:

        >>> strip_validation_section(
        ...     "# Report\\n\\n## Reference Validation\\n\\nWe discuss it.\\n"
        ...     "\\n## Conclusions\\n\\nImportant text.\\n"
        ... )
        '# Report\\n\\n## Reference Validation\\n\\nWe discuss it.\\n\\n## Conclusions\\n\\nImportant text.\\n'
    """
    return split_validation_sections(markdown, headings)[0]
