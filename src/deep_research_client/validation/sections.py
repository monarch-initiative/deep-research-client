"""The validation sections written into a report, and how to take them off again.

Both kinds of validation append a section to the report they checked, and both
have to remove whatever an earlier run left before checking again: re-validating
an annotated report would otherwise re-extract the identifiers its own section
lists, re-fetch the flagged ones, inflate the counts, and - for reference
validation - feed the validator's prose about fabrication into the report's
keywords.

The heading constants and the stripper live here rather than in either model
module because each needs to know about the other's section. A report carrying
both is stripped of both, in whatever order they were written.
"""

import re
from typing import Iterable, Optional

__all__ = [
    "GENERATED_SECTION_HEADINGS",
    "TERM_VALIDATION_SECTION_HEADING",
    "VALIDATION_SECTION_HEADING",
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


def strip_validation_section(
    markdown: str,
    headings: Optional[Iterable[str]] = None,
) -> str:
    """Remove previously written validation sections from a report.

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
            section is stripped of both.

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
    generated = set(headings) if headings is not None else set(GENERATED_SECTION_HEADINGS)
    text = markdown
    while True:
        found = _H2_HEADING_RE.findall(text)
        if not found or found[-1].strip() not in generated:
            break
        # Repeat, so a file left with stacked sections by an older run is cleaned
        # up rather than losing only the last of them.
        text = text[: text.rindex(found[-1])]
    return text.rstrip() + "\n" if text.strip() else ""
