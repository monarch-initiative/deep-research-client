"""Split a source into the units an extractor reads, each with its provenance.

A unit is a stretch of text small enough for one extraction request, together
with where it is: offsets into the source text, the heading path above it, and
for prose inside a structured document, the path of the field.
"""

import re
from typing import Any, Iterator, Optional

from .models import CitationHandle
from .parsing import TextUnit, section_citation

__all__ = ["markdown_units", "structured_units", "report_title"]

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})", re.MULTILINE)
#: A numbered reference entry: "1. ...", "[1] ...", a Markdown reference
#: definition "[1]: ...", or any of them as a list item. Footnotes ("[^1]: ...")
#: are not read: their markers ("[^1]") are not numeric markers either.
_NUMBERED_ENTRY = re.compile(
    r"^\s*(?:[-*+]\s+)?(?:\[(\d+)\]:?|(\d+)\.)\s+(.+?)\s*$", re.MULTILINE,
)

#: Sections this client writes around a provider's answer. The question is the
#: user's, not the source's; the bibliography and generated validation
#: sections are about the report rather than claims made in it.
_SKIPPED_TOP_SECTIONS = frozenset({
    "question", "citations", "artifacts", "reference validation", "term validation",
})

#: Headings of reference lists a provider writes inside its own answer. A
#: reference list names sources rather than making claims, so it and its
#: subsections are not sent to the extractor.
_REFERENCE_SECTIONS = frozenset({
    "references", "sources", "bibliography", "citations", "works cited",
    "literature cited", "further reading",
})

#: Reference lists that are not what an answer's "[n]" markers point at.
#: They are still skipped as units.
_NOT_THE_BIBLIOGRAPHY = frozenset({"further reading"})

#: Largest unit sent in one request. A section longer than this is split at
#: paragraph breaks, keeping its heading path.
DEFAULT_MAX_UNIT_CHARS = 6000


def _headings(text: str, start: int = 0, end: Optional[int] = None) -> list[re.Match[str]]:
    r"""The markdown headings in ``text[start:end]``, skipping fenced code blocks.

    A line starting with "#" inside a code fence is code (a shell comment, a
    Python comment), not a heading, and must not split a unit.

    >>> [m.group(2) for m in _headings("# A\n```\n# not a heading\n```\n## B\n")]
    ['A', 'B']
    """
    stop = len(text) if end is None else end
    fenced: list[tuple[int, int]] = []
    opening: Optional[re.Match[str]] = None
    for fence in _FENCE.finditer(text, 0, stop):
        if opening is None:
            opening = fence
        elif fence.group(1)[0] == opening.group(1)[0] and len(fence.group(1)) >= len(opening.group(1)):
            fenced.append((opening.start(), fence.end()))
            opening = None
    if opening is not None:
        fenced.append((opening.start(), stop))
    return [
        m for m in _HEADING.finditer(text, start, stop)
        if not any(a <= m.start() < b for a, b in fenced)
    ]


def _body_start(text: str) -> int:
    """Offset just past YAML frontmatter, or 0 when there is none.

    >>> _body_start("---\\na: 1\\n---\\n\\nBody")
    13
    >>> _body_start("No frontmatter")
    0
    """
    if not text.startswith("---\n"):
        return 0
    closing = text.find("\n---\n", 4)
    return 0 if closing < 0 else closing + len("\n---\n")


def _numbered_entries(text: str, heading: re.Match[str]) -> dict[int, str]:
    r"""The numbered list under a heading, up to the next heading, by number.

    >>> text = "## Refs\n\n1. Plain\n[2] Bracketed\n- [3] Bulleted\n* 4. Starred\n[5]: https://x.org/5\n"
    >>> _numbered_entries(text, _headings(text)[0])
    {1: 'Plain', 2: 'Bracketed', 3: 'Bulleted', 4: 'Starred', 5: 'https://x.org/5'}
    """
    following = next(iter(_headings(text, heading.end())), None)
    section = text[heading.end():following.start() if following else len(text)]
    return {
        int(m.group(1) or m.group(2)): m.group(3) for m in _NUMBERED_ENTRY.finditer(section)
    }


def _bibliography(text: str, start: int, end: Optional[int] = None) -> dict[int, str]:
    """The numbered references ``[n]`` markers resolve through, by number.

    This client's own ``## Citations`` list, which follows the answer, comes
    first. When it is missing or empty, the provider's own reference list
    inside the answer (``References``, ``Sources``... at any level) is used,
    so a provider that numbers its own sources still has them resolved. If
    the answer has several, the longest is taken (the first, on a tie), and
    a "Further reading" list never is: it numbers other works.

    Args:
        text: The report.
        start: Where the answer begins.
        end: Where it ends (exclusive); the end of ``text`` if None.

    Returns:
        Entry text by reference number; empty when neither list exists.

    >>> own = "# Title\\n\\nFBN1 [1].\\n\\n### References\\n\\n1. Dietz HC. PMID: 1852208\\n"
    >>> _bibliography(own, 0)
    {1: 'Dietz HC. PMID: 1852208'}
    >>> both = "## Output\\n\\nA [1].\\n\\n## Sources\\n\\n1. Theirs\\n\\n## Citations\\n\\n1. Ours\\n"
    >>> _bibliography(both, 0, both.index("## Citations"))
    {1: 'Ours'}
    >>> extra = "# T\\n\\nA [2].\\n\\n### References\\n\\n1. R1\\n2. R2\\n\\n### Further reading\\n\\n1. F1\\n2. F2\\n"
    >>> _bibliography(extra, 0)
    {1: 'R1', 2: 'R2'}
    """
    stop = len(text) if end is None else end
    ours = [
        m for m in _headings(text, start)
        if len(m.group(1)) == 2 and m.group(2).strip().lower() == "citations"
    ]
    if ours and (entries := _numbered_entries(text, ours[-1])):
        return entries
    theirs = [
        _numbered_entries(text, m) for m in _headings(text, start, stop)
        if (name := m.group(2).rstrip(":").strip().lower()) in _REFERENCE_SECTIONS
        and name not in _NOT_THE_BIBLIOGRAPHY
    ]
    return max(theirs, key=len, default={})


def _answer_region(text: str) -> tuple[int, int]:
    """Where the provider's answer is, skipping the sections this client adds.

    With a ``## Output`` heading (this client's report layout) the answer runs
    from there to the first of the trailing sections this client writes. A
    provider may use level-2 headings of its own, so trailing sections are
    found from the end: the longest run of known ones closing the document.
    Without ``## Output``, the whole body is the answer, minus the same tail.

    Args:
        text: The report.

    Returns:
        ``(start, end)`` offsets of the answer.
    """
    start = _body_start(text)
    level_two = [m for m in _headings(text, start) if len(m.group(1)) == 2]
    question = next((h for h in level_two if h.group(2).strip().lower() == "question"), None)
    for heading in level_two:
        if heading.group(2).strip().lower() == "output":
            if question is not None and question.start() < heading.start():
                start = _past_echoed_question(
                    text[question.end():heading.start()], text, heading.end(),
                )
            else:
                start = heading.end()
            break
    end = len(text)
    for heading in reversed([h for h in level_two if h.start() >= start]):
        if heading.group(2).strip().lower() not in _SKIPPED_TOP_SECTIONS:
            break
        end = heading.start()
    return start, end


#: How many of the question's closing lines must be found, in order, for the
#: answer to count as repeating the question.
_ECHO_TAIL_LINES = 3


def _past_echoed_question(question: str, text: str, answer_start: int) -> int:
    r"""Where the answer begins, past any repetition of the question.

    Some providers (Falcon) open their answer by repeating the whole prompt:
    a wrapper of their own, then the question word for word. That is the
    question, not the answer, and its template would otherwise be decomposed
    into claims. The answer counts as repeating the question only when it
    holds the question's first line and, after it, the question's closing
    lines in order; a report quoting one line of its question keeps it.

    Args:
        question: The text of this client's ``## Question`` section.
        text: The report.
        answer_start: Where the ``## Output`` section's text begins.

    Returns:
        The offset just past the repeated question, or ``answer_start``.

    >>> q = "\n# Template\n\n- Find the gene.\n- Find the drug.\n- Cite PMIDs.\n"
    >>> report = f"## Question\n{q}## Output\n\nQuestion: be expert.\n{q}\n## Report\n\nTBX1 matters.\n"
    >>> start, end = _answer_region(report)
    >>> report[start:].strip()
    '## Report\n\nTBX1 matters.'
    >>> quoting = f"## Question\n{q}## Output\n\n# Template\n\nTBX1 matters.\n"
    >>> _answer_region(quoting)[0] == quoting.index("## Output") + len("## Output")
    True
    """
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    if len(lines) <= _ECHO_TAIL_LINES:
        return answer_start
    first = text.find(lines[0], answer_start)
    if first < 0:
        return answer_start
    position = first + len(lines[0])
    for line in lines[-_ECHO_TAIL_LINES:]:
        found = text.find(line, position)
        if found < 0:
            return answer_start
        position = found + len(line)
    return position


def report_title(text: str) -> Optional[str]:
    """The report's first heading at its highest level inside its answer, if any.

    A report titled with ``#`` gives that; one whose sections start at ``##``
    (as Falcon's do) gives its first ``##`` heading.

    >>> report_title("## Output\\n\\n# Marfan syndrome\\n\\n## Genetics\\n\\nText")
    'Marfan syndrome'
    >>> report_title("## Output\\n\\n## Report: 22q11.2DS\\n\\n### Summary\\n\\nText")
    'Report: 22q11.2DS'
    """
    start, end = _answer_region(text)
    headings = _headings(text, start, end)
    if not headings:
        return None
    top = min(len(h.group(1)) for h in headings)
    return next(h.group(2).strip() for h in headings if len(h.group(1)) == top)


def _split_long(start: int, end: int, text: str, max_chars: int) -> Iterator[tuple[int, int]]:
    """Split ``text[start:end]`` at paragraph breaks into pieces of at most max_chars.

    A single paragraph longer than max_chars is kept whole rather than cut
    mid-sentence.
    """
    piece_start = start
    last_break = None
    for match in re.finditer(r"\n[ \t]*\n", text[start:end]):
        boundary = start + match.end()
        if boundary - piece_start > max_chars and last_break is not None:
            yield piece_start, last_break
            piece_start = last_break
        last_break = boundary
    if end - piece_start > max_chars and last_break is not None and last_break > piece_start:
        yield piece_start, last_break
        piece_start = last_break
    yield piece_start, end


def markdown_units(text: str, max_chars: int = DEFAULT_MAX_UNIT_CHARS) -> list[TextUnit]:
    """Split a markdown report into extraction units, by heading.

    Heading lines are not part of any unit; they become its section path.
    Every unit shares the report's bibliography, so ``[n]`` markers resolve.
    A reference list the provider wrote inside its answer ("## References",
    "### Sources"...) is not a unit, nor is anything under it.

    Args:
        text: The report, as read from disk; offsets index into it.
        max_chars: Largest unit; longer sections split at paragraph breaks.

    Returns:
        Units in document order, with blank ones dropped.

    >>> report = "## Output\\n\\n# Title\\n\\nIntro.\\n\\n## Genetics\\n\\nFBN1 [1].\\n\\n## Citations\\n\\n1. Ref\\n"
    >>> [(u.section, u.body.strip()) for u in markdown_units(report)]
    [('Title', 'Intro.'), ('Title > Genetics', 'FBN1 [1].')]
    >>> markdown_units(report)[1].bibliography
    {1: 'Ref'}
    >>> own = "# Title\\n\\nFBN1 [1].\\n\\n## References\\n\\n1. Dietz HC. Nature. 1991.\\n"
    >>> [u.section for u in markdown_units(own)]
    ['Title']

    A section headed by a cited work carries that work as its citation, and
    so do its subsections:

    >>> listing = "# Papers\\n\\n### [1] A study\\n\\n- PMID: 41258631\\n\\n#### Methods\\n\\nMice.\\n\\n### Notes\\n\\nX.\\n"
    >>> [(u.section, u.section_citation and u.section_citation.reference_id) for u in markdown_units(listing)]
    [('Papers > [1] A study', 'PMID:41258631'), ('Papers > [1] A study > Methods', 'PMID:41258631'), ('Papers > Notes', None)]
    """
    start, end = _answer_region(text)
    bibliography = _bibliography(text, start, end)

    units: list[TextUnit] = []
    # Each open heading: its level, its title, and the cited work it names.
    stack: list[tuple[int, str, Optional[CitationHandle]]] = []
    cursor = start
    headings = _headings(text, start, end)
    boundaries = [(h.start(), h.end(), len(h.group(1)), h.group(2).strip()) for h in headings]
    boundaries.append((end, end, 0, ""))

    for index, (heading_start, heading_end, level, title) in enumerate(boundaries):
        section = " > ".join(name for _, name, _ in stack) or None
        in_references = any(
            name.rstrip(":").strip().lower() in _REFERENCE_SECTIONS for _, name, _ in stack
        )
        cited = next((c for _, _, c in reversed(stack) if c is not None), None)
        if not in_references and text[cursor:heading_start].strip():
            for piece_start, piece_end in _split_long(cursor, heading_start, text, max_chars):
                if text[piece_start:piece_end].strip():
                    units.append(TextUnit(
                        text=text, start=piece_start, end=piece_end,
                        section=section, bibliography=bibliography,
                        section_citation=cited,
                    ))
        if level:
            # The work's own identifier lines sit under its heading, before
            # any subheading: a subsection such as "Related work" may list
            # other papers' identifiers. The citation still covers the
            # subsections, through the stack.
            own_end = boundaries[index + 1][0]
            stack = [entry for entry in stack if entry[0] < level]
            stack.append((level, title, section_citation(
                title, text[heading_end:own_end], bibliography,
            )))
        cursor = heading_end
    return units


def structured_units(data: Any, min_words: int = 6) -> list[TextUnit]:
    """The prose fields of a parsed YAML or JSON document, one unit each.

    Short strings -- identifiers, labels, enum values -- are not prose and are
    skipped. Each unit's text is the field's own value, so its offsets index
    into that value, and ``source_path`` says which field it is.

    Args:
        data: The parsed document.
        min_words: Fewest words for a string to count as prose.

    Returns:
        One unit per prose field, in document order.

    >>> doc = {"name": "X", "notes": [{"text": "Loss of FBN1 weakens the aortic wall over time."}]}
    >>> [(u.source_path, u.end) for u in structured_units(doc)]
    [('notes[0].text', 47)]
    """
    units: list[TextUnit] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, str) and len(value.split()) >= min_words:
            units.append(TextUnit(text=value, start=0, end=len(value), source_path=path))

    walk(data, "")
    return units
