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
_NUMBERED_ENTRY = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$", re.MULTILINE)

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
    """The numbered list under a heading, up to the next heading, by number."""
    following = next(iter(_headings(text, heading.end())), None)
    section = text[heading.end():following.start() if following else len(text)]
    return {int(m.group(1)): m.group(2) for m in _NUMBERED_ENTRY.finditer(section)}


def _bibliography(text: str, start: int, end: Optional[int] = None) -> dict[int, str]:
    """The numbered references ``[n]`` markers resolve through, by number.

    This client's own ``## Citations`` list, which follows the answer, comes
    first. When it is missing or empty, the provider's own reference list
    inside the answer (``References``, ``Sources``... at any level) is used,
    so a provider that numbers its own sources still has them resolved.

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
    """
    stop = len(text) if end is None else end
    ours = [
        m for m in _headings(text, start)
        if len(m.group(1)) == 2 and m.group(2).strip().lower() == "citations"
    ]
    if ours and (entries := _numbered_entries(text, ours[-1])):
        return entries
    theirs = [
        m for m in _headings(text, start, stop)
        if m.group(2).rstrip(":").strip().lower() in _REFERENCE_SECTIONS
    ]
    return _numbered_entries(text, theirs[-1]) if theirs else {}


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
    for heading in level_two:
        if heading.group(2).strip().lower() == "output":
            start = heading.end()
            break
    end = len(text)
    for heading in reversed([h for h in level_two if h.start() >= start]):
        if heading.group(2).strip().lower() not in _SKIPPED_TOP_SECTIONS:
            break
        end = heading.start()
    return start, end


def report_title(text: str) -> Optional[str]:
    """The report's first top-level heading inside its answer, if any.

    >>> report_title("## Output\\n\\n# Marfan syndrome\\n\\nText")
    'Marfan syndrome'
    """
    start, end = _answer_region(text)
    for heading in _headings(text, start, end):
        if len(heading.group(1)) == 1:
            return heading.group(2).strip()
    return None


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
