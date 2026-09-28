"""Split a source into the units an extractor reads, each with its provenance.

A unit is a stretch of text small enough for one extraction request, together
with where it is: offsets into the source text, the heading path above it, and
for prose inside a structured document, the path of the field.
"""

import re
from typing import Any, Iterator, Mapping, NamedTuple, Optional

from .models import CitationHandle
from .parsing import TextUnit, section_citation

__all__ = ["markdown_units", "read_citations_file", "report_title", "structured_units"]

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

#: Words that make a heading a reference list when the list under it is one,
#: as in "Key references (URLs in evidence)". The content check matters:
#: "Evidence sources (patient-level vs aggregated)" is prose, not a list.
_REFERENCE_WORDS = re.compile(r"\b(?:references|sources|bibliography|citations|works cited)\b", re.IGNORECASE)

#: A line that names a reference by identifier or link.
_IDENTIFIED = re.compile(r"https?://|\bdoi\b|\bPMID\b|\bPMC\d", re.IGNORECASE)

#: A list item or numbered entry.
_ENTRY_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+|\[\d+\]:?\s+)")


def _is_reference_list(title: str, own_text: str) -> bool:
    """Whether a heading and the text under it are a reference list.

    A heading named as one ("References", "Sources"...) always is. A heading
    that only contains such a word is one when at least half its lines are
    entries and at least half of those name a URL, DOI or PMID.

    >>> _is_reference_list("Sources", "Anything.")
    True
    >>> refs = "- Mustillo 2023. https://doi.org/10.1/x\\n- Biggs 2023. https://doi.org/10.1/y\\n"
    >>> _is_reference_list("Key references (URLs in evidence)", refs)
    True
    >>> _is_reference_list("1.4 Evidence sources (patient-level vs aggregated)", "Most evidence is aggregated.")
    False
    """
    if title.rstrip(":").strip().lower() in _REFERENCE_SECTIONS:
        return True
    if not _REFERENCE_WORDS.search(title):
        return False
    lines = [line for line in own_text.splitlines() if line.strip()]
    entries = [line for line in lines if _ENTRY_LINE.match(line)]
    identified = [line for line in entries if _IDENTIFIED.search(line)]
    return bool(lines) and 2 * len(entries) >= len(lines) and 2 * len(identified) >= len(entries)


class _OpenHeading(NamedTuple):
    """A heading whose section the loop is inside."""

    level: int
    title: str
    citation: Optional[CitationHandle]
    references: bool


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


#: The last header line of a citations file this client writes with
#: --separate-citations. The query above it can hold numbered lines of its own.
_CITATIONS_FILE_HEADER_END = re.compile(r"^\*\*Generated:\*\*.*$", re.MULTILINE)


def read_citations_file(text: str) -> dict[int, str]:
    """The numbered citations in a file written by ``research --separate-citations``.

    That file repeats the query, which can hold numbered lines of its own (a
    template's objectives), so only the entries after its ``**Generated:**``
    line are read. A file without that header is read whole.

    Args:
        text: The citations file.

    Returns:
        Entry text by reference number.

    >>> side = "# Citations for Research Query\\n\\n**Query:** Find:\\n1. Genes\\n**Provider:** perplexity\\n"
    >>> side += "**Generated:** 2026-02-03\\n\\n1. https://pmc.ncbi.nlm.nih.gov/articles/PMC4900471/\\n"
    >>> read_citations_file(side)
    {1: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4900471/'}
    """
    headers = list(_CITATIONS_FILE_HEADER_END.finditer(text))
    body = text[headers[-1].end():] if headers else text
    return {int(m.group(1) or m.group(2)): m.group(3) for m in _NUMBERED_ENTRY.finditer(body)}


def _bibliography(
    text: str, start: int, end: Optional[int] = None,
    citations_file: Optional[Mapping[int, str]] = None,
) -> dict[int, str]:
    """The numbered references ``[n]`` markers resolve through, by number.

    This client's own ``## Citations`` list, which follows the answer, comes
    first, then its separate citations file when one was given. When both are
    missing or empty, the provider's own reference list
    inside the answer (``References``, ``Sources``... at any level) is used,
    so a provider that numbers its own sources still has them resolved. If
    the answer has several, the longest is taken (the first, on a tie), and
    a "Further reading" list never is: it numbers other works.

    Args:
        text: The report.
        start: Where the answer begins.
        end: Where it ends (exclusive); the end of ``text`` if None.
        citations_file: Entries read from a separate citations file.

    Returns:
        Entry text by reference number; empty when no list exists.

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
    if citations_file:
        return dict(citations_file)
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
    r"""Split ``text[start:end]`` into pieces of at most max_chars.

    Pieces break at paragraph breaks. A piece still too long is broken
    between the rows of a table, which has no paragraph breaks: a 12 KB table
    sent whole can need a reply longer than any budget. A single paragraph
    longer than max_chars is kept whole rather than cut mid-sentence.

    >>> table = "| Gene | Role |\n|---|---|\n" + "".join(f"| G{i} | role {i} |\n" for i in range(40))
    >>> pieces = list(_split_long(0, len(table), table, 200))
    >>> len(pieces) > 1, all(b - a <= 200 for a, b in pieces), pieces[-1][1] == len(table)
    (True, True, True)
    >>> [a for a, _ in pieces[1:]] == [b for _, b in pieces[:-1]], table[pieces[1][0]:].startswith("| G")
    (True, True)
    >>> text = "One long paragraph " * 20
    >>> list(_split_long(0, len(text), text, 100))
    [(0, 380)]
    """
    for piece_start, piece_end in _pack(start, end, text, max_chars, r"\n[ \t]*\n"):
        if piece_end - piece_start <= max_chars:
            yield piece_start, piece_end
        else:
            yield from _pack(piece_start, piece_end, text, max_chars, _ROW_BREAK)


#: A line break between two table rows.
_ROW_BREAK = r"(?<=\|)[ \t]*\n(?=[ \t]*\|)"


def _pack(start: int, end: int, text: str, max_chars: int, breaks: str) -> Iterator[tuple[int, int]]:
    """Group ``text[start:end]`` into pieces of at most max_chars, cut only at ``breaks``."""
    piece_start = start
    last_break = None
    for match in re.finditer(breaks, text[start:end]):
        boundary = start + match.end()
        if boundary - piece_start > max_chars and last_break is not None and last_break > piece_start:
            yield piece_start, last_break
            piece_start = last_break
        last_break = boundary
    if end - piece_start > max_chars and last_break is not None and last_break > piece_start:
        yield piece_start, last_break
        piece_start = last_break
    yield piece_start, end


def _table_header(text: str, position: int) -> Optional[str]:
    r"""The header of the table a piece starting at ``position`` is cut from.

    Returns the table's first two lines (column names and the ``|---|``
    rule) when the line before ``position`` is a table row, else None.

    >>> table = "Intro.\n\n| Gene | Role |\n|---|---|\n| A | x |\n| B | y |\n"
    >>> _table_header(table, table.index("| B"))
    '| Gene | Role |\n|---|---|'
    >>> _table_header(table, table.index("| Gene")) is None
    True
    """
    before = text[:position].rstrip("\n").split("\n")
    if not before or not before[-1].lstrip().startswith("|"):
        return None
    first = len(before) - 1
    while first > 0 and before[first - 1].lstrip().startswith("|"):
        first -= 1
    header = before[first:first + 2]
    return "\n".join(header) if len(header) == 2 else None


def markdown_units(
    text: str, max_chars: int = DEFAULT_MAX_UNIT_CHARS,
    citations_file: Optional[Mapping[int, str]] = None,
) -> list[TextUnit]:
    """Split a markdown report into extraction units, by heading.

    Heading lines are not part of any unit; they become its section path.
    Every unit shares the report's bibliography, so ``[n]`` markers resolve.
    A reference list the provider wrote inside its answer ("## References",
    "### Sources"...) is not a unit, nor is anything under it.

    Args:
        text: The report, as read from disk; offsets index into it.
        max_chars: Largest unit; longer sections split at paragraph breaks.
        citations_file: Entries of the report's separate citations file
            (:func:`read_citations_file`), for a report written without its
            own ``## Citations`` list.

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
    bibliography = _bibliography(text, start, end, citations_file)

    units: list[TextUnit] = []
    stack: list[_OpenHeading] = []
    cursor = start
    headings = _headings(text, start, end)
    boundaries = [(h.start(), h.end(), len(h.group(1)), h.group(2).strip()) for h in headings]
    boundaries.append((end, end, 0, ""))

    for index, (heading_start, heading_end, level, title) in enumerate(boundaries):
        section = " > ".join(open_heading.title for open_heading in stack) or None
        in_references = any(open_heading.references for open_heading in stack)
        cited = next((h.citation for h in reversed(stack) if h.citation is not None), None)
        if not in_references and text[cursor:heading_start].strip():
            for piece_start, piece_end in _split_long(cursor, heading_start, text, max_chars):
                if text[piece_start:piece_end].strip():
                    units.append(TextUnit(
                        text=text, start=piece_start, end=piece_end,
                        section=section, bibliography=bibliography,
                        section_citation=cited,
                        context=_table_header(text, piece_start),
                    ))
        if level:
            # The work's own identifier lines sit under its heading, before
            # any subheading: a subsection such as "Related work" may list
            # other papers' identifiers. The citation still covers the
            # subsections, through the stack.
            own_end = boundaries[index + 1][0]
            own_text = text[heading_end:own_end]
            stack = [entry for entry in stack if entry.level < level]
            stack.append(_OpenHeading(
                level, title, section_citation(title, own_text, bibliography),
                _is_reference_list(title, own_text),
            ))
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
