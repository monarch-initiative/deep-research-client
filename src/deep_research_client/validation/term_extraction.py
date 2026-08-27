"""Extraction of ontology term CURIEs, and the labels reports write beside them.

Like :mod:`deep_research_client.validation.extraction`, this module is
deliberately dependency-free (standard library only) so that it can be used
without installing the optional ``terms`` extra. Resolving the extracted CURIEs
against an ontology lives in
:mod:`deep_research_client.validation.term_validator`.

Two things are extracted, and the second is the hard one. Finding ``HP:0001250``
in a report is a matter of shape. Deciding that the report meant it to be called
"Seizure" means deciding which nearby words are a label and which are prose, and
getting that wrong produces a confident accusation about a term that was cited
correctly. So labels are only read out of positions where a label is the only
thing the text can reasonably be - a table cell, an emphasised run, a bracket or
a colon immediately following the CURIE - and a term mentioned in flowing prose
is left with no reported label at all. That undercounts rather than invents, and
the term is still checked for existence.

What this deliberately does not see:

* A term named in flowing prose, as above. "Patients with aortic root dilation
  (HP:0002616) were followed" yields no label, because the run before the
  bracket is a clause rather than a name.
* A sentence continuing straight off a separator with neither bracket nor comma
  to end it - "HP:0001250 - Seizure was observed in three unrelated probands"
  reads the whole run as the label. A word cap would catch it, and would also
  truncate real labels, several of which run past eight words, so the trade is
  not worth taking. A finite-verb marker would be the way in if this turns up in
  practice.
* A comma-led aside opening with a function word, on a term whose own label is
  short: "HP:0001250 - Seizure, with onset in infancy" is read whole and scores
  as a mismatch against "Seizure" - synonyms do not rescue it, since the aside
  is no closer to any of them. Cutting it would mean cutting
  "Deafness, autosomal recessive, with or without vestibular dysfunction", which
  is a real name of the same shape, so there is no discriminator here that is
  not a guess. The exposure is bounded by the *canonical* label's length rather
  than the aside's: the same aside on "Marfan syndrome" scores as a variant, not
  a mismatch, and a variant does not fail a build.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Optional

# The shape of an ontology CURIE: a prefix, a colon, a local identifier.
#
# The prefix must start uppercase and carry at least two letters. Both
# constraints are load-bearing. Starting uppercase drops URLs (`https:`) and the
# lowercase words that end sentences before a numeral; requiring two letters
# drops clock times inside timestamps, where `...T10:30` would otherwise present
# `T10` as a prefix and `30` as a local identifier.
#
# The local identifier is digits, optionally behind up to three letters, which
# covers both the OBO numbering used by HP, MONDO, GO, CHEBI and UBERON
# (HP:0001250) and the letter-led schemes used by NCIT and MeSH (NCIT:C16814,
# MESH:D008382). Two digits is the floor: it keeps ratios and section numbers out
# while admitting every real ontology identifier, whose local parts are longer.
_CURIE_PREFIX = r"[A-Z][A-Za-z0-9_.]{1,29}"
_CURIE_LOCAL = r"[A-Za-z]{0,3}\d{2,12}"
_CURIE_PATTERN = re.compile(rf"\b(?P<prefix>{_CURIE_PREFIX}):(?P<local>{_CURIE_LOCAL})\b")

# An OBO PURL, which is how ontology terms are linked when they are not written
# as CURIEs: http://purl.obolibrary.org/obo/HP_0001250.
_PURL_PATTERN = re.compile(
    r"https?://purl\.obolibrary\.org/obo/(?P<prefix>[A-Za-z][A-Za-z0-9_.]{1,29})_(?P<local>[A-Za-z]{0,3}\d{2,12})\b"
)

# Prefixes that match the CURIE shape but never name an ontology term. The
# bibliographic ones are the important entries: PMID, DOI, PMC and GEO are
# checked by reference validation, and extracting them here would report every
# citation in a report twice, once as a reference and once as a term that no
# ontology contains.
NON_ONTOLOGY_PREFIXES = frozenset(
    {
        "ARXIV",
        "BIORXIV",
        "DOI",
        "FTP",
        "GDS",
        "GEO",
        "GSE",
        "HTTP",
        "HTTPS",
        "ISBN",
        "ISO",
        "ISSN",
        "MEDRXIV",
        "PMC",
        "PMCID",
        "PMID",
        "RFC",
        "RRID",
    }
)

# A prefix at least two letters long, per the pattern above, tested separately so
# the rule can be stated once and reused when a caller supplies a CURIE directly.
_TWO_LETTERS = re.compile(r"[A-Za-z].*[A-Za-z]")

# Markdown inline link, reduced to its text. A CURIE is very often written as a
# link to bioregistry or OLS, and the link target is not part of the label.
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")

# Wrappers a label is set off with, stripped from both ends of a candidate.
_LABEL_WRAPPERS = "*_`\"'“”‘’ \t"
# Punctuation that trails or leads a candidate once its wrappers are gone.
_LABEL_EDGE_PUNCTUATION = " \t,;:.–—-"

# Words that start a parenthetical which is not a label. A report writes
# `HP:0001250 (seen in 4 of 11 probands)` as readily as it writes
# `HP:0001250 (Seizure)`, and reading the first as a label manufactures a label
# mismatch for a term that was cited correctly. Kept deliberately short: these
# are the openers that actually recur, and every word here is one that no
# ontology label begins with.
_NON_LABEL_OPENERS = frozenset(
    {
        "according",
        "adapted",
        "also",
        "and",
        "as",
        "based",
        "but",
        "cited",
        "e.g",
        "eg",
        "found",
        "from",
        "i.e",
        "ie",
        "in",
        "including",
        "n",
        "detected",
        "estimated",
        "observed",
        "or",
        "per",
        "ref",
        "reported",
        "reviewed",
        "see",
        "seen",
        "source",
        "such",
        "the",
        "via",
        "which",
        "with",
    }
)

# Words that open an aside *about* a term rather than part of its name, used to
# cut a trailing clause off a candidate that is otherwise a good label.
#
# Deliberately NOT _NON_LABEL_OPENERS, though it started as that and was wrong.
# That set answers a different question - does this candidate *begin* as prose -
# and for that job function words like "with", "or" and "in" are safe, because no
# label begins with them. Plenty of labels *continue* with them: the real MONDO
# term "microcephaly, with or without chorioretinopathy, lymphedema, or
# intellectual disability" is cut to "microcephaly" by that set, which reports a
# correctly cited term as naming something else - the very failure the trimming
# exists to prevent, in a shape this repo's own subject matter is full of.
#
# So this set holds only reporting and citation words, which no ontology label
# contains anywhere. That is the stronger property, so every word here belongs in
# _NON_LABEL_OPENERS too - a word that cannot appear at all cannot appear first.
# Spelled out rather than derived, because the literal reads better than a set
# expression; a test pins the containment so the two cannot drift apart.
_TRAILING_CLAUSE_OPENERS = frozenset(
    {
        "cited",
        "detected",
        "e.g",
        "eg",
        "estimated",
        "found",
        "i.e",
        "ie",
        "n",
        "observed",
        "ref",
        "reported",
        "reviewed",
        "see",
        "seen",
        "source",
    }
)

# Longest a candidate label may be. Ontology labels run long - some GO labels
# pass 100 characters - but a run of prose that long is prose.
_MAX_LABEL_LENGTH = 120
# Most words a *preceding* run may have to still be read as a label. A following
# bracket is not subject to this: `HP:0001250 (...)` is a label position however
# long its contents.
_MAX_PRECEDING_LABEL_WORDS = 8

# A line that is part of a markdown table: pipe-delimited, at least two pipes.
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
# The dashes-and-colons line under a table header, which carries no content.
_TABLE_RULE = re.compile(r"^[\s|:-]+$")
# A list bullet or ordered-list marker at the start of a line.
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

# Label immediately before a bracketed CURIE, set off by emphasis, code or
# quotes: **Seizure** (HP:0001250). The wrapper is what makes this safe to read
# as a label, so an unwrapped run in the same position is not matched here.
_EMPHASISED_BEFORE = re.compile(
    r"(?P<label>"
    r"\*\*[^*\n]{2,120}?\*\*"
    r"|\*[^*\n]{2,120}?\*"
    r"|__[^_\n]{2,120}?__"
    r"|`[^`\n]{2,120}?`"
    r"|\"[^\"\n]{2,120}?\""
    r"|“[^”\n]{2,120}?”"
    r")\s*[(\[]\s*(?P<curie>" + _CURIE_PREFIX + r":" + _CURIE_LOCAL + r")\s*[)\]]"
)

# Label immediately after a CURIE, in brackets or behind a separator:
# HP:0001250 (Seizure), HP:0001250: Seizure, HP:0001250 - Seizure.
_BRACKETED_AFTER = re.compile(
    r"(?P<curie>" + _CURIE_PREFIX + r":" + _CURIE_LOCAL + r")"
    r"\s*[(\[]\s*(?P<label>[^()\[\]\n]{2,120}?)\s*[)\]]"
)
# Brackets end the label rather than being swallowed into it. Without that,
# "HP:0001250 - Seizure (observed in 3 patients)" runs to the closing bracket and
# reports a correctly cited term as naming something else - the "cries wolf"
# outcome this module exists to avoid. The bracketed form of the same aside is
# already handled, by _NON_LABEL_OPENERS; this holds the separator form to the
# same standard.
_SEPARATED_AFTER = re.compile(
    r"(?P<curie>" + _CURIE_PREFIX + r":" + _CURIE_LOCAL + r")"
    r"\s*(?:[:–—]|-{1,2})\s+"
    r"(?P<label>[^|\n()\[\]]{2,120}?)"
    r"\s*(?=[.;]\s|[|()\[\]]|$)"
)


@dataclass(frozen=True)
class FoundTerm:
    """An ontology term identifier discovered in report text.

    Attributes:
        term_id: The CURIE as written, normalized to ``PREFIX:LOCAL``.
        prefix: The CURIE's prefix, for example ``HP``.
        count: Number of times the CURIE occurs in the scanned text.
        labels: Labels the report wrote beside this CURIE, de-duplicated in
            first-appearance order. Empty when the report never named it in a
            position a label could be read out of.

    Examples:
        >>> FoundTerm(term_id="HP:0001250", prefix="HP").count
        1
        >>> FoundTerm(term_id="HP:0001250", prefix="HP").labels
        ()
    """

    term_id: str
    prefix: str
    count: int = 1
    labels: tuple[str, ...] = ()


def is_ontology_curie(text: str) -> bool:
    """Whether a string is a CURIE this module would extract as a term.

    Args:
        text: A candidate identifier.

    Returns:
        True if the string has the shape of an ontology CURIE and its prefix is
        not one of the bibliographic namespaces reference validation owns.

    Examples:
        >>> is_ontology_curie("HP:0001250")
        True
        >>> is_ontology_curie("NCIT:C16814")
        True
        >>> is_ontology_curie("PMID:7913883")
        False
        >>> is_ontology_curie("T10:30")
        False
        >>> is_ontology_curie("not a curie")
        False
    """
    match = _CURIE_PATTERN.fullmatch(text.strip())
    if match is None:
        return False
    return _prefix_is_ontological(match.group("prefix"))


def _prefix_is_ontological(prefix: str) -> bool:
    """Whether a matched prefix should be treated as naming an ontology."""
    if prefix.upper() in NON_ONTOLOGY_PREFIXES:
        return False
    return bool(_TWO_LETTERS.search(prefix))


def clean_label(raw: str) -> Optional[str]:
    """Reduce a candidate label to the words a report meant as the term's name.

    Markdown wrappers, link syntax and edge punctuation are stripped. Candidates
    that cannot be a label - a URL, another CURIE, a bare number, a parenthetical
    that opens with a word no ontology label opens with - are rejected outright.

    Args:
        raw: The candidate text, as matched.

    Returns:
        The cleaned label, or None if the candidate is not a label.

    Examples:
        >>> clean_label("**Seizure**")
        'Seizure'
        >>> clean_label("[Marfan syndrome](https://example.org)")
        'Marfan syndrome'
        >>> clean_label("  Long QT syndrome,  ")
        'Long QT syndrome'
        >>> clean_label("https://bioregistry.io/HP:0001250") is None
        True
        >>> clean_label("HP:0001250") is None
        True
        >>> clean_label("seen in 4 of 11 probands") is None
        True
        >>> clean_label("12") is None
        True
        >>> clean_label("x") is None
        True
    """
    text = _MARKDOWN_LINK.sub(r"\1", raw).strip()
    # Wrappers first, then the punctuation they were hiding: "**Seizure**,"
    # needs both passes to come out as "Seizure".
    text = text.strip(_LABEL_WRAPPERS).strip(_LABEL_EDGE_PUNCTUATION)
    text = text.strip(_LABEL_WRAPPERS).strip()
    text = re.sub(r"\s+", " ", text)

    if not (2 <= len(text) <= _MAX_LABEL_LENGTH):
        return None
    if "://" in text or text.lower().startswith("www."):
        return None
    if _CURIE_PATTERN.search(text):
        return None
    if not re.search(r"[A-Za-z]{2}", text):
        return None
    first_word = _first_word(text)
    if first_word in _NON_LABEL_OPENERS:
        return None
    text = _drop_trailing_clause(text)
    if not (2 <= len(text) <= _MAX_LABEL_LENGTH):
        return None
    return text


def _first_word(text: str) -> str:
    """The first word of a candidate, lowercased and stripped of punctuation.

    Examples:
        >>> _first_word("Reported in 4 probands")
        'reported'
        >>> _first_word("e.g. seizures")
        'e.g'
    """
    return re.split(r"[\s,]+", text.lower(), maxsplit=1)[0].rstrip(".")


def _drop_trailing_clause(label: str) -> str:
    """Cut a comma-led clause that no ontology label would carry.

    ``HP:0001250: Seizure, reported in 4 of 11 probands`` names the term
    "Seizure" and then says something about it. Reading the whole run as the
    label reports a correctly cited term as mislabelled, so the clause is cut at
    the first comma-separated segment that opens with a word from
    :data:`_TRAILING_CLAUSE_OPENERS` - the narrow set, not
    :data:`_NON_LABEL_OPENERS`, whose function words are safe to reject at the
    start of a candidate and would truncate a real name mid-label.

    Only such segments are cut, because commas are ordinary inside real labels -
    "Seizure, generalized" is one, and OMIM-derived disease names are built of
    them - and truncating at every comma would cost far more than it saves.

    Examples:
        >>> _drop_trailing_clause("Seizure, reported in 4 of 11 probands")
        'Seizure'
        >>> _drop_trailing_clause("Seizure, generalized")
        'Seizure, generalized'
        >>> _drop_trailing_clause("Ectopia lentis")
        'Ectopia lentis'

        A real disease name whose later segments open with function words
        survives whole:

        >>> _drop_trailing_clause(
        ...     "microcephaly, with or without chorioretinopathy, lymphedema, "
        ...     "or intellectual disability"
        ... )
        'microcephaly, with or without chorioretinopathy, lymphedema, or intellectual disability'
    """
    segments = label.split(",")
    kept = [segments[0]]
    for segment in segments[1:]:
        if _first_word(segment.strip()) in _TRAILING_CLAUSE_OPENERS:
            break
        kept.append(segment)
    return ",".join(kept).strip()


def _labelled_curies_in_line(line: str) -> list[tuple[str, str]]:
    """Find (CURIE, label) pairs on one line of a report.

    Only the positions described in the module docstring are read. A line is
    handled as a table row when it is pipe-delimited, because a table puts the
    label and the identifier in separate cells, where none of the inline
    patterns can see the pairing.

    Args:
        line: A single line of report text.

    Returns:
        Pairs in the order found. A CURIE with no readable label is absent; it
        is picked up separately by :func:`find_term_ids`.

    Examples:
        >>> _labelled_curies_in_line("| Seizure | HP:0001250 |")
        [('HP:0001250', 'Seizure')]
        >>> _labelled_curies_in_line("- **Marfan syndrome** (MONDO:0007947)")
        [('MONDO:0007947', 'Marfan syndrome')]
        >>> _labelled_curies_in_line("NCIT:C16814 - Echocardiography Test")
        [('NCIT:C16814', 'Echocardiography Test')]
        >>> _labelled_curies_in_line("Patients with dilation (MONDO:0007947) were followed.")
        []
    """
    if _TABLE_ROW.match(line):
        return _labelled_curies_in_table_row(line)

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _record(curie: str, raw_label: str) -> None:
        if not _prefix_is_ontological(curie.split(":", 1)[0]):
            return
        label = clean_label(raw_label)
        if label is None or (curie, label) in seen:
            return
        seen.add((curie, label))
        pairs.append((curie, label))

    for pattern in (_EMPHASISED_BEFORE, _BRACKETED_AFTER, _SEPARATED_AFTER):
        for match in pattern.finditer(line):
            _record(match.group("curie"), match.group("label"))

    # A list item whose text is short enough to be a name rather than a
    # sentence: "- Seizure (HP:0001250)". Prose lines are excluded, because the
    # run before a bracket in a sentence is a clause, not a label.
    bullet = _LIST_MARKER.match(line)
    if bullet:
        item = line[bullet.end() :]
        leading = _BRACKETED_CURIE_AT_START.match(item)
        if leading and len(leading.group("label").split()) <= _MAX_PRECEDING_LABEL_WORDS:
            _record(leading.group("curie"), leading.group("label"))

    return pairs


# A list item that names something and then gives its CURIE in brackets. Anchored
# at the start of the item so the label is the whole of what precedes the
# bracket, not an arbitrary tail of a sentence.
_BRACKETED_CURIE_AT_START = re.compile(
    r"^(?P<label>[^()\[\]|\n.;!?]{2,120}?)\s*[(\[]\s*"
    r"(?P<curie>" + _CURIE_PREFIX + r":" + _CURIE_LOCAL + r")\s*[)\]]"
)


def _labelled_curies_in_table_row(line: str) -> list[tuple[str, str]]:
    """Pair CURIEs with labels across the cells of a markdown table row.

    A cell holding exactly one CURIE takes its label from the cell to its left,
    which is the column order reports use, or from the cell to its right when the
    identifier leads the row. A cell holding both - "Seizure (HP:0001250)" - is
    read inline instead.

    Args:
        line: A pipe-delimited table row.

    Returns:
        Pairs in cell order.

    Examples:
        >>> _labelled_curies_in_table_row("| Seizure | HP:0001250 | common |")
        [('HP:0001250', 'Seizure')]
        >>> _labelled_curies_in_table_row("| HP:0001250 | Seizure |")
        [('HP:0001250', 'Seizure')]
        >>> _labelled_curies_in_table_row(
        ...     "| [Seizure](https://bioregistry.io/HP:0001250) | HP:0001250 | rare |"
        ... )
        [('HP:0001250', 'Seizure')]
        >>> _labelled_curies_in_table_row("| --- | --- |")
        []
    """
    if _TABLE_RULE.match(line):
        return []

    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    pairs: list[tuple[str, str]] = []

    for index, cell in enumerate(cells):
        curies = _CURIE_PATTERN.findall(cell)
        if len(curies) != 1:
            continue
        prefix, local = curies[0]
        if not _prefix_is_ontological(prefix):
            continue
        curie = f"{prefix}:{local}"

        stripped = _MARKDOWN_LINK.sub(r"\1", cell).strip().strip(_LABEL_WRAPPERS)
        if stripped != curie:
            # The cell says more than the identifier, so the label - if there is
            # one - is in this cell, not a neighbouring one.
            for pattern in (_EMPHASISED_BEFORE, _BRACKETED_AFTER, _SEPARATED_AFTER):
                match = pattern.search(cell)
                if match and match.group("curie") == curie:
                    label = clean_label(match.group("label"))
                    if label is not None:
                        pairs.append((curie, label))
                        break
            else:
                inline = _BRACKETED_CURIE_AT_START.match(cell)
                if inline and inline.group("curie") == curie:
                    label = clean_label(inline.group("label"))
                    if label is not None:
                        pairs.append((curie, label))
            continue

        # The cell to the left, when there is one, and otherwise the cell to the
        # right. Reports write the label column before the identifier column, so
        # preferring the left cell is what keeps a trailing "Notes" column from
        # being read as a label; falling back to the right serves the other
        # common layout, where the identifier leads the row.
        neighbour = index - 1 if index > 0 else index + 1
        if 0 <= neighbour < len(cells):
            # Links are reduced to their text first: a label cell that links to
            # bioregistry carries the CURIE in its href, and testing the raw cell
            # would discard the very cell holding the label.
            candidate = _MARKDOWN_LINK.sub(r"\1", cells[neighbour])
            if not _CURIE_PATTERN.search(candidate):
                label = clean_label(candidate)
                if label is not None:
                    pairs.append((curie, label))

    return pairs


def find_term_ids(text: str) -> list[FoundTerm]:
    """Find every ontology term CURIE in a block of text.

    CURIEs are de-duplicated while preserving first-appearance order, and each
    result records how many times the CURIE occurs and every label the report
    wrote beside it.

    ``count`` counts textual mentions, not citations: a markdown link spells its
    CURIE twice and so counts twice, exactly as reference extraction does.

    Args:
        text: Markdown or plain text to scan.

    Returns:
        Ordered list of unique terms found.

    Examples:
        >>> terms = find_term_ids("Seizures were frequent (HP:0001250), see HP:0001250.")
        >>> [(t.term_id, t.count) for t in terms]
        [('HP:0001250', 2)]
        >>> terms = find_term_ids("| Seizure | HP:0001250 |")
        >>> terms[0].labels
        ('Seizure',)
        >>> [t.term_id for t in find_term_ids("Cites PMID:7913883 and DOI:10.1038/ng1234.")]
        []
        >>> [t.term_id for t in find_term_ids("http://purl.obolibrary.org/obo/HP_0001250")]
        ['HP:0001250']
        >>> find_term_ids("no terms here")
        []
    """
    found: dict[str, FoundTerm] = {}

    def _add(term_id: str, prefix: str) -> None:
        existing = found.get(term_id)
        if existing is None:
            found[term_id] = FoundTerm(term_id=term_id, prefix=prefix, count=1)
        else:
            found[term_id] = FoundTerm(
                term_id=existing.term_id,
                prefix=existing.prefix,
                count=existing.count + 1,
                labels=existing.labels,
            )

    def _add_label(term_id: str, label: str) -> None:
        existing = found.get(term_id)
        if existing is None or label in existing.labels:
            return
        found[term_id] = FoundTerm(
            term_id=existing.term_id,
            prefix=existing.prefix,
            count=existing.count,
            labels=existing.labels + (label,),
        )

    for match in _CURIE_PATTERN.finditer(text):
        prefix = match.group("prefix")
        if _prefix_is_ontological(prefix):
            _add(f"{prefix}:{match.group('local')}", prefix)

    for match in _PURL_PATTERN.finditer(text):
        prefix = match.group("prefix")
        if _prefix_is_ontological(prefix):
            _add(f"{prefix}:{match.group('local')}", prefix)

    for line in text.splitlines():
        for curie, label in _labelled_curies_in_line(line):
            _add_label(curie, label)

    return list(found.values())


def extract_terms(
    markdown: str,
    citations: Optional[Iterable[str]] = None,
) -> list[FoundTerm]:
    """Extract ontology terms from a report body and its citation list.

    Args:
        markdown: The report body.
        citations: Optional citation strings (e.g. ``ResearchResult.citations``)
            scanned in addition to the body.

    Returns:
        Ordered list of unique terms, with occurrence counts and labels summed
        across the body and the citation list.

    Examples:
        >>> terms = extract_terms("Body cites HP:0001250.", ["Also MONDO:0007947"])
        >>> [t.term_id for t in terms]
        ['HP:0001250', 'MONDO:0007947']
        >>> extract_terms("")
        []
    """
    combined = markdown
    if citations:
        combined = "\n".join([markdown, *citations])
    return find_term_ids(combined)
