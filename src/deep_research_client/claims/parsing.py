"""Turn an extractor's reply into anchored claims.

Kept apart from the call that produces the reply, so everything that decides
what a claim *is* -- anchoring, citation resolution, entity handling -- is a
pure function of text and can be tested on a recorded reply without a model.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..evaluation.scorers import extract_json_object
from ..validation.extraction import find_reference_ids
from ..validation.term_extraction import is_ontology_curie
from .anchoring import NUMERIC_MARKER, locate_quote, nearest_passage
from .models import (
    AnchorStatus,
    CitationHandle,
    CitationScope,
    Claim,
    ClaimBasis,
    ClaimTopic,
    EntityMention,
    TextSpan,
    citation_status_for,
)

__all__ = [
    "TextUnit",
    "UnreadableReplyError",
    "citation_window",
    "claims_from_reply",
    "resolve_citations",
    "section_citation",
]

_URL = re.compile(r"https?://[^\s<>\]]+")

#: A citation marker that is only a reference number, bracketed or not.
_NUMBERED_MARKER = re.compile(r"^\[?\s*(\d+)\s*\]?$")


class UnreadableReplyError(ValueError):
    """An extractor reply that holds no readable claims list.

    Raised rather than read as "no claims": a unit whose reply was cut off or
    garbled would otherwise look like a unit that makes no claims, and the
    claim set would look complete when it is not.
    """


@dataclass(frozen=True)
class TextUnit:
    """A stretch of source text sent to the extractor in one request.

    Attributes:
        text: The text that offsets index into: the whole report, or the
            field's own text for prose inside a structured document.
        start: Where the unit begins in ``text``.
        end: Where it ends (exclusive).
        section: Heading path of the unit, when the source has headings.
        source_path: Path of the field, for prose inside a structured document.
        bibliography: Numbered reference list entries by number, used to
            resolve markers such as ``[3]``.
        section_citation: The cited work the unit's section is headed by, if
            any (see :func:`section_citation`). Every located claim in the
            unit is attributed to it.

    >>> TextUnit(text="abc", start=0, end=3).body
    'abc'
    """

    text: str
    start: int
    end: int
    section: Optional[str] = None
    source_path: Optional[str] = None
    bibliography: Mapping[int, str] = field(default_factory=dict)
    section_citation: Optional[CitationHandle] = None

    @property
    def body(self) -> str:
        """The unit's own text.

        Returns:
            ``text[start:end]``.
        """
        return self.text[self.start:self.end]


def _cites(number: int, body: str) -> bool:
    """Whether a numeric marker in the text cites a reference number.

    Only brackets that hold nothing but numbers, commas and dashes are
    markers, so "[Figure 3]" and a link label like "[3 cases](...)" are not.
    A range cites every number in it.

    >>> [n for n in range(1, 9) if _cites(n, "A [1]. B [2, 4-6]. C [Figure 3]. D [7 cases](https://x).")]
    [1, 2, 4, 5, 6]
    """
    for match in re.finditer(NUMERIC_MARKER, body):
        inside = match.group(0)[1:match.group(0).index("]")]
        for part in re.split(r"\s*,\s*", inside.strip()):
            bounds = [int(n) for n in re.split(r"\s*[–-]\s*", part)]
            if bounds[0] <= number <= bounds[-1]:
                return True
    return False


def _marker_present(marker: str, body: str) -> bool:
    """Whether a citation marker the extractor reported appears in the unit.

    A numbered marker is present when a numeric marker in the text cites that
    number, alone, in a list, or within a range.

    >>> _marker_present("[3]", "caused by FBN1 [2, 3].")
    True
    >>> _marker_present("[3]", "caused by FBN1 [2-5].")
    True
    >>> _marker_present("[3]", "as in [Figure 3].")
    False
    >>> _marker_present("PMID:123", "see PMID:123")
    True
    >>> _marker_present("[9]", "caused by FBN1 [2, 3].")
    False
    """
    numbered = _NUMBERED_MARKER.match(marker.strip())
    if numbered:
        return _cites(int(numbered.group(1)), body)
    return marker.strip() in body


def resolve_citations(
    markers: list[Any], body: str, bibliography: Mapping[int, str],
) -> list[CitationHandle]:
    """Keep the citation markers that are really there, and resolve them.

    Each is a citation of the claim's own sentence (scope SENTENCE).

    A marker the extractor reports but the text does not contain is dropped:
    a citation is a fact about the source, not something to take on trust.
    One reference gives one handle: a later marker that resolves to an
    identifier already kept (``[2]``, then its URL) is skipped.

    Args:
        markers: Markers as the extractor reported them.
        body: The unit text the claim came from.
        bibliography: Numbered reference entries, for resolving ``[n]``.

    Returns:
        One handle per distinct marker present, with a normalised identifier
        and URL where the marker or its bibliography entry gives one.

    >>> handles = resolve_citations(
    ...     ["[2]", "[7]", "PMID:15241795"],
    ...     "FBN1 variants [2] cause it (PMID:15241795).",
    ...     {2: "Dietz HC. Nature. 1991. https://doi.org/10.1038/352337a0"},
    ... )
    >>> [(h.marker, h.reference_id, h.scope) for h in handles]
    [('[2]', 'DOI:10.1038/352337a0', 'SENTENCE'), ('PMID:15241795', 'PMID:15241795', 'SENTENCE')]
    >>> same = resolve_citations(
    ...     ["[2]", "https://doi.org/10.1038/352337a0"],
    ...     "FBN1 variants [2](https://doi.org/10.1038/352337a0) cause it.",
    ...     {2: "Dietz HC. Nature. 1991. https://doi.org/10.1038/352337a0"},
    ... )
    >>> [h.marker for h in same]
    ['[2]']
    """
    handles: list[CitationHandle] = []
    seen: set[str] = set()
    for raw in markers:
        if not isinstance(raw, str) or not raw.strip():
            continue
        marker = raw.strip()
        if marker in seen or not _marker_present(marker, body):
            continue
        seen.add(marker)

        numbered = _NUMBERED_MARKER.match(marker)
        resolvable = marker
        if numbered and int(numbered.group(1)) in bibliography:
            resolvable = bibliography[int(numbered.group(1))]
        found = find_reference_ids(resolvable)
        reference_id = found[0].normalized_id if found else None
        if reference_id is not None and reference_id in {h.reference_id for h in handles}:
            continue
        url = _URL.search(resolvable)
        handles.append(CitationHandle(
            marker=marker,
            reference_id=reference_id,
            url=url.group(0).rstrip(".,;)") if url else None,
            scope=CitationScope.SENTENCE,
        ))
    return handles


#: A heading that names a cited work: "[3] Title", as reports that list
#: papers write them.
_CITED_HEADING = re.compile(r"^\[(\d+)\]\s+\S")

#: A metadata line giving the section's own work an identifier: "- PMID: 123",
#: "DOI: 10.1/x". Only such lines are read, so an identifier the section's
#: prose mentions in passing is not taken for the work's own.
_IDENTIFIER_LINE = re.compile(r"^[ \t]*(?:[-*+][ \t]+)?(?:PMID|PMCID|DOI)[ \t]*:.*$", re.IGNORECASE | re.MULTILINE)


def section_citation(
    heading: str, section_text: str, bibliography: Mapping[int, str],
) -> Optional[CitationHandle]:
    r"""The cited work a section is headed by, if its heading names one.

    A report that lists papers gives each its own section, headed "[n] Title".
    Everything in that section comes from work n, so it is a citation of every
    claim in the section even where no sentence carries a marker. The work is
    identified through its bibliography entry, or else through the section's
    own PMID or DOI metadata lines.

    Args:
        heading: The heading's text, without the leading ``#``.
        section_text: The section's own text, up to its first subheading.
        bibliography: Numbered reference entries, for resolving ``[n]``.

    Returns:
        A handle with scope SECTION, or None when the heading names no work.

    >>> body = "- Year: 2025\n- DOI: 10.1007/s10067-025-07811-3\n- PMID: 41258631\n- Summary: ..."
    >>> handle = section_citation("[1] Avascular necrosis in APS", body,
    ...                           {1: "Zankar R (2025). https://www.semanticscholar.org/paper/70595d"})
    >>> handle.marker, handle.reference_id, handle.url, handle.scope
    ('[1]', 'PMID:41258631', 'https://www.semanticscholar.org/paper/70595d', 'SECTION')
    >>> section_citation("Genetics", body, {}) is None
    True
    """
    numbered = _CITED_HEADING.match(heading.strip())
    if not numbered:
        return None
    entry = bibliography.get(int(numbered.group(1)), "")
    found = find_reference_ids(entry) or find_reference_ids(
        "\n".join(m.group(0) for m in _IDENTIFIER_LINE.finditer(section_text))
    )
    url = _URL.search(entry)
    return CitationHandle(
        marker=f"[{numbered.group(1)}]",
        reference_id=found[0].normalized_id if found else None,
        url=url.group(0).rstrip(".,;)") if url else None,
        scope=CitationScope.SECTION,
    )


#: Where a sentence, table row or list item ends: terminal punctuation
#: followed by whitespace or the end of the text, or a line break that starts
#: a new block. Numbered markers written
#: after the full stop (".[1]" or ". [1]", as several providers write them)
#: belong to the sentence before them, so the end runs on over them. The full
#: stop of "et al.", "e.g." and the like is not a sentence end.
#: Abbreviations whose full stop does not end a sentence. Checked case
#: insensitively, each a separate fixed-width lookbehind.
_ABBREVIATIONS = ("al", "e.g", "i.e", "vs", "fig", "figs", "approx", "cf", "ca", "resp")
_NOT_ABBREVIATION = "".join(
    rf"(?<!\b{re.escape(a)})(?<!\b{re.escape(a.capitalize())})" for a in _ABBREVIATIONS
)

#: A line break that starts a new block: a blank line, a table row, a heading,
#: a block quote, or a list item. Any other line break is a hard wrap inside a
#: paragraph, and a sentence runs on across it.
_BLOCK_BREAK = r"\n(?=[ \t]*(?:$|\n|[|#>]|[*+-][ \t]|\d+[.)][ \t]))"

_SENTENCE_END = re.compile(
    rf"(?:[!?]|{_NOT_ABBREVIATION}\.)(?:\s*{NUMERIC_MARKER})*(?=\s|$)|{_BLOCK_BREAK}"
)


def citation_window(unit: TextUnit, span: TextSpan) -> str:
    r"""The text a citation must appear in to count as attached to a claim.

    That is the sentence (or table row) containing the claim's span, from the
    previous sentence end to the next one after the span. A marker elsewhere in
    the section may belong to a different claim, so the model's say-so is not
    enough to attach it. A claim with no span gets no citations, since there is
    no sentence to check against.

    Args:
        unit: The unit the claim came from.
        span: The claim's located span.

    Returns:
        The window's text.

    >>> text = "A is B [1]. C causes D, which causes E [2].\nF [3]."
    >>> unit = TextUnit(text=text, start=0, end=len(text))
    >>> citation_window(unit, TextSpan(start=12, end=23, text="C causes D,"))
    ' C causes D, which causes E [2].'
    """
    # Searched over the whole unit, not the slice before the span: in a slice,
    # "$" matches at the span's start, so a hard wrap or a decimal point just
    # before it would read as a sentence end.
    starts = [
        m.end() for m in _SENTENCE_END.finditer(unit.text, unit.start, unit.end)
        if m.end() <= span.start
    ]
    window_start = starts[-1] if starts else unit.start
    # From the span's last character: a span that already ends its sentence
    # (a normalised match runs on over the full stop) must not reach into the
    # next one.
    after = _SENTENCE_END.search(unit.text, max(span.start, span.end - 1), unit.end)
    window_end = after.end() if after else unit.end
    return unit.text[window_start:window_end]


def _mention(value: Any) -> Optional[EntityMention]:
    """An entity mention from a reply value: a string, or ``{label, id}``.

    A label that is itself an ontology CURIE is taken as grounded.

    >>> _mention("MONDO:0007947")
    EntityMention(label='MONDO:0007947', id='MONDO:0007947')
    >>> _mention({"label": "FBN1", "id": "HGNC:3603"})
    EntityMention(label='FBN1', id='HGNC:3603')
    >>> _mention("") is None
    True
    """
    if isinstance(value, Mapping):
        label = value.get("label") or value.get("name")
        grounded = value.get("id")
    else:
        label, grounded = value, None
    if not isinstance(label, str) or not label.strip():
        return None
    label = label.strip()
    if grounded is None and is_ontology_curie(label):
        grounded = label
    return EntityMention(label=label, id=grounded if isinstance(grounded, str) else None)


def _text(value: Any) -> Optional[str]:
    """A non-empty stripped string, or None."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _member(value: Any, enum: type[ClaimTopic] | type[ClaimBasis]) -> Any:
    """The enum member a model's label names, or None if it names none.

    Models write these as ``"secondary_source"``, ``"Secondary source"`` or
    ``"SECONDARY-SOURCE"``; all mean the same member.

    >>> _member("secondary source", ClaimBasis).value, _member("Work", ClaimTopic).value
    ('SECONDARY_SOURCE', 'WORK')
    >>> _member("hearsay", ClaimBasis) is None, _member(None, ClaimTopic) is None
    (True, True)
    """
    text = _text(value)
    if text is None:
        return None
    return enum.__members__.get(re.sub(r"[\s-]+", "_", text).upper())


def claims_from_reply(reply: str, unit: TextUnit) -> list[Claim]:
    """Build claims from an extractor's JSON reply about one unit.

    Each claim's quote is located in the unit; one that cannot be found is
    kept as UNANCHORED with no span. Claim ids are provisional (``u1``,
    ``u2``...) and are renumbered across the whole source by the caller.

    Args:
        reply: The model's reply, expected to hold ``{"claims": [...]}``.
        unit: The unit the reply is about.

    Returns:
        The claims, in reply order. ``{"claims": []}`` yields none.

    Raises:
        UnreadableReplyError: If the reply holds no ``claims`` list, for
            example because it was cut off mid-object.

    >>> text = "## Genetics\\nMarfan syndrome is caused by FBN1 variants [1].\\n"
    >>> unit = TextUnit(text=text, start=12, end=len(text), section="Genetics")
    >>> reply = '{"claims": [{"claim": "FBN1 variants cause Marfan syndrome.",'
    >>> reply += ' "quote": "Marfan syndrome is caused by FBN1 variants",'
    >>> reply += ' "subject": "FBN1 variants", "predicate": "causes",'
    >>> reply += ' "object": "Marfan syndrome", "citations": ["[1]"]}]}'
    >>> claim = claims_from_reply(reply, unit)[0]
    >>> claim.anchor_status, claim.source_span.start, claim.section
    ('EXACT', 12, 'Genetics')
    >>> claim.subject.label, [c.marker for c in claim.citations], claim.citation_status
    ('FBN1 variants', ['[1]'], 'CITED')
    """
    parsed = extract_json_object(reply, key="claims")
    entries = parsed.get("claims") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        where = unit.section or unit.source_path or "the source"
        raise UnreadableReplyError(
            f"The claim-extraction reply for {where} holds no readable claims list: "
            f"{reply[:200]!r}"
        )

    claims: list[Claim] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        claim_text = _text(entry.get("claim"))
        if claim_text is None:
            continue

        quote = _text(entry.get("quote"))
        span, status = (
            locate_quote(quote, unit.text, unit.start, unit.end)
            if quote else (None, AnchorStatus.UNANCHORED)
        )
        negated = entry.get("negated")
        entities = [m for m in (_mention(e) for e in entry.get("entities") or []) if m]
        citations = resolve_citations(
            entry.get("citations") or [],
            citation_window(unit, span) if span is not None else "",
            unit.bibliography,
        )
        # A section headed by a cited work attributes all of it to that work.
        # Not for an unlocated claim, which may not come from the section, and
        # not again when the sentence already cites the same work.
        section = unit.section_citation
        if span is not None and section is not None and not any(
            c.marker == section.marker
            or (section.reference_id is not None and c.reference_id == section.reference_id)
            for c in citations
        ):
            citations.append(section)
        about = _member(entry.get("about"), ClaimTopic)
        basis = _member(entry.get("basis"), ClaimBasis)
        # Only a domain claim has a basis. So a basis with no readable "about"
        # says the claim is about the domain, while one on a claim the model
        # called a work claim means nothing and is not kept.
        if about is None and basis is not None:
            about = ClaimTopic.DOMAIN
        if about != ClaimTopic.DOMAIN:
            basis = None
        # In a section headed by a cited work, the source presents every claim
        # as that work's. That is structure, not judgement, so it is not left
        # to the model, which does not always apply it.
        if about == ClaimTopic.DOMAIN and span is not None and unit.section_citation is not None:
            basis = ClaimBasis.SECONDARY_SOURCE
        claims.append(Claim(
            id=f"u{len(claims) + 1}",
            claim_text=claim_text,
            source_span=span,
            source_path=unit.source_path,
            anchor_status=status,
            section=unit.section,
            subject=_mention(entry.get("subject")),
            predicate=_mention(entry.get("predicate")),
            object=_mention(entry.get("object")),
            negated=negated if isinstance(negated, bool) else None,
            qualifier=_text(entry.get("qualifier")),
            subject_qualifier=_text(entry.get("subject_qualifier")),
            object_qualifier=_text(entry.get("object_qualifier")),
            entities=entities or None,
            citations=citations or None,
            nearest_passage=(
                nearest_passage(quote, unit.text, unit.start, unit.end)
                if quote and status == AnchorStatus.UNANCHORED else None
            ),
            citation_status=citation_status_for(citations, status),
            about=about,
            basis=basis,
        ))
    return claims
