"""Turn an extractor's reply into anchored claims.

Kept apart from the call that produces the reply, so everything that decides
what a claim *is* -- anchoring, citation resolution, entity handling -- is a
pure function of text and can be tested on a recorded reply without a model.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ..evaluation.scorers import _extract_json_object
from ..validation.extraction import find_reference_ids
from ..validation.term_extraction import is_ontology_curie
from .anchoring import locate_quote
from .models import AnchorStatus, CitationHandle, Claim, EntityMention

__all__ = ["TextUnit", "claims_from_reply", "resolve_citations"]

_URL = re.compile(r"https?://[^\s<>\]]+")

#: A citation marker that is only a reference number, bracketed or not.
_NUMBERED_MARKER = re.compile(r"^\[?\s*(\d+)\s*\]?$")


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

    >>> TextUnit(text="abc", start=0, end=3).body
    'abc'
    """

    text: str
    start: int
    end: int
    section: Optional[str] = None
    source_path: Optional[str] = None
    bibliography: Mapping[int, str] = field(default_factory=dict)

    @property
    def body(self) -> str:
        """The unit's own text.

        Returns:
            ``text[start:end]``.
        """
        return self.text[self.start:self.end]


def _marker_present(marker: str, body: str) -> bool:
    """Whether a citation marker the extractor reported appears in the unit.

    Numbered markers are matched inside any bracket group, so "3" is present
    in "[2, 3]" as well as in "[3]".

    >>> _marker_present("[3]", "caused by FBN1 [2, 3].")
    True
    >>> _marker_present("PMID:123", "see PMID:123")
    True
    >>> _marker_present("[9]", "caused by FBN1 [2, 3].")
    False
    """
    numbered = _NUMBERED_MARKER.match(marker.strip())
    if numbered:
        number = numbered.group(1)
        return re.search(rf"\[[^\]]*\b{number}\b[^\]]*\]", body) is not None
    return marker.strip() in body


def resolve_citations(
    markers: list[Any], body: str, bibliography: Mapping[int, str],
) -> list[CitationHandle]:
    """Keep the citation markers that are really there, and resolve them.

    A marker the extractor reports but the text does not contain is dropped:
    a citation is a fact about the source, not something to take on trust.

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
    >>> [(h.marker, h.reference_id) for h in handles]
    [('[2]', 'DOI:10.1038/352337a0'), ('PMID:15241795', 'PMID:15241795')]
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
        url = _URL.search(resolvable)
        handles.append(CitationHandle(
            marker=marker,
            reference_id=found[0].normalized_id if found else None,
            url=url.group(0).rstrip(".,;)") if url else None,
        ))
    return handles


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


def claims_from_reply(reply: str, unit: TextUnit) -> list[Claim]:
    """Build claims from an extractor's JSON reply about one unit.

    Each claim's quote is located in the unit; one that cannot be found is
    kept as UNANCHORED with no span. Claim ids are provisional (``u1``,
    ``u2``...) and are renumbered across the whole source by the caller.

    Args:
        reply: The model's reply, expected to hold ``{"claims": [...]}``.
        unit: The unit the reply is about.

    Returns:
        The claims, in reply order. An unreadable reply yields none; the
        caller decides whether that is an error.

    >>> text = "## Genetics\\nMarfan syndrome is caused by FBN1 variants [1].\\n"
    >>> unit = TextUnit(text=text, start=12, end=len(text), section="Genetics")
    >>> reply = '{"claims": [{"claim": "FBN1 variants cause Marfan syndrome.",'
    >>> reply += ' "quote": "Marfan syndrome is caused by FBN1 variants",'
    >>> reply += ' "subject": "FBN1 variants", "predicate": "causes",'
    >>> reply += ' "object": "Marfan syndrome", "citations": ["[1]"]}]}'
    >>> claim = claims_from_reply(reply, unit)[0]
    >>> claim.anchor_status, claim.source_span.start, claim.section
    ('EXACT', 12, 'Genetics')
    >>> claim.subject.label, [c.marker for c in claim.citations]
    ('FBN1 variants', ['[1]'])
    """
    parsed = _extract_json_object(reply, key="claims")
    entries = parsed.get("claims") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return []

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
            citations=resolve_citations(
                entry.get("citations") or [], unit.body, unit.bibliography,
            ) or None,
        ))
    return claims
