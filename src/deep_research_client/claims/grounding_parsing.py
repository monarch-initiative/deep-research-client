"""Read a grounding choice from a model reply.

Kept apart from the call that produces the reply, so what a choice *is* -- the id
read, checked against the candidates offered, turned into a grounded term -- is a
pure function of the reply and the candidates, testable on a recorded reply with
no model and no ontology service.
"""

from typing import Optional, Sequence

from ..evaluation.scorers import extract_json_object
from .grounding import GroundingCandidate, GroundedTerm

__all__ = ["grounding_from_reply"]

#: Replies that mean "none of the candidates fits", besides a JSON null.
_NO_CHOICE = frozenset({"", "none", "null", "n/a", "na"})


def grounding_from_reply(
    reply: str, candidates: Sequence[GroundingCandidate]
) -> Optional[GroundedTerm]:
    """The candidate the reply chose, or None.

    The model is asked for the id of the one candidate that denotes the same
    concept, or none. An id that is not among the candidates is treated as no
    choice, not trusted: the model may not invent a CURIE the ontology search did
    not offer.

    Args:
        reply: The model's reply to a grounding prompt.
        candidates: The candidates it was shown.

    Returns:
        The chosen candidate as a GroundedTerm, or None when the reply chose none
        or named an id that was not offered.

    >>> cands = [GroundingCandidate(id="MONDO:0007947", label="Marfan syndrome"),
    ...          GroundingCandidate(id="MONDO:0011071", label="isolated ectopia lentis")]
    >>> grounding_from_reply('{"id": "MONDO:0007947", "reason": "exact match"}', cands)
    GroundedTerm(id='MONDO:0007947', label='Marfan syndrome')
    >>> grounding_from_reply('{"id": null, "reason": "no candidate fits"}', cands) is None
    True
    >>> grounding_from_reply('{"id": "MONDO:9999999"}', cands) is None  # not offered
    True
    """
    data = extract_json_object(reply, key="id")
    raw = data.get("id") if data else None
    if not isinstance(raw, str) or raw.strip().casefold() in _NO_CHOICE:
        return None
    chosen = raw.strip()
    by_id = {candidate.id: candidate for candidate in candidates}
    candidate = by_id.get(chosen)
    return GroundedTerm(id=candidate.id, label=candidate.label) if candidate else None
