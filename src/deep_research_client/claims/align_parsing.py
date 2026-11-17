"""Turn a relation-classification reply into a verdict.

Kept apart from the call that produces the reply, so what a verdict *is* -- the
relation read, the confidence clamped, the rationale kept -- is a pure function of
the reply text and can be tested on a recorded reply without a model.

A model sometimes states an answer, reconsiders, and states a corrected one: real
recorded replies do this. So the reply is read as "the last relation it committed
to", not the first: every balanced top-level JSON object is scanned, and the last
one carrying a known relation wins.
"""

import json
from dataclasses import dataclass
from typing import Optional

from .alignment_datamodel import SemanticRelation
from .parsing import UnreadableReplyError

__all__ = ["RelationVerdict", "relation_from_reply"]


@dataclass(frozen=True)
class RelationVerdict:
    """What classifying one pair produced.

    Attributes:
        semantic_relation: The relation the model named.
        confidence: Its confidence in 0..1, or None if it gave none or an
            unreadable one.
        rationale: Its one-sentence justification, or None.
    """

    semantic_relation: SemanticRelation
    confidence: Optional[float]
    rationale: Optional[str]


def _json_objects(text: str) -> list[dict]:
    """Every balanced top-level JSON object in the text, decoded, in order.

    A simple brace scan that ignores braces inside strings, so prose and several
    objects around the answer do not defeat it.

    >>> _json_objects('first {"a": 1} then {"b": {"c": 2}} end')
    [{'a': 1}, {'b': {'c': 2}}]
    """
    objects: list[dict] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    decoded = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, dict):
                    objects.append(decoded)
                start = None
    return objects


def _confidence(raw: object) -> Optional[float]:
    """A reply's confidence clamped to 0..1, or None when it is not a number.

    >>> _confidence(0.9), _confidence(1.5), _confidence(-1), _confidence("high")
    (0.9, 1.0, 0.0, None)
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return max(0.0, min(1.0, float(raw)))


def relation_from_reply(reply: str) -> RelationVerdict:
    """Read the semantic relation, confidence and rationale from a reply.

    Args:
        reply: The model's reply to a classification prompt.

    Returns:
        The verdict the reply states; if the model stated more than one relation,
        the last one it committed to.

    Raises:
        UnreadableReplyError: If no JSON object with a known ``relation`` is found;
            read as "no verdict" rather than a relation, a garbled reply would make
            a pair look classified when it is not.

    >>> v = relation_from_reply('{"relation": "identical", "confidence": 0.8, "rationale": "same."}')
    >>> v.semantic_relation.value, v.confidence, v.rationale
    ('IDENTICAL', 0.8, 'same.')

    A model that corrects itself commits to its last relation:

    >>> reply = ('{"relation": "subsumes", "confidence": 0.6}\\n'
    ...          'Wait, A is the more specific claim.\\n'
    ...          '{"relation": "subsumed_by", "confidence": 0.65}')
    >>> relation_from_reply(reply).semantic_relation.value
    'SUBSUMED_BY'

    >>> relation_from_reply('nonsense')
    Traceback (most recent call last):
    ...
    deep_research_client.claims.parsing.UnreadableReplyError: No relation in reply: 'nonsense'
    """
    chosen: Optional[dict] = None
    chosen_relation: Optional[SemanticRelation] = None
    for obj in _json_objects(reply):
        raw = obj.get("relation")
        if not isinstance(raw, str):
            continue
        try:
            relation = SemanticRelation(raw.strip().upper())
        except ValueError:
            continue
        chosen, chosen_relation = obj, relation
    if chosen is None or chosen_relation is None:
        raise UnreadableReplyError(f"No relation in reply: {reply[:200]!r}")
    rationale = chosen.get("rationale")
    return RelationVerdict(
        semantic_relation=chosen_relation,
        confidence=_confidence(chosen.get("confidence")),
        rationale=rationale.strip() if isinstance(rationale, str) and rationale.strip() else None,
    )
