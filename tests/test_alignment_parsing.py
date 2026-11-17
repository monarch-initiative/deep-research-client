"""Reading a relation verdict from a reply, as a pure function of the text."""

import pytest

from deep_research_client.claims.align_parsing import relation_from_reply
from deep_research_client.claims.parsing import UnreadableReplyError


@pytest.mark.parametrize(
    ("reply", "relation", "confidence"),
    [
        ('{"relation": "identical", "confidence": 0.9}', "IDENTICAL", 0.9),
        ('{"relation": "SUBSUMES"}', "SUBSUMES", None),
        ('  reasoning...\n{"relation": "close", "confidence": 1}', "CLOSE", 1.0),
        ('{"relation": "contradictory", "confidence": 2.5}', "CONTRADICTORY", 1.0),  # clamped
        ('{"relation": "unrelated", "confidence": "high"}', "UNRELATED", None),  # non-numeric
    ],
)
def test_a_well_formed_reply_yields_its_verdict(reply, relation, confidence) -> None:
    verdict = relation_from_reply(reply)
    assert verdict.semantic_relation.value == relation
    assert verdict.confidence == confidence


def test_a_self_correcting_reply_commits_to_its_last_relation() -> None:
    """Real models state an answer, reconsider, and state another; the last wins."""
    reply = (
        '{"relation": "subsumes", "confidence": 0.6}\n'
        "On reflection A is the more specific claim.\n"
        '{"relation": "subsumed_by", "confidence": 0.7}'
    )
    assert relation_from_reply(reply).semantic_relation.value == "SUBSUMED_BY"


def test_a_later_object_without_a_known_relation_does_not_override() -> None:
    """A trailing object that is not a verdict leaves the last real relation standing."""
    reply = '{"relation": "close"} and then {"note": "unsure"}'
    assert relation_from_reply(reply).semantic_relation.value == "CLOSE"


def test_an_unknown_relation_is_refused() -> None:
    with pytest.raises(UnreadableReplyError, match="No relation"):
        relation_from_reply('{"relation": "mostly_the_same"}')


def test_a_reply_with_no_json_is_refused() -> None:
    with pytest.raises(UnreadableReplyError, match="No relation"):
        relation_from_reply("They are about the same thing, I think.")


def test_a_blank_rationale_becomes_none() -> None:
    assert relation_from_reply('{"relation": "close", "rationale": "   "}').rationale is None
