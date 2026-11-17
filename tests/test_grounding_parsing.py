"""Reading a grounding choice from a reply, as a pure function of reply + candidates."""

import pytest

from deep_research_client.claims import GroundedTerm, GroundingCandidate
from deep_research_client.claims.grounding_parsing import grounding_from_reply

CANDS = [
    GroundingCandidate(id="MONDO:0007947", label="Marfan syndrome", definition="A disorder..."),
    GroundingCandidate(id="MONDO:0011071", label="isolated ectopia lentis"),
]


def test_a_chosen_candidate_becomes_a_grounded_term() -> None:
    term = grounding_from_reply('{"id": "MONDO:0007947", "reason": "exact"}', CANDS)
    assert term == GroundedTerm(id="MONDO:0007947", label="Marfan syndrome")


@pytest.mark.parametrize("reply", [
    '{"id": null, "reason": "none fit"}',
    '{"id": "", "reason": "none"}',
    '{"id": "none"}',
    'I could not find a match.',  # no JSON at all
])
def test_no_choice_yields_none(reply) -> None:
    assert grounding_from_reply(reply, CANDS) is None


def test_an_id_not_among_the_candidates_is_refused() -> None:
    """The model may not invent a CURIE the ontology search did not offer."""
    assert grounding_from_reply('{"id": "MONDO:9999999"}', CANDS) is None


def test_the_label_comes_from_the_candidate_not_the_reply() -> None:
    """A grounded term carries the ontology's own label, whatever the reply says."""
    term = grounding_from_reply('{"id": "MONDO:0011071", "label": "something else"}', CANDS)
    assert term == GroundedTerm(id="MONDO:0011071", label="isolated ectopia lentis")
