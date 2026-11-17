"""Real model replies to the grounding prompt, replayed through the parser.

The replies in tests/input/claims/grounding/<model>/ were written by a model
choosing among the OAK candidates committed in candidates/ (see manifest.yaml).
Each is passed straight to ``grounding_from_reply`` with those candidates; no
client and no ontology service are involved, so what is tested is the real parser
on real model output.

The first test keeps the recording honest: if the prompt, its version, or the
terms change, the recorded replies no longer answer what the code would ask.
Assertions about which term the model chose carry MODEL_DEPENDENT.
"""

import json
from functools import cache
from pathlib import Path

import pytest
import yaml

from deep_research_client.claims import GroundingCandidate
from deep_research_client.claims.grounding_llm import GROUNDING_PROMPT_VERSION, build_grounding_prompt
from deep_research_client.claims.grounding_parsing import grounding_from_reply

GROUND = Path(__file__).parent / "input" / "claims" / "grounding"
MANIFEST = yaml.safe_load((GROUND / "manifest.yaml").read_text(encoding="utf-8"))
MODELS = sorted(MANIFEST["recordings"])
MODEL_DEPENDENT = "MODEL_DEPENDENT (re-check after re-recording): "


@cache
def _candidates(index: int) -> tuple[GroundingCandidate, ...]:
    data = json.loads((GROUND / "candidates" / f"{index:02d}.json").read_text(encoding="utf-8"))
    return tuple(GroundingCandidate(**c) for c in data)


def _reply(model: str, index: int) -> str:
    return (GROUND / model / f"{index:02d}.txt").read_text(encoding="utf-8")


def test_the_recording_answers_the_prompts_this_code_sends() -> None:
    """Re-record (run record.py) whenever this fails."""
    assert MANIFEST["prompt_version"] == GROUNDING_PROMPT_VERSION
    system_file = (GROUND / "prompts" / "SYSTEM.txt").read_text(encoding="utf-8")
    for i, term in enumerate(MANIFEST["terms"], 1):
        system, user = build_grounding_prompt(term["label"], term["context"], _candidates(i))
        assert system["content"] == system_file
        assert user["content"] == (GROUND / "prompts" / f"{i:02d}.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("model", MODELS)
def test_every_reply_parses_to_an_offered_candidate_or_none(model) -> None:
    """Whatever the model chose, the parser returns one of the candidates or None."""
    for i in range(1, len(MANIFEST["terms"]) + 1):
        candidates = _candidates(i)
        term = grounding_from_reply(_reply(model, i), candidates)
        if term is not None:
            assert term.id in {c.id for c in candidates}


@pytest.mark.parametrize("model", MODELS)
def test_the_model_grounded_each_term_to_the_right_kind_of_ontology(model) -> None:
    """The three fixture terms have clear groundings; MODEL_DEPENDENT."""
    expected = {"Marfan syndrome": "MONDO:", "hyperandrogenism": "HP:", "sebaceous gland": "UBERON:"}
    for i, term in enumerate(MANIFEST["terms"], 1):
        prefix = expected[term["label"]]
        grounded = grounding_from_reply(_reply(model, i), _candidates(i))
        assert grounded is not None, MODEL_DEPENDENT + f"{term['label']} was grounded"
        assert grounded.id.startswith(prefix), MODEL_DEPENDENT + f"{term['label']} -> {prefix}"
