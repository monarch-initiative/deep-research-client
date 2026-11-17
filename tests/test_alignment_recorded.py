"""Real model replies to the relation prompt, replayed through the parser.

The replies in tests/input/claims/alignment/<model>/ were written by a model
answering exactly the prompts this code builds for the candidate pairs of the two
fixture claim sets (see manifest.yaml). Each is passed straight to
``relation_from_reply``; no client is involved, so what is tested is the real
parser on real model output, not a model stand-in.

The first test keeps the recording honest: if the prompt, its version, or the
fixtures change, the recorded replies no longer answer what the code would ask,
and it fails until they are re-recorded (run record.py).

A few assertions depend on what the model chose; they carry MODEL_DEPENDENT in
the failure message, so after a re-recording a failure there reads as a change in
the model's output, not a parser regression.
"""

import json
from functools import cache
from pathlib import Path

import pytest
import yaml

from deep_research_client.claims import (
    ClaimSet,
    EvidentialRelation,
    SemanticRelation,
    candidate_pairs,
    evidential_relation,
    shares_citation,
)
from deep_research_client.claims.align_llm import ALIGN_PROMPT_VERSION, build_align_prompt
from deep_research_client.claims.align_parsing import relation_from_reply

ALIGN = Path(__file__).parent / "input" / "claims" / "alignment"
MANIFEST = yaml.safe_load((ALIGN / "manifest.yaml").read_text(encoding="utf-8"))
MODELS = sorted(MANIFEST["recordings"])

MODEL_DEPENDENT = "MODEL_DEPENDENT (re-check after re-recording): "


@cache
def _pairs():
    """The candidate pairs the recording was made from, in file order (01, 02, ...)."""
    set_a = ClaimSet.model_validate(json.loads((ALIGN / MANIFEST["sources"]["a"]).read_text("utf-8")))
    set_b = ClaimSet.model_validate(json.loads((ALIGN / MANIFEST["sources"]["b"]).read_text("utf-8")))
    return candidate_pairs(set_a.claim_list, set_b.claim_list)


def _reply(model: str, index: int) -> str:
    return (ALIGN / model / f"{index:02d}.txt").read_text(encoding="utf-8")


def test_the_recording_answers_the_prompts_this_code_sends() -> None:
    """Re-record (run record.py) whenever this fails."""
    assert MANIFEST["prompt_version"] == ALIGN_PROMPT_VERSION
    pairs = _pairs()
    recorded = sorted(p.name for p in (ALIGN / "prompts").glob("[0-9][0-9].txt"))
    assert recorded == [f"{i:02d}.txt" for i in range(1, len(pairs) + 1)]
    system_file = (ALIGN / "prompts" / "SYSTEM.txt").read_text(encoding="utf-8")
    for i, pair in enumerate(pairs, 1):
        system, user = build_align_prompt(pair.claim_a, pair.claim_b)
        assert system["content"] == system_file
        assert user["content"] == (ALIGN / "prompts" / f"{i:02d}.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("model", MODELS)
def test_every_recorded_reply_parses_to_a_known_relation(model) -> None:
    """Whatever the model said, the parser reads one relation per pair."""
    for i in range(1, len(_pairs()) + 1):
        verdict = relation_from_reply(_reply(model, i))
        assert isinstance(verdict.semantic_relation, SemanticRelation)
        if verdict.confidence is not None:
            assert 0.0 <= verdict.confidence <= 1.0


@pytest.mark.parametrize("model", MODELS)
def test_the_derived_evidential_axis_holds_for_the_recorded_pairs(model) -> None:
    """Independent of the model, each pair's evidential relation is fixed by the claims."""
    pairs = _pairs()
    by_ids = {(p.claim_a.id, p.claim_b.id): p for p in pairs}
    # c1 (background) vs the curated FBN1 record (secondary): A_BACKGROUND_B_SECONDARY.
    fbn1 = by_ids[("c1", "genetic_factors[0]")]
    assert evidential_relation(fbn1.claim_a, fbn1.claim_b) == EvidentialRelation.A_BACKGROUND_B_SECONDARY
    # c2 vs the prevalence record: both secondary, and both cite PMID:1.
    prev = by_ids[("c2", "phenotypes[1]")]
    assert evidential_relation(prev.claim_a, prev.claim_b) == EvidentialRelation.BOTH_SECONDARY
    assert shares_citation(prev.claim_a, prev.claim_b) is True
    # c3 (observation) vs the curated phenotype (secondary): A_OBSERVATION_B_SECONDARY.
    pheno = by_ids[("c3", "phenotypes[0]")]
    assert evidential_relation(pheno.claim_a, pheno.claim_b) == EvidentialRelation.A_OBSERVATION_B_SECONDARY


@pytest.mark.parametrize("model", MODELS)
def test_the_model_classified_the_pairs_sensibly(model) -> None:
    """The fixtures were built to have clear relations; MODEL_DEPENDENT."""
    pairs = _pairs()
    index = {(p.claim_a.id, p.claim_b.id): i for i, p in enumerate(pairs, 1)}

    def relation(a, b):
        return relation_from_reply(_reply(model, index[(a, b)])).semantic_relation

    # Same causal claim, differing only in voice: identical.
    assert relation("c1", "genetic_factors[0]") == SemanticRelation.IDENTICAL, MODEL_DEPENDENT + "FBN1 pair"
    # 1 in 5000 versus 1 in 3000 for the same condition: a conflict (or a near miss).
    assert relation("c2", "phenotypes[1]") in {
        SemanticRelation.CONTRADICTORY, SemanticRelation.CLOSE,
    }, MODEL_DEPENDENT + "prevalence pair"
    # "the major manifestation" is more specific than "is a phenotype".
    assert relation("c3", "phenotypes[0]") in {
        SemanticRelation.SUBSUMED_BY, SemanticRelation.SUBSUMES,
        SemanticRelation.CLOSE, SemanticRelation.COMPLEMENTARY,
    }, MODEL_DEPENDENT + "aortic root pair"
