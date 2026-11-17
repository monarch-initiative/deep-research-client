"""Aligning two claim sets with a real model (issue #43, item 2).

The worked example the issue asks for: a deep research report's claims against a
curated structured source's claims. Runs through the local Claude Code CLI, so a
logged-in `claude` needs no API key. Marked ``llm`` and excluded by default.
"""

import json
import shutil
from pathlib import Path

import pytest

from deep_research_client.claims import (
    ClaimSet,
    SemanticRelation,
    aalign_claim_sets,
    acompare_claims,
    summarize_alignment,
)

ALIGN = Path(__file__).parent / "input" / "claims" / "alignment"

pytestmark = pytest.mark.llm


def _needs_claude() -> None:
    if shutil.which("claude") is None:
        pytest.skip("the `claude` CLI is not on PATH")


def _set(name: str) -> ClaimSet:
    return ClaimSet.model_validate(json.loads((ALIGN / name).read_text("utf-8")))


async def test_a_report_aligns_against_a_curated_source() -> None:
    """Report-vs-structured alignment classifies each candidate on both axes."""
    _needs_claude()
    from deep_research_client.claude_code_chat import DEFAULT_CLAUDE_CODE_MODEL, ClaudeCodeChatClient

    set_a = _set("report_a.claims.json")
    set_b = _set("gold_b.claims.json")
    alignment = await aalign_claim_sets(
        set_a, set_b, llm_client=ClaudeCodeChatClient(), model=DEFAULT_CLAUDE_CODE_MODEL,
    )

    # The three fixture candidate pairs are classified, each with a derived axis.
    # (The generated base sets use_enum_values, so slots read back as strings.)
    assert len(alignment.pair_list) == 3
    valid = {r.value for r in SemanticRelation}
    for pair in alignment.pair_list:
        assert pair.semantic_relation in valid
        assert pair.evidential_relation is not None
    assert alignment.aligner.prompt_version

    # The FBN1 causal claim is the same on both sides.
    fbn1 = next(p for p in alignment.pair_list if p.claim_a_id == "c1")
    assert fbn1.semantic_relation == SemanticRelation.IDENTICAL

    # Rubrics aggregate without another model call.
    summary = summarize_alignment(alignment, set_a, set_b, reference="b")
    completeness = summary.metric("completeness")
    assert completeness is not None and completeness.value is not None


async def test_compare_claims_classifies_a_single_pair() -> None:
    """The one-pair interface returns a classified pair with the derived axes."""
    _needs_claude()
    from deep_research_client.claude_code_chat import DEFAULT_CLAUDE_CODE_MODEL, ClaudeCodeChatClient

    set_a = _set("report_a.claims.json")
    set_b = _set("gold_b.claims.json")
    a = next(c for c in set_a.claim_list if c.id == "c1")
    b = next(c for c in set_b.claim_list if c.id == "genetic_factors[0]")

    pair = await acompare_claims(a, b, llm_client=ClaudeCodeChatClient(), model=DEFAULT_CLAUDE_CODE_MODEL)
    assert pair.semantic_relation == SemanticRelation.IDENTICAL
    assert pair.candidate_signals is None  # nothing proposed it; it was asked directly
