"""The enriched rubric end to end, with a real model, OAK and the reference API.

Runs the whole chain on the committed report/reference fixtures: ground the
candidate's entities (OAK + model), verify its citations (resolve against the
source), align it against the curated reference, and fold all of that into one
RubricSummary. This is the live counterpart to the offline test in
test_rubric_dismech.py. Marked ``llm`` and ``integration``: it needs a logged-in
`claude`, the `terms` and `validation` extras, and the network, so it is excluded
from the default suite (run with `just test-full`).
"""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("linkml_term_validator")  # the `terms` extra (OAK grounding)
pytest.importorskip("linkml_reference_validator")  # the `validation` extra (citations)

from deep_research_client.claims import (  # noqa: E402 - after importorskip
    CitationVerifier,
    ClaimSet,
    OakGrounder,
    aalign_claim_sets,
    aground_claims,
    averify_claims,
    biomedical_ontology_config,
    report_topic_text,
    summarize_alignment,
)

ALIGN = Path(__file__).parent / "input" / "claims" / "alignment"

pytestmark = [pytest.mark.llm, pytest.mark.integration]


def _set(name: str) -> ClaimSet:
    return ClaimSet.model_validate(json.loads((ALIGN / name).read_text("utf-8")))


async def test_the_enriched_rubric_folds_in_grounding_and_citations_live() -> None:
    """A report grounded, citation-checked and aligned, summarised in one rubric."""
    if shutil.which("claude") is None:
        pytest.skip("the `claude` CLI is not on PATH")
    from deep_research_client.claude_code_chat import DEFAULT_CLAUDE_CODE_MODEL, ClaudeCodeChatClient

    client = ClaudeCodeChatClient()
    candidate = _set("report_a.claims.json")  # a report, with entities and a PMID citation
    reference = _set("gold_b.claims.json")

    # Ground the candidate's entities (fills EntityMention.id, sets the grounder stamp).
    # await the async form: this test already runs in an event loop, so the sync
    # ground_claims (which calls asyncio.run) could not nest inside it.
    grounded = await aground_claims(
        candidate, OakGrounder(biomedical_ontology_config(), llm_client=client,
                               model=DEFAULT_CLAUDE_CODE_MODEL),
    )
    assert grounded.grounder is not None  # grounding ran

    # Verify the grounded candidate's citations against their sources.
    verification = await averify_claims(
        grounded, [CitationVerifier(topic_text=report_topic_text(grounded))],
    )

    # Align the candidate against the curated reference.
    alignment = await aalign_claim_sets(
        grounded, reference, llm_client=client, model=DEFAULT_CLAUDE_CODE_MODEL,
    )

    summary = summarize_alignment(
        alignment, grounded, reference, reference="b", verification=verification,
    )
    names = {m.name for m in summary.metrics}
    # The five alignment metrics, plus all three folded metrics, since the candidate
    # was grounded (it has entity mentions) and its citations were checked.
    assert {"relevance", "informativeness", "support", "completeness", "citation_quality"} <= names
    assert "grounding" in names
    assert "citation_accuracy" in names
    assert "review_burden" in names

    # Every metric reads back consistently (RubricSummary validates value == num/den).
    for name in ("grounding", "citation_accuracy", "review_burden"):
        metric = summary.metric(name)
        assert metric is not None and metric.denominator >= 1
