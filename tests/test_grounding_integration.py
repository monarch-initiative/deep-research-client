"""Grounding a claim set with real OAK ontology search and a real model.

Needs the `terms` extra, network (OLS), and a logged-in `claude`. Marked `llm`
and `integration`, so excluded by default.
"""

import shutil

import pytest

from deep_research_client.claims import (
    Claim,
    ClaimSet,
    EntityMention,
    ExtractorInfo,
    OakGrounder,
    SourceDocument,
    SourceType,
    aground_claims,
    biomedical_ontology_config,
)

pytestmark = [pytest.mark.llm, pytest.mark.integration]


def _needs_claude() -> None:
    if shutil.which("claude") is None:
        pytest.skip("the `claude` CLI is not on PATH")


async def test_grounds_a_disease_and_a_phenotype() -> None:
    """A disease label gets a MONDO id and a phenotype label an HP id."""
    _needs_claude()
    from deep_research_client.claude_code_chat import DEFAULT_CLAUDE_CODE_MODEL, ClaudeCodeChatClient

    def claim(cid, label, text):
        return Claim(id=cid, claim_text=text, anchor_status="EXACT", citation_status="UNCITED",
                     about="DOMAIN", entities=[EntityMention(label=label)])

    claim_set = ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="x"),
        claims=[
            claim("c1", "Marfan syndrome", "Marfan syndrome is caused by variants in FBN1."),
            claim("c2", "hyperandrogenism", "Hyperandrogenism occurs in adult female acne."),
        ],
    )
    grounder = OakGrounder(
        biomedical_ontology_config(),
        llm_client=ClaudeCodeChatClient(), model=DEFAULT_CLAUDE_CODE_MODEL,
    )
    grounded = await aground_claims(claim_set, grounder)

    by_label = {e.label: e.id for c in grounded.claim_list for e in (c.entities or [])}
    assert (by_label["Marfan syndrome"] or "").startswith("MONDO:")
    assert (by_label["hyperandrogenism"] or "").startswith("HP:")
    info = grounded.grounder
    assert info is not None and info.name == "oak-llm"
    assert (info.grounded or 0) >= 1
