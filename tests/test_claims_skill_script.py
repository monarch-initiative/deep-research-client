"""The claim-set summariser shipped with the review-claim-set skill.

It lives in .claude/skills/, outside the package, so it is loaded from its
path. It is run on a real curated extraction, and on a report claim set built
by the real parser from a recorded-style reply.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from deep_research_client.claims import extract_claims
from deep_research_client.claims.models import (
    ClaimSet,
    ExtractorInfo,
    SourceDocument,
    SourceType,
    content_sha256,
)
from deep_research_client.claims.parsing import TextUnit, claims_from_reply

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "tests" / "input" / "claims"
SCRIPT = REPO / ".claude" / "skills" / "review-claim-set" / "scripts" / "summarize_claims.py"


@pytest.fixture(scope="module")
def summarizer() -> ModuleType:
    """The script, imported from its path."""
    spec = importlib.util.spec_from_file_location("summarize_claims", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report_set(text: str) -> ClaimSet:
    """A report claim set with one anchored claim and one misquoted one."""
    unit = TextUnit(text=text, start=0, end=len(text), section="Genetics")
    reply = json.dumps({"claims": [
        {"claim": "FBN1 variants cause Marfan syndrome.", "about": "domain",
         "quote": "caused by pathogenic variants in FBN1", "basis": "background_knowledge"},
        {"claim": "Marfan syndrome is recessive.", "about": "domain",
         "quote": "Marfan syndrome is an autosomal recessive disorder caused by pathogenic variants in FBN1"},
    ]})
    return ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT,
                              content_sha256=content_sha256(text)),
        extractor=ExtractorInfo(name="llm-atomic", model="claude-sonnet-5", prompt_version="5"),
        claims=claims_from_reply(reply, unit),
    )


def test_a_curated_set_is_summarised_with_its_record_citations(summarizer, tmp_path):
    """Counts come from the set, and a structured source has no spans to check."""
    path = tmp_path / "dismech.claims.json"
    path.write_text(extract_claims(INPUT / "marfan_dismech.yaml").model_dump_json(), encoding="utf-8")

    summary = summarizer.summarize(
        summarizer.load(path), (INPUT / "marfan_dismech.yaml").read_text(encoding="utf-8"),
    )

    assert "- Claims: 5" in summary
    assert "| CITED | 3 |" in summary and "| UNCITED | 2 |" in summary
    assert "| RECORD | 3 |" in summary
    assert "no spans to check" in summary


def test_a_report_set_shows_its_gaps_and_where_to_look(summarizer):
    """The unanchored claim shows its nearest passage; the uncited domain claim is listed."""
    text = "Marfan syndrome is an autosomal dominant disorder caused by pathogenic variants in FBN1."

    summary = summarizer.summarize(_report_set(text), text)

    assert "- Extractor: llm-atomic, model claude-sonnet-5, prompt v5" in summary
    assert "0 spans differ from the source text" in summary
    assert "## Unanchored claims (1)" in summary
    assert 'nearest passage, score' in summary and "autosomal dominant disorder" in summary
    assert "## Uncited domain claims (1)\n\nu1 in Genetics" in summary


def test_a_changed_source_is_reported_before_any_span_is_trusted(summarizer):
    """If the source's hash differs, offsets are not checked at all."""
    text = "Marfan syndrome is an autosomal dominant disorder caused by pathogenic variants in FBN1."

    summary = summarizer.summarize(_report_set(text), text + " Edited.")

    assert "The source has changed since extraction" in summary
    assert "spans differ" not in summary


def test_a_span_that_no_longer_reads_as_the_source_is_named(summarizer):
    """A matching hash with a moved span is a bug, so the reader gets the ids to report."""
    text = "Marfan syndrome is an autosomal dominant disorder caused by pathogenic variants in FBN1."
    claims = _report_set(text)
    first = claims.claim_list[0]
    moved = first.model_copy(update={"source_span": first.source_span.model_copy(
        update={"start": first.source_span.start + 1, "end": first.source_span.end + 1})})
    claims = claims.model_copy(update={"claims": [moved, *claims.claim_list[1:]]})

    summary = summarizer.summarize(claims, text)

    assert "1 spans differ from the source text" in summary
    assert "Report these:\n  u1 in Genetics" in summary
