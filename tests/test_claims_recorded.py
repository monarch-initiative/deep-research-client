"""Real model replies to the claim-extraction prompt, replayed through the parser.

The replies in tests/input/claims/recorded/ were written by models answering
exactly the prompts this code builds (see manifest.yaml). Each is passed
straight to ``claims_from_reply`` with the unit it answered; no client is
involved, so what is tested is the real parser on real model output, not a
model stand-in.

The first test keeps the recording honest: if the prompt, its version, or the
units change, the recorded replies no longer answer what the code would ask,
and it fails until they are re-recorded.
"""

from pathlib import Path

import pytest
import yaml

from deep_research_client.claims import AnchorStatus
from deep_research_client.claims.llm import PROMPT_VERSION, build_prompt
from deep_research_client.claims.parsing import TextUnit, citation_window, claims_from_reply
from deep_research_client.claims.units import markdown_units, structured_units

INPUT = Path(__file__).parent / "input" / "claims"
RECORDED = INPUT / "recorded"
MANIFEST = yaml.safe_load((RECORDED / "manifest.yaml").read_text(encoding="utf-8"))
MODELS = sorted(MANIFEST["recordings"])


def _units() -> list[tuple[str, TextUnit]]:
    """The units the recording was made from, named as their prompt files are."""
    report = (INPUT / MANIFEST["sources"]["report"]).read_text(encoding="utf-8")
    notes = yaml.safe_load((INPUT / MANIFEST["sources"]["notes"]).read_text(encoding="utf-8"))
    units = [("report", u) for u in markdown_units(report)]
    units += [("notes", u) for u in structured_units(notes)]
    return [(f"{i:02d}-{kind}.txt", unit) for i, (kind, unit) in enumerate(units, 1)]


def test_the_recording_answers_the_prompts_this_code_sends():
    """Re-record (see manifest.yaml) whenever this fails."""
    assert MANIFEST["prompt_version"] == PROMPT_VERSION
    recorded = sorted(p.name for p in (RECORDED / "prompts").glob("[0-9][0-9]-*.txt"))
    units = _units()
    assert recorded == [name for name, _ in units]
    for name, unit in units:
        system, user = build_prompt(unit)
        assert (RECORDED / "prompts" / "SYSTEM.txt").read_text(encoding="utf-8") == system["content"]
        assert (RECORDED / "prompts" / name).read_text(encoding="utf-8") == user["content"], name


@pytest.mark.parametrize("model", MODELS)
def test_every_recorded_claim_anchors_and_its_span_reads_as_the_source(model):
    """Real replies anchor, and every span is the source's own text."""
    total = 0
    for name, unit in _units():
        claims = claims_from_reply((RECORDED / model / name).read_text(encoding="utf-8"), unit)
        assert claims, f"{model} {name}: no claims parsed"
        for claim in claims:
            total += 1
            assert claim.anchor_status != AnchorStatus.UNANCHORED, (model, name, claim.claim_text)
            span = claim.source_span
            assert unit.text[span.start:span.end] == span.text
            assert unit.start <= span.start < span.end <= unit.end, "the span stays in its unit"
            assert claim.section == unit.section and claim.source_path == unit.source_path
    assert total >= 9


@pytest.mark.parametrize("model", MODELS)
def test_recorded_citations_resolve_to_the_bibliography_entries(model):
    """Every numbered marker kept is in its claim's sentence and resolves."""
    expected = {
        "[1]": "DOI:10.1038/352337a0",
        "[2]": "PMID:15731757",
        "[3]": "PMID:8166794",
        "https://doi.org/10.1136/jmg.2009.072785": "DOI:10.1136/jmg.2009.072785",
    }
    seen = {}
    for name, unit in _units():
        for claim in claims_from_reply((RECORDED / model / name).read_text(encoding="utf-8"), unit):
            for citation in claim.citations or []:
                assert citation.marker in citation_window(unit, claim.source_span), claim.claim_text
                seen[citation.marker] = citation.reference_id
    assert seen == expected


@pytest.mark.parametrize("model", MODELS)
def test_the_recorded_negation_is_kept(model):
    """"It is not caused by variants in TGFBR2" is a negated claim, not a positive one."""
    name, unit = _units()[0]
    claims = claims_from_reply((RECORDED / model / name).read_text(encoding="utf-8"), unit)

    tgfbr2 = [c for c in claims if "TGFBR2" in c.claim_text and "Marfan" in c.claim_text]

    assert [c.negated for c in tgfbr2] == [True]


@pytest.mark.parametrize("model", MODELS)
def test_recorded_claims_from_structured_prose_keep_their_field(model):
    """Claims from a YAML prose leaf say which field they came from."""
    paths = set()
    for name, unit in _units():
        if name.endswith("-notes.txt"):
            reply = (RECORDED / model / name).read_text(encoding="utf-8")
            paths |= {c.source_path for c in claims_from_reply(reply, unit)}
    assert paths == {"sections[0].text", "sections[1].text"}
