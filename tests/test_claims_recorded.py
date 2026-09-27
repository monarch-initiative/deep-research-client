"""Real model replies to the claim-extraction prompt, replayed through the parser.

The replies in tests/input/claims/recorded/ were written by models answering
exactly the prompts this code builds (see manifest.yaml). Each is passed
straight to ``claims_from_reply`` with the unit it answered; no client is
involved, so what is tested is the real parser on real model output, not a
model stand-in.

The first test keeps the recording honest: if the prompt, its version, or the
units change, the recorded replies no longer answer what the code would ask,
and it fails until they are re-recorded.

Most assertions are about the code: whatever the model chose to extract, a
claim that anchors reads as the source, and a citation that is kept resolves
correctly. A few also depend on what the model chose, for example that it
reported a negation. Those carry MODEL_DEPENDENT in their failure message, so
after a re-recording a failure there is read as a change in the model's
output, not as a parser regression.
"""

from functools import cache
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

#: Prefix for assertions that test the recorded model's choices, not the code.
MODEL_DEPENDENT = "MODEL_DEPENDENT (re-check after re-recording): "


@cache
def _units() -> tuple[tuple[str, TextUnit], ...]:
    """The units the recording was made from, named as their prompt files are."""
    units = []
    for source in MANIFEST["sources"]:
        path = INPUT / source["file"]
        text = path.read_text(encoding="utf-8")
        built = markdown_units(text) if path.suffix == ".md" else structured_units(yaml.safe_load(text))
        units += [(source["name"], unit) for unit in built]
    return tuple((f"{i:02d}-{name}.txt", unit) for i, (name, unit) in enumerate(units, 1))


def _source(name: str) -> str:
    """The manifest name of the source a prompt file's unit came from: "06-provider.txt" is "provider"."""
    return name.split("-", 1)[1].removesuffix(".txt")


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
    for name, unit in _units():
        claims = claims_from_reply((RECORDED / model / name).read_text(encoding="utf-8"), unit)
        assert claims, f"{MODEL_DEPENDENT}{model} reported no claims for {name}"
        for claim in claims:
            assert claim.anchor_status != AnchorStatus.UNANCHORED, (
                f"{MODEL_DEPENDENT}{model} {name}: quote not in the source: {claim.claim_text}"
            )
            span = claim.source_span
            assert unit.text[span.start:span.end] == span.text
            assert unit.start <= span.start < span.end <= unit.end, "the span stays in its unit"
            assert claim.section == unit.section and claim.source_path == unit.source_path


@pytest.mark.parametrize("model", MODELS)
def test_recorded_citations_resolve_to_the_bibliography_entries(model):
    """Every marker kept is in its claim's sentence and resolves to its own entry.

    Which markers a model attaches is its choice, so this checks only that
    those kept resolve correctly, not that every one was cited.
    """
    expected = {
        ("report", "[1]"): "DOI:10.1038/352337a0",
        ("report", "[2]"): "PMID:15731757",
        ("report", "[3]"): "PMID:8166794",
        ("report", "https://doi.org/10.1136/jmg.2009.072785"): "DOI:10.1136/jmg.2009.072785",
        (
            "report", "[the 2010 Ghent criteria](https://doi.org/10.1136/jmg.2009.072785)",
        ): "DOI:10.1136/jmg.2009.072785",
    }
    seen: dict[tuple[str, str], str | None] = {}
    for name, unit in _units():
        for claim in claims_from_reply((RECORDED / model / name).read_text(encoding="utf-8"), unit):
            for citation in claim.citations or []:
                assert citation.marker in citation_window(unit, claim.source_span), claim.claim_text
                # One marker must resolve the same way in every claim citing it.
                key = (_source(name), citation.marker)
                assert seen.setdefault(key, citation.reference_id) == citation.reference_id
    assert seen.items() <= expected.items()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "prompt_file,words",
    [("01-report.txt", ("TGFBR2", "Marfan"))],
    ids=["not-caused-by-TGFBR2"],
)
def test_the_recorded_negation_is_kept(model, prompt_file, words):
    """A sentence saying a relationship does not hold gives a negated claim, not a positive one."""
    unit = dict(_units())[prompt_file]
    claims = claims_from_reply((RECORDED / model / prompt_file).read_text(encoding="utf-8"), unit)

    matching = [c for c in claims if all(w.lower() in c.claim_text.lower() for w in words)]

    assert [c.negated for c in matching] == [True], (
        f"{MODEL_DEPENDENT}{model} did not report one negated claim about {' and '.join(words)}"
    )


@pytest.mark.parametrize("model", MODELS)
def test_recorded_claims_from_structured_prose_keep_their_field(model):
    """Claims from a YAML prose leaf say which field they came from."""
    paths = set()
    for name, unit in _units():
        if name.endswith("-notes.txt"):
            reply = (RECORDED / model / name).read_text(encoding="utf-8")
            paths |= {c.source_path for c in claims_from_reply(reply, unit)}
    assert paths == {"sections[0].text", "sections[1].text"}, (
        f"{MODEL_DEPENDENT}{model} left a notes field with no claims"
    )
