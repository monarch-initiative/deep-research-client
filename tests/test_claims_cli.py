"""The `claims extract` command (issue #43).

Curated sources run end to end with no model. For prose sources, only the
refusal without a key is tested here; the model path is covered under the
``llm`` marker in test_claims_extract.py.
"""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from deep_research_client.claims import ClaimSet
from deep_research_client.cli import app

INPUT = Path(__file__).parent / "input" / "claims"
runner = CliRunner()


def test_the_claim_set_is_the_product_on_stdout():
    """Only the JSON claim set is on stdout, so it can be piped or redirected."""
    result = runner.invoke(app, ["claims", "extract", str(INPUT / "marfan_dismech.yaml")])

    assert result.exit_code == 0, result.output
    claims = ClaimSet.model_validate(json.loads(result.stdout))
    assert claims.extractor.name == "dismech"
    assert len(claims.claim_list) == 5


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_output_is_written_in_the_format_its_suffix_names(tmp_path, suffix):
    """A written claim set reads back as the same set."""
    output = tmp_path / f"fbn1.claims{suffix}"

    result = runner.invoke(
        app, ["claims", "extract", str(INPUT / "fbn1_gene_review.yaml"), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    text = output.read_text(encoding="utf-8")
    loaded = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    assert [c["source_path"] for c in ClaimSet.model_validate(loaded).model_dump()["claims"]] == [
        "existing_annotations[0]", "core_functions[0]",
    ]


def test_a_prose_source_without_a_key_is_refused_on_stderr(monkeypatch):
    """Markdown needs a model; saying how to provide one beats a traceback."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(app, ["claims", "extract", str(INPUT / "marfan_report.md")])

    assert result.exit_code == 1
    assert "Error: OPENAI_API_KEY is not set" in result.stderr
    assert "--llm-base-url" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "args,message",
    [
        (["claims", "extract", "absent.md"], "Error: File not found: absent.md"),
        (
            ["claims", "extract", str(INPUT / "marfan_dismech.yaml"), "--format", "pdf"],
            "Use one of: auto, markdown, dismech, gene-review, structured",
        ),
    ],
    ids=["missing-file", "unknown-format"],
)
def test_bad_input_is_an_error_on_stderr(args, message):
    """An error about the run, with nothing on stdout."""
    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert message in result.stderr
    assert result.stdout == ""


def test_a_malformed_json_source_is_reported_as_unparseable(tmp_path):
    """A syntax error in the file is not a bad --format."""
    source = tmp_path / "notes.json"
    source.write_text('{"name": "x",', encoding="utf-8")

    result = runner.invoke(app, ["claims", "extract", str(source)])

    assert result.exit_code == 1
    assert f"Error: Could not parse {source}" in result.stderr
    assert "Use one of" not in result.stderr
    assert result.stdout == ""
