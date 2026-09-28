"""The `claims extract` command (issue #43).

Curated sources run end to end with no model. For prose sources, only the
refusals without a key or backend are tested here; the model path is covered under the
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


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--llm-backend", "gemini"], "Unknown --llm-backend 'gemini'"),
        (["--llm-backend", "claude-code", "--llm-base-url", "http://localhost:8000"],
         "--llm-base-url is for the openai backend"),
    ],
)
def test_a_backend_that_cannot_run_is_refused_before_any_call(args, message):
    """A wrong backend choice is named on stderr, not discovered mid-extraction."""
    result = runner.invoke(app, ["claims", "extract", str(INPUT / "marfan_report.md"), *args])

    assert result.exit_code == 1
    assert message in result.stderr
    assert result.stdout == ""


def test_the_claude_code_backend_needs_the_cli_on_path(monkeypatch):
    """Without `claude` on PATH, the claude-code backend says so."""
    monkeypatch.setenv("PATH", "")

    result = runner.invoke(
        app, ["claims", "extract", str(INPUT / "marfan_report.md"), "--llm-backend", "claude-code"]
    )

    assert result.exit_code == 1
    assert "needs the `claude` CLI on PATH" in result.stderr


def test_a_missing_citations_file_is_refused(tmp_path):
    """An explicit --citations that is not there is named, not ignored."""
    result = runner.invoke(app, ["claims", "extract", str(INPUT / "marfan_report.md"),
                                 "--citations", str(tmp_path / "absent.citations.md")])

    assert result.exit_code == 1
    assert "Citations file not found" in result.stderr


def test_a_citations_file_for_a_curated_source_is_refused(tmp_path):
    """Curated files have no [n] markers for a citations file to resolve."""
    side = tmp_path / "side.citations.md"
    side.write_text("1. https://doi.org/10.1/x\n", encoding="utf-8")

    result = runner.invoke(app, ["claims", "extract", str(INPUT / "marfan_dismech.yaml"), "--citations", str(side)])

    assert result.exit_code == 1
    assert "dismech sources have none" in result.stderr
