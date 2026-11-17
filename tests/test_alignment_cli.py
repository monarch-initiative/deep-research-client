"""The `claims align` command: input validation and the refusals before any call.

The model path is covered end to end under the ``llm`` marker in
test_alignment_integration.py; here only the paths that need no model run.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app

ALIGN = Path(__file__).parent / "input" / "claims" / "alignment"
A = str(ALIGN / "report_a.claims.json")
B = str(ALIGN / "gold_b.claims.json")
runner = CliRunner()


def test_a_missing_input_is_named(tmp_path) -> None:
    result = runner.invoke(app, ["claims", "align", str(tmp_path / "nope.json"), B])
    assert result.exit_code == 1
    assert "File not found" in result.stderr


def test_a_file_that_is_not_a_claim_set_is_refused(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a claim set"}', encoding="utf-8")
    result = runner.invoke(app, ["claims", "align", str(bad), B])
    assert result.exit_code == 1
    assert "is not a valid claim set" in result.stderr


def test_an_unknown_reference_is_refused() -> None:
    result = runner.invoke(app, ["claims", "align", A, B, "--reference", "gold"])
    assert result.exit_code == 1
    assert "--reference must be a or b" in result.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--llm-backend", "gemini"], "Unknown --llm-backend 'gemini'"),
        (["--llm-backend", "claude-code", "--llm-base-url", "http://localhost:8000"],
         "--llm-base-url is for the openai backend"),
    ],
)
def test_a_backend_that_cannot_run_is_refused_before_any_call(args, message) -> None:
    result = runner.invoke(app, ["claims", "align", A, B, *args])
    assert result.exit_code == 1
    assert message in result.stderr


def test_rubric_is_refused_for_a_within_report_alignment() -> None:
    """The rubric compares two sources; it makes no sense aligning a set with itself."""
    result = runner.invoke(app, ["claims", "align", A, "--rubric"])
    assert result.exit_code == 1
    assert "--rubric compares two sources" in result.stderr


def test_without_a_key_the_reason_names_alignment(monkeypatch) -> None:
    """The no-key message says a model is needed to align, and how to provide one."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(app, ["claims", "align", A, B])
    assert result.exit_code == 1
    assert "OPENAI_API_KEY is not set, and claims are aligned by a model" in result.stderr
    assert "--llm-backend claude-code" in result.stderr
