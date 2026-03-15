"""Tests for CLI behaviors."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app, _collect_noop_research_option_warnings


runner = CliRunner(mix_stderr=False)


def _load_asta_api_key() -> str:
    """Load the Asta API key from env or the repo-local helper file."""
    key = Path(__file__).resolve().parents[1] / "asta_key"
    if key.exists():
        return key.read_text(encoding="utf-8").strip()
    pytest.skip("asta_key file not present")


def test_research_uses_input_file(tmp_path):
    """CLI should read the query directly from a file when requested."""
    query_file = tmp_path / "query.md"
    query_file.write_text("What is synthetic biology?", encoding="utf-8")

    result = runner.invoke(
        app,
        ["research", "--input-file", str(query_file), "--provider", "mock"],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )

    assert result.exit_code == 0, result.stderr
    assert "What is synthetic biology?" in result.stdout


def test_research_rejects_conflicting_query_sources(tmp_path):
    """Passing both inline query and file should fail fast with a clear error."""
    query_file = tmp_path / "conflict.md"
    query_file.write_text("File-based query", encoding="utf-8")

    result = runner.invoke(
        app,
        ["research", "Inline query", "--input-file",
            str(query_file), "--provider", "mock"],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )

    assert result.exit_code == 1
    combined_output = result.stdout + result.stderr
    assert "Provide the query either as an argument or via --input-file" in combined_output


def test_providers_asta_shows_required_credentials():
    """CLI should explain that Asta needs its retrieval credential."""
    result = runner.invoke(
        app,
        ["providers", "--provider", "asta"],
        env={},
    )

    assert result.exit_code == 0, result.stderr
    assert "Provider: asta - Not available" in result.stdout
    assert "ASTA_API_KEY" in result.stdout


def test_collect_noop_research_option_warnings_for_asta():
    """Asta should warn on CLI options that do not affect retrieval."""
    warnings = _collect_noop_research_option_warnings(
        "asta",
        model="custom-model",
        base_url="https://example.org",
        use_cborg=True,
        api_key_env="CUSTOM_KEY",
    )

    assert len(warnings) == 4
    assert any("--model" in warning for warning in warnings)
    assert any("--base-url" in warning for warning in warnings)
    assert any("--use-cborg" in warning for warning in warnings)
    assert any("--api-key-env" in warning for warning in warnings)


@pytest.mark.integration
def test_research_asta_warns_on_noop_model_and_writes_separate_citations(tmp_path):
    """Asta CLI should warn on --model but still honor output formatting options."""
    output_path = tmp_path / "asta-report.md"
    citations_path = tmp_path / "asta-citations.md"

    result = runner.invoke(
        app,
        [
            "research",
            "hyperlipidemia",
            "--provider",
            "asta",
            "--model",
            "ignored-model",
            "--output",
            str(output_path),
            "--separate-citations",
            str(citations_path),
        ],
        env={"ASTA_API_KEY": _load_asta_api_key()},
    )

    assert result.exit_code == 0, result.stderr
    combined_output = result.stdout + result.stderr
    assert "ignores --model" in combined_output
    assert output_path.exists()
    assert citations_path.exists()
    assert "## Citations" not in output_path.read_text(encoding="utf-8")
    assert "# Citations for Research Query" in citations_path.read_text(
        encoding="utf-8")
