"""Tests for CLI behaviors."""

from typer.testing import CliRunner

from deep_research_client.cli import app


runner = CliRunner(mix_stderr=False)


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
        ["research", "Inline query", "--input-file", str(query_file), "--provider", "mock"],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )

    assert result.exit_code == 1
    combined_output = result.stdout + result.stderr
    assert "Provide the query either as an argument or via --input-file" in combined_output
