"""Which stream each kind of CLI message lands on (issue #68).

The rule, stated beside the helpers in `cli.py`: stdout is what a command
produces, stderr is about the run, the logger is `-v` tracing. These tests pin
the stream, not just the text. The pinned click (8.5.0) keeps
`CliRunner`'s `result.stdout` and `result.stderr` apart, and `capsys` does the
same for a helper called directly, so a message moved to the wrong stream
fails here.

Tests for a single command's channels live beside that command's other tests
where a file for it exists (`test_cli.py`, `test_provider_health_cli.py`).
"""

import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client import cli as cli_module
from deep_research_client.cli import app
from deep_research_client.validation import (
    ReferenceCheck,
    ReferenceStatus,
    ReferenceValidationReport,
    TermCheck,
    TermStatus,
    TermValidationReport,
)

runner = CliRunner()


def test_reference_findings_are_warnings_on_stderr(capsys):
    """A finding is about this run's input, and must not join a printed report."""
    report = ReferenceValidationReport(
        references=[
            ReferenceCheck(
                reference_id="PMID:1", status=ReferenceStatus.NOT_FOUND, message="no such PMID"
            ),
            ReferenceCheck(reference_id="PMID:2", status=ReferenceStatus.VERIFIED),
        ]
    )

    cli_module._echo_validation_summary(report)
    captured = capsys.readouterr()

    assert "Warning: Unresolved reference: PMID:1 (no such PMID)" in captured.err
    assert captured.out == ""


def test_term_findings_are_warnings_on_stderr(capsys):
    """The term summary follows the same rule as the reference one."""
    report = TermValidationReport(
        terms=[
            TermCheck(
                term_id="HP:9999999", prefix="HP", status=TermStatus.NOT_FOUND, message="unknown"
            ),
            TermCheck(term_id="HP:0000001", prefix="HP", status=TermStatus.VERIFIED),
        ]
    )

    cli_module._echo_term_validation_summary(report)
    captured = capsys.readouterr()

    assert "Warning: Unresolved term: HP:9999999 (unknown)" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("command", ["validate-references", "validate-terms"])
def test_validate_commands_report_a_missing_file_on_stderr(command, tmp_path: Path):
    """An input that does not exist ends the run; nothing is produced."""
    missing = tmp_path / "absent.md"

    result = runner.invoke(app, [command, str(missing)])

    assert result.exit_code == 1
    assert f"Error: File not found: {missing}" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize(
    "args,message",
    [
        (["models", "--cost", "priceless"], "Error: Invalid --cost value 'priceless'"),
        (
            ["models", "--provider", "flacon"],
            "Error: Unknown provider, or no model cards for 'flacon'",
        ),
    ],
    ids=["bad-filter", "unknown-provider"],
)
def test_models_rejects_bad_input_on_stderr(args, message):
    """A rejected filter is about the run; the listing on stdout stays empty."""
    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert message in result.stderr
    assert result.stdout == ""


def test_browse_files_warns_per_skipped_source_and_errors_when_none_remain(tmp_path: Path):
    """Each skipped source is a warning; nothing left to browse is an error."""
    not_markdown = tmp_path / "notes.txt"
    not_markdown.write_text("plain text", encoding="utf-8")
    absent = tmp_path / "absent.md"

    result = runner.invoke(
        app, ["browse-files", str(not_markdown), str(absent), "-o", str(tmp_path / "out")]
    )

    assert result.exit_code == 1
    assert f"Warning: Skipping non-markdown file: {not_markdown}" in result.stderr
    assert f"Warning: Source not found, skipping: {absent}" in result.stderr
    assert "Error: No markdown files found" in result.stderr
    assert result.stdout == ""


_HAS_BROWSER_EXTRA = importlib.util.find_spec("linkml_browser") is not None


@pytest.mark.skipif(_HAS_BROWSER_EXTRA, reason="needs the browser extra to be absent")
def test_browse_files_reports_the_missing_extra_and_its_install_lines_on_stderr(tmp_path: Path):
    """The extra is not on PyPI, so this is the path most installs actually take."""
    report = tmp_path / "report.md"
    report.write_text("# A report\n", encoding="utf-8")

    result = runner.invoke(app, ["browse-files", str(report), "-o", str(tmp_path / "out")])

    assert result.exit_code == 1
    lines = result.stderr.splitlines()
    at = next(i for i, line in enumerate(lines) if line.endswith("not installed. Install with:"))
    assert lines[at].startswith("Error: ")
    assert lines[at + 1] == "  pip install deep-research-client[browser]"
    assert result.stdout == ""


@pytest.mark.skipif(not _HAS_BROWSER_EXTRA, reason="the extra check runs first without it")
def test_browse_files_refuses_an_existing_output_directory_on_stderr(tmp_path: Path):
    """The --force hint belongs with the error it resolves."""
    report = tmp_path / "report.md"
    report.write_text("# A report\n", encoding="utf-8")
    existing = tmp_path / "out"
    existing.mkdir()

    result = runner.invoke(app, ["browse-files", str(report), "-o", str(existing)])

    assert result.exit_code == 1
    lines = result.stderr.splitlines()
    at = lines.index(f"Error: Output directory exists: {existing}")
    assert lines[at + 1] == "Use --force to overwrite"
    assert result.stdout == ""
