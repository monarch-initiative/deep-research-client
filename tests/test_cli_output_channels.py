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
