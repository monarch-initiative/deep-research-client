"""A validator that fails still leaves a record in the saved report (#124).

Failures are real, not simulated: each failing validator is given a cache
directory under a regular file, so creating it raises ``NotADirectoryError``.
The validator that is meant to succeed runs offline: PMIDs are skipped, and
terms are resolved only from an empty label cache. A failing reference validator
never reaches the network either: it raises while creating its cache, before
the first lookup.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.markdown_parser import parse_frontmatter
from deep_research_client.models import ResearchResult
from deep_research_client.processing import ResultFormatter
from deep_research_client.validation import (
    TERM_VALIDATION_SECTION_HEADING,
    VALIDATION_SECTION_HEADING,
    ValidationFailure,
    strip_validation_section,
)

QUERY = "Widget coloration (PMID:12345678) involves GO:0008150."
NOT_COMPLETED = "**Not completed.**"


@pytest.fixture
def unusable_dir(tmp_path: Path) -> Path:
    """A cache path that cannot be created: its parent is a regular file."""
    blocker = tmp_path / "a-file"
    blocker.write_text("", encoding="utf-8")
    return blocker / "cache"


def _research_args(
    tmp_path: Path, unusable_dir: Path, failing: set[str], output: bool = True
) -> list[str]:
    """Arguments for a mock research run where the ``failing`` validators raise."""
    reference_cache = unusable_dir if "reference" in failing else tmp_path / "references"
    term_cache = unusable_dir if "term" in failing else tmp_path / "terms"
    args = [
        "research", QUERY, "--provider", "mock", "--no-cache",
        "--validation-cache-dir", str(reference_cache),
        "--validation-email", "test@example.org",
        "--validate-terms", "--term-cache-dir", str(term_cache), "--term-offline",
    ]
    if "reference" not in failing:
        # Offline: a skipped PMID is reported unverifiable without a lookup. Only
        # here, because the cache directory is first created for a lookup, so a
        # run with nothing to look up would never reach the failure.
        args += ["--validation-skip-prefix", "PMID"]
    if output:
        args += ["--output", str(tmp_path / "report.md")]
    return args


@pytest.mark.parametrize(
    "failing",
    [{"reference"}, {"term"}, {"reference", "term"}],
    ids=["reference-fails", "term-fails", "both-fail"],
)
def test_a_failed_validator_is_recorded_in_the_report(
    failing: set[str], tmp_path: Path, unusable_dir: Path, monkeypatch
) -> None:
    """The failure is in the frontmatter and the body; the run still exits 3."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")

    result = CliRunner().invoke(app, _research_args(tmp_path, unusable_dir, failing))

    assert result.exit_code == 3, result.output
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)
    assert "## Output" in body
    for kind, key, heading, label, count_key in (
        ("reference", "reference_validation", VALIDATION_SECTION_HEADING,
         "Reference validation", "total_references"),
        ("term", "term_validation", TERM_VALIDATION_SECTION_HEADING,
         "Term validation", "total_terms"),
    ):
        block = frontmatter[key]
        assert body.count(heading) == 1
        section = body[body.index(heading):]
        if kind in failing:
            assert block["status"] == "failed"
            assert block["error"].startswith(f"{label} failed: ")
            assert "Not a directory" in block["error"]
            assert block["needs_review"] is True
            assert count_key not in block, "a failure must not carry counts"
            assert NOT_COMPLETED in section.split("\n## ")[0]
            # The same line went to stderr and into the file.
            assert block["error"] in result.stderr
        else:
            assert "status" not in block
            assert count_key in block
            assert NOT_COMPLETED not in section.split("\n## ")[0]


def test_both_failing_still_prints_both_records_to_stdout(
    tmp_path: Path, unusable_dir: Path, monkeypatch
) -> None:
    """Without --output the records are printed after the report."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")

    result = CliRunner().invoke(
        app, _research_args(tmp_path, unusable_dir, {"reference", "term"}, output=False)
    )

    assert result.exit_code == 3, result.output
    assert result.stdout.count(NOT_COMPLETED) == 2
    assert VALIDATION_SECTION_HEADING in result.stdout
    assert TERM_VALIDATION_SECTION_HEADING in result.stdout


def test_revalidating_replaces_the_failure_record(
    tmp_path: Path, unusable_dir: Path, monkeypatch
) -> None:
    """Each retry the records suggest replaces its own record and leaves the other alone."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    report = tmp_path / "report.md"
    first = CliRunner().invoke(app, _research_args(tmp_path, unusable_dir, {"reference", "term"}))
    assert first.exit_code == 3, first.output

    retry = CliRunner().invoke(app, [
        "validate-references", str(report), "--in-place",
        "--cache-dir", str(tmp_path / "references"), "--skip-prefix", "PMID",
        "--email", "test@example.org",
    ])

    assert retry.exit_code == 0, retry.output
    frontmatter, body = parse_frontmatter(report.read_text(encoding="utf-8"))
    assert "status" not in frontmatter["reference_validation"]
    assert "total_references" in frontmatter["reference_validation"]
    assert frontmatter["term_validation"]["status"] == "failed"
    assert body.count(VALIDATION_SECTION_HEADING) == 1
    assert body.count(TERM_VALIDATION_SECTION_HEADING) == 1
    assert body.count(NOT_COMPLETED) == 1
    assert body.index(VALIDATION_SECTION_HEADING) < body.index(TERM_VALIDATION_SECTION_HEADING)

    term_retry = CliRunner().invoke(app, [
        "validate-terms", str(report), "--in-place",
        "--cache-dir", str(tmp_path / "terms"), "--offline",
    ])

    assert term_retry.exit_code == 0, term_retry.output
    frontmatter, body = parse_frontmatter(report.read_text(encoding="utf-8"))
    assert "status" not in frontmatter["term_validation"]
    assert "total_terms" in frontmatter["term_validation"]
    assert "total_references" in frontmatter["reference_validation"]
    assert body.count(TERM_VALIDATION_SECTION_HEADING) == 1
    assert NOT_COMPLETED not in body


@pytest.mark.parametrize(
    ("message", "kept", "removed"),
    [
        (
            "503 Server Error for url: https://data.example.org/terms?apikey=s3cret&id=GO",
            "https://data.example.org/terms?<query removed>",
            "s3cret",
        ),
        (
            "HTTPSConnectionPool(host='x.org', port=443): Max retries exceeded with "
            "url: /api/search?q=GO&email=someone@example.org (Caused by timeout)",
            "url: /api/search?<query removed> (Caused by timeout)",
            "someone@example.org",
        ),
        (
            # Unencoded punctuation inside the query must not end the redaction
            # early and leave the rest of the query, key included, behind.
            "500 Server Error for url: https://x.org/s?q=label(GO)&apikey=s3cret",
            "https://x.org/s?<query removed>",
            "s3cret",
        ),
    ],
)
def test_query_strings_are_kept_out_of_the_record(message: str, kept: str, removed: str) -> None:
    """A request URL in an error can carry an email or a key; the file keeps neither."""
    failure = ValidationFailure.from_exception("term", ConnectionError(message))

    assert kept in failure.error
    assert removed not in failure.error
    assert removed not in failure.to_markdown()


def test_a_failure_record_comes_off_again_like_a_report() -> None:
    """Re-validating must not read a previous failure back in as report content."""
    result = ResearchResult(query="q", markdown="Body", provider="mock")
    failure = ValidationFailure.from_exception("reference", OSError("unreachable"))

    formatted = ResultFormatter().format_full_markdown(
        result, reference_validation=failure,
        term_validation=ValidationFailure.from_exception("term", OSError("unreachable")),
    )

    assert strip_validation_section(formatted).endswith("Body\n")
