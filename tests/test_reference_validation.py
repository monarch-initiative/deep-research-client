"""Tests for reference validation.

Offline tests exercise the real ``linkml-reference-validator`` code paths without
network access in two ways: ``file:`` references, which it resolves from the
local filesystem, and pre-seeded entries in its on-disk reference cache, which
stand in for PubMed records. Tests that need a live bibliographic API to assert
that an identifier genuinely does not exist are marked ``integration``.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.models import ResearchResult
from deep_research_client.processing import ResultFormatter
from deep_research_client.validation import (
    FoundReference,
    QuotedClaim,
    ReferenceCheck,
    ReferenceStatus,
    ReferenceValidationReport,
    ReferenceValidator,
    extract_evidence,
    extract_quoted_claims,
    extract_references,
    find_reference_ids,
)

PAPER_TITLE = "Widget Coloration in Wild Populations"
PAPER_BODY = (
    "We surveyed 1200 widgets across twelve sites. Widgets are blue in 90% of "
    "observed cases, and green in the remainder."
)
CACHED_PMID = "PMID:12345678"


@pytest.fixture
def paper(tmp_path: Path) -> Path:
    """A local file standing in for a fetchable reference."""
    path = tmp_path / "widget_paper.md"
    path.write_text(f"# {PAPER_TITLE}\n\n{PAPER_BODY}\n", encoding="utf-8")
    return path


@pytest.fixture
def paper_ref(paper: Path) -> str:
    """The reference identifier for the local paper."""
    return f"file:{paper}"


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """An isolated reference cache directory."""
    path = tmp_path / "references_cache"
    path.mkdir()
    return path


@pytest.fixture
def validator(cache_dir: Path) -> ReferenceValidator:
    """A validator with an isolated reference cache."""
    return ReferenceValidator(cache_dir=cache_dir)


@pytest.fixture
def seeded_cache(cache_dir: Path) -> Path:
    """A reference cache pre-populated with one PubMed record.

    Seeding the cache lets the full pipeline - extraction, resolution, quote
    checking - run against a known PMID without contacting NCBI.
    """
    from linkml_reference_validator.etl.reference_fetcher import ReferenceFetcher
    from linkml_reference_validator.models import ReferenceValidationConfig

    fetcher = ReferenceFetcher(ReferenceValidationConfig(cache_dir=cache_dir))
    fetcher.get_cache_path(CACHED_PMID).write_text(
        "---\n"
        f"reference_id: {CACHED_PMID}\n"
        f"title: {PAPER_TITLE}\n"
        "year: '2019'\n"
        "journal: Journal of Widgets\n"
        "content_type: abstract_only\n"
        "full_text_attempted: true\n"
        "---\n\n"
        f"# {PAPER_TITLE}\n\n"
        "## Content\n\n"
        f"{PAPER_BODY}\n",
        encoding="utf-8",
    )
    return cache_dir


def _reference(reference_id: str, count: int = 1) -> FoundReference:
    """Build a FoundReference for an identifier the extractor does not scan for."""
    return FoundReference(normalized_id=reference_id, raw=reference_id, count=count)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Shown in PMID:7913883.", ["PMID:7913883"]),
        ("Shown in PMID: 7913883.", ["PMID:7913883"]),
        ("See doi:10.1038/ng1234 for details.", ["DOI:10.1038/ng1234"]),
        ("See https://doi.org/10.1038/ng1234.", ["DOI:10.1038/ng1234"]),
        ("See https://pubmed.ncbi.nlm.nih.gov/12345678", ["PMID:12345678"]),
        ("(DOI:10.1038/ng1234)", ["DOI:10.1038/ng1234"]),
        ("No identifiers at all.", []),
        ("PMID:123", []),  # too short to be a PMID
    ],
)
def test_find_reference_ids(text: str, expected: list[str]) -> None:
    assert [r.normalized_id for r in find_reference_ids(text)] == expected


def test_find_reference_ids_counts_repeats() -> None:
    refs = find_reference_ids("PMID:7913883 and again PMID:7913883 and PMID:12345678")
    counts = {r.normalized_id: r.count for r in refs}
    assert counts == {"PMID:7913883": 2, "PMID:12345678": 1}


def test_extract_references_includes_citation_list() -> None:
    refs = extract_references(
        "The body cites PMID:7913883.",
        citations=["Some Author (1994). A paper. PMID:12345678"],
    )
    assert [r.normalized_id for r in refs] == ["PMID:7913883", "PMID:12345678"]


@pytest.mark.parametrize(
    "markdown,expected",
    [
        (
            'They report "widgets are blue in 90% of cases" (PMID:7913883).',
            [("widgets are blue in 90% of cases", "PMID:7913883")],
        ),
        (
            'They report "widgets are blue in 90% of cases" [DOI:10.1038/ng1234]',
            [("widgets are blue in 90% of cases", "DOI:10.1038/ng1234")],
        ),
        # Too short to be treated as an attributed quotation
        ('A single "gene" (PMID:7913883).', []),
        # A quotation with no attached citation is not checkable
        ('"an unattributed quotation of ample length here"', []),
    ],
)
def test_extract_quoted_claims(markdown: str, expected: list[tuple[str, str]]) -> None:
    claims = extract_quoted_claims(markdown)
    assert [(c.quote, c.reference_id) for c in claims] == expected


def test_extract_evidence_combines_both() -> None:
    evidence = extract_evidence(
        'The authors state "widgets are blue in 90% of cases" (PMID:7913883). '
        "See also PMID:12345678."
    )
    assert [r.normalized_id for r in evidence.references] == ["PMID:7913883", "PMID:12345678"]
    assert len(evidence.quoted_claims) == 1


def test_evaluation_scorer_shares_extraction() -> None:
    """The eval framework's citation extractor is backed by the same patterns."""
    from deep_research_client.evaluation.scorers import extract_citations_from_markdown

    citations = extract_citations_from_markdown("Shown (PMID:7913883) and (DOI:10.1038/ng1234).")

    assert [c.normalized_id for c in citations] == ["PMID:7913883", "DOI:10.1038/ng1234"]


# ---------------------------------------------------------------------------
# Resolving references
# ---------------------------------------------------------------------------


def test_resolvable_reference_is_verified(validator: ReferenceValidator, paper_ref: str) -> None:
    report = validator.validate_references([_reference(paper_ref)])

    assert report.total_references == 1
    check = report.references[0]
    assert check.status == ReferenceStatus.VERIFIED
    assert check.title == PAPER_TITLE
    assert report.confabulation_rate == 0.0
    assert not report.has_confabulations


def test_unresolvable_reference_is_flagged(
    validator: ReferenceValidator, tmp_path: Path
) -> None:
    report = validator.validate_references([_reference(f"file:{tmp_path / 'absent.md'}")])

    assert report.references[0].status == ReferenceStatus.NOT_FOUND
    assert report.not_found_count == 1
    assert report.confabulation_rate == 1.0
    assert report.has_confabulations


def test_unknown_prefix_is_unverifiable(validator: ReferenceValidator) -> None:
    report = validator.validate_references([_reference("WIDGETDB:0001")])

    assert report.references[0].status == ReferenceStatus.UNVERIFIABLE
    assert report.unverifiable_count == 1
    assert not report.has_confabulations


def test_skip_prefixes_are_unverifiable(cache_dir: Path, paper_ref: str) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, skip_prefixes=["file"])

    report = validator.validate_references([_reference(paper_ref)])

    assert report.references[0].status == ReferenceStatus.UNVERIFIABLE
    assert not report.has_confabulations


def test_occurrence_count_is_carried_through(
    validator: ReferenceValidator, paper_ref: str
) -> None:
    report = validator.validate_references([_reference(paper_ref, count=4)])

    assert report.references[0].occurrences == 4


# ---------------------------------------------------------------------------
# Checking quoted claims
# ---------------------------------------------------------------------------


def test_supported_quote_passes(validator: ReferenceValidator, paper_ref: str) -> None:
    report = validator.validate_references(
        [_reference(paper_ref)],
        [QuotedClaim(quote="Widgets are blue in 90% of observed cases", reference_id=paper_ref)],
    )

    assert report.quotes_checked == 1
    assert report.quotes_valid_count == 1
    assert not report.unsupported_quotes


def test_fabricated_quote_is_flagged(validator: ReferenceValidator, paper_ref: str) -> None:
    report = validator.validate_references(
        [_reference(paper_ref)],
        [QuotedClaim(quote="Widgets are magenta in every case", reference_id=paper_ref)],
    )

    assert report.quotes_checked == 1
    assert report.quotes_valid_count == 0
    assert report.unsupported_quotes[0].reference_id == paper_ref
    assert report.has_confabulations


def test_quote_for_unresolved_reference_is_reported(
    validator: ReferenceValidator, tmp_path: Path
) -> None:
    missing_ref = f"file:{tmp_path / 'absent.md'}"

    report = validator.validate_references(
        [_reference(missing_ref)],
        [QuotedClaim(quote="Widgets are blue", reference_id=missing_ref)],
    )

    assert report.references[0].status == ReferenceStatus.NOT_FOUND
    assert report.unsupported_quotes[0].message == (
        "Reference did not resolve, so the quote cannot be checked"
    )


def test_quotes_for_skipped_prefixes_are_not_checked(cache_dir: Path, paper_ref: str) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, skip_prefixes=["file"])

    report = validator.validate_references(
        [_reference(paper_ref)],
        [QuotedClaim(quote="Widgets are magenta", reference_id=paper_ref)],
    )

    assert report.quotes_checked == 0
    assert not report.has_confabulations


def test_quotes_for_truncated_references_are_skipped(
    tmp_path: Path, cache_dir: Path, paper_ref: str
) -> None:
    second = tmp_path / "second_paper.md"
    second.write_text("# Second Paper\n\nUnrelated content.\n", encoding="utf-8")
    second_ref = f"file:{second}"
    validator = ReferenceValidator(cache_dir=cache_dir, max_references=1)

    report = validator.validate_references(
        [_reference(paper_ref), _reference(second_ref)],
        [
            QuotedClaim(quote="Widgets are blue", reference_id=paper_ref),
            QuotedClaim(quote="Unrelated content", reference_id=second_ref),
        ],
    )

    assert report.truncated is True
    assert report.total_references == 1
    assert report.quotes_checked == 1
    assert report.supporting_text[0].reference_id == paper_ref
    assert "stopped early" in report.to_markdown()


# ---------------------------------------------------------------------------
# End-to-end validation of report text
# ---------------------------------------------------------------------------


def test_validate_markdown_end_to_end(seeded_cache: Path) -> None:
    validator = ReferenceValidator(cache_dir=seeded_cache)

    report = validator.validate_markdown(
        f'The authors report that "Widgets are blue in 90% of observed cases" ({CACHED_PMID}), '
        f'though others claim "widgets are uniformly magenta year round" ({CACHED_PMID}).'
    )

    assert report.total_references == 1
    assert report.references[0].status == ReferenceStatus.VERIFIED
    assert report.references[0].journal == "Journal of Widgets"
    assert report.quotes_checked == 2
    assert report.quotes_valid_count == 1
    assert report.unsupported_quotes[0].quote == "widgets are uniformly magenta year round"


def test_check_quotes_can_be_disabled(seeded_cache: Path) -> None:
    validator = ReferenceValidator(cache_dir=seeded_cache)

    report = validator.validate_markdown(
        f'They claim "widgets are uniformly magenta year round" ({CACHED_PMID}).',
        check_quotes=False,
    )

    assert report.quotes_checked == 0
    assert report.total_references == 1
    assert not report.has_confabulations


def test_validate_result_uses_citations(seeded_cache: Path) -> None:
    validator = ReferenceValidator(cache_dir=seeded_cache)
    result = ResearchResult(
        markdown="A report body with no inline identifiers.",
        citations=[f"Widget authors (2019). {PAPER_TITLE}. {CACHED_PMID}"],
        provider="mock",
        query="widgets",
    )

    report = validator.validate_result(result)

    assert report.total_references == 1
    assert report.references[0].status == ReferenceStatus.VERIFIED


def test_report_with_no_references(validator: ReferenceValidator) -> None:
    report = validator.validate_markdown("A report that cites nothing at all.")

    assert report.total_references == 0
    assert report.confabulation_rate == 0.0
    assert "No PMID or DOI references" in report.to_markdown()


# ---------------------------------------------------------------------------
# Generated data model
# ---------------------------------------------------------------------------


def test_datamodel_matches_linkml_schema() -> None:
    """datamodel.py is generated; regenerate it with `just gen-datamodel`.

    Guards against the checked-in Pydantic model drifting from the LinkML
    schema that is its source of truth.
    """
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    schema = Path("src/deep_research_client/validation/reference_validation.yaml")
    generated = repo_root / "src/deep_research_client/validation/datamodel.py"

    # Resolve the generator next to the running interpreter so the comparison
    # uses the linkml version this environment pins, not whatever is on PATH.
    gen_pydantic = shutil.which("gen-pydantic", path=str(Path(sys.executable).parent))
    assert gen_pydantic, "gen-pydantic not found; install the dev dependency group"

    regenerated = subprocess.run(
        [gen_pydantic, str(schema)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert regenerated == generated.read_text(encoding="utf-8"), (
        "datamodel.py is out of date with reference_validation.yaml; "
        "run `just gen-datamodel`"
    )


def test_schema_documents_every_status() -> None:
    """Every ReferenceStatus value carries a description in the schema."""
    import yaml

    schema_path = (
        Path(__file__).resolve().parent.parent
        / "src/deep_research_client/validation/reference_validation.yaml"
    )
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    permissible = schema["enums"]["ReferenceStatus"]["permissible_values"]

    assert set(permissible) == {status.value for status in ReferenceStatus}
    assert all(value.get("description") for value in permissible.values())


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_report_markdown_lists_unresolved() -> None:
    report = ReferenceValidationReport(
        references=[
            ReferenceCheck(reference_id="PMID:7913883", status=ReferenceStatus.VERIFIED),
            ReferenceCheck(
                reference_id="PMID:99999999",
                status=ReferenceStatus.NOT_FOUND,
                occurrences=3,
                message="Identifier did not resolve to a record",
            ),
        ]
    )

    markdown = report.to_markdown()

    assert "## Reference Validation" in markdown
    assert "Unresolved references" in markdown
    assert "PMID:99999999" in markdown
    assert "cited 3x" in markdown
    assert "PMID:7913883" not in markdown


def test_report_summary_is_yaml_friendly() -> None:
    report = ReferenceValidationReport(
        references=[
            ReferenceCheck(reference_id="PMID:1234567", status=ReferenceStatus.VERIFIED),
            ReferenceCheck(reference_id="PMID:99999999", status=ReferenceStatus.NOT_FOUND),
        ]
    )

    summary = report.summary()

    assert summary["total_references"] == 2
    assert summary["verified"] == 1
    assert summary["not_found"] == 1
    assert summary["confabulation_rate"] == 0.5
    assert summary["unresolved_references"] == ["PMID:99999999"]


def test_formatter_embeds_validation_report() -> None:
    result = ResearchResult(
        markdown="Findings cite PMID:99999999.",
        provider="mock",
        query="widgets",
    )
    report = ReferenceValidationReport(
        references=[
            ReferenceCheck(
                reference_id="PMID:99999999",
                status=ReferenceStatus.NOT_FOUND,
                message="Identifier did not resolve to a record",
            )
        ]
    )

    output = ResultFormatter().format_full_markdown(result, reference_validation=report)

    assert "reference_validation:" in output
    assert "not_found: 1" in output
    assert "## Reference Validation" in output


def test_formatter_unchanged_without_validation() -> None:
    result = ResearchResult(markdown="Findings.", provider="mock", query="widgets")

    output = ResultFormatter().format_full_markdown(result)

    assert "reference_validation" not in output
    assert "## Reference Validation" not in output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_report(path: Path, body: str) -> Path:
    path.write_text(
        f"---\nprovider: mock\n---\n\n## Question\n\nwidgets\n\n## Output\n\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_cli_validate_references_reports_to_stdout(tmp_path: Path, seeded_cache: Path) -> None:
    report_file = _write_report(
        tmp_path / "report.md",
        f'The authors report "Widgets are blue in 90% of observed cases" ({CACHED_PMID}).',
    )

    result = CliRunner().invoke(
        app, ["validate-references", str(report_file), "--cache-dir", str(seeded_cache)]
    )

    assert result.exit_code == 0, result.output
    assert "## Reference Validation" in result.output
    assert "All extracted references resolved successfully." in result.output


def test_cli_validate_references_flags_unsupported_quote(
    tmp_path: Path, seeded_cache: Path
) -> None:
    report_file = _write_report(
        tmp_path / "report.md",
        f'The authors report "widgets are uniformly magenta year round" ({CACHED_PMID}).',
    )

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_file),
            "--cache-dir",
            str(seeded_cache),
            "--in-place",
            "--fail-on-unresolved",
        ],
    )

    assert result.exit_code == 2, result.output
    written = report_file.read_text(encoding="utf-8")
    assert "Quotes not found in the cited source" in written


def test_cli_validate_references_json_output(tmp_path: Path, seeded_cache: Path) -> None:
    report_file = _write_report(tmp_path / "report.md", f"Established previously ({CACHED_PMID}).")
    json_path = tmp_path / "validation.json"

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_file),
            "--cache-dir",
            str(seeded_cache),
            "--json",
            str(json_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["references"][0]["status"] == ReferenceStatus.VERIFIED.value
    assert payload["references"][0]["title"] == PAPER_TITLE


def test_cli_validate_references_no_check_quotes(tmp_path: Path, seeded_cache: Path) -> None:
    report_file = _write_report(
        tmp_path / "report.md",
        f'They claim "widgets are uniformly magenta year round" ({CACHED_PMID}).',
    )

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_file),
            "--cache-dir",
            str(seeded_cache),
            "--no-check-quotes",
            "--fail-on-unresolved",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Quoted claims checked" not in result.output


def test_cli_validate_references_rejects_missing_file(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["validate-references", str(tmp_path / "absent.md")])

    assert result.exit_code == 1


def test_cli_validate_references_rejects_multi_file_output(tmp_path: Path) -> None:
    first = _write_report(tmp_path / "a.md", "nothing cited")
    second = _write_report(tmp_path / "b.md", "nothing cited")

    result = CliRunner().invoke(
        app,
        ["validate-references", str(first), str(second), "--output", str(tmp_path / "out.md")],
    )

    assert result.exit_code == 1


def test_cli_research_validates_references(tmp_path: Path, seeded_cache: Path, monkeypatch) -> None:
    """--validate-references embeds a validation section in the saved report."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        app,
        [
            "research",
            "widgets",
            "--provider",
            "mock",
            "--no-cache",
            "--output",
            str(output),
            "--validate-references",
            "--validation-cache-dir",
            str(seeded_cache),
        ],
    )

    assert result.exit_code == 0, result.output
    written = output.read_text(encoding="utf-8")
    assert "## Reference Validation" in written
    assert "reference_validation:" in written


# ---------------------------------------------------------------------------
# Integration: live bibliographic APIs
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_pmid_resolves(cache_dir: Path) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, email="jhc@lbl.gov")

    report = validator.validate_markdown("Achondroplasia is caused by FGFR3 (PMID:7913883).")

    assert report.references[0].status == ReferenceStatus.VERIFIED
    assert "FGFR3" in (report.references[0].title or "")


@pytest.mark.integration
def test_fabricated_pmid_is_flagged(cache_dir: Path) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, email="jhc@lbl.gov")

    report = validator.validate_markdown("A confident but invented claim (PMID:99999999).")

    assert report.references[0].status == ReferenceStatus.NOT_FOUND
    assert report.has_confabulations


@pytest.mark.integration
def test_fabricated_doi_is_flagged(cache_dir: Path) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, email="jhc@lbl.gov")

    report = validator.validate_markdown("Reported previously (DOI:10.9999/not.a.real.doi).")

    assert report.references[0].status == ReferenceStatus.NOT_FOUND


@pytest.mark.integration
def test_real_quote_is_verified_against_pubmed(cache_dir: Path) -> None:
    """A verbatim quote from the abstract passes, an invented one does not."""
    validator = ReferenceValidator(cache_dir=cache_dir, email="jhc@lbl.gov")

    report = validator.validate_markdown(
        'The authors report that "Achondroplasia (ACH) is the most common genetic form '
        'of dwarfism" (PMID:7913883), but also that "ACH is caused by a deletion of '
        'chromosome 21" (PMID:7913883).'
    )

    assert report.quotes_checked == 2
    assert report.quotes_valid_count == 1
    assert "deletion of chromosome 21" in report.unsupported_quotes[0].quote
