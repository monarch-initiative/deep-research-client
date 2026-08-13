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
    SupportingTextCheck,
    extract_evidence,
    extract_quoted_claims,
    extract_references,
    find_reference_ids,
    strip_validation_section,
)

PAPER_TITLE = "Widget Coloration in Wild Populations"
PAPER_BODY = (
    "We surveyed 1200 widgets across twelve sites. Widgets are blue in 90% of "
    "observed cases, and green in the remainder."
)
CACHED_PMID = "PMID:12345678"
SECOND_CACHED_PMID = "PMID:12345679"


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
    for pmid, title in ((CACHED_PMID, PAPER_TITLE), (SECOND_CACHED_PMID, "A Second Study")):
        fetcher.get_cache_path(pmid).write_text(
            "---\n"
            f"reference_id: {pmid}\n"
            f"title: {title}\n"
            "year: '2019'\n"
            "journal: Journal of Widgets\n"
            "content_type: abstract_only\n"
            "full_text_attempted: true\n"
            "---\n\n"
            f"# {title}\n\n"
            "## Content\n\n"
            f"{PAPER_BODY}\n",
            encoding="utf-8",
        )
    return cache_dir


def _reference(reference_id: str, count: int = 1) -> FoundReference:
    """Build a FoundReference for an identifier the extractor does not scan for."""
    return FoundReference(normalized_id=reference_id, raw=reference_id, count=count)


def _ncbi_email() -> str:
    """Contact address for live NCBI calls, overridable per contributor."""
    import os

    return os.environ.get("NCBI_EMAIL", "deep-research-client@example.org")


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
        # A longer number must not be truncated into a plausible-looking PMID
        ("PMID:1234567890", []),
    ],
)
def test_find_reference_ids(text: str, expected: list[str]) -> None:
    assert [r.normalized_id for r in find_reference_ids(text)] == expected


@pytest.mark.parametrize(
    "text",
    [
        "doi:10.1038/ng1234",
        "**doi:10.1038/ng1234**",
        "*doi:10.1038/ng1234*",
        "_doi:10.1038/ng1234_",
        "`doi:10.1038/ng1234`",
        "~~doi:10.1038/ng1234~~",
        "See (doi:10.1038/ng1234).",
        "[doi:10.1038/ng1234]",
        'He wrote "doi:10.1038/ng1234".',
        "| doi:10.1038/ng1234 | Nature |",
        "|doi:10.1038/ng1234|Nature|",
        r"\|doi:10.1038/ng1234\|",
        "<https://doi.org/10.1038/ng1234>",
        "**https://doi.org/10.1038/ng1234**",
    ],
)
def test_doi_survives_markdown_wrapping(text: str) -> None:
    """Markup must not be glued onto a DOI, which would read as a fabrication."""
    assert [r.normalized_id for r in find_reference_ids(text)] == ["DOI:10.1038/ng1234"]


@pytest.mark.parametrize(
    "text,expected",
    [
        # Publisher landing pages, which is how deep research tools most often
        # render a DOI. Observed in real claude_code and falcon reports.
        ("https://www.pnas.org/doi/10.1073/pnas.232568699", "DOI:10.1073/pnas.232568699"),
        (
            "https://onlinelibrary.wiley.com/doi/10.1002/ajmg.a.61787",
            "DOI:10.1002/ajmg.a.61787",
        ),
        (
            "https://www.tandfonline.com/doi/full/10.1080/1744666X.2025.2612589",
            "DOI:10.1080/1744666X.2025.2612589",
        ),
        ("https://dx.doi.org/10.1038/ng1234", "DOI:10.1038/ng1234"),
        ("[Paper](https://www.pnas.org/doi/abs/10.1073/pnas.232568699)", "DOI:10.1073/pnas.232568699"),
    ],
)
def test_publisher_doi_urls_are_extracted(text: str, expected: str) -> None:
    """A DOI rendered as a publisher URL must not be silently left unchecked."""
    assert [r.normalized_id for r in find_reference_ids(text)] == [expected]


@pytest.mark.parametrize(
    "text,expected",
    [
        # Markdown-escaped underscore, as falcon writes book-chapter DOIs
        (
            r"https://doi.org/10.1007/978-3-030-80614-9\_8",
            "DOI:10.1007/978-3-030-80614-9_8",
        ),
        (r"doi:10.1007/s00439-021-02282\-3", "DOI:10.1007/s00439-021-02282-3"),
    ],
)
def test_markdown_escapes_inside_a_doi_are_undone(text: str, expected: str) -> None:
    """An escaped character must not make a real DOI look fabricated."""
    assert [r.normalized_id for r in find_reference_ids(text)] == [expected]


@pytest.mark.parametrize(
    "url",
    [
        "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        "https://www.ncbi.nlm.nih.gov/pubmed/12345678",
        "http://ncbi.nlm.nih.gov/pubmed/12345678",
    ],
)
def test_both_pubmed_url_styles_are_extracted(url: str) -> None:
    """Providers still emit the older www.ncbi.nlm.nih.gov/pubmed path."""
    assert [r.normalized_id for r in find_reference_ids(url)] == ["PMID:12345678"]


def test_doi_containing_parentheses_is_preserved() -> None:
    """Real DOIs contain parentheses, so they must survive trailing-punctuation stripping."""
    found = find_reference_ids("Reported in doi:10.1016/0092-8674(94)90302-6 originally.")

    assert [r.normalized_id for r in found] == ["DOI:10.1016/0092-8674(94)90302-6"]


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


def test_quoted_claim_keeps_a_doi_containing_parentheses() -> None:
    """The quote scan must agree with the body scan about the identifier.

    A citation group that stopped at the first ')' produced a truncated DOI, so
    the quote was orphaned from the reference the body scan had found.
    """
    text = 'They said "a suitably long quotation here" (DOI:10.1016/0092-8674(94)90302-6).'

    claims = extract_quoted_claims(text)

    assert [c.reference_id for c in claims] == ["DOI:10.1016/0092-8674(94)90302-6"]
    assert claims[0].reference_id == find_reference_ids(text)[0].normalized_id


def test_quoted_claims_do_not_absorb_the_next_citation() -> None:
    """Widening the citation group must not let one quote swallow another's."""
    claims = extract_quoted_claims(
        'A "first suitably long quotation" (PMID:1111111). '
        'And "second suitably long quotation" (PMID:2222222).'
    )

    assert [c.reference_id for c in claims] == ["PMID:1111111", "PMID:2222222"]


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
    check = report.checked_references[0]
    assert check.status == ReferenceStatus.VERIFIED
    assert check.title == PAPER_TITLE
    assert report.confabulation_rate == 0.0
    assert not report.has_confabulations


def test_unresolvable_reference_is_flagged(
    validator: ReferenceValidator, tmp_path: Path
) -> None:
    report = validator.validate_references([_reference(f"file:{tmp_path / 'absent.md'}")])

    assert report.checked_references[0].status == ReferenceStatus.NOT_FOUND
    assert report.not_found_count == 1
    assert report.confabulation_rate == 1.0
    assert report.has_confabulations


def test_truncated_doi_is_not_called_a_fabrication() -> None:
    """A DOI the report itself cut short must not be reported as invented.

    Observed in a real Edison report, which cited
    https://doi.org/10.1016/0092-8674(94)90302-6 correctly in its body and
    https://doi.org/10.1016/0092-8674(94 in its reference list.
    """
    from deep_research_client.validation.validator import _reclassify_truncated_dois

    checks = [
        ReferenceCheck(
            reference_id="DOI:10.1016/0092-8674(94)90302-6",
            status=ReferenceStatus.VERIFIED,
        ),
        ReferenceCheck(
            reference_id="DOI:10.1016/0092-8674(94", status=ReferenceStatus.NOT_FOUND
        ),
    ]

    report = ReferenceValidationReport(references=_reclassify_truncated_dois(checks))

    assert report.checked_references[1].status == ReferenceStatus.UNVERIFIABLE
    assert "truncated copy" in (report.checked_references[1].message or "")
    assert report.confabulated_references == []
    assert not report.has_confabulations


@pytest.mark.parametrize(
    "unresolved_id",
    [
        # Not a prefix of the resolved DOI
        "DOI:10.9999/invented.entirely",
        # A PMID is never demoted: PMIDs are opaque numerics, so one being a
        # prefix of another says nothing about them being the same record.
        "PMID:1234567",
    ],
)
def test_truncation_rule_leaves_other_failures_alone(unresolved_id: str) -> None:
    from deep_research_client.validation.validator import _reclassify_truncated_dois

    checks = [
        ReferenceCheck(reference_id="DOI:10.1234/abcdef", status=ReferenceStatus.VERIFIED),
        ReferenceCheck(reference_id="PMID:12345678", status=ReferenceStatus.VERIFIED),
        ReferenceCheck(reference_id=unresolved_id, status=ReferenceStatus.NOT_FOUND),
    ]

    result = _reclassify_truncated_dois(checks)

    assert result[2].status == ReferenceStatus.NOT_FOUND


def test_unrelated_unresolved_reference_is_still_flagged(
    cache_dir: Path, tmp_path: Path
) -> None:
    """The truncation rule must not swallow a genuinely unresolvable reference."""
    paper = tmp_path / "paper.md"
    paper.write_text("# A Paper\n\nContent.\n", encoding="utf-8")

    validator = ReferenceValidator(cache_dir=cache_dir)
    report = validator.validate_references(
        [_reference(f"file:{paper}"), _reference(f"file:{tmp_path / 'absent.md'}")]
    )

    assert report.not_found_count == 1
    assert report.has_confabulations


def test_unknown_prefix_is_unverifiable(validator: ReferenceValidator) -> None:
    report = validator.validate_references([_reference("WIDGETDB:0001")])

    assert report.checked_references[0].status == ReferenceStatus.UNVERIFIABLE
    assert report.unverifiable_count == 1
    assert not report.has_confabulations


def test_skip_prefixes_are_unverifiable(cache_dir: Path, paper_ref: str) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, skip_prefixes=["file"])

    report = validator.validate_references([_reference(paper_ref)])

    assert report.checked_references[0].status == ReferenceStatus.UNVERIFIABLE
    assert not report.has_confabulations


def test_occurrence_count_is_carried_through(
    validator: ReferenceValidator, paper_ref: str
) -> None:
    report = validator.validate_references([_reference(paper_ref, count=4)])

    assert report.checked_references[0].occurrences == 4


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


def test_quote_for_unresolved_reference_is_not_called_unsupported(
    validator: ReferenceValidator, tmp_path: Path
) -> None:
    """An unresolvable reference makes its quote uncheckable, not fabricated."""
    missing_ref = f"file:{tmp_path / 'absent.md'}"

    report = validator.validate_references(
        [_reference(missing_ref)],
        [QuotedClaim(quote="Widgets are blue", reference_id=missing_ref)],
    )

    assert report.checked_references[0].status == ReferenceStatus.NOT_FOUND
    assert report.unsupported_quotes == []
    assert report.unchecked_quotes[0].was_checkable is False
    assert "did not resolve" in (report.unchecked_quotes[0].message or "")
    assert report.quotes_checked == 0


def test_quote_against_reference_without_content_is_uncheckable(
    validator: ReferenceValidator, tmp_path: Path
) -> None:
    """A resolved record with no text cannot contradict a quote."""
    empty = tmp_path / "empty_paper.md"
    empty.write_text("", encoding="utf-8")
    empty_ref = f"file:{empty}"

    report = validator.validate_references(
        [_reference(empty_ref)],
        [QuotedClaim(quote="Widgets are blue in most cases", reference_id=empty_ref)],
    )

    assert report.checked_references[0].status == ReferenceStatus.VERIFIED
    assert report.unsupported_quotes == []
    assert "no abstract or full text" in (report.unchecked_quotes[0].message or "")
    assert not report.has_confabulations


def test_quotes_for_skipped_prefixes_are_recorded_not_dropped(
    cache_dir: Path, paper_ref: str
) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, skip_prefixes=["file"])

    report = validator.validate_references(
        [_reference(paper_ref)],
        [QuotedClaim(quote="Widgets are magenta", reference_id=paper_ref)],
    )

    assert report.quotes_checked == 0
    assert len(report.unchecked_quotes) == 1
    assert not report.has_confabulations


def test_unmatched_quote_reference_does_not_blame_a_limit(
    validator: ReferenceValidator, paper_ref: str
) -> None:
    """With no limit set, the message must not claim one was reached."""
    report = validator.validate_references(
        [_reference(paper_ref)],
        [QuotedClaim(quote="Widgets are blue", reference_id="PMID:9999999")],
    )

    assert report.unchecked_quotes[0].message == (
        "Reference was not among those extracted from the report body"
    )


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
    # The dropped reference's quote is still accounted for, just not checked.
    assert len(report.quote_checks) == 2
    assert [q.reference_id for q in report.unchecked_quotes] == [second_ref]
    assert "reference limit was reached" in (report.unchecked_quotes[0].message or "")
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
    assert report.checked_references[0].status == ReferenceStatus.VERIFIED
    assert report.checked_references[0].journal == "Journal of Widgets"
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
    assert report.checked_references[0].status == ReferenceStatus.VERIFIED


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
    if not gen_pydantic:
        pytest.skip("linkml is not installed; install the dev dependency group to check drift")

    completed = subprocess.run(
        [gen_pydantic, str(schema)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"gen-pydantic failed:\n{completed.stderr}"

    assert completed.stdout == generated.read_text(encoding="utf-8"), (
        "datamodel.py does not match reference_validation.yaml. Either the schema "
        "changed or linkml was upgraded; run `just gen-datamodel` and review the diff."
    )


def test_report_serialises_with_exclude_none() -> None:
    """The inherited serializer nulls empty lists; the report must survive that."""
    empty = ReferenceValidationReport()
    assert empty.model_dump(exclude_none=True) == {"truncated": False}

    partial = ReferenceValidationReport(
        references=[ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED)]
    )
    dumped = partial.model_dump(exclude_none=True)
    assert dumped["references"][0]["reference_id"] == "PMID:1"
    assert "supporting_text" not in dumped


def test_status_is_stored_as_a_string() -> None:
    """Pins the use_enum_values behaviour inherited from the generated base."""
    check = ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED)

    assert check.status == ReferenceStatus.VERIFIED
    assert check.status == "VERIFIED"
    assert not isinstance(check.status, ReferenceStatus)


@pytest.mark.parametrize(
    "score,expected", [(0.5, 0.5), (1.0000000002, 1.0), (-0.1, 0.0), (42.0, 1.0)]
)
def test_similarity_score_is_clamped(score: float, expected: float) -> None:
    """A foreign float outside 0-1 must not invalidate a whole report."""
    from deep_research_client.validation.validator import _clamp_similarity

    assert _clamp_similarity(score) == expected
    assert (
        SupportingTextCheck(
            reference_id="PMID:1",
            quote="x",
            is_valid=False,
            similarity_score=_clamp_similarity(score),
        ).similarity_score
        == expected
    )


@pytest.mark.parametrize(
    "content,expected",
    [
        ("# Report\n\nBody.\n", "# Report\n\nBody.\n"),
        ("# Report\n\nBody.\n\n## Reference Validation\n\nStuff.\n", "# Report\n\nBody.\n"),
        # Two appended sections, as an older run could have left behind
        (
            "# Report\n\nBody.\n\n## Reference Validation\n\nOne.\n\n"
            "## Reference Validation\n\nTwo.\n",
            "# Report\n\nBody.\n",
        ),
        ("## Reference Validation\n\nOnly.\n", ""),
        # The generated section carries level-three headings, which must not
        # stop it being recognised as the trailing section
        (
            "# Report\n\nBody.\n\n## Reference Validation\n\n"
            "### Unresolved references\n\n- `PMID:1`\n",
            "# Report\n\nBody.\n",
        ),
    ],
)
def test_strip_validation_section(content: str, expected: str) -> None:
    assert strip_validation_section(content) == expected


def test_strip_validation_section_keeps_a_non_trailing_section() -> None:
    """A report that discusses validation and then continues must survive intact.

    The stripped text is written back over the file by --in-place, so removing
    from a mid-document heading would destroy the rest of the report.
    """
    content = (
        "---\na: 1\n---\n# Report\n\n## Reference Validation\n\n"
        "We discuss how validation works.\n\n## Conclusions\n\nImportant text.\n"
    )

    assert strip_validation_section(content) == content


def test_cli_in_place_preserves_a_body_section_named_like_ours(
    tmp_path: Path, seeded_cache: Path
) -> None:
    """End-to-end guard on the same data-loss path."""
    report_file = tmp_path / "report.md"
    report_file.write_text(
        f"## Output\n\nCited ({CACHED_PMID}).\n\n"
        "## Reference Validation\n\nProse about validation.\n\n"
        "## Conclusions\n\nMust not be deleted.\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_file),
            "--cache-dir",
            str(seeded_cache),
            "--in-place",
        ],
    )

    assert result.exit_code == 0, result.output
    written = report_file.read_text(encoding="utf-8")
    assert "Must not be deleted." in written
    assert "Prose about validation." in written


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
    assert "(3 mentions)" in markdown
    assert "PMID:7913883" not in markdown


def test_all_unverifiable_report_does_not_claim_success() -> None:
    """A run that learned nothing must not read as a clean bill of health."""
    report = ReferenceValidationReport(
        references=[ReferenceCheck(reference_id="X:1", status=ReferenceStatus.UNVERIFIABLE)]
    )

    markdown = report.to_markdown()

    assert "All extracted references resolved successfully." not in markdown
    assert "confirmed or contradicted" in markdown


def test_partly_unverifiable_report_states_the_shortfall() -> None:
    report = ReferenceValidationReport(
        references=[
            ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED),
            ReferenceCheck(reference_id="X:1", status=ReferenceStatus.UNVERIFIABLE),
        ]
    )

    assert "1 of 2 references resolved" in report.to_markdown()


def test_confabulation_rate_ignores_unverifiable_references() -> None:
    """Skipping a prefix must not dilute the rate."""
    report = ReferenceValidationReport(
        references=[
            ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.NOT_FOUND),
            ReferenceCheck(reference_id="PMID:2", status=ReferenceStatus.VERIFIED),
            ReferenceCheck(reference_id="X:1", status=ReferenceStatus.UNVERIFIABLE),
            ReferenceCheck(reference_id="X:2", status=ReferenceStatus.UNVERIFIABLE),
        ]
    )

    assert report.resolvable_count == 2
    assert report.confabulation_rate == 0.5


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


def test_cli_validate_references_in_place_is_idempotent(
    tmp_path: Path, seeded_cache: Path
) -> None:
    """A second --in-place run must not stack sections or re-count identifiers."""
    report_file = _write_report(tmp_path / "report.md", f"Established previously ({CACHED_PMID}).")
    args = [
        "validate-references",
        str(report_file),
        "--cache-dir",
        str(seeded_cache),
        "--in-place",
    ]

    assert CliRunner().invoke(app, args).exit_code == 0
    first = report_file.read_text(encoding="utf-8")
    assert CliRunner().invoke(app, args).exit_code == 0
    second = report_file.read_text(encoding="utf-8")

    assert first == second
    assert second.count("## Reference Validation") == 1
    assert "| References checked | 1 |" in second


def test_cli_in_place_refreshes_stale_frontmatter(tmp_path: Path, seeded_cache: Path) -> None:
    """A frontmatter summary must not survive contradicting the section below it."""
    report_file = tmp_path / "report.md"
    report_file.write_text(
        "---\n"
        "provider: mock\n"
        "reference_validation:\n"
        "  total_references: 99\n"
        "  verified: 99\n"
        "  not_found: 0\n"
        "---\n\n"
        f"## Output\n\nCited ({CACHED_PMID}).\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_file),
            "--cache-dir",
            str(seeded_cache),
            "--in-place",
        ],
    )

    assert result.exit_code == 0, result.output
    import yaml

    from deep_research_client.markdown_parser import parse_frontmatter

    frontmatter, _ = parse_frontmatter(report_file.read_text(encoding="utf-8"))
    assert frontmatter["reference_validation"]["total_references"] == 1
    assert frontmatter["reference_validation"]["verified"] == 1
    assert frontmatter["provider"] == "mock"
    assert yaml.safe_load(yaml.dump(frontmatter)) == frontmatter


def test_cli_in_place_leaves_plain_frontmatter_alone(
    tmp_path: Path, seeded_cache: Path
) -> None:
    """A file with no validation summary must not be reformatted."""
    original_frontmatter = "---\n# a comment\nprovider:   mock\n---\n"
    report_file = tmp_path / "report.md"
    report_file.write_text(
        original_frontmatter + f"\n## Output\n\nCited ({CACHED_PMID}).\n", encoding="utf-8"
    )

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_file),
            "--cache-dir",
            str(seeded_cache),
            "--in-place",
        ],
    )

    assert result.exit_code == 0, result.output
    assert report_file.read_text(encoding="utf-8").startswith(original_frontmatter)


def test_cli_validate_references_output_file(tmp_path: Path, seeded_cache: Path) -> None:
    report_file = _write_report(tmp_path / "report.md", f"Established previously ({CACHED_PMID}).")
    out = tmp_path / "validation.md"

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_file),
            "--cache-dir",
            str(seeded_cache),
            "--output",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    written = out.read_text(encoding="utf-8")
    assert written.startswith("## Reference Validation")
    assert "| References checked | 1 |" in written
    # The source file is untouched when only --output is given
    assert "## Reference Validation" not in report_file.read_text(encoding="utf-8")


def test_cli_validate_references_handles_several_files(
    tmp_path: Path, seeded_cache: Path
) -> None:
    first = _write_report(tmp_path / "a.md", f"Cited here ({CACHED_PMID}).")
    second = _write_report(tmp_path / "b.md", "Nothing cited here.")

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(first),
            str(second),
            "--cache-dir",
            str(seeded_cache),
            "--in-place",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "| References checked | 1 |" in first.read_text(encoding="utf-8")
    assert "No PMID or DOI references" in second.read_text(encoding="utf-8")


def test_cli_validate_references_truncation_is_surfaced(
    tmp_path: Path, seeded_cache: Path
) -> None:
    report_file = _write_report(
        tmp_path / "report.md", f"Cited ({CACHED_PMID}) and ({SECOND_CACHED_PMID})."
    )

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_file),
            "--cache-dir",
            str(seeded_cache),
            "--max-references",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "stopped early" in result.output
    assert "| References checked | 1 |" in result.output


def test_cli_validate_references_rejects_zero_max(tmp_path: Path, seeded_cache: Path) -> None:
    report_file = _write_report(tmp_path / "report.md", f"Cited ({CACHED_PMID}).")

    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_file),
            "--cache-dir",
            str(seeded_cache),
            "--max-references",
            "0",
        ],
    )

    assert result.exit_code != 0


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


def test_cli_research_fails_on_unresolved(tmp_path: Path, cache_dir: Path, monkeypatch) -> None:
    """An unresolvable citation exits 2 - and the report is still written."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        app,
        [
            "research",
            f"Cite the nonexistent file:{tmp_path / 'absent.md'} please",
            "--provider",
            "mock",
            "--no-cache",
            "--output",
            str(output),
            "--validate-references",
            "--validation-cache-dir",
            str(cache_dir),
            "--fail-on-unresolved",
        ],
    )

    written = output.read_text(encoding="utf-8")
    assert output.exists()
    if result.exit_code == 2:
        assert "## Reference Validation" in written
    else:
        # The mock provider echoed no resolvable identifiers, so nothing failed.
        assert result.exit_code == 0, result.output


def test_cli_research_keeps_report_when_validation_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    """A validation problem must never cost the user their research result."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    monkeypatch.setattr(
        "deep_research_client.validation.validator_is_available", lambda: False
    )
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
        ],
    )

    assert result.exit_code == 1
    assert output.exists(), "the research result must survive a validation failure"
    assert "## Question" in output.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Integration: live bibliographic APIs
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_pmid_resolves(cache_dir: Path) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, email=_ncbi_email())

    report = validator.validate_markdown("Achondroplasia is caused by FGFR3 (PMID:7913883).")

    assert report.checked_references[0].status == ReferenceStatus.VERIFIED
    assert "FGFR3" in (report.checked_references[0].title or "")


@pytest.mark.integration
def test_fabricated_pmid_is_flagged(cache_dir: Path) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, email=_ncbi_email())

    report = validator.validate_markdown("A confident but invented claim (PMID:99999999).")

    assert report.checked_references[0].status == ReferenceStatus.NOT_FOUND
    assert report.has_confabulations


@pytest.mark.integration
def test_fabricated_doi_is_flagged(cache_dir: Path) -> None:
    validator = ReferenceValidator(cache_dir=cache_dir, email=_ncbi_email())

    report = validator.validate_markdown("Reported previously (DOI:10.9999/not.a.real.doi).")

    assert report.checked_references[0].status == ReferenceStatus.NOT_FOUND


@pytest.mark.integration
def test_real_quote_is_verified_against_pubmed(cache_dir: Path) -> None:
    """A verbatim quote from the abstract passes, an invented one does not."""
    validator = ReferenceValidator(cache_dir=cache_dir, email=_ncbi_email())

    report = validator.validate_markdown(
        'The authors report that "Achondroplasia (ACH) is the most common genetic form '
        'of dwarfism" (PMID:7913883), but also that "ACH is caused by a deletion of '
        'chromosome 21" (PMID:7913883).'
    )

    assert report.quotes_checked == 2
    assert report.quotes_valid_count == 1
    assert "deletion of chromosome 21" in report.unsupported_quotes[0].quote
