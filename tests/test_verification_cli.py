"""The `claims verify` command: triage-only runs and refusals.

The live citation path (`--check-citations`) is covered under the integration
marker in test_verification_integration.py; here only the paths that need neither
a network nor the optional extra run.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.claims import (
    AlignedClaimPair,
    AlignedSource,
    AlignedSourceType,
    AlignerInfo,
    AnchorStatus,
    Claim,
    ClaimAlignment,
    ClaimBasis,
    ClaimSet,
    ClaimTopic,
    CitationStatus,
    EvidentialRelation,
    ExtractorInfo,
    SemanticRelation,
    SourceDocument,
    SourceType,
)
from deep_research_client.cli import app

CLAIMS = str(Path(__file__).parent / "input" / "claims" / "alignment" / "report_a.claims.json")
runner = CliRunner()


def test_triage_only_run_emits_a_report(tmp_path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["claims", "verify", CLAIMS, "-o", str(out)])
    assert result.exit_code == 0, result.stderr
    assert "review flags" in result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verifier"]["name"] == "triage"
    assert "verifications" not in report  # triage-only: no verifier verdicts


@pytest.mark.parametrize(
    "flag",
    [["--email", "me@example.org"], ["--cache", "refs"], ["--trust-cache"]],
)
def test_citation_options_warn_without_check_citations(flag: list[str], tmp_path) -> None:
    """An option that only affects citation checking says so when checking is off."""
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["claims", "verify", CLAIMS, "-o", str(out), *flag])
    assert result.exit_code == 0, result.stderr
    assert f"{flag[0]} has no effect without --check-citations" in result.stderr


def test_triage_only_run_does_not_warn_about_citation_options(tmp_path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["claims", "verify", CLAIMS, "-o", str(out)])
    assert result.exit_code == 0, result.stderr
    assert "--check-citations" not in result.stderr


def _write_contradiction_set(tmp_path) -> tuple[Path, Path]:
    """A tiny claim set and a within-source alignment that contradicts itself."""
    def claim(cid: str) -> Claim:
        return Claim(id=cid, claim_text="Acne is purely bacterial.", anchor_status=AnchorStatus.EXACT,
                     about=ClaimTopic.DOMAIN, basis=ClaimBasis.BACKGROUND_KNOWLEDGE,
                     citation_status=CitationStatus.UNCITED)

    claim_set = ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="x"), claims=[claim("c1"), claim("c2")],
    )
    alignment = ClaimAlignment(
        source_a=AlignedSource(id="r.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id="r.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        aligner=AlignerInfo(name="llm-pairwise"), within_source=True,
        pairs=[AlignedClaimPair(id="p1", claim_a_id="c1", claim_b_id="c2",
                                claim_a_text="Acne is purely bacterial.",
                                claim_b_text="Acne is not bacterial.",
                                semantic_relation=SemanticRelation.CONTRADICTORY,
                                evidential_relation=EvidentialRelation.BOTH_BACKGROUND)],
    )
    claims_path = tmp_path / "c.json"
    align_path = tmp_path / "a.json"
    claims_path.write_text(claim_set.model_dump_json(exclude_none=True), encoding="utf-8")
    align_path.write_text(alignment.model_dump_json(exclude_none=True), encoding="utf-8")
    return claims_path, align_path


def test_alignment_contributes_contradiction_flags(tmp_path) -> None:
    claims_path, align_path = _write_contradiction_set(tmp_path)
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["claims", "verify", str(claims_path), "--alignment", str(align_path),
                                 "-o", str(out)])
    assert result.exit_code == 0, result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    kinds = {f["kind"] for f in report["flags"]}
    assert "SELF_CONTRADICTION" in kinds
    assert "need a human" in result.stderr


def test_a_missing_claims_file_is_refused() -> None:
    result = runner.invoke(app, ["claims", "verify", "nope.json"])
    assert result.exit_code == 1
    assert "File not found" in result.stderr


def test_a_missing_alignment_file_is_refused() -> None:
    result = runner.invoke(app, ["claims", "verify", CLAIMS, "--alignment", "nope.json"])
    assert result.exit_code == 1
    assert "Alignment file not found" in result.stderr


def test_an_unparseable_alignment_is_refused(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    result = runner.invoke(app, ["claims", "verify", CLAIMS, "--alignment", str(bad)])
    assert result.exit_code == 1
    assert "Could not parse" in result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("linkml_reference_validator") is not None,
    reason="runs only where the `validation` extra is absent; it is in the dev group, so the "
           "standard CI pytest jobs install it and skip this -- run it with `--no-dev`",
)
def test_check_citations_names_the_validation_extra_when_missing() -> None:
    """With linkml-reference-validator not installed, `--check-citations` says how to install it."""
    result = runner.invoke(app, ["claims", "verify", CLAIMS, "--check-citations"])
    assert result.exit_code == 1
    assert "deep_research_client[validation]" in result.stderr
