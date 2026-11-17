"""The verification datamodel and its generated code (issue #43, item 5)."""

from pathlib import Path

import pytest

from deep_research_client.claims import (
    ClaimVerification,
    Decidability,
    FlagKind,
    ReviewFlag,
    SourceType,
    VerificationReport,
    VerificationStatus,
    VerifiedSourceType,
    VerifierInfo,
)


def _report(flags=None, verifications=None) -> VerificationReport:
    return VerificationReport(
        source_id="r.md",
        source_type=VerifiedSourceType.MARKDOWN_REPORT,
        verifier=VerifierInfo(name="triage"),
        flags=flags,
        verifications=verifications,
    )


def test_views_read_flags_and_verdicts() -> None:
    report = _report(
        flags=[
            ReviewFlag(kind=FlagKind.UNSUPPORTED_CLAIM, decidability=Decidability.AGENT,
                       claim_ids=["c1"], message="c1 cites nothing"),
            ReviewFlag(kind=FlagKind.CONTRADICTION, decidability=Decidability.HUMAN,
                       claim_ids=["c2", "c3"], pair_id="p1", message="c2 vs c3"),
        ],
        verifications=[
            ClaimVerification(claim_id="c4", status=VerificationStatus.SUPPORTED,
                              method="citation:resolution", reference_id="PMID:1"),
        ],
    )
    assert report.flag_count == 2
    assert [f.kind for f in report.needs_human] == [FlagKind.CONTRADICTION.value]
    assert [f.kind for f in report.needs_agent] == [FlagKind.UNSUPPORTED_CLAIM.value]
    assert [f.claim_ids for f in report.flags_for("c2")] == [["c2", "c3"]]
    assert report.flags_of_kind(FlagKind.CONTRADICTION)[0].pair_id == "p1"
    verdict = report.verification_for("c4")
    assert verdict is not None and verdict.status == VerificationStatus.SUPPORTED.value
    assert report.verification_for("nope") is None


def test_empty_report_reads_cleanly() -> None:
    """A triage-only report with no flags and no verdicts reads through the views."""
    report = _report()
    assert report.flag_count == 0
    assert report.needs_human == []
    assert report.verification_list == []
    assert report.flags_for("c1") == []


def test_verified_source_type_covers_every_claims_source_type() -> None:
    """VerifiedSourceType mirrors claims.SourceType; a new source type must be added here."""
    assert {e.value for e in SourceType} == {e.value for e in VerifiedSourceType}


def test_datamodel_matches_linkml_schema() -> None:
    """claims/verification_datamodel.py is generated; regenerate with `just gen-datamodel`."""
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    schema = Path("src/deep_research_client/claims/verification.yaml")
    generated = repo_root / "src/deep_research_client/claims/verification_datamodel.py"

    gen_pydantic = shutil.which("gen-pydantic", path=str(Path(sys.executable).parent))
    if not gen_pydantic:
        pytest.skip("linkml is not installed; install the dev dependency group to check drift")

    completed = subprocess.run(
        [gen_pydantic, str(schema)], cwd=repo_root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, f"gen-pydantic failed:\n{completed.stderr}"
    assert completed.stdout == generated.read_text(encoding="utf-8"), (
        "verification_datamodel.py does not match verification.yaml. Either the schema changed "
        "or linkml was upgraded; run `just gen-datamodel` and review the diff."
    )
