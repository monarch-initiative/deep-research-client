"""The claims datamodel and its generated code (issue #43)."""

from pathlib import Path

import pytest

from deep_research_client.claims import (
    AnchorStatus,
    Claim,
    ClaimSet,
    ExtractorInfo,
    SourceDocument,
    SourceType,
    TextSpan,
)


def test_datamodel_matches_linkml_schema() -> None:
    """claims/datamodel.py is generated; regenerate it with `just gen-datamodel`."""
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    schema = Path("src/deep_research_client/claims/claims.yaml")
    generated = repo_root / "src/deep_research_client/claims/datamodel.py"

    gen_pydantic = shutil.which("gen-pydantic", path=str(Path(sys.executable).parent))
    if not gen_pydantic:
        pytest.skip("linkml is not installed; install the dev dependency group to check drift")

    completed = subprocess.run(
        [gen_pydantic, str(schema)], cwd=repo_root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, f"gen-pydantic failed:\n{completed.stderr}"
    assert completed.stdout == generated.read_text(encoding="utf-8"), (
        "claims/datamodel.py does not match claims.yaml. Either the schema changed "
        "or linkml was upgraded; run `just gen-datamodel` and review the diff."
    )


def test_a_claim_set_round_trips_through_json() -> None:
    """The set is the unit written to disk and read back by alignment later."""
    claims = ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="llm-atomic", model="m", prompt_version="1"),
        claims=[Claim(
            id="c1", claim_text="A causes B.", anchor_status=AnchorStatus.EXACT,
            source_span=TextSpan(start=4, end=15, text="A causes B."),
        )],
    )

    again = ClaimSet.model_validate_json(claims.model_dump_json())

    assert again == claims
    assert again.claim_list[0].anchor_status == AnchorStatus.EXACT


def test_a_moved_span_is_reported() -> None:
    """A span whose recorded text no longer matches the source is caught."""
    claims = ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="x"),
        claims=[Claim(
            id="c1", claim_text="A causes B.", anchor_status=AnchorStatus.EXACT,
            source_span=TextSpan(start=0, end=11, text="A causes B."),
        )],
    )

    assert claims.mismatched_spans("A causes B.") == []
    assert [c.id for c in claims.mismatched_spans("Now: A causes B.")] == ["c1"]
