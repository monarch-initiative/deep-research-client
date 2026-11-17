"""The alignment datamodel and its generated code (issue #43, item 2)."""

from pathlib import Path

import pytest

from deep_research_client.claims import (
    AlignedClaimPair,
    AlignedSource,
    AlignedSourceType,
    AlignerInfo,
    ClaimAlignment,
    EvidentialRelation,
    RubricMetric,
    RubricSummary,
    SemanticRelation,
    SourceType,
)


def test_the_committed_worked_example_still_validates() -> None:
    """The docs example data must keep matching the schema if it changes later."""
    import json

    examples = Path(__file__).resolve().parent.parent / "examples" / "within-report-alignment"
    from deep_research_client.claims import ClaimSet

    claims = ClaimSet.model_validate(
        json.loads((examples / "acne_vulgaris.claims.json").read_text(encoding="utf-8"))
    )
    alignment = ClaimAlignment.model_validate(
        json.loads((examples / "acne_vulgaris.self-align.json").read_text(encoding="utf-8"))
    )
    assert len(claims.claim_list) == 267
    assert len(alignment.pair_list) == 114
    assert alignment.within_source is True
    assert alignment.source_a.id == alignment.source_b.id  # a within-source alignment


def test_aligned_source_type_covers_every_claims_source_type() -> None:
    """AlignedSourceType mirrors claims.SourceType; a new source type must be added here.

    `align._aligned_source` maps one to the other by value, so divergence would
    raise at runtime rather than here.
    """
    assert {e.value for e in SourceType} == {e.value for e in AlignedSourceType}


def test_datamodel_matches_linkml_schema() -> None:
    """claims/alignment_datamodel.py is generated; regenerate with `just gen-datamodel`."""
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    schema = Path("src/deep_research_client/claims/alignment.yaml")
    generated = repo_root / "src/deep_research_client/claims/alignment_datamodel.py"

    gen_pydantic = shutil.which("gen-pydantic", path=str(Path(sys.executable).parent))
    if not gen_pydantic:
        pytest.skip("linkml is not installed; install the dev dependency group to check drift")

    completed = subprocess.run(
        [gen_pydantic, str(schema)], cwd=repo_root, capture_output=True, text=True,
    )
    assert completed.returncode == 0, f"gen-pydantic failed:\n{completed.stderr}"
    assert completed.stdout == generated.read_text(encoding="utf-8"), (
        "alignment_datamodel.py does not match alignment.yaml. Either the schema changed "
        "or linkml was upgraded; run `just gen-datamodel` and review the diff."
    )


def _pair(confidence: float | None = None, rationale: str | None = None) -> AlignedClaimPair:
    return AlignedClaimPair(
        id="p1", claim_a_id="c1", claim_b_id="phenotypes[0]",
        claim_a_text="A.", claim_b_text="A.",
        semantic_relation=SemanticRelation.IDENTICAL,
        evidential_relation=EvidentialRelation.BOTH_SECONDARY,
        confidence=confidence, rationale=rationale,
    )


def _alignment(pairs) -> ClaimAlignment:
    return ClaimAlignment(
        source_a=AlignedSource(id="a.md", source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id="b.yaml", source_type=AlignedSourceType.STRUCTURED_DOCUMENT),
        aligner=AlignerInfo(name="llm-pairwise", model="m", prompt_version="1"),
        pairs=pairs,
    )


def test_an_alignment_round_trips_through_json() -> None:
    """The alignment is the unit written to disk and read back by rubrics later."""
    alignment = _alignment([_pair(confidence=0.9, rationale="same.")])
    loaded = ClaimAlignment.model_validate(alignment.model_dump(mode="json"))
    assert loaded.pair_list[0].semantic_relation == SemanticRelation.IDENTICAL
    assert loaded.pair_list[0].evidential_relation == EvidentialRelation.BOTH_SECONDARY


def test_a_pair_with_confidence_outside_0_to_1_is_refused() -> None:
    """A confidence a reader would misread is caught on construction and on load."""
    with pytest.raises(ValueError, match="confidence 1.5, outside"):
        _alignment([_pair(confidence=1.5)])


@pytest.mark.parametrize(
    ("numerator", "denominator", "value", "ok"),
    [(1, 2, 0.5, True), (0, 0, None, True), (1, 2, 0.9, False), (0, 0, 0.0, False), (3, 2, 1.5, False)],
)
def test_a_rubric_metric_value_must_follow_from_its_counts(numerator, denominator, value, ok) -> None:
    """value is numerator/denominator, or null exactly when denominator is zero."""
    metric = {"name": "completeness", "numerator": numerator, "denominator": denominator, "value": value}
    if ok:
        RubricSummary(reference="b", metrics=[RubricMetric(**metric)])
    else:
        with pytest.raises(ValueError):
            RubricSummary(reference="b", metrics=[RubricMetric(**metric)])
