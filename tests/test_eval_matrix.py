"""Tests for the matrix runner.

These drive real cells through the mock provider rather than stubbing the
client, so the output layout, the resume logic and the TSV writers are all
exercised end to end without reaching a network.
"""

import asyncio
import json

import pytest

from deep_research_client.client import DeepResearchClient
from deep_research_client.evaluation.datamodel import (
    AnswerSpec,
    AnswerType,
    ArmSpec,
    CellResult,
    CellStatus,
    EvalSet,
    EvalTask,
    ScoreDisposition,
)
from deep_research_client.evaluation.matrix import (
    MatrixConfig,
    RunLayout,
    _arm_params,
    load_arms,
    parse_arm_flag,
    run_matrix,
    safe_segment,
    score_by_arm,
    write_results_tsv,
)


@pytest.fixture
def mock_client(monkeypatch):
    """A client with only the mock provider registered."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    return DeepResearchClient()


@pytest.fixture
def report_eval_set() -> EvalSet:
    return EvalSet(
        name="demo",
        tasks=[
            EvalTask(id="t1", prompt="First question?", answer_type=AnswerType.REPORT),
            EvalTask(id="t2", prompt="Second question?", answer_type=AnswerType.REPORT),
        ],
    )


# ---------------------------------------------------------------------------
# Arm parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag,expected_id,expected_provider,expected_model", [
    ("falcon", "falcon", "falcon", None),
    ("openai:o3-deep-research", "openai__o3-deep-research", "openai", "o3-deep-research"),
    ("edison=falcon", "edison", "falcon", None),
    ("baseline=claude_code:opus", "baseline", "claude_code", "opus"),
])
def test_parse_arm_flag(flag, expected_id, expected_provider, expected_model):
    arm = parse_arm_flag(flag)
    assert (arm.id, arm.provider, arm.model) == (expected_id, expected_provider, expected_model)


def test_parse_arm_flag_rejects_an_empty_provider():
    with pytest.raises(ValueError, match="Could not read a provider"):
        parse_arm_flag("justanid=")


def test_load_arms_round_trips_provider_params(tmp_path):
    """A baseline arm is often the same provider with different params."""
    path = tmp_path / "arms.yaml"
    path.write_text(
        "arms:\n"
        "  - id: agent\n    provider: claude_code\n"
        "  - id: agent-noweb\n    provider: claude_code\n"
        "    description: Closed-book control\n"
        "    params:\n      allowed_tools: []\n"
    )
    arms = load_arms(path)
    assert [a.id for a in arms] == ["agent", "agent-noweb"]
    assert _arm_params(arms[0]) == {}
    assert _arm_params(arms[1]) == {"allowed_tools": []}
    assert arms[1].description == "Closed-book control"


def test_load_arms_requires_a_provider(tmp_path):
    path = tmp_path / "arms.yaml"
    path.write_text("arms:\n  - id: nameless\n")
    with pytest.raises(ValueError, match="no 'provider'"):
        load_arms(path)


@pytest.mark.parametrize("raw,expected", [
    ("LitQA2__e3b5/a4af", "LitQA2__e3b5_a4af"),
    ("MONDO:0007037", "MONDO_0007037"),
    ("../escape", "escape"),
    ("///", "unnamed"),
])
def test_safe_segment_keeps_ids_inside_the_run_directory(raw, expected):
    """Task ids come from benchmark data and become path segments."""
    assert safe_segment(raw) == expected


# ---------------------------------------------------------------------------
# Running the matrix
# ---------------------------------------------------------------------------


def test_run_matrix_writes_the_documented_layout(tmp_path, mock_client, report_eval_set):
    arms = [ArmSpec(id="a1", provider="mock"), ArmSpec(id="a2", provider="mock")]
    manifest = asyncio.run(run_matrix(
        report_eval_set, arms, MatrixConfig(output_dir=tmp_path / "run"), client=mock_client,
    ))

    root = tmp_path / "run"
    assert len(manifest.cells) == 4  # 2 tasks x 2 arms
    assert (root / "manifest.json").exists()
    assert (root / "results.tsv").exists()

    for task_id in ("t1", "t2"):
        for arm_id in ("a1", "a2"):
            cell_dir = root / task_id / arm_id
            assert (cell_dir / "prompt.md").exists()
            assert (cell_dir / "output.md").exists()
            assert (cell_dir / "cell.json").exists()


def test_report_run_writes_no_scores_file(tmp_path, mock_client, report_eval_set):
    """Report tasks need an LLM judge, so a report-only run scores nothing."""
    asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=tmp_path / "run"), client=mock_client,
    ))
    assert not (tmp_path / "run" / "scores.tsv").exists()


def test_prompt_saved_is_what_the_provider_was_sent(tmp_path, mock_client):
    """The saved prompt has to include the lettered options, or a run is unauditable."""
    eval_set = EvalSet(name="mcq", tasks=[EvalTask(
        id="m1", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine"]),
    )])
    asyncio.run(run_matrix(
        eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=tmp_path / "run"), client=mock_client,
    ))

    prompt = (tmp_path / "run" / "m1" / "a1" / "prompt.md").read_text()
    assert "Which base?" in prompt
    assert "Thymine" in prompt and "Guanine" in prompt
    assert "Answer: X" in prompt


def test_mcq_run_grades_during_the_run(tmp_path, mock_client):
    """Grading multiple choice is free, so it happens without a second pass."""
    eval_set = EvalSet(name="mcq", tasks=[EvalTask(
        id="m1", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine"]),
    )])
    manifest = asyncio.run(run_matrix(
        eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=tmp_path / "run"), client=mock_client,
    ))

    assert manifest.cells[0].disposition is not None
    assert (tmp_path / "run" / "m1" / "a1" / "answer.json").exists()
    assert (tmp_path / "run" / "scores.tsv").exists()


def test_a_failing_arm_does_not_lose_the_others(tmp_path, mock_client, report_eval_set):
    """The point of a matrix is that cells are independent."""
    arms = [ArmSpec(id="good", provider="mock"), ArmSpec(id="bad", provider="not_a_provider")]
    manifest = asyncio.run(run_matrix(
        report_eval_set, arms, MatrixConfig(output_dir=tmp_path / "run"), client=mock_client,
    ))

    by_arm: dict[str, list[CellResult]] = {}
    for cell in manifest.cells:
        by_arm.setdefault(cell.arm_id, []).append(cell)

    assert all(c.status == CellStatus.COMPLETED for c in by_arm["good"])
    assert all(c.status == CellStatus.FAILED for c in by_arm["bad"])
    assert all(c.error for c in by_arm["bad"])


def test_resume_skips_completed_cells(tmp_path, mock_client, report_eval_set):
    """An interrupted run should cost only the cells that never finished."""
    run_dir = tmp_path / "run"
    config = MatrixConfig(output_dir=run_dir)
    asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="a1", provider="mock")], config, client=mock_client,
    ))

    marker = run_dir / "t1" / "a1" / "output.md"
    marker.write_text("EDITED BY TEST")

    asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="a1", provider="mock")], config, client=mock_client,
    ))
    assert marker.read_text() == "EDITED BY TEST"

    asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=run_dir, resume=False), client=mock_client,
    ))
    assert marker.read_text() != "EDITED BY TEST"


def test_failed_cells_are_retried_on_resume(tmp_path, mock_client, report_eval_set):
    """Resume must not treat a failure as done, or a transient error is permanent."""
    run_dir = tmp_path / "run"
    asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="bad", provider="not_a_provider")],
        MatrixConfig(output_dir=run_dir), client=mock_client,
    ))

    manifest = asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="bad", provider="mock")],
        MatrixConfig(output_dir=run_dir), client=mock_client,
    ))
    assert all(c.status == CellStatus.COMPLETED for c in manifest.cells)


def test_manifest_carries_the_provenance_a_score_needs(tmp_path, mock_client):
    """A number without its dataset revision cannot be reproduced or checked."""
    eval_set = EvalSet(
        name="pinned", source="futurehouse/lab-bench", source_revision="abc123",
        is_partial=True, partial_reason="20% withheld",
        tasks=[EvalTask(id="t1", prompt="Q?", answer_type=AnswerType.REPORT)],
    )
    manifest = asyncio.run(run_matrix(
        eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=tmp_path / "run"), client=mock_client,
    ))

    assert manifest.eval_set_revision == "abc123"
    assert manifest.is_partial
    assert manifest.partial_reason == "20% withheld"
    assert manifest.client_version

    stored = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert stored["eval_set_revision"] == "abc123"


def test_empty_inputs_are_rejected(tmp_path, mock_client, report_eval_set):
    with pytest.raises(ValueError, match="no tasks"):
        asyncio.run(run_matrix(
            EvalSet(name="empty", tasks=[]), [ArmSpec(id="a", provider="mock")],
            MatrixConfig(output_dir=tmp_path / "r1"), client=mock_client,
        ))
    with pytest.raises(ValueError, match="No arms"):
        asyncio.run(run_matrix(
            report_eval_set, [], MatrixConfig(output_dir=tmp_path / "r2"), client=mock_client,
        ))


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def test_results_tsv_survives_a_multiline_error(tmp_path):
    """A traceback in an error message would otherwise corrupt the row."""
    layout = RunLayout(tmp_path)
    write_results_tsv(layout, [CellResult(
        task_id="t1", arm_id="a1", status=CellStatus.FAILED,
        error="line one\nline two\twith a tab",
    )])

    lines = layout.results_path.read_text().strip().split("\n")
    assert len(lines) == 2
    header, row = (line.split("\t") for line in lines)
    assert len(row) == len(header)
    assert row[header.index("error")] == "line one line two with a tab"


def test_score_by_arm_ignores_report_cells():
    """Only multiple-choice tasks have an option to have chosen."""
    eval_set = EvalSet(name="mixed", tasks=[
        EvalTask(id="m1", prompt="Q?", answer_type=AnswerType.MULTIPLE_CHOICE,
                 answer_spec=AnswerSpec(ideal="A")),
        EvalTask(id="r1", prompt="Q?", answer_type=AnswerType.REPORT),
    ])
    cells = [
        CellResult(task_id="m1", arm_id="a1", status=CellStatus.COMPLETED,
                   disposition=ScoreDisposition.SCORED, correct=True),
        CellResult(task_id="r1", arm_id="a1", status=CellStatus.COMPLETED),
    ]
    scores = score_by_arm(eval_set, cells)
    assert scores["a1"].total == 1
    assert scores["a1"].accuracy == pytest.approx(1.0)


def test_score_by_arm_is_empty_without_multiple_choice_tasks():
    eval_set = EvalSet(name="reports", tasks=[
        EvalTask(id="r1", prompt="Q?", answer_type=AnswerType.REPORT),
    ])
    assert score_by_arm(eval_set, []) == {}
