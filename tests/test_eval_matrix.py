"""Tests for the matrix runner.

These drive real cells through the mock provider rather than stubbing the
client, so the output layout, the resume logic and the TSV writers are all
exercised end to end without reaching a network.
"""

import asyncio
import json
from pathlib import Path

import pytest

from deep_research_client.client import DeepResearchClient
from deep_research_client.evaluation import mcq
from deep_research_client.evaluation.datamodel import (
    AnswerSpec,
    AnswerType,
    ArmSpec,
    CellResult,
    CellStatus,
    EvalSet,
    EvalTask,
    MetadataItem,
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


@pytest.mark.parametrize("raw,stem", [
    ("LitQA2__e3b5/a4af", "LitQA2__e3b5_a4af"),
    ("MONDO:0007037", "MONDO_0007037"),
    ("../escape", "escape"),
    ("///", "unnamed"),
    ("..", "unnamed"),
])
def test_safe_segment_keeps_ids_inside_the_run_directory(raw, stem):
    """Task ids come from benchmark data and become path segments.

    A rewritten id keeps a readable stem but must not be usable to climb out of
    the run directory or to name a file outside it.
    """
    segment = safe_segment(raw)
    assert segment.startswith(stem)
    assert "/" not in segment and "\\" not in segment
    assert not segment.startswith(".")
    assert Path("/run", segment).resolve().parent == Path("/run")


def test_safe_segment_leaves_an_already_safe_id_alone():
    """No suffix on ids that need no rewriting, so paths stay readable."""
    assert safe_segment("LitQA2__abc-123") == "LitQA2__abc-123"
    assert safe_segment("plain_id") == "plain_id"


@pytest.mark.parametrize("first,second", [
    ("MONDO:0007037", "MONDO_0007037"),
    ("a/b", "a_b"),
    ("..", "unnamed"),
])
def test_distinct_ids_never_share_a_directory(first, second):
    """Sanitising is lossy, so it must not merge two tasks into one cell.

    Uniqueness is checked on the raw id, but the directory name is the
    sanitized one. Without disambiguation the second task would overwrite the
    first's output and, on resume, report the first's result as its own.
    """
    assert first != second
    assert safe_segment(first) != safe_segment(second)


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


def test_mcq_run_grades_only_when_asked(tmp_path, mock_client):
    """A run materialises results; grading is opt-in.

    The current grader reads answers out of prose with regular expressions and
    has produced plausible-but-wrong numbers before, so it must never run by
    accident. Materialising the outputs is what a run is for; how they are
    scored can be decided later without paying for the outputs again.
    """
    eval_set = EvalSet(name="mcq", tasks=[EvalTask(
        id="m1", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine"]),
    )])
    ungraded = asyncio.run(run_matrix(
        eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=tmp_path / "plain"), client=mock_client,
    ))
    assert ungraded.cells[0].disposition is None
    assert not (tmp_path / "plain" / "scores.tsv").exists()
    assert not (tmp_path / "plain" / "m1" / "a1" / "answer.json").exists()
    # The response itself is still on disk, so it can be scored later.
    assert (tmp_path / "plain" / "m1" / "a1" / "output.md").exists()

    graded = asyncio.run(run_matrix(
        eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=tmp_path / "graded", grade=True), client=mock_client,
    ))
    assert graded.cells[0].disposition is not None
    assert (tmp_path / "graded" / "m1" / "a1" / "answer.json").exists()
    assert (tmp_path / "graded" / "scores.tsv").exists()


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


# ---------------------------------------------------------------------------
# End-to-end scoring, via the mock provider
#
# The mock cannot know which option is correct, so it answers by position. That
# is what makes these assertions possible: option order is deterministic for a
# given task id, so the score an "always A" arm deserves can be computed here
# independently of the code under test, and asserted exactly.
# ---------------------------------------------------------------------------


def _mcq_eval_set(n: int = 12, abstention: str | None = None) -> EvalSet:
    return EvalSet(name="scored", tasks=[
        EvalTask(
            id=f"q{i}",
            prompt=f"Question {i}?",
            answer_type=AnswerType.MULTIPLE_CHOICE,
            answer_spec=AnswerSpec(
                ideal=f"right-{i}",
                distractors=[f"wrong-{i}-a", f"wrong-{i}-b", f"wrong-{i}-c"],
                abstention_option=abstention,
            ),
        )
        for i in range(n)
    ])


def _mock_arm(arm_id: str, policy: str) -> ArmSpec:
    return ArmSpec(
        id=arm_id, provider="mock",
        params=[MetadataItem(key="answer_policy", value=json.dumps(policy))],
    )


def _expected_correct_for_first(eval_set: EvalSet) -> int:
    """How many tasks an 'always answer A' arm should get right."""
    return sum(
        1 for task in (eval_set.tasks or []) if mcq.present_choices(task)[0].is_ideal
    )


def test_always_first_arm_scores_exactly_what_it_should(tmp_path, mock_client):
    eval_set = _mcq_eval_set()
    expected = _expected_correct_for_first(eval_set)

    manifest = asyncio.run(run_matrix(
        eval_set, [_mock_arm("always-a", "first")],
        MatrixConfig(output_dir=tmp_path / "run", grade=True), client=mock_client,
    ))

    score = score_by_arm(eval_set, manifest.cells)["always-a"]
    assert score.total == len(eval_set.tasks)
    assert score.attempted == len(eval_set.tasks)  # it answers every question
    assert score.correct == expected
    assert score.accuracy == pytest.approx(expected / len(eval_set.tasks))
    assert score.coverage == pytest.approx(1.0)
    assert score.precision == pytest.approx(expected / len(eval_set.tasks))
    assert score.extraction_failures == 0


def test_always_last_arm_abstains_on_every_question(tmp_path, mock_client):
    """With an abstention offered last, 'always last' must read as declining."""
    eval_set = _mcq_eval_set(abstention="Insufficient information to answer this question.")

    manifest = asyncio.run(run_matrix(
        eval_set, [_mock_arm("decliner", "last")],
        MatrixConfig(output_dir=tmp_path / "run", grade=True), client=mock_client,
    ))

    score = score_by_arm(eval_set, manifest.cells)["decliner"]
    assert score.abstained == len(eval_set.tasks)
    assert score.attempted == 0
    assert score.coverage == pytest.approx(0.0)
    assert score.correct == 0
    # Precision over zero attempts is zero, not a division error.
    assert score.precision == pytest.approx(0.0)


def test_an_echoing_arm_scores_the_same_as_a_terse_one(tmp_path, mock_client):
    """The regression, end to end.

    A provider that restates every option before answering must score exactly
    the same as one that answers tersely. Before the extractor learned to strip
    restatements, the echoing arm scored as abstaining on every question.
    """
    eval_set = _mcq_eval_set(abstention="Insufficient information to answer this question.")

    manifest = asyncio.run(run_matrix(
        eval_set, [_mock_arm("terse", "first"), _mock_arm("echoing", "echo")],
        MatrixConfig(output_dir=tmp_path / "run", grade=True), client=mock_client,
    ))

    scores = score_by_arm(eval_set, manifest.cells)
    assert scores["echoing"].correct == scores["terse"].correct
    assert scores["echoing"].accuracy == pytest.approx(scores["terse"].accuracy)
    assert scores["echoing"].abstained == 0
    assert scores["echoing"].correct == _expected_correct_for_first(eval_set)


def test_silent_arm_is_extraction_failure_not_abstention(tmp_path, mock_client):
    """A provider that never answers must not be recorded as declining."""
    eval_set = _mcq_eval_set(abstention="Insufficient information to answer this question.")

    manifest = asyncio.run(run_matrix(
        eval_set, [_mock_arm("silent", "none")],
        MatrixConfig(output_dir=tmp_path / "run", grade=True), client=mock_client,
    ))

    score = score_by_arm(eval_set, manifest.cells)["silent"]
    assert score.abstained == 0
    assert score.extraction_failures == len(eval_set.tasks)
    assert score.coverage == pytest.approx(0.0)


def test_scores_tsv_matches_the_computed_scores(tmp_path, mock_client):
    """The written file is the artifact people read; it must agree with the API."""
    eval_set = _mcq_eval_set()
    manifest = asyncio.run(run_matrix(
        eval_set, [_mock_arm("always-a", "first")],
        MatrixConfig(output_dir=tmp_path / "run", grade=True), client=mock_client,
    ))

    score = score_by_arm(eval_set, manifest.cells)["always-a"]
    rows = (tmp_path / "run" / "scores.tsv").read_text().strip().split("\n")
    header, row = (line.split("\t") for line in rows)
    values = dict(zip(header, row))

    assert values["arm_id"] == "always-a"
    assert int(values["correct"]) == score.correct
    assert float(values["accuracy"]) == pytest.approx(score.accuracy, abs=1e-4)
    assert float(values["coverage"]) == pytest.approx(score.coverage, abs=1e-4)


# ---------------------------------------------------------------------------
# Resume semantics
# ---------------------------------------------------------------------------


def test_grading_a_run_that_was_materialised_earlier(tmp_path, mock_client):
    """The documented workflow: materialise now, decide grading later.

    Resuming with grading on must score the responses already on disk rather
    than reuse their ungraded cells. Without this the whole separation is
    hollow — the run says it can be scored later, and later produces silence.
    """
    eval_set = _mcq_eval_set(6)
    arm = _mock_arm("always-a", "first")
    run_dir = tmp_path / "run"

    first = asyncio.run(run_matrix(
        eval_set, [arm], MatrixConfig(output_dir=run_dir), client=mock_client,
    ))
    assert all(c.disposition is None for c in first.cells)
    assert not (run_dir / "scores.tsv").exists()

    second = asyncio.run(run_matrix(
        eval_set, [arm], MatrixConfig(output_dir=run_dir, grade=True), client=mock_client,
    ))
    assert all(c.disposition is not None for c in second.cells)

    score = score_by_arm(eval_set, second.cells)["always-a"]
    assert score.total == len(eval_set.tasks)
    assert score.correct == _expected_correct_for_first(eval_set)
    assert (run_dir / "scores.tsv").exists()


def test_grading_a_resumed_run_does_not_call_the_provider_again(tmp_path, mock_client):
    """Scoring later must not cost the provider calls again."""
    eval_set = _mcq_eval_set(3)
    arm = _mock_arm("always-a", "first")
    run_dir = tmp_path / "run"

    asyncio.run(run_matrix(
        eval_set, [arm], MatrixConfig(output_dir=run_dir), client=mock_client,
    ))
    marker = "SENTINEL RESPONSE\n\nAnswer: A"
    (run_dir / "q0" / "always-a" / "output.md").write_text(marker)

    asyncio.run(run_matrix(
        eval_set, [arm], MatrixConfig(output_dir=run_dir, grade=True), client=mock_client,
    ))
    # Untouched: graded from disk, not re-fetched.
    assert (run_dir / "q0" / "always-a" / "output.md").read_text() == marker


def test_resume_reruns_a_cell_whose_question_changed(tmp_path, mock_client):
    """Editing the eval set must not leave stale answers in a new manifest.

    A new distractor reshuffles the options, so the stored answer was given to a
    different question than the one this run records.
    """
    run_dir = tmp_path / "run"
    task = EvalTask(
        id="m1", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine"]),
    )
    asyncio.run(run_matrix(
        EvalSet(name="v1", tasks=[task]), [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=run_dir), client=mock_client,
    ))
    (run_dir / "m1" / "a1" / "output.md").write_text("STALE")

    edited = EvalTask(
        id="m1", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine", "Cytosine"]),
    )
    asyncio.run(run_matrix(
        EvalSet(name="v2", tasks=[edited]), [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=run_dir), client=mock_client,
    ))
    assert (run_dir / "m1" / "a1" / "output.md").read_text() != "STALE"
    assert "Cytosine" in (run_dir / "m1" / "a1" / "prompt.md").read_text()


def test_summary_files_exist_before_the_run_finishes(tmp_path, mock_client):
    """An interrupted run must still leave a readable results.tsv and manifest."""
    seen: list[Path] = []
    eval_set = _mcq_eval_set(4)

    def on_cell(_cell):
        seen.append(tmp_path / "run" / "results.tsv")
        assert (tmp_path / "run" / "results.tsv").exists()
        assert (tmp_path / "run" / "manifest.json").exists()

    asyncio.run(run_matrix(
        eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=tmp_path / "run", on_cell=on_cell), client=mock_client,
    ))
    assert len(seen) == len(eval_set.tasks)


def test_a_resumed_cell_with_no_saved_response_is_rerun(tmp_path, mock_client):
    """Returning it ungraded would drop it from the aggregate silently."""
    eval_set = _mcq_eval_set(2)
    arm = _mock_arm("always-a", "first")
    run_dir = tmp_path / "run"

    asyncio.run(run_matrix(
        eval_set, [arm], MatrixConfig(output_dir=run_dir), client=mock_client,
    ))
    (run_dir / "q0" / "always-a" / "output.md").unlink()

    manifest = asyncio.run(run_matrix(
        eval_set, [arm], MatrixConfig(output_dir=run_dir, grade=True), client=mock_client,
    ))
    assert all(c.disposition is not None for c in manifest.cells)
    assert (run_dir / "q0" / "always-a" / "output.md").exists()
    assert score_by_arm(eval_set, manifest.cells)["always-a"].total == 2


def test_cell_files_are_written_atomically(tmp_path, mock_client, report_eval_set):
    """A half-written cell.json raises on resume and takes the whole run down.

    Atomic writes leave no partial files behind, so no stray temporaries and
    every artefact parses.
    """
    run_dir = tmp_path / "run"
    asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=run_dir), client=mock_client,
    ))

    leftovers = [p for p in run_dir.rglob(".*") if p.is_file()]
    assert leftovers == [], f"temporary files left behind: {leftovers}"
    for cell_json in run_dir.rglob("cell.json"):
        json.loads(cell_json.read_text())


def test_written_files_get_the_mode_an_ordinary_write_would_give(
    tmp_path, mock_client, report_eval_set
):
    """Atomic writes must not silently make everything owner-only.

    mkstemp creates 0600 and os.replace keeps that mode, so without restoring
    the default a shared benchmark cache stops being readable by anyone but
    whoever fetched it first.

    Compared against a sibling written the ordinary way rather than against an
    absolute mode: what the fix promises is parity with a normal write, and
    asserting group-readability instead would fail under a strict umask on a
    machine where the code is behaving exactly as intended.
    """
    import stat

    run_dir = tmp_path / "run"
    asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=run_dir), client=mock_client,
    ))

    reference = run_dir / "reference.txt"
    reference.write_text("written the ordinary way")
    expected = stat.S_IMODE(reference.stat().st_mode)

    for written in (run_dir / "results.tsv", run_dir / "manifest.json"):
        mode = stat.S_IMODE(written.stat().st_mode)
        assert mode == expected, f"{written.name} is {oct(mode)}, not {oct(expected)}"


def test_atomic_write_preserves_an_existing_files_mode(tmp_path):
    """A cache deliberately opened up to a group must not close on refresh."""
    import stat

    from deep_research_client.evaluation._fs import atomic_write

    path = tmp_path / "cache.json"
    path.write_text("{}")
    path.chmod(0o664)

    atomic_write(path, '{"refreshed": true}')
    assert stat.S_IMODE(path.stat().st_mode) == 0o664
    assert path.read_text() == '{"refreshed": true}'
