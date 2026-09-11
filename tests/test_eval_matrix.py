"""Tests for the matrix runner.

These drive real cells through the mock provider rather than stubbing the
client, so the output layout, the resume logic and the TSV writers are all
exercised end to end without reaching a network.
"""

import ast
import asyncio
import json
import os
import pathlib
import re
from pathlib import Path

import pytest

from deep_research_client.client import DeepResearchClient
from deep_research_client.models import CacheConfig
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
def mock_client(monkeypatch, tmp_path_factory):
    """A client with only the mock provider registered, and no cache.

    Caching is off deliberately. `DeepResearchClient()` defaults to a cache in
    ``~/.deep_research_cache`` shared by every run on the machine, so a mock
    response computed once is replayed for every later run of a test that asks
    the same question -- and these tests do ask stable questions. That makes a
    test pass on data recorded before the code it is meant to exercise existed:
    a tripwire raised inside `_mock_answer` never fired, while the response
    still arrived complete with its answer line.
    """
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    return DeepResearchClient(cache_config=CacheConfig(
        enabled=False, directory=str(tmp_path_factory.mktemp("nocache")),
    ))


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
        "    params:\n      allowed_tools: []\n", encoding="utf-8")
    arms = load_arms(path)
    assert [a.id for a in arms] == ["agent", "agent-noweb"]
    assert _arm_params(arms[0]) == {}
    assert _arm_params(arms[1]) == {"allowed_tools": []}
    assert arms[1].description == "Closed-book control"


def test_load_arms_requires_a_provider(tmp_path):
    path = tmp_path / "arms.yaml"
    path.write_text("arms:\n  - id: nameless\n", encoding="utf-8")
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

    prompt = (tmp_path / "run" / "m1" / "a1" / "prompt.md").read_text(encoding="utf-8")
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
    marker.write_text("EDITED BY TEST", encoding="utf-8")

    asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="a1", provider="mock")], config, client=mock_client,
    ))
    assert marker.read_text(encoding="utf-8") == "EDITED BY TEST"

    asyncio.run(run_matrix(
        report_eval_set, [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=run_dir, resume=False), client=mock_client,
    ))
    assert marker.read_text(encoding="utf-8") != "EDITED BY TEST"


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

    stored = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
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

    lines = layout.results_path.read_text(encoding="utf-8").strip().split("\n")
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


def test_a_question_that_looks_like_an_option_does_not_derail_the_policy(
    tmp_path, mock_client,
):
    """The mock's whole value is that its score is computable in advance.

    `_MCQ_OPTION` matched anywhere in the prompt, so a question opening with
    "E. coli ..." parsed as option E -- ahead of the real A/B/C. An arm with
    answer_policy="first" then answered E, a letter never offered, and the cell
    scored EXTRACTION_FAILED instead of the first position the policy promises.

    No LitQA2 question happens to trip this (checked: 0 of 199), so the baseline
    this provider produced stands. The guarantee is what is under test, not the
    corpus that currently exercises it.
    """
    eval_set = EvalSet(name="ecoli", tasks=[
        EvalTask(
            id="ecoli",
            prompt="E. coli grows anaerobically under which condition?",
            answer_type=AnswerType.MULTIPLE_CHOICE,
            answer_spec=AnswerSpec(
                ideal="Fermentation", distractors=["Respiration", "Glycolysis"],
            ),
        ),
    ])
    offered = [c.letter for c in mcq.present_choices(eval_set.tasks[0])]

    run_dir = tmp_path / "run"
    manifest = asyncio.run(run_matrix(
        eval_set, [_mock_arm("always-a", "first")],
        MatrixConfig(output_dir=run_dir, grade=True), client=mock_client,
    ))

    # The letter written, which is what the fix changes.
    response = (run_dir / "ecoli" / "always-a" / "output.md").read_text(encoding="utf-8")
    answered = re.findall(r"^Answer:\s*([A-Z])\s*$", response, re.MULTILINE)
    assert answered, "the mock wrote no answer line at all"
    assert answered[-1] in offered, (
        f"mock answered {answered[-1]!r}, which was never offered ({offered})"
    )
    assert answered[-1] == offered[0], "answer_policy='first' must name position one"

    # And the harm it caused: a letter off the list cannot be extracted, so an
    # arm that answers every question was recorded as having answered none.
    score = score_by_arm(eval_set, manifest.cells)["always-a"]
    assert score.extraction_failures == 0
    assert score.attempted == 1


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
    rows = (tmp_path / "run" / "scores.tsv").read_text(encoding="utf-8").strip().split("\n")
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
    (run_dir / "q0" / "always-a" / "output.md").write_text(marker, encoding="utf-8")

    asyncio.run(run_matrix(
        eval_set, [arm], MatrixConfig(output_dir=run_dir, grade=True), client=mock_client,
    ))
    # Untouched: graded from disk, not re-fetched.
    assert (run_dir / "q0" / "always-a" / "output.md").read_text(encoding="utf-8") == marker


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
    (run_dir / "m1" / "a1" / "output.md").write_text("STALE", encoding="utf-8")

    edited = EvalTask(
        id="m1", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine", "Cytosine"]),
    )
    asyncio.run(run_matrix(
        EvalSet(name="v2", tasks=[edited]), [ArmSpec(id="a1", provider="mock")],
        MatrixConfig(output_dir=run_dir), client=mock_client,
    ))
    assert (run_dir / "m1" / "a1" / "output.md").read_text(encoding="utf-8") != "STALE"
    assert "Cytosine" in (run_dir / "m1" / "a1" / "prompt.md").read_text(encoding="utf-8")


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
        json.loads(cell_json.read_text(encoding="utf-8"))


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
    reference.write_text("written the ordinary way", encoding="utf-8")
    expected = stat.S_IMODE(reference.stat().st_mode)

    for written in (run_dir / "results.tsv", run_dir / "manifest.json"):
        mode = stat.S_IMODE(written.stat().st_mode)
        assert mode == expected, f"{written.name} is {oct(mode)}, not {oct(expected)}"


def test_atomic_write_preserves_an_existing_files_mode(tmp_path):
    """A cache deliberately opened up to a group must not close on refresh."""
    import stat

    from deep_research_client.evaluation._fs import atomic_write

    path = tmp_path / "cache.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o664)

    atomic_write(path, '{"refreshed": true}')
    assert stat.S_IMODE(path.stat().st_mode) == 0o664
    assert path.read_text(encoding="utf-8") == '{"refreshed": true}'


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------


def _cache_client(tmp_path, monkeypatch, enabled=True):
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    return DeepResearchClient(cache_config=CacheConfig(
        enabled=enabled, directory=str(tmp_path / "cache"),
    ))


def test_a_replayed_cell_is_recorded_as_cached(tmp_path, monkeypatch):
    """A replay is a measurement that did not happen in the run reporting it.

    The cache re-stamps `start_time`, `end_time` and `duration_seconds` for the
    current run, so nothing downstream can tell a months-old report from a live
    one. `ResearchResult.cached` already says which it was; `_run_cell` copied
    five other fields off the result and dropped that one.
    """
    client = _cache_client(tmp_path, monkeypatch)
    eval_set = EvalSet(name="c", tasks=[
        EvalTask(id="t1", prompt="Cache probe question?", answer_type=AnswerType.REPORT),
    ])

    first = asyncio.run(run_matrix(
        eval_set, [_mock_arm("a", "none")],
        MatrixConfig(output_dir=tmp_path / "r1"), client=client,
    ))
    assert first.cells[0].cached is False

    second = asyncio.run(run_matrix(
        eval_set, [_mock_arm("a", "none")],
        MatrixConfig(output_dir=tmp_path / "r2"), client=client,
    ))
    assert second.cells[0].cached is True, (
        "a second run over the same prompt was served from the cache but "
        "recorded as though the provider had answered"
    )


def test_two_arms_with_the_same_configuration_are_one_sample(tmp_path, monkeypatch):
    """The run-to-run variance check a matrix exists for.

    `--arm a=falcon --arm b=falcon` is how you ask what a provider's spread
    looks like. The cache key is (prompt, provider, model, params), so the
    second arm replays the first and `results.tsv` reports one sample as two.
    Run serially, because concurrently the two arms race the cache and both
    can miss -- which makes the defect intermittent rather than absent.
    """
    client = _cache_client(tmp_path, monkeypatch)
    eval_set = EvalSet(name="v", tasks=[
        EvalTask(id="t1", prompt="Variance probe?", answer_type=AnswerType.REPORT),
    ])

    manifest = asyncio.run(run_matrix(
        eval_set, [_mock_arm("a", "none"), _mock_arm("b", "none")],
        MatrixConfig(output_dir=tmp_path / "run", concurrency=1), client=client,
    ))

    # Which arm is the replay depends on execution order, and asserting that
    # made the test order-dependent -- it failed alone and passed with the
    # file. The property is that exactly one of the two was measured.
    replayed = [c.arm_id for c in manifest.cells if c.cached]
    assert len(manifest.cells) == 2
    assert len(replayed) == 1, (
        f"expected one of two identical arms to be a replay, got {replayed}"
    )


def test_disabling_the_cache_makes_every_cell_a_live_call(tmp_path, monkeypatch):
    """`--no-resume` re-runs a cell; it does not re-call the provider.

    Without a way to defeat the cache, a user who suspects a bad run cannot
    force a real one -- the same bytes come back under a fresh duration.
    """
    client = _cache_client(tmp_path, monkeypatch, enabled=False)
    eval_set = EvalSet(name="n", tasks=[
        EvalTask(id="t1", prompt="No-cache probe?", answer_type=AnswerType.REPORT),
    ])
    config = MatrixConfig(output_dir=tmp_path / "r1")

    asyncio.run(run_matrix(eval_set, [_mock_arm("a", "none")], config, client=client))
    second = asyncio.run(run_matrix(
        eval_set, [_mock_arm("a", "none")],
        MatrixConfig(output_dir=tmp_path / "r2"), client=client,
    ))
    assert second.cells[0].cached is False


def test_the_cache_column_is_written_to_results_tsv(tmp_path, monkeypatch):
    """Visible in the artefact a score is computed from, not only in memory."""
    client = _cache_client(tmp_path, monkeypatch)
    eval_set = EvalSet(name="c", tasks=[
        EvalTask(id="t1", prompt="TSV cache probe?", answer_type=AnswerType.REPORT),
    ])
    run_dir = tmp_path / "run"
    asyncio.run(run_matrix(
        eval_set, [_mock_arm("a", "none")],
        MatrixConfig(output_dir=run_dir), client=client,
    ))

    header, row = (run_dir / "results.tsv").read_text(encoding="utf-8").splitlines()[:2]
    assert "cached" in header.split("\t")
    assert row.split("\t")[header.split("\t").index("cached")] == "false"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_a_resumed_run_survives_non_ascii_under_an_ascii_locale(tmp_path):
    """Every write here is UTF-8 by argument; the reads were UTF-8 by luck.

    `atomic_write` passes an explicit encoding, for the reason its docstring
    gives -- reports carry non-ASCII and the bytes should not depend on the
    machine. The matching reads were bare `read_text()`, which uses the
    interpreter's locale encoding. Where that bites is `_completed_cell`,
    called from `one()` above the semaphore and outside any `try`: a µ or a °C
    in a prompt (LitQA2 has plenty) aborted the entire resumed matrix with a
    traceback out of `as_completed` -- not one FAILED cell, the whole run.

    Run in a subprocess, because the condition cannot be created in-process.
    Monkeypatching `locale.getpreferredencoding` does nothing: `TextIOWrapper`
    reads `locale.getencoding()` at the C level. And the environment alone is
    not enough either -- PEP 538 coercion turns `LC_ALL=C` into C.UTF-8, so
    `PYTHONCOERCECLOCALE=0` and `PYTHONUTF8=0` are both needed to get an
    interpreter whose preferred encoding is really ASCII. That an in-process
    version of this test passed with the fix reverted is exactly why it is
    written this way.
    """
    import subprocess
    import sys

    # The child keeps its non-ASCII as \\u escapes, so the script file is itself
    # pure ASCII: an interpreter configured for ASCII must be able to read it.
    # Simplifying those to literals would make the script unreadable by the very
    # interpreter it exists to configure.
    script = tmp_path / "resume_under_ascii.py"
    script.write_text(
        "import asyncio, codecs, os, sys, locale\n"
        # Exit 77 means "could not get an ASCII interpreter" -- a skip, not a
        # failure. Normalised through codecs.lookup because the same encoding is
        # spelt ANSI_X3.4-1968 on glibc and US-ASCII on macOS, and on Windows the
        # coercion variables do not apply at all.
        # The lookup itself can raise: getpreferredencoding can return '' or
        # an alias no codec is registered for on some libc/locale pairs. That
        # is "could not get an ASCII interpreter" too, not the defect.
        "try:\n"
        "    _enc = codecs.lookup(locale.getpreferredencoding(False)).name\n"
        "except LookupError:\n"
        "    print('unregistered')\n"
        "    sys.exit(77)\n"
        "if _enc != 'ascii':\n"
        "    print(_enc)\n"
        "    sys.exit(77)\n"
        "os.environ['ENABLE_MOCK_PROVIDER'] = 'true'\n"
        "from pathlib import Path\n"
        "from deep_research_client.client import DeepResearchClient\n"
        "from deep_research_client.models import CacheConfig\n"
        "from deep_research_client.evaluation.datamodel import (\n"
        "    AnswerType, ArmSpec, EvalSet, EvalTask)\n"
        "from deep_research_client.evaluation.matrix import run_matrix, MatrixConfig\n"
        "out = Path(sys.argv[1])\n"
        "client = DeepResearchClient(cache_config=CacheConfig(\n"
        "    enabled=False, directory=str(out / 'cache')))\n"
        "es = EvalSet(name='enc', tasks=[EvalTask(\n"
        "    id='t1', prompt='Concentration was 5 \\u00b5M at 37 \\u00b0C - why?',\n"
        "    answer_type=AnswerType.REPORT)])\n"
        "arm = ArmSpec(id='a', provider='mock')\n"
        "asyncio.run(run_matrix(es, [arm], MatrixConfig(output_dir=out / 'run'),\n"
        "                       client=client))\n"
        "m = asyncio.run(run_matrix(es, [arm], MatrixConfig(output_dir=out / 'run'),\n"
        "                           client=client))\n"
        "assert len(m.cells) == 1, m.cells\n"
        "assert m.cells[0].resumed is True\n"
        "print('OK')\n",
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONUTF8": "0",
    }
    done = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        capture_output=True, env=env,
        # The parent decodes the child's output, so `text=True` alone would use
        # the *parent's* locale -- the sixth locale-dependent construct in this
        # repo, inside the test that exists for locale-dependence. On a failure
        # the tail printed is a traceback over a prompt containing µ and °C.
        encoding="utf-8", errors="replace",
    )
    if done.returncode == 77:
        pytest.skip(
            "this interpreter cannot be put into an ASCII locale "
            f"(got {done.stdout.strip()!r}); the structural check still applies"
        )
    assert done.returncode == 0, (
        "a resumed run died under an ASCII locale:\n"
        + done.stdout[-2000:] + done.stderr[-2000:]
    )
    assert "OK" in done.stdout


def _positions(func, *, drop_self: bool = False) -> tuple[int, int | None]:
    """Where ``encoding`` and ``mode`` sit in a call to `func`.

    Read from the signature rather than written down. Hand-copied indices are
    themselves a claim, and three of the four added in the previous commit were
    wrong -- `os.fdopen`'s encoding is at 3, not 2, and the `tempfile` factories
    take `mode` as their *first* parameter rather than a leading file argument.
    Derived here, they cannot drift from the stdlib or from a future Python.

    `drop_self` is for methods reached through their class: `Path.open` has
    `self` at 0, which a call like ``p.open("w")`` does not pass.
    """
    import inspect

    names = list(inspect.signature(func).parameters)
    if drop_self:
        names = names[1:]
    return names.index("encoding"), (names.index("mode") if "mode" in names else None)


def _text_io_signatures() -> dict[tuple[str, bool], tuple[int, int | None, bool]]:
    """Every constructor that can open a locale-encoded text handle.

    Enumerated from what the property can be violated *with*, not from what the
    code currently happens to call. Three consecutive rounds of review found the
    earlier versions of this check one construct short of the one that mattered
    -- first `open`, then `write_text`, then `os.fdopen`, which is how
    `atomic_write` writes every artifact in the package and is the very line
    whose docstring this invariant comes from.

    Both spellings are registered for everything reachable through a module,
    because ``from tempfile import NamedTemporaryFile`` is an `ast.Name` call
    and would otherwise be skipped in silence. Only `open` differs between the
    two, since the builtin carries a leading `file` argument where the bound
    method carries `self`.

    The third element inverts the question: `open` and `os.fdopen` are text
    unless told otherwise, while the `tempfile` factories are binary unless told
    otherwise, so only a tempfile call that *names* a text mode needs an
    encoding.
    """
    import builtins
    import io
    import os
    import tempfile

    table: dict[tuple[str, bool], tuple[int, int | None, bool]] = {
        ("read_text", True): (*_positions(pathlib.Path.read_text, drop_self=True), False),
        ("write_text", True): (*_positions(pathlib.Path.write_text, drop_self=True), False),
        ("open", True): (*_positions(pathlib.Path.open, drop_self=True), False),
        ("open", False): (*_positions(builtins.open), False),
    }
    # Reached through a module, so the receiver is not an argument and the two
    # spellings share one set of positions.
    for name, func, binary_default in [
        ("fdopen", os.fdopen, False),
        ("TextIOWrapper", io.TextIOWrapper, False),
        ("NamedTemporaryFile", tempfile.NamedTemporaryFile, True),
        ("TemporaryFile", tempfile.TemporaryFile, True),
        ("SpooledTemporaryFile", tempfile.SpooledTemporaryFile, True),
    ]:
        entry = (*_positions(func), binary_default)
        table[(name, True)] = entry
        table[(name, False)] = entry
    return table


#: Mode strings are drawn from this alphabet. Checked rather than searching for
#: a "b", because `mode_at` is only a guess for a method call on an unknown
#: object: `zipfile.ZipFile(p).open("notebook.md")` would otherwise have its
#: *filename* read as a mode, find the "b" in "notebook", and be skipped -- a
#: false negative, in the construct the skip is written about.
_MODE_CHARACTERS = set("rwxabt+")


def _is_binary_mode(node: "ast.expr") -> bool:
    """Whether this argument is a mode string that opens a binary handle."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return False
    mode = node.value
    if not mode or not set(mode) <= _MODE_CHARACTERS:
        return False  # not a mode at all; treat the call as text
    return "b" in mode


def _zipfile_bindings(tree: "ast.AST") -> set[str]:
    """Names bound to a `ZipFile` in this module.

    So `zf.open(member)` can be recognised as the bytes-yielding call it is,
    from the constructor rather than from how the variable is spelled -- a skip
    list of likely names is a false negative for every other spelling, in the
    one construct it claims to be about. Both `zipfile.ZipFile(...)` and a
    direct `ZipFile(...)` import are recognised.

    Module-scoped rather than function-scoped, so a name bound to a ZipFile
    anywhere in a file exempts `<name>.open(...)` throughout it. That is a known
    over-reach, kept because narrowing it costs a scope walk for a construct
    this repo uses twice.
    """
    def is_zipfile_call(node: "ast.expr") -> bool:
        return isinstance(node, ast.Call) and (
            getattr(node.func, "attr", None) == "ZipFile"
            or getattr(node.func, "id", None) == "ZipFile"
        )

    bound = {
        item.optional_vars.id
        for stmt in ast.walk(tree)
        if isinstance(stmt, ast.With)
        for item in stmt.items
        if isinstance(item.optional_vars, ast.Name) and is_zipfile_call(item.context_expr)
    }
    bound |= {
        stmt.targets[0].id
        for stmt in ast.walk(tree)
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and is_zipfile_call(stmt.value)
    }
    return bound


def _offenders_in_tree(tree: "ast.AST", where: str, line_offset: int = 0) -> list[str]:
    """Calls in one parsed tree that open a text handle without an encoding."""
    signatures = _text_io_signatures()
    zipfile_handles = _zipfile_bindings(tree)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        if isinstance(func, ast.Attribute):
            name, is_method = func.attr, True
            owner = getattr(func.value, "id", "")
            if (owner, name) == ("os", "open"):
                continue  # takes and returns a file descriptor
            if name == "open" and owner in zipfile_handles:
                continue  # ZipFile.open always yields bytes
        elif isinstance(func, ast.Name):
            name, is_method = func.id, False
        else:
            continue

        signature = signatures.get((name, is_method))
        if signature is None:
            continue
        encoding_at, mode_at, binary_default = signature

        if any(k.arg == "encoding" for k in node.keywords):
            continue
        if len(node.args) > encoding_at:
            continue  # a positional encoding, in the position it belongs

        mode = next(
            (k.value for k in node.keywords if k.arg == "mode"),
            node.args[mode_at] if mode_at is not None and len(node.args) > mode_at
            else None,
        )
        if mode is None:
            if binary_default:
                continue  # tempfile defaults to "w+b"; nothing to encode
        elif _is_binary_mode(mode):
            continue

        offenders.append(f"{where}:{node.lineno + line_offset} {name}")
    return offenders


def _encoding_offenders(root: pathlib.Path) -> list[str]:
    """Text reads and writes under `root` that do not name an encoding.

    Covers module code and doctest bodies alike. Doctests are included because
    `just test` runs them, so a bare text handle in one is a live violation --
    and because the two that existed were found by reading rather than by this
    check, which is the same hand-fix-without-a-guard that produced the check in
    the first place. `ast` sees a docstring as a string, so their examples are
    parsed separately and reported against the line they sit on.
    """
    import doctest

    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        name = str(path.relative_to(root))
        offenders.extend(_offenders_in_tree(tree, name))

        for holder in ast.walk(tree):
            if not isinstance(
                holder, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                continue
            docstring = ast.get_docstring(holder, clean=False)
            if not docstring or ">>>" not in docstring:
                continue
            # Line of the docstring's opening quote, so offsets land in the file.
            base = holder.body[0].lineno if holder.body else 1
            for example in doctest.DocTestParser().get_examples(docstring):
                try:
                    example_tree = ast.parse(example.source)
                except SyntaxError:
                    continue  # a fragment, or output mistaken for source
                offenders.extend(
                    _offenders_in_tree(example_tree, name, base + example.lineno - 1)
                )
    return offenders


def test_every_text_read_and_write_in_the_package_names_its_encoding():
    """The write side was argued for; the read side was never written down.

    Structural rather than behavioural, deliberately: the failure needs a
    non-UTF-8 interpreter to appear, so a reader added under a UTF-8 default
    would pass every other test in this file and break only on someone else's
    machine.

    Parsed rather than grepped, because a line-based check both missed the
    multi-line calls that do pass an encoding and, on its first writing,
    mis-grouped its own `or`/`and` so it never checked writes at all.
    """
    package = pathlib.Path(__file__).resolve().parent.parent / "src" / "deep_research_client"
    offenders = _encoding_offenders(package)
    assert not offenders, (
        "these read or write text without naming an encoding, so the bytes "
        f"depend on the machine's locale: {', '.join(offenders)}"
    )


def test_the_tests_hold_themselves_to_the_same_rule():
    """The suite that proves the property was full of bare reads itself.

    Under the locale the subprocess test above creates, this suite would have
    failed on its own fixtures -- and two files had already drifted apart about
    it, one passing `encoding="utf-8"` for the same `results.tsv` read the
    other did bare.
    """
    tests = pathlib.Path(__file__).resolve().parent
    offenders = _encoding_offenders(tests)
    assert not offenders, (
        "test files read or write text without naming an encoding: "
        f"{', '.join(offenders)}"
    )


def test_the_manifest_records_the_cache_state_the_client_actually_had(
    tmp_path, monkeypatch,
):
    """`config.use_cache` is not it when the caller supplies a client.

    The config field applies only when `run_matrix` builds the client, which is
    documented on the field itself -- so recording it in the manifest made the
    manifest assert the default for every library caller and every test here,
    where the fixture exists precisely to turn caching off. This file's job is
    provenance; it must not be the one field that can state the reverse of what
    happened.
    """
    client = _cache_client(tmp_path, monkeypatch, enabled=False)
    eval_set = EvalSet(name="m", tasks=[
        EvalTask(id="t1", prompt="Manifest probe?", answer_type=AnswerType.REPORT),
    ])

    # use_cache left at its default True, while the client has caching off.
    manifest = asyncio.run(run_matrix(
        eval_set, [_mock_arm("a", "none")],
        MatrixConfig(output_dir=tmp_path / "run"), client=client,
    ))
    assert manifest.cache_enabled is False
    # And no directory, because none was read. Recording one beside
    # `cache_enabled: false` would name somewhere this run never consulted --
    # the same family as the field it was added to close.
    assert manifest.cache_dir is None

    # With the cache on, the directory is provenance worth having: replays from
    # a per-project cache are a different history from the shared default.
    on = _cache_client(tmp_path, monkeypatch, enabled=True)
    used = asyncio.run(run_matrix(
        eval_set, [_mock_arm("a", "none")],
        MatrixConfig(output_dir=tmp_path / "run2"), client=on,
    ))
    assert used.cache_enabled is True
    assert used.cache_dir == str(tmp_path / "cache")
