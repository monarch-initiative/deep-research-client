"""Tests for the `eval` CLI commands.

The matrix runner and the adapters are covered at library level; these cover the
commands themselves — flag plumbing, task selection, and the end-of-run messages.
The messages matter more than they look: several of them exist precisely because
an earlier version of this code stayed silent about something a user needed to
know, and a message nothing asserts on is a message that can quietly disappear.

Everything here runs through the mock provider, so no network and no spend.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.evaluation.adapters import get_adapter

runner = CliRunner()

EVAL_INPUT = Path(__file__).parent / "input" / "eval"


@pytest.fixture(autouse=True)
def enable_mock(monkeypatch):
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# eval load / adapters
# ---------------------------------------------------------------------------


def test_eval_adapters_lists_every_registered_format():
    result = runner.invoke(app, ["eval", "adapters"])
    assert result.exit_code == 0
    for name in ("yaml", "tsv", "lab-bench", "dismech", "ai-gene-review"):
        assert name in result.stdout


def test_eval_load_summarises_shapes():
    result = runner.invoke(app, ["eval", "load", str(EVAL_INPUT / "example_evalset.yaml")])
    assert result.exit_code == 0
    assert "coscientist-demo" in result.stdout
    assert "MULTIPLE_CHOICE=1" in result.stdout
    assert "REPORT=1" in result.stdout


def test_eval_load_rejects_a_degenerate_task_before_anything_is_spent(tmp_path):
    """`eval load` exists to catch this before providers are paid for."""
    path = _write(tmp_path / "bad.yaml",
                  "tasks:\n  - id: q\n    prompt: Which?\n"
                  "    answer_type: MULTIPLE_CHOICE\n    ideal: only\n")
    result = runner.invoke(app, ["eval", "load", str(path)])
    assert result.exit_code == 1
    # Reported, not raised: a malformed eval set is this command's expected
    # output, and its neighbours report bad input the same way.
    assert "no usable distractors" in result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


# ---------------------------------------------------------------------------
# eval run
# ---------------------------------------------------------------------------


def test_eval_run_requires_an_arm():
    result = runner.invoke(app, ["eval", "run", str(EVAL_INPUT / "example_evalset.yaml")])
    assert result.exit_code == 1
    assert "--arm" in result.stdout


def test_eval_run_rejects_duplicate_arm_ids():
    result = runner.invoke(app, [
        "eval", "run", str(EVAL_INPUT / "example_evalset.yaml"),
        "--arm", "x=mock", "--arm", "x=mock",
    ])
    assert result.exit_code == 1
    assert "unique" in result.stdout


def test_eval_run_dry_run_calls_no_provider(tmp_path):
    result = runner.invoke(app, [
        "eval", "run", str(EVAL_INPUT / "example_evalset.yaml"),
        "--arm", "mock", "--dry-run", "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code == 0
    assert "no providers were called" in result.stdout
    assert not (tmp_path / "run").exists()


def test_eval_run_says_results_were_not_scored(tmp_path):
    path = _write(tmp_path / "reports.yaml", "tasks:\n  - id: r1\n    prompt: Why?\n")
    result = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code == 0
    assert "materialised, not scored" in result.stdout


def test_eval_run_with_grade_and_no_mcq_tasks_says_so(tmp_path):
    """Asking to grade a set with nothing gradable must not be answered with silence."""
    path = _write(tmp_path / "reports.yaml", "tasks:\n  - id: r1\n    prompt: Why?\n")
    result = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--grade",
        "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code == 0
    assert "Nothing to grade" in result.stdout


def test_eval_run_flags_short_answer_tasks_as_unscoreable(tmp_path):
    """The plainest eval set anyone writes infers a shape nothing scores."""
    path = _write(tmp_path / "short.yaml",
                  "tasks:\n  - id: s1\n    prompt: Capital of France?\n    ideal: Paris\n")
    result = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code == 0
    assert "SHORT_ANSWER" in result.stdout
    assert "nothing in this client scores yet" in result.stdout


def test_the_unscoreable_warning_arrives_before_any_provider_is_called(tmp_path):
    """A warning about wasting money is worth nothing after the money is spent.

    It used to be computed after `run_matrix` returned, so a real run printed it
    once every provider call had been made, and `--dry-run` -- the command
    documented as the way to price a run first -- returned before it and never
    printed it at all.
    """
    path = _write(tmp_path / "short.yaml",
                  "tasks:\n  - id: s1\n    prompt: Capital of France?\n    ideal: Paris\n")
    result = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--dry-run",
        "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code == 0
    assert "nothing in this client scores yet" in result.stdout
    # Nothing ran, so there is nothing to have paid for.
    assert "no providers were called" in result.stdout
    assert not (tmp_path / "run").exists()

    # On a real run it has to precede the per-cell progress, not trail it.
    real = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--output-dir", str(tmp_path / "run2"),
    ])
    assert real.exit_code == 0
    assert real.stdout.index("nothing in this client scores yet") < real.stdout.index("[1/1]")


def test_eval_load_says_what_follows_from_the_shapes_it_reports(tmp_path):
    """`Shapes: SHORT_ANSWER=1` states a fact without stating its consequence."""
    path = _write(tmp_path / "short.yaml",
                  "tasks:\n  - id: s1\n    prompt: Capital of France?\n    ideal: Paris\n")
    result = runner.invoke(app, ["eval", "load", str(path)])
    assert result.exit_code == 0
    assert "SHORT_ANSWER=1" in result.stdout
    assert "nothing in this client scores yet" in result.stdout


def test_eval_run_grades_multiple_choice_and_warns_about_the_extractor(tmp_path):
    path = _write(tmp_path / "mcq.yaml",
                  "tasks:\n  - id: m1\n    prompt: Which base pairs with adenine?\n"
                  "    ideal: Thymine\n    distractors: [Guanine, Cytosine]\n")
    result = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--grade",
        "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code == 0
    assert "acc" in result.stdout and "cov" in result.stdout
    assert "provisional regex extractor" in result.stdout


def test_eval_run_limit_selects_a_prefix(tmp_path):
    result = runner.invoke(app, [
        "eval", "run", str(EVAL_INPUT / "example_evalset.yaml"),
        "--arm", "mock", "--limit", "1", "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code == 0
    assert "1 cells" in result.stdout
    assert len(list((tmp_path / "run").glob("*/mock/cell.json"))) == 1


def test_eval_run_rejects_an_unknown_task_id(tmp_path):
    result = runner.invoke(app, [
        "eval", "run", str(EVAL_INPUT / "example_evalset.yaml"),
        "--arm", "mock", "--task-id", "no_such_task",
        "--output-dir", str(tmp_path / "run"),
    ])
    assert result.exit_code == 1
    assert "no_such_task" in result.stdout


# ---------------------------------------------------------------------------
# eval score
# ---------------------------------------------------------------------------


def test_eval_score_names_the_shapes_it_actually_found(tmp_path):
    """The message used to claim the set was multiple choice regardless."""
    path = _write(tmp_path / "short.yaml",
                  "tasks:\n  - id: s1\n    prompt: Capital of France?\n    ideal: Paris\n")
    report = _write(tmp_path / "report.md", "Paris is the capital of France.")

    result = runner.invoke(app, ["eval", "score", str(report), "--source", str(path)])
    assert result.exit_code == 1
    assert "SHORT_ANSWER" in result.stdout
    assert "not scored by this client" in result.stdout


def test_eval_score_points_multiple_choice_sets_at_the_right_command(tmp_path):
    path = _write(tmp_path / "mcq.yaml",
                  "tasks:\n  - id: m1\n    prompt: Which?\n    ideal: Thymine\n"
                  "    distractors: [Guanine]\n")
    report = _write(tmp_path / "report.md", "Answer: A")

    result = runner.invoke(app, ["eval", "score", str(report), "--source", str(path)])
    assert result.exit_code == 1
    assert "eval run --grade" in result.stdout


@pytest.mark.parametrize("command", ["run", "score"])
def test_every_command_reports_a_malformed_eval_set_the_same_way(tmp_path, command):
    """One command printing a message while the next tracebacks is the defect.

    `eval run` is the one the user reaches with a wallet open, and it already
    reports a bad `--arm` with a message a dozen lines earlier.
    """
    path = _write(tmp_path / "bad.yaml",
                  "tasks:\n  - id: q\n    prompt: Which?\n"
                  "    answer_type: MULTIPLE_CHOICE\n    ideal: only\n")
    args = (
        ["eval", "run", str(path), "--arm", "mock", "--output-dir", str(tmp_path / "r")]
        if command == "run"
        else ["eval", "score", str(_write(tmp_path / "rep.md", "x")), "--source", str(path)]
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "Could not load the eval set" in result.stdout


def test_eval_load_reports_the_number_of_options_actually_asked(tmp_path):
    """Counting distractors omits the abstention and includes blanks never shown."""
    path = _write(tmp_path / "mcq.yaml",
                  "tasks:\n  - id: m1\n    prompt: Which base?\n    ideal: Thymine\n"
                  "    distractors: [Guanine, Cytosine]\n"
                  "    abstention_option: Insufficient information.\n")
    result = runner.invoke(app, ["eval", "load", str(path)])
    assert result.exit_code == 0
    assert "4 options" in result.stdout


def test_the_tabular_adapters_drop_blanks_before_they_reach_the_spec(tmp_path):
    """What this actually covers, which is not what it used to claim.

    It was written as a test of `usable_distractors` and asserted `3 options`
    on a TSV with `Guanine||Cytosine`. But `_split_list` drops empty parts, so
    the blank never reached the `AnswerSpec` and `usable_distractors` had
    nothing to filter -- the assertion held with the filter reverted. The
    adapter-level guarantee is real and worth pinning; it is just a different
    guarantee, and `usable_distractors` is covered at library level instead,
    where a spec can actually carry a blank.
    """
    path = _write(tmp_path / "blanks.tsv",
                  "id\tquestion\tideal\tdistractors\n"
                  "m1\tWhich base?\tThymine\tGuanine||Cytosine\n")
    result = runner.invoke(app, ["eval", "load", str(path), "--adapter", "tsv"])
    assert result.exit_code == 0
    assert "3 options" in result.stdout

    eval_set = get_adapter("tsv").load(path)
    assert eval_set.tasks[0].answer_spec.distractors == ["Guanine", "Cytosine"]


def test_eval_load_reports_too_many_options_as_a_message(tmp_path):
    """It renders a question to count options, so this must not traceback."""
    distractors = ", ".join(f"d{i}" for i in range(26))
    path = _write(tmp_path / "big.yaml",
                  f"tasks:\n  - id: big\n    prompt: Q?\n    ideal: right\n"
                  f"    distractors: [{distractors}]\n")
    result = runner.invoke(app, ["eval", "load", str(path)])
    assert result.exit_code == 1
    assert "Could not load the eval set" in result.stdout
    assert "26 letters" in result.stdout
