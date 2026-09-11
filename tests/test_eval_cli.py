"""Tests for the `eval` CLI commands.

The matrix runner and the adapters are covered at library level; these cover the
commands themselves — flag plumbing, task selection, and the end-of-run messages.
The messages matter more than they look: several of them exist precisely because
an earlier version of this code stayed silent about something a user needed to
know, and a message nothing asserts on is a message that can quietly disappear.

Everything here runs through the mock provider, so no network and no spend.
"""

from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.evaluation.adapters import get_adapter

runner = CliRunner()

EVAL_INPUT = Path(__file__).parent / "input" / "eval"


@pytest.fixture(autouse=True)
def enable_mock(monkeypatch, tmp_path_factory):
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")

    # `eval run` builds its own client inside `run_matrix`, so these tests
    # cannot hand it a cache-disabled one the way the matrix tests can. Without
    # this they read and write the developer's real ~/.deep_research_cache,
    # keyed on prompts these tests deliberately keep stable -- so a test could
    # pass on a response recorded before the code under test existed, and a
    # first run on a clean machine would exercise a different path from every
    # run after it. `CacheManager` resolves its default as
    # `Path.home() / ".deep_research_cache"`, so moving HOME moves the cache,
    # and catches anything else that writes to the home directory too.
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))


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
    # Named, because the errors underneath carry a row number and not a file.
    assert str(path) in result.stdout


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


# ---------------------------------------------------------------------------
# eval fetch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arg", ["", ",", " , "])
def test_eval_fetch_refuses_an_empty_subset_argument(arg):
    """It used to exit 0 having printed nothing: no download, no error, no sign.

    The same three inputs `LabBenchAdapter.load` refuses, one command over.
    """
    result = runner.invoke(app, ["eval", "fetch", arg])
    assert result.exit_code == 1
    assert "No subset named" in result.stdout


@pytest.mark.parametrize("exc,expected", [
    (ValueError("LitQA2 returned 198 rows, expected 199"), "Could not fetch the dataset"),
    (httpx.HTTPError("503 from the datasets server"), "Could not reach the dataset"),
    # fetch_subset now raises this itself, at the read, so every command that
    # reaches that read reports the same thing; the CLI only has to surface it.
    # See test_a_truncated_cache_file_names_itself_and_the_remedy for the read.
    (ValueError("The cached LAB-Bench file /tmp/x.json is not valid JSON (...). "
                "re-fetch it with refresh=True"), "refresh=True"),
])
def test_eval_fetch_reports_expected_failures_rather_than_raising(monkeypatch, exc, expected):
    """Its siblings all report these; this command tracebacked.

    Both are expected outcomes of running it: the row-count guard exists so a
    user finds out about upstream drift, and this is the command that downloads
    hundreds of megabytes, so an outage lands here more than anywhere else.
    """
    from deep_research_client.evaluation.adapters import lab_bench

    monkeypatch.setattr(lab_bench, "_resolve_or_fall_back", lambda *a, **k: "deadbeef")

    def boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(lab_bench, "fetch_subset", boom)

    result = runner.invoke(app, ["eval", "fetch", "LitQA2"])
    assert result.exit_code == 1
    assert expected in result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_eval_fetch_resolves_the_revision_once_for_many_subsets(monkeypatch):
    """Resolving per subset can split the cache across two revisions.

    `newest_cached_revision` requires one revision covering every subset asked
    for, so a revision that changes mid-fetch leaves every byte on disk and a
    later offline load failing with a connection error -- against a cache this
    very command assembled.
    """
    from deep_research_client.evaluation.adapters import lab_bench

    resolves = []

    def count_resolve(client, wanted, cache_dir=None):
        resolves.append(tuple(wanted))
        return "deadbeefcafe"

    monkeypatch.setattr(lab_bench, "_resolve_or_fall_back", count_resolve)
    monkeypatch.setattr(
        lab_bench, "fetch_subset",
        lambda name, **kw: ([{"id": 1}], kw["resolved_revision"]),
    )

    result = runner.invoke(app, ["eval", "fetch", "LitQA2,SuppQA,DbQA"])
    assert result.exit_code == 0
    assert len(resolves) == 1, f"resolved {len(resolves)} times, expected once"


def test_eval_fetch_says_a_multimodal_subset_has_no_path_to_a_run(monkeypatch):
    """Fetching one is allowed, but `eval load` refuses it, so say so."""
    from deep_research_client.evaluation.adapters import lab_bench

    monkeypatch.setattr(lab_bench, "_resolve_or_fall_back", lambda *a, **k: "deadbeef")
    monkeypatch.setattr(
        lab_bench, "fetch_subset", lambda name, **kw: ([{"id": 1}], "deadbeef"))

    result = runner.invoke(app, ["eval", "fetch", "FigQA"])
    assert result.exit_code == 0
    assert "no path from this download to a run" in result.stdout


def test_the_score_hint_names_a_path_that_exists(tmp_path):
    """`safe_segment` rewrites ids, so <task_id> was neither real nor guessable."""
    path = _write(tmp_path / "colon.yaml",
                  'tasks:\n  - id: "HP:0001156"\n    prompt: What mechanisms?\n')
    run_dir = tmp_path / "run"
    result = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--output-dir", str(run_dir),
    ])
    assert result.exit_code == 0

    hint = next(ln for ln in result.stdout.splitlines() if "eval score" in ln)
    printed = Path(hint.split("eval score")[1].split()[0])
    assert printed.exists(), f"hint names a path that does not exist: {printed}"


# ---------------------------------------------------------------------------
# eval run and the response cache
# ---------------------------------------------------------------------------


def _run_twice(tmp_path, extra=()):
    """Run the same eval set twice under one HOME, so the cache persists."""
    path = _write(tmp_path / "cache.yaml",
                  "tasks:\n  - id: t1\n    prompt: CLI cache probe question?\n")
    outs = []
    for i in (1, 2):
        result = runner.invoke(app, [
            "eval", "run", str(path), "--arm", "mock",
            "--output-dir", str(tmp_path / f"run{i}"), *extra,
        ])
        assert result.exit_code == 0, result.stdout
        outs.append(result.stdout)
    return outs


def test_eval_run_says_when_cells_were_replayed_rather_than_measured(tmp_path):
    """Silence here means a score describes calls that never happened.

    The cache re-stamps this run's timings, so a replayed cell reads as fresh
    everywhere downstream -- in cell.json, in results.tsv and in the manifest.
    """
    first, second = _run_twice(tmp_path)
    assert "replayed from the response cache" not in first
    # Distinct output dirs, so the second run is a cache replay, not a resume.
    assert "1 replayed from the response cache" in second
    assert "0 resumed from a previous run" in second


def test_eval_run_no_cache_forces_a_live_call(tmp_path):
    """`--no-resume` re-runs the cell; only this re-calls the provider."""
    first, second = _run_twice(tmp_path, extra=("--no-cache",))
    assert "replayed from the response cache" not in first
    assert "replayed from the response cache" not in second


def test_the_cached_column_marks_which_rows_were_replays(tmp_path):
    """The end-of-run message points at this column, so it has to be there."""
    _run_twice(tmp_path)
    header, row = (tmp_path / "run2" / "results.tsv").read_text().splitlines()[:2]
    columns = header.split("\t")
    assert "cached" in columns
    assert row.split("\t")[columns.index("cached")] == "true"


def test_the_score_hint_skips_an_arm_whose_cell_failed(tmp_path):
    """Naming a real arm only helps if that arm actually wrote an output.

    The first version of this test named its arm `broken=mock`, which sets the
    arm *id* and nothing else -- `parse_arm_flag` carries no provider params, and
    `MockProvider` raises only when `error_type` or `include_error` is set. So
    both cells completed, `arms[0]` had an output.md anyway, and the test passed
    with the fix reverted. Making an arm fail needs the `--arms` file.
    """
    path = _write(tmp_path / "rep.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    arms = _write(tmp_path / "arms.yaml",
                  "arms:\n"
                  "  - id: broken\n    provider: mock\n"
                  "    params:\n      error_type: transient\n"
                  "  - id: ok\n    provider: mock\n")
    run_dir = tmp_path / "run"
    result = runner.invoke(app, [
        "eval", "run", str(path), "--arms", str(arms),
        "--output-dir", str(run_dir),
    ])
    assert result.exit_code == 0

    # The premise the docstring claims: the first arm really failed.
    assert not (run_dir / "r1" / "broken" / "output.md").exists()

    hint = [ln for ln in result.stdout.splitlines() if "eval score" in ln]
    assert hint, "a report task was run, so the scoring hint must be printed"
    printed = Path(hint[0].split("eval score")[1].split()[0])
    assert printed.exists(), f"hint names a missing path: {printed}"
    assert "/ok/" in str(printed), "the hint should name the arm that succeeded"


def test_a_fully_resumed_run_says_it_measured_nothing(tmp_path):
    """The worst case was silence: "1/1 cells completed" for a run that called
    no provider at all, because resumed cells carried the earlier run's flags
    and a resumed cell stored with cached=false looked measured.
    """
    path = _write(tmp_path / "res.yaml",
                  "tasks:\n  - id: t1\n    prompt: Resume accounting probe?\n")
    out = tmp_path / "run"
    first = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--output-dir", str(out)])
    assert first.exit_code == 0

    second = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--output-dir", str(out)])
    assert second.exit_code == 0
    assert "0 measured in this run" in second.stdout
    assert "1 resumed from a previous run" in second.stdout


def test_the_remedy_offered_for_a_resumed_run_actually_reaches_the_cells(tmp_path):
    """--no-cache alone cannot: resume skips the cell before the client is used.

    The message used to offer exactly that, so following it changed nothing.
    """
    path = _write(tmp_path / "res.yaml",
                  "tasks:\n  - id: t1\n    prompt: Remedy probe?\n")
    out = tmp_path / "run"
    runner.invoke(app, ["eval", "run", str(path), "--arm", "mock", "--output-dir", str(out)])

    resumed = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--output-dir", str(out)])
    assert "--no-resume --no-cache" in resumed.stdout, (
        "a resumed run must not offer a remedy that resume itself defeats"
    )

    # And following it measures the cell again.
    forced = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock",
        "--no-resume", "--no-cache", "--output-dir", str(out)])
    assert forced.exit_code == 0
    assert "resumed from a previous run" not in forced.stdout


def test_identically_configured_arms_are_flagged_before_the_run(tmp_path):
    """Whether they are one sample or two is decided by scheduling.

    At -j 1 the second arm replays the first; at the default concurrency both
    usually reach the provider first. So it cannot be reported reliably after
    the fact -- but it is perfectly detectable before any money is spent.
    """
    path = _write(tmp_path / "v.yaml",
                  "tasks:\n  - id: t1\n    prompt: Variance probe?\n")
    result = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "a=mock", "--arm", "b=mock",
        "--dry-run", "--output-dir", str(tmp_path / "run")])
    assert result.exit_code == 0
    assert "identically configured" in result.stdout
    assert "a, b" in result.stdout
    # Printed before the dry-run return, so before any provider call.
    assert result.stdout.index("identically configured") < result.stdout.index("Dry run")


@pytest.mark.parametrize("args", [
    ("--arm", "a=mock", "--arm", "b=mock", "--no-cache"),
    ("--arm", "a=mock:m1", "--arm", "b=mock:m2"),
])
def test_the_identical_arms_note_stays_quiet_when_it_does_not_apply(tmp_path, args):
    """With the cache off, or with genuinely different arms, there is no issue."""
    path = _write(tmp_path / "v.yaml",
                  "tasks:\n  - id: t1\n    prompt: Variance probe?\n")
    result = runner.invoke(app, [
        "eval", "run", str(path), *args, "--dry-run",
        "--output-dir", str(tmp_path / "run")])
    assert result.exit_code == 0
    assert "identically configured" not in result.stdout
