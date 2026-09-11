"""Tests for the `eval` CLI commands.

The matrix runner and the adapters are covered at library level; these cover the
commands themselves — flag plumbing, task selection, and the end-of-run messages.
The messages matter more than they look: several of them exist precisely because
an earlier version of this code stayed silent about something a user needed to
know, and a message nothing asserts on is a message that can quietly disappear.

Everything here runs through the mock provider, so no network and no spend.
"""

import json
import logging
import re
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.evaluation.adapters import get_adapter
from deep_research_client.evaluation.models import MCQScore

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
    path.write_text(body, encoding="utf-8")
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


def test_a_dimension_on_a_dead_scale_is_not_printed_as_a_number(tmp_path, monkeypatch):
    """The header and the lines under it have to answer one question.

    `scored_dimensions` was changed to filter on `normalized_score`, so a
    dimension carrying a score against a non-positive scale counts as unscored
    and the aggregate says `not measured`. The per-dimension loop kept asking
    `d.score is None` -- a third accessor -- and printed a number beneath that
    header, against a `/5` the dimension does not have.
    """
    from deep_research_client.evaluation import runner as eval_runner
    from deep_research_client.evaluation.models import (
        EvalResult, RACEDimension, RACEScore,
    )

    result = EvalResult(task_id="r1", provider="mock")
    result.race_score = RACEScore(dimensions=[
        RACEDimension(dimension="comprehensiveness", score=3.0, max_score=0.0),
        RACEDimension(dimension="accuracy", score=4.0, max_score=5.0),
        # A scale that is NOT 5, so the hardcoded `/5` is observable: with it,
        # both of the scored dimensions render the same way and nothing here
        # can tell a literal from the field.
        RACEDimension(dimension="organization", score=7.0, max_score=10.0),
    ])

    async def fake_score(*a, **k):
        return result

    # `eval score` imports `score_output` inside the command body, so the
    # patch on the module is what it picks up.
    monkeypatch.setattr(eval_runner, "score_output", fake_score)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    path = _write(tmp_path / "t.yaml", "tasks:\n  - id: r1\n    prompt: What?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia.")

    out = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1"])
    assert out.exit_code == 0, out.stdout

    lines = {ln.strip().split(":")[0]: ln.strip() for ln in out.stdout.splitlines()}
    assert lines["comprehensiveness"] == "comprehensiveness: unscored", (
        "a dimension the aggregate counts as unscored must not print a number"
    )
    assert lines["accuracy"] == "accuracy: 4.0/5"
    assert lines["organization"] == "organization: 7.0/10", (
        "the denominator is the dimension's own, not a literal 5"
    )


def test_the_grade_table_discloses_records_it_could_not_use(tmp_path, monkeypatch):
    """A harness gap that moves a rate says so where the rate is printed.

    `EXTRACTION_FAILED` gets a column AND a note under the table. The unusable
    record had only a `logger.warning`, which goes to stderr while the table
    goes to stdout -- so a user who redirects the table, or reads `scores.tsv`
    later, has nothing.

    In THIS file rather than beside the matrix tests: `eval run` builds its own
    client inside `run_matrix`, so a CLI-driven test cannot be handed a
    cache-disabled one, and only this file's autouse fixture moves HOME away
    from the developer's real `~/.deep_research_cache`.

    `score_by_arm` is stubbed, so this covers the CLI branch; the arithmetic
    that produces a non-zero `unusable` is covered at library level in
    `test_eval_matrix`.
    """
    from deep_research_client.evaluation import matrix as matrix_mod

    def one_unusable(eval_set, cells):
        return {"decliner": MCQScore(
            total=1, attempted=1, correct=0, unusable=1,
        )}

    monkeypatch.setattr(matrix_mod, "score_by_arm", one_unusable)
    path = _write(tmp_path / "mcq.yaml",
                  "tasks:\n  - id: m1\n    prompt: Which base pairs with adenine?\n"
                  "    ideal: Thymine\n    distractors: [Guanine]\n")

    out = runner.invoke(app, [
        "eval", "run", str(path), "--arm", "mock", "--grade",
        "--output-dir", str(tmp_path / "run")])

    assert out.exit_code == 0, out.stdout
    assert "no correctness" in out.stdout, (
        "the table that printed the rate must say what was left out of it"
    )
    assert "unusable column" in out.stdout
    # And it names accuracy, which is the column those records actually cost:
    # they are in `total` and cannot be in `correct`, so they lower it exactly
    # as a wrong answer would. The note used to explain only the two columns
    # they did not hurt.
    assert "accuracy" in out.stdout.split("no correctness")[1][:400]


def _make_cells_unusable(run_dir: Path) -> list[str]:
    """Rewrite every stored cell into the shape that has no correctness.

    `SCORED` with `correct` absent is what a hand-edited or older-format
    `cell.json` holds, and `_completed_cell` only re-grades a cell whose
    disposition is None -- so a resume hands these straight through to the
    scorer, which is the path a user actually meets them on.
    """
    arms = []
    for path in sorted(run_dir.rglob("cell.json")):
        cell = json.loads(path.read_text(encoding="utf-8"))
        cell["disposition"] = "SCORED"
        cell.pop("correct", None)
        path.write_text(json.dumps(cell), encoding="utf-8")
        arms.append(cell["arm_id"])
    return arms


def test_the_page_quotes_the_extraction_failures_note_as_the_command_prints_it(
    tmp_path,
):
    """Built from the command, the way the `eval score` sibling guard is.

    The note gained "and against accuracy, which is over every question
    asked"; the worked transcript went on quoting the wording from before it,
    so the page showed a sentence the command could no longer produce -- in
    the transcript the surrounding prose teaches line by line. The guard that
    should have caught it asserted the four words "no recoverable answer",
    which survive any rewording of the rest.

    The sibling `unusable` note had its accuracy clause asserted in the same
    commit that added both. This one had none, and it is the one the page
    quotes.
    """
    path = _write(tmp_path / "mcq.yaml",
                  "tasks:\n  - id: m1\n    prompt: Which base pairs with adenine?\n"
                  "    ideal: Thymine\n    distractors: [Guanine, Cytosine]\n")
    # `none` never states an answer, so the extractor recovers nothing --
    # the page's `silent` arm, which is the row the prose is about.
    arms = _write(tmp_path / "arms.yaml",
                  "arms:\n  - id: silent\n    provider: mock\n"
                  "    params:\n      answer_policy: none\n")

    out = runner.invoke(app, [
        "eval", "run", str(path), "--arms", str(arms), "--grade",
        "--output-dir", str(tmp_path / "run")])
    assert out.exit_code == 0, out.stdout

    # The condition really was reproduced: an extraction failure, not an
    # abstention -- otherwise the note never prints and everything below
    # passes on an empty string.
    scores = (tmp_path / "run" / "scores.tsv").read_text(encoding="utf-8")
    header, row = [ln.split("\t") for ln in scores.strip().splitlines()]
    assert dict(zip(header, row))["extraction_failures"] == "1"

    printed = out.stdout.split("no recoverable answer")[1]
    printed = printed.split("scores.tsv.")[0] + "scores.tsv."
    assert "against accuracy" in printed, (
        f"the note no longer names accuracy, which these responses cost: {printed!r}"
    )

    page = (Path(__file__).parent.parent
            / "docs" / "how-to" / "evaluate-providers.md").read_text(encoding="utf-8")
    # Compare word sequences: the page hard-wraps the sentence at a different
    # width from the terminal, so only the wording is common to both.
    assert " ".join(printed.split()) in " ".join(page.split()), (
        "the how-to's worked transcript quotes an extraction-failures note "
        "the command does not print; it reads:\n"
        + " ".join(printed.split())
    )


def _scores_tsv_header(tmp_path: Path, score: MCQScore) -> list[str]:
    """The column names `write_scores_tsv` actually emits, by running it.

    Takes the caller's `tmp_path` rather than calling `tempfile.mkdtemp`,
    which hands you a directory and no one to remove it -- every run of the
    suite left a `scores.tsv` in the system temp. This file already carries
    an autouse fixture because of an earlier test that wrote outside
    `tmp_path`, into the developer's real response cache.
    """
    from deep_research_client.evaluation.matrix import RunLayout, write_scores_tsv

    root = tmp_path / "header-probe"
    root.mkdir(parents=True, exist_ok=True)
    write_scores_tsv(RunLayout(root=root), {"a": score})
    return (root / "scores.tsv").read_text(encoding="utf-8").splitlines()[0].split("\t")


def test_the_page_names_every_column_the_writer_emits(tmp_path):
    """DERIVED from `write_scores_tsv`, so a new column cannot leave the page.

    The how-to tells a reader that `cov` beside the em dash does not say which
    case it is, and points at `scores.tsv`'s columns as what does. That list
    was a case short until `skipped` was added.

    The first version of this guard read the writer's column names and then
    used them only as a subset self-check on a hardcoded set of five. Adding a
    sixth column left both assertions passing and the page unchecked for the
    new name -- so it caught a removal or a rename, and not the addition it
    was named for, which is the event that prompted it. A guard's list is a
    describer too.

    Derived, every emitted column must be named on the page or exempted here
    by name, so a new one fails until someone decides which.
    """

    # The writer's own tuple, read by running it rather than by parsing its
    # source: a regex over `inspect.getsource` depended on a trailing comma
    # and matched quoted words in the docstring too.
    emitted = _scores_tsv_header(
        tmp_path, MCQScore(total=1, attempted=0, correct=0, abstained=1))
    assert "skipped" in emitted, emitted

    page = (Path(__file__).parent.parent
            / "docs" / "how-to" / "evaluate-providers.md").read_text(encoding="utf-8")

    # `arm_id` identifies the row; the three rates are documented by the
    # paragraphs above, which the sibling guards pin sentence by sentence.
    exempt = {"arm_id", "accuracy", "coverage", "precision"}
    missing = [c for c in emitted if c not in exempt and f"`{c}`" not in page]
    assert not missing, (
        f"docs/how-to/evaluate-providers.md does not name {missing}, which "
        f"scores.tsv carries and the em-dash paragraph sends a reader to. "
        f"Name it on the page, or add it to `exempt` with a reason."
    )


def test_a_graded_run_warns_once_per_arm_and_not_twice(tmp_path):
    """The scores are computed once and read twice, not computed twice.

    `score_mcq` logs one warning per arm whose records carry no correctness,
    and a graded run used to derive the scores twice over the same cells --
    once in `run_matrix` to write `scores.tsv`, once in the CLI to print the
    table. So a two-arm run emitted four lines, and the two for one arm were
    byte-identical, which is exactly what naming the arm in that warning was
    added to prevent. Grading is idempotent in its NUMBERS and not in its
    OUTPUT, so "it only recomputes" was never the whole cost.

    Counted on a handler of our own rather than `caplog`: the CLI calls
    `logging.basicConfig(force=True)`, which closes every root handler,
    pytest's capture among them, so `caplog` sees nothing here no matter what
    is logged.
    """
    path = _write(tmp_path / "mcq.yaml",
                  "tasks:\n  - id: m1\n    prompt: Which base pairs with adenine?\n"
                  "    ideal: Thymine\n    distractors: [Guanine]\n")
    run_dir = tmp_path / "run"
    argv = ["eval", "run", str(path), "--arm", "alpha=mock", "--arm", "beta=mock",
            "--output-dir", str(run_dir)]

    assert runner.invoke(app, argv).exit_code == 0
    assert sorted(_make_cells_unusable(run_dir)) == ["alpha", "beta"]

    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Collect()
    mcq_logger = logging.getLogger("deep_research_client.evaluation.mcq")
    mcq_logger.addHandler(handler)
    try:
        out = runner.invoke(app, argv + ["--grade"])
    finally:
        mcq_logger.removeHandler(handler)

    assert out.exit_code == 0, out.stdout
    # The condition really was reproduced -- without this the count below
    # passes just as well when nothing is unusable and nothing is logged.
    # Read as a VALUE, not as a substring: `unusable` is one of
    # `write_scores_tsv`'s column names, so `"unusable" in scores.tsv` is
    # true of every file it writes and was a pre-check that could not fail.
    scores = (run_dir / "scores.tsv").read_text(encoding="utf-8")
    header, *rows = [ln.split("\t") for ln in scores.strip().splitlines()]
    by_arm = {r[0]: dict(zip(header, r)) for r in rows}
    assert {a: by_arm[a]["unusable"] for a in ("alpha", "beta")} == {
        "alpha": "1", "beta": "1",
    }
    assert "no correctness" in out.stdout

    warnings = [m for m in records if "no recorded correctness" in m]
    assert len(warnings) == 2, (
        f"one warning per arm, not one per arm per pass over the cells: {warnings}"
    )
    # `startswith`, not `split(":")[0]` and not a fixed-width slice: the
    # first read the arm id as everything before the FIRST colon, which stops
    # discriminating the moment an id contains one (`safe_segment` exists
    # because ids do), and the second silently compared "arm beta: " with a
    # trailing space against the shorter id.
    alpha, beta = sorted(warnings)
    assert alpha.startswith("arm alpha:"), alpha
    assert beta.startswith("arm beta:"), beta
    # And it costs what the stdout note says it costs. These are two
    # disclosures of the same records on two streams, and stderr is the one a
    # user is left with when the table is redirected -- so they must not
    # disagree about which rates moved. The note names three; the warning
    # named two until this assertion existed.
    for rate in ("attempted", "precision", "accuracy"):
        assert all(rate in m for m in warnings), (rate, warnings)


def test_an_arm_that_attempted_nothing_shows_no_precision(tmp_path, monkeypatch):
    """The eighth rate, and the one the enumeration reached but did not gate.

    Precision is correct-over-attempted, so an arm that attempted nothing has
    none. It printed `0.000` in a column beside arms that did attempt, which
    reads as "answered and got them all wrong" -- and the ways in are the
    ordinary ones: every question declined, every response unreadable by the
    provisional extractor, an endpoint down for the whole run (a failed
    multiple-choice cell is given PROVIDER_ERROR precisely so the arm appears
    rather than vanishing from the comparison), every attempted answer
    carrying no recorded correctness, or any mixture of those. This arm is
    the `cov 0.000` shape; the mixtures, which put the dash beside a coverage
    that is neither 0 nor 1, are pinned in `test_eval_adapters`.

    The reasoning offered for leaving it was that `cov 0.000` sits beside it
    and does say so. That is the same trade -- a disambiguator next to a rate
    -- that the citation lines rejected one command over, and there the
    disambiguator was in the same sentence rather than an adjacent column.
    And it does not even hold in general: coverage does not identify which
    way an arm reached an absent precision.
    """
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    path = _write(tmp_path / "mcq.yaml",
                  "tasks:\n  - id: m1\n    prompt: Which base pairs with adenine?\n"
                  "    ideal: Thymine\n    distractors: [Guanine, Cytosine]\n"
                  "    abstention_option: Insufficient information\n")
    arms = _write(tmp_path / "arms.yaml",
                  "arms:\n  - id: decliner\n    provider: mock\n"
                  "    params:\n      answer_policy: last\n")

    result = runner.invoke(app, [
        "eval", "run", str(path), "--arms", str(arms), "--grade",
        "--output-dir", str(tmp_path / "run"),
    ])

    assert result.exit_code == 0, result.stdout
    # From the score table, not the arm listing above it, which also names
    # the arm and would satisfy a bare substring search with no numbers on it.
    table = result.stdout.split("Multiple-choice scores:", 1)[-1]
    row = next((ln for ln in table.splitlines() if "decliner" in ln), None)
    assert row is not None, result.stdout
    assert "0.000   0.000" in row, f"the arm must still show a measured zero coverage: {row}"
    assert "—" in row, f"precision over no attempts must be absent, not zero: {row}"
    # Aligned under its header, not merely present. The dash branch once wrote
    # six literal spaces where its sibling formats to `:>7`, so the width lived
    # in a place that could drift from the header it has to line up with.
    header = next(ln for ln in table.splitlines() if "prec" in ln)
    assert row.index("—") == header.index("prec") + len("prec") - 1, (
        f"the dash is not right-aligned under the prec header:\n{header}\n{row}"
    )
    # The page teaches this row; build the claim from the command and assert
    # the page carries it, the way the `eval score` sibling guard does. Two
    # commits in a row have changed a rendering and updated the page by hand.
    page = (Path(__file__).parent.parent
            / "docs" / "how-to" / "evaluate-providers.md").read_text(encoding="utf-8")
    # A DISCRIMINATING slice, not `row.split()[3]` -- that is the em dash
    # alone, and the page is full of them, two in the prose right under the
    # table this guards. It passed with the worked table deleted outright.
    # The three rate columns together pin the widths and the dash at once.
    # Split on the n column by shape, not on the literal "   0/" -- that
    # depended on `correct` rendering as " 0" under `:>2`, so a fixture with
    # ten or more correct answers would silently swallow the rest of the row
    # rather than fail.
    quoted = re.split(r"\s+\d+/", row[row.index("0.000"):])[0].rstrip()
    assert quoted in page, (
        f"the how-to does not show {quoted!r} in its worked --grade table, "
        f"which is what the command prints for an arm that attempted nothing"
    )
    # The unconditional note: printed under every graded table, and the one
    # that qualifies the whole thing. The page omitted it while quoting the
    # other.
    assert "provisional regex extractor" in result.stdout
    assert "provisional regex extractor" in page
    # The conditional one belongs to the page's `silent` row, not to this
    # fixture -- this arm abstains, so it has no extraction failures and the
    # command rightly stays quiet about them.
    assert "no recoverable answer" not in result.stdout
    # Only that the page HAS the note; its wording is built from the command
    # and compared in `test_the_page_quotes_the_extraction_failures_note_as_
    # the_command_prints_it`, because this arm cannot produce one.
    assert "no recoverable answer" in page

    # And in the artifact, where a spreadsheet would average the column.
    scores = (tmp_path / "run" / "scores.tsv").read_text(encoding="utf-8")
    header, *rows = [ln.split("\t") for ln in scores.strip().splitlines()]
    cell = dict(zip(header, rows[0]))
    assert cell["attempted"] == "0"
    assert cell["precision"] == "", f"precision should be empty, got {cell['precision']!r}"


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
    # Asserted on the TSV rather than on the phrasing of a zero, which a
    # reworded message that only listed non-zero categories would fail while
    # being perfectly correct.
    assert "replayed from the response cache" in second
    header, row = (tmp_path / "run2" / "results.tsv").read_text(
        encoding="utf-8").splitlines()[:2]
    columns = header.split("\t")
    assert row.split("\t")[columns.index("cached")] == "true"
    assert row.split("\t")[columns.index("resumed")] == "false"


def test_eval_run_no_cache_forces_a_live_call(tmp_path):
    """`--no-resume` re-runs the cell; only this re-calls the provider."""
    first, second = _run_twice(tmp_path, extra=("--no-cache",))
    assert "replayed from the response cache" not in first
    assert "replayed from the response cache" not in second


def test_the_cached_column_marks_which_rows_were_replays(tmp_path):
    """The end-of-run message points at this column, so it has to be there."""
    _run_twice(tmp_path)
    header, row = (tmp_path / "run2" / "results.tsv").read_text(encoding="utf-8").splitlines()[:2]
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


def test_a_failed_cell_is_not_counted_as_measured(tmp_path):
    """`Only the measured cells describe the provider as it is now.`

    A FAILED cell has cached=None, so it fell into `measured` -- and a cell
    that raised describes nothing. On a resumed run whose remaining cells all
    failed, the operator read "N measured in this run" three lines under
    "N failed".
    """
    path = _write(tmp_path / "rep.yaml",
                  "tasks:\n  - id: r1\n    prompt: Failure accounting probe?\n")
    arms = _write(tmp_path / "arms.yaml",
                  "arms:\n"
                  "  - id: broken\n    provider: mock\n"
                  "    params:\n      error_type: transient\n"
                  "  - id: ok\n    provider: mock\n")
    out = tmp_path / "run"

    runner.invoke(app, ["eval", "run", str(path), "--arms", str(arms),
                        "--output-dir", str(out)])
    # Re-run so the summary prints at all: it is shown when anything resumed.
    second = runner.invoke(app, ["eval", "run", str(path), "--arms", str(arms),
                                 "--output-dir", str(out)])
    assert second.exit_code == 0

    summary = next(ln for ln in second.stdout.splitlines() if "measured in this run" in ln)
    assert "0 measured in this run" in summary, (
        f"a failed cell was counted as measured: {summary}"
    )
    assert "1 failed" in summary


def test_a_fresh_run_reports_how_many_cells_failed(tmp_path):
    """The accounting paragraph is gated on a resume or a replay.

    So on a fresh run -- the first run, and every `--no-resume --no-cache`
    re-run -- it never prints, and dropping the count from the failure header
    left a 50-failure run showing ten lines and no total at all.
    """
    path = _write(tmp_path / "rep.yaml",
                  "tasks:\n" + "".join(
                      f"  - id: t{i}\n    prompt: Probe {i}?\n" for i in range(12)))
    arms = _write(tmp_path / "arms.yaml",
                  "arms:\n  - id: broken\n    provider: mock\n"
                  "    params:\n      error_type: transient\n")
    result = runner.invoke(app, [
        "eval", "run", str(path), "--arms", str(arms),
        "--output-dir", str(tmp_path / "run")])
    assert result.exit_code == 0

    assert "12 failed:" in result.stdout
    # And says so when the list is truncated, which it silently was.
    assert "and 2 more" in result.stdout
    # The accounting paragraph is absent on a fresh run, which is the premise.
    assert "measured in this run" not in result.stdout


def test_eval_score_without_an_api_key_offers_a_remedy_that_works(tmp_path, monkeypatch):
    """The command's own advice used to traceback.

    `AsyncOpenAI` raises on an empty key at *construction*, and the client was
    built unconditionally -- so `--no-fact --no-recall --no-race`, the remedy
    the warning offered, died before any scoring. There was no way to get the
    intrinsic scores without a key at all.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia.")

    # Without the remedy: a message naming it, not a traceback.
    refused = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1"])
    assert refused.exit_code == 1
    assert "--no-fact --no-recall --no-race" in refused.stdout
    assert refused.exception is None or isinstance(refused.exception, SystemExit)

    # With it: the intrinsic scores the message promised.
    ok = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-fact", "--no-recall", "--no-race"])
    assert ok.exit_code == 0, ok.stdout
    assert "Factual Spot Checks" in ok.stdout


def test_an_unscored_race_dimension_prints_rather_than_raising(tmp_path, monkeypatch):
    """`score` is Optional now, and this loop is outside runner._run's except.

    Formatting None with `:.1f` raises TypeError, so a judge that failed on one
    dimension took down the whole command after every scorer had finished.

    Driven through the command rather than through the models: an earlier
    version of this test rebuilt the CLI's own rendering expression and
    asserted on that, which stayed green with `cli.py` reverted to the code
    that crashed. The only thing that can prove this is the command's output.
    """
    from deep_research_client.evaluation import scorers

    calls: list[str] = []

    async def flaky_judge(prompt, llm_client, model="gpt-4o-mini"):
        calls.append(prompt)
        # First RACE dimension answers; the rest fail, the way a judge that
        # goes away mid-report does. A judge that failed on *every* dimension
        # would not reach this branch: score_race would still return, but the
        # mixed case is the one that has to render two shapes in one list.
        if len(calls) == 1:
            return '{"score": 4, "explanation": "fine"}'
        raise RuntimeError("judge unreachable")

    monkeypatch.setattr(scorers, "_llm_judge", flaky_judge)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-by-the-stub")

    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia.")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-fact", "--no-recall", "--no-intrinsic"])

    assert result.exit_code == 0, result.stdout
    assert result.exception is None, result.exception
    assert "comprehensiveness: 4.0/5" in result.stdout
    assert "unscored" in result.stdout
    # Averaged over what was scored, not over a zero for the ones that failed.
    assert "over 1/4 dimensions" in result.stdout


def test_eval_score_says_when_nothing_compared_a_value(tmp_path, monkeypatch):
    """`accuracy 0.00` led the line for a task that measured no accuracy.

    A task with no rubric -- the commonest kind, and what `eval run` produces
    for a bare list of questions -- has no spot check that compares anything.
    Printing a rate of 0.00 over a denominator of zero reads as "every fact in
    this report was wrong"; the rate is absent, not small.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia.")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-fact", "--no-recall", "--no-race"])

    assert result.exit_code == 0, result.stdout
    assert "no accuracy (no check compared a value)" in result.stdout
    assert "accuracy 0.00" not in result.stdout


def test_eval_score_reports_a_report_the_judge_only_partly_saw(tmp_path, monkeypatch):
    """RACE cuts a long report to MAX_REPORT_CHARS and records what it cut.

    Nothing printed it, so a comprehensiveness score computed on a report's
    opening third was indistinguishable from one computed on all of it -- the
    same "a number with no statement of what it covers" the unresolvable counts
    exist to prevent.
    """
    from deep_research_client.evaluation import scorers

    async def judge(prompt, llm_client, model="gpt-4o-mini"):
        # Answers BOTH scorers: RACE reads `score`, claim recall reads
        # `matched`. With only `score`, claim recall found no verdict, counted
        # the claim unjudged, and this test's assertion that its line carries
        # the truncation note was pinning a line that said nothing was judged.
        return '{"score": 3, "explanation": "ok", "matched": true}'

    monkeypatch.setattr(scorers, "_llm_judge", judge)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-by-the-stub")

    # With a reference claim, so claim recall actually asks the judge something
    # and truncates like RACE does; without one it judges nothing, and nothing
    # judged is not a truncated judgement.
    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n"
                  "    answer_type: REPORT\n"
                  "    rubric:\n"
                  "      reference_claims:\n"
                  "        - name: fgfr3\n"
                  "          category: mechanism\n"
                  "          description: FGFR3 mutations cause achondroplasia.\n")
    long_report = "FGFR3 drives achondroplasia. " * 1000
    assert len(long_report) > scorers.MAX_REPORT_CHARS
    report = _write(tmp_path / "r.md", long_report)

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-fact", "--no-intrinsic"])

    assert result.exit_code == 0, result.stdout
    note = f"judged on {scorers.MAX_REPORT_CHARS:,} of {len(long_report):,} characters"
    # Both judge-backed scorers cut the report and both record the pair, so both
    # lines have to say so -- one of them printing it is what made the other
    # line's silence easy to miss.
    lines = {
        ln.strip().split(":")[0]: ln
        for ln in result.stdout.splitlines() if note in ln
    }
    assert set(lines) == {"RACE", "Claim Recall"}, result.stdout


def test_eval_score_reports_alignment_lookups_that_failed(tmp_path, monkeypatch):
    """Its sibling printed this and it did not.

    With PubMed unreachable, every citation-claim pair is unresolvable, so the
    line reads `0/0 (0.00)` -- identical to a report whose citations support
    nothing at all.
    """
    from deep_research_client.evaluation import scorers

    async def unreachable(pmid, client=None):
        return {"exists": False, "title": None, "year": None,
                "error": "ConnectTimeout", "lookup_failed": True}

    monkeypatch.setattr(scorers, "fetch_pubmed_metadata", unreachable)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia [PMID:7913883].")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-fact", "--no-recall", "--no-race"])

    assert result.exit_code == 0, result.stdout
    # On the alignment line specifically. Its sibling above prints the same
    # phrase, so a bare substring assertion over the whole output passed with
    # this line reverted -- the failure mode this test exists to catch.
    alignment = next(
        ln for ln in result.stdout.splitlines() if "Citation-Claim Alignment" in ln
    )
    assert "1 with nothing to align against" in alignment


def test_a_keyless_local_endpoint_is_not_refused(tmp_path, monkeypatch):
    """`--llm-base-url` exists for vLLM, Ollama and LM Studio, which take any key.

    Making the client construction lazy fixed a traceback and introduced this:
    the command began exiting 1 whenever the key env var was empty, whatever
    `--llm-base-url` said, and offered as its remedy "turn off the judge you
    just configured". The refusal belongs to the default endpoint, not to the
    judge.
    """
    from deep_research_client.evaluation import scorers

    async def judge(prompt, llm_client, model="gpt-4o-mini"):
        return '{"score": 4, "explanation": "ok"}'

    monkeypatch.setattr(scorers, "_llm_judge", judge)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia.")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--llm-base-url", "http://localhost:8000/v1",
        "--no-fact", "--no-recall", "--no-intrinsic"])

    assert result.exit_code == 0, result.stdout
    assert "RACE" in result.stdout


def test_the_keyless_refusal_names_every_way_out(tmp_path, monkeypatch):
    """Against the default endpoint the refusal stands, but it used to offer
    only one of the three remedies the command actually has."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia.")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1"])

    assert result.exit_code == 1
    assert "--llm-api-key-env" in result.stdout
    assert "--llm-base-url" in result.stdout
    assert "--no-fact --no-recall --no-race" in result.stdout


def test_the_fact_line_says_what_it_could_not_judge(tmp_path, monkeypatch):
    """The fourth score line with no unmeasured count.

    `score_fact` drops every pair whose verdict is None and reports
    `total_citations` over what is left, so a report citing only DOIs -- whose
    abstracts this scorer cannot fetch -- printed `accuracy=0.00,
    effective_citations=0/0`, which reads as a report whose citations support
    nothing. Its three siblings were fixed; this one was not touched.
    """
    from deep_research_client.evaluation import scorers

    async def judge(prompt, llm_client, model="gpt-4o-mini"):
        return '{"supported": true, "explanation": "yes"}'

    monkeypatch.setattr(scorers, "_llm_judge", judge)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-by-the-stub")

    async def abstract(pmid, client=None):
        return "FGFR3 mutations cause achondroplasia."

    monkeypatch.setattr(scorers, "fetch_pubmed_abstract", abstract)

    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    # Two PMIDs the judge rules on and two DOIs, whose abstracts this scorer
    # cannot fetch. A mix, not four DOIs: with nothing judged, `total_citations`
    # and the judged count coincide at 0, so the line reads `0/0` whichever
    # number it is printed over and the assertion below cannot tell them apart.
    report = _write(tmp_path / "r.md",
                    "FGFR3 drives achondroplasia (PMID:7913883). "
                    "It is dominant (PMID:12345678). "
                    "Growth is affected (DOI:10.1038/ng1234). "
                    "So is the skull (DOI:10.1038/ng5678).")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-recall", "--no-race", "--no-intrinsic"])

    assert result.exit_code == 0, result.stdout
    fact_line = next(ln for ln in result.stdout.splitlines() if "FACT:" in ln)
    assert "2 not judged" in fact_line
    # Over the judged pairs, which is the rate's own denominator. Printed over
    # every pair it would read 2/4 beside an accuracy of 1.00, letting a reader
    # divide and get a different number from the one next to it.
    assert "effective_citations=2/2" in fact_line


def test_a_placeholder_key_is_announced_rather_than_sent_silently(tmp_path, monkeypatch):
    """The skip is keyed on "a custom base URL", not on "needs no key".

    The commonest custom base URL after localhost is a corporate or cloud
    proxy, which does check keys. Silently substituting a placeholder there
    trades one pre-flight message naming three remedies for a 401 inside every
    judge call, after the report has been read and the intrinsic scorers have
    run. The skip stays -- the local case is why it exists -- but it says so.
    """
    from deep_research_client.evaluation import scorers

    async def judge(prompt, llm_client, model="gpt-4o-mini"):
        return '{"score": 4, "explanation": "ok"}'

    monkeypatch.setattr(scorers, "_llm_judge", judge)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia.")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--llm-base-url", "https://llm.corp.example/v1",
        "--no-fact", "--no-recall", "--no-intrinsic"])

    assert result.exit_code == 0, result.stdout
    assert "placeholder key" in result.stdout
    assert "answer 401" in result.stdout


def test_the_docs_do_not_claim_the_skip_is_narrower_than_it_is(tmp_path):
    """The runtime note landed a round before the pages describing the flag.

    Both docs presented the condition as the thing it stands for -- "an endpoint
    that needs no key" -- while the code skips for any custom base URL. A reader
    configuring a corporate proxy learned it from a 401, not from the page that
    documents the flag.
    """
    root = Path(__file__).parent.parent
    for page in ("docs/reference/cli.md", "docs/how-to/evaluate-providers.md"):
        text = (root / page).read_text(encoding="utf-8")
        assert "--llm-base-url" in text, page
        # Both facts in ONE paragraph. Asserting them over the whole page let
        # "placeholder" be satisfied by an unrelated `--template PATH | Template
        # file with variable placeholders` row, so only the "401" half
        # discriminated and rewording the real paragraph to say "Unauthorized"
        # would have left this green on a stale page.
        paragraphs = [p for p in text.split("\n\n") if "--llm-base-url" in p]
        assert any(
            "401" in p and "placeholder" in p for p in paragraphs
        ), (
            f"{page} has no --llm-base-url paragraph saying both that a "
            f"placeholder key is sent and that a key-checking endpoint "
            f"answers 401"
        )


def test_a_task_with_no_rubric_prints_no_rates_it_did_not_measure(tmp_path, monkeypatch):
    """Three lines had the same shape and only one had been fixed.

    The spot-check line learned to say "no accuracy (no check compared a
    value)" rather than `accuracy 0.00`; claim recall and topic coverage two
    lines away still printed `0.00 (0/0)` and `0/0 (0.00)`. The trigger is
    identical -- a task whose rubric is absent or partial, which is every task
    from a benchmark that ships without one, and the documented
    `--source questions.yaml` path -- so a user scoring a rubric-less task read
    three lines that look like a report which covered nothing.
    """
    from deep_research_client.evaluation import scorers

    async def judge(prompt, llm_client, model="gpt-4o-mini"):
        return '{"matched": true}'

    monkeypatch.setattr(scorers, "_llm_judge", judge)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-by-the-stub")

    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia.")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-fact", "--no-race"])

    assert result.exit_code == 0, result.stdout
    assert "Claim Recall: no reference claims to match against" in result.stdout
    assert "Topic Coverage: no expected topics to cover" in result.stdout
    assert "no accuracy (no check compared a value)" in result.stdout
    # None of the three may render as a measured zero.
    for absent in ("Claim Recall: 0.00", "Topic Coverage: 0/0", "accuracy 0.00"):
        assert absent not in result.stdout, absent


def test_a_judge_that_cannot_be_reached_is_not_a_report_that_scored_zero(
    tmp_path, monkeypatch,
):
    """The three judge-backed lines, from the commonest judge failure there is.

    Gating the citation lines on their own denominators was done by enumerating
    the *lines* that print a rate, and RACE has no citation count to be reached
    that way -- so `overall=0.00 over 0/4 dimensions` survived, which is the
    same rate-over-nothing with a disambiguating suffix the citation lines had
    just stopped printing. All four dimensions fail together whenever the judge
    itself is unreachable, so it is the commonest RACE outcome, not the rarest.

    The same run covers two more: `FACT: … none checkable` called an outage a
    property of the citations, when the PMID here was entirely checkable; and
    the truncation note was appended to a line saying nothing was judged,
    because `judged_chars` records what was *sent*.
    """
    from deep_research_client.evaluation import scorers

    async def dead_judge(prompt, llm_client, model="gpt-4o-mini"):
        raise RuntimeError("judge endpoint is down")

    async def abstract(pmid, client=None):
        return "FGFR3 gain-of-function underlies achondroplasia."

    monkeypatch.setattr(scorers, "_llm_judge", dead_judge)
    # FACT is left on deliberately -- it is one of the three lines under test
    # -- so the PMID below is fetched. Stubbed, because this file's docstring
    # promises no network, and because the premise matters: with the abstract
    # in hand the pair is checkable and the judge is what failed, which is the
    # distinction the line is being asserted on. Unstubbed it also passes, by
    # the other route (no abstract, so nothing to judge) and after a 30-second
    # timeout on a blackholed network.
    monkeypatch.setattr(scorers, "fetch_pubmed_abstract", abstract)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-by-the-stub")

    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n"
                  "    rubric:\n      reference_claims:\n"
                  "        - name: fgfr3\n          category: mechanism\n"
                  "          description: FGFR3 mutations cause achondroplasia.\n")
    # Long enough to be truncated, so the note fires if it is not gated, and
    # carrying a citation so FACT has a pair it could have judged.
    report = _write(tmp_path / "r.md",
                    "FGFR3 drives achondroplasia [PMID:7913883]. "
                    + "Filler sentence. " * 3000)

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-intrinsic"])

    assert result.exit_code == 0, result.stdout
    assert "RACE: not measured, 0/4 dimensions scored" in result.stdout
    assert "FACT: not measured, 1 not judged" in result.stdout
    assert "Claim Recall: not measured, 1 not judged" in result.stdout
    # No rate, and nothing that says a judgement was made on part of the report.
    for absent in ("overall=0.00", "none checkable", "judged on "):
        assert absent not in result.stdout, absent


def test_the_docs_quote_a_line_the_command_can_actually_print(tmp_path, monkeypatch):
    """The page quoted `Citation Verifiability: 0/0 (0.00), 12 not checked`.

    That was real output once. Gating the rate on its own denominator made it
    unreachable, and the page went on quoting it -- as the worked example for
    the one case it exists to explain, an accession-only bibliography. Build
    the sentence from the command and assert the page carries that wording, so
    the next rewording of the line takes the page with it.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md",
                    "FGFR3 drives achondroplasia (PMC11000121). "
                    "Deposited under GSE68086.")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-fact", "--no-recall", "--no-race"])
    assert result.exit_code == 0, result.stdout

    line = next(ln for ln in result.stdout.splitlines()
                if "Citation Verifiability" in ln).strip()
    # The page's example uses a different count, so compare the wording with
    # the number taken out rather than the whole sentence.
    shape = line.replace("2 not checked", "N not checked")
    assert "N not checked" in shape, line

    page = (Path(__file__).parent.parent
            / "docs" / "how-to" / "evaluate-providers.md").read_text(encoding="utf-8")
    prefix = shape.split("N not checked")[0]
    assert prefix in page, (
        f"docs/how-to/evaluate-providers.md does not quote {prefix!r}, which is "
        f"what the command now prints for an accession-only bibliography"
    )


def test_a_report_without_citations_prints_no_rates_it_did_not_measure(
    tmp_path, monkeypatch,
):
    """The same defect as its predecessor above, one trigger over.

    That test fixed the three lines a rubric-less task renders. A report that
    cites nothing drives three more from a zero denominator -- FACT, citation
    verifiability and citation-claim alignment -- and they printed `0/0` and
    `0.00` beside two lines that do say "there is nothing here", in the same
    output. A provider that returned prose without a bibliography is the
    commonest case there is, and it read as a provider whose every citation
    was fabricated.
    """
    from deep_research_client.evaluation import scorers

    async def judge(prompt, llm_client, model="gpt-4o-mini"):
        return '{"supported": true}'

    monkeypatch.setattr(scorers, "_llm_judge", judge)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used-by-the-stub")

    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md", "FGFR3 drives achondroplasia.")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-recall", "--no-race"])

    assert result.exit_code == 0, result.stdout
    assert "FACT: no citations to verify" in result.stdout
    assert "Citation Verifiability: no citations to check" in result.stdout
    assert "Citation-Claim Alignment: no citation-claim pairs" in result.stdout
    # A measured zero on any of the three is the defect, whichever line it is
    # on: assert over the output rather than per-line so a regression that
    # moves between them still fails.
    for absent in ("accuracy=0.00", "0/0 ("):
        assert absent not in result.stdout, absent
    # `Factual Spot Checks: 0/0 present` is deliberately outside that list: it
    # is a count of checks, not a rate over them, and the line says separately
    # that no accuracy was measured. Stated here because it is the only bare
    # `0/0` left in this output, and a carve-out nobody wrote down is one
    # someone else will read as an oversight.
    assert "Factual Spot Checks: 0/0 present" in result.stdout


def test_the_verifiability_line_says_not_checked_rather_than_could_not_be(
    tmp_path, monkeypatch,
):
    """An identifier this scorer has no resolver for was never looked up.

    "Could not be looked up" reads as an attempt that failed. The alignment
    line was reworded when it gained this same second cause; its sibling was
    not, in the same commit.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = _write(tmp_path / "t.yaml",
                  "tasks:\n  - id: r1\n    prompt: What mechanisms?\n")
    report = _write(tmp_path / "r.md",
                    "FGFR3 drives achondroplasia (PMC11000121). "
                    "Deposited under GSE68086.")

    result = runner.invoke(app, [
        "eval", "score", str(report), "--source", str(path), "--task-id", "r1",
        "--no-fact", "--no-recall", "--no-race"])

    assert result.exit_code == 0, result.stdout
    line = next(ln for ln in result.stdout.splitlines() if "Citation Verifiability" in ln)
    assert "2 not checked" in line
    assert "could not be looked up" not in line
