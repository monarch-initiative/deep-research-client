"""Tests for the generic eval-set model, its adapters, and MCQ scoring.

The unit tests here never reach the network. LAB-Bench in particular is tested
against a small synthetic fixture in the upstream row shape rather than against
real questions: the real dataset ships a contamination canary, is CC-BY-SA-4.0
where this project is BSD-3-Clause, and would make `just test` depend on
HuggingFace being up. Tests that need the real thing are marked ``integration``.
"""

from pathlib import Path

import pytest

from deep_research_client.evaluation import mcq
from deep_research_client.evaluation.adapters import available_adapters, get_adapter
from deep_research_client.evaluation.adapters.lab_bench import (
    SUBSETS,
    _task_from_row,
    text_only_subsets,
)
from deep_research_client.evaluation.adapters.monarch import build_rubric
from deep_research_client.evaluation.datamodel import (
    AnswerSpec,
    AnswerType,
    EvalTask,
    ExpectedTopic,
    Rubric,
    ScoreDisposition,
    SpotCheck,
)
from deep_research_client.evaluation.models import DROutput, MCQAnswer
from deep_research_client.evaluation.scorers import (
    score_factual_spot_checks,
    score_topic_coverage,
)

EVAL_INPUT = Path(__file__).parent / "input" / "eval"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_every_registered_adapter_resolves():
    """A name in the registry must actually import and instantiate."""
    for name in available_adapters():
        adapter = get_adapter(name)
        assert adapter.name == name
        assert adapter.description


def test_unknown_adapter_names_the_alternatives():
    with pytest.raises(ValueError, match="Unknown eval-set adapter"):
        get_adapter("no-such-benchmark")


# ---------------------------------------------------------------------------
# YAML / TSV adapters
# ---------------------------------------------------------------------------


def test_yaml_adapter_reads_mixed_shapes():
    eval_set = get_adapter("yaml").load(EVAL_INPUT / "example_evalset.yaml")
    assert eval_set.name == "coscientist-demo"
    by_id = {t.id: t for t in eval_set.tasks}
    assert by_id["fgfr3_mech"].answer_type == AnswerType.REPORT
    assert by_id["fgfr3_residue"].answer_type == AnswerType.MULTIPLE_CHOICE
    assert by_id["capital_of_france"].answer_type == AnswerType.SHORT_ANSWER


def test_yaml_adapter_keeps_unknown_columns_as_metadata():
    eval_set = get_adapter("yaml").load(EVAL_INPUT / "example_evalset.yaml")
    task = next(t for t in eval_set.tasks if t.id == "capital_of_france")
    assert [(m.key, m.value) for m in task.metadata] == [("difficulty", "trivial")]


def test_tsv_adapter_reads_pipe_separated_distractors():
    eval_set = get_adapter("tsv").load(EVAL_INPUT / "example_evalset.tsv")
    task = next(t for t in eval_set.tasks if t.id == "q_mcq")
    assert task.answer_type == AnswerType.MULTIPLE_CHOICE
    assert task.answer_spec.distractors == ["Guanine", "Cytosine"]
    assert task.tags == ["biology", "basics"]


@pytest.mark.parametrize("adapter_name,filename", [
    ("yaml", "example_evalset.yaml"),
    ("tsv", "example_evalset.tsv"),
])
def test_adapters_reject_missing_files(adapter_name, filename):
    with pytest.raises(FileNotFoundError):
        get_adapter(adapter_name).load(EVAL_INPUT / f"absent-{filename}")


def test_duplicate_task_ids_are_rejected(tmp_path):
    """Task ids become directory names, so a collision must not be silent."""
    path = tmp_path / "dupes.yaml"
    path.write_text(
        "tasks:\n"
        "  - id: same\n    prompt: first\n"
        "  - id: same\n    prompt: second\n"
    )
    with pytest.raises(ValueError, match="duplicate task ids: same"):
        get_adapter("yaml").load(path)


def test_declared_mcq_without_ideal_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("tasks:\n  - id: x\n    prompt: p\n    answer_type: MULTIPLE_CHOICE\n")
    with pytest.raises(ValueError, match="no 'ideal' answer"):
        get_adapter("yaml").load(path)


# ---------------------------------------------------------------------------
# LAB-Bench adapter
# ---------------------------------------------------------------------------


def test_lab_bench_row_maps_onto_the_generic_model():
    row = {
        "id": "abc-123",
        "question": "Which antibiotic?",
        "ideal": "ampicillin",
        "distractors": ["gentamicin", "meropenem"],
        "subtask": "litqa2-v1-public",
        "canary": "lab-bench:5c41:DO-NOT-PROPAGATE",
    }
    task = _task_from_row(row, "LitQA2", "Insufficient information.")
    assert task.id == "LitQA2__abc-123"
    assert task.source_id == "abc-123"
    assert task.answer_type == AnswerType.MULTIPLE_CHOICE
    assert task.answer_spec.ideal == "ampicillin"
    assert task.task_type == "litqa2-v1-public"


def test_lab_bench_canary_never_reaches_the_eval_set():
    """The canary is a contamination marker; it must not be stored or prompted."""
    row = {
        "id": "abc", "question": "Q", "ideal": "A", "distractors": ["B"],
        "canary": "lab-bench:5c41:DO-NOT-PROPAGATE",
    }
    task = _task_from_row(row, "LitQA2", None)
    assert "DO-NOT-PROPAGATE" not in task.model_dump_json()


def test_multimodal_subsets_are_refused_rather_than_silently_scored():
    """FigQA/TableQA questions are about images this client cannot present."""
    for subset in ("FigQA", "TableQA"):
        assert subset not in text_only_subsets()
        with pytest.raises(ValueError, match="figures or tables"):
            get_adapter("lab-bench").load(subset)


def test_unknown_lab_bench_subset_is_rejected():
    from deep_research_client.evaluation.adapters.lab_bench import fetch_subset

    with pytest.raises(ValueError, match="Unknown LAB-Bench subset"):
        fetch_subset("NotASubset")


@pytest.mark.integration
def test_lab_bench_litqa2_downloads_at_the_expected_size():
    """Guards the pin: a row-count drift means upstream changed under us."""
    eval_set = get_adapter("lab-bench").load("LitQA2")
    assert len(eval_set.tasks) == SUBSETS["LitQA2"][0]
    assert eval_set.is_partial
    assert eval_set.source_revision
    assert eval_set.license == "CC-BY-SA-4.0"


# ---------------------------------------------------------------------------
# Multiple-choice presentation, extraction and scoring
# ---------------------------------------------------------------------------


def _mcq_task(**kwargs) -> EvalTask:
    spec = AnswerSpec(
        ideal=kwargs.pop("ideal", "Thymine"),
        distractors=kwargs.pop("distractors", ["Guanine", "Cytosine"]),
        abstention_option=kwargs.pop("abstention_option", None),
    )
    return EvalTask(
        id=kwargs.pop("id", "t1"),
        prompt=kwargs.pop("prompt", "Which base pairs with adenine?"),
        answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=spec,
    )


def test_option_order_is_stable_across_runs():
    """Providers must see identical questions, and a rerun must reproduce them."""
    task = _mcq_task()
    first = [c.text for c in mcq.present_choices(task)]
    second = [c.text for c in mcq.present_choices(task)]
    assert first == second


def test_option_order_differs_between_tasks():
    """Seeding on the task id must not put the answer in the same slot every time."""
    orders = {
        tuple(c.is_ideal for c in mcq.present_choices(_mcq_task(id=f"t{i}")))
        for i in range(25)
    }
    assert len(orders) > 1


def test_abstention_option_is_presented_last():
    task = _mcq_task(abstention_option="Insufficient information.")
    choices = mcq.present_choices(task)
    assert choices[-1].is_abstention
    assert sum(c.is_abstention for c in choices) == 1


@pytest.mark.parametrize("response_template,expected_disposition", [
    ("Answer: {letter}", ScoreDisposition.SCORED),
    ("A long report.\n\nFinal answer: {letter}", ScoreDisposition.SCORED),
    ("{letter}", ScoreDisposition.SCORED),
    ("I have no idea.", ScoreDisposition.EXTRACTION_FAILED),
    ("", ScoreDisposition.EXTRACTION_FAILED),
])
def test_extraction_dispositions(response_template, expected_disposition):
    task = _mcq_task()
    choices = mcq.present_choices(task)
    ideal = next(c for c in choices if c.is_ideal)
    answer = mcq.grade("t1", "p", response_template.format(letter=ideal.letter), choices)
    assert answer.disposition == expected_disposition


def test_extraction_prefers_the_concluding_letter():
    """A report weighing options before concluding must be read at its conclusion."""
    task = _mcq_task()
    choices = mcq.present_choices(task)
    ideal = next(c for c in choices if c.is_ideal)
    wrong = next(c for c in choices if not c.is_ideal)
    response = f"Option {wrong.letter} is tempting, but wrong.\n\nAnswer: {ideal.letter}"
    assert mcq.grade("t1", "p", response, choices).correct


def test_ambiguous_text_match_is_an_extraction_failure_not_a_guess():
    """Two option texts in the response means the report discussed, not chose."""
    task = _mcq_task()
    choices = mcq.present_choices(task)
    response = "It could be Thymine or Guanine; both are plausible."
    assert mcq.grade("t1", "p", response, choices).disposition == (
        ScoreDisposition.EXTRACTION_FAILED
    )


def test_abstention_is_not_counted_as_a_wrong_answer():
    task = _mcq_task(abstention_option="Insufficient information.")
    choices = mcq.present_choices(task)
    answer = mcq.grade("t1", "p", f"Answer: {choices[-1].letter}", choices)
    assert answer.disposition == ScoreDisposition.ABSTAINED
    assert not answer.correct


def test_provider_error_is_recorded_separately():
    task = _mcq_task()
    choices = mcq.present_choices(task)
    answer = mcq.grade("t1", "p", None, choices, error="quota exhausted")
    assert answer.disposition == ScoreDisposition.PROVIDER_ERROR
    assert answer.error == "quota exhausted"


def test_score_mcq_follows_lab_bench_metric_definitions():
    """accuracy = correct/total, coverage = attempted/total, precision = correct/attempted."""
    answers = [
        MCQAnswer(task_id="1", provider="p", disposition=ScoreDisposition.SCORED, correct=True),
        MCQAnswer(task_id="2", provider="p", disposition=ScoreDisposition.SCORED, correct=True),
        MCQAnswer(task_id="3", provider="p", disposition=ScoreDisposition.SCORED, correct=False),
        MCQAnswer(task_id="4", provider="p", disposition=ScoreDisposition.ABSTAINED),
        MCQAnswer(task_id="5", provider="p", disposition=ScoreDisposition.PROVIDER_ERROR),
    ]
    score = mcq.score_mcq(answers)
    assert (score.total, score.attempted, score.correct) == (5, 3, 2)
    assert score.accuracy == pytest.approx(2 / 5)
    assert score.coverage == pytest.approx(3 / 5)
    assert score.precision == pytest.approx(2 / 3)
    assert score.abstained == 1
    assert score.provider_errors == 1


def test_empty_score_does_not_divide_by_zero():
    score = mcq.score_mcq([])
    assert (score.accuracy, score.coverage, score.precision) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Rubric-driven intrinsic scorers
# ---------------------------------------------------------------------------


def _report_task(rubric: Rubric) -> EvalTask:
    return EvalTask(id="t", prompt="p", answer_type=AnswerType.REPORT, rubric=rubric)


def test_spot_check_distinguishes_wrong_value_from_absent_fact():
    """Presence and accuracy are different failures and must not be conflated."""
    rubric = Rubric(spot_checks=[
        SpotCheck(name="length", pattern=r"(\d+)\s*amino acid", expected="1863"),
    ])
    task = _report_task(rubric)

    right = score_factual_spot_checks(
        DROutput(task_id="t", provider="m", raw_markdown="a 1863 amino acid protein"), task)
    assert (right.present_count, right.correct_count) == (1, 1)

    wrong = score_factual_spot_checks(
        DROutput(task_id="t", provider="m", raw_markdown="a 999 amino acid protein"), task)
    assert (wrong.present_count, wrong.correct_count) == (1, 0)

    absent = score_factual_spot_checks(
        DROutput(task_id="t", provider="m", raw_markdown="no numbers here"), task)
    assert (absent.present_count, absent.correct_count) == (0, 0)


def test_presence_only_spot_check_needs_no_expected_value():
    rubric = Rubric(spot_checks=[SpotCheck(name="ring", pattern=r"RING domain")])
    task = _report_task(rubric)
    score = score_factual_spot_checks(
        DROutput(task_id="t", provider="m", raw_markdown="It has a RING domain."), task)
    assert score.correct_count == 1


def test_topic_coverage_counts_any_matching_keyword():
    rubric = Rubric(expected_topics=[
        ExpectedTopic(name="repair", keywords=["homologous recombination", "DNA repair"]),
        ExpectedTopic(name="epidemiology", keywords=["prevalence", "incidence"]),
    ])
    score = score_topic_coverage(
        DROutput(task_id="t", provider="m", raw_markdown="BRCA1 acts in DNA repair."),
        _report_task(rubric),
    )
    assert (score.covered_count, score.total_topics) == (1, 2)
    assert score.coverage_rate == pytest.approx(0.5)


def test_scorers_tolerate_a_task_with_no_rubric():
    """A benchmark that ships no rubric must score empty, not crash."""
    task = EvalTask(id="t", prompt="p", answer_type=AnswerType.REPORT)
    out = DROutput(task_id="t", provider="m", raw_markdown="text")
    assert score_factual_spot_checks(out, task).total_checks == 0
    assert score_topic_coverage(out, task).total_topics == 0


def test_bundled_rubrics_load_and_merge_subject_checks():
    """Bio knowledge lives in rubric data now, not in the scorer."""
    generic = build_rubric("gene_function", [])
    specific = build_rubric("gene_function", [], subject="BRCA1")
    assert generic.expected_topics
    assert len(specific.spot_checks) > len(generic.spot_checks)
    assert "protein_length" in {c.name for c in specific.spot_checks}
    assert "protein_length" not in {c.name for c in generic.spot_checks}


# ---------------------------------------------------------------------------
# Generated data model
# ---------------------------------------------------------------------------


def test_datamodel_matches_linkml_schema() -> None:
    """datamodel.py is generated; regenerate it with `just gen-datamodel-eval`.

    Guards against the checked-in Pydantic model drifting from the LinkML
    schema that is its source of truth.
    """
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    schema = Path("src/deep_research_client/evaluation/evaluation.yaml")
    generated = repo_root / "src/deep_research_client/evaluation/datamodel.py"

    # Resolve the generator next to the running interpreter so the comparison
    # uses the linkml version this environment pins, not whatever is on PATH.
    gen_pydantic = shutil.which("gen-pydantic", path=str(Path(sys.executable).parent))
    if not gen_pydantic:
        pytest.skip("linkml is not installed; install the dev dependency group to check drift")

    completed = subprocess.run(
        [gen_pydantic, str(schema)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"gen-pydantic failed:\n{completed.stderr}"

    assert completed.stdout == generated.read_text(encoding="utf-8"), (
        "datamodel.py does not match evaluation.yaml. Either the schema changed "
        "or linkml was upgraded; run `just gen-datamodel-eval` and review the diff."
    )
