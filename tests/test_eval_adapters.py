"""Tests for the generic eval-set model, its adapters, and MCQ scoring.

The unit tests here never reach the network. LAB-Bench in particular is tested
against a small synthetic fixture in the upstream row shape rather than against
real questions: the real dataset ships a contamination canary, is CC-BY-SA-4.0
where this project is BSD-3-Clause, and would make `just test` depend on
HuggingFace being up. Tests that need the real thing are marked ``integration``.
"""

import re
from pathlib import Path

import httpx
import pytest

from deep_research_client.evaluation import mcq
from deep_research_client.evaluation.adapters import (
    available_adapters,
    get_adapter,
    lab_bench,
)
from deep_research_client.evaluation.adapters.lab_bench import (
    SUBSETS,
    _task_from_row,
    text_only_subsets,
)
from deep_research_client.evaluation.adapters.monarch import build_rubric
from deep_research_client.evaluation.adapters.tabular import (
    _task_from_row as _tabular_task_from_row,
)
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
    , encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task ids: same"):
        get_adapter("yaml").load(path)


def test_declared_mcq_without_ideal_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("tasks:\n  - id: x\n    prompt: p\n    answer_type: MULTIPLE_CHOICE\n", encoding="utf-8")
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


@pytest.mark.parametrize("source,match", [
    ("NotASubset", "Unknown LAB-Bench subset"),
    # A real name alongside a bad one must still name the bad one -- and must
    # not download the real one first while finding that out.
    ("LitQA2,NotASubset", "NotASubset"),
    # One input short of the check above: an empty list passed every guard,
    # made the revision request anyway, and returned an EvalSet named
    # "lab-bench-" with zero tasks and a partial_reason describing a benchmark
    # it had not loaded.
    ("", "No LAB-Bench subset named"),
    (",", "No LAB-Bench subset named"),
    (" , ", "No LAB-Bench subset named"),
])
def test_a_bad_subset_argument_is_refused_before_the_network(monkeypatch, source, match):
    """The absence of the network is the assertion, not a property of the runner.

    Asserting only the message does not discriminate: `fetch_subset` refuses an
    unknown name as its own first statement, so with this guard reverted the
    same ValueError still comes out -- just after `_resolve_or_fall_back` has
    made a revision request, and for "LitQA2,NotASubset" after LitQA2 has been
    downloaded in full. Verified by reverting the guard: the message assertion
    stayed green. A tripwire on the revision lookup is what actually holds the
    property, on a networked machine and an isolated one alike.
    """
    def tripwire(*args, **kwargs):
        raise AssertionError("network reached before the subset name was checked")

    monkeypatch.setattr(lab_bench, "resolve_revision", tripwire)
    monkeypatch.setattr(lab_bench, "_resolve_or_fall_back", tripwire)

    with pytest.raises(ValueError, match=match):
        get_adapter("lab-bench").load(source)


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


# ---------------------------------------------------------------------------
# Echoed-prompt handling
#
# Deep research tools routinely restate the question and every option before
# answering. That restatement looks exactly like a series of answers, so it has
# to be removed before extraction - otherwise the last option listed wins, which
# is the abstention whenever one is offered.
# ---------------------------------------------------------------------------


def _abstaining_task() -> EvalTask:
    return EvalTask(
        id="echo",
        prompt="Which base pairs with adenine?",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(
            ideal="Thymine",
            distractors=["Guanine", "Cytosine", "Uracil"],
            abstention_option="Insufficient information to answer this question.",
        ),
    )


def test_echoed_option_list_is_not_read_as_an_abstention():
    """The regression: a restated option list scored as choosing the last option."""
    task = _abstaining_task()
    choices = mcq.present_choices(task)
    ideal = next(c for c in choices if c.is_ideal)

    echoed = "The user asked:\n\n" + "\n".join(f"{c.letter}. {c.text}" for c in choices)
    response = f"{echoed}\n\nAfter reviewing the literature.\n\nAnswer: {ideal.letter}"

    answer = mcq.grade("echo", "p", response, choices)
    assert answer.disposition == ScoreDisposition.SCORED
    assert answer.correct


def test_echoed_option_list_without_an_answer_is_an_extraction_failure():
    """Quoting the options is not answering, and must not count as abstaining."""
    task = _abstaining_task()
    choices = mcq.present_choices(task)
    echoed = "You asked:\n\n" + "\n".join(f"{c.letter}. {c.text}" for c in choices)

    answer = mcq.grade("echo", "p", echoed + "\n\nThis is a report with no verdict.", choices)
    assert answer.disposition == ScoreDisposition.EXTRACTION_FAILED


def test_naming_a_single_option_still_counts_as_choosing_it():
    """Only a run of two or more restated options is a quotation, not a choice."""
    task = _abstaining_task()
    choices = mcq.present_choices(task)
    ideal = next(c for c in choices if c.is_ideal)

    answer = mcq.grade("echo", "p", f"{ideal.letter}. {ideal.text}", choices)
    assert answer.disposition == ScoreDisposition.SCORED
    assert answer.correct


def test_echoed_instruction_line_is_ignored():
    """The prompt's own 'Answer: X' instruction must not be read as an answer."""
    task = _abstaining_task()
    choices = mcq.present_choices(task)
    prompt_echo = mcq.format_prompt(task, choices)

    answer = mcq.grade("echo", "p", prompt_echo + "\n\nI cannot determine this.", choices)
    assert answer.disposition == ScoreDisposition.EXTRACTION_FAILED


def test_short_numeric_options_are_not_matched_by_text_alone():
    """A bare quantity appearing in a report is not evidence of a choice.

    LAB-Bench options are often quantities - "6%", "17", "2.7 fold" - and a
    report of any length mentions numbers constantly. Matching those as answers
    invents choices the provider never made, which is worse than recovering
    nothing: an extraction failure is visible in its own column, while a
    fabricated answer silently enters the accuracy.
    """
    task = EvalTask(
        id="numeric",
        prompt="By what percentage?",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="6%", distractors=["12%", "20%"]),
    )
    choices = mcq.present_choices(task)
    report = "The assay showed roughly 6% of cells responding, though this is preliminary."

    assert mcq.grade("numeric", "p", report, choices).disposition == (
        ScoreDisposition.EXTRACTION_FAILED
    )


def test_distinctive_option_text_is_still_matched():
    """The guard must not break recovery from a report that names its choice."""
    task = EvalTask(
        id="named",
        prompt="Which antibiotic?",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="ciprofloxacin", distractors=["ampicillin", "meropenem"]),
    )
    choices = mcq.present_choices(task)
    answer = mcq.grade("named", "p", "We conclude the resistance is to ciprofloxacin.", choices)
    assert answer.disposition == ScoreDisposition.SCORED
    assert answer.correct


@pytest.mark.parametrize("verdict", [
    "**Answer: D**",
    "## Answer: D",
    "**Final answer: D**",
    "`Answer: D`",
    "Answer: **D**",
])
def test_an_emphasised_verdict_is_still_read(verdict):
    """Real reports bold or head their verdict.

    Found on live Edison output, which ends its analysis with "**Answer: D**"
    and then appends a References section. A pattern anchored to the end of the
    line never reaches past the trailing marks, so the verdict was missed and a
    correct answer was recorded as an extraction failure - which reads as a low
    score rather than as a bug.
    """
    task = EvalTask(
        id="emph", prompt="Which antibiotic?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(
            ideal="ciprofloxacin",
            distractors=["ampicillin", "gentamicin", "meropenem"],
        ),
    )
    choices = mcq.present_choices(task)
    target = choices[3]  # whichever option is at D
    response = f"A long analysis.\n\n{verdict.replace('D', target.letter)}\n\nReferences\n\n1. Darby et al."

    answer = mcq.grade("emph", "edison", response, choices)
    assert answer.disposition == ScoreDisposition.SCORED
    assert answer.chosen_letter == target.letter


def test_a_verdict_followed_by_references_is_still_found():
    """The answer is often not the last thing in the document."""
    task = EvalTask(
        id="refs", prompt="Which?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="right", distractors=["wrong-a", "wrong-b"]),
    )
    choices = mcq.present_choices(task)
    ideal = next(c for c in choices if c.is_ideal)
    response = (
        f"## Interpretation\n\nSome reasoning.\n\n**Answer: {ideal.letter}**\n\n"
        "References\n\n1. Author, A. (2024). A paper. doi:10.1/2\n"
        "2. Author, B. (2023). Another paper. doi:10.3/4\n"
    )
    assert mcq.grade("refs", "p", response, choices).correct


@pytest.mark.parametrize("tail,description", [
    ("References\n\nB. Jones et al., 2019. A paper.", "author initial in a reference list"),
    ("C. elegans was not part of this study.", "species abbreviation in prose"),
    ("1. Paper one\n\nA. Author, 2020, Journal.", "numbered reference list"),
])
def test_a_letter_starting_a_line_is_not_an_answer(tail, description):
    """Author initials and species abbreviations look exactly like choices.

    "^[A-Z][.)] " matches "B. Jones et al." and "C. elegans", and reference
    lists sit at the end of a report where the last match wins. A provider that
    omitted its verdict was being scored on a citation.
    """
    task = EvalTask(
        id="refs", prompt="Which organism?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(
            ideal="Saccharomyces cerevisiae",
            distractors=["Drosophila melanogaster", "Arabidopsis thaliana"],
        ),
    )
    choices = mcq.present_choices(task)
    answer = mcq.grade("refs", "p", f"The analysis is inconclusive.\n\n{tail}\n", choices)
    assert answer.disposition == ScoreDisposition.EXTRACTION_FAILED, description


def test_a_labelled_line_repeating_the_option_text_is_an_answer():
    """The counterpart: "A. Saccharomyces cerevisiae" is a choice, not a citation."""
    task = EvalTask(
        id="labelled", prompt="Which organism?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(
            ideal="Saccharomyces cerevisiae",
            distractors=["Drosophila melanogaster", "Arabidopsis thaliana"],
        ),
    )
    choices = mcq.present_choices(task)
    ideal = next(c for c in choices if c.is_ideal)
    answer = mcq.grade(
        "labelled", "p", f"After review:\n\n{ideal.letter}. {ideal.text}\n", choices)
    assert answer.disposition == ScoreDisposition.SCORED
    assert answer.correct


def test_lab_bench_refuses_a_revision_it_cannot_serve(monkeypatch):
    """A requested revision must not be used to relabel current data.

    The datasets-server rows endpoint always serves the current revision, so
    filing a download under a requested sha would stamp current data with an old
    one — the false provenance the pin exists to prevent.
    """
    from deep_research_client.evaluation.adapters import lab_bench

    monkeypatch.setattr(lab_bench, "resolve_revision", lambda client=None: "currentsha")
    with pytest.raises(ValueError, match="but oldsha was requested"):
        lab_bench.fetch_subset("LitQA2", revision="oldsha")


def test_lab_bench_uses_a_cached_revision_when_it_cannot_reach_the_api(monkeypatch, tmp_path):
    """An outage must not turn a complete local copy into a failed run."""
    import json as json_mod

    from deep_research_client.evaluation.adapters import lab_bench

    cached = tmp_path / "eval_datasets" / "lab-bench" / "cachedsha"
    cached.mkdir(parents=True)
    rows = [
        {"id": str(i), "question": "Q?", "ideal": "A", "distractors": ["B"]}
        for i in range(lab_bench.SUBSETS["LitQA2"][0])
    ]
    (cached / "LitQA2.json").write_text(json_mod.dumps(rows), encoding="utf-8")

    def unreachable(client=None):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(lab_bench, "resolve_revision", unreachable)
    got, revision = lab_bench.fetch_subset("LitQA2", cache_dir=tmp_path)
    assert revision == "cachedsha"
    assert len(got) == len(rows)


def test_lab_bench_reports_a_truncated_cache(monkeypatch, tmp_path):
    """A short cached file must be caught, not silently shrink the benchmark."""
    import json as json_mod

    from deep_research_client.evaluation.adapters import lab_bench

    cached = tmp_path / "eval_datasets" / "lab-bench" / "somesha"
    cached.mkdir(parents=True)
    (cached / "LitQA2.json").write_text(json_mod.dumps([{"id": "1"}]), encoding="utf-8")

    monkeypatch.setattr(lab_bench, "resolve_revision", lambda client=None: "somesha")
    with pytest.raises(ValueError, match="cached copy of"):
        lab_bench.fetch_subset("LitQA2", cache_dir=tmp_path)


def test_check_unique_ids_rejects_duplicates():
    """The shared guard itself."""
    from deep_research_client.evaluation.adapters.base import check_unique_ids

    task = EvalTask(id="dupe", prompt="p", answer_type=AnswerType.REPORT)
    with pytest.raises(ValueError, match="duplicate task ids: dupe"):
        check_unique_ids([task, task], "somewhere")


def test_an_adapter_actually_applies_the_uniqueness_guard(tmp_path):
    """Calling the guard in a test proves nothing about adapters calling it.

    Ids become directory names, so a duplicate means one task's results
    overwrite another's. This goes through a real adapter so that an adapter
    dropping the check fails here.
    """
    path = tmp_path / "dupes.yaml"
    path.write_text(
        "tasks:\n"
        "  - id: same\n    prompt: first\n"
        "  - id: same\n    prompt: second\n"
    , encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate task ids: same"):
        get_adapter("yaml").load(path)


def test_a_report_discussing_each_option_states_no_answer():
    """Walking through the options is not choosing one.

    Requiring a labelled line to repeat the option's own text stops author
    initials being read as answers, but a report that gives each option its own
    heading restates several of them — and those headings need not be
    consecutive, so the echo strip leaves them. Taking the last would return
    whichever option was discussed last.
    """
    task = EvalTask(
        id="walkthrough", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine", "Cytosine"]),
    )
    choices = mcq.present_choices(task)
    report = "\n\n".join(
        f"{c.letter}. {c.text}\n\nA paragraph about {c.text} reaching no conclusion."
        for c in choices
    )
    assert mcq.grade("walkthrough", "p", report, choices).disposition == (
        ScoreDisposition.EXTRACTION_FAILED
    )


def test_a_walkthrough_that_ends_in_a_verdict_is_read():
    """The counterpart: an explicit verdict after the walkthrough still counts."""
    task = EvalTask(
        id="walkthrough2", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine", "Cytosine"]),
    )
    choices = mcq.present_choices(task)
    ideal = next(c for c in choices if c.is_ideal)
    report = "\n\n".join(f"{c.letter}. {c.text}\n\nSome discussion." for c in choices)
    answer = mcq.grade("walkthrough2", "p", f"{report}\n\nAnswer: {ideal.letter}", choices)
    assert answer.disposition == ScoreDisposition.SCORED
    assert answer.correct


@pytest.mark.parametrize("heading", ["**{L}**", "### {L}", "**{L})**", "`{L}`"])
def test_emphasised_option_headings_are_not_a_verdict(heading):
    """Stripping emphasis turns per-option headings into bare letters.

    "**A**", "### A" and "**A)**" all arrive at the bare-letter pattern as a
    lone letter, so a report that walks through its options under such headings
    and never commits was scored as whichever came last — the abstention, when
    one is offered. That is the module docstring's own first defect, reached by
    a different route.
    """
    task = EvalTask(
        id="headings", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(
            ideal="Thymine", distractors=["Guanine", "Cytosine"],
            abstention_option="Insufficient information to answer this question.",
        ),
    )
    choices = mcq.present_choices(task)
    report = "\n\n".join(
        heading.format(L=c.letter) + f"\n\nA paragraph about {c.text}." for c in choices
    )
    assert mcq.grade("headings", "p", report, choices).disposition == (
        ScoreDisposition.EXTRACTION_FAILED
    )


@pytest.mark.parametrize("heading", ["**{L}**", "### {L}"])
def test_emphasised_headings_followed_by_a_verdict_are_read(heading):
    """The counterpart: an explicit verdict after the walkthrough still counts."""
    task = EvalTask(
        id="headings2", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine", "Cytosine"]),
    )
    choices = mcq.present_choices(task)
    ideal = next(c for c in choices if c.is_ideal)
    report = "\n\n".join(heading.format(L=c.letter) + "\n\nDiscussion." for c in choices)
    answer = mcq.grade("headings2", "p", f"{report}\n\nAnswer: {ideal.letter}", choices)
    assert answer.disposition == ScoreDisposition.SCORED
    assert answer.correct


def test_a_single_emphasised_letter_is_still_a_verdict():
    """One marked letter is a choice; several are a table of contents."""
    task = EvalTask(
        id="one", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine", "Cytosine"]),
    )
    choices = mcq.present_choices(task)
    ideal = next(c for c in choices if c.is_ideal)
    answer = mcq.grade("one", "p", f"My conclusion:\n\n**{ideal.letter}**\n", choices)
    assert answer.disposition == ScoreDisposition.SCORED
    assert answer.correct


def _split_cache(tmp_path):
    """A cache holding two subsets under different revisions."""
    import json as json_mod

    root = tmp_path / "eval_datasets" / "lab-bench"
    (root / "rev1").mkdir(parents=True)
    (root / "rev2").mkdir(parents=True)
    (root / "rev1" / "LitQA2.json").write_text(json_mod.dumps([]), encoding="utf-8")
    (root / "rev2" / "SuppQA.json").write_text(json_mod.dumps([]), encoding="utf-8")


def test_newest_cached_revision_needs_every_subset_under_one_revision(tmp_path):
    """The helper itself."""
    from deep_research_client.evaluation.adapters.lab_bench import newest_cached_revision

    _split_cache(tmp_path)
    assert newest_cached_revision("LitQA2", tmp_path) == "rev1"
    assert newest_cached_revision("SuppQA", tmp_path) == "rev2"
    assert newest_cached_revision(["LitQA2", "SuppQA"], tmp_path) is None


def test_the_offline_fallback_itself_requires_one_revision(monkeypatch, tmp_path):
    """Testing the helper leaves the caller free to pass only the first subset.

    That was the actual finding, and asserting on the helper would stay green if
    it came back. This goes through the fallback.
    """
    from deep_research_client.evaluation.adapters import lab_bench

    _split_cache(tmp_path)

    def unreachable(client=None):
        raise httpx.ConnectError("no network")

    monkeypatch.setattr(lab_bench, "resolve_revision", unreachable)

    # One subset is fully cached, so the fallback can serve it.
    assert lab_bench._resolve_or_fall_back(None, ["LitQA2"], tmp_path) == "rev1"

    # Both together are not under any single revision, so it must not pretend.
    with pytest.raises(httpx.ConnectError):
        lab_bench._resolve_or_fall_back(None, ["LitQA2", "SuppQA"], tmp_path)


# ---------------------------------------------------------------------------
# Degenerate task shapes
# ---------------------------------------------------------------------------


def test_a_multiple_choice_task_needs_more_than_one_option():
    """One option and a right answer is not a question.

    Every arm answers it correctly, and the run reports perfect accuracy for
    having asked nothing — a number with no question behind it.
    """
    task = EvalTask(
        id="solo", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=[]),
    )
    with pytest.raises(ValueError, match="at least two"):
        mcq.present_choices(task)


def test_an_abstention_does_not_count_towards_the_two_options():
    """Otherwise "one right answer, or decline" would pass as a question."""
    task = EvalTask(
        id="solo_abstain", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(
            ideal="Thymine", distractors=[],
            abstention_option="Insufficient information to answer this question.",
        ),
    )
    with pytest.raises(ValueError, match="at least two"):
        mcq.present_choices(task)


@pytest.mark.parametrize("body,expected", [
    ("    answer_type: MULTIPLE_CHOICE\n    ideal: Thymine\n",
     "offers no usable distractors"),
    # An ideal repeated as its only distractor is both a duplicate and a
    # one-option task. The duplicate message is the one that names the cell to
    # edit, so it is the one that has to win.
    ("    ideal: Thymine\n    distractors: [Thymine]\n",
     "repeats its ideal answer"),
    ("    ideal: Thymine\n    distractors: [Thymine, Guanine]\n",
     "repeats its ideal answer"),
    ("    ideal: Thymine\n    distractors: ['  thymine ', Guanine]\n",
     "repeats its ideal answer"),
    # ...but with no duplicate to name, the generic count still has to outrank
    # the abstention checks, or a one-option task reads as a collision.
    ("    answer_type: MULTIPLE_CHOICE\n    ideal: Unknown\n"
     "    abstention_option: unknown\n",
     "offers no usable distractors"),
])
def test_a_degenerate_set_is_refused_when_it_loads(tmp_path, body, expected):
    """Caught at load, not at cell 43 of a run that has already been paid for.

    The realistic ways in are a spreadsheet conversion that forgets the
    distractors, and an edit that leaves an option duplicated.

    The two orderings this parametrize pins pull in opposite directions, which
    is why both are here: the duplicate check must outrank the generic count,
    and the generic count must outrank the abstention checks.
    """
    path = tmp_path / "degenerate.yaml"
    path.write_text(f"tasks:\n  - id: q1\n    prompt: Which base?\n{body}", encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        get_adapter("yaml").load(path)


def test_distractors_duplicating_each_other_are_allowed(tmp_path):
    """Untidy but harmless, and real benchmarks contain them.

    Both duplicates are wrong however the model answers, so no score changes.
    Two LitQA2 questions have this shape; refusing it would make a published
    benchmark unloadable over a defect that costs nothing.
    """
    path = tmp_path / "dupe_distractors.yaml"
    path.write_text(
        "tasks:\n  - id: q1\n    prompt: Which base?\n"
        "    ideal: Thymine\n    distractors: [Guanine, Guanine]\n"
    , encoding="utf-8")
    eval_set = get_adapter("yaml").load(path)
    assert len(mcq.present_choices(eval_set.tasks[0])) == 3


def test_options_differing_only_in_punctuation_are_distinct():
    """Real questions distinguish options by exactly what `_normalize` strips.

    LAB-Bench offers "CD8- / IGNF+" against "CD8-/IGNF -"; folding punctuation
    away to compare options would reject that question as degenerate.
    """
    task = EvalTask(
        id="signs", prompt="Which population?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="CD8- / IGNF+", distractors=["CD8-/IGNF -"]),
    )
    assert len(mcq.present_choices(task)) == 2


def test_present_choices_keeps_its_own_guard_as_a_backstop():
    """Tasks built in code never pass through an adapter."""
    task = EvalTask(
        id="in_code", prompt="Which?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Thymine", "Guanine"]),
    )
    with pytest.raises(ValueError, match="repeats its ideal answer"):
        mcq.present_choices(task)


def test_a_superseded_verdict_does_not_resurface():
    """A final verdict naming an option that was never offered is a failure.

    Reaching past it to an earlier letter would report a choice the report had
    already withdrawn.
    """
    task = EvalTask(
        id="superseded", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine", "Cytosine"]),
    )
    choices = mcq.present_choices(task)
    report = "At first I thought Answer: A\n\nOn reflection, Answer: F"
    assert mcq.grade("superseded", "p", report, choices).disposition == (
        ScoreDisposition.EXTRACTION_FAILED
    )


def test_a_task_with_no_ideal_answer_is_refused():
    """Accuracy 0.000 for every arm is the same defect as accuracy 1.000.

    A blank ideal renders as a lettered option with no text, flagged correct.
    No provider can choose it, so every arm is marked wrong and nothing reports
    that the question was unanswerable.
    """
    for ideal in ("", "   "):
        spec = AnswerSpec(ideal=ideal, distractors=["Guanine", "Cytosine"])
        assert "no ideal answer" in (mcq.degenerate_reason(spec) or "")


def test_a_benchmark_row_missing_its_answer_field_is_refused():
    """The realistic way in: an upstream rename of `ideal`.

    `_task_from_row` defaults it to "", and the row-count guard counts rows
    without looking inside them — so the whole subset would load, run, cost
    money and score zero.
    """
    from deep_research_client.evaluation.adapters.base import check_task_shapes
    from deep_research_client.evaluation.adapters.lab_bench import _task_from_row

    row = {"id": "abc", "question": "Which antibiotic?", "answer": "ampicillin",
           "distractors": ["gentamicin", "meropenem"]}
    task = _task_from_row(row, "LitQA2", None)
    with pytest.raises(ValueError, match="no ideal answer"):
        check_task_shapes([task], "LitQA2")


def test_a_task_with_an_empty_prompt_is_refused():
    """Worse than a bad answer: a blank question sent to every arm."""
    from deep_research_client.evaluation.adapters.base import check_task_shapes

    task = EvalTask(id="blank", prompt="   ", answer_type=AnswerType.REPORT)
    with pytest.raises(ValueError, match="empty prompt"):
        check_task_shapes([task], "somewhere")


def test_the_validator_and_the_renderer_see_the_same_options():
    """A guard that checks a different question than the one asked guards nothing."""
    spec = AnswerSpec(ideal="Thymine", distractors=["Guanine", "", "   "])
    task = EvalTask(
        id="blanks", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=spec,
    )
    presented = mcq.present_choices(task)
    assert len(presented) == 2
    assert {c.text for c in presented} == {"Thymine", "Guanine"}
    assert mcq.degenerate_reason(spec) is None


@pytest.mark.parametrize("spec_kwargs,expected", [
    (dict(ideal="Unknown", distractors=["X", "Y"], abstention_option="unknown"),
     "abstention text as the ideal"),
    (dict(ideal="A", distractors=["X", "Cannot be determined"],
          abstention_option="cannot be determined"),
     "repeats its abstention text"),
])
def test_the_abstention_is_not_exempt_from_the_duplicate_rule(spec_kwargs, expected):
    """The abstention is appended after everything the guard looked at.

    Without comparing it too, the duplicate rule is simply routed around — and
    the collision is not far-fetched: "Unknown" and "Cannot be determined" serve
    both as a harness abstention and as real answers to biology questions. With
    the ideal duplicated, naming one line scores correct and naming the other
    records an abstention, at random.
    """
    assert expected in (mcq.degenerate_reason(AnswerSpec(**spec_kwargs)) or "")


def test_an_ordinary_abstention_is_still_allowed():
    """The rule must not refuse the normal case it exists alongside."""
    spec = AnswerSpec(
        ideal="Thymine", distractors=["Guanine", "Cytosine"],
        abstention_option="Insufficient information to answer this question.",
    )
    assert mcq.degenerate_reason(spec) is None


def test_a_null_distractor_does_not_become_an_option_reading_None():
    """`str(None)` is non-blank, so it passes every guard while being no option."""
    from deep_research_client.evaluation.adapters.lab_bench import _task_from_row

    row = {"id": "z", "question": "Q?", "ideal": "A", "distractors": ["B", None]}
    task = _task_from_row(row, "LitQA2", None)
    assert task.answer_spec.distractors == ["B"]
    assert "None" not in {c.text for c in mcq.present_choices(task)}


def test_too_many_options_is_refused_at_load_not_at_render():
    """The letter budget was the last shape guard still firing mid-run.

    Checked only in `present_choices`, it validated clean and then raised from
    inside a paid run — and from `eval load`, which renders a question to count
    its options and so tracebacked in the command taught not to.
    """
    spec = AnswerSpec(ideal="right", distractors=[f"d{i}" for i in range(26)])
    assert "more than the 26 letters" in (mcq.degenerate_reason(spec) or "")


def test_the_letter_budget_counts_the_abstention_too():
    """25 distractors plus an ideal plus an abstention is 27 options."""
    spec = AnswerSpec(
        ideal="right", distractors=[f"d{i}" for i in range(25)],
        abstention_option="Insufficient information.",
    )
    assert "more than the 26 letters" in (mcq.degenerate_reason(spec) or "")

    without = AnswerSpec(ideal="right", distractors=[f"d{i}" for i in range(25)])
    assert mcq.degenerate_reason(without) is None


@pytest.mark.parametrize("row,expected", [
    ({"id": "  ", "question": "Q?"}, "task_1"),
    ({"id": "", "question": "Q?"}, "task_1"),
    ({"question": "Q?"}, "task_1"),
    ({"id": " m1 ", "question": "Q?"}, "m1"),
])
def test_a_blank_id_column_falls_back_to_the_positional_id(row, expected):
    """A whitespace id used to survive as an empty string.

    `str(row.get("id") or f"task_{n}").strip()` stripped after the fallback, not
    before it, so "  " was truthy, the fallback never fired, and the task got an
    empty id. Nothing downstream refused it: `check_unique_ids` only looks for
    collisions, `check_task_shapes` checks the prompt, and `safe_segment("")`
    quietly writes the results under `unnamed-<digest>`.
    """
    assert _tabular_task_from_row(row, 0, "|").id == expected


def test_a_blank_distractor_is_never_presented_as_an_option():
    """The distractor half of the rule, tested where a blank can actually reach.

    Both tabular adapters drop empty parts in `_split_list`, so no CLI route
    carries a blank distractor into an `AnswerSpec` -- which means a test going
    through them cannot tell whether `usable_distractors` filters or not. A spec
    built in code can, and library callers build them that way.
    """
    task = EvalTask(
        id="blank_d", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(
            ideal="Thymine", distractors=["Guanine", "", "   ", "Cytosine"],
        ),
    )
    choices = mcq.present_choices(task)
    # Four without the filter, three with it: the assertion discriminates.
    assert len(choices) == 3
    assert all(c.text.strip() for c in choices)

    option_lines = [
        line for line in mcq.format_prompt(task, choices).splitlines()
        if re.fullmatch(r"[A-Z]\.\s*", line)
    ]
    assert option_lines == []


def test_a_blank_abstention_is_treated_as_no_abstention():
    """A space in a spreadsheet cell is truthy, and would render a bare letter.

    The task would then offer a way to decline that no provider can take, while
    the prompt says otherwise — the same rule `usable_distractors` enforces, in
    the third place an option comes from.
    """
    task = EvalTask(
        id="ws", prompt="Which base?", answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(
            ideal="Thymine", distractors=["Guanine"], abstention_option="   ",
        ),
    )
    choices = mcq.present_choices(task)
    assert len(choices) == 2
    assert not any(c.is_abstention for c in choices)
    assert all(c.text.strip() for c in choices)

    # The visible symptom being prevented: a rendered line that is a bare letter.
    option_lines = [
        line for line in mcq.format_prompt(task, choices).splitlines()
        if re.fullmatch(r"[A-Z]\.\s*", line)
    ]
    assert option_lines == []


def test_a_truncated_cache_file_names_itself_and_the_remedy(tmp_path, monkeypatch):
    """Guarded at the read, so every command that reaches it reports the same.

    The remedy first landed on `eval fetch` only, which left `eval load`, `eval
    run` and `eval score` printing a bare `Expecting value: line 1 column 1`
    through `_load_eval_set_or_exit`'s `except ValueError` -- a JSONDecodeError
    being a ValueError. `eval run` is the one the user reaches with a wallet
    open.
    """
    from deep_research_client.evaluation.adapters import lab_bench

    revision = "cafebabe" * 5
    cached = tmp_path / "eval_datasets" / "lab-bench" / revision
    cached.mkdir(parents=True)
    truncated = cached / "LitQA2.json"
    truncated.write_text('[{"id": "q1", "question": "Wh', encoding="utf-8")

    monkeypatch.setattr(lab_bench, "resolve_revision", lambda client=None: revision)

    with pytest.raises(ValueError) as excinfo:
        lab_bench.fetch_subset("LitQA2", cache_dir=str(tmp_path))

    message = str(excinfo.value)
    assert "not valid JSON" in message
    assert str(truncated) in message, "the user is told a file is bad but not which"
    assert "refresh" in message


def test_the_resolve_and_download_calls_use_their_own_timeouts(monkeypatch, tmp_path):
    """The call sites are the property; the constants alone are not.

    Comparing `_RESOLVE_TIMEOUT < _DOWNLOAD_TIMEOUT` stayed green with all three
    call sites reverted to the download timeout -- the constants would still be
    correctly ordered and nothing would use the smaller one. What the fix
    changed is the timeout each client is constructed with, so that is what is
    recorded here.

    It matters because the resolve call gates the offline fallback: its timeout
    is how long a user waits before a complete local cache is used instead.

    Whether the calls themselves succeed is irrelevant and deliberately not
    asserted -- this must record the same thing on a machine with network and
    one without.
    """
    from deep_research_client.evaluation.adapters import lab_bench

    seen: list[float | None] = []
    real_client = lab_bench.httpx.Client

    def recording_client(*args, **kwargs):
        seen.append(kwargs.get("timeout"))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(lab_bench.httpx, "Client", recording_client)

    try:
        lab_bench.resolve_revision()
    except Exception:  # noqa: BLE001 - the network outcome is not the property
        pass
    assert seen, "resolve_revision built no client"
    assert seen[0] == lab_bench._RESOLVE_TIMEOUT, (
        f"the revision lookup was built with timeout {seen[0]!r}, "
        f"expected _RESOLVE_TIMEOUT ({lab_bench._RESOLVE_TIMEOUT})"
    )

    seen.clear()
    monkeypatch.setattr(lab_bench, "_resolve_or_fall_back", lambda *a, **k: "deadbeef")
    monkeypatch.setattr(lab_bench, "_fetch_rows", lambda subset, client: [])
    try:
        lab_bench.fetch_subset("LitQA2", cache_dir=str(tmp_path))
    except Exception:  # noqa: BLE001 - the row-count guard fires; not the property
        pass
    assert seen, "fetch_subset built no client"
    assert seen[0] == lab_bench._DOWNLOAD_TIMEOUT, (
        f"the row download was built with timeout {seen[0]!r}, "
        f"expected _DOWNLOAD_TIMEOUT ({lab_bench._DOWNLOAD_TIMEOUT})"
    )
