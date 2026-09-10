"""Provisional scoring for multiple-choice tasks.

STATUS: a stopgap, not the intended design. Reading a provider's answer out of
its prose with regular expressions is brittle, and demonstrably so - three
separate defects showed up within a day of first use, each of which produced
plausible-looking numbers rather than an obvious failure:

- a restated option list read as the provider choosing the last option, which is
  the abstention whenever one is offered;
- a bare quantity such as "6%" appearing anywhere in a report read as choosing
  that option, inventing an answer never given;
- a verdict written "**Answer: D**" read as no answer at all, which would have
  discarded every correct answer from any provider that bolds its conclusion.

Each was found by running the thing, not by reading it, and each would have been
invisible in the resulting table. That is the argument against this approach
rather than a list of fixed bugs: the next such defect is equally likely to look
like a score.

The intended replacement is an LLM judge, as the report scorers already use
(``score_fact``, ``score_claim_recall`` and ``score_race`` in ``scorers.py`` all
call one). Deciding which option a report settled on is a reading-comprehension
task, and asking a model - ideally with structured output, so the answer comes
back as a field rather than as prose to be re-parsed - is both more robust and
more honest about its own uncertainty than a regular expression can be.

Until then this module stays off the default path: ``eval run`` materialises
results without grading unless asked. What is here is well tested against the
failures found so far, and is fine for a quick look; it should not be the basis
of a published number.

Three things happen here.

*Presentation* turns a task's ideal answer and distractors into lettered
options. The order is shuffled, but deterministically from the task id, so that
every provider sees the same question in the same order and a rerun reproduces
it. This part is sound and an LLM judge would keep it unchanged.

*Extraction* recovers which option a provider chose. This is the brittle part
described above. Failures are counted separately as ``EXTRACTION_FAILED`` and
never folded into wrong answers, because they are a defect of this harness
rather than of the provider.

*Scoring* follows LAB-Bench (Laurent et al. 2024): accuracy over all questions,
coverage as the fraction attempted, and precision over attempted questions
alone. This part is arithmetic over dispositions and is independent of how the
dispositions were obtained, so it survives the replacement of the extractor.
"""

import random
import re
import string
from dataclasses import dataclass

from .datamodel import EvalTask, ScoreDisposition
from .models import MCQAnswer, MCQScore

#: Letters assigned to options, in order.
_LETTERS = string.ascii_uppercase

#: Patterns tried, in order, to recover a letter from a provider's response.
#: Ordered most explicit first: a report that states "Answer: C" means it, while
#: a bare "C" somewhere in prose is far weaker evidence.
_LETTER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s*\(?([A-Z])\)?[.\s)]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\(?([A-Z])\)?\s*$", re.MULTILINE),
)

#: A line that opens with a letter marker, capturing the letter and the rest of
#: the line: "C. Thymine". Matched separately from the patterns above because
#: the remainder has to be checked - a bare "^[A-Z][.)] " matches an author
#: initial in a reference list ("B. Jones et al., 2019") and a species
#: abbreviation in prose ("C. elegans was not studied"), both of which are
#: common in the tail of a research report and neither of which is an answer.
_LABELLED_LINE = re.compile(r"^\s*\(?([A-Z])\)?[.):]\s+(.*?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Choice:
    """One lettered option as presented to a provider."""

    letter: str
    text: str
    is_ideal: bool
    is_abstention: bool


def present_choices(task: EvalTask, seed: str | None = None) -> list[Choice]:
    """Build the lettered options for a multiple-choice task.

    The correct answer and distractors are shuffled with a seed derived from the
    task id, so the ordering is stable across providers and across runs. Any
    abstention option is appended last rather than shuffled in, matching the
    convention of presenting "none of the above"-style options at the end.

    Args:
        task: A MULTIPLE_CHOICE task carrying an ``answer_spec``.
        seed: Override for the shuffle seed; defaults to the task id.

    Returns:
        The options, in presentation order.

    Raises:
        ValueError: If the task has no answer spec.

    >>> from .datamodel import AnswerSpec, AnswerType
    >>> task = EvalTask(id="t1", prompt="Which?", answer_type=AnswerType.MULTIPLE_CHOICE,
    ...                 answer_spec=AnswerSpec(ideal="right", distractors=["wrong1", "wrong2"],
    ...                                        abstention_option="Insufficient information."))
    >>> choices = present_choices(task)
    >>> [c.letter for c in choices]
    ['A', 'B', 'C', 'D']
    >>> choices[-1].is_abstention
    True
    >>> sum(c.is_ideal for c in choices)
    1
    >>> [c.letter for c in present_choices(task)] == [c.letter for c in choices]
    True
    """
    if task.answer_spec is None:
        raise ValueError(f"Task {task.id!r} has no answer_spec to present")

    spec = task.answer_spec
    options = [(spec.ideal, True)] + [(d, False) for d in (spec.distractors or [])]
    random.Random(seed or task.id).shuffle(options)

    if spec.abstention_option:
        options.append((spec.abstention_option, False))

    if len(options) > len(_LETTERS):
        raise ValueError(
            f"Task {task.id!r} has {len(options)} options, more than the "
            f"{len(_LETTERS)} letters available"
        )

    return [
        Choice(
            letter=_LETTERS[i],
            text=text,
            is_ideal=is_ideal,
            is_abstention=bool(spec.abstention_option) and i == len(options) - 1,
        )
        for i, (text, is_ideal) in enumerate(options)
    ]


def format_prompt(task: EvalTask, choices: list[Choice]) -> str:
    """Render a multiple-choice task as a prompt.

    The instruction to end with an explicit ``Answer: <letter>`` line exists to
    make extraction tractable. It does not constrain how the provider reaches
    the answer, so a deep research tool is still free to return a full report
    above that line.

    >>> from .datamodel import AnswerSpec, AnswerType
    >>> task = EvalTask(id="t1", prompt="Which base pairs with adenine?",
    ...                 answer_type=AnswerType.MULTIPLE_CHOICE,
    ...                 answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine"]))
    >>> print(format_prompt(task, present_choices(task)))
    Which base pairs with adenine?
    <BLANKLINE>
    A. Thymine
    B. Guanine
    <BLANKLINE>
    End your response with a line of the form "Answer: X", where X is the letter
    of the single best option.
    """
    lines = [task.prompt, ""]
    lines.extend(f"{c.letter}. {c.text}" for c in choices)
    lines.extend(["", *_INSTRUCTION_LINES])
    return "\n".join(lines)


#: The instruction appended to every multiple-choice prompt. Held as a constant
#: because a provider that quotes the prompt back quotes this too, and the quoted
#: copy contains the very "Answer: X" shape the extractor looks for.
_INSTRUCTION_LINES = (
    'End your response with a line of the form "Answer: X", where X is the letter',
    "of the single best option.",
)

#: Leading list markers to ignore when deciding whether a line restates an option.
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]\s*)?(?:\(?([A-Za-z])\)?[.):]\s*)?")


def _strip_echoed_prompt(response: str, choices: list["Choice"]) -> str:
    """Remove a restatement of the question's own options from a response.

    Deep research tools routinely open by quoting the question and listing every
    option before answering. Left in place that block is indistinguishable from
    an answer: each option line looks exactly like a provider stating its choice,
    and taking the last such line silently returns whichever option was listed
    last - the abstention, whenever one is offered. Coverage collapses and every
    arm looks identically indecisive.

    Only a *run* of consecutive lines restating two or more distinct options in
    presentation order is removed. A provider naming one option, even as
    "C. Thymine", is choosing, not quoting, and survives.

    >>> from .datamodel import AnswerSpec, AnswerType
    >>> task = EvalTask(id="t1", prompt="Which?", answer_type=AnswerType.MULTIPLE_CHOICE,
    ...                 answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine"],
    ...                                        abstention_option="Not enough information."))
    >>> choices = present_choices(task)
    >>> echoed = "You asked:\\nA. Thymine\\nB. Guanine\\nC. Not enough information.\\n\\nAnswer: A"
    >>> "Guanine" in _strip_echoed_prompt(echoed, choices)
    False
    >>> _strip_echoed_prompt("I pick B. Guanine here.", choices)
    'I pick B. Guanine here.'
    """
    by_text = {_normalize(c.text): c.letter for c in choices if c.text.strip()}
    if len(by_text) < 2:
        return response

    lines = response.split("\n")
    matched: list[str | None] = []
    for line in lines:
        stripped = _LIST_MARKER.sub("", line).strip()
        matched.append(by_text.get(_normalize(stripped)))

    drop: set[int] = set()
    start = 0
    while start < len(lines):
        if matched[start] is None:
            start += 1
            continue
        end = start
        seen = []
        while end < len(lines) and (matched[end] is not None or not lines[end].strip()):
            if matched[end] is not None:
                seen.append(matched[end])
            end += 1
        # Two or more distinct options in a row is a restatement of the list,
        # not a choice among them.
        if len(set(seen)) >= 2:
            drop.update(range(start, end))
        start = max(end, start + 1)

    kept = [
        line for i, line in enumerate(lines)
        if i not in drop and line.strip() not in _INSTRUCTION_LINES
    ]
    return "\n".join(kept)


#: Markdown emphasis and heading marks, stripped before looking for an answer.
#: Real reports write their verdict as "**Answer: D**" or "## Answer: D", and a
#: pattern anchored to the end of the line will not see past the trailing marks.
_EMPHASIS = re.compile(r"[*`#]+")


def _strip_emphasis(text: str) -> str:
    """Remove markdown emphasis and heading marks from a line.

    >>> _strip_emphasis("**Answer: D**")
    'Answer: D'
    >>> _strip_emphasis("## Final answer: B")
    ' Final answer: B'
    """
    return _EMPHASIS.sub("", text)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace and punctuation for loose matching.

    >>> _normalize("  The  RING-domain. ")
    'the ring domain'
    """
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


#: Shortest normalized option text that may be matched on its own. Benchmark
#: options are often bare quantities - "6%", "17", "2.7" - and a report of any
#: length mentions numbers constantly, so matching those as answers invents
#: choices the provider never made. That is worse than recovering nothing: an
#: extraction failure is visible in its own column, while a fabricated answer
#: silently enters the accuracy.
_MIN_DISCRIMINATING_LENGTH = 4


def _is_discriminating(text: str) -> bool:
    """Whether an option's text is distinctive enough to identify a choice by.

    >>> _is_discriminating("Thymine")
    True
    >>> _is_discriminating("2.7 fold")
    True
    >>> _is_discriminating("6%")
    False
    >>> _is_discriminating("17")
    False
    >>> _is_discriminating("")
    False
    """
    normalized = _normalize(text)
    return (
        len(normalized) >= _MIN_DISCRIMINATING_LENGTH
        and any(char.isalpha() for char in normalized)
    )


def extract_choice(response: str, choices: list[Choice]) -> Choice | None:
    """Recover the option a provider chose from its response text.

    Tries explicit letter markers first, then falls back to matching the full
    text of an option. Returns ``None`` when nothing can be recovered, which the
    caller records as ``EXTRACTION_FAILED`` rather than as a wrong answer.

    Args:
        response: The provider's full response.
        choices: The options as presented.

    Returns:
        The chosen option, or ``None``.

    >>> from .datamodel import AnswerSpec, AnswerType
    >>> task = EvalTask(id="t1", prompt="Which?", answer_type=AnswerType.MULTIPLE_CHOICE,
    ...                 answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine"]))
    >>> choices = present_choices(task)
    >>> extract_choice("A long report...\\n\\nAnswer: A", choices).text
    'Thymine'
    >>> extract_choice("I conclude the answer is Thymine.", choices).text
    'Thymine'
    >>> extract_choice("No idea whatsoever.", choices) is None
    True
    """
    if not response.strip():
        return None

    by_letter = {c.letter: c for c in choices}
    tail = _strip_echoed_prompt(response, choices).strip()
    if not tail:
        return None
    # Emphasis has to go before the line patterns run: they anchor to the end of
    # a line, which "**Answer: D**" never reaches.
    tail = _strip_emphasis(tail)

    for pattern in _LETTER_PATTERNS:
        matches = pattern.findall(tail)
        if not matches:
            continue
        # Last match wins: a report that discusses options before concluding
        # states its conclusion at the end.
        letter = matches[-1].upper()
        if letter in by_letter:
            return by_letter[letter]

    # A labelled line counts only when what follows the letter is that option's
    # own text. "C. Arabidopsis thaliana" is a choice; "C. elegans was not
    # studied" and "B. Jones et al., 2019" are not.
    for letter, remainder in reversed(_LABELLED_LINE.findall(tail)):
        choice = by_letter.get(letter.upper())
        if choice is not None and _normalize(remainder) == _normalize(choice.text):
            return choice

    normalized_response = _normalize(tail)
    text_matches = [
        c for c in choices
        if _is_discriminating(c.text) and _normalize(c.text) in normalized_response
    ]
    # Only trust a text match when exactly one option's text appears; two
    # matches means the report discussed the options rather than choosing one.
    if len(text_matches) == 1:
        return text_matches[0]

    return None


def grade(
    task_id: str,
    provider: str,
    response: str | None,
    choices: list[Choice],
    error: str | None = None,
) -> MCQAnswer:
    """Grade one provider response against the options it was shown.

    Args:
        task_id: The task's id.
        provider: Provider (or arm) that produced the response.
        response: The provider's response, or ``None`` if it failed.
        choices: Options as presented.
        error: Provider error message, when the call failed.

    Returns:
        The graded answer, with its disposition set.

    >>> from .datamodel import AnswerSpec, AnswerType
    >>> task = EvalTask(id="t1", prompt="Which?", answer_type=AnswerType.MULTIPLE_CHOICE,
    ...                 answer_spec=AnswerSpec(ideal="Thymine", distractors=["Guanine"]))
    >>> choices = present_choices(task)
    >>> grade("t1", "mock", "Answer: A", choices).correct
    True
    >>> grade("t1", "mock", "Answer: B", choices).correct
    False
    >>> grade("t1", "mock", None, choices, error="quota").disposition.value
    'PROVIDER_ERROR'
    >>> grade("t1", "mock", "waffle", choices).disposition.value
    'EXTRACTION_FAILED'
    """
    if error is not None or response is None:
        return MCQAnswer(
            task_id=task_id,
            provider=provider,
            disposition=ScoreDisposition.PROVIDER_ERROR,
            error=error or "no response",
        )

    chosen = extract_choice(response, choices)
    if chosen is None:
        return MCQAnswer(
            task_id=task_id,
            provider=provider,
            disposition=ScoreDisposition.EXTRACTION_FAILED,
        )

    if chosen.is_abstention:
        return MCQAnswer(
            task_id=task_id,
            provider=provider,
            chosen_letter=chosen.letter,
            chosen_text=chosen.text,
            disposition=ScoreDisposition.ABSTAINED,
        )

    return MCQAnswer(
        task_id=task_id,
        provider=provider,
        chosen_letter=chosen.letter,
        chosen_text=chosen.text,
        disposition=ScoreDisposition.SCORED,
        correct=chosen.is_ideal,
    )


def score_mcq(answers: list[MCQAnswer]) -> MCQScore:
    """Aggregate graded answers into LAB-Bench's three headline metrics.

    Following Laurent et al. (2024): accuracy is correct over all questions,
    coverage is the fraction attempted, and precision is correct over attempted.
    A question is "attempted" when an option was actually chosen, so abstentions,
    provider errors and extraction failures all reduce coverage.

    Args:
        answers: Graded answers, one per task.

    Returns:
        The aggregate score.

    >>> from .models import MCQAnswer
    >>> answers = [
    ...     MCQAnswer(task_id="1", provider="p", disposition="SCORED", correct=True),
    ...     MCQAnswer(task_id="2", provider="p", disposition="SCORED", correct=False),
    ...     MCQAnswer(task_id="3", provider="p", disposition="ABSTAINED"),
    ...     MCQAnswer(task_id="4", provider="p", disposition="EXTRACTION_FAILED"),
    ... ]
    >>> s = score_mcq(answers)
    >>> s.total, s.attempted, s.correct
    (4, 2, 1)
    >>> s.accuracy, s.coverage, s.precision
    (0.25, 0.5, 0.5)
    >>> score_mcq([]).accuracy
    0.0
    """
    total = len(answers)
    attempted = sum(1 for a in answers if a.disposition == ScoreDisposition.SCORED)
    correct = sum(1 for a in answers if a.correct)

    return MCQScore(
        total=total,
        attempted=attempted,
        correct=correct,
        abstained=sum(1 for a in answers if a.disposition == ScoreDisposition.ABSTAINED),
        extraction_failures=sum(
            1 for a in answers if a.disposition == ScoreDisposition.EXTRACTION_FAILED
        ),
        provider_errors=sum(
            1 for a in answers if a.disposition == ScoreDisposition.PROVIDER_ERROR
        ),
        accuracy=correct / total if total else 0.0,
        coverage=attempted / total if total else 0.0,
        precision=correct / attempted if attempted else 0.0,
        answers=answers,
    )
