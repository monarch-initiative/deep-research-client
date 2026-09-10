"""Provisional scoring for multiple-choice tasks.

STATUS: a stopgap, not the intended design. Reading a provider's answer out of
its prose with regular expressions is brittle, and demonstrably so - six
separate defects surfaced within a day of first use, every one of which produced
a plausible-looking number rather than an obvious failure:

- a restated option list read as the provider choosing the last option, which is
  the abstention whenever one is offered;
- a bare quantity such as "6%" appearing anywhere in a report read as choosing
  that option, inventing an answer never given;
- a verdict written "**Answer: D**" read as no answer at all, which would have
  discarded every correct answer from any provider that bolds its conclusion;
- an author initial in a reference list ("B. Jones et al., 2019") or a species
  abbreviation in prose ("C. elegans was not studied") read as a choice;
- a report giving each option its own heading, and choosing none of them, read
  as choosing whichever it discussed last;
- the same walkthrough with emphasised headings ("**A**", "### A"), which the
  emphasis strip turns into bare letters, read as the last option again.

The first three were found by running it and the last three by reading it, which
makes the argument stronger rather than weaker: each fix was correct and each
left another way in, because deciding what a report concluded is not a pattern
matching problem. This is the case for replacing the extractor with a judge, not
a list of bugs since fixed - a seventh is as likely to look like a score.

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

from .datamodel import AnswerSpec, EvalTask, ScoreDisposition
from .models import MCQAnswer, MCQScore

#: Letters assigned to options, in order.
_LETTERS = string.ascii_uppercase

#: An explicit verdict: "Answer: D", "the final answer is B". Saying so is
#: unambiguous, so the last one in the document wins - a report may weigh the
#: options aloud before committing.
_EXPLICIT_ANSWER = re.compile(
    r"(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s*\(?([A-Z])\)?[.\s)]*$",
    re.IGNORECASE | re.MULTILINE,
)

#: A line that is nothing but a letter. Far weaker evidence than it looks:
#: emphasis is stripped before matching, so per-option headings written "**A**",
#: "### A" or "**A)**" all arrive here as bare letters. A report that walks
#: through its options under such headings and never commits would otherwise be
#: scored as whichever option came last - and with an abstention offered last,
#: that is the abstention. So this resolves only when a single distinct letter
#: is marked this way.
_BARE_LETTER_LINE = re.compile(r"^\s*\(?([A-Z])\)?\s*$", re.MULTILINE)

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


def _comparable(text: str) -> str:
    """Fold a option's text for comparing it with another option's.

    Deliberately conservative - case and whitespace only. ``_normalize`` strips
    punctuation, which is right for finding an option's text inside a report and
    wrong for telling two options apart: real benchmark questions distinguish
    options by exactly the characters it removes. LAB-Bench offers "CD8- / IGNF+"
    against "CD8-/IGNF -", and treating those as the same option would reject a
    perfectly good question.

    >>> _comparable("  Thymine  ") == _comparable("thymine")
    True
    >>> _comparable("CD8- / IGNF+") == _comparable("CD8-/IGNF -")
    False

    The asymmetry with ``_normalize`` is deliberate, not an oversight: on the
    very questions this keeps apart, the downstream text fallback still sees two
    matches and declines to resolve. Being unable to recover an answer from
    prose is the safe failure; refusing a valid question is not.
    """
    return " ".join(text.split()).casefold()


def usable_distractors(spec: AnswerSpec) -> list[str]:
    """The distractors that will actually be presented as options.

    Blank entries are dropped here rather than in one place and not the other:
    the shape that gets validated has to be the shape that gets asked, or the
    load-time guard is checking a question nobody sees. A blank option renders
    as a bare letter, cannot be chosen, and is skipped by every defence in this
    module while still occupying a slot in the prompt.

    >>> usable_distractors(AnswerSpec(ideal="A", distractors=["B", "", "  ", "C"]))
    ['B', 'C']
    """
    return [d for d in (spec.distractors or []) if d and d.strip()]


def degenerate_reason(spec: AnswerSpec) -> str | None:
    """Why this spec cannot pose an answerable question, or None if it can.

    Two shapes are refused, and only two - both because they produce a number
    rather than an error:

    - No ideal answer at all. The correct option renders blank, no provider can
      choose it, and every arm is marked wrong - accuracy 0.000 across the
      matrix, which is the same defect as accuracy 1.000 and just as quiet.
    - Fewer than two distinct options. One option and a right answer is not a
      question: every arm answers it correctly.
    - The ideal answer repeated among the distractors. Two lettered options then
      read identically with only one flagged correct, so a provider that knows
      the answer is marked wrong half the time, at random.

    Distractors that duplicate *each other* are not refused. Both are wrong
    however the model answers, so no accuracy changes, and real benchmarks
    contain them: two LitQA2 questions do, and rejecting those would make a
    published benchmark unloadable over a defect that costs nothing. It is not
    quite free - a response that names the duplicated option in prose rather
    than by letter matches two choices, so the exactly-one rule declines and the
    cell becomes an extraction failure, moving coverage without moving accuracy.
    That fails in the safe direction, which is why it is tolerated.

    >>> degenerate_reason(AnswerSpec(ideal="Thymine", distractors=["Guanine"])) is None
    True
    >>> degenerate_reason(AnswerSpec(ideal="Thymine", distractors=["Guanine", "Guanine"])) is None
    True
    >>> print(degenerate_reason(AnswerSpec(ideal="Thymine", distractors=[])))
    offers 1 distinct option(s) besides any abstention; at least two are needed for the answer to mean anything
    >>> print(degenerate_reason(AnswerSpec(ideal="Thymine", distractors=["thymine "])))
    repeats its ideal answer among the distractors, so two options read identically and only one counts as correct
    >>> print(degenerate_reason(AnswerSpec(ideal="  ", distractors=["Guanine", "Cytosine"])))
    has no ideal answer, so its correct option would render blank and every arm would be marked wrong
    """
    if not (spec.ideal or "").strip():
        return (
            "has no ideal answer, so its correct option would render blank and "
            "every arm would be marked wrong"
        )

    ideal = _comparable(spec.ideal)
    distractors = [_comparable(d) for d in usable_distractors(spec)]

    if ideal in distractors:
        return (
            "repeats its ideal answer among the distractors, so two options read "
            "identically and only one counts as correct"
        )

    distinct = len({t for t in [ideal, *distractors] if t})
    if distinct < 2:
        return (
            f"offers {distinct} distinct option(s) besides any abstention; at "
            f"least two are needed for the answer to mean anything"
        )
    return None


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
    reason = degenerate_reason(spec)
    if reason is not None:
        # One option and a right answer is not a question: every arm answers it
        # correctly and the run reports perfect accuracy for having asked
        # nothing. Reachable from an authored eval set that declares
        # multiple_choice and forgets the distractors, and from a benchmark row
        # whose distractors field is empty or renamed - neither of which
        # anything downstream would notice.
        raise ValueError(
            f"Task {task.id!r} is multiple choice but {reason}."
        )

    options = [(spec.ideal, True)] + [(d, False) for d in usable_distractors(spec)]
    random.Random(seed or task.id).shuffle(options)

    # Appended after the guard: declining is not one of the things being chosen
    # between, so "one right answer, or say you don't know" is still not a
    # question.
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

    # An explicit verdict is unambiguous, so the last one wins - and it wins
    # outright. Reaching past a final "Answer: F" on a four-option question to
    # an earlier, superseded letter would report a choice the report had already
    # withdrawn; a verdict naming an option that was never offered is a failure
    # to extract, which is what the rest of this module does with an answer it
    # cannot trust.
    explicit = _EXPLICIT_ANSWER.findall(tail)
    if explicit:
        # Deliberately returns None when the letter is not on offer: this is the
        # one place a *match* produces an extraction failure, because a report
        # that ends by naming an option that does not exist has not chosen one.
        return by_letter.get(explicit[-1].upper())

    # A bare letter is not, so it resolves only when exactly one is marked.
    bare = {
        letter.upper()
        for letter in _BARE_LETTER_LINE.findall(tail)
        if letter.upper() in by_letter
    }
    if len(bare) == 1:
        return by_letter[bare.pop()]

    # A labelled line counts only when what follows the letter is that option's
    # own text. "C. Arabidopsis thaliana" is a choice; "C. elegans was not
    # studied" and "B. Jones et al., 2019" are not.
    #
    # And only when exactly one option is labelled that way. A report that walks
    # through the options under their own headings restates several of them
    # without ever choosing, and those headings are not always consecutive, so
    # `_strip_echoed_prompt` leaves them in place. Taking the last would return
    # whichever option the report happened to discuss last - the same mistake in
    # a different guise. This mirrors the exactly-one rule used for text matches
    # below.
    labelled = {
        letter.upper()
        for letter, remainder in _LABELLED_LINE.findall(tail)
        if (choice := by_letter.get(letter.upper())) is not None
        and _normalize(remainder) == _normalize(choice.text)
    }
    if len(labelled) == 1:
        return by_letter[labelled.pop()]

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
