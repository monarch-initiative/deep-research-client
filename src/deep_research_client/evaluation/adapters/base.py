"""The adapter contract: how an upstream benchmark becomes an ``EvalSet``.

Every benchmark this framework can read is an adapter. The framework itself
knows nothing about diseases, genes, or literature questions; it knows that an
adapter turns some source into tasks, and that each task declares the shape of
answer it expects.

Adding a benchmark therefore means writing one class here, not editing the
runner, the scorers, or the schema.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from ..datamodel import AnswerType, EvalSet, EvalTask
from ..datamodel_helpers import is_prefix_match


class EvalSetAdapter(ABC):
    """Converts one upstream benchmark format into an :class:`EvalSet`.

    Subclasses declare a ``name`` (how the CLI addresses them) and implement
    :meth:`load`. Anything an adapter cannot express in the schema's named slots
    belongs in ``EvalTask.metadata`` rather than being dropped.
    """

    #: Name used to select this adapter, e.g. on the command line.
    name: ClassVar[str]

    #: One-line description, shown when listing adapters.
    description: ClassVar[str]

    #: Whether :meth:`load` reaches the network. Adapters that do are excluded
    #: from offline test runs and warn before downloading.
    requires_network: ClassVar[bool] = False

    @abstractmethod
    def load(self, source: str | Path, **options: Any) -> EvalSet:
        """Build an :class:`EvalSet` from ``source``.

        Args:
            source: Whatever identifies the data for this adapter - a path, a
                directory, or a dataset subset name.
            **options: Adapter-specific options.

        Returns:
            An EvalSet whose provenance slots are filled in as fully as the
            source allows.
        """
        raise NotImplementedError


def check_unique_ids(tasks: list[EvalTask], source: str) -> None:
    """Raise if two tasks share an id.

    Task ids become directory names in run output, so a duplicate means one
    task's results overwrite another's and, on a resumed run, the second reports
    the first's answer as its own. Every adapter is subject to this, not just the
    ones that read user-authored files: an adapter that derives ids from a
    filename or a fallback field can collide just as easily as a hand-written
    eval set can.

    Args:
        tasks: The tasks an adapter produced.
        source: What was loaded, for the error message.

    Raises:
        ValueError: If any id appears more than once.

    >>> check_unique_ids([EvalTask(id="a", prompt="p", answer_type="REPORT")], "x")
    >>> check_unique_ids([
    ...     EvalTask(id="a", prompt="p", answer_type="REPORT"),
    ...     EvalTask(id="a", prompt="q", answer_type="REPORT"),
    ... ], "x")
    Traceback (most recent call last):
        ...
    ValueError: x: duplicate task ids: a
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for task in tasks:
        if task.id in seen and task.id not in duplicates:
            duplicates.append(task.id)
        seen.add(task.id)
    if duplicates:
        raise ValueError(f"{source}: duplicate task ids: {', '.join(duplicates)}")


def check_task_shapes(tasks: list[EvalTask], source: str) -> None:
    """Raise if any task is shaped so that it cannot be answered meaningfully.

    Checked when an eval set is loaded rather than when a cell runs. The
    invariant is fully determinable from the eval set, and a matrix run costs
    real money: discovering at cell 43 that task 44 is malformed aborts the run
    and wastes every call already made. Loading is also where ``eval load`` looks,
    which is the command whose job is to catch this before anything is spent.

    ``present_choices`` keeps its own guard as a backstop, for tasks built in
    code rather than loaded through an adapter.

    Args:
        tasks: The tasks an adapter produced.
        source: What was loaded, for the error message.

    Raises:
        ValueError: If a task has an empty prompt, or a multiple-choice task
            cannot pose an answerable question - see
            ``mcq.degenerate_reason``.

    >>> from ..datamodel import AnswerSpec
    >>> ok = EvalTask(id="a", prompt="p", answer_type="MULTIPLE_CHOICE",
    ...               answer_spec=AnswerSpec(ideal="x", distractors=["y"]))
    >>> check_task_shapes([ok], "x")
    >>> bad = EvalTask(id="b", prompt="p", answer_type="MULTIPLE_CHOICE",
    ...                answer_spec=AnswerSpec(ideal="x", distractors=[]))
    >>> check_task_shapes([bad], "somewhere")
    Traceback (most recent call last):
        ...
    ValueError: somewhere: task 'b' is multiple choice but offers no usable distractors, so it presents one option besides any abstention; at least two are needed for the answer to mean anything
    """
    # Imported here rather than at module scope: mcq imports the datamodel, and
    # hoisting this would make adapters and mcq import each other.
    from ..mcq import degenerate_reason

    for task in tasks:
        if not task.prompt.strip():
            raise ValueError(
                f"{source}: task {task.id!r} has an empty prompt; it would be "
                f"sent to every arm as a blank question"
            )
        if task.answer_type != AnswerType.MULTIPLE_CHOICE:
            continue
        if task.answer_spec is None:
            raise ValueError(
                f"{source}: task {task.id!r} is multiple choice but has no answer_spec"
            )
        reason = degenerate_reason(task.answer_spec)
        if reason is not None:
            raise ValueError(
                f"{source}: task {task.id!r} is multiple choice but {reason}"
            )


def check_rubrics(tasks: list[EvalTask], source: str) -> None:
    r"""Raise if any task's rubric cannot do what it says it does.

    A rubric is the one kind of authored data that reaches a regex compiler.
    Until this check existed, a malformed pattern loaded without complaint and
    raised ``re.error`` during scoring, inside ``score_intrinsic`` -- whose
    caller catches every exception and drops all four intrinsic sub-scores, so
    one bad character in one check silently cost a report its citation
    verifiability, alignment, and topic coverage as well. Compiling at load time
    turns that into a message naming the check, before a matrix run spends
    anything.

    ``match: prefix`` is refused when nothing can be compared -- a pattern with
    no capturing group, or a check with no ``expected``. In both cases the
    comparison the author asked for cannot happen and the check quietly becomes
    presence-only, which is a coverage signal wearing an accuracy label. A
    groupless pattern with the default ``exact`` style is *not* refused: that is
    the documented way to write a presence-only check, and most of the bundled
    ones are exactly that.

    An empty ``rubric`` block is refused for the same reason: it is
    indistinguishable in effect from no rubric at all, so writing one is always
    a mistake -- an unfinished edit, or the wrong nesting level. (A task with no
    rubric at all is fine, and scores identically. The asymmetry is about
    authoring intent: writing the key says a rubric was meant.)

    All three of a rubric's parts are checked, not just the patterns. A topic
    with no keywords cannot be covered by any report and a topic with a blank
    keyword is covered by every report including an empty one; a reference
    claim with a blank description asks the judge to look for nothing. Each is
    a declared measurement that settles nothing, which is the property the
    ``prefix`` refusals are about, arrived at by a different one-character slip.

    Args:
        tasks: The tasks an adapter produced.
        source: What was loaded, for the error message.

    Raises:
        ValueError: If a spot-check pattern does not compile, a ``prefix``
            check has nothing to compare, or a rubric block is empty.

    >>> from ..datamodel import Rubric, SpotCheck
    >>> ok = EvalTask(id="a", prompt="p", answer_type="REPORT", rubric=Rubric(
    ...     spot_checks=[SpotCheck(name="present", pattern=r"\bBARD1\b")]))
    >>> check_rubrics([ok], "x")
    >>> bad = EvalTask(id="b", prompt="p", answer_type="REPORT", rubric=Rubric(
    ...     spot_checks=[SpotCheck(name="chromosome", pattern=r"chr\s+(17")]))
    >>> try:                                    # message, not just the class:
    ...     check_rubrics([bad], "somewhere")   # a chained re.error would make
    ... except ValueError as exc:               # a Traceback example match on
    ...     print(exc)                          # the class name alone
    somewhere: task 'b' spot check 'chromosome' has a pattern that is not a
    valid regular expression: missing ), unterminated subpattern at position 6

    A ``prefix`` check with nothing to compare is refused too, rather than
    degrading to a presence-only check that reports coverage as accuracy:

    >>> vague = EvalTask(id="c", prompt="p", answer_type="REPORT", rubric=Rubric(
    ...     spot_checks=[SpotCheck(name="locus", pattern=r"chromosome\s+17",
    ...                            expected="17q21.31", match="prefix")]))
    >>> try:
    ...     check_rubrics([vague], "somewhere")
    ... except ValueError as exc:
    ...     print(exc)
    somewhere: task 'c' spot check 'locus' asks for 'match: prefix' but its
    pattern has no capturing group, so no value is captured to compare against
    '17q21.31'; add a group around the part that carries the answer
    """
    for task in tasks:
        if task.rubric is not None and not any((
            task.rubric.reference_claims,
            task.rubric.spot_checks,
            task.rubric.expected_topics,
        )):
            raise ValueError(
                f"{source}: task {task.id!r} has a 'rubric' with no "
                f"reference_claims, spot_checks or expected_topics, so it "
                f"scores nothing; fill it in or remove it"
            )
        for spec in (task.rubric.spot_checks if task.rubric else None) or []:
            try:
                compiled = re.compile(spec.pattern)
            except re.error as exc:
                raise ValueError(
                    f"{source}: task {task.id!r} spot check {spec.name!r} has a "
                    f"pattern that is not a valid regular expression: {exc}"
                ) from exc
            if compiled.match("") is not None:
                raise ValueError(
                    f"{source}: task {task.id!r} spot check {spec.name!r} has a "
                    f"pattern that matches the empty string, so it reports "
                    f"itself present in a report that says nothing; anchor it "
                    f"or require at least one character"
                )
            if not is_prefix_match(spec):
                continue
            if spec.expected is None:
                raise ValueError(
                    f"{source}: task {task.id!r} spot check {spec.name!r} asks "
                    f"for 'match: prefix' but has no 'expected' value, so there "
                    f"is nothing to compare the report against"
                )
            if not compiled.groups:
                raise ValueError(
                    f"{source}: task {task.id!r} spot check {spec.name!r} asks "
                    f"for 'match: prefix' but its pattern has no capturing "
                    f"group, so no value is captured to compare against "
                    f"{spec.expected!r}; add a group around the part that "
                    f"carries the answer"
                )

        for topic in (task.rubric.expected_topics if task.rubric else None) or []:
            # The same shape the `prefix` refusals are for: a declared
            # measurement that cannot measure. No keyword is a topic no report
            # can ever cover, so every report loses a point it could not win;
            # an empty-string keyword is a substring of everything, so every
            # report covers the topic, an empty one included.
            usable = [kw for kw in topic.keywords if kw.strip()]
            if not usable:
                raise ValueError(
                    f"{source}: task {task.id!r} topic {topic.name!r} has no "
                    f"usable keywords, so no report can ever cover it"
                )
            if len(usable) != len(topic.keywords):
                raise ValueError(
                    f"{source}: task {task.id!r} topic {topic.name!r} has a "
                    f"blank keyword, which is a substring of every report, so "
                    f"the topic would count as covered by all of them"
                )

        for claim in (task.rubric.reference_claims if task.rubric else None) or []:
            # The judge is asked whether the report covers this claim, and is
            # sent the description as the claim's text.
            if not claim.description.strip():
                raise ValueError(
                    f"{source}: task {task.id!r} reference claim {claim.name!r} "
                    f"has a blank description, which is what the judge is asked "
                    f"to look for"
                )


def validate_tasks(tasks: list[EvalTask], source: str) -> None:
    """Run every load-time check an adapter's output is subject to.

    One call so that an adapter cannot pick up one guard and miss the next.
    """
    check_unique_ids(tasks, source)
    check_task_shapes(tasks, source)
    check_rubrics(tasks, source)
