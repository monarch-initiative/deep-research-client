"""The adapter contract: how an upstream benchmark becomes an ``EvalSet``.

Every benchmark this framework can read is an adapter. The framework itself
knows nothing about diseases, genes, or literature questions; it knows that an
adapter turns some source into tasks, and that each task declares the shape of
answer it expects.

Adding a benchmark therefore means writing one class here, not editing the
runner, the scorers, or the schema.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from ..datamodel import AnswerType, EvalSet, EvalTask


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


def validate_tasks(tasks: list[EvalTask], source: str) -> None:
    """Run every load-time check an adapter's output is subject to.

    One call so that an adapter cannot pick up one guard and miss the next.
    """
    check_unique_ids(tasks, source)
    check_task_shapes(tasks, source)
