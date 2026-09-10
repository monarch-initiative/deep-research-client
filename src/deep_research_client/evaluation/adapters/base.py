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

from ..datamodel import EvalSet, EvalTask


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
