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

from ..datamodel import EvalSet


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
