"""Registry of eval-set adapters.

Adapters are resolved lazily by name, mirroring the provider registry in
``deep_research_client.client``: importing an adapter should not cost anything
until it is actually used, because some of them pull in optional dependencies or
reach the network.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import EvalSetAdapter

#: Adapter name -> (module path, class name).
ADAPTER_CLASSES: dict[str, tuple[str, str]] = {
    "yaml": ("deep_research_client.evaluation.adapters.tabular", "YamlAdapter"),
    "tsv": ("deep_research_client.evaluation.adapters.tabular", "TsvAdapter"),
    "lab-bench": ("deep_research_client.evaluation.adapters.lab_bench", "LabBenchAdapter"),
    "dismech": ("deep_research_client.evaluation.adapters.monarch", "DismechAdapter"),
    "ai-gene-review": ("deep_research_client.evaluation.adapters.monarch", "GeneReviewAdapter"),
}


def available_adapters() -> list[str]:
    """Return the names of every registered adapter, in registration order.

    >>> "lab-bench" in available_adapters()
    True
    >>> "yaml" in available_adapters()
    True
    """
    return list(ADAPTER_CLASSES)


def get_adapter(name: str) -> "EvalSetAdapter":
    """Instantiate the adapter registered under ``name``.

    Args:
        name: Registered adapter name.

    Returns:
        A ready-to-use adapter instance.

    Raises:
        ValueError: If no adapter is registered under that name.

    >>> get_adapter("lab-bench").name
    'lab-bench'
    >>> get_adapter("nope")
    Traceback (most recent call last):
        ...
    ValueError: Unknown eval-set adapter: 'nope'. Available: yaml, tsv, lab-bench, dismech, ai-gene-review
    """
    if name not in ADAPTER_CLASSES:
        raise ValueError(
            f"Unknown eval-set adapter: {name!r}. "
            f"Available: {', '.join(available_adapters())}"
        )
    module_name, class_name = ADAPTER_CLASSES[name]
    adapter_class = getattr(importlib.import_module(module_name), class_name)
    return adapter_class()  # type: ignore[no-any-return]


__all__ = ["ADAPTER_CLASSES", "available_adapters", "get_adapter"]
