try:
    from deep_research_client._version import __version__, __version_tuple__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)

from .client import DeepResearchClient
from .models import ResearchResult, ProviderConfig, CacheConfig
from .processing import ResearchProcessor, TemplateProcessor, ResultFormatter
from .model_cards import (
    ModelCard,
    ProviderModelCards,
    CostLevel,
    TimeEstimate,
    ModelCapability,
    get_provider_model_cards,
    list_all_models,
    find_models_by_cost,
    find_models_by_capability,
    resolve_model_alias,
    list_all_aliases
)

__all__ = [
    "DeepResearchClient",
    "ResearchResult",
    "ProviderConfig",
    "CacheConfig",
    "ResearchProcessor",
    "TemplateProcessor",
    "ResultFormatter",
    "ModelCard",
    "ProviderModelCards",
    "CostLevel",
    "TimeEstimate",
    "ModelCapability",
    "get_provider_model_cards",
    "list_all_models",
    "find_models_by_cost",
    "find_models_by_capability",
    "resolve_model_alias",
    "list_all_aliases"
]
