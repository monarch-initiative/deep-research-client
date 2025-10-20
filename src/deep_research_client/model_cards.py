"""Model cards for research providers - descriptions, costs, and capabilities."""

from enum import Enum
from typing import Dict, Optional, List
from pydantic import BaseModel, Field, ConfigDict


class CostLevel(str, Enum):
    """Cost levels for research models."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TimeEstimate(str, Enum):
    """Estimated response time categories."""
    FAST = "fast"          # < 30 seconds
    MEDIUM = "medium"      # 30 seconds - 2 minutes
    SLOW = "slow"          # 2-10 minutes
    VERY_SLOW = "very_slow"  # > 10 minutes


class ModelCapability(str, Enum):
    """Model capabilities."""
    WEB_SEARCH = "web_search"
    ACADEMIC_SEARCH = "academic_search"
    SCIENTIFIC_LITERATURE = "scientific_literature"
    CITATION_TRACKING = "citation_tracking"
    REAL_TIME_DATA = "real_time_data"
    CODE_INTERPRETATION = "code_interpretation"
    VISUAL_ANALYSIS = "visual_analysis"
    MULTI_LANGUAGE = "multi_language"


class ModelCard(BaseModel):
    """Information card for a research model."""

    name: str = Field(description="Model identifier")
    display_name: str = Field(description="Human-readable model name")
    description: str = Field(description="Detailed description of model capabilities")
    cost_level: CostLevel = Field(description="Relative cost level")
    time_estimate: TimeEstimate = Field(description="Expected response time")
    capabilities: List[ModelCapability] = Field(
        default_factory=list,
        description="List of model capabilities"
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Short aliases for convenient CLI usage"
    )

    # Optional detailed information
    context_window: Optional[int] = Field(default=None, description="Context window size in tokens")
    max_output: Optional[int] = Field(default=None, description="Maximum output tokens")
    pricing_notes: Optional[str] = Field(default=None, description="Additional pricing information")
    use_cases: List[str] = Field(
        default_factory=list,
        description="Recommended use cases"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations"
    )

    model_config = ConfigDict(use_enum_values=True)


class ProviderModelCards(BaseModel):
    """Collection of model cards for a provider."""

    provider_name: str = Field(description="Provider identifier")
    default_model: str = Field(description="Default model name")
    models: Dict[str, ModelCard] = Field(description="Map of model names to model cards")

    def get_model_card(self, model_name: str) -> Optional[ModelCard]:
        """Get model card by name."""
        return self.models.get(model_name)

    def resolve_model_name(self, model_or_alias: str) -> Optional[str]:
        """Resolve a model name or alias to the full model name.

        Args:
            model_or_alias: Either a full model name or a short alias

        Returns:
            Full model name if found, None otherwise
        """
        # First check if it's a direct model name
        if model_or_alias in self.models:
            return model_or_alias

        # Then check aliases
        for model_name, card in self.models.items():
            if model_or_alias in card.aliases:
                return model_name

        return None

    def get_model_card_by_alias(self, model_or_alias: str) -> Optional[ModelCard]:
        """Get model card by name or alias."""
        resolved_name = self.resolve_model_name(model_or_alias)
        if resolved_name:
            return self.models[resolved_name]
        return None

    def list_models(self) -> List[str]:
        """List available model names."""
        return list(self.models.keys())

    def list_aliases(self) -> Dict[str, str]:
        """List all aliases mapped to their full model names."""
        alias_map = {}
        for model_name, card in self.models.items():
            for alias in card.aliases:
                alias_map[alias] = model_name
        return alias_map

    def get_models_by_cost(self, cost_level: CostLevel) -> List[ModelCard]:
        """Get models filtered by cost level."""
        return [card for card in self.models.values() if card.cost_level == cost_level]

    def get_models_by_time(self, time_estimate: TimeEstimate) -> List[ModelCard]:
        """Get models filtered by time estimate."""
        return [card for card in self.models.values() if card.time_estimate == time_estimate]

    def get_models_with_capability(self, capability: ModelCapability) -> List[ModelCard]:
        """Get models that have a specific capability."""
        return [card for card in self.models.values() if capability in card.capabilities]


def create_openai_model_cards() -> ProviderModelCards:
    """Create model cards for OpenAI provider."""

    o3_deep_research = ModelCard(
        name="o3-deep-research-2025-06-26",
        display_name="OpenAI o3 Deep Research",
        description=(
            "Comprehensive deep research model optimized for in-depth analysis and "
            "synthesis. Performs exhaustive web searches with multiple iterations and "
            "produces analyst-grade reports with detailed citations."
        ),
        cost_level=CostLevel.VERY_HIGH,
        time_estimate=TimeEstimate.VERY_SLOW,
        capabilities=[
            ModelCapability.WEB_SEARCH,
            ModelCapability.REAL_TIME_DATA,
            ModelCapability.CODE_INTERPRETATION,
            ModelCapability.CITATION_TRACKING
        ],
        aliases=["o3", "o3-deep", "o3dr"],
        context_window=128000,
        pricing_notes="$10/$40 per million input/output tokens + $10/1K web searches + $0.03/code interpreter session",
        use_cases=[
            "Comprehensive research reports",
            "Academic literature reviews",
            "Market research analysis",
            "Technical deep dives",
            "Multi-source fact checking"
        ],
        limitations=[
            "Very high cost per query",
            "Long response times (5-15 minutes)",
            "Limited monthly usage quotas",
            "Requires patience for complex queries"
        ]
    )

    o4_mini_deep_research = ModelCard(
        name="o4-mini-deep-research-2025-06-26",
        display_name="OpenAI o4-mini Deep Research",
        description=(
            "Lightweight and cost-effective deep research model designed for faster "
            "responses while maintaining research quality. Ideal for latency-sensitive "
            "use cases and frequent queries."
        ),
        cost_level=CostLevel.MEDIUM,
        time_estimate=TimeEstimate.MEDIUM,
        capabilities=[
            ModelCapability.WEB_SEARCH,
            ModelCapability.REAL_TIME_DATA,
            ModelCapability.CODE_INTERPRETATION,
            ModelCapability.CITATION_TRACKING
        ],
        aliases=["o4m", "o4-mini", "o4mini", "mini"],
        context_window=128000,
        pricing_notes="$2/$8 per million input/output tokens + tool usage fees",
        use_cases=[
            "Quick research queries",
            "Fact checking",
            "News summaries",
            "Educational content",
            "Rapid prototyping of research"
        ],
        limitations=[
            "Less comprehensive than o3",
            "May require follow-up queries for complex topics",
            "Still subject to tool usage costs"
        ]
    )

    return ProviderModelCards(
        provider_name="openai",
        default_model="o3-deep-research-2025-06-26",
        models={
            "o3-deep-research-2025-06-26": o3_deep_research,
            "o4-mini-deep-research-2025-06-26": o4_mini_deep_research
        }
    )


def create_perplexity_model_cards() -> ProviderModelCards:
    """Create model cards for Perplexity provider."""

    sonar_deep_research = ModelCard(
        name="sonar-deep-research",
        display_name="Perplexity Sonar Deep Research",
        description=(
            "Comprehensive research model with real-time web search and extensive "
            "source analysis. Optimized for thorough investigation with multiple "
            "search iterations and source validation."
        ),
        cost_level=CostLevel.HIGH,
        time_estimate=TimeEstimate.SLOW,
        capabilities=[
            ModelCapability.WEB_SEARCH,
            ModelCapability.REAL_TIME_DATA,
            ModelCapability.CITATION_TRACKING,
            ModelCapability.MULTI_LANGUAGE
        ],
        aliases=["deep", "deep-research", "sdr"],
        context_window=200000,
        pricing_notes="Higher cost per query, includes comprehensive web search",
        use_cases=[
            "In-depth research projects",
            "Academic literature reviews",
            "Current events analysis",
            "Multi-source verification",
            "Comprehensive fact-checking"
        ],
        limitations=[
            "Higher cost than basic models",
            "Longer response times",
            "May over-research simple queries"
        ]
    )

    sonar_pro = ModelCard(
        name="sonar-pro",
        display_name="Perplexity Sonar Pro",
        description=(
            "Fast and efficient search model with enhanced reasoning capabilities. "
            "Balanced approach between speed and research depth, suitable for "
            "professional use cases."
        ),
        cost_level=CostLevel.MEDIUM,
        time_estimate=TimeEstimate.MEDIUM,
        capabilities=[
            ModelCapability.WEB_SEARCH,
            ModelCapability.REAL_TIME_DATA,
            ModelCapability.CITATION_TRACKING
        ],
        aliases=["pro", "sp"],
        context_window=200000,
        pricing_notes="Mid-tier pricing with good performance",
        use_cases=[
            "Business research",
            "News analysis",
            "Quick fact-checking",
            "Professional reports",
            "Market intelligence"
        ],
        limitations=[
            "Less comprehensive than deep research",
            "May require follow-up for complex topics"
        ]
    )

    sonar = ModelCard(
        name="sonar",
        display_name="Perplexity Sonar",
        description=(
            "Standard search model providing quick answers with real-time web search. "
            "Cost-effective option for basic research needs and frequent queries."
        ),
        cost_level=CostLevel.LOW,
        time_estimate=TimeEstimate.FAST,
        capabilities=[
            ModelCapability.WEB_SEARCH,
            ModelCapability.REAL_TIME_DATA,
            ModelCapability.CITATION_TRACKING
        ],
        aliases=["basic", "fast", "s"],
        context_window=100000,
        pricing_notes="Most cost-effective option",
        use_cases=[
            "Quick questions",
            "Basic fact-checking",
            "News updates",
            "Simple research queries",
            "High-frequency usage"
        ],
        limitations=[
            "Limited depth for complex topics",
            "Fewer search iterations",
            "Less comprehensive analysis"
        ]
    )

    return ProviderModelCards(
        provider_name="perplexity",
        default_model="sonar-deep-research",
        models={
            "sonar-deep-research": sonar_deep_research,
            "sonar-pro": sonar_pro,
            "sonar": sonar
        }
    )


def create_falcon_model_cards() -> ProviderModelCards:
    """Create model cards for FutureHouse Falcon provider."""

    falcon_api = ModelCard(
        name="FutureHouse Falcon API",
        display_name="FutureHouse Falcon",
        description=(
            "Specialized scientific literature search and synthesis model with access "
            "to curated academic databases and scientific literature. Optimized for "
            "research-quality academic analysis."
        ),
        cost_level=CostLevel.HIGH,
        time_estimate=TimeEstimate.SLOW,
        capabilities=[
            ModelCapability.ACADEMIC_SEARCH,
            ModelCapability.SCIENTIFIC_LITERATURE,
            ModelCapability.CITATION_TRACKING
        ],
        aliases=["falcon", "fh", "science"],
        pricing_notes="Academic research pricing, varies by usage",
        use_cases=[
            "Scientific literature reviews",
            "Academic research synthesis",
            "Medical research analysis",
            "Technical paper discovery",
            "Grant writing support"
        ],
        limitations=[
            "Limited to academic/scientific sources",
            "No general web search",
            "Filtering reliability issues noted",
            "Specialized domain focus"
        ]
    )

    return ProviderModelCards(
        provider_name="falcon",
        default_model="FutureHouse Falcon API",
        models={
            "FutureHouse Falcon API": falcon_api
        }
    )


def create_consensus_model_cards() -> ProviderModelCards:
    """Create model cards for Consensus provider."""

    consensus_search = ModelCard(
        name="Consensus Academic Search",
        display_name="Consensus AI Academic Search",
        description=(
            "Peer-reviewed academic paper search and analysis focused on evidence-based "
            "research. Provides structured analysis of academic literature with "
            "study quality assessment and meta-analysis capabilities."
        ),
        cost_level=CostLevel.LOW,
        time_estimate=TimeEstimate.FAST,
        capabilities=[
            ModelCapability.ACADEMIC_SEARCH,
            ModelCapability.CITATION_TRACKING
        ],
        aliases=["consensus", "academic", "papers", "c"],
        pricing_notes="$6.99/month for premium, free tier available",
        use_cases=[
            "Academic literature search",
            "Evidence-based research",
            "Meta-analysis preparation",
            "Systematic reviews",
            "Research validation"
        ],
        limitations=[
            "Academic papers only",
            "No web search capability",
            "Limited to peer-reviewed content",
            "Requires API approval"
        ]
    )

    return ProviderModelCards(
        provider_name="consensus",
        default_model="Consensus Academic Search",
        models={
            "Consensus Academic Search": consensus_search
        }
    )


# Registry of all provider model cards
PROVIDER_MODEL_CARDS: Dict[str, ProviderModelCards] = {
    "openai": create_openai_model_cards(),
    "perplexity": create_perplexity_model_cards(),
    "falcon": create_falcon_model_cards(),
    "consensus": create_consensus_model_cards()
}


def get_provider_model_cards(provider_name: str) -> Optional[ProviderModelCards]:
    """Get model cards for a specific provider."""
    return PROVIDER_MODEL_CARDS.get(provider_name)


def list_all_models() -> Dict[str, List[str]]:
    """List all available models by provider."""
    return {
        provider: cards.list_models()
        for provider, cards in PROVIDER_MODEL_CARDS.items()
    }


def find_models_by_cost(cost_level: CostLevel) -> Dict[str, List[ModelCard]]:
    """Find models across all providers by cost level."""
    result = {}
    for provider, cards in PROVIDER_MODEL_CARDS.items():
        models = cards.get_models_by_cost(cost_level)
        if models:
            result[provider] = models
    return result


def find_models_by_capability(capability: ModelCapability) -> Dict[str, List[ModelCard]]:
    """Find models across all providers by capability."""
    result = {}
    for provider, cards in PROVIDER_MODEL_CARDS.items():
        models = cards.get_models_with_capability(capability)
        if models:
            result[provider] = models
    return result


def resolve_model_alias(provider_name: str, model_or_alias: str) -> Optional[str]:
    """Resolve a model alias to the full model name for a specific provider.

    Args:
        provider_name: Name of the provider
        model_or_alias: Model name or alias to resolve

    Returns:
        Full model name if found, None if provider or model not found
    """
    cards = get_provider_model_cards(provider_name)
    if cards:
        return cards.resolve_model_name(model_or_alias)
    return None


def list_all_aliases() -> Dict[str, Dict[str, str]]:
    """List all aliases across all providers.

    Returns:
        Dict mapping provider names to their alias->model_name mappings
    """
    result = {}
    for provider, cards in PROVIDER_MODEL_CARDS.items():
        aliases = cards.list_aliases()
        if aliases:
            result[provider] = aliases
    return result