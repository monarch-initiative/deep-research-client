"""Provider-specific parameter models using Pydantic for validation."""

from typing import Optional, Literal, List, Type, Any, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator


class BaseProviderParams(BaseModel):
    """Base provider parameters that all providers can accept."""

    model: Optional[str] = Field(default=None, description="Model to use for this provider")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt to override the default research prompt"
    )
    allowed_domains: List[str] = Field(
        default_factory=list,
        description=(
            "Harmonized parameter: Filter web search to specific domains (max 20). "
            "Only include results from these domains. "
            "Example: ['wikipedia.org', 'github.com']. "
            "Use domain names without protocols (http/https)."
        )
    )

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        validate_assignment=True  # Validate on assignment
    )


class PerplexityParams(BaseProviderParams):
    """Parameters specific to Perplexity AI provider.

    Note: Both `allowed_domains` (harmonized) and `search_domain_filter` (Perplexity-specific)
    are supported. If `allowed_domains` is provided and `search_domain_filter` is empty,
    `allowed_domains` will be used. The Perplexity-specific `search_domain_filter` supports
    both allowlist and denylist (prefix with '-' to exclude).
    """

    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Reasoning effort level for Perplexity"
    )
    search_recency_filter: Optional[str] = Field(
        default=None,
        description="Filter sources by recency (e.g., 'month', 'week', 'year')"
    )
    search_domain_filter: List[str] = Field(
        default_factory=list,
        description=(
            "Provider-specific alias: Filter search results by domains or URLs. "
            "Supports allowlist (include) and denylist (exclude) modes. "
            "Maximum 20 domains/URLs per request.\n"
            "Examples:\n"
            "  Allowlist: ['wikipedia.org', 'github.com'] - only these domains\n"
            "  Denylist: ['-reddit.com', '-quora.com'] - exclude these domains\n"
            "  Mixed: ['github.com', 'stackoverflow.com', '-reddit.com']\n"
            "Can use domain names (e.g., 'wikipedia.org') or specific URLs.\n"
            "Use simple domain names without protocols (http/https).\n"
            "Note: You can also use the harmonized `allowed_domains` parameter instead."
        )
    )
    return_citations: bool = Field(
        default=True,
        description="Whether to return structured citations"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )

    @model_validator(mode='after')
    def sync_domain_filters(self):
        """Sync allowed_domains with search_domain_filter.

        If allowed_domains is provided but search_domain_filter is empty,
        use allowed_domains as search_domain_filter.
        """
        if self.allowed_domains and not self.search_domain_filter:
            self.search_domain_filter = self.allowed_domains
        return self


class OpenAIParams(BaseProviderParams):
    """Parameters specific to OpenAI provider.

    Supports the harmonized `allowed_domains` parameter (inherited from BaseProviderParams)
    to filter web search results to specific domains (max 20).
    """

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens in response"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )


class FalconParams(BaseProviderParams):
    """Parameters specific to Falcon provider."""

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens in response"
    )


class ConsensusParams(BaseProviderParams):
    """Parameters specific to Consensus provider."""

    year_min: Optional[int] = Field(
        default=None,
        description="Minimum publication year for papers"
    )
    year_max: Optional[int] = Field(
        default=None,
        description="Maximum publication year for papers"
    )
    study_types: List[str] = Field(
        default_factory=list,
        description="Filter by study types (e.g., 'RCT', 'Systematic Review')"
    )
    sample_size_min: Optional[int] = Field(
        default=None,
        gt=0,
        description="Minimum sample size for studies"
    )


class MockParams(BaseProviderParams):
    """Parameters specific to Mock provider for testing."""

    response_delay: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Artificial delay in seconds to simulate API call"
    )
    response_length: Literal["short", "medium", "long"] = Field(
        default="medium",
        description="Length of mock response"
    )
    include_error: bool = Field(
        default=False,
        description="Whether to simulate an error response"
    )
    custom_response: Optional[str] = Field(
        default=None,
        description="Custom response text instead of default"
    )


# Registry mapping provider names to their parameter models
PROVIDER_PARAMS_REGISTRY: dict[str, Type[BaseProviderParams]] = {
    "perplexity": PerplexityParams,
    "openai": OpenAIParams,
    "falcon": FalconParams,
    "consensus": ConsensusParams,
    "mock": MockParams,
}


def get_provider_params_class(provider_name: str) -> type[BaseProviderParams]:
    """Get the parameter model class for a provider.

    Args:
        provider_name: Name of the provider

    Returns:
        Parameter model class for the provider

    Raises:
        ValueError: If provider is not found in registry
    """
    params_class = PROVIDER_PARAMS_REGISTRY.get(provider_name)
    if not params_class:
        raise ValueError(f"No parameter model found for provider: {provider_name}")
    return params_class


def create_provider_params(
    provider_name: str,
    model: Optional[str] = None,
    provider_params: Optional[Dict[str, Any]] = None
) -> BaseProviderParams:
    """Create and validate provider parameters.

    Args:
        provider_name: Name of the provider
        model: Model to use (overrides provider default)
        provider_params: Provider-specific parameters

    Returns:
        Validated provider parameters instance

    Raises:
        ValueError: If validation fails or provider not found
    """
    params_class = get_provider_params_class(provider_name)

    # Prepare parameter data
    param_data: Dict[str, Any] = {}
    if model:
        param_data["model"] = model
    if provider_params:
        param_data.update(provider_params)

    # Validate and create parameters
    try:
        return params_class(**param_data)
    except Exception as e:
        raise ValueError(f"Invalid parameters for {provider_name}: {e}")