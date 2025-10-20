"""Main client for deep research tools."""

import asyncio
import os
import time
from datetime import datetime
from typing import Optional, List

from .cache import CacheManager
from .models import ResearchResult, ProviderConfig, CacheConfig
from .providers import ProviderRegistry, ResearchProvider
from .providers.openai import OpenAIProvider
from .providers.falcon import FalconProvider
from .providers.perplexity import PerplexityProvider
from .providers.consensus import ConsensusProvider
from .providers.mock import MockProvider
from .provider_params import create_provider_params


class DeepResearchClient:
    """Main client for accessing deep research tools."""

    def __init__(
        self,
        cache_config: Optional[CacheConfig] = None,
        provider_configs: Optional[dict[str, ProviderConfig]] = None
    ):
        """Initialize the client with optional configurations.

        Args:
            cache_config: Cache configuration (uses defaults if None)
            provider_configs: Provider configurations (auto-detects from env if None)
        """
        # Initialize cache
        self.cache_config = cache_config or CacheConfig()
        self.cache = CacheManager(self.cache_config)

        # Initialize provider registry
        self.registry = ProviderRegistry()

        # Setup providers
        if provider_configs:
            self._setup_providers_from_config(provider_configs)
        else:
            self._setup_providers_from_env()

    def _setup_providers_from_env(self) -> None:
        """Setup providers by auto-detecting from environment variables."""
        # OpenAI provider
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            config = ProviderConfig(
                name="openai",
                api_key=openai_key,
                enabled=True
            )
            self.registry.register(OpenAIProvider(config, None))

        # Falcon provider
        falcon_key = os.getenv("FUTUREHOUSE_API_KEY")
        if falcon_key:
            config = ProviderConfig(
                name="falcon",
                api_key=falcon_key,
                enabled=True
            )
            self.registry.register(FalconProvider(config, None))

        # Perplexity provider
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if perplexity_key:
            config = ProviderConfig(
                name="perplexity",
                api_key=perplexity_key,
                enabled=True
            )
            self.registry.register(PerplexityProvider(config, None))

        # Consensus provider
        consensus_key = os.getenv("CONSENSUS_API_KEY")
        if consensus_key:
            config = ProviderConfig(
                name="consensus",
                api_key=consensus_key,
                enabled=True
            )
            self.registry.register(ConsensusProvider(config, None))

        # Mock provider only if explicitly requested via environment
        if os.getenv("ENABLE_MOCK_PROVIDER", "").lower() in ("true", "1", "yes"):
            mock_config = ProviderConfig(
                name="mock",
                api_key="mock-key",  # Not required but needed for config
                enabled=True
            )
            mock_provider = MockProvider(mock_config, None)
            self.registry.register(mock_provider)

    def _setup_providers_from_config(self, configs: dict[str, ProviderConfig]) -> None:
        """Setup providers from provided configurations."""
        for name, config in configs.items():
            provider: ResearchProvider
            if name == "openai":
                provider = OpenAIProvider(config, None)
            elif name == "falcon":
                provider = FalconProvider(config, None)
            elif name == "perplexity":
                provider = PerplexityProvider(config, None)
            elif name == "consensus":
                provider = ConsensusProvider(config, None)
            elif name == "mock":
                provider = MockProvider(config, None)
            else:
                raise ValueError(f"Unknown provider: {name}")

            self.registry.register(provider)

    def _create_provider_with_params(self, provider_name: str, model: Optional[str] = None, provider_params: Optional[dict] = None) -> 'ResearchProvider':
        """Create a provider instance with custom parameters.

        Args:
            provider_name: Name of the provider to create
            model: Model to use (overrides default)
            provider_params: Provider-specific parameters

        Returns:
            Provider instance with custom parameters

        Raises:
            ValueError: If provider not found or parameters are invalid
        """
        # Get the base provider config
        base_provider = self.registry.get_provider(provider_name)
        if not base_provider:
            raise ValueError(f"Provider '{provider_name}' not found")

        config = base_provider.config

        # Create validated parameters using Pydantic models
        params = create_provider_params(provider_name, model, provider_params)

        # Get provider class and create instance
        provider_class = type(base_provider)
        return provider_class(config, params)

    def research(self, query: str, provider: Optional[str] = None, template_info: Optional[dict] = None, model: Optional[str] = None, provider_params: Optional[dict] = None) -> ResearchResult:
        """Perform research on the given query.

        Args:
            query: The research question or topic
            provider: Specific provider to use (uses first available if None)
            template_info: Template information if query was generated from template
            model: Model to use for the provider (overrides provider default)
            provider_params: Provider-specific parameters

        Returns:
            ResearchResult with markdown content and citations

        Raises:
            ValueError: If no providers are available or specified provider not found
        """
        return asyncio.run(self.aresearch(query, provider, template_info, model, provider_params))

    async def aresearch(self, query: str, provider: Optional[str] = None, template_info: Optional[dict] = None, model: Optional[str] = None, provider_params: Optional[dict] = None) -> ResearchResult:
        """Async version of research method."""
        start_time = datetime.now()
        start_timestamp = time.time()

        # Get provider
        if provider:
            base_provider = self.registry.get_provider(provider)
            if not base_provider:
                raise ValueError(f"Provider '{provider}' not found")
            if not base_provider.is_available():
                raise ValueError(f"Provider '{provider}' is not available")

            # Create new instance with custom parameters if needed
            if provider_params or model:
                research_provider = self._create_provider_with_params(provider, model, provider_params or {})
            else:
                research_provider = base_provider
        else:
            base_provider = self.registry.get_first_available()
            if not base_provider:
                raise ValueError("No research providers available")

            provider_name = base_provider.name
            # Create new instance with custom parameters if needed
            if provider_params or model:
                research_provider = self._create_provider_with_params(provider_name, model, provider_params or {})
            else:
                research_provider = base_provider

        # Check cache first
        cached_result = await self.cache.get(query, research_provider.name, model, provider_params)
        if cached_result:
            # Update timing for cached results
            end_time = datetime.now()
            cached_result.start_time = start_time
            cached_result.end_time = end_time
            cached_result.duration_seconds = time.time() - start_timestamp
            return cached_result

        # Perform research
        result = await research_provider.research(query)

        # Add timing and metadata
        end_time = datetime.now()
        result.start_time = start_time
        result.end_time = end_time
        result.duration_seconds = time.time() - start_timestamp

        # Add template information if provided
        if template_info:
            result.template_file = template_info.get('template_file')
            result.template_variables = template_info.get('template_variables')

        # Add provider configuration
        result.model = getattr(research_provider, 'model', None)
        result.provider_config = {
            'timeout': research_provider.config.timeout,
            'max_retries': research_provider.config.max_retries,
        }

        # Add provider-specific parameters if they exist
        if hasattr(research_provider, 'params') and research_provider.params:
            # Convert Pydantic model to dict, excluding None values and model field
            params_dict = research_provider.params.model_dump(exclude_none=True, exclude={'model'})
            if params_dict:
                result.provider_config['parameters'] = params_dict

        # Cache the result
        await self.cache.set(query, research_provider.name, result, model, provider_params)

        return result

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [p.name for p in self.registry.get_available_providers()]

    def clear_cache(self) -> int:
        """Clear all cached results and return count of files removed."""
        return self.cache.clear_cache()

    def list_cached_files(self) -> list:
        """List all cached files with human-readable names."""
        cache_files = self.cache.list_cache_files()
        return [{"path": str(f), "name": f.name} for f in cache_files]