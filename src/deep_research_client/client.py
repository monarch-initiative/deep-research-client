"""Main client for deep research tools."""

import asyncio
import importlib
import importlib.util
import os
import time
from datetime import datetime
from typing import Any, Optional

from .cache import CacheManager
from .models import ResearchResult, ProviderConfig, CacheConfig, QueryMetadata
from .providers import ProviderRegistry, ResearchProvider
from .provider_params import BaseProviderParams, create_provider_params

PROVIDER_CLASS_PATHS: dict[str, tuple[str, str]] = {
    "openai": ("deep_research_client.providers.openai", "OpenAIProvider"),
    "falcon": ("deep_research_client.providers.falcon", "FalconProvider"),
    "asta": ("deep_research_client.providers.asta", "AstaProvider"),
    "perplexity": ("deep_research_client.providers.perplexity", "PerplexityProvider"),
    "consensus": ("deep_research_client.providers.consensus", "ConsensusProvider"),
    "cyberian": ("deep_research_client.providers.cyberian", "CyberianProvider"),
    "openscientist": ("deep_research_client.providers.openscientist", "OpenScientistProvider"),
    "claude_code": ("deep_research_client.providers.claude_code", "ClaudeCodeProvider"),
    "biomni": ("deep_research_client.providers.biomni", "BiomniProvider"),
    "mock": ("deep_research_client.providers.mock", "MockProvider"),
}


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
            self.registry.register(self._create_provider("openai", config))

        # Edison provider (formerly Falcon/FutureHouse)
        edison_key = os.getenv("EDISON_API_KEY") or os.getenv("FUTUREHOUSE_API_KEY")
        if edison_key:
            # Show deprecation warning if using old env var
            if not os.getenv("EDISON_API_KEY") and os.getenv("FUTUREHOUSE_API_KEY"):
                import warnings
                warnings.warn(
                    "FUTUREHOUSE_API_KEY is deprecated. Please use EDISON_API_KEY instead.",
                    DeprecationWarning,
                    stacklevel=2
                )
            config = ProviderConfig(
                name="falcon",
                api_key=edison_key,
                enabled=True
            )
            self.registry.register(self._create_provider("falcon", config))

        # Asta provider
        asta_key = os.getenv("ASTA_API_KEY")
        if asta_key:
            config = ProviderConfig(
                name="asta",
                api_key=asta_key,
                enabled=True,
            )
            self.registry.register(self._create_provider("asta", config))

        # Perplexity provider
        perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        if perplexity_key:
            config = ProviderConfig(
                name="perplexity",
                api_key=perplexity_key,
                enabled=True
            )
            self.registry.register(self._create_provider("perplexity", config))

        # Consensus provider
        consensus_key = os.getenv("CONSENSUS_API_KEY")
        if consensus_key:
            config = ProviderConfig(
                name="consensus",
                api_key=consensus_key,
                enabled=True
            )
            self.registry.register(self._create_provider("consensus", config))

        # OpenScientist provider
        openscientist_key = os.getenv("OPENSCIENTIST_API_KEY")
        if openscientist_key:
            openscientist_url = os.getenv("OPENSCIENTIST_URL", "https://www.openscientist.io")
            config = ProviderConfig(
                name="openscientist",
                api_key=openscientist_key,
                base_url=openscientist_url,
                enabled=True,
                timeout=3600,  # 1 hour default for long-running jobs
            )
            self.registry.register(self._create_provider("openscientist", config))

        # Cyberian provider - check if cyberian is installed
        try:
            import cyberian  # type: ignore[import-not-found, import-untyped]  # noqa: F401
            cyberian_config = ProviderConfig(
                name="cyberian",
                api_key=None,  # Not required for cyberian
                enabled=True,
                timeout=1800  # 30 minutes for long-running workflows
            )
            self.registry.register(self._create_provider("cyberian", cyberian_config))
        except ImportError:
            pass  # Cyberian not installed, skip

        # Biomni provider - check if the optional biomni package is installed.
        # Biomni configures its own underlying LLM (via ANTHROPIC_API_KEY etc.),
        # so no provider API key is required at this layer.
        # Set DISABLE_BIOMNI_PROVIDER=true to opt out of auto-detection.
        if (
            os.getenv("DISABLE_BIOMNI_PROVIDER", "").lower() not in ("true", "1", "yes")
            and importlib.util.find_spec("biomni") is not None
        ):
            biomni_config = ProviderConfig(
                name="biomni",
                api_key=None,  # Not required; biomni authenticates its own LLM
                enabled=True,
                timeout=3600,  # Agentic runs with local code execution are slow
            )
            self.registry.register(self._create_provider("biomni", biomni_config))

        # Claude Code provider - available whenever the `claude` CLI is on PATH.
        # No API key required; auth/billing is handled by the local installation.
        # Set DISABLE_CLAUDE_CODE_PROVIDER=true to opt out of auto-detection.
        #
        # NOTE: auto-detection deliberately probes the default "claude" executable
        # rather than constructing the provider and calling is_available(), to keep
        # env setup import-light. A custom claude_executable (a ClaudeCodeParams
        # field, not env-derived) is therefore not honored here; pass an explicit
        # provider_config to register the provider against a non-default binary.
        import shutil
        if (
            os.getenv("DISABLE_CLAUDE_CODE_PROVIDER", "").lower() not in ("true", "1", "yes")
            and shutil.which("claude") is not None
        ):
            claude_code_config = ProviderConfig(
                name="claude_code",
                api_key=None,  # Not required for Claude Code
                enabled=True,
                timeout=1800,  # 30 minutes for long-running agentic research
            )
            self.registry.register(self._create_provider("claude_code", claude_code_config))

        # Mock provider only if explicitly requested via environment
        if os.getenv("ENABLE_MOCK_PROVIDER", "").lower() in ("true", "1", "yes"):
            mock_config = ProviderConfig(
                name="mock",
                api_key="mock-key",  # Not required but needed for config
                enabled=True
            )
            self.registry.register(self._create_provider("mock", mock_config))

    def _setup_providers_from_config(self, configs: dict[str, ProviderConfig]) -> None:
        """Setup providers from provided configurations."""
        for name, config in configs.items():
            self.registry.register(self._create_provider(name, config))

    def _get_provider_class(self, provider_name: str) -> type[ResearchProvider]:
        """Resolve a provider class only when it is actually needed."""
        class_path = PROVIDER_CLASS_PATHS.get(provider_name)
        if class_path is None:
            raise ValueError(f"Unknown provider: {provider_name}")

        module_name, class_name = class_path
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _create_provider(
        self,
        provider_name: str,
        config: ProviderConfig,
        params: str | BaseProviderParams | None = None,
    ) -> ResearchProvider:
        """Instantiate a provider via the lazy class loader."""
        provider_class = self._get_provider_class(provider_name)
        return provider_class(config, params)

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
        return self._create_provider(provider_name, config, params)

    def _get_cache_provider_params(
        self,
        research_provider: 'ResearchProvider',
        provider_params: Optional[dict] = None,
    ) -> Optional[dict]:
        """Build effective cache parameters, including provider-specific cache busting."""
        effective_params = dict(provider_params or {})

        # Asta response parsing and paper metadata changed after initial release;
        # keep stale cache entries from shadowing current live results.
        if research_provider.name == "asta":
            effective_params["_cache_version"] = "snippet-v5"
        elif research_provider.name in {"falcon", "openscientist"}:
            effective_params["_cache_version"] = "artifacts-v1"
        # The Claude Code prompt scaffolding (inline-report directive) and run
        # provenance capture changed how results are produced; bump to keep
        # stale cache entries from shadowing current live results.
        elif research_provider.name == "claude_code":
            effective_params["_cache_version"] = "inline-report-v1"

        return effective_params or None

    def research(
        self,
        query: str,
        provider: Optional[str] = None,
        template_info: Optional[dict] = None,
        model: Optional[str] = None,
        provider_params: Optional[dict] = None,
        metadata: Optional[dict] = None
    ) -> ResearchResult:
        """Perform research on the given query.

        Args:
            query: The research question or topic
            provider: Specific provider to use (uses first available if None)
            template_info: Template information if query was generated from template
            model: Model to use for the provider (overrides provider default)
            provider_params: Provider-specific parameters
            metadata: Publication-style metadata (title, abstract, keywords, author, contributors)

        Returns:
            ResearchResult with markdown content and citations

        Raises:
            ValueError: If no providers are available or specified provider not found
        """
        return asyncio.run(self.aresearch(query, provider, template_info, model, provider_params, metadata))

    async def aresearch(
        self,
        query: str,
        provider: Optional[str] = None,
        template_info: Optional[dict] = None,
        model: Optional[str] = None,
        provider_params: Optional[dict] = None,
        metadata: Optional[dict] = None
    ) -> ResearchResult:
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
        cache_provider_params = self._get_cache_provider_params(research_provider, provider_params)
        cached_result = await self.cache.get(query, research_provider.name, model, cache_provider_params)
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

        # Add publication-style metadata if provided
        if metadata:
            if 'title' in metadata:
                result.title = metadata['title']
            if 'abstract' in metadata:
                result.abstract = metadata['abstract']
            if 'keywords' in metadata:
                result.keywords = metadata['keywords']
            if 'author' in metadata or 'contributors' in metadata:
                result.query_metadata = QueryMetadata(
                    author=metadata.get('author'),
                    contributors=metadata.get('contributors', [])
                )

        # Add provider configuration. Prefer a model the provider already recorded
        # on the result (e.g. the actual model id reported by the run) over the
        # provider's configured/default model.
        if result.model is None:
            result.model = getattr(research_provider, 'model', None)
        result.provider_config = {
            'timeout': research_provider.config.timeout,
            'max_retries': research_provider.config.max_retries,
        }

        # Add provider-specific parameters if they exist
        params = getattr(research_provider, "params", None)
        if isinstance(params, BaseProviderParams):
            # Convert Pydantic model to dict, excluding None values and model field
            params_dict = params.model_dump(exclude_none=True, exclude={'model'})
            if params_dict:
                result.provider_config['parameters'] = params_dict

        # Cache the result
        await self.cache.set(query, research_provider.name, result, model, cache_provider_params)

        return result

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return [p.name for p in self.registry.get_available_providers()]

    def clear_cache(self) -> int:
        """Clear all cached results and return count of files removed."""
        return self.cache.clear_cache()

    def list_cached_files(self) -> list:
        """List all cached files with human-readable names."""
        cache_files = self.cache.list_cache_files()
        return [{"path": str(f), "name": f.name} for f in cache_files]

    def get_cache_info(self) -> list[dict[str, Any]]:
        """Get detailed info for all cached files.

        Returns list of dicts with metadata including query, provider,
        model, timing info, and file stats.
        """
        return self.cache.get_cache_info()

    def search_cache(self, keyword: str, context_chars: int = 60, max_snippets: int = 3) -> list[dict[str, Any]]:
        """Search cached files for keyword in query or content.

        Args:
            keyword: Case-insensitive keyword to search for
            context_chars: Characters of context around matches
            max_snippets: Maximum snippets per field

        Returns:
            List of cache info dicts that match the keyword
        """
        return self.cache.search_cache(keyword, context_chars, max_snippets)

    def export_cache_for_browser(self, include_content: bool = False) -> list[dict[str, Any]]:
        """Export cache data in format suitable for linkml-browser.

        Args:
            include_content: If True, include full markdown and citations

        Returns:
            List of dicts with standardized fields for faceted browsing
        """
        return self.cache.export_for_browser(include_content=include_content)
