"""Research provider interfaces and base classes."""

from abc import ABC, abstractmethod
from typing import Optional, Union, TYPE_CHECKING

from ..models import ResearchResult, ProviderConfig
from ..model_cards import ProviderModelCards

if TYPE_CHECKING:
    from ..provider_params import BaseProviderParams


class ResearchProvider(ABC):
    """Abstract base class for deep research providers."""

    def __init__(self, config: ProviderConfig, params_or_model: Optional[Union[str, "BaseProviderParams"]] = None):
        """Initialize provider with configuration.

        Args:
            config: Provider configuration
            params_or_model: Optional model name string or provider parameters object
        """
        self.config = config
        # Extract model name if params_or_model is a string
        if isinstance(params_or_model, str):
            model = params_or_model
        elif params_or_model is not None and hasattr(params_or_model, 'model'):
            model = params_or_model.model  # type: ignore
        else:
            model = None
        self.model = self._resolve_model(model) if model else self.get_default_model()

    def _resolve_model(self, model_or_alias: str) -> str:
        """Resolve model alias to full model name.

        Args:
            model_or_alias: Model name or alias

        Returns:
            Full model name, or original input if not found
        """
        # Get model cards for this provider
        cards = self.model_cards()
        if cards:
            resolved = cards.resolve_model_name(model_or_alias)
            if resolved:
                return resolved

        # Return original if not found (might be a valid model name we don't know about)
        return model_or_alias

    def get_default_model(self) -> str:
        """Get the default model for this provider. Should be overridden."""
        return "default"

    @abstractmethod
    async def research(self, query: str) -> ResearchResult:
        """Perform research for the given query.

        Args:
            query: The research question or topic

        Returns:
            ResearchResult with markdown content and citations
        """
        pass

    def is_available(self) -> bool:
        """Check if provider is available (has API key, etc.)."""
        return self.config.enabled and self.config.api_key is not None

    @property
    def name(self) -> str:
        """Get provider name."""
        return self.config.name

    @classmethod
    def model_cards(cls) -> Optional[ProviderModelCards]:
        """Get model cards for this provider.

        Should be overridden by subclasses to provide model information.

        Returns:
            ProviderModelCards with model descriptions, costs, and capabilities
        """
        return None


class ProviderRegistry:
    """Registry for managing research providers."""

    def __init__(self):
        self._providers: dict[str, ResearchProvider] = {}

    def register(self, provider: ResearchProvider) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider

    def get_provider(self, name: str) -> Optional[ResearchProvider]:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_available_providers(self) -> list[ResearchProvider]:
        """Get all available providers."""
        return [p for p in self._providers.values() if p.is_available()]

    def get_first_available(self) -> Optional[ResearchProvider]:
        """Get the first available provider."""
        available = self.get_available_providers()
        return available[0] if available else None