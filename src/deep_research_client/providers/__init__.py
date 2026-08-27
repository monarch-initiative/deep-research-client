"""Research provider interfaces and base classes."""

from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union, TYPE_CHECKING

from ..models import ResearchResult, ProviderConfig, ProviderHealth
from ..model_cards import ProviderModelCards

if TYPE_CHECKING:
    from ..provider_params import BaseProviderParams


class ResearchProvider(ABC):
    """Abstract base class for deep research providers."""

    #: Environment variable that supplies this provider's credential, if it
    #: needs one. Set by subclasses so the base can name it in error messages.
    credential_env_var: ClassVar[Optional[str]] = None

    #: Human-facing name for this provider's credential, e.g. "OpenAI".
    credential_label: ClassVar[Optional[str]] = None

    #: Whether this provider's output is real research. False for a provider
    #: that fabricates its reports, which keeps it out of the *automatic*
    #: fallback ordering: a run that ran out of credits should fail rather
    #: than quietly hand a curation record a made-up report. Naming such a
    #: provider explicitly still honours it -- that is a deliberate choice,
    #: not a stand-in nobody asked for.
    produces_real_reports: ClassVar[bool] = True

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

    def unavailable_reason(self) -> str:
        """Explain why this provider is unavailable.

        Called when a caller explicitly requests a provider that fails
        :meth:`is_available`. Subclasses should override this when they can say
        something more useful than "no API key" -- for example a stub provider
        whose upstream service does not exist yet.

        Returns:
            Human-readable explanation suitable for an error message
        """
        if not self.config.enabled:
            return f"Provider '{self.name}' is disabled"
        if self.credential_env_var:
            label = self.credential_label or self.name
            return f"no {label} API key configured (set {self.credential_env_var})"
        return f"Provider '{self.name}' is not available"

    async def check_health(self) -> ProviderHealth:
        """Probe whether this provider can actually take work right now.

        :meth:`is_available` only answers "is a key configured". This answers
        the more useful question, at the cost of a network round trip.
        Subclasses should override with the cheapest authenticated call their
        API offers -- never a full research run.

        The default implementation performs no probe, so ``reachable`` is None:
        configuration is all we know.

        Returns:
            Health record for this provider
        """
        if not self.is_available():
            return ProviderHealth(
                provider=self.name,
                configured=False,
                reachable=False,
                detail=self.unavailable_reason(),
            )
        return ProviderHealth(provider=self.name, configured=True)

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
