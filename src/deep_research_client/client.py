"""Main client for deep research tools."""

import asyncio
import importlib
import logging
import os
import time
from datetime import datetime
from collections.abc import Sequence
from typing import Any, Optional, Union

from .cache import CacheManager
from .exceptions import ProviderError, ProviderNotConfiguredError, is_fallback_worthy
from .models import (
    ResearchResult,
    ProviderAttempt,
    ProviderConfig,
    CacheConfig,
    QueryMetadata,
)
from .providers import ProviderRegistry, ResearchProvider
from .provider_params import BaseProviderParams, create_provider_params

logger = logging.getLogger(__name__)

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
    "deeper_med": ("deep_research_client.providers.deeper_med", "DeeperMedProvider"),
    "mock": ("deep_research_client.providers.mock", "MockProvider"),
}


#: Providers whose registration is gated on an environment variable rather than
#: on the provider's own availability, so the provider cannot explain itself.
REGISTRATION_GATES: dict[str, str] = {
    # Phrased as requirements, not findings: absence from the registry has
    # more than one cause (explicit provider_configs skip env detection
    # entirely), so a sentence asserting *why* would sometimes be false.
    "claude_code": (
        "requires the local Claude Code CLI, with DISABLE_CLAUDE_CODE_PROVIDER unset"
    ),
    "biomni": (
        "requires an upstream Biomni environment and deep-research-client[biomni], "
        "with DISABLE_BIOMNI_PROVIDER unset"
    ),
    "mock": "set ENABLE_MOCK_PROVIDER=true to enable the mock provider",
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

        # Biomni provider - require the complete core Python runtime, rather
        # than only the biomni package spec: biomni 0.0.8 under-declares A1's
        # eager imports. Biomni configures its own underlying LLM (via
        # ANTHROPIC_API_KEY etc.), so no provider API key is required here.
        # Set DISABLE_BIOMNI_PROVIDER=true to opt out of auto-detection.
        from .providers.biomni import (
            BIOMNI_DEFAULT_TIMEOUT,
            missing_biomni_runtime_modules,
        )

        if (
            os.getenv("DISABLE_BIOMNI_PROVIDER", "").lower() not in ("true", "1", "yes")
            and not missing_biomni_runtime_modules()
        ):
            biomni_config = ProviderConfig(
                name="biomni",
                api_key=None,  # Not required; biomni authenticates its own LLM
                enabled=True,
                # Agentic runs with local code execution are slow.
                timeout=BIOMNI_DEFAULT_TIMEOUT,
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

        # DeepER-Med is a stub with no upstream endpoint. It is registered
        # unconditionally and needs no credentials: is_available() is always False,
        # so it can never be auto-selected by get_first_available(). Registering it
        # means an explicit request for it reports why it cannot run (with the
        # arXiv pointer) instead of a bare "Provider not found".
        deeper_med_config = ProviderConfig(
            name="deeper_med",
            api_key=None,  # No upstream service exists to authenticate against
            enabled=True,
        )
        self.registry.register(self._create_provider("deeper_med", deeper_med_config))

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

    def unregistered_reason(self, provider_name: str) -> str:
        """Explain why a known provider cannot do work for this client.

        Public because the CLI is now a first-class caller: it renders this in
        `providers` and `providers --check` rather than keeping a second
        opinion of its own.

        A provider that *can* work is answered here rather than below, because
        the explanation below is built from a throwaway instance carrying no
        credentials -- so for a registered, key-bearing provider it would
        confidently report the key as missing. Callers in this package guard
        the call already; a public method must not depend on their doing so.

        The gate is availability rather than registration: deeper_med is
        registered unconditionally and still needs its own explanation.

        Args:
            provider_name: Canonical name of a provider in PROVIDER_CLASS_PATHS.

        Returns:
            Human-readable explanation of what is missing, or a statement that
            the provider is in fact usable
        """
        if provider_name in self.get_available_providers():
            return f"'{provider_name}' is available"
        return self._unregistered_reason(provider_name)

    def _unregistered_reason(self, provider_name: str) -> str:
        """Explain why a known provider never made it into the registry.

        Asks the provider class itself where it can answer, so the wording
        matches every other surface. But registration and availability are not
        the same gate: the providers in REGISTRATION_GATES are held back by an
        environment variable while considering themselves perfectly available,
        and asking those why they are unavailable produces a confident wrong
        answer -- telling a reader to install a CLI they already have, for
        instance.

        Args:
            provider_name: Canonical name of a provider in PROVIDER_CLASS_PATHS.

        Returns:
            Human-readable explanation of what is missing
        """
        try:
            provider_class = self._get_provider_class(provider_name)
            provider = provider_class(ProviderConfig(name=provider_name))
        except Exception:
            # A diagnostic path must not fail with a second, unrelated error.
            # The class attributes are readable even when the instance is not,
            # so fall back to those before giving up on saying anything useful.
            logger.debug("Could not build %s to explain itself:", provider_name, exc_info=True)
            return self._reason_from_class_attributes(provider_name)

        if provider.is_available():
            # The class has nothing to explain: the gate is outside it.
            return REGISTRATION_GATES.get(
                provider_name, f"'{provider_name}' is not registered in this environment"
            )
        return provider.unavailable_reason()

    def _reason_from_class_attributes(self, provider_name: str) -> str:
        """Explain a provider without instantiating it.

        Args:
            provider_name: Canonical name of a provider in PROVIDER_CLASS_PATHS.

        Returns:
            Human-readable explanation of what is missing
        """
        gate = REGISTRATION_GATES.get(provider_name)
        if gate:
            return gate
        paths = PROVIDER_CLASS_PATHS.get(provider_name)
        if paths is None:
            return f"'{provider_name}' is not configured"
        module_name, class_name = paths
        try:
            provider_class = getattr(importlib.import_module(module_name), class_name)
        except Exception:
            return f"'{provider_name}' is not configured"
        if provider_class.credential_env_var:
            label = provider_class.credential_label or provider_name
            return f"no {label} API key configured (set {provider_class.credential_env_var})"
        return f"'{provider_name}' is not configured"

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

    @staticmethod
    def _get_cache_provider_params(
        provider_name: str,
        provider_params: Optional[dict] = None,
    ) -> Optional[dict]:
        """Build effective cache parameters, including provider-specific cache busting.

        Takes a name rather than a provider, because a cache key needs nothing
        an instance has: that is what lets a cached report be served for a
        provider this run cannot construct.

        Args:
            provider_name: Name of the provider whose cache entry is wanted.
            provider_params: Provider-specific parameters for this run.

        Returns:
            Parameters to key the cache entry by, or None when there are none.
        """
        effective_params = dict(provider_params or {})

        # Asta response parsing and paper metadata changed after initial release;
        # keep stale cache entries from shadowing current live results.
        if provider_name == "asta":
            effective_params["_cache_version"] = "snippet-v5"
        elif provider_name in {"falcon", "openscientist"}:
            effective_params["_cache_version"] = "artifacts-v1"
        # The Claude Code prompt scaffolding (inline-report directive) and run
        # provenance capture changed how results are produced; bump to keep
        # stale cache entries from shadowing current live results.
        elif provider_name == "claude_code":
            effective_params["_cache_version"] = "inline-report-v1"

        return effective_params or None

    def _fallback_candidates(
        self,
        provider: Optional[str],
        fallback: Optional[Union[bool, str, Sequence[str]]],
    ) -> list[str]:
        """Work out which providers to try, in order.

        Args:
            provider: Provider the caller named, if any.
            fallback: False or None for no fallback, True to fall back to
                whatever else is available, or an ordered list of provider
                names. A single name may be given as a bare string.

        Logs at INFO when a fallback was requested but only one candidate
        survives, since the run is then about to behave as though no fallback
        had been asked for. Planning and saying so live together because the
        reason -- which filter or which absent name -- is known only here.

        Returns:
            Provider names to try, most-preferred first, without duplicates.
            With True, the automatic ones follow registration order.

        Raises:
            ValueError: If a named fallback provider does not exist, or there
                is nothing at all to try.
        """
        if isinstance(fallback, str):
            # A Sequence[str] accepts a str, and list("openai") would quietly
            # become six one-letter providers. The singular form is the obvious
            # thing to reach for next to a list, so honour it.
            extra = [fallback]
        elif fallback is None or isinstance(fallback, bool):
            # None arrives the same way the bare string does -- a wrapper
            # passing config.get("fallback") for an absent key -- and means
            # the same thing as False. Falling through would reach list(None).
            extra = (
                [
                    candidate.name
                    for candidate in self.registry.get_available_providers()
                    # A provider that invents its reports is not a stand-in for
                    # one that does the work. Name it and it is honoured.
                    if candidate.produces_real_reports
                ]
                if fallback
                else []
            )
        else:
            # An explicit list is an instruction, so names are kept even when
            # unavailable: each one then reports why it could not be used,
            # rather than vanishing from the trail without explanation.
            extra = list(fallback)

        # Names are checked before any provider is called. A typo found
        # mid-run would abort after the first provider had already been paid
        # for, never reach the good candidate behind it, and replace the
        # failure that started the fallback with a bare "not found".
        unknown = [name for name in extra if not self.knows_provider(name)]
        if unknown:
            raise ValueError(
                f"Provider '{unknown[0]}' not found"
                if len(unknown) == 1
                else f"Providers not found: {', '.join(unknown)}"
            )

        if provider:
            ordered = [provider]
        else:
            first_available = self.registry.get_first_available()
            ordered = [first_available.name] if first_available else []

        for name in extra:
            if name not in ordered:
                ordered.append(name)

        if not ordered:
            raise ValueError("No research providers available")

        # A fallback was asked for and there is nobody to fall back to. The
        # run is about to behave exactly as it would with no flag at all, and
        # without this nothing distinguishes that from "the fallback ran and
        # also failed" -- the trail is empty either way at position 0.
        #
        # Both routes are easy to hit: the automatic ordering drops providers
        # that do not produce real reports, and a named fallback that repeats
        # the primary deduplicates away. The reason is the invisible part, so
        # say which one it was.
        if fallback and len(ordered) == 1:
            # `is True`, not truthiness: a list is truthy too, and the
            # produces_real_reports filter above runs only on the automatic
            # route. An explicitly named provider that invents its reports is
            # honoured, so on the list route a mock sitting in the registry
            # was not excluded for fabricating -- it simply was not named.
            # Blaming the filter there would tell an operator their own
            # --fallback-provider mock would be dropped, which is the reverse
            # of what happens.
            excluded = (
                [
                    candidate.name
                    for candidate in self.registry.get_available_providers()
                    if not candidate.produces_real_reports
                    and candidate.name != ordered[0]
                ]
                if fallback is True
                else []
            )
            # Three reasons, not two. "named or available" as one string
            # asserts on the list route that nobody else is here, when the
            # truth is usually that nobody else was listed -- and listing one
            # is the fix the operator needs pointing at.
            if excluded:
                reason = (
                    "the other available provider(s) do not produce real "
                    f"reports ({', '.join(excluded)})"
                )
            elif fallback is True:
                reason = "no other provider is available"
            else:
                reason = "no other provider was named"
            logger.info(
                "Fallback was requested, but %s is the only candidate: %s. "
                "The run will behave as though no fallback was asked for",
                ordered[0],
                reason,
            )
        return ordered

    def _prepare_provider(
        self,
        provider_name: str,
        model: Optional[str],
        provider_params: Optional[dict],
    ) -> ResearchProvider:
        """Resolve one candidate into a provider instance ready to be called.

        Args:
            provider_name: Name of the provider to prepare.
            model: Model override, if the caller gave one.
            provider_params: Provider-specific parameters, if any.

        Returns:
            A provider instance configured for this run.

        Raises:
            ProviderNotConfiguredError: If the provider is known but not set up.
            ValueError: If the name is not a provider at all.
        """
        base_provider = self.registry.get_provider(provider_name)
        if not base_provider:
            # Providers register only once their credential is present, so
            # a known name that is absent means "not set up", not "no such
            # thing". Saying "not found" sends the caller hunting a typo.
            if self.knows_provider(provider_name):
                raise ProviderNotConfiguredError(
                    provider_name, self._unregistered_reason(provider_name)
                )
            raise ValueError(f"Provider '{provider_name}' not found")
        if not base_provider.is_available():
            raise ProviderNotConfiguredError(
                provider_name, base_provider.unavailable_reason()
            )

        # Create new instance with custom parameters if needed
        if provider_params or model:
            return self._create_provider_with_params(
                provider_name, model, provider_params or {}
            )
        return base_provider

    @staticmethod
    def _warn_falling_back(
        failed_provider: str,
        next_provider: str,
        exc: BaseException,
        model: Optional[str],
        provider_params: Optional[dict],
        overridden_provider: str,
    ) -> None:
        """Say which provider is taking over, and what is being dropped to do it.

        Args:
            failed_provider: The provider that could not do the work.
            next_provider: The provider about to be tried.
            exc: The failure that ended the previous attempt.
            model: Model override the caller asked for, if any.
            provider_params: Provider-specific parameters the caller asked for.
            overridden_provider: The one candidate the overrides were applied
                to. Not the same as ``failed_provider`` past the first hop,
                where the provider that just failed had no overrides either --
                and a message about provenance must not assert otherwise.
        """
        logger.warning(
            "Provider %s cannot do this run: %s. Falling back to %s",
            failed_provider, exc, next_provider,
        )
        dropped = [
            label
            for label, value in (("--model", model), ("--param", provider_params))
            if value
        ]
        if dropped:
            logger.warning(
                "%s runs on its own defaults: %s applied to %s, not to it",
                next_provider, " and ".join(dropped), overridden_provider,
            )

    @staticmethod
    def _attach_trail(
        exc: BaseException, failed: list[ProviderAttempt], candidate: str
    ) -> None:
        """Record every provider tried on the failure that ends the run.

        With no result there is no ``provider_attempts`` to read, which is
        exactly the case a caller most wants it. This puts the trail on the
        exception instead -- including the candidate whose failure is being
        raised, so the list is the whole run and not everything before it.

        A failure we did not classify has nowhere to carry it -- a provider
        that cannot recognise its own SDK error re-raises it bare, and a plain
        exception has no field for this. That is the run where the fallback
        machinery worked hardest, so the trail is logged rather than lost.

        The log names the candidate that ended the run rather than claiming
        every one failed: an unclassified failure is not fallback-worthy, so
        it ends the run wherever it happens, and candidates behind it are
        left untried.

        Args:
            exc: The failure about to be re-raised.
            failed: Attempts that failed earlier, in order.
            candidate: The provider whose failure ends the run.
        """
        if not failed:
            return
        trail = (*failed, ProviderAttempt.from_exception(candidate, exc))
        if isinstance(exc, ProviderError):
            exc.provider_attempts = trail
            return
        logger.warning(
            "Provider %s failed with %s, which cannot carry the trail. "
            "The run ends here; any candidate after it was not tried. "
            "Providers tried:\n%s",
            candidate,
            type(exc).__name__,
            ProviderAttempt.render_trail(trail),
        )

    @staticmethod
    def _record_provenance(
        result: ResearchResult,
        requested: Optional[str],
        failed: list[ProviderAttempt],
        used: str,
    ) -> None:
        """Stamp a result with who was asked, who was tried, and who answered.

        Called after the result has been cached, never before: these fields
        describe *this* run, and replaying them out of a cache file would
        credit a later run with a fallback that never happened.

        Announcing the fallback is part of stamping it rather than a separate
        step at each return, so a path that returns a result cannot forget to
        mention that someone else produced it -- which the cache-hit path did.

        Args:
            result: The result to stamp.
            requested: Provider the caller named, if any.
            failed: Attempts that failed, in the order they were tried.
            used: Provider that produced the result.
        """
        result.requested_provider = requested
        result.provider_attempts = [
            *failed,
            ProviderAttempt(provider=used, succeeded=True),
        ]
        if result.fell_back:
            logger.warning(
                "Report produced by %s, not the provider first tried (%s)",
                used, failed[0].provider,
            )

    def research(
        self,
        query: str,
        provider: Optional[str] = None,
        template_info: Optional[dict] = None,
        model: Optional[str] = None,
        provider_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
        fallback: Optional[Union[bool, str, Sequence[str]]] = False,
    ) -> ResearchResult:
        """Perform research on the given query.

        Args:
            query: The research question or topic
            provider: Specific provider to use (uses first available if None)
            template_info: Template information if query was generated from template
            model: Model to use for the provider (overrides provider default)
            provider_params: Provider-specific parameters
            metadata: Publication-style metadata (title, abstract, keywords, author, contributors)
            fallback: Opt in to trying another provider when this one cannot do
                the work. False (the default) or None never switches. True
                falls back to whatever else is available, in registration
                order, excluding providers that do not do real research. A
                list -- or a bare string, for one -- names them explicitly, in
                preference order, and replaces the automatic ordering.

        Returns:
            ResearchResult with markdown content and citations

        Raises:
            ProviderNotConfiguredError: If a known provider is not set up, or
                the requested provider has no credential configured
            ValueError: If no providers are available, or the name is not a
                provider at all
        """
        return asyncio.run(
            self.aresearch(
                query,
                provider,
                template_info,
                model,
                provider_params,
                metadata,
                fallback,
            )
        )

    async def aresearch(
        self,
        query: str,
        provider: Optional[str] = None,
        template_info: Optional[dict] = None,
        model: Optional[str] = None,
        provider_params: Optional[dict] = None,
        metadata: Optional[dict] = None,
        fallback: Optional[Union[bool, str, Sequence[str]]] = False,
    ) -> ResearchResult:
        """Async version of research method.

        Falling back is opt-in and never silent: whoever produced the report is
        recorded on it, along with everyone who was tried first and why they
        could not. Only failures that mean "this provider cannot do the work"
        are followed -- a throttle or a 5xx says to wait, not to switch, so
        those propagate as they always did.

        Nothing is cached or returned until a provider actually succeeds, so a
        run that ends without one raises rather than producing a partial
        result -- whether it exhausted the candidates or stopped at a failure
        that did not justify trying the next.
        """
        start_time = datetime.now()
        start_timestamp = time.time()

        candidates = self._fallback_candidates(provider, fallback)
        failed: list[ProviderAttempt] = []

        async def serve_cached(
            candidate: str,
            cached_model: Optional[str],
            cache_params: Optional[dict],
            unreachable: Optional[BaseException] = None,
        ) -> Optional[ResearchResult]:
            """Return this candidate's cached report, stamped for this run.

            Closes over what does not vary across candidates, so the two places
            that may serve a cached result cannot drift on how they stamp one --
            the same reason the fallback warning lives inside _record_provenance
            rather than at each return.

            Args:
                candidate: Provider whose cache entry is wanted.
                cached_model: Model to key the entry by, if any.
                cache_params: Provider parameters to key the entry by.
                unreachable: The failure that stopped this provider running, when
                    the entry is being served in place of a live call. Announced
                    before the result is stamped, so the reason reaches the log
                    ahead of the outcome it caused.

            Returns:
                The stamped result, or None when nothing is cached.
            """
            cached = await self.cache.get(query, candidate, cached_model, cache_params)
            if cached is None:
                return None
            if unreachable is not None:
                # Say it even though the run succeeds: the next uncached query
                # will not, and nothing else records why. No attempt is
                # appended, because the cached report really was produced by
                # this provider -- so this line is the only thing standing
                # between a revoked credential and silence.
                logger.warning(
                    "Provider %s cannot do this run: %s. "
                    "Serving the report it produced for this query earlier",
                    candidate, unreachable,
                )
            cached.start_time = start_time
            cached.end_time = datetime.now()
            cached.duration_seconds = time.time() - start_timestamp
            self._record_provenance(cached, provider, failed, candidate)
            return cached

        for position, candidate in enumerate(candidates):
            # The last candidate has nowhere to fall back to, so its failure is
            # the run's failure and propagates untouched -- same type, same
            # traceback a caller would have seen with no fallback at all.
            last = position == len(candidates) - 1

            # A model name and provider parameters were chosen for the provider
            # the caller named; they mean nothing to a different one, and an
            # unknown parameter is a hard schema error. So they apply to the
            # first candidate only, and a fallback runs on its own defaults --
            # said out loud below, because it is a change to what was asked.
            first = position == 0
            effective_model = model if first else None
            effective_params = provider_params if first else None

            cache_provider_params = self._get_cache_provider_params(
                candidate, effective_params
            )

            try:
                research_provider = self._prepare_provider(
                    candidate, effective_model, effective_params
                )
            except Exception as exc:
                if last or not is_fallback_worthy(exc):
                    # Carry the trail out on the exception, then re-raise bare:
                    # same object, same type, same traceback, and __cause__
                    # left to whoever set it.
                    self._attach_trail(exc, failed, candidate)
                    raise
                # Reading a report off disk needs no credential, so a
                # provider we can no longer reach can still serve one it
                # produced earlier, rather than the next candidate being
                # called for a report we already hold.
                #
                # The predicate is "another candidate remains", not "someone
                # would otherwise be billed" -- the two come apart, since a
                # remaining candidate may be a permanently unavailable stub.
                # It is deliberately the narrower one: on the last candidate
                # the run fails exactly as it did before any of this, which is
                # what keeps the default, no-fallback path byte-identical.
                served = await serve_cached(
                    candidate, effective_model, cache_provider_params, exc
                )
                if served is not None:
                    return served
                failed.append(ProviderAttempt.from_exception(candidate, exc))
                self._warn_falling_back(
                    candidate, candidates[position + 1], exc, model,
                    provider_params, candidates[0],
                )
                continue

            # The ordinary read, for a provider we were able to prepare.
            served = await serve_cached(
                candidate, effective_model, cache_provider_params
            )
            if served is not None:
                return served

            # Perform research
            try:
                result = await research_provider.research(query)
            except Exception as exc:
                if last or not is_fallback_worthy(exc):
                    # Carry the trail out on the exception, then re-raise bare:
                    # same object, same type, same traceback, and __cause__
                    # left to whoever set it.
                    self._attach_trail(exc, failed, candidate)
                    raise
                failed.append(ProviderAttempt.from_exception(candidate, exc))
                self._warn_falling_back(
                    candidate, candidates[position + 1], exc, model,
                    provider_params, candidates[0],
                )
                continue

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
            await self.cache.set(
                query, candidate, result, effective_model, cache_provider_params
            )

            # After caching, so that which providers this run tried never
            # becomes part of what a later run reads back.
            self._record_provenance(result, provider, failed, candidate)
            return result

        # Unreachable: _fallback_candidates never returns an empty list, and the
        # last candidate re-raises rather than falling out of the loop.
        raise ValueError("No research providers available")

    def knows_provider(self, name: str) -> bool:
        """Report whether a name is a provider at all, configured or not.

        The distinction this draws is the one that decides which error a caller
        gets: a known provider with no credential is *not configured* and can
        be fallen back from, while an unrecognised name is a typo and the
        remedy is a list of the names that exist.

        Every site that has to tell those apart asks this -- the fallback list,
        the CLI's pre-check, and the resolver that picks between the two error
        types. Spelling it out instead is how it came to be missing from one of
        them.

        Configuration is a different question, and deliberately not this one:
        ``falcon`` is a provider whether or not a key is set for it, while a
        transposed letter is not.

        >>> client = DeepResearchClient(
        ...     cache_config=CacheConfig(enabled=False),
        ...     provider_configs={"mock": ProviderConfig(name="mock")},
        ... )
        >>> client.knows_provider("falcon"), client.knows_provider("falcn")
        (True, False)

        The registry is consulted as well as the known names, so a provider put
        there directly under a name no environment variable produces still
        counts:

        >>> from deep_research_client.providers.mock import MockProvider
        >>> client.registry.register(MockProvider(ProviderConfig(name="spare")))
        >>> client.knows_provider("spare")
        True

        Args:
            name: Provider name to check.

        Returns:
            True when the name is one this client could resolve.
        """
        return name in PROVIDER_CLASS_PATHS or self.registry.get_provider(name) is not None

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
