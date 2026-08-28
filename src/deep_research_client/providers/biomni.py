"""Biomni biomedical co-scientist provider.

`Biomni <https://github.com/snap-stanford/Biomni>`_ is a general-purpose
biomedical AI agent from Stanford SNAP. It wraps a large toolbox of biomedical
software and curated databases and executes generated code to plan and carry
out research tasks. Unlike a classic "deep research" provider that searches the
literature and writes a report, Biomni is a *co-scientist*: it forms and tests
hypotheses, runs code against biomedical data, and can design experiments. A
conventional literature-synthesis run is essentially a subset of what it does.

This provider wraps the local ``biomni`` Python package (``biomni.agent.A1``).
The ``biomni`` extra supplies its core Python runtime, but Biomni's scientific
toolbox still requires one of the upstream conda environments. Because Biomni
executes generated code locally and downloads a large (~11GB) data lake, run it
only in a trusted/sandboxed environment.
"""

import asyncio
from datetime import datetime
import importlib.util
import logging
import os
from typing import Any, List, Optional, Sequence

from . import ResearchProvider
from ..exceptions import ProviderNotInstalledError, classify_exception
from ..models import ProviderConfig, ResearchResult
from ..provider_params import BiomniParams
from ..model_cards import ProviderModelCards, create_biomni_model_cards
from ..validation.extraction import find_reference_ids

logger = logging.getLogger(__name__)

# Biomni-generated analyses can be long-running. A1 applies this value to each
# generated code execution; it is not a deadline for the whole agent run.
BIOMNI_DEFAULT_TIMEOUT = 3600
DEFAULT_DATA_PATH = "./biomni_data"

# Biomni 0.0.8's wheel declares only pydantic/langchain/python-dotenv, while
# A1 imports pandas and langchain-openai eagerly and its default Claude path
# imports langchain-anthropic during construction. Keep the availability gate
# aligned with the dependencies supplied by our optional extra: checking only
# for ``biomni`` registers a provider that cannot import, and a failed automatic
# fallback then surfaces as an unclassified ValueError.
BIOMNI_EAGER_RUNTIME_MODULES = (
    "biomni",
    "pandas",
    "langchain_openai",
)
BIOMNI_SOURCE_MODULES = {
    "Anthropic": "langchain_anthropic",
    "Ollama": "langchain_ollama",
    "Bedrock": "langchain_aws",
}
# These adapters are intentionally not in the Biomni extra: they are needed
# only when a caller selects the corresponding non-default backend. Keys above
# match Biomni's case-sensitive SourceType. OpenAI, AzureOpenAI, Gemini, Groq,
# and Custom all use langchain_openai, which A1 imports eagerly and the extra
# already supplies.
BIOMNI_EXTERNAL_SOURCE_PACKAGES = {
    "langchain_ollama": "langchain-ollama",
    "langchain_aws": "langchain-aws",
}
BIOMNI_EXTRA_RUNTIME_MODULES = frozenset(
    (*BIOMNI_EAGER_RUNTIME_MODULES, BIOMNI_SOURCE_MODULES["Anthropic"])
)


def missing_biomni_runtime_modules(source: Optional[str] = None) -> list[str]:
    """Return core Python modules required by the default Biomni A1 path.

    ``langchain_openai`` is eager even for Anthropic because Biomni imports its
    tool retriever at module load. A source-specific adapter is added where the
    selected backend needs one. ``None`` means Biomni's default, Anthropic.

    This deliberately covers agent construction, not every optional tool in
    Biomni's external conda environment.

    Args:
        source: Biomni LLM source, or None for its default Anthropic path.

    Returns:
        Import-module names that are not installed
    """
    modules = list(BIOMNI_EAGER_RUNTIME_MODULES)
    source_module = BIOMNI_SOURCE_MODULES.get(source or "Anthropic")
    if source_module:
        modules.append(source_module)
    return [
        module
        for module in modules
        if importlib.util.find_spec(module) is None
    ]


def biomni_runtime_unavailable_reason(
    source: Optional[str], missing: Sequence[str]
) -> str:
    """Explain how to supply missing Biomni runtime modules.

    The default Anthropic adapter is part of ``deep-research-client[biomni]``;
    Ollama and Bedrock adapters are deliberately separate installs. Keeping
    that distinction here prevents the provider from recommending a reinstall
    that cannot supply the selected backend.

    Args:
        source: Biomni LLM source, or None for its default Anthropic path.
        missing: Import-module names that could not be found.

    Returns:
        Human-readable diagnosis and installation remedy.

    >>> biomni_runtime_unavailable_reason("Ollama", ["langchain_ollama"])
    "the Biomni Ollama backend requires Python module 'langchain_ollama' (install it with `pip install langchain-ollama` inside the upstream Biomni environment)"
    """
    missing_modules = list(dict.fromkeys(module for module in missing if module))
    if not missing_modules:
        return (
            "the Biomni runtime reported a missing Python module without naming it "
            "(inspect the Biomni logs and verify the upstream environment)"
        )

    extra_modules: list[str] = []
    external_modules: list[str] = []
    environment_modules: list[str] = []
    for module in missing_modules:
        if module in BIOMNI_EXTRA_RUNTIME_MODULES:
            extra_modules.append(module)
        elif module in BIOMNI_EXTERNAL_SOURCE_PACKAGES:
            external_modules.append(module)
        else:
            environment_modules.append(module)

    reasons: list[str] = []
    if "biomni" in extra_modules:
        reasons.append(
            "the biomni package is not installed "
            "(install the upstream Biomni environment, then "
            "pip install deep-research-client[biomni])"
        )
    elif extra_modules:
        reasons.append(
            "the Biomni Python runtime is incomplete; missing "
            f"{', '.join(extra_modules)} (reinstall deep-research-client[biomni] "
            "inside the upstream Biomni environment)"
        )
    backend = source or "selected"
    reasons.extend(
        f"the Biomni {backend} backend requires Python module '{module}' "
        f"(install it with `pip install {BIOMNI_EXTERNAL_SOURCE_PACKAGES[module]}` "
        "inside the upstream Biomni environment)"
        for module in external_modules
    )
    if environment_modules:
        modules = ", ".join(environment_modules)
        pronoun = "it" if len(environment_modules) == 1 else "them"
        reasons.append(
            f"the Biomni runtime is missing {modules} "
            f"(install {pronoun} inside the upstream Biomni environment)"
        )
    return "; ".join(reasons)


class BiomniProvider(ResearchProvider):
    """Provider that runs the local Biomni ``A1`` biomedical agent.

    Requires the optional ``biomni`` package. The underlying LLM (Claude by
    default) is configured through Biomni itself and authenticated via the
    usual provider environment variables (e.g. ``ANTHROPIC_API_KEY``).
    """

    def __init__(self, config: ProviderConfig, params: Optional[BiomniParams] = None):
        """Initialize the Biomni provider.

        Args:
            config: Provider configuration.
            params: Biomni-specific parameters.
        """
        self.params = params or BiomniParams()
        super().__init__(config, self.params.model)

        self.data_path = (
            self.params.path
            or os.getenv("BIOMNI_DATA_PATH")
            or DEFAULT_DATA_PATH
        )

    def get_default_model(self) -> str:
        """Return the default research model card identifier."""
        return "biomni-a1"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for the Biomni provider."""
        return create_biomni_model_cards()

    def unavailable_reason(self) -> str:
        """Name the missing runtime package, since no credential fixes it.

        Biomni needs no API key of its own -- it drives whatever LLM it is
        configured with -- so the base class's "no API key configured" would
        send the reader looking for a variable that does not exist.

        Returns:
            Human-readable explanation suitable for an error message
        """
        if not self.config.enabled:
            return super().unavailable_reason()

        missing = missing_biomni_runtime_modules(self.params.source)
        if missing:
            return biomni_runtime_unavailable_reason(self.params.source, missing)

        return super().unavailable_reason()

    def is_available(self) -> bool:
        """Check whether the core Python runtime for default A1 is importable."""
        if not self.config.enabled:
            return False
        missing = missing_biomni_runtime_modules(self.params.source)
        if missing:
            # Debug, not warning: availability is polled by provider-listing
            # paths and on every run, and unavailable_reason() already carries
            # the user-facing wording for the one place that needs it.
            logger.debug("Biomni runtime incomplete; missing: %s", ", ".join(missing))
            return False
        return True

    async def research(self, query: str) -> ResearchResult:
        """Run a Biomni agent task and return the result as a research report.

        Args:
            query: The biomedical task or research question.

        Returns:
            ResearchResult with the agent's final answer as markdown and any
            PMID/DOI citations extracted from it.
        """
        # Validate the query first so a caller error surfaces regardless of
        # whether the optional package happens to be installed.
        if not query or not query.strip():
            raise ValueError("Research query must not be empty.")
        if not self.is_available():
            raise ProviderNotInstalledError(self.name, self.unavailable_reason())

        start_time = datetime.now()
        logger.info("Starting Biomni agent run (data path: %s)", self.data_path)
        logger.debug("Query: %s%s", query[:100], "..." if len(query) > 100 else "")

        try:
            raw = await asyncio.to_thread(self._run_agent, query)
        except ModuleNotFoundError as e:
            # The core imports are checked by is_available(), but Biomni's
            # generated code and scientific tools can reach deeper optional
            # modules from its external environment. Preserve missing modules
            # as typed configuration failures so --fallback can move on. A plain
            # ImportError may instead mean an installed-but-incompatible package;
            # the generic branch below preserves that distinction and its detail.
            missing = e.name or str(e)
            raise ProviderNotInstalledError(
                self.name,
                biomni_runtime_unavailable_reason(self.params.source, [missing]),
            ) from e
        except Exception as e:  # noqa: BLE001 - surface a clean provider error
            logger.error("Biomni agent run failed: %s", e)
            logger.debug("Error details:", exc_info=True)
            # Biomni drives an LLM of its own, so an auth or quota failure from
            # that model arrives here; classify it rather than flattening every
            # cause into one message.
            classified = classify_exception(self.name, e)
            if classified is not None:
                raise classified from e
            raise ValueError(f"Biomni agent error: {e}") from e

        markdown = self._result_to_markdown(raw)
        if not markdown.strip():
            # A run that produced nothing may still have spent an hour of LLM
            # time and local compute; returning it as an ordinary success gives
            # the caller no way to tell that apart from a real answer.
            logger.warning(
                "Biomni run produced an empty report. The agent finished without "
                "returning text -- check the biomni logs for the underlying failure."
            )
        citations = self._extract_citations(markdown)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            "Biomni run complete in %.1fs: %d chars, %d citations",
            duration, len(markdown), len(citations),
        )

        return ResearchResult(
            markdown=markdown,
            citations=citations,
            provider=self.name,
            query=query,
            model=self.model,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )

    def _agent_kwargs(self) -> dict[str, Any]:
        """Build the keyword arguments for a Biomni ``A1`` agent.

        Split out from :meth:`_build_agent` so the precedence rules below can be
        tested in the ordinary base-install CI jobs. Dedicated clean-extra jobs
        separately exercise the real import, signature, and construction path.

        Returns:
            Keyword arguments to construct ``A1`` with
        """
        kwargs: dict[str, Any] = {
            "path": self.data_path,
            "use_tool_retriever": self.params.use_tool_retriever,
        }
        if self.params.llm:
            kwargs["llm"] = self.params.llm
        if self.params.source:
            kwargs["source"] = self.params.source
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        # A1's parameter is `timeout_seconds` (not `timeout`). Let an explicit
        # BiomniParams.timeout win over the config default, then the module
        # default.
        timeout = self.params.timeout or self.config.timeout or BIOMNI_DEFAULT_TIMEOUT
        kwargs["timeout_seconds"] = timeout
        if self.params.skip_data_lake:
            # An empty expected-files list tells A1 not to load/download the lake.
            kwargs["expected_data_lake_files"] = []

        return kwargs

    def _build_agent(self) -> Any:
        """Construct a Biomni ``A1`` agent from the configured parameters."""
        from biomni.agent import A1  # type: ignore[import-not-found, import-untyped]

        kwargs = self._agent_kwargs()
        logger.debug("Constructing Biomni A1 with: %s", sorted(kwargs))
        return A1(**kwargs)

    def _run_agent(self, query: str) -> Any:
        """Run the Biomni agent synchronously and return its raw output.

        A fresh agent per call, deliberately: ``research()`` hands this to
        ``asyncio.to_thread``, so a cached agent would be shared across
        concurrent runs, and ``A1`` carries per-run conversation state that is
        not documented as thread-safe. The cost is real -- construction sets up
        the tool retriever and validates the data lake -- so a caller issuing
        many queries pays it each time; run isolation is judged the better
        trade until upstream states otherwise.
        """
        agent = self._build_agent()
        return agent.go(query)

    @staticmethod
    def _result_to_markdown(result: Any) -> str:
        """Normalize Biomni's ``go()`` return value into markdown text.

        ``A1.go`` may return the final answer string, or a ``(log, output)``
        pair, or a list of messages. This coerces those shapes into a single
        markdown string, preferring the final answer.
        """
        if isinstance(result, str):
            return result
        if isinstance(result, (tuple, list)):
            if len(result) == 2 and isinstance(result[1], str):
                return result[1]
            parts = [p for p in result if isinstance(p, str)]
            if parts:
                return parts[-1]
            if result:
                logger.warning(
                    "Biomni returned a %s of non-string parts; stringifying the last "
                    "one as the report. This usually means A1.go's return shape "
                    "changed and the parsing here needs updating.",
                    type(result).__name__,
                )
                return str(result[-1])
            # Debug, not warning: research() warns once for every empty shape,
            # so warning here too would put two lines in the log for one run.
            logger.debug("Biomni returned an empty collection; no report text.")
            return ""
        for attr in ("content", "output", "final_answer", "answer"):
            value = getattr(result, attr, None)
            if isinstance(value, str):
                return value
        logger.warning(
            "Biomni returned an unrecognised %s with no text attribute; stringifying "
            "it as the report. This usually means A1.go's return shape changed and "
            "the parsing here needs updating.",
            type(result).__name__,
        )
        return str(result)

    @staticmethod
    def _extract_citations(markdown: str) -> List[str]:
        """Extract reference identifiers from the report text.

        Delegates to :func:`deep_research_client.validation.find_reference_ids`,
        which owns the identifier patterns shared with reference validation, so
        a Biomni report is read exactly as every other report is. Those patterns
        already handle what a hand-rolled pair here missed: a DOI written as a
        markdown link, a bare PubMed URL, and the PMC/GEO accessions a
        biomedical agent emits routinely.

        >>> BiomniProvider._extract_citations("See PMID: 12345678 and doi:10.1038/nature12373")
        ['PMID:12345678', 'DOI:10.1038/nature12373']
        >>> BiomniProvider._extract_citations("PMID:11111111 again PMID: 11111111")
        ['PMID:11111111']

        A DOI inside a markdown link is captured as the identifier alone, not
        as the identifier plus the link's trailing furniture:

        >>> BiomniProvider._extract_citations(
        ...     "[doi:10.1038/nature12373](https://doi.org/10.1038/nature12373)"
        ... )
        ['DOI:10.1038/nature12373']

        A 1976 paper's six-digit PMID is a real citation, not a truncation:

        >>> BiomniProvider._extract_citations("Shown decades ago (PMID: 942051).")
        ['PMID:942051']
        """
        return [found.normalized_id for found in find_reference_ids(markdown)]
