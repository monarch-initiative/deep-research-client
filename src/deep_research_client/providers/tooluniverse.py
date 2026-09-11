"""Local ToolUniverse co-scientist using a smolagents tool adapter.

ToolUniverse supplies scientific tools, not a standalone research-question API.
This provider composes those tools with a smolagents CodeAgent, which plans,
executes Python, investigates evidence, and returns an inline markdown report.
Install the optional ``tooluniverse`` extra and use a trusted environment for
the agent's local code execution. The base install never imports either SDK.
"""

import asyncio
from datetime import datetime
import importlib.util
from typing import Any

from . import ResearchProvider
from ..exceptions import (
    ProviderNotConfiguredError,
    ProviderNotInstalledError,
    classify_exception,
)
from ..model_cards import ProviderModelCards, create_tooluniverse_model_cards
from ..models import ProviderConfig, ResearchResult
from ..provider_params import ToolUniverseParams
from ..toolsets.tooluniverse import ToolUniverseToolset
from ..validation.extraction import find_reference_ids

DEFAULT_REQUEST_TIMEOUT = 120
RESEARCH_INSTRUCTIONS = """Act as a scientific co-investigator. Develop hypotheses,
use the available scientific tools and Python to investigate them, and separate
observations from speculation. Ground factual claims in retrieved evidence.
Return the complete final report as a markdown string via final_answer, including
methods, findings, limitations, and references with actual PMID, DOI, or source
URLs. Do not invent citations or claim an experiment was performed when it was
only proposed. Do not return a filename or a promise to write a report later."""


def missing_tooluniverse_runtime_modules() -> list[str]:
    """Return missing optional modules without importing their heavy SDKs."""
    return [
        name for name in ("tooluniverse", "smolagents")
        if importlib.util.find_spec(name) is None
    ]


class ToolUniverseProvider(ResearchProvider):
    """Run a fresh local scientific agent for each research question.

    ``ProviderConfig.api_key`` authenticates the underlying LLM, and
    ``base_url`` selects an OpenAI-compatible endpoint. Auto-detection supplies
    these from TOOLUNIVERSE_API_KEY (falling back to OPENAI_API_KEY) and
    TOOLUNIVERSE_BASE_URL. Individual scientific tools may need separate keys.
    """

    credential_env_var = "TOOLUNIVERSE_API_KEY"
    credential_label = "ToolUniverse underlying LLM"

    def __init__(
        self, config: ProviderConfig, params: ToolUniverseParams | None = None,
    ) -> None:
        """Initialize configuration without constructing an agent or making requests."""
        self.params = params or ToolUniverseParams()
        super().__init__(config, self.params.model)

    def get_default_model(self) -> str:
        """Return the research model card identity, distinct from the LLM ID."""
        return "tooluniverse-coscientist"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Return the ToolUniverse co-scientist model card."""
        return create_tooluniverse_model_cards()

    def is_available(self) -> bool:
        """Check credentials and optional runtime presence without network access."""
        return bool(
            self.config.enabled and self.config.api_key
            and not missing_tooluniverse_runtime_modules()
        )

    def unavailable_reason(self) -> str:
        """Explain disabled configuration, missing credentials, or missing SDKs."""
        if not self.config.enabled:
            return super().unavailable_reason()
        if not self.config.api_key:
            return (
                "no underlying LLM API key configured (set TOOLUNIVERSE_API_KEY "
                "or OPENAI_API_KEY, or pass ProviderConfig.api_key)"
            )
        missing = missing_tooluniverse_runtime_modules()
        if missing:
            return (
                f"missing Python modules: {', '.join(missing)}; "
                "install with `uv add 'deep-research-client[tooluniverse]'`"
            )
        return super().unavailable_reason()

    def _model_kwargs(self) -> dict[str, Any]:
        """Build LLM configuration with an explicit per-request timeout."""
        return {
            "model_id": self.params.llm,
            "api_key": self.config.api_key,
            "api_base": self.config.base_url,
            "client_kwargs": {
                "timeout": self.params.timeout or self.config.timeout or DEFAULT_REQUEST_TIMEOUT,
            },
        }

    def _build_agent(self, universe: Any) -> Any:
        """Load selected tools and construct the real upstream CodeAgent.

        Unknown or unavailable tool names fail before an LLM request. Exposing
        only the requested tools keeps the model context bounded and prevents
        silently substituting an agent with no scientific tools.
        """
        from smolagents import CodeAgent, OpenAIModel  # type: ignore[import-not-found, import-untyped]
        from ._tooluniverse_tools import create_tooluniverse_tool

        try:
            ToolUniverseToolset(tools=self.params.tools).load_tools(universe)
        except ValueError as error:
            raise ProviderNotConfiguredError(self.name, str(error)) from error
        tools = [create_tooluniverse_tool(name, universe) for name in self.params.tools]
        return CodeAgent(
            tools=tools,
            model=OpenAIModel(**self._model_kwargs()),
            instructions=self.params.system_prompt or RESEARCH_INSTRUCTIONS,
            max_steps=self.params.max_steps,
            verbosity_level=0,
            return_full_result=True,
        )

    def _run_agent(self, query: str) -> str:
        """Run synchronously with independent conversation and tool state per call."""
        from tooluniverse import ToolUniverse  # type: ignore[import-not-found, import-untyped]

        universe = ToolUniverse()
        try:
            agent = self._build_agent(universe)
            try:
                result = agent.run(query)
                return self._result_to_markdown(result)
            finally:
                agent.model.client.close()
        finally:
            universe.close()

    @staticmethod
    def _result_to_markdown(result: Any) -> str:
        """Accept only a successful RunResult containing non-empty report text.

        smolagents can synthesize a final answer after exhausting max_steps;
        its ``max_steps_error`` state must not be cached as successful research.
        """
        if result.state != "success":
            raise ValueError(f"ToolUniverse agent did not complete: {result.state}")
        if not isinstance(result.output, str) or not result.output.strip():
            raise ValueError("ToolUniverse agent returned no markdown report")
        return result.output

    async def research(self, query: str) -> ResearchResult:
        """Investigate a question and return markdown, citations, and run timing."""
        if not query or not query.strip():
            raise ValueError("Research query must not be empty.")
        if not self.config.enabled or not self.config.api_key:
            raise ProviderNotConfiguredError(self.name, self.unavailable_reason())
        if missing_tooluniverse_runtime_modules():
            raise ProviderNotInstalledError(self.name, self.unavailable_reason())

        start_time = datetime.now()
        try:
            markdown = await asyncio.to_thread(self._run_agent, query)
        except Exception as error:
            # Translate known backend failures for the client's fallback policy;
            # preserve unknown errors and their traceback rather than hiding them.
            classified = classify_exception(self.name, error)
            if classified is not None and classified is not error:
                raise classified from error
            raise
        end_time = datetime.now()
        return ResearchResult(
            markdown=markdown,
            citations=[reference.normalized_id for reference in find_reference_ids(markdown)],
            provider=self.name,
            query=query,
            model=self.model,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            run_metadata={
                "agent_runtime": "smolagents.CodeAgent",
                "llm": self.params.llm,
                "toolsets": [ToolUniverseToolset(tools=self.params.tools).provenance()],
            },
        )
