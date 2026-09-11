"""ToolUniverse tests using real configuration, SDK objects, and scientific tools.

Base-install tests need no SDK. Clean-extra tests construct the real agent and
exercise its Python executor without an LLM request. Integration tests contact
PubMed or an explicitly enabled LLM backend; no SDKs or services are mocked.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from deep_research_client.client import DeepResearchClient
from deep_research_client.exceptions import ProviderNotConfiguredError, ProviderNotInstalledError
from deep_research_client.model_cards import ProviderArchetype, ResearchCapability
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.provider_params import ToolUniverseParams, create_provider_params
from deep_research_client.providers.tooluniverse import (
    DEFAULT_REQUEST_TIMEOUT,
    ToolUniverseProvider,
    missing_tooluniverse_runtime_modules,
)


def make_provider(**params: Any) -> ToolUniverseProvider:
    """Construct a provider with a non-working key for offline configuration tests."""
    return ToolUniverseProvider(
        ProviderConfig(name="tooluniverse", api_key="offline-test-key"),
        ToolUniverseParams(**params),
    )


def require_runtime() -> None:
    """Skip SDK tests on base installs, but require them in the clean-extra job."""
    missing = missing_tooluniverse_runtime_modules()
    if missing:
        reason = f"ToolUniverse runtime required: {', '.join(missing)}"
        if os.getenv("REQUIRE_TOOLUNIVERSE_RUNTIME") == "1":
            pytest.fail(reason)
        pytest.skip(reason)


def test_model_card_and_params() -> None:
    """The co-scientist identity is distinct from its configurable LLM."""
    params = create_provider_params("tooluniverse", "tu", {"llm": "custom-model"})
    assert isinstance(params, ToolUniverseParams)
    provider = ToolUniverseProvider(ProviderConfig(name="tooluniverse"), params)
    assert provider.model == "tooluniverse-coscientist"
    assert provider.params.llm == "custom-model"
    card = provider.model_cards().get_model_card(provider.model)
    assert card is not None
    assert card.archetype == ProviderArchetype.co_scientist
    assert ResearchCapability.code_interpretation in card.capabilities
    assert ResearchCapability.hypothesis_generation in card.capabilities


@pytest.mark.parametrize("params", [
    {"tools": []}, {"tools": [""]}, {"tools": [" PubMed_get_article"]},
    {"tools": ["PubMed_get_article", "PubMed_get_article"]},
    {"max_steps": 0}, {"max_steps": 101}, {"timeout": 0},
    {"llm": ""}, {"unknown": True}, {"allowed_domains": ["example.org"]},
])
def test_invalid_parameters_fail_fast(params: dict[str, Any]) -> None:
    """Reject invalid requests before any tools or models are constructed."""
    with pytest.raises(ValidationError):
        ToolUniverseParams(**params)


@pytest.mark.parametrize("param_timeout,config_timeout,expected", [
    (None, None, DEFAULT_REQUEST_TIMEOUT), (15, 30, 15), (None, 30, 30),
])
def test_model_configuration(
    param_timeout: int | None, config_timeout: int | None, expected: int,
) -> None:
    """Explicit params outrank config, which outranks the request timeout default."""
    provider = ToolUniverseProvider(
        ProviderConfig(
            name="tooluniverse", api_key="offline-test-key",
            base_url="https://example.org/v1", timeout=config_timeout,
        ),
        ToolUniverseParams(llm="custom-model", timeout=param_timeout),
    )
    assert provider._model_kwargs() == {
        "model_id": "custom-model", "api_key": "offline-test-key",
        "api_base": "https://example.org/v1", "client_kwargs": {"timeout": expected},
    }


def test_availability_requires_key_and_runtime() -> None:
    """Import detection cannot make an unconfigured model available."""
    assert make_provider().is_available() == (not missing_tooluniverse_runtime_modules())
    assert not ToolUniverseProvider(ProviderConfig(name="tooluniverse")).is_available()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "  "])
async def test_empty_query_rejected(query: str) -> None:
    """Caller errors surface even without the optional extra."""
    with pytest.raises(ValueError, match="must not be empty"):
        await make_provider().research(query)


@pytest.mark.asyncio
async def test_disabled_provider() -> None:
    """Disabling a provider is reported independently of its installation."""
    provider = ToolUniverseProvider(ProviderConfig(name="tooluniverse", enabled=False))
    assert not provider.is_available()
    with pytest.raises(ProviderNotConfiguredError, match="disabled"):
        await provider.research("Investigate scurvy")


@pytest.mark.asyncio
async def test_missing_runtime() -> None:
    """A configured LLM cannot compensate for missing scientific agent packages."""
    if not missing_tooluniverse_runtime_modules():
        pytest.skip("Optional runtime is installed")
    with pytest.raises(ProviderNotInstalledError, match=r"deep-research-client\[tooluniverse\]"):
        await make_provider().research("Investigate scurvy")


def test_explicit_client_registration(tmp_path: Path) -> None:
    """Client factory and parameter dispatch know the new provider."""
    client = DeepResearchClient(
        cache_config=CacheConfig(directory=str(tmp_path)),
        provider_configs={"tooluniverse": ProviderConfig(name="tooluniverse")},
    )
    assert isinstance(client.registry.get_provider("tooluniverse"), ToolUniverseProvider)


def test_environment_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-detection uses the dedicated credential and supports opting out."""
    monkeypatch.setenv("TOOLUNIVERSE_API_KEY", "offline-test-key")
    monkeypatch.setenv("TOOLUNIVERSE_BASE_URL", "https://example.org/v1")
    monkeypatch.delenv("DISABLE_TOOLUNIVERSE_PROVIDER", raising=False)
    client = DeepResearchClient(cache_config=CacheConfig(directory=str(tmp_path)))
    provider = client.registry.get_provider("tooluniverse")
    if missing_tooluniverse_runtime_modules():
        assert provider is None
    else:
        assert provider is not None
        assert provider.config.api_key == "offline-test-key"
        assert provider.config.base_url == "https://example.org/v1"
    monkeypatch.setenv("DISABLE_TOOLUNIVERSE_PROVIDER", "true")
    client = DeepResearchClient(cache_config=CacheConfig(directory=str(tmp_path)))
    assert client.registry.get_provider("tooluniverse") is None


def test_build_agent_with_clean_extra(tmp_path: Path) -> None:
    """Construct actual default tools/model and run Python without contacting an LLM."""
    require_runtime()
    from tooluniverse import ToolUniverse  # type: ignore[import-not-found, import-untyped]

    provider = make_provider(system_prompt="Custom scientific instructions.", max_steps=3)
    universe = ToolUniverse(workspace=str(tmp_path))
    try:
        agent = provider._build_agent(universe)
        try:
            assert set(agent.tools) == {*provider.params.tools, "final_answer"}
            assert agent.max_steps == 3
            assert agent.return_full_result
            assert "Custom scientific instructions." in agent.system_prompt
            assert agent.model.model_id == provider.params.llm
            assert agent.model.client.timeout == DEFAULT_REQUEST_TIMEOUT
            # PubMed's union result schema must not break construction or be
            # rewritten in the ToolUniverse registry to placate the adapter.
            assert universe.all_tool_dict["PubMed_get_article"]["return_schema"]["type"] == [
                "object", "string",
            ]
            search_tool = agent.tools["PubMed_search_articles"]
            assert search_tool.inputs["query"]["type"] == "string"
            assert not search_tool.inputs["query"].get("nullable", False)
            assert search_tool.inputs["limit"]["nullable"] is True
            assert search_tool.inputs["limit"]["default"] == 10
            # Real local code execution is a core co-scientist capability.
            # CodeAgent.run sends tools before executing its first step.
            agent.python_executor.send_tools(agent.tools)
            output = agent.python_executor("sum([1, 2, 3])")
            assert output.output == 6
        finally:
            agent.model.client.close()
    finally:
        universe.close()


def test_unknown_tool_fails_before_llm_request(tmp_path: Path) -> None:
    """The upstream adapter otherwise silently creates an unusable empty tool."""
    require_runtime()
    from tooluniverse import ToolUniverse  # type: ignore[import-not-found, import-untyped]

    universe = ToolUniverse(workspace=str(tmp_path))
    try:
        with pytest.raises(ProviderNotConfiguredError, match="No_such_scientific_tool"):
            make_provider(tools=["No_such_scientific_tool"])._build_agent(universe)
    finally:
        universe.close()


@pytest.mark.parametrize("output,state,error", [
    ("# Findings\nPMID:942051", "success", None),
    ("An incomplete answer", "max_steps_error", "did not complete"),
    (None, "success", "no markdown report"),
    (" ", "success", "no markdown report"),
    ({"answer": "not markdown"}, "success", "no markdown report"),
])
def test_real_run_result_parsing(output: Any, state: str, error: str | None) -> None:
    """Use the actual upstream result class, including exhausted/empty runs."""
    require_runtime()
    from smolagents.agents import RunResult  # type: ignore[import-not-found, import-untyped]

    result = RunResult(output=output, state=state)
    if error:
        with pytest.raises(ValueError, match=error):
            ToolUniverseProvider._result_to_markdown(result)
    else:
        assert ToolUniverseProvider._result_to_markdown(result) == output


@pytest.mark.integration
@pytest.mark.parametrize("positional", [False, True])
def test_pubmed_tool_integration(tmp_path: Path, positional: bool) -> None:
    """Retrieve a real PubMed record through the scientific tool adapter."""
    require_runtime()
    from tooluniverse import ToolUniverse  # type: ignore[import-not-found, import-untyped]

    universe = ToolUniverse(workspace=str(tmp_path))
    try:
        agent = make_provider()._build_agent(universe)
        try:
            tool = agent.tools["PubMed_get_article"]
            result = tool("942051") if positional else tool(pmid="942051")
            assert "942051" in str(result)
            assert "title" in str(result)
            assert "Error executing tool" not in str(result)
        finally:
            agent.model.client.close()
    finally:
        universe.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_integration() -> None:
    """Run a real paid scientific investigation only when explicitly enabled."""
    require_runtime()
    if os.getenv("RUN_TOOLUNIVERSE_INTEGRATION") != "1":
        pytest.skip("Set RUN_TOOLUNIVERSE_INTEGRATION=1 to enable a paid LLM run")
    key = os.getenv("TOOLUNIVERSE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.fail("A ToolUniverse underlying LLM API key is required")
    provider = ToolUniverseProvider(
        ProviderConfig(
            name="tooluniverse", api_key=key, base_url=os.getenv("TOOLUNIVERSE_BASE_URL"),
        ),
        ToolUniverseParams(max_steps=5),
    )
    result = await provider.research(
        "Retrieve PubMed article PMID:942051 with PubMed_get_article. "
        "Write a brief markdown report of its title and findings, citing PMID:942051."
    )
    assert len(result.markdown) > 100
    assert "PMID:942051" in result.citations
    assert result.provider == "tooluniverse"
    assert result.model == "tooluniverse-coscientist"
    assert result.duration_seconds is not None and result.duration_seconds > 0
