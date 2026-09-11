"""ToolUniverse composition tests using real tool schemas, MCP sessions, and agents.

No mock servers or agent doubles: offline tests execute ToolUniverse's local
usage-tips tool. Paid host-agent tests require an explicit integration flag.
"""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from deep_research_client.cache import CacheManager
from deep_research_client.client import DeepResearchClient
from deep_research_client.exceptions import ProviderNotInstalledError
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.provider_params import (
    BiomniParams, ClaudeCodeParams, CyberianParams, ToolUniverseParams, create_provider_params,
)
from deep_research_client.providers.claude_code import ClaudeCodeProvider
from deep_research_client.providers.cyberian import CyberianProvider
from deep_research_client.toolsets.tooluniverse import ToolUniverseToolset

LOCAL_TOOL = "ToolUniverse_get_usage_tips"


def require_tooluniverse() -> None:
    """Skip SDK tests on the base install, require them in toolset-extra CI."""
    if importlib.util.find_spec("tooluniverse") is None:
        if os.getenv("REQUIRE_TOOLUNIVERSE_RUNTIME") == "1":
            pytest.fail("ToolUniverse SDK is missing")
        pytest.skip("ToolUniverse SDK is not installed")


@pytest.mark.parametrize("provider", ["claude_code", "biomni", "cyberian"])
@pytest.mark.parametrize("cli_strings", [False, True])
def test_hosts_share_configuration(provider: str, cli_strings: bool) -> None:
    """Each local host accepts the same shorthand and explicit selection."""
    selection = {"tools": [LOCAL_TOOL]}
    default = create_provider_params(provider, provider_params={"tooluniverse": "true" if cli_strings else True})
    explicit = create_provider_params(provider, provider_params={"tooluniverse": json.dumps(selection) if cli_strings else selection})
    disabled = create_provider_params(provider, provider_params={"tooluniverse": "false" if cli_strings else False})
    assert default.model_dump()["tooluniverse"]["tools"] == ToolUniverseParams().tools
    assert explicit.model_dump()["tooluniverse"] == {"tools": [LOCAL_TOOL]}
    assert disabled.model_dump()["tooluniverse"] is None


@pytest.mark.parametrize("provider", ["openscientist", "openai", "falcon", "perplexity", "asta"])
def test_unsupported_hosts_reject_mixin(provider: str) -> None:
    """Remote/retrieval APIs must not silently ignore the requested composition."""
    with pytest.raises(ValueError, match="tooluniverse"):
        create_provider_params(provider, provider_params={"tooluniverse": True})


@pytest.mark.parametrize("tools", [[], [""], ["PubMed_*"], ["PubMed_get_article", "PubMed_get_article"]])
def test_invalid_selection(tools: list[str]) -> None:
    """Reject ambiguous selections for both mixins and the standalone agent."""
    with pytest.raises(ValidationError):
        ToolUniverseToolset(tools=tools)
    with pytest.raises(ValidationError):
        ToolUniverseParams(tools=tools)


def test_claude_command_preserves_permissions_and_host_model() -> None:
    """Adding scientific tools grants only the requested tools, without changing the LLM."""
    params = ClaudeCodeParams.model_validate({"tooluniverse": {"tools": [LOCAL_TOOL]}, "model": "sonnet"})
    provider = ClaudeCodeProvider(ProviderConfig(name="claude_code"), params)
    command = provider._build_command()
    config = json.loads(command[command.index("--mcp-config") + 1])
    server = config["mcpServers"]["tu"]
    assert server["command"] == sys.executable
    assert json.loads(server["args"][-1]) == [LOCAL_TOOL]
    assert command[command.index("--model") + 1] == "sonnet"
    assert command[command.index("--allowedTools") + 1].split(",") == [
        "WebSearch", "WebFetch", f"mcp__tu__{LOCAL_TOOL}",
    ]
    assert "--dangerously-skip-permissions" not in command
    assert params.allowed_tools == ["WebSearch", "WebFetch"]


@pytest.mark.parametrize("params", [{"agent_type": "codex"}, {"manage_server": False}])
def test_cyberian_rejects_hosts_it_cannot_configure(params: dict[str, Any]) -> None:
    """A pre-existing server or unsupported agent cannot inherit new workspace MCP settings."""
    with pytest.raises(ValidationError, match="requires agent_type='claude' and manage_server=true"):
        CyberianParams.model_validate({"tooluniverse": True, **params})


def test_cyberian_workspace_configuration(tmp_path: Path) -> None:
    """Configure local MCP discovery and permissions without altering workflow instructions."""
    toolset = ToolUniverseToolset(tools=[LOCAL_TOOL])
    provider = CyberianProvider(
        ProviderConfig(name="cyberian"), CyberianParams(tooluniverse=toolset),
    )
    provider._prepare_tooluniverse_workdir(str(tmp_path))
    assert json.loads((tmp_path / ".mcp.json").read_text()) == toolset.claude_mcp_config()
    assert json.loads((tmp_path / ".claude/settings.local.json").read_text()) == {
        "enabledMcpjsonServers": ["tu"], "permissions": {"allow": [f"mcp__tu__{LOCAL_TOOL}"]},
    }
    with pytest.raises(FileExistsError):
        provider._prepare_tooluniverse_workdir(str(tmp_path))


def test_biomni_configuration_keeps_credentials_out_of_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Biomni resolves references before passing its environment to the MCP child."""
    monkeypatch.setenv("NCBI_API_KEY", "secret-scientific-key")
    toolset = ToolUniverseToolset(tools=[LOCAL_TOOL])
    server = toolset.biomni_mcp_config("test_server")["mcp_servers"]["test_server"]
    assert server["command"][0] == sys.executable
    assert json.loads(server["command"][-1]) == [LOCAL_TOOL]
    assert server["env"]["NCBI_API_KEY"] == "${NCBI_API_KEY}"
    assert "secret-scientific-key" not in json.dumps(server)


def test_composition_cache_identity(tmp_path: Path) -> None:
    """Equivalent shorthand shares a cache entry; different toolsets and hosts do not."""
    params = DeepResearchClient._get_cache_provider_params
    shorthand = params("claude_code", {"tooluniverse": True})
    explicit = params("claude_code", {"tooluniverse": ToolUniverseToolset().model_dump()})
    assert shorthand == explicit
    cache = CacheManager(CacheConfig(directory=str(tmp_path)))
    mixed = cache._get_cache_filename("query", "claude_code", provider_params=shorthand)
    assert mixed != cache._get_cache_filename("query", "claude_code", provider_params=params("claude_code"))
    assert mixed != cache._get_cache_filename("query", "biomni", provider_params=shorthand)
    custom = params("claude_code", {"tooluniverse": {"tools": [LOCAL_TOOL]}})
    assert mixed != cache._get_cache_filename("query", "claude_code", provider_params=custom)


def test_missing_mixin_install_explains_toolset_extra() -> None:
    """The composed SDK does not require smolagents or its own LLM credential."""
    if importlib.util.find_spec("tooluniverse") is not None:
        pytest.skip("ToolUniverse is installed")
    with pytest.raises(ProviderNotInstalledError, match="tooluniverse-tools"):
        ToolUniverseToolset().prepare("claude_code")


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [ToolUniverseParams().tools, [LOCAL_TOOL]])
async def test_real_mcp_bridge(tools: list[str]) -> None:
    """A real stdio child advertises only the selection and executes local SDK tools."""
    require_tooluniverse()
    from mcp import ClientSession, StdioServerParameters  # type: ignore[import-not-found, import-untyped]
    from mcp.client.stdio import stdio_client  # type: ignore[import-not-found, import-untyped]

    server = ToolUniverseToolset(tools=tools).mcp_server()
    async with asyncio.timeout(60):
        async with stdio_client(StdioServerParameters(**server, env=os.environ.copy())) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert {tool.name for tool in listed.tools} == set(tools)
                assert all(tool.inputSchema["type"] == "object" for tool in listed.tools)
                denied = await session.call_tool("Not_selected", {})
                assert denied.isError
                if LOCAL_TOOL in tools:
                    result = await session.call_tool(LOCAL_TOOL, {"topic": "loading"})
                    assert not result.isError
                    assert "loading" in str(result.content)


def test_real_biomni_mcp_attachment(tmp_path: Path) -> None:
    """Construct A1 without a data lake or LLM request and invoke TU through its MCP API."""
    require_tooluniverse()
    from deep_research_client.providers.biomni import BiomniProvider, missing_biomni_runtime_modules

    if missing_biomni_runtime_modules():
        if os.getenv("REQUIRE_BIOMNI_RUNTIME") == "1":
            pytest.fail("Biomni SDK is missing")
        pytest.skip("Biomni runtime is not installed")
    provider = BiomniProvider(
        ProviderConfig(name="biomni", api_key="offline-test-key"),
        BiomniParams(
            path=str(tmp_path), skip_data_lake=True, use_tool_retriever=False,
            tooluniverse=ToolUniverseToolset(tools=[LOCAL_TOOL]),
        ),
    )
    agent = provider._build_agent()
    with asyncio.Runner() as runner:
        runner.get_loop()
        with provider._attach_tooluniverse(agent):
            assert LOCAL_TOOL in agent.list_custom_tools()
            result = agent.get_custom_tool(LOCAL_TOOL)(topic="loading")
            assert "loading" in str(result)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_config", [False, True])
async def test_real_claude_uses_tooluniverse(tmp_path: Path, workspace_config: bool) -> None:
    """Prove both inline Claude and Cyberian-style workspace configs reach a real tool call."""
    require_tooluniverse()
    if os.getenv("RUN_TOOLUNIVERSE_CLAUDE_INTEGRATION") != "1":
        pytest.skip("Set RUN_TOOLUNIVERSE_CLAUDE_INTEGRATION=1 for a paid Claude run")
    toolset = ToolUniverseToolset(tools=[LOCAL_TOOL])
    params = ClaudeCodeParams(
        model="haiku", working_dir=str(tmp_path), timeout=120,
        tooluniverse=None if workspace_config else toolset,
        allowed_tools=[f"mcp__tu__{LOCAL_TOOL}"] if workspace_config else [],
        extra_args=["--max-turns", "4"],
    )
    provider = ClaudeCodeProvider(ProviderConfig(name="claude_code"), params)
    if not provider.is_available():
        pytest.fail(provider.unavailable_reason())
    if workspace_config:
        CyberianProvider(ProviderConfig(name="cyberian"), CyberianParams(tooluniverse=toolset))._prepare_tooluniverse_workdir(str(tmp_path))
    stdout, stderr, code = await provider._run_process(
        provider._build_command(),
        f"Call the MCP tool {LOCAL_TOOL} with topic='loading'. "
        "Then summarize its returned usage tips in a short paragraph. You must call the tool.",
    )
    assert code == 0, stderr
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    tool_calls = [
        block["name"] for event in events if event.get("type") == "assistant"
        for block in event.get("message", {}).get("content", []) if block.get("type") == "tool_use"
    ]
    assert f"mcp__tu__{LOCAL_TOOL}" in tool_calls
    terminal = [event for event in events if event.get("type") == "result"][-1]
    assert not terminal.get("is_error")
    assert not terminal.get("permission_denials")
