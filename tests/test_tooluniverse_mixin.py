"""ToolUniverse composition tests using real tool schemas, MCP sessions, and agents.

No mock servers or agent doubles: offline tests execute ToolUniverse's local
usage-tips tool. Paid host-agent tests require an explicit integration flag.
"""

import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
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
from deep_research_client.toolsets.tooluniverse import (
    ToolUniverseMixin, ToolUniverseToolset, tooluniverse_result_is_error,
)

LOCAL_TOOL = "ToolUniverse_get_usage_tips"


def mcp_child_pids() -> set[int]:
    """Inspect real child processes so Biomni tests can detect orphaned MCP servers."""
    if shutil.which("ps") is None:
        pytest.skip("Process lifecycle verification requires ps")
    result = subprocess.run(["ps", "-axo", "pid,ppid,command"], text=True, capture_output=True, check=True)
    children = set()
    for line in result.stdout.splitlines()[1:]:
        pid, parent, command = line.strip().split(None, 2)
        if int(parent) == os.getpid() and "deep_research_client.toolsets.tooluniverse_mcp" in command:
            children.add(int(pid))
    return children


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
    assert explicit.model_dump()["tooluniverse"] == ToolUniverseToolset(tools=[LOCAL_TOOL]).model_dump()
    assert disabled.model_dump()["tooluniverse"] is None
    assert isinstance(explicit, ToolUniverseMixin)
    assert explicit.toolset_run_metadata() == {"toolsets": [{"name": "tooluniverse", "tools": [LOCAL_TOOL]}]}
    assert isinstance(disabled, ToolUniverseMixin)
    assert disabled.toolset_run_metadata() == {}


@pytest.mark.parametrize("result,expected", [
    ("Error executing tool PubMed: failed", True), ("Error: unavailable", True),
    ("Error extracting content: corrupt input", True), ({"error": "invalid argument"}, True),
    ({"status": "error"}, True), ({"success": False}, True),
    ("Error rates improved in the experiment", False), ({"error": None, "data": [1]}, False),
    ({"status": "success", "data": "Error: quoted source text"}, False), ([1, 2], False),
])
def test_tool_failure_classification(result: Any, expected: bool) -> None:
    """Recognize failure payloads without scanning legitimate evidence for error words."""
    assert tooluniverse_result_is_error(result) is expected


@pytest.mark.skipif(os.getenv("REQUIRE_NO_SMOLAGENTS") != "1", reason="Toolset-only environment check")
def test_toolset_extra_does_not_install_smolagents() -> None:
    """Fail the dedicated CI job if the toolset begins pulling in an agent framework."""
    assert importlib.util.find_spec("tooluniverse") is not None
    assert importlib.util.find_spec("mcp") is not None
    assert importlib.util.find_spec("smolagents") is None


def test_native_stdout_cannot_corrupt_protocol() -> None:
    """Exercise real fd writes and exceptional cleanup in a separate Python process."""
    code = '''
import os
from deep_research_client.toolsets.tooluniverse_mcp import protocol_stdout
try:
    with protocol_stdout() as output:
        print("python diagnostic")
        os.write(1, b"native diagnostic\\n")
        output.write("protocol output\\n")
        raise RuntimeError("test error")
except RuntimeError:
    pass
os.write(1, b"restored output\\n")
'''
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    assert result.stdout == "protocol output\nrestored output\n"
    assert "python diagnostic" in result.stderr
    assert "native diagnostic" in result.stderr


def test_explicit_scientific_environment_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom scientific credentials are opt-in; host credentials stay out by default."""
    monkeypatch.setenv("CUSTOM_SCIENCE_KEY", "custom-key")
    toolset = ToolUniverseToolset(env_vars=["CUSTOM_SCIENCE_KEY"])
    assert toolset.biomni_mcp_config("tu")["mcp_servers"]["tu"]["env"] == {
        "CUSTOM_SCIENCE_KEY": "${CUSTOM_SCIENCE_KEY}",
    }
    with pytest.raises(ValidationError, match="portable environment"):
        ToolUniverseToolset(env_vars=["INVALID-NAME"])


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
    assert json.loads(server["args"][-1])["tools"] == [LOCAL_TOOL]
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-llm-key")
    monkeypatch.setenv("GITHUB_TOKEN", "ci-token")
    toolset = ToolUniverseToolset(tools=[LOCAL_TOOL])
    server = toolset.biomni_mcp_config("test_server")["mcp_servers"]["test_server"]
    assert server["command"][0] == sys.executable
    assert json.loads(server["command"][-1])["tools"] == [LOCAL_TOOL]
    assert server["env"] == {"NCBI_API_KEY": "${NCBI_API_KEY}"}
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
    selection = [LOCAL_TOOL, "PubMed_get_article"]
    assert params("claude_code", {"tooluniverse": {"tools": selection}}) == params(
        "claude_code", {"tooluniverse": {"tools": list(reversed(selection))}},
    )
    assert params("tooluniverse", {"tools": selection}) == params(
        "tooluniverse", {"tools": list(reversed(selection))},
    )


def test_missing_mixin_install_explains_toolset_extra() -> None:
    """The composed SDK does not require smolagents or its own LLM credential."""
    if importlib.util.find_spec("tooluniverse") is not None:
        pytest.skip("ToolUniverse is installed")
    with pytest.raises(ProviderNotInstalledError, match="tooluniverse-tools"):
        ToolUniverseToolset().prepare("claude_code")


@pytest.mark.asyncio
@pytest.mark.parametrize("tools", [ToolUniverseParams().tools, [LOCAL_TOOL]])
async def test_real_mcp_bridge(tools: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real stdio child advertises only the selection and executes local SDK tools."""
    require_tooluniverse()
    from mcp import ClientSession, StdioServerParameters  # type: ignore[import-not-found, import-untyped]
    from mcp.client.stdio import stdio_client  # type: ignore[import-not-found, import-untyped]

    # Explicit local workspace and no inherited remote profile: loading schemas
    # reads packaged JSON; the usage-tips operation itself is entirely offline.
    monkeypatch.delenv("TOOLUNIVERSE_PROFILE", raising=False)
    server = ToolUniverseToolset(tools=tools, workspace=str(tmp_path)).mcp_server()
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
                    failed = await session.call_tool(LOCAL_TOOL, {"topic": "invalid-topic"})
                    assert failed.isError


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
    before = mcp_child_pids()
    modules_before = {name for name in sys.modules if name.startswith("mcp_servers.tu_")}
    with asyncio.Runner() as runner:
        asyncio.set_event_loop(runner.get_loop())
        with provider._attach_tooluniverse(agent):
            assert LOCAL_TOOL in agent.list_custom_tools()
            assert mcp_child_pids() == before  # discovery's stdio session has already closed
            result = agent.get_custom_tool(LOCAL_TOOL)(topic="loading")
            assert "loading" in str(result)
            assert mcp_child_pids() == before  # calls own short-lived stdio sessions too
    assert mcp_child_pids() == before
    assert {name for name in sys.modules if name.startswith("mcp_servers.tu_")} == modules_before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_claude_composed_report_provenance(tmp_path: Path) -> None:
    """Run the real research entry point and assert composition survives in its result."""
    require_tooluniverse()
    if os.getenv("RUN_TOOLUNIVERSE_CLAUDE_INTEGRATION") != "1":
        pytest.skip("Set RUN_TOOLUNIVERSE_CLAUDE_INTEGRATION=1 for a paid Claude run")
    toolset = ToolUniverseToolset(tools=[LOCAL_TOOL])
    provider = ClaudeCodeProvider(
        ProviderConfig(name="claude_code"),
        ClaudeCodeParams(model="haiku", working_dir=str(tmp_path), timeout=120, tooluniverse=toolset),
    )
    result = await provider.research(
        f"Call {LOCAL_TOOL} with topic='loading'. Explain its returned tool-loading guidance "
        "in a markdown report of at least 200 words."
    )
    assert result.provider == "claude_code"
    assert result.run_metadata is not None
    assert result.run_metadata["toolsets"] == [{"name": "tooluniverse", "tools": [LOCAL_TOOL]}]
    assert not result.run_metadata.get("permission_denials")


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
