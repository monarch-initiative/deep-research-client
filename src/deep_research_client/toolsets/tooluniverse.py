"""Shared ToolUniverse selection and MCP configuration, independent of agents."""

from contextlib import contextmanager
import importlib.util
import json
import os
import sys
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..exceptions import ProviderNotConfiguredError, ProviderNotInstalledError


def default_tooluniverse_tools() -> list[str]:
    """Return an independent copy of the default biomedical tool selection."""
    return [
        "PubMed_search_articles",
        "PubMed_get_article",
        "EuropePMC_search_articles",
        "OpenTargets_get_disease_id_description_by_name",
        "OpenTargets_get_associated_targets_by_disease_efoId",
    ]


class ToolUniverseToolset(BaseModel):
    """Select scientific tools independently of the host LLM and agent loop.

    >>> config = ToolUniverseToolset(tools=["PubMed_get_article"])
    >>> config.claude_allowed_tools()
    ['mcp__tu__PubMed_get_article']
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    tools: list[str] = Field(
        default_factory=default_tooluniverse_tools, min_length=1,
        description="Exact ToolUniverse tool names exposed to the host agent (no wildcards)",
    )

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, tools: list[str]) -> list[str]:
        """Reject ambiguous names before loading tools or launching an agent."""
        if any(not name.isidentifier() for name in tools):
            raise ValueError("tools must contain exact Python identifier names, without whitespace or wildcards")
        if len(set(tools)) != len(tools):
            raise ValueError("tools must not contain duplicate names")
        return tools

    def require_runtime(self, provider: str) -> None:
        """Check the optional scientific SDK without requiring smolagents or an LLM key."""
        if importlib.util.find_spec("tooluniverse") is None:
            raise ProviderNotInstalledError(
                provider, "ToolUniverse mixin requires `uv add 'deep-research-client[tooluniverse-tools]'`",
            )

    def load_tools(self, universe: Any) -> None:
        """Load the selected tools and fail if any requested name is unavailable."""
        universe.load_tools(include_tools=self.tools)
        missing = [name for name in self.tools if name not in universe.all_tool_dict]
        if missing:
            raise ValueError(
                f"ToolUniverse tools not loaded: {', '.join(missing)}. "
                "Check exact tool names and any tool-specific API keys or dependencies."
            )

    def prepare(self, provider: str) -> None:
        """Validate installation and tool selection before the host spends LLM tokens."""
        self.require_runtime(provider)
        try:
            with self.open_universe():
                pass
        except ValueError as error:
            raise ProviderNotConfiguredError(provider, str(error)) from error

    @contextmanager
    def open_universe(self) -> Iterator[Any]:
        """Own the SDK lifetime independently of any particular agent runtime."""
        from tooluniverse import ToolUniverse  # type: ignore[import-not-found, import-untyped]

        universe = ToolUniverse()
        try:
            self.load_tools(universe)
            yield universe
        finally:
            universe.close()

    def mcp_server(self) -> dict[str, Any]:
        """Describe a stdio server using this installation's interpreter and tools.

        The small MCP bridge exposes exactly the selection, with no generic
        dispatcher that could bypass it. It requires no smolagents dependency.
        """
        return {
            "command": sys.executable,
            "args": [
                "-m", "deep_research_client.toolsets.tooluniverse_mcp",
                "--tools", json.dumps(self.tools),
            ],
        }

    def claude_mcp_config(self) -> dict[str, Any]:
        """Return Claude-compatible configuration without changing global settings."""
        return {"mcpServers": {"tu": self.mcp_server()}}

    def claude_allowed_tools(self) -> list[str]:
        """Name only the selected MCP tools for Claude's permission allowlist."""
        return [f"mcp__tu__{name}" for name in self.tools]

    def biomni_mcp_config(self, server_name: str) -> dict[str, Any]:
        """Translate to Biomni's YAML format, forwarding environment by reference.

        Biomni's MCP subprocess otherwise gets only MCP's small default
        environment, losing scientific API credentials. Its ${VAR} expansion
        lets the temporary configuration avoid containing credential values.
        """
        server = self.mcp_server()
        return {"mcp_servers": {server_name: {
            "command": [server["command"], *server["args"]],
            "env": {key: "${" + key + "}" for key in os.environ},
        }}}

    def provenance(self) -> dict[str, Any]:
        """Describe the requested composition without secrets or machine paths."""
        return {"name": "tooluniverse", "tools": self.tools.copy()}


class ToolUniverseMixin(BaseModel):
    """Parameters shared by hosts that can attach ToolUniverse scientific tools.

    True enables the default selection; a configuration object selects tools.
    False/None disables the mixin. Credentials and reasoning stay with the host.
    """

    tooluniverse: ToolUniverseToolset | None = Field(
        default=None, description="ToolUniverse tools: true for defaults, or {\"tools\": [...]}",
    )

    @field_validator("tooluniverse", mode="before")
    @classmethod
    def normalize_tooluniverse(cls, value: Any) -> Any:
        """Allow the concise CLI spelling --param tooluniverse=true."""
        if isinstance(value, str):
            value = json.loads(value)
        if value is True:
            return ToolUniverseToolset()
        if value is False:
            return None
        return value
