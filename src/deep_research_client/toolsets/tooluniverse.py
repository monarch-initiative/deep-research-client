"""Shared ToolUniverse selection and MCP configuration, independent of agents."""

from contextlib import contextmanager
import importlib.util
import json
import os
import re
import sys
from typing import Any, Iterator

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..exceptions import ProviderNotConfiguredError, ProviderNotInstalledError


def default_tooluniverse_tools() -> list[str]:
    """Return an independent copy of the default biomedical tool selection."""
    return sorted([
        "PubMed_search_articles",
        "PubMed_get_article",
        "EuropePMC_search_articles",
        "OpenTargets_get_disease_id_description_by_name",
        "OpenTargets_get_associated_targets_by_disease_efoId",
    ])


def tooluniverse_result_is_error(result: Any) -> bool:
    """Recognize SDK failures without mistaking scientific prose for an error.

    >>> tooluniverse_result_is_error("Error executing tool PubMed: unavailable")
    True
    >>> tooluniverse_result_is_error({"status": "error", "error": "Invalid topic"})
    True
    >>> tooluniverse_result_is_error("Error rates were lower in the treated group.")
    False
    """
    if isinstance(result, dict):
        return bool(result.get("error")) or result.get("status") == "error" or result.get("success") is False
    if isinstance(result, str):
        return re.match(
            r"^\s*Error(?::|\s+(?:executing|running|calling|fetching|retrieving|extracting)\b)",
            result, re.IGNORECASE,
        ) is not None
    return False


class ToolUniverseSelection(BaseModel):
    """Shared scientific selection and SDK lifetime, independent of MCP or agents."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    tools: list[str] = Field(
        default_factory=default_tooluniverse_tools, min_length=1,
        description="Exact ToolUniverse tool names exposed to the host agent (no wildcards)",
    )
    workspace: str | None = Field(
        default=None,
        description="ToolUniverse workspace; when unset, honors TOOLUNIVERSE_HOME or ./.tooluniverse",
    )

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, tools: list[str]) -> list[str]:
        """Reject ambiguous names before loading tools or launching an agent."""
        if any(not name.isidentifier() for name in tools):
            raise ValueError("tools must contain exact Python identifier names, without whitespace or wildcards")
        if len(set(tools)) != len(tools):
            raise ValueError("tools must not contain duplicate names")
        return sorted(tools)

    def require_runtime(self, provider: str) -> None:
        """Check the optional scientific SDK without requiring smolagents or an LLM key."""
        if any(importlib.util.find_spec(name) is None for name in ("tooluniverse", "mcp")):
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
        with self.open_universe(provider):
            pass

    @contextmanager
    def open_universe(self, provider: str = "tooluniverse") -> Iterator[Any]:
        """Own the SDK lifetime independently of any particular agent runtime."""
        from tooluniverse import ToolUniverse  # type: ignore[import-not-found, import-untyped]

        universe = ToolUniverse(workspace=self.workspace)
        try:
            # Only selection errors are configuration failures. Exceptions
            # from the caller's run must retain their original classification.
            try:
                self.load_tools(universe)
            except ValueError as error:
                raise ProviderNotConfiguredError(provider, str(error)) from error
            yield universe
        finally:
            universe.close()

    def provenance(self) -> dict[str, Any]:
        """Describe the requested composition without secrets or machine paths."""
        return {"name": "tooluniverse", "tools": sorted(self.tools)}


class ToolUniverseToolset(ToolUniverseSelection):
    """Attach a scientific selection to a host using MCP.

    >>> config = ToolUniverseToolset(tools=["PubMed_get_article"])
    >>> config.claude_allowed_tools()
    ['mcp__tu__PubMed_get_article']
    """

    env_vars: list[str] = Field(
        default_factory=lambda: ["NCBI_API_KEY"],
        description="Environment variable names to forward to Biomni's scientific MCP child",
    )

    @field_validator("env_vars")
    @classmethod
    def validate_environment_names(cls, names: list[str]) -> list[str]:
        """Require portable names for explicit credential forwarding."""
        if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in names):
            raise ValueError("env_vars must contain portable environment variable names")
        return sorted(set(names))

    def mcp_server(self) -> dict[str, Any]:
        """Describe a stdio server using this installation's interpreter and tools.

        The small MCP bridge exposes exactly the selection, with no generic
        dispatcher that could bypass it. It requires no smolagents dependency.
        Serialize only toolset fields so subclasses cannot send host parameters.
        """
        return {
            "command": sys.executable,
            "args": [
                "-m", "deep_research_client.toolsets.tooluniverse_mcp",
                "--config", self.model_dump_json(include=set(ToolUniverseToolset.model_fields)),
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

        MCP already forwards basic process variables (PATH, HOME, etc.). Only
        explicitly selected scientific credential names are added. Biomni's
        ${VAR} expansion keeps credential values out of the temporary file.
        """
        server = self.mcp_server()
        return {"mcp_servers": {server_name: {
            "command": [server["command"], *server["args"]],
            "env": {key: "${" + key + "}" for key in self.env_vars if key in os.environ},
        }}}


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

    def toolset_run_metadata(self) -> dict[str, Any]:
        """Return shared composition provenance for the host's ResearchResult."""
        return {"toolsets": [self.tooluniverse.provenance()]} if self.tooluniverse else {}
