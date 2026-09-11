"""Stdio MCP bridge exposing an explicit ToolUniverse selection to host agents.

Invoked by ToolUniverseToolset; the module is importable on a base installation.
ToolUniverse diagnostics go to stderr so they cannot corrupt the MCP stream.
"""

import argparse
import asyncio
from contextlib import redirect_stdout
import json
import sys
from typing import Any

from .tooluniverse import ToolUniverseToolset


async def serve(config: ToolUniverseToolset) -> None:
    """Serve selected scientific tools until the host closes its stdio session."""
    from mcp import types  # type: ignore[import-not-found, import-untyped]
    from mcp.server.lowlevel import Server  # type: ignore[import-not-found, import-untyped]
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found, import-untyped]

    server = Server("deep-research-tooluniverse")
    # Capture the real stdout stream before redirecting noisy SDK diagnostics.
    async with stdio_server() as (reader, writer):
        with redirect_stdout(sys.stderr), config.open_universe() as universe:

            @server.list_tools()
            async def list_tools() -> list[types.Tool]:
                """Advertise only configured tools with their original input schemas."""
                return [types.Tool(
                    name=name,
                    description=universe.all_tool_dict[name]["description"],
                    inputSchema=universe.all_tool_dict[name].get("parameter", {"type": "object"}),
                ) for name in config.tools]

            @server.call_tool()
            async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
                """Execute an allowed tool and preserve structured JSON results."""
                if name not in config.tools:
                    raise ValueError(f"Tool is not enabled: {name}")
                result = await asyncio.to_thread(
                    universe.run_one_function, {"name": name, "arguments": arguments},
                )
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(result, default=str))],
                    isError=isinstance(result, dict) and bool(result.get("error")),
                )

            await server.run(reader, writer, server.create_initialization_options())


def main() -> None:
    """Parse the selection and start the scientific MCP server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", required=True, help="JSON array of exact ToolUniverse tool names")
    args = parser.parse_args()
    asyncio.run(serve(ToolUniverseToolset(tools=json.loads(args.tools))))


if __name__ == "__main__":
    main()
