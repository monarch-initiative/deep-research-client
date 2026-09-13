"""Stdio MCP bridge exposing an explicit ToolUniverse selection to host agents.

Invoked by ToolUniverseToolset; the module is importable on a base installation.
ToolUniverse diagnostics go to stderr so they cannot corrupt the MCP stream.
"""

import argparse
import asyncio
from contextlib import contextmanager, redirect_stdout
import json
import os
import sys
from typing import Any, Iterator, TextIO

from .tooluniverse import ToolUniverseToolset, tooluniverse_result_is_error


@contextmanager
def protocol_stdout() -> Iterator[TextIO]:
    """Reserve the protocol pipe and send Python and native stdout noise to stderr.

    Used only inside the dedicated server process: fd 1 is process-wide. The
    MCP writer owns a duplicate of the original pipe, so native os.write/printf
    calls cannot corrupt it. Restore fd 1 on both normal and exceptional exit.
    """
    sys.stdout.flush()
    with os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1) as output:
        os.dup2(2, 1)
        try:
            with redirect_stdout(sys.stderr):
                yield output
        finally:
            sys.stderr.flush()
            os.dup2(output.fileno(), 1)


async def serve(config: ToolUniverseToolset) -> None:
    """Serve selected scientific tools until the host closes its stdio session."""
    from mcp import types  # type: ignore[import-not-found, import-untyped]
    from mcp.server.lowlevel import Server  # type: ignore[import-not-found, import-untyped]
    from mcp.server.stdio import stdio_server  # type: ignore[import-not-found, import-untyped]
    import anyio

    server = Server("deep-research-tooluniverse")
    with protocol_stdout() as output, config.open_universe() as universe:
        async with stdio_server(stdout=anyio.wrap_file(output)) as (reader, writer):

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
                    isError=tooluniverse_result_is_error(result),
                )

            await server.run(reader, writer, server.create_initialization_options())


def main() -> None:
    """Parse the selection and start the scientific MCP server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON ToolUniverse toolset configuration")
    args = parser.parse_args()
    asyncio.run(serve(ToolUniverseToolset.model_validate_json(args.config)))


if __name__ == "__main__":
    main()
