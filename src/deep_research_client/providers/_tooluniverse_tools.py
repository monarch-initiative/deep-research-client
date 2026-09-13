"""Optional smolagents bridge for ToolUniverse's scientific tool schemas.

Imported only when constructing an agent. ToolUniverse 1.4.1's bundled adapter
fails on union return types (including PubMed_get_article) and inserts None for
omitted optional arguments. This bridge preserves structured results and lets
ToolUniverse apply its own argument defaults and validation.
"""

from copy import deepcopy
import json
from typing import Any

def create_tooluniverse_tool(name: str, universe: Any) -> Any:
    """Expose one scientific tool while keeping the optional SDK import lazy."""
    from smolagents import Tool  # type: ignore[import-not-found, import-untyped]

    class ToolUniverseTool(Tool):
        """Bridge runtime scientific schemas to smolagents' Tool interface."""

        # Like smolagents' own LangChain/Gradio bridges, parameters come from a
        # runtime schema rather than a statically declared forward signature.
        skip_forward_signature_validation = True
        output_type = "any"

        def __init__(self) -> None:
            """Copy input metadata while retaining the original SDK for execution."""
            config = universe.all_tool_dict[name]
            self.name = name
            self.description = config["description"]
            schema = config.get("parameter", {})
            required = set(schema.get("required", []))
            self.inputs = deepcopy(schema.get("properties", {}))
            for parameter_name, parameter in self.inputs.items():
                parameter.setdefault("type", "any")
                parameter.setdefault("description", "")
                if parameter_name not in required:
                    parameter["nullable"] = True
            # smolagents treats output_schema as proof of a dict result, but
            # ToolUniverse also returns lists/scalars/unions. Describe the real
            # schema without giving the agent a false promise of a dictionary.
            if config.get("return_schema"):
                self.description += "\nReturn schema: " + json.dumps(config["return_schema"])
            super().__init__()

        def forward(self, *args: Any, **arguments: Any) -> Any:
            """Execute with exactly the arguments supplied by the agent."""
            if len(args) > len(self.inputs):
                raise TypeError(f"Too many positional arguments for {self.name}")
            positional = dict(zip(self.inputs, args))
            duplicates = positional.keys() & arguments.keys()
            if duplicates:
                raise TypeError(f"Duplicate arguments for {self.name}: {sorted(duplicates)}")
            arguments = {**positional, **arguments}
            return universe.run_one_function({"name": self.name, "arguments": arguments})

    return ToolUniverseTool()
