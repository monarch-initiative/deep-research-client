"""Re-record the claim-extraction replies that tests/test_claims_recorded.py replays.

Writes the prompts this code builds to prompts/, then sends each through the
local Claude Code CLI (ClaudeCodeChatClient) for every model in the manifest,
and writes each reply as the model gave it. Needs a logged-in `claude` CLI.

    uv run python tests/input/claims/recorded/record.py

Then set prompt_version in manifest.yaml to the one printed, and run
`uv run pytest tests/test_claims_recorded.py`.
"""

import asyncio
from pathlib import Path

import yaml

from deep_research_client.claims.llm import DEFAULT_MAX_TOKENS, PROMPT_VERSION, build_prompt
from deep_research_client.claims.units import markdown_units, structured_units
from deep_research_client.claude_code_chat import ClaudeCodeChatClient

RECORDED = Path(__file__).parent
INPUT = RECORDED.parent


async def main() -> None:
    """Write the prompts, then record every model's reply to each."""
    manifest = yaml.safe_load((RECORDED / "manifest.yaml").read_text(encoding="utf-8"))
    units = []
    for source in manifest["sources"]:
        path = INPUT / source["file"]
        text = path.read_text(encoding="utf-8")
        built = markdown_units(text) if path.suffix == ".md" else structured_units(yaml.safe_load(text))
        units += [(source["name"], unit) for unit in built]
    named = [(f"{i:02d}-{name}.txt", unit) for i, (name, unit) in enumerate(units, 1)]

    prompts = RECORDED / "prompts"
    for stale in prompts.glob("*.txt"):
        stale.unlink()
    for name, unit in named:
        system, user = build_prompt(unit)
        (prompts / "SYSTEM.txt").write_text(system["content"], encoding="utf-8")
        (prompts / name).write_text(user["content"], encoding="utf-8")

    client = ClaudeCodeChatClient()
    for model in manifest["recordings"]:
        out = RECORDED / model
        for stale in out.glob("*.txt"):
            stale.unlink()
        replies = await asyncio.gather(*(
            client.chat.completions.create(
                model=model, messages=build_prompt(unit), max_tokens=DEFAULT_MAX_TOKENS,
            )
            for _, unit in named
        ))
        for (name, _), reply in zip(named, replies):
            choice = reply.choices[0]
            if choice.finish_reason != "stop":
                raise SystemExit(f"{model} {name}: reply ended with {choice.finish_reason}")
            (out / name).write_text(choice.message.content, encoding="utf-8")
        print(f"{model}: {len(named)} replies from {replies[0].model}")
    print(f"prompt_version: {PROMPT_VERSION}")


if __name__ == "__main__":
    asyncio.run(main())
