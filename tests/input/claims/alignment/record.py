"""Re-record the relation-classification replies that test_alignment_recorded.py replays.

Loads the two fixture claim sets, generates the candidate pairs exactly as
``align`` would, writes the prompts this code builds to prompts/, then sends each
through the local Claude Code CLI (ClaudeCodeChatClient) for every model in the
manifest and writes each reply verbatim. Needs a logged-in `claude` CLI.

    uv run python tests/input/claims/alignment/record.py

Then set prompt_version in manifest.yaml to the one printed, and run
`uv run pytest tests/test_alignment_recorded.py`.
"""

import asyncio
import json
from pathlib import Path

import yaml

from deep_research_client.claims import ClaimSet, candidate_pairs
from deep_research_client.claims.align_llm import ALIGN_PROMPT_VERSION, DEFAULT_ALIGN_MAX_TOKENS, build_align_prompt
from deep_research_client.claude_code_chat import ClaudeCodeChatClient

HERE = Path(__file__).parent


def _pairs():
    """The candidate pairs, in the order the test will recompute them."""
    set_a = ClaimSet.model_validate(json.loads((HERE / "report_a.claims.json").read_text("utf-8")))
    set_b = ClaimSet.model_validate(json.loads((HERE / "gold_b.claims.json").read_text("utf-8")))
    return candidate_pairs(set_a.claim_list, set_b.claim_list)


async def main() -> None:
    """Write the prompts, then record every model's reply to each pair."""
    manifest = yaml.safe_load((HERE / "manifest.yaml").read_text(encoding="utf-8"))
    pairs = _pairs()
    named = [(f"{i:02d}.txt", pair) for i, pair in enumerate(pairs, 1)]

    prompts = HERE / "prompts"
    prompts.mkdir(exist_ok=True)
    for stale in prompts.glob("*.txt"):
        stale.unlink()
    for name, pair in named:
        system, user = build_align_prompt(pair.claim_a, pair.claim_b)
        (prompts / "SYSTEM.txt").write_text(system["content"], encoding="utf-8")
        (prompts / name).write_text(user["content"], encoding="utf-8")

    client = ClaudeCodeChatClient()
    for model in manifest["recordings"]:
        out = HERE / model
        out.mkdir(exist_ok=True)
        for stale in out.glob("*.txt"):
            stale.unlink()
        replies = await asyncio.gather(*(
            client.chat.completions.create(
                model=model, messages=build_align_prompt(pair.claim_a, pair.claim_b),
                max_tokens=DEFAULT_ALIGN_MAX_TOKENS,
            )
            for _, pair in named
        ))
        for (name, _), reply in zip(named, replies):
            choice = reply.choices[0]
            if choice.finish_reason != "stop":
                raise SystemExit(f"{model} {name}: reply ended with {choice.finish_reason}")
            (out / name).write_text(choice.message.content, encoding="utf-8")

    print(f"Recorded {len(named)} pairs for {list(manifest['recordings'])}.")
    print(f"ALIGN_PROMPT_VERSION is {ALIGN_PROMPT_VERSION}; set prompt_version in manifest.yaml to match.")


if __name__ == "__main__":
    asyncio.run(main())
