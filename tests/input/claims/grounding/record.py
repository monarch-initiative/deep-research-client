"""Re-record the grounding replies that test_grounding_recorded.py replays.

For each term in manifest.yaml, searches the biomedical ontologies with OAK
(needs the `terms` extra and network), writes the candidates it found and the
prompt this code builds, then sends the prompt through the local Claude Code CLI
for every model in the manifest and writes each reply verbatim. The committed
candidates let the test run offline.

    uv run python tests/input/claims/grounding/record.py

Then set prompt_version in manifest.yaml to the one printed, and run
`uv run pytest tests/test_grounding_recorded.py`.
"""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from deep_research_client.claims import biomedical_ontology_config
from deep_research_client.claims.grounding_llm import GROUNDING_PROMPT_VERSION, build_grounding_prompt
from deep_research_client.claims.grounding_oak import OakGrounder
from deep_research_client.claude_code_chat import ClaudeCodeChatClient

HERE = Path(__file__).parent


async def main() -> None:
    manifest = yaml.safe_load((HERE / "manifest.yaml").read_text(encoding="utf-8"))
    terms = manifest["terms"]
    searcher = OakGrounder(biomedical_ontology_config(), llm_client=None, model="unused")

    (HERE / "candidates").mkdir(exist_ok=True)
    prompts = HERE / "prompts"
    prompts.mkdir(exist_ok=True)
    for stale in prompts.glob("*.txt"):
        stale.unlink()

    named = []
    for i, term in enumerate(terms, 1):
        candidates = searcher._candidates(term["label"])
        if not candidates:
            raise SystemExit(f"{term['label']}: OAK returned no candidates; pick another term")
        (HERE / "candidates" / f"{i:02d}.json").write_text(
            json.dumps([asdict(c) for c in candidates], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        system, user = build_grounding_prompt(term["label"], term["context"], candidates)
        (prompts / "SYSTEM.txt").write_text(system["content"], encoding="utf-8")
        (prompts / f"{i:02d}.txt").write_text(user["content"], encoding="utf-8")
        named.append((f"{i:02d}.txt", term, candidates))

    client = ClaudeCodeChatClient()
    for model in manifest["recordings"]:
        out = HERE / model
        out.mkdir(exist_ok=True)
        for stale in out.glob("*.txt"):
            stale.unlink()
        replies = await asyncio.gather(*(
            client.chat.completions.create(
                model=model, messages=build_grounding_prompt(term["label"], term["context"], cands),
                max_tokens=1024,
            )
            for _, term, cands in named
        ))
        for (name, _, _), reply in zip(named, replies):
            choice = reply.choices[0]
            if choice.finish_reason != "stop":
                raise SystemExit(f"{model} {name}: reply ended with {choice.finish_reason}")
            (out / name).write_text(choice.message.content, encoding="utf-8")

    print(f"Recorded {len(named)} terms for {list(manifest['recordings'])}.")
    print(f"GROUNDING_PROMPT_VERSION is {GROUNDING_PROMPT_VERSION}; set prompt_version to match.")


if __name__ == "__main__":
    asyncio.run(main())
