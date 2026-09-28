"""Summarise a claim set from `deep-research-client claims extract`, as markdown.

    uv run python .claude/skills/review-claim-set/scripts/summarize_claims.py \
        report.claims.json --source report.md

Loading the set runs its consistency rules, so a set whose slots contradict
each other is refused here too. With --source, the source is checked against
the set: its SHA-256, and for a markdown report every span's text.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import yaml

from deep_research_client.claims import ClaimSet, SourceType
from deep_research_client.claims.models import content_sha256, ids_by_section


def load(path: Path) -> ClaimSet:
    """Read a claim set written as JSON or YAML."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if path.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    return ClaimSet.model_validate(data)


def _table(title: str, counts: Counter, unit: str = "Claims") -> list[str]:
    """A two-column count table, largest first."""
    lines = [f"| {title} | {unit} |", "|---|---|"]
    lines += [f"| {key} | {count} |" for key, count in counts.most_common()]
    return lines + [""]


def summarize(claims: ClaimSet, source_text: Optional[str] = None) -> str:
    """The summary, as markdown.

    Args:
        claims: The claim set.
        source_text: The source it was extracted from, to check it against.

    Returns:
        Provenance, counts, the source check, and the claims to look at.
    """
    cl = claims.claim_list
    extractor = claims.extractor
    lines = [
        f"# Claims from {claims.source.title or claims.source.id}",
        "",
        f"- Source: `{claims.source.id}` ({claims.source.source_type})",
        f"- Extractor: {extractor.name}"
        + (f", model {extractor.model}" if extractor.model else "")
        + (f", prompt v{extractor.prompt_version}" if extractor.prompt_version else ""),
        f"- Claims: {len(cl)}",
        "",
    ]
    lines += _table("anchor_status", Counter(str(c.anchor_status) for c in cl))
    lines += _table("citation_status", Counter(str(c.citation_status) for c in cl))
    lines += _table("about / basis", Counter(f"{c.about or '-'} / {c.basis or '-'}" for c in cl))
    handles = [h for c in cl for h in c.citations or []]
    if handles:
        resolved = sum(1 for h in handles if h.reference_id)
        lines += _table("citation scope", Counter(str(h.scope) for h in handles), unit="Citations")
        lines += [f"{resolved} of {len(handles)} citations resolve to an identifier.", ""]

    if source_text is not None:
        lines += ["## Source check", ""]
        if claims.source.content_sha256 and content_sha256(source_text) != claims.source.content_sha256:
            lines += ["- **The source has changed since extraction**: its SHA-256 differs. "
                      "Offsets no longer apply; re-extract.", ""]
        elif claims.source.source_type == SourceType.MARKDOWN_REPORT:
            moved = claims.mismatched_spans(source_text)
            lines += [f"- SHA-256 matches. {len(moved)} spans differ from the source text.", ""]
            if moved:
                lines += ["  With the hash unchanged this is a bug in extraction. Report these:",
                          f"  {ids_by_section(moved)}", ""]
        else:
            lines += ["- SHA-256 matches. Structured claims are located by `source_path`, "
                      "so there are no spans to check.", ""]

    unanchored = claims.unanchored_claims
    if unanchored:
        lines += [f"## Unanchored claims ({len(unanchored)})", "",
                  "Their quote is not in the source. Do not treat them as the source's claims.", ""]
        for c in unanchored:
            lines.append(f"- `{c.id}` ({c.section or c.source_path or 'no section'}): {c.claim_text}")
            near = c.nearest_passage
            if near is not None:
                lines.append(f"  - nearest passage, score {near.score}: \"{near.text}\"")
        lines.append("")

    conflicting = claims.cited_background_claims
    if conflicting:
        lines += [f"## Background knowledge that carries a citation ({len(conflicting)})", "",
                  "Either the model's basis or the citation it attached is wrong.", "",
                  ids_by_section(conflicting), ""]

    # For a curated record, UNCITED means the record lists no evidence.
    uncited_domain = [c for c in cl if c.about == "DOMAIN" and c.citation_status == "UNCITED"]
    if uncited_domain:
        lines += [f"## Uncited domain claims ({len(uncited_domain)})", "",
                  ids_by_section(uncited_domain), ""]
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    """Print the summary of one claim set."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("claims", type=Path, help="A claim set (.json, .yaml or .yml)")
    parser.add_argument("--source", type=Path, help="The source it was extracted from")
    args = parser.parse_args(argv)
    source_text = args.source.read_text(encoding="utf-8") if args.source else None
    sys.stdout.write(summarize(load(args.claims), source_text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
