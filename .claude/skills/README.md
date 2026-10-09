# Deep Research Client Skills

These agent skills support deep-research-client workflows. They live under
`.claude/skills/` for discovery in Claude Code and can also be installed for
other supported agents with the [skills CLI](https://skills.sh/).

## Available Skills

- [run-deep-research](run-deep-research/SKILL.md): choose a provider and model,
  preserve cached research, and assess reports and citations.
- [gene-set-enrichment](gene-set-enrichment/SKILL.md): research gene sets with
  structured summaries and gene groupings.

## Installation

Preview the available skills, then choose the research workflow:

```bash
npx skills add monarch-initiative/deep-research-client --list
npx skills add monarch-initiative/deep-research-client --skill run-deep-research
```

Installation defaults to the current project. Add `-g` for user-wide installation
and `-a codex` or `-a claude-code` to select an agent. See the
[root README](../../README.md#agent-skills) for runtime setup and manual copying.
Installing a skill does not install DRC or configure provider credentials.

### extract-claims

Extracts the atomic claims a source makes with `deep-research-client claims extract`: a deep research report, a dismech or ai-gene-review file, or any YAML/JSON document with prose. Each claim is anchored to the passage that states it, with the citations the source attaches.

**Key Features:**
- Picks the model backend by checking what works: a logged-in Claude Code (no API key) or an OpenAI-compatible key. Curated files need no model.
- Explains each stderr warning and each error, with the fix
- Summarises the result with `review-claim-set`'s script

**Example prompts:**
- "What does report.md claim?"
- "Extract the claims from this Asta report"
- "Turn Marfan_Syndrome.yaml into a claim set"

### review-claim-set

Reads, audits and filters a claim set (`*.claims.json` or `*.claims.yaml`): what was checked by code and what was the model's judgement, which claims failed to anchor and why, and subsets such as cited domain claims.

**Key Features:**
- `scripts/summarize_claims.py` prints counts, checks the set against its source (SHA-256 and every span), and lists unanchored claims with their nearest passage, cited background claims and uncited domain claims, by section
- Python recipes for filtering by topic, citation status and section

**Example prompts:**
- "Can I trust the claims in report.claims.json?"
- "Why didn't these claims anchor?"
- "Give me only the cited claims about the disease, not the papers"

Both are adapted from `docs/how-to/extract-claims.md`, which stays the full reference.

## Adding More Skills

Create a directory `.claude/skills/your-skill-name/` containing `SKILL.md` with
YAML frontmatter:

```yaml
---
name: your-skill-name
description: Describe the workflow and when an agent should use it.
---
```

Keep the instructions focused on the task. Use directory-relative links for
supporting files bundled with the skill, and avoid assuming that the user has a
checkout of this repository.

## License

Skills follow this project's BSD-3-Clause license.
