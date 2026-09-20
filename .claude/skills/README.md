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
