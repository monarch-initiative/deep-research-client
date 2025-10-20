# Deep Research Client Skills

This directory contains Claude Code skills for the deep-research-client project.

## Available Skills

### run-deep-research

A comprehensive skill for performing deep research using multiple AI research providers (OpenAI, Falcon, Perplexity, Consensus).

**Key Features:**
- Support for multiple research providers with different speed/depth trade-offs
- Smart caching to avoid expensive re-queries
- Template-based research with variable substitution
- Rich markdown output with citations and metadata

**Installation:**

To use this skill locally:

```bash
# Copy the skill to your local skills directory
cp -r .claude/skills/run-deep-research ~/.claude/skills/
```

Or to use it project-wide, it's already available in `.claude/skills/run-deep-research/`.

**Usage:**

Once installed, Claude Code will automatically use this skill when you request deep research tasks. The skill will:

1. Ask you to choose between fast/light or comprehensive/slow approaches
2. Help you select the appropriate provider and model
3. Execute the research query
4. Save results with proper citations and metadata

**Example prompts that trigger this skill:**
- "Research the latest developments in CRISPR gene editing"
- "I need a comprehensive analysis of quantum computing trends"
- "Do a literature review on machine learning interpretability"
- "Research gene TP53 in humans focusing on cancer associations"

## Adding More Skills

To add additional skills to this project:

1. **Simple skill**: Create a single `.md` file like `skill-name.md` with YAML frontmatter:
   ```yaml
   ---
   name: your-skill-name
   description: Brief description of what your skill does (200 char max)
   ---
   ```

2. **Complex skill** (with templates/examples): Create a directory with `SKILL.md`:
   ```
   .claude/skills/skill-name/
   ├── SKILL.md          # Main skill with YAML frontmatter
   └── examples/         # Optional supporting files
       └── template.md
   ```

3. Add the skill instructions and documentation in the markdown body

## Skill Development Guidelines

When creating skills for this project:

- Follow the YAML frontmatter format with `name` and `description`
- Include clear "When to Use" sections
- Provide concrete usage examples with `uv run` commands
- Document all environment variables and configuration
- Include workflow steps for Claude to follow
- Add common patterns and use cases

## License

Skills in this directory are part of the deep-research-client project and follow the same BSD-3-Clause license.
