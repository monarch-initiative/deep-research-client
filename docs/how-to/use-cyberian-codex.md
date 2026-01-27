# How to Use Cyberian with Codex

This guide shows how to use the Cyberian provider with OpenAI Codex for agent-based deep research.

## Overview

Cyberian is an agent-based research provider that uses AI coding assistants (like Codex or Claude) to perform iterative research tasks. Unlike API-based providers, Cyberian:

- Runs locally using agentapi to communicate with AI agents
- Creates files (PLAN.md, REPORT.md, citations/) in a workspace
- Can iterate on research, refining results over multiple passes
- Supports multiple agent backends (Codex, Claude, Aider, etc.)

## Prerequisites

### 1. Install deep-research-client with cyberian

```bash
pip install "deep-research-client[cyberian]"
# or with uv
uv pip install "deep-research-client[cyberian]"
```

### 2. Install agentapi

Cyberian uses [agentapi](https://github.com/coder/agentapi) to communicate with AI agents:

```bash
# macOS
brew install coder/tap/agentapi

# Or download from releases
# https://github.com/coder/agentapi/releases
```

### 3. Install Codex

```bash
npm install -g @openai/codex
# or
brew install openai-codex
```

Verify installation:

```bash
which agentapi  # Should show path
which codex     # Should show path
```

## Basic Usage

### Simple Query

```bash
deep-research-client research "What is the function of the TP53 gene?" \
  --provider cyberian \
  --param agent_type=codex
```

### With Output File

```bash
deep-research-client research "Review CRISPR base editing mechanisms" \
  --provider cyberian \
  --param agent_type=codex \
  --output crispr-review.md
```

## Configuration Options

### Using workdir_base for Trusted Workspace

By default, Cyberian creates temporary directories for each research task. Use `workdir_base` to place workspaces in a trusted location (useful for Codex sandbox bypass):

```bash
deep-research-client research "Analyze autophagy pathways" \
  --provider cyberian \
  --param agent_type=codex \
  --param workdir_base=/path/to/trusted/workspace
```

### Limiting Iterations

Use `max_iterations` to limit how many times the research loop runs (useful for testing or cost control):

```bash
# Run at most 2 iterations
deep-research-client research "Brief overview of BRCA1" \
  --provider cyberian \
  --param agent_type=codex \
  --param max_iterations=2
```

### Custom Port

If the default port (3284) is in use:

```bash
deep-research-client research "Query here" \
  --provider cyberian \
  --param agent_type=codex \
  --param port=4000
```

## Advanced Usage

### Using an External agentapi Server

For debugging or custom configurations, run agentapi manually:

```bash
# Terminal 1: Start agentapi with Codex
agentapi server codex --port 3299 -- --dangerously-bypass-approvals-and-sandbox

# Terminal 2: Run research with manage_server=false
deep-research-client research "Your query" \
  --provider cyberian \
  --param agent_type=codex \
  --param port=3299 \
  --param manage_server=false
```

### Running Parallel Queries

Run multiple research queries concurrently using different ports:

```python
import asyncio
from deep_research_client import DeepResearchClient
from deep_research_client.provider_params import CyberianParams

async def parallel_research():
    client = DeepResearchClient()

    # Configure two providers with different ports
    params1 = CyberianParams(
        agent_type="codex",
        port=3284,
        workdir_base="./research/query1",
        max_iterations=3,
    )

    params2 = CyberianParams(
        agent_type="codex",
        port=3285,
        workdir_base="./research/query2",
        max_iterations=3,
    )

    # Run concurrently
    results = await asyncio.gather(
        client.research_async(
            "What is the role of p53 in cancer?",
            provider="cyberian",
            provider_params=params1,
        ),
        client.research_async(
            "How does BRCA1 affect DNA repair?",
            provider="cyberian",
            provider_params=params2,
        ),
    )

    for result in results:
        print(f"Query: {result.query}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Report length: {len(result.markdown)} chars")
        print("---")

asyncio.run(parallel_research())
```

### Using Claude Instead of Codex

```bash
deep-research-client research "Your query" \
  --provider cyberian \
  --param agent_type=claude
```

## Workflow Customization

Cyberian uses YAML workflow files to define research tasks. The default `deep-research.yaml` workflow includes:

1. Initial search and planning
2. Iterative refinement until quality criteria are met
3. Final report generation with citations

### Using a Custom Workflow

```bash
deep-research-client research "Your query" \
  --provider cyberian \
  --param agent_type=codex \
  --param workflow_file=/path/to/custom-workflow.yaml
```

## Troubleshooting

### "agentapi not found in PATH"

Ensure agentapi is installed and in your PATH:

```bash
which agentapi
# If not found, install it or add to PATH
export PATH="$PATH:/path/to/agentapi"
```

### "Codex startup can take longer"

Codex may take 10-30 seconds to start. If you see timeouts, either:

1. Increase the timeout in provider config
2. Use an external server with `manage_server=false`

### Connection Refused

If agentapi fails to start:

```bash
# Check if port is in use
lsof -i :3284

# Kill existing process if needed
kill $(lsof -t -i :3284)
```

### Research Takes Too Long

Use `max_iterations` to limit the research loop:

```bash
--param max_iterations=2
```

Or use a simpler workflow for quick tests.

## Comparison with Other Providers

| Aspect | Cyberian + Codex | OpenAI Deep Research | Perplexity |
|--------|------------------|---------------------|------------|
| Speed | Slow (10-30 min) | Slow (5-15 min) | Fast (seconds) |
| Cost | Variable | High | Low-Medium |
| Depth | Very thorough | Thorough | Variable |
| Citations | Creates files | Inline | Inline |
| Offline | Local execution | API call | API call |
| Customizable | Yes (workflows) | Limited | Limited |

## See Also

- [Provider Reference](../reference/providers.md) - Full provider documentation
- [Configuration Reference](../reference/configuration.md) - CyberianParams details
- [Choose a Provider](choose-provider.md) - Provider comparison guide
