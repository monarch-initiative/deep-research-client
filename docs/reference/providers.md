# Provider Reference

Complete reference for all supported research providers.

## Overview

| Provider | Env Variable | Strengths | Speed |
|----------|--------------|-----------|-------|
| OpenAI | `OPENAI_API_KEY` | Most comprehensive | Slow |
| Perplexity | `PERPLEXITY_API_KEY` | Real-time web, multiple speeds | Fast-Slow |
| Edison | `EDISON_API_KEY` | Scientific literature | Slow |
| Asta | `ASTA_API_KEY` | Semantic Scholar-scale literature retrieval + snippets | Fast |
| Consensus | `CONSENSUS_API_KEY` | Academic papers | Fast |
| OpenScientist | `OPENSCIENTIST_API_KEY` | Autonomous research, PMID citations | Very slow |
| Cyberian | (local agents) | Agent-based, thorough | Very slow |

## OpenAI Deep Research

### Setup

```bash
export OPENAI_API_KEY="your-key"
```

### Models

| Model | Aliases | Description |
|-------|---------|-------------|
| `o3-deep-research-2025-06-26` | o3, o3-deep, o3dr | Most comprehensive |
| `o4-mini-deep-research-2025-06-26` | o4m, o4-mini, mini | Balanced speed/quality |

### Parameters

```python
from deep_research_client.provider_params import OpenAIParams

params = OpenAIParams(
    allowed_domains=["pubmed.ncbi.nlm.nih.gov"],  # Filter to domains
    temperature=0.2,
    max_tokens=4000,
    top_p=0.95
)
```

### Characteristics

- **Cost**: High to Very High
- **Speed**: 2-15 minutes
- **Context Window**: 128K tokens
- **Capabilities**: Web search, code interpretation, comprehensive synthesis

---

## Perplexity AI

### Setup

```bash
export PERPLEXITY_API_KEY="your-key"
```

### Models

| Model | Aliases | Description |
|-------|---------|-------------|
| `sonar-deep-research` | deep, deep-research, sdr | Comprehensive |
| `sonar-pro` | pro, sp | Balanced |
| `sonar` | basic, fast, s | Fastest |

### Parameters

```python
from deep_research_client.provider_params import PerplexityParams

params = PerplexityParams(
    allowed_domains=["wikipedia.org", "github.com"],
    reasoning_effort="high",  # low, medium, high
    search_recency_filter="month"  # day, week, month, year
)

# Or use native domain filter with deny-list
params = PerplexityParams(
    search_domain_filter=[
        "github.com",       # Allow
        "-reddit.com",      # Deny (prefix with -)
    ]
)
```

### Characteristics

- **Cost**: Low to High (depends on model)
- **Speed**: Seconds to minutes
- **Context Window**: 100K-200K tokens
- **Capabilities**: Real-time web search, recent data

---

## Edison Scientific (Falcon)

### Setup

```bash
export EDISON_API_KEY="your-key"
```

### Models

| Model | Aliases | Description |
|-------|---------|-------------|
| `Edison Scientific Literature` | falcon, edison, eds, science | Scientific papers |

### Parameters

```python
from deep_research_client.provider_params import FalconParams

params = FalconParams(
    temperature=0.1,
    max_tokens=8000
)
```

### Characteristics

- **Cost**: High
- **Speed**: 2-5 minutes
- **Capabilities**: Scientific literature, powered by PaperQA3
- **Artifacts**: Edison output artifacts are fetched from the completed task. Image artifacts such as diagrams, charts, and figures are written beside saved reports and embedded in the generated Markdown; other artifact files are linked from an `Artifacts` section.

---

## Asta

### Setup

```bash
export ASTA_API_KEY="your-key"
```

### Models

| Model | Aliases | Description |
|-------|---------|-------------|
| `Asta Scientific Corpus Retrieval` | asta, retrieval, snippets | Retrieval-only paper and snippet lookup |

### Parameters

```python
from deep_research_client.provider_params import AstaParams

params = AstaParams(
    query_char_limit=500,
    paper_limit=50,
    snippet_limit=20,
    publication_date_range="2021:",
    venues="Nature,Science"
)
```

### Characteristics

- **Cost**: Free
- **Speed**: Usually a few seconds
- **Capabilities**: Scientific literature retrieval, snippet search, direct evidence reporting

---

## Consensus

### Setup

```bash
export CONSENSUS_API_KEY="your-key"
```

Note: Requires application approval.

### Models

| Model | Aliases | Description |
|-------|---------|-------------|
| `Consensus Academic Search` | consensus, academic, papers, c | Peer-reviewed only |

### Characteristics

- **Cost**: Low
- **Speed**: Seconds
- **Capabilities**: Academic papers only, evidence-based summaries

---

## OpenScientist

### Setup

```bash
export OPENSCIENTIST_API_KEY="name:secret"
# Optional: custom instance URL
export OPENSCIENTIST_URL="https://www.openscientist.io"
```

**Important**: Your account must be approved by an administrator at [openscientist.io](https://openscientist.io) before you can create jobs. Until approved, the API returns `403 Forbidden`.

### Models

| Model | Aliases | Description |
|-------|---------|-------------|
| `openscientist-autonomous` | openscientist, autonomous-research | Iterative hypothesis-driven research |

### Parameters

```python
from deep_research_client.provider_params import OpenScientistParams

params = OpenScientistParams(
    max_iterations=5,              # Research iterations (1-20)
    use_hypotheses=False,          # Enable hypothesis tracking
    investigation_mode="autonomous",  # "autonomous" or "coinvestigate"
    poll_interval=30,              # Seconds between status checks
    timeout=3600,                  # Max wait time (1-2 hours recommended)
)
```

### Characteristics

- **Cost**: Variable (uses Claude under the hood)
- **Speed**: 10-60+ minutes (iterative multi-step research)
- **Capabilities**: PubMed search, code execution, hypothesis-driven research
- **Citations**: PMID format with deduplication

### When to Use

- Disease pathophysiology research with PubMed citations
- Hypothesis-driven scientific literature reviews
- Biomedical mechanism discovery
- Evidence synthesis with structured PMID references

### Limitations

- Requires account approval at openscientist.io
- Very slow (designed for comprehensive research)
- PubMed-focused (may not cover all scientific domains)
- Cost scales with number of iterations

---

## Cyberian (Agent-Based)

### Setup

```bash
pip install deep-research-client[cyberian]
```

Cyberian uses local AI agents (Claude, Aider, etc.) - no separate API key needed.

### Parameters

```python
from deep_research_client.provider_params import CyberianParams

params = CyberianParams(
    agent_type="claude",       # claude, aider, cursor, goose, codex
    workflow_file=None,        # Custom workflow file
    port=3284,                 # agentapi server port
    skip_permissions=True,     # Skip permission checks
    manage_server=True,        # Start/stop agentapi server (set False for external server)
    sources="academic papers", # Source guidance
    workdir_base=None,         # Base directory for workspaces (default: system temp)
    max_iterations=None,       # Limit looping task iterations (requires cyberian >= 0.3.0)
)
```

If you want to run a pre-configured agentapi server (for example, Codex with yolo mode),
start it manually and set `manage_server=False` so deep-research-client does not restart it.

### Characteristics

- **Cost**: Variable (depends on agent)
- **Speed**: 10-30+ minutes
- **Capabilities**: Iterative research, citation management, comprehensive synthesis

### When to Use

- Comprehensive literature reviews
- Deep technical research
- Multi-source citation management

### Testing and Iteration Control

Use `max_iterations` to limit how many times looping tasks (like the `iterate` subtask) can run.
This is passed through to cyberian's TaskRunner and requires **cyberian >= 0.3.0**.

```bash
# Limit to 2 iterations for testing
deep-research-client research --provider cyberian "query" \
    --param agent_type=codex \
    --param max_iterations=2
```

This is useful for:

- **Testing**: Run a quick verification without full research
- **Cost control**: Limit agent API calls
- **Debugging**: Inspect intermediate results after fixed iterations

---

## Provider Detection

Providers are auto-detected based on environment variables:

```bash
# Check available providers
deep-research-client providers
```

## Adding Custom Providers

Create a new provider in `src/deep_research_client/providers/`:

```python
from . import ResearchProvider
from ..models import ResearchResult

class NewProvider(ResearchProvider):
    async def research(self, query: str) -> ResearchResult:
        # Implementation
        return ResearchResult(...)
```
