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
| Lightcone | (local `lc` CLI) | Runs an astra.yaml analysis spec, artifacts | Very slow |
| Claude Code | (local `claude` CLI) | Agentic web research, no API key | Slow |

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
    save_artifacts=True,           # Preserve useful ZIP artifacts
    artifact_max_bytes=5 * 1024 * 1024,  # Per-artifact extraction limit
)
```

### Characteristics

- **Cost**: Variable (uses Claude under the hood)
- **Speed**: 10-60+ minutes (iterative multi-step research)
- **Capabilities**: PubMed search, code execution, hypothesis-driven research
- **Citations**: PMID format with deduplication
- **Artifacts**: Useful figures, small structured files, and rendered reports from the OpenScientist artifact ZIP are returned as `ResearchArtifact` entries. Runtime scaffolding, logs, transcripts, archives, and oversized files are skipped by default.

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

## Claude Code

Claude Code is a local command-line tool rather than an HTTP API. This provider
shells out to the `claude` binary in non-interactive ("print") mode, pipes the
research prompt to it via stdin, and lets Claude Code's own agentic tools (web
search, web fetch, file reading) carry out the research. The final answer comes
back as a cited markdown report.

### Setup

No API key is needed — authentication and billing are handled by your local
Claude Code installation. Just make sure the CLI is installed and on your PATH:

```bash
claude --version   # should succeed
```

The provider is auto-detected whenever `claude` is found on PATH. Set
`DISABLE_CLAUDE_CODE_PROVIDER=true` to opt out of auto-detection.

### Parameters

```python
from deep_research_client.provider_params import ClaudeCodeParams

params = ClaudeCodeParams(
    model="opus",                 # optional; forwarded to `claude --model`
    allowed_tools=["WebSearch", "WebFetch"],  # tool allowlist (default: read-only research set)
    skip_permissions=False,       # default; True bypasses ALL checks (see Security)
    add_dirs=["/data/papers"],    # optional --add-dir entries
    working_dir="/tmp/research",  # optional cwd for the run
    extra_args=["--max-turns", "30"],  # escape hatch for unmodeled flags
)
```

When `skip_permissions` is `False` (the default), you can also set
`permission_mode` (e.g. `"plan"` or `"acceptEdits"`).

### Security

In non-interactive mode the run is driven by an agent, and
`get_first_available()` may select this provider for an arbitrary query whenever
`claude` is on PATH. To keep that safe by default:

- `allowed_tools` defaults to a **read-only research set** (`["WebSearch", "WebFetch"]`),
  passed via `--allowedTools`. Tools not on the list are auto-denied (without
  blocking the run), so the agent cannot edit files or run shell commands.
- `skip_permissions` defaults to **`False`**. Setting it `True` adds
  `--dangerously-skip-permissions`, which bypasses every permission check and
  **makes `allowed_tools` a no-op** (all tools become available). Only enable it
  in trusted, sandboxed environments.

Widen `allowed_tools` if a task genuinely needs more. The most common case is
**research over local documents**: the default set is web-only and deliberately
omits the `Read` tool, so to let Claude Code read files you have supplied (for
example in an `add_dirs` path) you must add it explicitly:

```python
params = ClaudeCodeParams(
    allowed_tools=["WebSearch", "WebFetch", "Read"],
    add_dirs=["/data/papers"],
)
```

`Read` is left out of the default because it grants the agent read access to the
local filesystem, which is unnecessary for purely web-based research and is a
mild information-disclosure surface if the query is untrusted. Add it only when
reading local documents is actually part of the task.

### Usage

```bash
deep-research-client research --provider claude_code "your research question"
```

See [a full example report](../examples/claude-code-deep-research-example.md)
produced by this provider, including the YAML frontmatter that records the
actual model(s) used and run provenance (`run_metadata`).

### Characteristics

- **Cost**: Handled by your Claude Code subscription / API key
- **Speed**: Slow (agentic, multi-step)
- **Capabilities**: Web search, citation tracking, code interpretation
- **Auth**: None required by this client; relies on local Claude Code

### Limitations

- Requires the `claude` CLI installed and authenticated locally
- Restricted to a read-only research toolset by default; broaden `allowed_tools`
  (or enable `skip_permissions` in a sandbox) for tasks that need more
- Non-deterministic results

---

## Lightcone (ASTRA)

Lightcone is an *agentic-analysis* provider, not a literature one. The local
`lc` CLI materializes an **ASTRA** (*Agentic Schema for Transparent Research
Analysis*) specification — an `astra.yaml` file — into a tree of outputs,
recording the methodological decisions and their provenance. This provider is a
**spec runner**: the research "query" is a path to an ASTRA project directory
(or an `astra.yaml` file), *not* a free-text question. The spec already declares
the inputs, decisions, and expected outputs; the provider drives `lc` in the
project directory and harvests the materialized output tree into the standard
`ResearchResult` (report `markdown` plus non-text `artifacts`).

An `astra.yaml` encodes a hypothesis *as the analysis that would test it*. For a
disease-mechanism ("dismech") hypothesis, for example, the `name`/`hypothesis`
state the claim while `decisions` capture each methodological fork (enrichment
background, DE method, significance cutoff) with the literature that justifies
each option; separate `universes/*.yaml` files pin one option per decision so
you can materialize several and see whether the verdict is decision-robust. A
runnable sample lives in `tests/input/astra_dismech/`.

### How it runs

The provider invokes `lc run` in the project directory. Per the Lightcone CLI,
`lc run [OPTIONS] [OUTPUTS]...` generates a Snakefile and dispatches it through
Snakemake + Dask; `--universe NAME` restricts execution to one universe (all of
`universes/*.yaml` otherwise), and positional output IDs restrict which outputs
are built. If your recipes run in containers, run `lc build` first to prepare
images.

Materialized outputs are discovered **deterministically** rather than by
guessing an output directory: `lc` writes a `.lightcone-manifest.json` sidecar
next to each output, recording its `output_id`, `universe_id`, `code_version`,
`data_version`, `decisions`, `recipe`, and more. The provider finds those
sidecars, harvests the sibling output files as `artifacts`, and surfaces the
per-output provenance under `run_metadata["outputs"]`.

### Setup

No API key is needed — Lightcone drives a local agent (Claude Code) that handles
auth/billing. Install the CLI and make sure it is on your PATH:

```bash
lc --version   # should succeed
```

The provider is auto-detected whenever `lc` is found on PATH. Set
`DISABLE_LIGHTCONE_PROVIDER=true` to opt out of auto-detection.

### Parameters

```python
from deep_research_client.provider_params import LightconeParams

params = LightconeParams(
    universe="baseline",          # optional; else all universes/*.yaml are built
    outputs=["mito_enrichment"],  # optional output IDs to materialize (else all)
    jobs=4,                       # optional parallel Snakemake jobs (--jobs)
    force=False,                  # rebuild everything, ignoring staleness (--force)
    materialize_args=["run"],     # the `lc` subcommand (overridable escape hatch)
    scan_subdir="",               # restrict manifest scan to a subdir (default: whole project)
    working_dir="/data/my-astra-project",  # explicit project dir (else derived from query)
    save_artifacts=True,          # harvest output files discovered via manifests
    extra_args=[],                # escape hatch for unmodeled flags
)
```

### Usage

```bash
# The "query" is a path to an ASTRA project (or its astra.yaml)
deep-research-client research --provider lightcone ./my-astra-project
```

### Security

Materializing an ASTRA spec runs an agent that **executes code**. Only run
Lightcone against trusted projects in a sandboxed environment.

### Characteristics

- **Cost**: Handled by the local agent (Claude Code) it drives
- **Speed**: Very slow (executes a full analysis)
- **Capabilities**: Code interpretation, citation tracking, decision provenance
- **Auth**: None required by this client; relies on local `lc` / Claude Code

### Limitations

- Requires the `lc` CLI installed and an ASTRA project to run against
- Input is a project/spec path, not a free-text research question
- Executes code as part of the analysis (possibly in containers); run only in a
  trusted environment
- Non-deterministic results

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
