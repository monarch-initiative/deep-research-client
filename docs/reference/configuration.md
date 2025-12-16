# Configuration Reference

Reference for all configuration options.

## Environment Variables

### Provider API Keys

| Variable | Provider | Required |
|----------|----------|----------|
| `OPENAI_API_KEY` | OpenAI Deep Research | For OpenAI |
| `EDISON_API_KEY` | Edison Scientific | For Edison |
| `PERPLEXITY_API_KEY` | Perplexity AI | For Perplexity |
| `CONSENSUS_API_KEY` | Consensus | For Consensus |
| `CBORG_API_KEY` | CBORG Proxy | For CBORG |

### Example Setup

```bash
# Add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-..."
export PERPLEXITY_API_KEY="pplx-..."
export EDISON_API_KEY="..."
export CONSENSUS_API_KEY="..."
```

## Python Configuration

### DeepResearchClient

```python
from deep_research_client import DeepResearchClient, ProviderConfig, CacheConfig

client = DeepResearchClient(
    cache_config=CacheConfig(...),
    provider_configs={...}
)
```

### CacheConfig

```python
from deep_research_client import CacheConfig

cache_config = CacheConfig(
    enabled=True,                    # Enable/disable caching
    directory="~/.deep_research_cache"  # Cache location
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | True | Enable caching |
| `directory` | str | `~/.deep_research_cache` | Cache directory path |

### ProviderConfig

```python
from deep_research_client import ProviderConfig

config = ProviderConfig(
    name="openai",
    api_key="your-key",
    base_url=None,        # Custom endpoint
    timeout=600,          # Request timeout (seconds)
    max_retries=3,        # Retry count
    enabled=True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Provider name |
| `api_key` | str | None | API key (uses env var if not set) |
| `base_url` | str | None | Custom API endpoint |
| `timeout` | int | 600 | Request timeout in seconds |
| `max_retries` | int | 3 | Number of retries |
| `enabled` | bool | True | Enable provider |

### Full Configuration Example

```python
import os
from deep_research_client import (
    DeepResearchClient,
    ProviderConfig,
    CacheConfig
)

# Custom cache settings
cache_config = CacheConfig(
    enabled=True,
    directory="./my_project_cache"
)

# Custom provider settings
provider_configs = {
    "openai": ProviderConfig(
        name="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=900,  # 15 minutes for deep research
        max_retries=5
    ),
    "perplexity": ProviderConfig(
        name="perplexity",
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        timeout=300,
        enabled=True
    )
}

client = DeepResearchClient(
    cache_config=cache_config,
    provider_configs=provider_configs
)
```

## Provider Parameters

### OpenAIParams

```python
from deep_research_client.provider_params import OpenAIParams

params = OpenAIParams(
    allowed_domains=["example.com"],  # Limit to domains
    temperature=0.2,
    max_tokens=4000,
    top_p=0.95
)

result = client.research(
    "query",
    provider="openai",
    provider_params=params
)
```

### PerplexityParams

```python
from deep_research_client.provider_params import PerplexityParams

params = PerplexityParams(
    allowed_domains=["wikipedia.org"],
    reasoning_effort="high",           # low, medium, high
    search_recency_filter="month",     # day, week, month, year
    search_domain_filter=[             # Native filter with deny support
        "github.com",
        "-reddit.com"                  # Deny with - prefix
    ]
)
```

### FalconParams

```python
from deep_research_client.provider_params import FalconParams

params = FalconParams(
    temperature=0.1,
    max_tokens=8000
)
```

### CyberianParams

```python
from deep_research_client.provider_params import CyberianParams

params = CyberianParams(
    agent_type="claude",       # claude, aider, cursor, goose
    workflow_file=None,        # Custom workflow YAML
    port=3284,                 # agentapi port
    skip_permissions=True,
    sources="academic papers"
)
```

## Cache Directory Structure

```
~/.deep_research_cache/
├── openai-what-is-crispr-a1b2c3d4.json
├── perplexity-machine-learning-f6e5d4c3.json
└── edison-protein-folding-b8f7e2a9.json
```

### Cache File Format

```json
{
  "markdown": "# Research Report\n...",
  "citations": ["Citation 1", "Citation 2"],
  "provider": "openai",
  "model": "o3-deep-research-2025-06-26",
  "query": "original query text"
}
```

## Output Format

Research output is Markdown with YAML frontmatter containing publication-style metadata:

```yaml
---
# Publication metadata (optional, user-provided)
title: "CFAP300 Gene Function Review"
abstract: "A comprehensive review of CFAP300 gene function..."
keywords:
  - genetics
  - cilia
  - rare disease
author: "Jane Doe"
contributors:
  - "John Smith"
  - "Alice Johnson"

# Execution metadata
provider: perplexity
model: sonar-pro
cached: false
start_time: '2025-01-15T10:30:00.000000'
end_time: '2025-01-15T10:30:45.000000'
duration_seconds: 45.0
citation_count: 12

# Template metadata (if template used)
template_file: gene_research.md.j2
template_variables:
  gene: TP53
  organism: human

# Provider configuration
provider_config:
  timeout: 600
  max_retries: 3

# Edit history (if present)
edit_history:
  - author: "Jane Doe"
    date: '2025-01-15T10:30:00.000000'
    summary: "Initial research"
  - author: "John Smith"
    date: '2025-01-16T14:00:00.000000'
    summary: "Added clinical relevance section"
---

## Question

Your query here

## Output

Research content...

## Citations

1. https://...
2. https://...
```

### Frontmatter Fields

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Title for the research report |
| `abstract` | string | Abstract or summary |
| `keywords` | list | Keywords/tags for the research |
| `author` | string | Primary author |
| `contributors` | list | List of contributors |
| `provider` | string | Provider used (always present) |
| `model` | string | Model used |
| `cached` | boolean | Whether result was from cache |
| `start_time` | datetime | When research started |
| `end_time` | datetime | When research completed |
| `duration_seconds` | float | Total duration |
| `citation_count` | integer | Number of citations |
| `template_file` | string | Template file used (if any) |
| `template_variables` | object | Variables passed to template |
| `provider_config` | object | Provider configuration |
| `edit_history` | list | History of edits (if any) |
