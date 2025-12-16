# Architecture

This page explains how deep-research-client is designed.

## Overview

```
┌─────────────────────────────────────────────────┐
│                      CLI                         │
│              (typer-based commands)              │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│              DeepResearchClient                  │
│         (orchestrates research flow)             │
└─────────────────────┬───────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│   Cache   │  │ Templates │  │ Providers │
└───────────┘  └───────────┘  └─────┬─────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │           │         │         │           │
              ▼           ▼         ▼         ▼           ▼
         ┌────────┐ ┌──────────┐ ┌───────┐ ┌───────────┐ ┌──────────┐
         │ OpenAI │ │Perplexity│ │Edison │ │ Consensus │ │Cyberian  │
         └────────┘ └──────────┘ └───────┘ └───────────┘ └──────────┘
```

## Core Components

### DeepResearchClient

The main entry point for all research operations.

**Location:** `src/deep_research_client/client.py`

**Responsibilities:**

- Provider selection and initialization
- Cache management
- Template processing
- Result formatting

### Providers

Each provider implements the `ResearchProvider` interface.

**Location:** `src/deep_research_client/providers/`

**Interface:**

```python
class ResearchProvider:
    async def research(self, query: str, **kwargs) -> ResearchResult
```

**Implementations:**

- `openai.py` - OpenAI Deep Research
- `perplexity.py` - Perplexity AI
- `edison.py` - Edison Scientific (formerly Falcon)
- `consensus.py` - Consensus Academic Search
- `cyberian.py` - Cyberian agent-based research

### Models

Pydantic models for type safety and validation.

**Location:** `src/deep_research_client/models.py`

**Key models:**

- `ResearchResult` - Standard output from all providers
- `ProviderConfig` - Provider configuration
- `CacheConfig` - Cache configuration

### Cache

File-based caching to avoid re-queries.

**Location:** `src/deep_research_client/cache.py`

**Key features:**

- SHA256-based cache keys
- Human-readable filenames
- JSON storage format

### Template Processing

Variable substitution for reusable queries.

**Location:** `src/deep_research_client/processing/`

**Components:**

- `template_processor.py` - Handles both f-string and Jinja2 templates
- `result_formatter.py` - Formats output with YAML frontmatter

### Model Cards

Model metadata and discovery system.

**Location:** `src/deep_research_client/model_cards.py`

**Features:**

- Model capabilities and limitations
- Cost and speed estimates
- Alias resolution

## Request Flow

### CLI Request

```
1. User runs: deep-research-client research "query"
2. CLI parses arguments (cli.py)
3. Creates DeepResearchClient instance
4. Checks cache for existing result
5. If cached: return cached result
6. If not cached:
   a. Select provider
   b. Process template (if used)
   c. Call provider.research()
   d. Cache result
   e. Format output
7. Print or save result
```

### Python API Request

```python
client = DeepResearchClient()
result = client.research("query", provider="perplexity")
```

Flow:

```
1. Check cache
2. Select provider instance
3. Call provider.research() asynchronously
4. Cache result
5. Return ResearchResult
```

## Cache Design

### Why File-Based?

- Simple to implement and debug
- Human-readable cache entries
- Easy to backup or transfer
- No database dependencies

### Cache Key Generation

```python
key = sha256(f"{provider}:{model}:{query}").hexdigest()[:8]
filename = f"{provider}-{slugify(query)}-{key}.json"
```

### Cache Location

Default: `~/.deep_research_cache/`

This is in the user's home directory because:

- Shared across projects
- Persists across working directories
- Easy to find and manage

## Template System

### Format Detection

```
File extension .j2/.jinja → Jinja2
Frontmatter format: jinja → Jinja2
Otherwise → f-string
```

### Processing Pipeline

```
1. Read template file
2. Extract frontmatter (if any)
3. Detect format
4. Substitute variables
5. Return rendered query + metadata
```

## Provider Implementation

### Adding a New Provider

1. Create `src/deep_research_client/providers/newprovider.py`
2. Implement `ResearchProvider` interface
3. Add model cards in `model_cards.py`
4. Register in `client.py`

### Provider Interface

```python
from . import ResearchProvider
from ..models import ResearchResult

class NewProvider(ResearchProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config

    async def research(
        self,
        query: str,
        model: str | None = None,
        **kwargs
    ) -> ResearchResult:
        # Call external API
        # Parse response
        # Extract citations
        return ResearchResult(
            markdown=content,
            citations=citations,
            provider=self.name,
            model=model_used,
            duration_seconds=elapsed
        )
```

## Output Format

All providers return `ResearchResult`:

```python
@dataclass
class ResearchResult:
    markdown: str           # Research content
    citations: list[str]    # Source URLs
    provider: str           # Provider name
    model: str              # Model used
    duration_seconds: float # Time taken
    cached: bool = False    # From cache?
```

The CLI formats this with YAML frontmatter:

```yaml
---
provider: perplexity
model: sonar-pro
cached: false
duration_seconds: 45.2
citation_count: 12
---

## Question
...

## Output
...

## Citations
...
```

## Error Handling

Providers handle errors at the API level. Common errors:

- API key missing/invalid
- Rate limiting
- Timeout
- Provider unavailable

The client surfaces these as exceptions rather than catching silently.
