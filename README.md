
# deep-research-client

A simple Python wrapper for multiple deep research tools including OpenAI Deep Research, Edison Scientific (formerly FutureHouse Falcon), Perplexity AI, Consensus Academic Search, and Cyberian agent-based research.

## Features

- 🔍 **Multiple Providers**: Support for OpenAI Deep Research, Edison Scientific, Perplexity AI, Consensus, and Cyberian (agent-based)
- 📚 **Rich Output**: Returns comprehensive markdown reports with citations
- 💾 **Smart Caching**: File-based caching to avoid expensive re-queries
- 🔧 **Simple Configuration**: Auto-detects providers from environment variables
- 📝 **Library + CLI**: Use as a Python library or command-line tool
- 📋 **Advanced Templates**: Support for both simple f-string and powerful Jinja2 templates
- 🏗️ **Extensible**: Easy to add new research providers

## Installation

```bash
# Install with pip
pip install deep-research-client

# Or use uvx to run directly without installation
uvx deep-research-client research "What is CRISPR?"

# Or add to a uv project
uv add deep-research-client

# For development
git clone <repo-url>
cd deep-research-client
uv sync
```

## Quick Start

### Set up API Keys

```bash
# For OpenAI Deep Research
export OPENAI_API_KEY="your-openai-key"

# For Edison Scientific (formerly FutureHouse Falcon)
export EDISON_API_KEY="your-edison-key"

# For Perplexity AI
export PERPLEXITY_API_KEY="your-perplexity-key"

# For Consensus AI (requires application approval)
export CONSENSUS_API_KEY="your-consensus-key"

# For Cyberian (agent-based research) - requires cyberian installation
pip install deep-research-client[cyberian]
# Cyberian uses your local AI agents (Claude, etc.) - no separate API key needed
```

### Command Line Usage

```bash
# Basic research query
deep-research-client research "What is CRISPR gene editing?"

# Run without installation using uvx
uvx deep-research-client research "What is CRISPR gene editing?"

# Use specific provider and model
deep-research-client research "Explain quantum computing" --provider perplexity --model sonar-pro

# Save to file (citations included by default)
deep-research-client research "Machine learning trends 2024" --output report.md

# Save citations to separate file
deep-research-client research "AI trends 2024" --output report.md --separate-citations

# Use simple template with variables (f-string style)
deep-research-client research --template gene_research.md --var "gene=TP53" --var "organism=human"

# Use advanced Jinja2 template with conditionals
deep-research-client research --template gene_advanced.md.j2 \
  --var "gene=BRCA1" \
  --var "organism=human" \
  --var "detail_level=high"

# List available providers
deep-research-client providers

# List available models (with costs, speeds, and capabilities)
deep-research-client models
deep-research-client models --provider perplexity --detailed

# Cache management
deep-research-client list-cache   # Show cached files
deep-research-client clear-cache  # Remove all cache
```

### Python Library Usage

```python
from deep_research_client import DeepResearchClient

# Initialize client (auto-detects providers from env vars)
client = DeepResearchClient()

# Perform research with model selection
result = client.research("What are the latest developments in AI?", provider="perplexity", model="sonar-pro")

print(result.markdown)  # Full markdown report
print(f"Provider: {result.provider}")
print(f"Model: {result.model}")
print(f"Duration: {result.duration_seconds:.2f}s")
print(f"Citations: {len(result.citations)}")
print(f"Cached: {result.cached}")
```

### Advanced Configuration

```python
from deep_research_client import DeepResearchClient, ProviderConfig, CacheConfig

# Custom cache settings
cache_config = CacheConfig(
    enabled=True,
    directory="./my_cache"  # Custom cache location
)

# Custom provider settings
provider_configs = {
    "openai": ProviderConfig(
        name="openai",
        api_key="your-key",
        timeout=900,  # 15 minutes
        max_retries=5
    )
}

client = DeepResearchClient(
    cache_config=cache_config,
    provider_configs=provider_configs
)
```

### Provider-Specific Parameters

Each provider supports specific parameters for fine-tuning behavior. The library provides both **harmonized parameters** (work across all providers) and **provider-specific parameters** (native API options).

#### Harmonized Parameters

These parameters work consistently across all providers:

- `allowed_domains`: Filter web search to specific domains (max 20)
  ```python
  client.research(
      "What is CRISPR?",
      provider="openai",
      provider_params={"allowed_domains": ["wikipedia.org", "nih.gov"]}
  )
  ```

#### OpenAI-Specific Parameters

```python
from deep_research_client.provider_params import OpenAIParams

params = OpenAIParams(
    allowed_domains=["pubmed.ncbi.nlm.nih.gov", "clinicaltrials.gov"],
    temperature=0.2,
    max_tokens=4000,
    top_p=0.95
)

result = client.research("Cancer immunotherapy research", provider="openai", provider_params=params)
```

#### Perplexity-Specific Parameters

Perplexity supports both the harmonized `allowed_domains` and its native `search_domain_filter` (which supports deny-lists):

```python
from deep_research_client.provider_params import PerplexityParams

# Using harmonized parameter
params = PerplexityParams(
    allowed_domains=["wikipedia.org", "github.com"],  # Allowlist only
    reasoning_effort="high",
    search_recency_filter="month"
)

# Or using Perplexity's native parameter for more control
params = PerplexityParams(
    search_domain_filter=[
        "github.com",           # Allow
        "stackoverflow.com",    # Allow
        "-reddit.com",          # Deny (prefix with -)
        "-quora.com"            # Deny
    ],
    reasoning_effort="high"
)

result = client.research("Python best practices", provider="perplexity", provider_params=params)
```

#### Falcon-Specific Parameters

```python
from deep_research_client.provider_params import FalconParams

params = FalconParams(
    temperature=0.1,
    max_tokens=8000
)

result = client.research("Protein folding mechanisms", provider="falcon", provider_params=params)
```

#### Cyberian-Specific Parameters (Agent-Based Research)

Cyberian is unique among the providers - it uses local AI agents (Claude, Aider, etc.) running through the cyberian workflow system to perform iterative, multi-step research.

```python
from deep_research_client.provider_params import CyberianParams

# Install cyberian provider support first
# pip install deep-research-client[cyberian]

params = CyberianParams(
    agent_type="claude",           # Agent to use: claude, aider, cursor, goose
    workflow_file=None,            # Custom workflow (defaults to deep-research.yaml)
    port=3284,                     # agentapi server port
    skip_permissions=True,         # Skip permission checks for agent
    sources="academic papers"      # Source guidance for research
)

# Note: Cyberian research is slow but thorough
# - Typically takes 10-30 minutes per query
# - Performs iterative research with citation management
# - Creates structured reports with REPORT.md and citations/
result = client.research(
    "What are the mechanisms of autophagy in cancer?",
    provider="cyberian",
    provider_params=params
)

# The result includes:
# - Comprehensive markdown report (from REPORT.md)
# - Citations extracted from citations/ directory
# - Timing information for the full workflow
print(result.markdown)
print(f"Citations found: {len(result.citations)}")
print(f"Research took: {result.duration_seconds / 60:.1f} minutes")
```

**When to use Cyberian:**
- 📚 Comprehensive literature reviews
- 🔬 Deep technical research requiring synthesis
- 📊 Multi-source citation management
- 🔄 Iterative hypothesis exploration

**Trade-offs:**
- ⏱️ **Slow**: 10-30+ minutes per query (vs seconds for API providers)
- 💰 **Variable cost**: Depends on agent and research depth
- 🖥️ **Local compute**: Requires agentapi and agent setup
- 🎯 **Thorough**: More comprehensive than API-based providers

#### CLI Usage with Provider Parameters

Currently, provider-specific parameters are primarily accessible via the Python API. For CLI usage, use default parameters or create custom wrapper scripts.

## Using Proxies and OpenAI-Compatible Endpoints

The client supports using proxy services and OpenAI-compatible endpoints, enabling you to route requests through institutional proxies like CBORG, or use alternative services like Azure OpenAI, LiteLLM, or OpenRouter.

### CBORG (Berkeley Lab's AI Portal)

CBORG provides an OpenAI-compatible proxy for institutional access with cost management and tracking.

```bash
# Set up CBORG API key
export CBORG_API_KEY="your-cborg-key"

# Use CBORG proxy with convenient shortcut
deep-research-client research "Quantum computing advances" --use-cborg

# Run without installation using uvx
uvx deep-research-client research "AI ethics" --use-cborg
```

The `--use-cborg` flag automatically:
- Sets the base URL to `https://api.cborg.lbl.gov`
- Uses the `CBORG_API_KEY` environment variable
- Maintains compatibility with all OpenAI models

### Custom OpenAI-Compatible Endpoints

Use any OpenAI-compatible service with `--base-url`:

```bash
# Azure OpenAI
deep-research-client research "AI trends" \
  --base-url https://your-resource.openai.azure.com \
  --api-key-env AZURE_OPENAI_KEY

# LiteLLM proxy (local or hosted)
deep-research-client research "ML developments" \
  --base-url http://localhost:4000 \
  --api-key-env LITELLM_API_KEY

# OpenRouter
deep-research-client research "Technology review" \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY

# Custom proxy with default OPENAI_API_KEY
deep-research-client research "Research query" \
  --base-url https://api.example.com
```

### Proxy Configuration Options

| Option | Description | Example |
|--------|-------------|---------|
| `--use-cborg` | Use CBORG proxy (shortcut) | `--use-cborg` |
| `--base-url <url>` | Custom API endpoint URL | `--base-url https://api.example.com` |
| `--api-key-env <name>` | Environment variable for API key | `--api-key-env CBORG_API_KEY` |

### Python API with Proxies

```python
import os
from deep_research_client import DeepResearchClient, ProviderConfig

# CBORG configuration
cborg_config = {
    "openai": ProviderConfig(
        name="openai",
        api_key=os.getenv("CBORG_API_KEY"),
        base_url="https://api.cborg.lbl.gov",
        enabled=True
    )
}

client = DeepResearchClient(provider_configs=cborg_config)
result = client.research("Latest AI developments")

# Custom endpoint (e.g., Azure OpenAI)
azure_config = {
    "openai": ProviderConfig(
        name="openai",
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        base_url="https://your-resource.openai.azure.com",
        enabled=True
    )
}

client = DeepResearchClient(provider_configs=azure_config)
result = client.research("AI ethics research")
```

### Common Proxy Use Cases

**Institutional Access (CBORG)**
```bash
# Keep personal OpenAI key for direct access
export OPENAI_API_KEY="sk-personal-key"

# Use CBORG for institutional billing
export CBORG_API_KEY="cborg-institutional-key"
deep-research-client research "Scientific topic" --use-cborg
```

**LiteLLM Multi-Provider Gateway**
```bash
# Run LiteLLM proxy locally
litellm --port 4000

# Use it with deep-research-client
export LITELLM_API_KEY="your-key"
deep-research-client research "Research query" \
  --base-url http://localhost:4000 \
  --api-key-env LITELLM_API_KEY
```

**Azure OpenAI Deployment**
```bash
export AZURE_OPENAI_KEY="your-azure-key"
deep-research-client research "Enterprise research" \
  --base-url https://your-resource.openai.azure.com \
  --api-key-env AZURE_OPENAI_KEY
```

### Benefits of Proxy Usage

- **Institutional Billing**: Route through services like CBORG for cost management
- **Compliance**: Meet data residency and privacy requirements
- **Multi-Provider**: Use LiteLLM to access multiple LLM providers through one interface
- **Fallback**: Implement automatic provider fallback for reliability
- **Cost Control**: Add budget limits and rate limiting through gateway services

## Caching

The client uses intelligent file-based caching to avoid expensive re-queries:

- **Location**: Caches stored in `~/.deep_research_cache/` directory (user's home)
- **Persistence**: Cache files never expire (permanent until manually cleared)
- **Human-Readable Names**: Files named like `openai-what-is-crispr-gene-a1b2c3d4.json`
- **Uniqueness**: SHA256 hash suffix ensures no collisions
- **Management**:
  - `deep-research-client clear-cache` - Remove all cached files
  - `deep-research-client list-cache` - Show all cached files
  - `--no-cache` CLI flag to bypass cache for single query

### Cache File Structure
```
~/.deep_research_cache/
├── openai-what-is-crispr-gene-a1b2c3d4.json
├── falcon-machine-learning-trends-f6e5d4c3.json
├── openai-quantum-computing-basics-b8f7e2a9.json
└── ...
```

Each cache file contains:
```json
{
  "markdown": "# Research Report\n...",
  "citations": ["Citation 1", "Citation 2"],
  "provider": "openai",
  "query": "original query text"
}
```

## Templates

Use template files with variable substitution for reusable research queries. The client supports both **simple f-string templates** and **powerful Jinja2 templates** with conditionals, loops, and filters.

### Template Formats

#### F-String Templates (Simple)

Create a template file with `{variable}` placeholders:

```markdown
# Gene Research Template

Please research the gene {gene} in {organism}, focusing on:

1. Function and molecular mechanisms
2. Disease associations
3. Expression patterns in {tissue} tissue
4. Recent discoveries since {year}

Gene: {gene}
Organism: {organism}
Tissue: {tissue}
Year: {year}
```

#### Jinja2 Templates (Advanced)

For more complex templates with conditionals, loops, and logic, use Jinja2 syntax with `{{variable}}` and save with `.j2` extension:

```jinja
# Gene Research: {{gene}}

Research the {{gene}} gene in {{organism}}.

{% if detail_level == "high" %}
Provide comprehensive information including:
- Detailed molecular mechanisms
- All known disease associations
- Complete expression atlas
- Evolutionary conservation
{% else %}
Provide a concise overview covering:
- Primary function
- Major disease links
- Key expression patterns
{% endif %}

Focus areas:
{% for topic in topics %}
- {{topic|capitalize}}
{% endfor %}

{% if year %}
Emphasize discoveries from {{year}} onwards.
{% endif %}
```

### Format Detection

Templates are automatically detected as Jinja2 based on:

1. **File extension**: `.j2`, `.jinja`, `.jinja2` (e.g., `template.md.j2`)
2. **Frontmatter**: YAML frontmatter with `format: jinja` in `.md` files
3. **Default**: F-string format for backward compatibility

Example with frontmatter:
```markdown
---
format: jinja
description: Advanced gene research
---
Research {{gene}}{% if organism %} in {{organism}}{% endif %}.
```

### Using Templates

```bash
# Simple f-string template
deep-research-client research \
  --template gene_research.md \
  --var "gene=TP53" \
  --var "organism=human" \
  --var "tissue=brain" \
  --var "year=2020"

# Jinja2 template with conditionals
deep-research-client research \
  --template gene_advanced.md.j2 \
  --var "gene=BRCA1" \
  --var "organism=human" \
  --var "detail_level=high"

# With uvx (no installation)
uvx deep-research-client research \
  --template gene_research.md \
  --var "gene=TP53" \
  --var "organism=human"

# Missing variables will show helpful error
deep-research-client research --template gene_research.md
# Error: Template requires variables: gene, organism, tissue, year
# Use --var key=value for each variable
```

### Template Examples

The repository includes example templates:
- `templates/gene_family.md` - Simple f-string template for gene families
- `templates/gene_jinja.md.j2` - Jinja2 template with conditionals
- `templates/gene_frontmatter.md` - Markdown with frontmatter-based format detection

### Python API with Templates

```python
from deep_research_client import DeepResearchClient
from deep_research_client.processing import TemplateProcessor
from pathlib import Path

# Initialize processor (auto-detects format)
template_processor = TemplateProcessor()

# Simple f-string template
query = template_processor.render_template(
    Path("gene_research.md"),
    {"gene": "TP53", "organism": "human", "tissue": "brain", "year": "2020"}
)

# Jinja2 template with advanced features
query = template_processor.render_template(
    Path("gene_advanced.md.j2"),
    {
        "gene": "BRCA1",
        "organism": "human",
        "detail_level": "high",
        "topics": ["function", "mutations", "expression"],
        "year": "2020"
    }
)

# Process template with metadata (includes format info)
rendered_query, metadata = template_processor.process_template(
    Path("gene_research.md.j2"),
    {"gene": "TP53", "organism": "human"}
)
print(f"Format used: {metadata['template_format']}")  # 'jinja' or 'fstring'

# Use with client
client = DeepResearchClient()
result = client.research(query)
```

### Jinja2 Features

Jinja2 templates support powerful features:

- **Conditionals**: `{% if condition %}...{% endif %}`
- **Loops**: `{% for item in list %}...{% endfor %}`
- **Filters**: `{{gene|upper}}`, `{{organism|capitalize}}`
- **Default values**: `{{year|default('2024')}}`
- **Comments**: `{# This is a comment #}`

See [docs/templates.md](docs/templates.md) for comprehensive template documentation.

## Output Format

The client outputs structured markdown with YAML frontmatter containing metadata:

```yaml
---
provider: perplexity
model: sonar-pro
cached: false
start_time: '2025-10-18T17:43:49.437056'
end_time: '2025-10-18T17:44:08.922200'
duration_seconds: 19.49
template_file: examples/gene_research.md.j2  # If template used
template_format: jinja                        # Template format: 'jinja' or 'fstring'
template_variables:                           # If template used
  gene: TP53
  organism: human
  detail_level: high
provider_config:
  timeout: 600
  max_retries: 3
citation_count: 18
---

## Question

What is machine learning?

## Output

**Machine learning** is a branch of artificial intelligence...

## Citations

1. https://www.ibm.com/topics/machine-learning
2. https://www.coursera.org/articles/what-is-machine-learning
...
```

### Citation Handling

- **Default**: Citations included at end of markdown output
- **Separate file**: Use `--separate-citations` to save citations to `.citations.md` file
- **Library usage**: Access citations via `result.citations` list

## Model Discovery & Introspection

The client includes comprehensive model discovery capabilities to help you find the right model for your needs.

### Listing Available Models

```bash
# List all available models across all providers
deep-research-client models

# Show models for a specific provider
deep-research-client models --provider openai
deep-research-client models --provider perplexity

# Filter by cost level
deep-research-client models --cost low      # Budget-friendly options
deep-research-client models --cost medium   # Balanced options
deep-research-client models --cost high     # Premium options
deep-research-client models --cost very_high  # Most expensive/comprehensive

# Filter by capability
deep-research-client models --capability web_search
deep-research-client models --capability academic_search
deep-research-client models --capability scientific_literature

# Show detailed information including pricing, use cases, and limitations
deep-research-client models --detailed

# Combine filters
deep-research-client models --provider perplexity --cost low --detailed
```

### Model Aliases

Most models support convenient short aliases:

```bash
# These are equivalent
deep-research-client research "AI trends" --provider openai --model o3-deep-research-2025-06-26
deep-research-client research "AI trends" --provider openai --model o3
deep-research-client research "AI trends" --provider openai --model o3-deep

# Perplexity aliases
--model sonar-deep-research  # Can use: deep, deep-research, sdr
--model sonar-pro            # Can use: pro, sp
--model sonar                # Can use: basic, fast, s

# Using uvx with aliases
uvx deep-research-client research "AI" --provider openai --model o3
```

### Model Selection

Use `--model` to override the default:

```bash
# Use faster Perplexity model
deep-research-client research "AI trends" --provider perplexity --model sonar-pro

# Use default deep research model (comprehensive but slower)
deep-research-client research "AI trends" --provider perplexity  # Uses sonar-deep-research

# Override OpenAI model using alias
deep-research-client research "AI trends" --provider openai --model o4-mini
```

### Available Models by Provider

| Provider | Model | Aliases | Cost | Speed | Context | Key Features |
|----------|-------|---------|------|-------|---------|--------------|
| **OpenAI** | `o3-deep-research-2025-06-26` | o3, o3-deep, o3dr | 💰💰💰💰 | 🐢 | 128K | Most comprehensive, web search, code analysis |
| **OpenAI** | `o4-mini-deep-research-2025-06-26` | o4m, o4-mini, mini | 💰💰 | ⏳ | 128K | Balanced speed/quality |
| **Perplexity** | `sonar-deep-research` | deep, deep-research, sdr | 💰💰💰 | 🐌 | 200K | Comprehensive with real-time data |
| **Perplexity** | `sonar-pro` | pro, sp | 💰💰 | ⏳ | 200K | Fast with good quality |
| **Perplexity** | `sonar` | basic, fast, s | 💰 | ⚡ | 100K | Fastest, budget-friendly |
| **Edison** | `Edison Scientific Literature` | falcon, edison, eds, science | 💰💰💰 | 🐌 | - | Scientific literature focus |
| **Consensus** | `Consensus Academic Search` | consensus, academic, papers, c | 💰 | ⚡ | - | Peer-reviewed papers only |

**Cost Legend:** 💰 = Low, 💰💰 = Medium, 💰💰💰 = High, 💰💰💰💰 = Very High
**Speed Legend:** ⚡ = Fast, ⏳ = Medium, 🐌 = Slow, 🐢 = Very Slow

### Model Capabilities

Each model has different capabilities:

| Capability | Models |
|------------|--------|
| **Web Search** | OpenAI (o3, o4-mini), Perplexity (all) |
| **Academic Search** | Edison, Consensus |
| **Scientific Literature** | Edison |
| **Real-time Data** | OpenAI (all), Perplexity (all) |
| **Citation Tracking** | All providers |
| **Code Interpretation** | OpenAI (o3) |

### Python API for Model Discovery

```python
from deep_research_client.model_cards import (
    get_provider_model_cards,
    list_all_models,
    find_models_by_cost,
    find_models_by_capability,
    resolve_model_alias,
    CostLevel,
    ModelCapability
)

# Get all models for a provider
openai_cards = get_provider_model_cards("openai")
models = openai_cards.list_models()
print(models)  # ['o3-deep-research-2025-06-26', 'o4-mini-deep-research-2025-06-26']

# Get detailed model information
card = openai_cards.get_model_card("o3-deep-research-2025-06-26")
print(f"Cost: {card.cost_level}")
print(f"Speed: {card.time_estimate}")
print(f"Capabilities: {card.capabilities}")
print(f"Context Window: {card.context_window}")
print(f"Use Cases: {card.use_cases}")
print(f"Limitations: {card.limitations}")

# Find models across all providers by criteria
cheap_models = find_models_by_cost(CostLevel.LOW)
# Returns: {'perplexity': ['sonar'], 'consensus': ['Consensus Academic Search']}

web_search_models = find_models_by_capability(ModelCapability.WEB_SEARCH)
# Returns: {'openai': ['o3-deep-research-2025-06-26', ...], 'perplexity': [...]}

# Resolve aliases to full model names
full_name = resolve_model_alias("openai", "o3")
print(full_name)  # 'o3-deep-research-2025-06-26'

# List all models across all providers
all_models = list_all_models()
# Returns: {'openai': [...], 'perplexity': [...], 'falcon': [...], ...}

# Get models by speed
fast_models = openai_cards.get_models_by_time(TimeEstimate.FAST)
```

### Model Selection Guidelines

**For comprehensive research:**
- OpenAI o3 (most thorough, expensive, slow)
- Perplexity sonar-deep-research (thorough with real-time data)

**For balanced performance:**
- OpenAI o4-mini (good quality, reasonable cost)
- Perplexity sonar-pro (fast with good coverage)

**For quick lookups:**
- Perplexity sonar (fastest, cheapest)

**For academic/scientific research:**
- Edison (scientific literature focused, powered by PaperQA3)
- Consensus (peer-reviewed papers only)

**Budget considerations:**
```bash
# Most expensive but most comprehensive
deep-research-client research "complex topic" --provider openai --model o3

# Good balance of cost and quality
deep-research-client research "complex topic" --provider perplexity --model pro

# Cheapest option for quick answers
deep-research-client research "simple question" --provider perplexity --model sonar
```

## Supported Providers

| Provider | Environment Variable | Model/Service | Strengths |
|----------|---------------------|---------------|-----------|
| OpenAI | `OPENAI_API_KEY` | o3-deep-research-2025-06-26 | Deep research, comprehensive reports |
| Edison | `EDISON_API_KEY` | Edison Scientific Literature | Scientific literature focus, powered by PaperQA3 |
| Perplexity | `PERPLEXITY_API_KEY` | sonar-deep-research | Real-time web search, recent sources |
| Consensus | `CONSENSUS_API_KEY` | Consensus Academic Search | Peer-reviewed academic papers, evidence-based research |

## Development

### Essential Commands

```bash
# Run tests
just test

# Type checking
just mypy

# Format code
just format

# Install dependencies
uv sync

# Run CLI locally
uv run deep-research-client --help
```

### Adding New Providers

1. Create a new provider class in `src/deep_research_client/providers/`
2. Inherit from `ResearchProvider`
3. Implement the `research()` method
4. Register in `client.py`

Example:
```python
from . import ResearchProvider
from ..models import ResearchResult

class NewProvider(ResearchProvider):
    async def research(self, query: str) -> ResearchResult:
        # Your implementation
        return ResearchResult(...)
```

## Repository Structure

* [docs/](docs/) - mkdocs documentation
* [src/deep_research_client/](src/deep_research_client/) - main package
  * [client.py](src/deep_research_client/client.py) - main client class
  * [models.py](src/deep_research_client/models.py) - Pydantic models
  * [cache.py](src/deep_research_client/cache.py) - caching implementation
  * [providers/](src/deep_research_client/providers/) - research providers
* [tests/](tests/) - test suite

## Potential Future Providers

The following providers were researched but not yet implemented:

### Research-Focused Providers
- **Semantic Scholar API** - Free access to 200+ million academic papers with comprehensive metadata
- **Elicit AI** - AI research assistant focused on answering research questions
- **Scite.ai** - Citation-based research with smart citations and paper analysis

### General AI Providers with Research Capabilities
- **Anthropic Claude** - Has web search API but lacks dedicated deep research mode
- **Google Gemini** - Deep Research feature exists but requires allowlist API access
- **Exa (Metaphor)** - AI-powered search API designed for research workflows

### Specialized Tools
- **Qatalog** - Enterprise knowledge search and research
- **Azure AI Search** - Microsoft's cognitive search capabilities
- **IBM Watson Discovery** - Enterprise document analysis and research

These providers could be added based on user demand and API availability. Contributions welcome!

## Claude Code Skill

This repository includes a Claude Code skill that makes it easy to use deep-research-client directly from Claude Code.

### Installation

To use the skill:

```bash
# Copy to your local Claude skills directory
cp -r .claude/skills/run-deep-research ~/.claude/skills/

# Or use it project-wide (already available in this repo)
# Claude Code will auto-detect skills in .claude/skills/
```

### Features

The skill provides:
- **Guided research workflow**: Automatically asks about speed vs depth preferences
- **Provider selection**: Helps choose the right provider and model for your needs
- **Template support**: Easy access to research templates for common patterns
- **Smart defaults**: Leverages caching and best practices automatically

### Usage with Claude Code

Once installed, simply ask Claude to research topics:

```
"Research the latest developments in CRISPR gene editing"
"Do a quick overview of quantum computing (fast approach)"
"I need a comprehensive analysis of TP53 gene in humans"
```

Claude will automatically:
1. Ask whether you want a fast/light or comprehensive/slow approach
2. Select the appropriate provider and model
3. Execute the research with proper caching
4. Save results with citations and metadata

See [.claude/skills/README.md](.claude/skills/README.md) for more details.

## License

BSD-3-Clause
