# deep-research-client

A CLI tool for AI-powered deep research across multiple providers.

## Quick Start

```bash
# Install
pip install deep-research-client

# Set up an API key (e.g., Perplexity)
export PERPLEXITY_API_KEY="your-key"

# Run a research query
deep-research-client research "Research the human CFAP300 gene including molecular function, disease associations, and evolutionary conservation"
```

Or run without installation:

```bash
uvx deep-research-client research "Analyze recent advances in solid-state battery technology for electric vehicles, focusing on materials science breakthroughs since 2023"
```

## Features

- **Multiple Providers**: OpenAI, Perplexity, Edison, Consensus, Cyberian
- **Smart Caching**: Avoid expensive re-queries
- **Templates**: Create reusable research queries with variables
- **Model Selection**: Choose speed vs comprehensiveness
- **Citations**: All results include source references

## Supported Providers

| Provider | Best For | Speed |
|----------|----------|-------|
| [OpenAI](reference/providers.md#openai-deep-research) | Comprehensive reports | Slow |
| [Perplexity](reference/providers.md#perplexity-ai) | General web research | Fast-Slow |
| [Edison](reference/providers.md#edison-scientific-falcon) | Scientific literature | Slow |
| [Consensus](reference/providers.md#consensus) | Academic papers | Fast |
| [Cyberian](reference/providers.md#cyberian-agent-based) | Agent-based deep research | Very Slow |

## Documentation

This documentation follows the [Diátaxis](https://diataxis.fr/) framework:

### Tutorials (Learning)

Step-by-step guides for getting started:

- [Getting Started](tutorials/getting-started.md) - Installation and setup
- [Your First Query](tutorials/first-query.md) - Run your first research

### How-to Guides (Tasks)

Solve specific problems:

- [Choose a Provider](how-to/choose-provider.md) - Pick the right provider
- [Use Templates](how-to/templates.md) - Create reusable queries
- [Manage Cache](how-to/cache.md) - Control caching behavior
- [Use Proxies](how-to/proxies.md) - Route through CBORG or custom endpoints

### Reference (Information)

Technical details:

- [CLI Reference](reference/cli.md) - All commands and options
- [Providers](reference/providers.md) - Provider specifications
- [Models](reference/models.md) - Available models
- [Configuration](reference/configuration.md) - All config options

### Explanation (Understanding)

Background and concepts:

- [Provider Comparison](explanation/providers.md) - How providers differ
- [Architecture](explanation/architecture.md) - How the tool works

## Common Commands

```bash
# Research with default provider
deep-research-client research "Review the current evidence on microplastics in human tissue and potential health implications"

# Use specific provider for scientific literature
deep-research-client research "Summarize mechanisms of ferroptosis in cancer therapy" --provider edison

# Save comprehensive report to file
deep-research-client research "Analyze the state of nuclear fusion research including ITER progress and private sector developments" --output fusion-report.md

# List available providers
deep-research-client providers

# List available models
deep-research-client models

# Check cache
deep-research-client list-cache
```

## Links

- [GitHub Repository](https://github.com/ai4curation/deep-research-client)
- [PyPI Package](https://pypi.org/project/deep-research-client/)
