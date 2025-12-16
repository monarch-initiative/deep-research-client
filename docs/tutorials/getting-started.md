# Getting Started

This tutorial walks you through installing deep-research-client and setting up your first provider.

## What You'll Learn

- How to install the CLI tool
- How to configure API keys
- How to verify your setup

## Prerequisites

- Python 3.10 or later
- An API key from at least one provider (see [Providers](../reference/providers.md))

## Installation

### Option 1: Install with pip (Recommended)

```bash
pip install deep-research-client
```

### Option 2: Run with uvx (No Installation)

If you just want to try it out without installing:

```bash
uvx deep-research-client --help
```

### Option 3: Add to a uv Project

```bash
uv add deep-research-client
```

### Option 4: Development Installation

```bash
git clone https://github.com/ai4curation/deep-research-client
cd deep-research-client
uv sync
```

## Configure Your API Keys

Set up at least one provider. You only need the providers you plan to use.

### Perplexity (Easiest to Start)

```bash
export PERPLEXITY_API_KEY="your-perplexity-key"
```

Get your key at: [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)

### OpenAI Deep Research

```bash
export OPENAI_API_KEY="your-openai-key"
```

Get your key at: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### Edison Scientific (Formerly Falcon)

```bash
export EDISON_API_KEY="your-edison-key"
```

### Consensus (Academic Papers)

```bash
export CONSENSUS_API_KEY="your-consensus-key"
```

Note: Consensus requires application approval.

## Verify Your Setup

Check which providers are available:

```bash
deep-research-client providers
```

You should see your configured provider(s) listed with a checkmark.

## Next Steps

Now that you're set up, continue to [Your First Query](first-query.md) to run your first research request.
