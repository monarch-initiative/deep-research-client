# How to Use Proxies

Route requests through proxy services like CBORG, LiteLLM, or Azure OpenAI.

## CBORG (Berkeley Lab)

CBORG provides institutional access with cost management.

### Setup

```bash
export CBORG_API_KEY="your-cborg-key"
```

### Usage

```bash
deep-research-client research "Quantum computing advances" --use-cborg
```

The `--use-cborg` flag automatically:

- Sets base URL to `https://api.cborg.lbl.gov`
- Uses `CBORG_API_KEY` environment variable

## Custom OpenAI-Compatible Endpoints

Use any OpenAI-compatible API with `--base-url`:

### Azure OpenAI

```bash
export AZURE_OPENAI_KEY="your-azure-key"

deep-research-client research "AI trends" \
  --base-url https://your-resource.openai.azure.com \
  --api-key-env AZURE_OPENAI_KEY
```

### LiteLLM Proxy

```bash
# Start LiteLLM locally
litellm --port 4000

# Use it
export LITELLM_API_KEY="your-key"
deep-research-client research "ML developments" \
  --base-url http://localhost:4000 \
  --api-key-env LITELLM_API_KEY
```

### OpenRouter

```bash
export OPENROUTER_API_KEY="your-key"

deep-research-client research "Technology review" \
  --base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY
```

## Python API

Configure proxies programmatically:

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
```

## Options Reference

| Option | Description |
|--------|-------------|
| `--use-cborg` | Use CBORG proxy (shortcut) |
| `--base-url <url>` | Custom API endpoint URL |
| `--api-key-env <name>` | Environment variable name for API key |

## Common Use Cases

### Institutional Billing

Keep personal key for direct access, use CBORG for institutional work:

```bash
# Personal
export OPENAI_API_KEY="sk-personal-key"
deep-research-client research "Personal project"

# Institutional
export CBORG_API_KEY="cborg-institutional-key"
deep-research-client research "Lab project" --use-cborg
```

### Multi-Provider Gateway

Use LiteLLM to access multiple providers through one interface:

```bash
litellm --port 4000

deep-research-client research "Research query" \
  --base-url http://localhost:4000 \
  --api-key-env LITELLM_API_KEY
```

## Benefits

- **Institutional Billing**: Track costs through CBORG or similar
- **Compliance**: Meet data residency requirements
- **Fallback**: Use gateways for automatic provider failover
- **Budget Control**: Add rate limiting through proxy services
