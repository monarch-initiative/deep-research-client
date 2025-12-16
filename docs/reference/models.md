# Model Reference

Complete reference for all available models.

## Model Overview

| Provider | Model | Aliases | Cost | Speed | Context |
|----------|-------|---------|------|-------|---------|
| OpenAI | o3-deep-research-2025-06-26 | o3, o3-deep | Very High | Very Slow | 128K |
| OpenAI | o4-mini-deep-research-2025-06-26 | o4m, o4-mini | High | Medium | 128K |
| Perplexity | sonar-deep-research | deep, sdr | High | Slow | 200K |
| Perplexity | sonar-pro | pro, sp | Medium | Medium | 200K |
| Perplexity | sonar | basic, fast, s | Low | Fast | 100K |
| Edison | Edison Scientific Literature | falcon, edison | High | Slow | - |
| Consensus | Consensus Academic Search | consensus, c | Low | Fast | - |

## Using Model Aliases

Aliases provide shortcuts:

```bash
# These are equivalent
deep-research-client research "AI" --provider openai --model o3-deep-research-2025-06-26
deep-research-client research "AI" --provider openai --model o3
```

## Model Capabilities

| Capability | Models |
|------------|--------|
| Web Search | OpenAI (all), Perplexity (all) |
| Academic Search | Edison, Consensus |
| Scientific Literature | Edison |
| Real-time Data | OpenAI (all), Perplexity (all) |
| Code Interpretation | OpenAI (o3) |
| Citation Tracking | All |

## Model Selection Guide

### By Use Case

| Use Case | Recommended Model |
|----------|-------------------|
| Quick lookup | Perplexity sonar |
| General research | Perplexity sonar-pro |
| Comprehensive report | OpenAI o3 |
| Scientific research | Edison |
| Academic citations | Consensus |
| Budget-friendly | Perplexity sonar, Consensus |

### By Budget

**Lowest cost:**
```bash
deep-research-client research "topic" --provider perplexity --model sonar
```

**Best value:**
```bash
deep-research-client research "topic" --provider perplexity --model pro
```

**Most comprehensive:**
```bash
deep-research-client research "topic" --provider openai --model o3
```

## Discovering Models

### List All Models

```bash
deep-research-client models
```

### Filter by Provider

```bash
deep-research-client models --provider perplexity
```

### Filter by Cost

```bash
deep-research-client models --cost low
deep-research-client models --cost medium
deep-research-client models --cost high
deep-research-client models --cost very_high
```

### Filter by Capability

```bash
deep-research-client models --capability web_search
deep-research-client models --capability academic_search
```

### Detailed Information

```bash
deep-research-client models --detailed
```

Shows:

- Pricing information
- Time estimates
- Use cases
- Limitations

## Python API

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

# Get models for a provider
cards = get_provider_model_cards("openai")
models = cards.list_models()

# Get model details
card = cards.get_model_card("o3-deep-research-2025-06-26")
print(card.cost_level)
print(card.time_estimate)
print(card.capabilities)

# Find models by criteria
cheap = find_models_by_cost(CostLevel.LOW)
web_search = find_models_by_capability(ModelCapability.WEB_SEARCH)

# Resolve alias
full_name = resolve_model_alias("openai", "o3")
# Returns: 'o3-deep-research-2025-06-26'
```

## Cost Levels

| Level | Description | Examples |
|-------|-------------|----------|
| LOW | Budget-friendly | sonar, Consensus |
| MEDIUM | Balanced | sonar-pro, o4-mini |
| HIGH | Premium | sonar-deep-research, Edison |
| VERY_HIGH | Most expensive | o3 |

## Time Estimates

| Level | Description | Typical Duration |
|-------|-------------|------------------|
| FAST | Quick responses | < 30 seconds |
| MEDIUM | Moderate wait | 30s - 2 minutes |
| SLOW | Longer processing | 2-5 minutes |
| VERY_SLOW | Extended research | 5-15+ minutes |
