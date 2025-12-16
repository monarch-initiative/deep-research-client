# How to Choose a Provider

This guide helps you select the right provider for your research needs.

## Quick Decision Guide

| Need | Best Provider | Command |
|------|---------------|---------|
| Quick summary | Perplexity (sonar) | `--provider perplexity --model sonar` |
| General research | Perplexity (sonar-pro) | `--provider perplexity --model pro` |
| Comprehensive report | OpenAI (o3) | `--provider openai` |
| Scientific literature | Edison | `--provider edison` |
| Academic papers only | Consensus | `--provider consensus` |

## Provider Comparison

### Perplexity AI

Best for: **General web research with real-time data**

```bash
# Lighter research task
deep-research-client research "Summarize recent developments in room-temperature superconductor claims" \
  --provider perplexity --model sonar

# Comprehensive web research
deep-research-client research "Analyze the current landscape of AI chip startups and their differentiation strategies" \
  --provider perplexity --model sonar-deep-research
```

Strengths:

- Real-time web search
- Multiple speed/quality options
- Good citation quality
- Reasonable pricing

### OpenAI Deep Research

Best for: **Comprehensive, in-depth research reports**

```bash
deep-research-client research "Comprehensive analysis of CRISPR base editing versus prime editing: mechanisms, applications, limitations, and recent clinical developments" \
  --provider openai
```

Strengths:

- Most thorough analysis
- Excellent synthesis
- Code interpretation capability

Trade-offs:

- Slowest (can take several minutes)
- Most expensive

### Edison Scientific (Formerly Falcon)

Best for: **Scientific literature review**

```bash
deep-research-client research "Review molecular mechanisms of autophagy regulation and its role in cancer progression" \
  --provider edison
```

Strengths:

- Powered by PaperQA3
- Focus on peer-reviewed papers
- Good for biomedical research

### Consensus

Best for: **Academic paper search**

```bash
deep-research-client research "Synthesize clinical evidence on intermittent fasting effects on metabolic health markers" \
  --provider consensus
```

Strengths:

- Only peer-reviewed sources
- Evidence-based summaries
- Good for scientific claims

## Budget Considerations

From cheapest to most expensive:

1. **Perplexity sonar** - Best value for quick queries
2. **Consensus** - Good value for academic research
3. **Perplexity sonar-pro** - Mid-range, good quality
4. **Edison** - Mid-range, scientific focus
5. **Perplexity sonar-deep-research** - Higher cost, comprehensive
6. **OpenAI o4-mini** - Higher cost, good quality
7. **OpenAI o3** - Highest cost, most comprehensive

Example for budget-conscious usage:

```bash
# Use cheapest for initial exploration
deep-research-client research "Overview of CAR-T cell therapy landscape" \
  --provider perplexity --model sonar

# Then use comprehensive for final deep dive
deep-research-client research "Detailed analysis of CAR-T manufacturing challenges and novel approaches to reduce vein-to-vein time" \
  --provider openai
```

## Speed Considerations

From fastest to slowest:

1. **Perplexity sonar** - Seconds
2. **Consensus** - Seconds
3. **Perplexity sonar-pro** - ~30 seconds
4. **Perplexity sonar-deep-research** - 1-3 minutes
5. **Edison** - 2-5 minutes
6. **OpenAI o4-mini** - 2-5 minutes
7. **OpenAI o3** - 5-15 minutes

## Use Multiple Providers

For thorough research, combine providers:

```bash
# Quick landscape overview
deep-research-client research "Overview of alpha-synuclein aggregation in Parkinson's disease" \
  --provider perplexity --model sonar \
  --output synuclein-overview.md

# Scientific literature deep-dive
deep-research-client research "Molecular mechanisms of alpha-synuclein fibril formation and propagation" \
  --provider edison \
  --output synuclein-mechanisms.md

# Clinical evidence synthesis
deep-research-client research "Clinical trial evidence for alpha-synuclein targeting therapies" \
  --provider consensus \
  --output synuclein-trials.md
```

## Check Available Models

List all available models and their characteristics:

```bash
# All models
deep-research-client models

# Filter by provider
deep-research-client models --provider perplexity

# Show detailed info
deep-research-client models --detailed

# Filter by cost
deep-research-client models --cost low
```

## See Also

- [Provider Reference](../reference/providers.md) - Full provider documentation
- [Model Reference](../reference/models.md) - Complete model specifications
