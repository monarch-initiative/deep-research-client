# Understanding Providers

This page explains how different providers work and when to use each one.

## What is a "Deep Research" Provider?

Traditional AI chatbots answer questions from their training data. Deep research providers go further:

1. **Search the web** for current information
2. **Read and synthesize** multiple sources
3. **Provide citations** for their claims
4. **Generate comprehensive reports** rather than brief answers

## How Providers Differ

### Data Sources

| Provider | Primary Sources |
|----------|-----------------|
| OpenAI | Web search, multiple domains |
| Perplexity | Real-time web, news, forums |
| Edison | Scientific papers (via PaperQA3) |
| Consensus | Peer-reviewed academic papers only |

### Research Depth

**Shallow (seconds):** Quick web search, summarize top results

- Perplexity sonar
- Consensus

**Medium (30s - 2min):** More thorough search, cross-reference sources

- Perplexity sonar-pro

**Deep (2-15min):** Extensive research, iterative refinement, comprehensive synthesis

- OpenAI o3
- Perplexity sonar-deep-research
- Edison

### Citation Quality

**Best for academic citations:**

- Consensus - only peer-reviewed papers
- Edison - scientific literature focus

**Best for web citations:**

- Perplexity - good source attribution
- OpenAI - comprehensive but less consistent formatting

## OpenAI Deep Research

OpenAI's approach uses their o3/o4 models with web browsing capabilities.

**How it works:**

1. Formulates search queries based on your question
2. Browses multiple web pages
3. Synthesizes information across sources
4. Generates a comprehensive report

**Strengths:**

- Most thorough analysis
- Can interpret code and data
- Excellent synthesis across diverse topics

**Limitations:**

- Slowest option (5-15 minutes)
- Most expensive
- May have access restrictions on some sites

## Perplexity AI

Perplexity combines LLM reasoning with real-time web search.

**How it works:**

1. Parses your query into search intents
2. Performs web searches
3. Reads and extracts relevant content
4. Synthesizes with source attribution

**Three tiers:**

- **sonar**: Fast, basic web search
- **sonar-pro**: More thorough, better reasoning
- **sonar-deep-research**: Extensive multi-step research

**Strengths:**

- Multiple speed/quality options
- Good at current events
- Transparent citations

**Limitations:**

- Deep research mode can be slow
- Quality varies by topic

## Edison Scientific (Falcon)

Edison (formerly FutureHouse Falcon) specializes in scientific literature.

**How it works:**

1. Powered by PaperQA3 - an AI system designed for scientific papers
2. Searches scientific databases
3. Reads full paper content (not just abstracts)
4. Synthesizes findings with proper academic citations

**Strengths:**

- Deep scientific literature coverage
- Full paper analysis, not just abstracts
- Good for biomedical and technical research

**Limitations:**

- Slower than general web search
- Less useful for non-scientific topics
- May miss very recent papers

## Consensus

Consensus focuses exclusively on peer-reviewed research.

**How it works:**

1. Searches a curated database of academic papers
2. Extracts findings relevant to your question
3. Provides evidence-based summaries
4. Only cites peer-reviewed sources

**Strengths:**

- Only peer-reviewed sources
- Good for "does X cause Y" type questions
- Fast responses

**Limitations:**

- Limited to their indexed papers
- Not comprehensive for obscure topics
- Can't access recent papers quickly

## Combining Providers

For thorough research, use multiple providers:

```bash
# Quick overview
deep-research-client research "topic" --provider perplexity --model sonar

# Scientific depth
deep-research-client research "topic" --provider edison

# Academic consensus
deep-research-client research "topic" --provider consensus

# Comprehensive synthesis
deep-research-client research "topic" --provider openai
```

## Cost vs Quality Trade-offs

The relationship between cost and quality isn't always linear:

**Best value for general queries:**
Perplexity sonar-pro - good quality at moderate cost

**Best value for academic queries:**
Consensus - low cost, high-quality academic sources

**When to pay premium:**
Use OpenAI o3 when you need the most thorough analysis and can wait 10+ minutes

## Provider Selection Flowchart

```
Is it about scientific research?
├── Yes → Is it about peer-reviewed evidence?
│   ├── Yes → Consensus
│   └── No → Edison
└── No → How much time do you have?
    ├── Seconds → Perplexity sonar
    ├── Minutes → Perplexity sonar-pro
    └── 10+ minutes → OpenAI o3
```
