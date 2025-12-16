# Your First Query

This tutorial walks you through running your first research query and understanding the output.

## What You'll Learn

- How to run a basic research query
- How to read the output format
- How to save results to a file

## Prerequisites

Complete [Getting Started](getting-started.md) first.

## Run a Research Query

Let's research a topic in depth:

```bash
deep-research-client research "Research the human CFAP300 gene including molecular function, disease associations, and evolutionary conservation"
```

Or using uvx without installation:

```bash
uvx deep-research-client research "Analyze recent clinical trial evidence for SGLT2 inhibitors in heart failure treatment"
```

## Understanding the Output

The output is Markdown with YAML frontmatter containing metadata:

```yaml
---
provider: perplexity
model: sonar-deep-research
cached: false
start_time: '2025-01-15T10:30:00.000000'
end_time: '2025-01-15T10:30:45.000000'
duration_seconds: 45.0
citation_count: 12
---

## Question

Research the human CFAP300 gene including molecular function, disease associations, and evolutionary conservation

## Output

CFAP300 (Cilia and Flagella Associated Protein 300) is a protein-coding gene...

## Citations

1. https://www.ncbi.nlm.nih.gov/gene/85016
2. https://www.uniprot.org/uniprotkb/...
```

Key fields:

- **provider**: Which AI service answered your query
- **model**: The specific model used
- **cached**: Whether this came from cache (saves time/money on repeat queries)
- **duration_seconds**: How long the query took
- **citation_count**: Number of sources cited

## Save to a File

Save the research report:

```bash
deep-research-client research "Review mechanisms of resistance to checkpoint inhibitor immunotherapy in solid tumors" --output immunotherapy-resistance.md
```

Save citations to a separate file:

```bash
deep-research-client research "Analyze the role of gut microbiome in neurodegenerative diseases" \
  --output microbiome-neurodegeneration.md \
  --separate-citations
```

This creates:

- `microbiome-neurodegeneration.md` - The main report
- `microbiome-neurodegeneration.citations.md` - Just the citations

## Choose a Different Provider

Use a specific provider:

```bash
# Use Perplexity for web-based research
deep-research-client research "Survey current approaches to large language model alignment and safety" --provider perplexity

# Use OpenAI for comprehensive synthesis
deep-research-client research "Comprehensive review of mRNA vaccine technology developments beyond COVID-19" --provider openai
```

## Use a Faster Model

For quicker results when deep research isn't needed:

```bash
# Fast Perplexity model for lighter queries
deep-research-client research "Summarize recent papers on protein language models" \
  --provider perplexity \
  --model sonar
```

## Check the Cache

If you run the same query twice, the second time is instant:

```bash
# First run - calls the API (may take minutes for deep research)
deep-research-client research "Review evidence for ketogenic diet in epilepsy treatment"

# Second run - returns cached result immediately
deep-research-client research "Review evidence for ketogenic diet in epilepsy treatment"
```

List your cached queries:

```bash
deep-research-client list-cache
```

## Next Steps

- [Choose a Provider](../how-to/choose-provider.md) - Pick the right provider for your needs
- [Use Templates](../how-to/templates.md) - Create reusable research queries
- [CLI Reference](../reference/cli.md) - Full command documentation
