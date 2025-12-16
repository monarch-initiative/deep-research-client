# CLI Reference

Complete reference for all `deep-research-client` commands.

## Global Options

```
--verbose, -v    Increase verbosity (-v, -vv, -vvv)
--help           Show help and exit
```

## Commands

### research

Perform deep research on a query.

```bash
deep-research-client research [OPTIONS] [QUERY]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `QUERY` | Research query or question (not needed if using `--template`) |

#### Options

| Option | Description |
|--------|-------------|
| `--provider TEXT` | Provider to use: openai, edison, perplexity, consensus, cyberian |
| `--model TEXT` | Model to use (overrides provider default) |
| `--output PATH` | Output file path (prints to stdout if not provided) |
| `--no-cache` | Disable caching for this query |
| `--separate-citations PATH` | Save citations to separate file |
| `--cache-dir PATH` | Override cache directory (default: `~/.deep_research_cache`) |
| `--template PATH` | Template file with variable placeholders |
| `--var TEXT` | Template variable as `key=value` (repeatable) |
| `--param TEXT` | Provider-specific parameter as `key=value` (repeatable) |
| `--base-url TEXT` | Custom base URL for API endpoint |
| `--use-cborg` | Use CBORG proxy (`api.cborg.lbl.gov`) |
| `--api-key-env TEXT` | Environment variable name for API key |
| `--title TEXT` | Title for the research report |
| `--abstract TEXT` | Abstract or summary for the research |
| `--keyword TEXT` | Keyword/tag for the research (repeatable) |
| `--author TEXT` | Primary author of the research |
| `--contributor TEXT` | Contributor to the research (repeatable) |

#### Examples

```bash
# Research a gene with comprehensive information
deep-research-client research "Research the human CFAP300 gene including molecular function, disease associations, and evolutionary conservation"

# Use specific provider and model for tech research
deep-research-client research "Analyze current approaches to federated learning for privacy-preserving machine learning" \
  --provider perplexity \
  --model sonar-pro

# Save comprehensive report to file
deep-research-client research "Review the evidence on long-term effects of COVID-19 on cardiovascular health" \
  --output long-covid-cardio.md

# Separate citations file
deep-research-client research "Survey ethical considerations in clinical AI deployment" \
  --output ai-ethics.md \
  --separate-citations

# Use template
deep-research-client research \
  --template template.md \
  --var "gene=TP53" \
  --var "organism=human"

# Provider-specific parameters
deep-research-client research "Medical research" \
  --provider perplexity \
  --param "reasoning_effort=high" \
  --param "search_recency_filter=week"

# Skip cache
deep-research-client research "Current events" --no-cache

# Use CBORG proxy
deep-research-client research "Quantum computing" --use-cborg

# Add publication-style metadata
deep-research-client research "CFAP300 gene function" \
  --title "CFAP300 Gene Function Review" \
  --author "Jane Doe" \
  --keyword "genetics" \
  --keyword "cilia" \
  --contributor "John Smith"

# Custom endpoint
deep-research-client research "AI" \
  --base-url https://api.example.com \
  --api-key-env CUSTOM_API_KEY
```

---

### providers

List available research providers.

```bash
deep-research-client providers [OPTIONS]
```

#### Options

| Option | Description |
|--------|-------------|
| `--show-params` | Show available parameters for each provider |
| `--provider TEXT` | Show details for specific provider only |

#### Examples

```bash
# List all providers
deep-research-client providers

# Show parameters
deep-research-client providers --show-params

# Specific provider
deep-research-client providers --provider perplexity --show-params
```

---

### models

Show available models and their characteristics.

```bash
deep-research-client models [OPTIONS]
```

#### Options

| Option | Description |
|--------|-------------|
| `--provider TEXT` | Show models for specific provider |
| `--cost TEXT` | Filter by cost: low, medium, high, very_high |
| `--capability TEXT` | Filter by capability: web_search, academic_search, etc. |
| `--detailed` | Show detailed model information |

#### Examples

```bash
# List all models
deep-research-client models

# Filter by provider
deep-research-client models --provider perplexity

# Filter by cost
deep-research-client models --cost low

# Detailed view
deep-research-client models --detailed

# Combined filters
deep-research-client models --provider perplexity --cost medium --detailed
```

---

### list-cache

List cached research files with metadata.

```bash
deep-research-client list-cache [OPTIONS]
```

#### Options

| Option | Description |
|--------|-------------|
| `--detailed, -d` | Show detailed metadata for each entry |
| `--provider, -p TEXT` | Filter by provider name |
| `--limit, -n INT` | Limit number of results |

#### Examples

```bash
# List all cached files
deep-research-client list-cache

# Detailed view with query, model, duration, citations
deep-research-client list-cache --detailed

# Filter by provider
deep-research-client list-cache --provider perplexity

# Show only last 10 entries
deep-research-client list-cache -n 10

# Combined
deep-research-client list-cache -p openai -d -n 5
```

---

### search-cache

Search cached research files by keyword.

```bash
deep-research-client search-cache [OPTIONS] KEYWORD
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `KEYWORD` | Keyword to search for in queries and content |

#### Options

| Option | Description |
|--------|-------------|
| `--detailed, -d` | Show detailed metadata for each match |
| `--query-only, -q` | Only search in queries, not content |
| `--context, -c INT` | Characters of context around matches (default: 60) |
| `--max-snippets, -m INT` | Maximum snippets to show per match (default: 3) |
| `--no-snippets` | Hide match snippets |

#### Examples

```bash
# Search for keyword
deep-research-client search-cache "BRCA1"

# With more context
deep-research-client search-cache "CRISPR" --context 100

# More snippets
deep-research-client search-cache "mutation" --max-snippets 5

# Only search queries
deep-research-client search-cache "gene" --query-only

# No snippets
deep-research-client search-cache "protein" --no-snippets
```

---

### browse-cache

Generate a standalone HTML browser for cached research results.

```bash
deep-research-client browse-cache [OPTIONS] OUTPUT_DIR
```

Requires the `browser` optional dependency:
```bash
pip install deep-research-client[browser]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `OUTPUT_DIR` | Output directory for browser files |

#### Options

| Option | Description |
|--------|-------------|
| `--title, -t TEXT` | Browser title |
| `--description, -d TEXT` | Browser description |
| `--force, -f` | Overwrite existing directory |
| `--export-only` | Only export JSON data, don't generate browser |
| `--no-pages` | Skip generating individual HTML pages |
| `--template PATH` | Custom Jinja2 template for individual pages |

#### Examples

```bash
# Generate browser with individual pages
deep-research-client browse-cache ./browser

# Overwrite existing
deep-research-client browse-cache ./browser --force

# Custom title
deep-research-client browse-cache ./browser -t "My Research Archive"

# Browser only (no individual pages)
deep-research-client browse-cache ./browser --no-pages

# Export JSON for customization
deep-research-client browse-cache ./data --export-only

# Custom template
deep-research-client browse-cache ./browser --template my-template.j2
```

#### Output

```
output_dir/
├── index.html      # Faceted browser
├── data.js         # Browser data
├── schema.js       # Browser schema
└── pages/          # Individual result pages
    ├── openai-xxx.html
    └── ...
```

---

### clear-cache

Clear all cached research results.

```bash
deep-research-client clear-cache
```

Removes all files from `~/.deep_research_cache/`.

---

## Environment Variables

| Variable | Provider | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | OpenAI | OpenAI API key |
| `EDISON_API_KEY` | Edison | Edison Scientific API key |
| `PERPLEXITY_API_KEY` | Perplexity | Perplexity AI API key |
| `CONSENSUS_API_KEY` | Consensus | Consensus API key |
| `CBORG_API_KEY` | CBORG | CBORG proxy API key |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid options, API failure, etc.) |

## Shell Completion

Install shell completion:

```bash
# Bash
deep-research-client --install-completion bash

# Zsh
deep-research-client --install-completion zsh

# Fish
deep-research-client --install-completion fish
```
