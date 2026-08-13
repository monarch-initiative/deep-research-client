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
| `--validate-references` | Resolve every cited identifier and append a validation section |
| `--validation-cache-dir PATH` | Directory for cached reference lookups (default: `./references_cache`) |
| `--validation-email TEXT` | Contact email for the NCBI Entrez API (defaults to `$NCBI_EMAIL`) |
| `--validation-full-text` | Fetch full text as well as abstracts when validating (~23x slower) |
| `--validation-max-references INT` | Stop after validating this many references |
| `--validation-skip-prefix TEXT` | Identifier prefix to report as unverifiable rather than resolving (repeatable) |
| `--validation-rate-limit-delay FLOAT` | Seconds to wait between lookups (default: 0.5) |
| `--fail-on-unresolved` | Exit with code 2 if any reference or quote fails validation |

When `--output` is provided, any non-text artifacts recovered with the report are written beside it in an `OUTPUT_STEM_artifacts/` directory and linked from the generated markdown.

The `--validate-*` options require the optional `validation` extra; see [validate-references](#validate-references) and the [Validate References how-to](../how-to/validate-references.md). Validation runs *after* the report has been written or printed, so a lookup service being unreachable never costs you the research result; that case exits with code `3`. On a cold cache it adds roughly 10-25% to the wall time of the research call, and next to nothing on a warm one.

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

# Check every cited identifier before trusting the report
deep-research-client research "Statins and myopathy risk" \
  --output statins.md \
  --validate-references
```

---

### validate-references

Check that the references cited in a saved report actually exist, and that quotes attributed to them really appear in the source. Handles PMIDs, DOIs, PMC accessions and GEO accessions.

```bash
deep-research-client validate-references [OPTIONS] FILES...
```

Requires the optional `validation` extra:

```bash
pip install "deep_research_client[validation]"
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `FILES...` | One or more markdown report files to validate |

#### Options

| Option | Description |
|--------|-------------|
| `--check-quotes / --no-check-quotes` | Check quoted claims against the text of the reference they cite (default: on) |
| `--cache-dir PATH` | Directory for cached reference lookups (default: `./references_cache`) |
| `--email TEXT` | Contact email for the NCBI Entrez API (defaults to `$NCBI_EMAIL`) |
| `--full-text` | Fetch full text as well as abstracts (~23x slower, better quote checks) |
| `--max-references INT` | Stop after validating this many references per file |
| `--skip-prefix TEXT` | Identifier prefix to report as unverifiable rather than resolving (repeatable) |
| `--rate-limit-delay FLOAT` | Seconds to wait between lookups (default: 0.5); lowering it risks rate-limit errors being reported as unresolved references |
| `--in-place` | Replace or append the validation section in each input file |
| `--output PATH` | Write the markdown validation report to a file (single input file only) |
| `--json PATH` | Write the validation report as JSON (single input file only) |
| `--fail-on-unresolved` | Exit with code 2 if any reference or quote fails validation |

#### Notes

- Every PMID, DOI, PMC accession and GEO accession in the file is resolved against PubMed, Crossref, DataCite and Entrez. Identifiers that do not resolve are reported as likely confabulations — but see the caveat on [what the outcomes mean](../how-to/validate-references.md#what-the-outcomes-mean), since a lookup that failed for network reasons is indistinguishable from one that failed because the record does not exist.
- Quotes are checked only when they are directly attributed, as in `"quoted text" (PMID:12345678)`. A quote whose reference could not be fetched is reported as *not checked* rather than as unsupported.
- Fetched references are cached on disk, so re-running over the same corpus is fast and polite to the upstream APIs. A cold run costs roughly 1.6s per PMID or PMC accession and 6.5s per DOI; a warm one is 1-2s for a whole report. `--full-text` multiplies the cold cost by about 23. See [how long it takes](../how-to/validate-references.md#how-long-it-takes).
- `--in-place` is idempotent: an existing validation section is replaced rather than appended to, so repeated runs neither stack sections nor re-count the identifiers a previous run listed.
- Exit codes: `0` success, `1` usage, input or filesystem error, `2` validation found problems (only with `--fail-on-unresolved`), `3` a lookup service was unreachable.

#### Examples

```bash
# Validate a saved report
deep-research-client validate-references report.md

# Validate several reports and append the results to each
deep-research-client validate-references reports/*.md --in-place

# Fail a pipeline when any citation is fabricated
deep-research-client validate-references report.md --fail-on-unresolved

# Existence checks only, capped at 20 references
deep-research-client validate-references report.md --no-check-quotes --max-references 20

# Machine-readable output
deep-research-client validate-references report.md --json validation.json
```

---

### edison-trajectory

Retrieve an existing Edison trajectory by ID, preserving any report artifacts recovered from the verbose trajectory payload.

```bash
deep-research-client edison-trajectory [OPTIONS] TRAJECTORY_ID
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `TRAJECTORY_ID` | Existing Edison trajectory/task ID |

#### Options

| Option | Description |
|--------|-------------|
| `--output PATH` | Output file path (prints to stdout if not provided) |
| `--separate-citations PATH` | Save citations to a separate file |

#### Notes

- Requires `EDISON_API_KEY` (or legacy `FUTUREHOUSE_API_KEY`) to be set.
- When `--output` is provided, recovered artifacts are written beside the report in an `OUTPUT_STEM_artifacts/` directory.
- Frontmatter includes `trajectory_id` and `artifact_sources` so it is clear where recovered artifacts came from.

#### Examples

```bash
# Retrieve an Edison trajectory into the current directory
deep-research-client edison-trajectory 784d73d5-da42-402e-9701-6c5b44beab14 \
  --output edison-report.md

# Write citations separately
deep-research-client edison-trajectory 784d73d5-da42-402e-9701-6c5b44beab14 \
  --output edison-report.md \
  --separate-citations edison-report.citations.md
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

### browse-files

Generate a standalone HTML browser from markdown research files.

```bash
deep-research-client browse-files [OPTIONS] SOURCES... -o OUTPUT_DIR
```

Requires the `browser` optional dependency:
```bash
pip install deep-research-client[browser]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `SOURCES...` | One or more directories or files to include |

#### Options

| Option | Description |
|--------|-------------|
| `--output, -o PATH` | Output directory for browser files (required) |
| `--pattern, -p TEXT` | Glob pattern for finding files in directories (default: `**/*.md`) |
| `--title, -t TEXT` | Browser title |
| `--description TEXT` | Browser description |
| `--force, -f` | Overwrite existing directory |
| `--export-only` | Only export JSON data, don't generate browser |
| `--no-pages` | Skip generating individual HTML pages |
| `--template PATH` | Custom Jinja2 template for individual pages |

#### Examples

```bash
# Browse all markdown files in a directory
deep-research-client browse-files ./research-outputs -o ./browser

# Use a specific glob pattern
deep-research-client browse-files ./docs -o ./browser -p "*.md"

# Browse a single file
deep-research-client browse-files ./my-research.md -o ./browser

# Multiple sources (directories and files)
deep-research-client browse-files ./dir1 ./dir2 ./extra.md -o ./browser

# Recursively find files matching pattern
deep-research-client browse-files ./notes -o ./browser -p "research/**/*.md"

# Export JSON only
deep-research-client browse-files ./research -o ./data --export-only

# Custom title
deep-research-client browse-files ./research -o ./browser -t "Research Archive"
```

#### Output

Same structure as `browse-cache`:

```
output_dir/
├── index.html      # Faceted browser
├── data.js         # Browser data
├── schema.js       # Browser schema
└── pages/          # Individual result pages
    ├── file1.html
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
| `ASTA_API_KEY` | Asta | Asta retrieval API key |
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
