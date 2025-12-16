# How to Manage Cache

The client caches all research results to avoid expensive re-queries.

## How Caching Works

- **Location**: `~/.deep_research_cache/`
- **Persistence**: Cache never expires (manual clearing only)
- **Key components**: provider, query, model, and provider parameters

When you run the same query twice, the second call returns instantly from cache.

## What Makes a Cache Hit?

A cached result is returned only when **all** of these match exactly:

1. **Provider** - same provider (e.g., `perplexity`)
2. **Query** - identical query text (case-sensitive)
3. **Model** - same model (or both unspecified)
4. **Provider parameters** - same parameters (or both unspecified)

This means:

```bash
# These are DIFFERENT cache entries:
deep-research-client research "Research TP53 gene" --provider perplexity
deep-research-client research "Research TP53 gene" --provider openai
deep-research-client research "Research TP53 gene" --provider perplexity --model sonar
deep-research-client research "research TP53 gene" --provider perplexity  # lowercase 'r'
```

For full details on the cache key algorithm, see [Cache Reference](../reference/cache.md).

## View Cached Results

List all cached queries with metadata:

```bash
deep-research-client list-cache
```

Output shows provider, size, and date for each entry:

```
Found 42 cached files (15.2MB) in ~/.deep_research_cache/:

  [openai] openai-what-is-crispr-gene-a1b2c3d4.json (75.3KB, 2025-12-13 00:06)
  [perplexity] perplexity-gene-research-for-func-82faf01b.json (53.6KB, 2025-12-12 07:13)
  [falcon] falcon-gene-research-for-func-646a20c3.json (34.1KB, 2025-12-04 10:39)
```

### Detailed View

Show full metadata including query, model, duration, and citation count:

```bash
deep-research-client list-cache --detailed
```

```
  [openai] openai-gene-research-for-functional-a-0ed516e3.json
    Modified: 2025-12-13 00:06  Size: 75.3KB
    Query: # Gene Research for Functional Annotation...
    Model: o3-deep-research-2025-06-26
    Duration: 1007.8s
    Citations: 100
```

### Filter by Provider

```bash
deep-research-client list-cache --provider perplexity
deep-research-client list-cache -p openai
```

### Limit Results

```bash
deep-research-client list-cache --limit 10
deep-research-client list-cache -n 5
```

## Search Cache

Find cached results by keyword (searches both queries and content):

```bash
deep-research-client search-cache "BRCA1"
```

Output shows matching files with context snippets:

```
Found 8 cached files matching 'BRCA1':

  [perplexity] perplexity-gene-set-enrichment-7b968613.json (71.8KB, 2025-10-19) [match in: query, content]
      [query] ...list of genes from human. ## Gene Set TP53 BRCA1 BRCA2 PTEN RB1...
      [content] ...enrichment analysis on five genes: TP53, BRCA1, BRCA2, PTEN, and RB1...
```

### Search Options

```bash
# More context around matches (default: 60 chars)
deep-research-client search-cache "CRISPR" --context 100

# More snippets per match (default: 3)
deep-research-client search-cache "mutation" --max-snippets 5

# Only search in query text, not content
deep-research-client search-cache "gene" --query-only

# Hide snippets (just show matching files)
deep-research-client search-cache "protein" --no-snippets

# Detailed metadata with snippets
deep-research-client search-cache "kinase" --detailed
```

## Browse Cache (Interactive Browser)

Generate a standalone HTML browser for exploring cached results:

```bash
# Install browser dependency
pip install deep-research-client[browser]

# Generate browser
deep-research-client browse-cache ./my-browser

# Open in browser
open ./my-browser/index.html
```

This creates a faceted browser with filters for provider, model, date, and keywords, plus individual HTML pages for each research result.

For full documentation including custom templates, hosting options, and workflow examples, see [Browse Cache](browse-cache.md)

## Bypass Cache

Skip the cache for a fresh result:

```bash
deep-research-client research "What is CRISPR?" --no-cache
```

This still saves the result to cache for future queries.

## Clear Cache

Remove all cached results:

```bash
deep-research-client clear-cache
```

## Clear Specific Cache Files

Navigate to the cache directory and remove specific files:

```bash
ls ~/.deep_research_cache/
rm ~/.deep_research_cache/openai-specific-query*.json
```

## Check If Result Is Cached

The output frontmatter shows cache status:

```yaml
---
provider: perplexity
cached: true    # <-- This result came from cache
---
```

## Cache File Format

Each cache file is JSON:

```json
{
  "markdown": "# Research Report\n...",
  "citations": ["Citation 1", "Citation 2"],
  "provider": "openai",
  "model": "o3-deep-research-2025-06-26",
  "query": "original query text"
}
```

## Custom Cache Location

In Python, configure a custom cache directory:

```python
from deep_research_client import DeepResearchClient, CacheConfig

cache_config = CacheConfig(
    enabled=True,
    directory="./project_cache"
)

client = DeepResearchClient(cache_config=cache_config)
```

## Disable Caching Entirely

In Python:

```python
from deep_research_client import DeepResearchClient, CacheConfig

cache_config = CacheConfig(enabled=False)
client = DeepResearchClient(cache_config=cache_config)
```

## Tips

- **Don't clear cache unnecessarily** - Deep research queries are expensive
- **Different models = different cache entries** - Changing models creates new cache entries
- **Templates expand before caching** - The expanded query is cached, not the template

## See Also

- [Cache Reference](../reference/cache.md) - Cache key algorithm and data model
- [Configuration Reference](../reference/configuration.md) - CacheConfig options
