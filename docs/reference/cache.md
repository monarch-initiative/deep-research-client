# Cache Reference

Technical reference for the caching system.

## Cache Key Algorithm

The cache key determines whether a cached result can be reused. It is computed from multiple components to ensure uniqueness.

### Key Components

The following values are combined to create the cache key:

| Component | Required | Description |
|-----------|----------|-------------|
| `provider` | Yes | Provider name (e.g., `perplexity`, `openai`) |
| `query` | Yes | The full query text (case-sensitive) |
| `model` | No | Model name if specified |
| `provider_params` | No | Provider-specific parameters if specified |

### Key Generation Process

1. **Build content string**:

   ```
   {provider}:{query}
   ```

   If model is specified, append:

   ```
   |model:{model}
   ```

   If provider_params are specified, sort them alphabetically and append:

   ```
   |params:{key1}={value1},{key2}={value2},...
   ```

2. **Hash the content**:

   ```python
   hash = sha256(content.encode()).hexdigest()
   ```

3. **Extract suffix**: Take the last 8 characters of the hash.

4. **Sanitize query for filename**:
   - Convert to lowercase
   - Keep only alphanumeric characters, spaces, and hyphens
   - Replace spaces with hyphens
   - Collapse multiple hyphens
   - Truncate to 30 characters

5. **Build filename**:

   ```
   {provider}-{sanitized_query}-{hash_suffix}.json
   ```

### Examples

**Simple query:**

```
Query: "Research TP53 gene function"
Provider: perplexity

Content: "perplexity:Research TP53 gene function"
Hash: sha256(...) = "a1b2c3d4e5f6..."
Suffix: "e5f6..." (last 8 chars)
Sanitized: "research-tp53-gene-function"

Filename: perplexity-research-tp53-gene-function-e5f67890.json
```

**With model:**

```
Query: "Research TP53 gene function"
Provider: perplexity
Model: sonar-pro

Content: "perplexity:Research TP53 gene function|model:sonar-pro"
→ Different hash suffix than without model
```

**With parameters:**

```
Query: "Research TP53 gene function"
Provider: perplexity
Params: {reasoning_effort: "high", search_recency_filter: "month"}

Content: "perplexity:Research TP53 gene function|params:reasoning_effort=high,search_recency_filter=month"
→ Different hash suffix
```

## Cache File Format

Cache files are JSON with the full `ResearchResult` model.

### Schema

```json
{
  "markdown": "string - Full markdown content of research result",
  "citations": ["string - List of citation URLs"],
  "provider": "string - Provider name",
  "model": "string | null - Model used",
  "query": "string - Original query text",
  "start_time": "string | null - ISO 8601 timestamp",
  "end_time": "string | null - ISO 8601 timestamp",
  "duration_seconds": "number | null - Query duration",
  "cached": false,
  "raw_response": "object | null - Provider-specific raw response"
}
```

### Example File

`~/.deep_research_cache/perplexity-research-tp53-gene-function-e5f67890.json`:

```json
{
  "markdown": "# TP53 Gene Function\n\nTP53 (tumor protein p53) is a crucial tumor suppressor gene...\n\n## Citations\n\n1. https://www.ncbi.nlm.nih.gov/gene/7157\n2. https://www.uniprot.org/uniprotkb/P04637",
  "citations": [
    "https://www.ncbi.nlm.nih.gov/gene/7157",
    "https://www.uniprot.org/uniprotkb/P04637"
  ],
  "provider": "perplexity",
  "model": "sonar-deep-research",
  "query": "Research TP53 gene function",
  "start_time": "2025-01-15T10:30:00.000000",
  "end_time": "2025-01-15T10:31:45.000000",
  "duration_seconds": 105.2,
  "cached": false,
  "raw_response": null
}
```

## Cache Directory Structure

```
~/.deep_research_cache/
├── openai-comprehensive-analysis-of-crispr-a1b2c3d4.json
├── perplexity-research-tp53-gene-function-e5f67890.json
├── perplexity-research-tp53-gene-function-12345678.json  # Different model
├── edison-molecular-mechanisms-of-autoph-fedcba98.json
└── consensus-clinical-evidence-on-interm-87654321.json
```

Note: The same query text can have multiple cache files if different models or parameters were used (different hash suffixes).

## Cache Behavior

### Cache Hits

A cache hit occurs when a file exists matching the computed filename. On hit:

- The cached `ResearchResult` is loaded
- `cached` field is set to `True`
- No API call is made

### Cache Misses

On cache miss:

- API call is made to the provider
- Result is saved to cache file
- `cached` field remains `False` in saved file

### No Expiration

Cache entries never expire automatically. They persist until:

- Manually cleared with `deep-research-client clear-cache`
- Manually deleted from the filesystem
- Cache file becomes corrupted (auto-removed on next access)

### Corrupted Files

If a cache file cannot be parsed as valid JSON, it is automatically deleted and treated as a cache miss.

## CacheManager API

```python
from deep_research_client.cache import CacheManager
from deep_research_client.models import CacheConfig

# Initialize
config = CacheConfig(enabled=True, directory="~/.deep_research_cache")
cache = CacheManager(config)

# Get cached result
result = await cache.get(
    query="Research TP53",
    provider="perplexity",
    model="sonar-pro",           # Optional
    provider_params={"key": "value"}  # Optional
)

# Save to cache
await cache.set(
    query="Research TP53",
    provider="perplexity",
    result=research_result,
    model="sonar-pro",
    provider_params={"key": "value"}
)

# List all cache files
files = cache.list_cache_files()  # Returns list[Path]

# Get detailed cache info
info = cache.get_cache_info()  # Returns list[dict] with metadata

# Search cache by keyword
matches = cache.search_cache("TP53")  # Searches query and content

# Clear all cache
count = cache.clear_cache()  # Returns number of files deleted
```

## See Also

- [How to Manage Cache](../how-to/cache.md) - Practical cache management
- [Configuration Reference](configuration.md) - CacheConfig options
