# How to Browse Cached Research

Generate a standalone HTML browser to explore and navigate your cached research results.

## Quick Start

```bash
# Install browser dependency
pip install deep-research-client[browser]

# Generate browser
deep-research-client browse-cache ./my-browser

# Open in browser
open ./my-browser/index.html
```

## What You Get

The browser generates:

```
my-browser/
├── index.html      # Faceted browser interface
├── data.js         # Research data
├── schema.js       # Browser configuration
└── pages/          # Individual result pages
    ├── openai-gene-research-xxx.html
    ├── perplexity-gene-research-yyy.html
    └── ...
```

### Faceted Browser (`index.html`)

An interactive interface with:

- **Provider filter** - Filter by OpenAI, Perplexity, Falcon, etc.
- **Model filter** - Filter by specific model used
- **Date filter** - Filter by research date
- **Keywords filter** - Filter by assigned keywords (if any)
- **Full-text search** - Search across queries and titles
- **Clickable results** - Each entry links to its full page

### Individual Pages (`pages/`)

Each cached result gets its own HTML page with:

- **Metadata header** - Provider, model, date, duration, citation count
- **Full content** - Complete research report with formatting
- **Citations section** - All citations listed at the bottom
- **Navigation** - Link back to the browser

## Basic Usage

### Generate Browser

```bash
deep-research-client browse-cache ./browser
```

### Overwrite Existing

```bash
deep-research-client browse-cache ./browser --force
```

### Custom Title and Description

```bash
deep-research-client browse-cache ./browser \
  --title "Gene Research Archive" \
  --description "Cached research results from 2024-2025"
```

## Advanced Options

### Skip Individual Pages

Generate only the browser interface (faster, smaller output):

```bash
deep-research-client browse-cache ./browser --no-pages
```

### Export Data Only

Export JSON files without generating the browser (useful for customization):

```bash
deep-research-client browse-cache ./data --export-only
```

This creates:
- `cache_data.json` - All research entries
- `schema.json` - Browser schema configuration

You can then customize these and use `linkml-browser deploy` directly.

### Custom Page Template

Use your own Jinja2 template for individual pages:

```bash
deep-research-client browse-cache ./browser --template my-template.j2
```

## Creating Custom Templates

Individual pages are rendered using Jinja2 templates. Create your own to customize the appearance.

### Available Variables

| Variable | Type | Description |
|----------|------|-------------|
| `title` | str | Research title (may be empty) |
| `query_preview` | str | First 200 chars of query |
| `provider` | str | Provider name (openai, perplexity, etc.) |
| `model` | str | Model used |
| `date` | str | Date string (YYYY-MM-DD) |
| `duration_seconds` | float | How long research took |
| `citation_count` | int | Number of citations |
| `keywords` | list | List of keyword strings |
| `author` | str | Author name (if set) |
| `filename` | str | Original cache filename |
| `content_html` | str | Markdown converted to HTML (use `| safe`) |
| `citations` | list | List of citation strings |

### Example Template

```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ title or "Research Result" }}</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: auto; padding: 20px; }
        .meta { background: #f5f5f5; padding: 15px; margin-bottom: 20px; }
        .content { line-height: 1.6; }
        .citations { border-top: 1px solid #ccc; margin-top: 30px; padding-top: 20px; }
    </style>
</head>
<body>
    <a href="index.html">&larr; Back</a>

    <div class="meta">
        <strong>Provider:</strong> {{ provider }}<br>
        <strong>Model:</strong> {{ model }}<br>
        <strong>Date:</strong> {{ date }}<br>
        <strong>Duration:</strong> {{ "%.1f"|format(duration_seconds or 0) }}s
    </div>

    <div class="content">
        {{ content_html | safe }}
    </div>

    {% if citations %}
    <div class="citations">
        <h2>Citations ({{ citations|length }})</h2>
        <ol>
        {% for cite in citations %}
            <li>{{ cite }}</li>
        {% endfor %}
        </ol>
    </div>
    {% endif %}
</body>
</html>
```

### Using Your Template

```bash
deep-research-client browse-cache ./browser --template my-template.j2
```

## Hosting the Browser

The generated browser is completely standalone - no server required.

### Local Viewing

```bash
open ./browser/index.html
```

### GitHub Pages

1. Generate browser to `docs/browser/` or a `gh-pages` branch
2. Enable GitHub Pages in repository settings
3. Browser is available at `https://username.github.io/repo/browser/`

### Any Web Server

Upload the entire output directory to any static hosting:
- Netlify
- Vercel
- AWS S3
- Any web server

## Workflow Examples

### Research Archive

Build a browsable archive of all your research:

```bash
# Generate comprehensive browser
deep-research-client browse-cache ./research-archive \
  --title "Research Archive" \
  --description "All cached deep research results" \
  --force
```

### Provider Comparison

After running queries across multiple providers, browse and compare results:

```bash
# Run same query on different providers
deep-research-client research "CFAP300 gene function" --provider openai
deep-research-client research "CFAP300 gene function" --provider perplexity
deep-research-client research "CFAP300 gene function" --provider falcon

# Generate browser to compare
deep-research-client browse-cache ./comparison -f
```

### Sharing Results

Generate a browser to share with colleagues:

```bash
# Generate with descriptive title
deep-research-client browse-cache ./share \
  --title "Gene Function Research - Q4 2024" \
  --description "Research results for annotation review"

# Zip for sharing
zip -r research-browser.zip ./share/
```

## Troubleshooting

### "linkml-browser not installed"

Install the browser optional dependency:

```bash
pip install deep-research-client[browser]
# or
uv add deep-research-client[browser]
```

### "Output directory exists"

Use `--force` to overwrite:

```bash
deep-research-client browse-cache ./browser --force
```

### Links Not Clickable

If "View" shows as text instead of a link, ensure you have the latest version. The browser post-processes `index.html` to enable URL handling.

### Large Cache Takes Long

For very large caches, skip individual page generation:

```bash
deep-research-client browse-cache ./browser --no-pages
```

## See Also

- [Manage Cache](cache.md) - List, search, and manage cached results
- [CLI Reference](../reference/cli.md#browse-cache) - Complete command reference
