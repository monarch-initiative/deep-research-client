# How to Browse Markdown Research Files

Generate a standalone HTML browser from markdown research files with YAML frontmatter.

## Overview

While `browse-cache` works with the JSON cache files, `browse-files` parses markdown files directly - the same format output by the `research` command with `-o` flag. This is useful when you:

- Have research results saved as markdown files (not in cache)
- Want to browse a curated collection of results
- Need to generate a browser from files shared by others
- Want to combine results from multiple sources

## Quick Start

```bash
# Install browser dependency
pip install deep-research-client[browser]

# Generate browser from a directory of markdown files
deep-research-client browse-files ./research-outputs -o ./browser

# Open in browser
open ./browser/index.html
```

## Input Format

The command expects markdown files with YAML frontmatter - the standard output format from `deep-research-client research -o file.md`:

```markdown
---
provider: perplexity
model: sonar-pro
title: My Research Title
keywords:
  - gene
  - function
duration_seconds: 45.2
---

## Question

What is the function of BRCA1?

## Output

# BRCA1 Gene Function

BRCA1 is a tumor suppressor gene...

## Citations

1. Wikipedia - BRCA1
2. NCBI Gene Database
```

### Supported Frontmatter Fields

| Field | Type | Description |
|-------|------|-------------|
| `provider` | str | Provider name (perplexity, openai, etc.) |
| `model` | str | Model used |
| `title` | str | Research title (auto-extracted if not set) |
| `keywords` | list | Keywords for faceted filtering |
| `duration_seconds` | float | How long research took |
| `author` | str | Author attribution |
| `contributors` | list | Additional contributors |
| `template_file` | str | Template used (if any) |
| `template_variables` | dict | Template variables (if any) |
| `provider_config` | dict | Provider configuration used |

### Expected Sections

The parser looks for these standard sections:

- `## Question` - The original query
- `## Output` - The research content
- `## Citations` - Numbered list of citations

Files without these sections still work - the entire body becomes the content.

## Basic Usage

### Browse a Directory

```bash
# All .md files recursively
deep-research-client browse-files ./research -o ./browser

# Only top-level files
deep-research-client browse-files ./research -o ./browser -p "*.md"

# Specific subdirectory pattern
deep-research-client browse-files ./notes -o ./browser -p "research/**/*.md"
```

### Browse a Single File

```bash
deep-research-client browse-files ./my-research.md -o ./browser
```

### Browse Multiple Sources

```bash
# Combine files from multiple directories and individual files
deep-research-client browse-files ./dir1 ./dir2 ./extra.md -o ./browser
```

### Custom Title and Description

```bash
deep-research-client browse-files ./research -o ./browser \
  --title "Gene Research Archive" \
  --description "Curated research results 2024-2025"
```

## Advanced Options

### Glob Patterns

Use `-p/--pattern` to filter which files to include in directories:

```bash
# Only files starting with "gene-"
deep-research-client browse-files ./research -o ./browser -p "gene-*.md"

# Files in specific subdirectories
deep-research-client browse-files ./docs -o ./browser -p "research/**/*.md"

# Multiple levels deep
deep-research-client browse-files ./notes -o ./browser -p "**/research/*.md"
```

### Skip Individual Pages

Generate only the browser interface (faster, smaller output):

```bash
deep-research-client browse-files ./research -o ./browser --no-pages
```

### Export Data Only

Export JSON files without generating the browser:

```bash
deep-research-client browse-files ./research -o ./data --export-only
```

This creates:
- `files_data.json` - All parsed entries
- `schema.json` - Browser schema configuration

### Custom Page Template

Use your own Jinja2 template for individual pages:

```bash
deep-research-client browse-files ./research -o ./browser --template my-template.j2
```

See [Browse Cache - Custom Templates](browse-cache.md#creating-custom-templates) for template details.

## Workflow Examples

### Curated Research Collection

Organize and browse a curated set of research results:

```bash
# Create a directory with selected results
mkdir curated-research
cp important-result-1.md curated-research/
cp important-result-2.md curated-research/

# Generate browser
deep-research-client browse-files ./curated-research -o ./browser \
  --title "Key Research Findings"
```

### Team Research Sharing

Generate a browser from research files shared by team members:

```bash
# Clone/download shared research files
git clone https://github.com/team/research-files

# Generate browser
deep-research-client browse-files ./research-files -o ./team-browser \
  --title "Team Research Archive"
```

### Combining Multiple Sources

Browse results from different locations directly (no need to copy):

```bash
# Pass multiple directories and files as sources
deep-research-client browse-files \
  ~/project1/research \
  ~/project2/outputs \
  ~/standalone-result.md \
  -o ./browser -f
```

### Processing External Markdown

Browse markdown files that don't follow the exact format (they'll still work):

```bash
# Any markdown with frontmatter works
deep-research-client browse-files ./external-docs -o ./browser

# Title extracted from first heading if not in frontmatter
# Provider defaults to "unknown" if not specified
```

## Comparison: browse-files vs browse-cache

| Feature | browse-cache | browse-files |
|---------|--------------|--------------|
| Input | JSON cache files | Markdown files |
| Source | `~/.deep_research_cache/` | Any directory |
| Format | Internal JSON | Markdown + YAML frontmatter |
| Use case | Browse your cache | Browse any markdown files |
| File selection | All cache files | Glob patterns |

### When to Use Each

**Use `browse-cache`** when:
- Browsing your own research cache
- Quick exploration of recent queries
- No need for file organization

**Use `browse-files`** when:
- Files are organized in directories
- Sharing results with others
- Curating a specific collection
- Working with exported markdown files

## Troubleshooting

### "No markdown files found"

Check your pattern and directory:

```bash
# List what would be matched
ls ./research/**/*.md

# Try a simpler pattern
deep-research-client browse-files ./research -o ./browser -p "*.md"
```

### Missing Metadata

Files without frontmatter still work but show "unknown" for provider/model. Add frontmatter:

```markdown
---
provider: perplexity
model: sonar-pro
---

# Your content here
```

### Title Not Extracted

Title is extracted from:
1. `title` field in frontmatter (preferred)
2. First `#` heading in the content

Ensure your content has a heading:

```markdown
---
provider: test
---

## Question

My query

## Output

# This Becomes the Title

Content here...
```

## See Also

- [Browse Cache](browse-cache.md) - Browse JSON cache files
- [Manage Cache](cache.md) - List and search cache
- [CLI Reference](../reference/cli.md#browse-files) - Complete command reference
