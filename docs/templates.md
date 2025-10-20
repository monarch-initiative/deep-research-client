# Template Support

The deep-research-client supports both **f-string** style templates and **Jinja2** templates, allowing you to create flexible, reusable research queries.

## Template Formats

### F-String Templates (Legacy)

F-string templates use simple `{variable}` placeholders:

```markdown
Research the {gene} gene in {organism}.
```

These templates are straightforward and work well for simple variable substitution.

### Jinja2 Templates

Jinja2 templates offer advanced features including:
- Conditional logic
- Loops
- Filters and transformations
- Template inheritance

Example with conditionals:

```jinja
Research the {{gene}} gene in {{organism}}.
{% if detail_level == "high" %}
Include detailed information about:
- Molecular function
- Biological processes
- Known mutations
{% else %}
Provide a brief overview.
{% endif %}
```

Example with loops:

```jinja
Research topics for {{gene}}:
{% for topic in topics %}
- {{topic}}
{% endfor %}
```

## Format Detection

The template format is automatically detected based on:

### 1. File Extension

Files with these extensions are treated as Jinja2 templates:
- `.j2`
- `.jinja`
- `.jinja2`

Examples:
- `gene_research.md.j2` → Jinja2
- `query.jinja` → Jinja2
- `template.md` → F-string (default)

### 2. YAML Frontmatter

For `.md` files, you can specify the format in YAML frontmatter:

```markdown
---
format: jinja
description: Advanced gene research template
---
Research the {{gene}} gene{% if organism %} in {{organism}}{% endif %}.
```

Supported format values:
- `jinja`, `jinja2` → Jinja2 format
- `fstring`, `f-string`, `python` → F-string format

### 3. Default Behavior

If no extension or frontmatter indicates the format:
- **Default: F-string** (for backward compatibility)

## Usage Examples

### Using Templates with the CLI

```bash
# F-string template
deep-research-client research \
  --template templates/gene_family.md \
  --var family=HOX

# Jinja2 template with file extension
deep-research-client research \
  --template templates/gene_jinja.md.j2 \
  --var gene=BRCA1 \
  --var organism=human \
  --var detail_level=high

# Jinja2 template with frontmatter
deep-research-client research \
  --template templates/gene_frontmatter.md \
  --var gene=TP53 \
  --var organism=mouse
```

### Using Templates Programmatically

```python
from pathlib import Path
from deep_research_client.processing import TemplateProcessor

tp = TemplateProcessor()

# Render an f-string template
result = tp.render_template(
    Path('templates/gene_family.md'),
    {'family': 'HOX'}
)

# Render a Jinja2 template
result = tp.render_template(
    Path('templates/gene_jinja.md.j2'),
    {
        'gene': 'BRCA1',
        'organism': 'human',
        'detail_level': 'high'
    }
)

# Process template with metadata
rendered_query, metadata = tp.process_template(
    Path('templates/gene_jinja.md.j2'),
    {'gene': 'BRCA1', 'organism': 'human', 'detail_level': 'high'}
)
# metadata contains: template_file, template_variables, template_format
```

## Template Examples

### Example 1: Simple F-String Template

**File:** `templates/simple_gene.md`

```markdown
Research the {gene} gene in {organism}.
```

**Usage:**
```bash
deep-research-client research --template templates/simple_gene.md \
  --var gene=TP53 --var organism=human
```

### Example 2: Jinja2 with Conditionals

**File:** `templates/detailed_gene.md.j2`

```jinja
Research the {{gene}} gene in {{organism}}.

{% if include_mutations %}
Focus on:
- Known pathogenic mutations
- Structural variants
- Clinical significance
{% endif %}

{% if include_expression %}
Include expression data:
- Tissue specificity
- Developmental stages
- Disease states
{% endif %}
```

**Usage:**
```bash
deep-research-client research --template templates/detailed_gene.md.j2 \
  --var gene=BRCA1 \
  --var organism=human \
  --var include_mutations=true \
  --var include_expression=true
```

### Example 3: Jinja2 with Loops

**File:** `templates/multi_topic.md.j2`

```jinja
Comprehensive research on {{gene}}:

{% for topic in topics %}
## {{topic|title}}

Provide detailed information about {{gene}} related to {{topic}}.

{% endfor %}
```

**Usage:**
```python
from pathlib import Path
from deep_research_client.processing import TemplateProcessor

tp = TemplateProcessor()
result = tp.render_template(
    Path('templates/multi_topic.md.j2'),
    {
        'gene': 'TP53',
        'topics': ['function', 'expression', 'mutations', 'pathways']
    }
)
```

### Example 4: Frontmatter-Based Format

**File:** `templates/frontmatter_example.md`

```markdown
---
format: jinja
author: Research Team
version: 1.0
---
# {{title}}

Research question: {{question}}

{% if background %}
## Background
{{background}}
{% endif %}

## Specific Questions
{% for q in specific_questions %}
{{loop.index}}. {{q}}
{% endfor %}
```

## Jinja2 Features

### Filters

Jinja2 includes many built-in filters:

```jinja
Gene: {{gene|upper}}
Organism: {{organism|capitalize}}
Date: {{date|default('2024-01-01')}}
```

### Conditionals

```jinja
{% if condition %}
  content when true
{% elif other_condition %}
  alternative content
{% else %}
  default content
{% endif %}
```

### Loops

```jinja
{% for item in list %}
  - {{item}}
{% endfor %}

{% for key, value in dict.items() %}
  {{key}}: {{value}}
{% endfor %}
```

### Inline Conditionals

```jinja
Research {{gene}}{% if organism %} in {{organism}}{% endif %}.
```

## Metadata Tracking

When using templates, the output includes metadata about the template used:

```yaml
---
provider: openai
model: gpt-4
template_file: templates/gene_jinja.md.j2
template_format: jinja
template_variables:
  gene: BRCA1
  organism: human
  detail_level: high
---
```

## Best Practices

1. **Use file extensions** for clarity (`.j2` for Jinja2 templates)
2. **Add frontmatter** to `.md` files when using Jinja2 features
3. **Keep it simple** - use f-strings for basic substitution
4. **Use Jinja2** when you need:
   - Conditional content
   - Loops over lists
   - Filters and transformations
   - Complex logic
5. **Document your templates** with frontmatter metadata
6. **Test templates** with various variable combinations

## Migration Guide

If you have existing f-string templates:

1. **No changes needed** - they continue to work as-is
2. **Optional upgrade** - convert to `.md.j2` or add frontmatter if you want Jinja2 features
3. **Gradual adoption** - mix both formats in the same project

Example migration:

**Before (f-string):**
```markdown
Research {gene} in {organism}.
```

**After (Jinja2 with enhancements):**
```jinja
Research {{gene}}{% if organism %} in {{organism}}{% endif %}.

{% if detail_level == "comprehensive" %}
Include:
- Molecular function
- Disease associations
- Expression patterns
{% endif %}
```

## Troubleshooting

### Template Variables Not Found

**Error:** `Missing template variables: gene, organism`

**Solution:** Ensure all required variables are provided via `--var` flags or in the variables dictionary.

### Jinja2 Syntax Errors

**Error:** `TemplateSyntaxError: unexpected '}'`

**Solution:** Check that Jinja2 templates use `{{variable}}` (double braces), not `{variable}` (single braces).

### Format Not Detected

**Issue:** Template uses Jinja2 syntax but is parsed as f-string.

**Solution:**
- Rename file with `.j2` extension, or
- Add frontmatter with `format: jinja`

## Reference

- [Jinja2 Documentation](https://jinja.palletsprojects.com/)
- [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)
