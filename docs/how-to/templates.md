# How to Use Templates

Templates let you create reusable research queries with variable substitution.

## Basic Template Usage

### Create a Template File

Create `gene_research.md`:

```markdown
Research the {gene} gene in {organism}, focusing on:

1. Molecular function
2. Disease associations
3. Expression patterns
```

### Use the Template

```bash
deep-research-client research \
  --template gene_research.md \
  --var "gene=TP53" \
  --var "organism=human"
```

This sends the query:

> Research the TP53 gene in human, focusing on:
> 1. Molecular function
> 2. Disease associations
> 3. Expression patterns

## Template Formats

### F-String Templates (Simple)

Use `{variable}` placeholders:

```markdown
Research {topic} in {context}.
Include information about {aspect}.
```

### Jinja2 Templates (Advanced)

For conditionals and loops, use Jinja2 with `.j2` extension.

Create `research.md.j2`:

```jinja
Research {{topic}} in {{context}}.

{% if detailed %}
Provide comprehensive analysis including:
- Historical background
- Current state
- Future directions
{% else %}
Provide a brief overview.
{% endif %}

{% if aspects %}
Focus on:
{% for aspect in aspects %}
- {{aspect}}
{% endfor %}
{% endif %}
```

Use it:

```bash
deep-research-client research \
  --template research.md.j2 \
  --var "topic=quantum computing" \
  --var "context=healthcare" \
  --var "detailed=true"
```

## Jinja2 Detection

Templates are detected as Jinja2 based on:

1. **File extension**: `.j2`, `.jinja`, `.jinja2`
2. **Frontmatter**: Add `format: jinja` to `.md` files

Example with frontmatter:

```markdown
---
format: jinja
---
Research {{gene}}{% if organism %} in {{organism}}{% endif %}.
```

## Useful Jinja2 Features

### Conditionals

```jinja
{% if detail_level == "high" %}
Include detailed analysis.
{% elif detail_level == "medium" %}
Include moderate detail.
{% else %}
Keep it brief.
{% endif %}
```

### Loops

```jinja
Topics to cover:
{% for topic in topics %}
- {{topic|capitalize}}
{% endfor %}
```

### Default Values

```jinja
Research {{gene}} in {{organism|default('human')}}.
Since {{year|default('2020')}}, focus on recent developments.
```

### Filters

```jinja
Gene: {{gene|upper}}
Organism: {{organism|capitalize}}
```

## Example Templates

### Gene Research Template

`gene_template.md.j2`:

```jinja
# Research: {{gene}} Gene

Research the {{gene}} gene in {{organism|default('human')}}.

## Areas of Focus

{% if include_function %}
- Molecular function and mechanisms
{% endif %}
{% if include_disease %}
- Disease associations and clinical relevance
{% endif %}
{% if include_expression %}
- Expression patterns across tissues
{% endif %}

{% if year %}
Emphasize discoveries from {{year}} onwards.
{% endif %}
```

Usage:

```bash
deep-research-client research \
  --template gene_template.md.j2 \
  --var "gene=BRCA1" \
  --var "include_function=true" \
  --var "include_disease=true" \
  --var "year=2020"
```

### Drug Research Template

`drug_template.md`:

```markdown
Research the drug {drug_name} for treating {condition}.

Include:
1. Mechanism of action
2. Clinical trial results
3. Side effects and contraindications
4. Comparison with alternatives
```

Usage:

```bash
deep-research-client research \
  --template drug_template.md \
  --var "drug_name=metformin" \
  --var "condition=type 2 diabetes"
```

## Template Output Metadata

When using templates, the output frontmatter includes template info:

```yaml
---
provider: perplexity
template_file: gene_template.md.j2
template_format: jinja
template_variables:
  gene: BRCA1
  organism: human
---
```

## Troubleshooting

### Missing Variables Error

```
Error: Template requires variables: gene, organism
Use --var key=value for each variable
```

Solution: Provide all required variables with `--var`.

### Wrong Format Detection

If your Jinja2 template is parsed as f-string:

- Rename to `.md.j2` extension, or
- Add frontmatter: `format: jinja`

### Syntax Errors

Jinja2 uses `{{variable}}` (double braces), not `{variable}`.

## See Also

- [CLI Reference](../reference/cli.md) - Full `--template` documentation
