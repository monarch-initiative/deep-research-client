# Validate References

Deep research providers confabulate. A report will happily attach `PMID:41258632` to a
claim when no such record exists, or quote a sentence that never appeared in the paper it
cites. Shipping a report with an unverified list of PMIDs and DOIs pushes that problem
downstream.

`deep-research-client` integrates
[linkml-reference-validator](https://pypi.org/project/linkml-reference-validator/) so a
report's references can be checked as soon as it is produced.

Two things are checked:

1. **Existence.** Every PMID and DOI in the report is resolved against PubMed, Crossref
   and DataCite. Identifiers that do not resolve are reported as likely fabrications.
2. **Supporting text.** Any quote directly attributed to a reference - written as
   `"quoted text" (PMID:12345678)` - is checked against the abstract or full text of that
   reference using deterministic substring matching.

## Install

Reference validation pulls in bibliographic tooling that most users do not need, so it is
an optional extra:

```bash
pip install "deep_research_client[validation]"
```

Or, in a uv project:

```bash
uv add "deep_research_client[validation]"
```

NCBI asks for a contact address on Entrez requests. Set it once:

```bash
export NCBI_EMAIL="you@example.org"
```

## Validate a report you already have

```bash
deep-research-client validate-references report.md
```

```markdown
## Reference Validation

Checked with `linkml-reference-validator` 0.2.1.

| Outcome | Count |
| --- | --- |
| References checked | 3 |
| Resolved | 1 |
| Unresolved (possible confabulation) | 2 |
| Unverifiable | 0 |
| Quoted claims checked | 2 |
| Quoted claims found in source | 1 |

### Unresolved references

These identifiers did not resolve to a record and may be fabricated:

- `PMID:99999998` (cited 2x) - Identifier did not resolve to a record
- `DOI:10.9999/totally.made.up` (cited 1x) - Identifier did not resolve to a record

### Quotes not found in the cited source

- `PMID:7913883`: "widget-based therapy reverses achondroplasia in adults"
  - closest text in source: "Achondroplasia (ACH) is the most common genetic form of dwarfism"
```

Use `--in-place` to append that section to the report itself, or `--json` to get the same
result as structured data.

## Validate as part of the research run

```bash
deep-research-client research "Statins and myopathy risk" \
  --output statins.md \
  --validate-references
```

The saved report gains both a `## Reference Validation` section and a frontmatter summary:

```yaml
reference_validation:
  total_references: 42
  verified: 40
  not_found: 2
  unverifiable: 0
  confabulation_rate: 0.048
  quotes_checked: 6
  quotes_valid: 5
  unresolved_references:
  - PMID:99999998
  - DOI:10.9999/totally.made.up
  validator_version: 0.2.1
```

## Fail a pipeline on bad citations

Both commands accept `--fail-on-unresolved`, which exits with code `2` when any reference
fails to resolve or any attributed quote is not found in its source:

```bash
deep-research-client validate-references report.md --fail-on-unresolved
```

## Keep it fast

Resolution is network-bound and sequential, so a bibliography of a hundred references takes
a few minutes on the first pass.

- **Cache.** Fetched references are written to `./references_cache` by default and reused on
  later runs. Point `--cache-dir` at a shared directory - or commit it - to make repeated
  validation across a corpus effectively free.
- **Cap the work.** `--max-references 20` stops after the first twenty references; the
  report says so explicitly rather than pretending it covered everything.
- **Skip existence-only checks you do not need.** `--no-check-quotes` skips supporting text
  validation.
- **Skip prefixes you cannot resolve.** `--skip-prefix SRA --skip-prefix BIOPROJECT` marks
  those identifiers unverifiable instead of reporting them as missing.

Full text retrieval is off by default because it is much slower. Turn it on with
`--full-text` when quote checking matters: without it, a quote drawn from the body of a
paper will be reported as not found because only the abstract was searched.

## Use it from Python

```python
from deep_research_client import DeepResearchClient, ReferenceValidator

client = DeepResearchClient()
result = client.research("Statins and myopathy risk")

validator = ReferenceValidator(cache_dir="references_cache", email="you@example.org")
report = validator.validate_result(result)

print(report.confabulation_rate)
for check in report.confabulated_references:
    print(check.reference_id, check.message)
for check in report.unsupported_quotes:
    print(check.reference_id, check.quote)
```

`report.to_markdown()` renders the section shown above, and `report.summary()` gives the
frontmatter dictionary. `ReferenceValidationReport` is a Pydantic model, so
`model_dump_json()` works as usual.

## How quotes are recognised

Only quotes with an adjacent citation are checkable, so the extractor looks for a
double-quoted span of at least 20 characters followed by a parenthesised or bracketed
identifier:

```markdown
The authors report that "Achondroplasia (ACH) is the most common genetic form of
dwarfism" (PMID:7913883).
```

Both straight and typographic quotation marks work, as do square brackets. A report that
paraphrases rather than quotes gets existence checking only. If your reports use a
different convention, pass your own pattern to
`ReferenceValidator.validate_markdown(..., quote_pattern=...)`; capture group 1 must be the
quote and group 2 the citation.

## What the outcomes mean

| Outcome | Meaning |
|---------|---------|
| Resolved | The identifier corresponds to a real record |
| Unresolved | The identifier did not resolve; treat it as fabricated until shown otherwise |
| Unverifiable | The prefix was skipped, or no resolver exists for it |

A resolved reference means the paper is real. It does not mean the paper supports the
claim attached to it - that is what quote checking, and the LLM-judged scorers in
`deep_research_client.evaluation`, are for.
