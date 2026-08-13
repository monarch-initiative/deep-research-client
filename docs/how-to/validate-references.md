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
   and DataCite. Identifiers that do not resolve are flagged as suspect - see
   [what the outcomes mean](#what-the-outcomes-mean) for how much weight that carries.
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
| Quoted claims checked | 1 |
| Quoted claims found in source | 1 |
| Quoted claims with nothing to check against | 1 |

### Unresolved references

These identifiers did not resolve to a record and may be fabricated. A lookup that failed
for transport reasons is indistinguishable from one that failed because the record does not
exist, so spot-check before acting on them:

- `PMID:99999998` (2 mentions) - Identifier did not resolve to a record
- `DOI:10.9999/totally.made.up` (1 mention) - Identifier did not resolve to a record

### Quotes that could not be checked

There was no text to compare these against, so they are neither confirmed nor contradicted:

- `PMID:99999998`: "widget-based therapy reverses achondroplasia in adults"
  - Reference did not resolve, so the quote could not be checked
```

Use `--in-place` to write that section into the report itself, or `--json` to get the same
result as structured data. `--in-place` replaces any section a previous run left behind, so
it is safe to re-run over a whole corpus.

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
  quotes_not_checkable: 1
  unresolved_references:
  - PMID:99999998
  - DOI:10.9999/totally.made.up
  validator_version: 0.2.1
```

## Fail a pipeline on bad citations

Both commands accept `--fail-on-unresolved`, which exits with code `2` when any reference
fails to resolve or any attributed quote is checked and not found in its source. Quotes
that could not be checked at all do not trip it:

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
- **Slow down or speed up.** `--rate-limit-delay` sets the pause between lookups (0.5s by
  default). Raise it if an API starts rejecting requests; lower it when working entirely
  from a warm cache.
- **Skip existence-only checks you do not need.** `--no-check-quotes` skips supporting text
  validation.
- **Skip a whole identifier type.** `--skip-prefix DOI` marks those references unverifiable
  without looking them up, which is useful when Crossref is slow and you only care about
  PubMed. Extraction only ever produces `PMID:` and `DOI:` identifiers, so those are the
  only two prefixes this can skip; it takes other prefixes for reports validated through
  the Python API, which accepts any identifier the underlying library can resolve.

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

## The data model

The shape of a validation report is defined in LinkML, at
`src/deep_research_client/validation/reference_validation.yaml`. That schema is the source
of truth: `datamodel.py` is generated from it with

```bash
just gen-datamodel
```

Edit the YAML, regenerate, and commit both. A test compares the checked-in Pydantic model
against a fresh generation and fails if they have drifted, so the two cannot quietly
diverge.

`models.py` sits on top of the generated model and adds what a schema cannot express:
counts, `confabulation_rate`, and the markdown and frontmatter rendering. Those are
computed from the schema's slots rather than stored, so they never need to be kept in sync
by hand.

Because the schema is LinkML, the usual tooling applies to it - `gen-json-schema`,
`gen-docs`, `linkml-validate` against a serialised report, and so on.

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
| Unresolved | The identifier did not resolve; treat it as suspect until shown otherwise |
| Unverifiable | The prefix was skipped, or no resolver exists for it — nothing was learned either way |

A resolved reference means the paper is real. It does not mean the paper supports the
claim attached to it - that is what quote checking, and the LLM-judged scorers in
`deep_research_client.evaluation`, are for.

`confabulation_rate` is computed over resolved plus unresolved references only.
Unverifiable ones are left out of the denominator, so skipping a prefix cannot dilute the
figure.

**"Unresolved" is not proof of fabrication.** The underlying fetcher returns nothing both
for an identifier that does not exist and for a lookup that failed in transit - a timeout,
an HTTP 500, an NCBI rate limit. The two are indistinguishable from here. In practice a
handful of unresolved references among many resolved ones is a strong signal; a report in
which *every* reference failed is almost always a network problem, and the rendered
section says so rather than accusing the whole bibliography. Spot-check before acting.

Quotes get a third outcome. A quote is only reported as *not found in the cited source*
when there was a source to search: if the reference did not resolve, exposed no abstract or
full text, was skipped by prefix, or fell outside `--max-references`, the quote is listed
under "could not be checked" instead. Those do not count towards `--fail-on-unresolved`,
because an unavailable source is not evidence against a quote.

## A note on enum values

The generated model sets LinkML's `use_enum_values`, so `check.status` holds the enum's
*value* rather than the member:

```python
check.status == ReferenceStatus.VERIFIED   # True - ReferenceStatus is a str enum
isinstance(check.status, ReferenceStatus)  # False
check.status.value                         # AttributeError
```

Compare against members; do not reach for attributes on them. Type checkers believe the
annotation here, so this is one place where a clean `mypy` run will not save you.
