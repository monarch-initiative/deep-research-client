# Validate References

Deep research providers confabulate. A report will happily attach `PMID:41258632` to a
claim when no such record exists, or quote a sentence that never appeared in the paper it
cites. Shipping a report with an unverified list of PMIDs and DOIs pushes that problem
downstream.

`deep-research-client` integrates
[linkml-reference-validator](https://pypi.org/project/linkml-reference-validator/) so a
report's references can be checked as soon as it is produced.

Two things are checked:

1. **Existence.** Every PMID, DOI, PMC accession and GEO accession in the report is
   resolved against PubMed, Crossref, DataCite and Entrez. Identifiers that do not resolve are flagged as suspect -
   see [what the outcomes mean](#what-the-outcomes-mean) for how much weight that carries.
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
it is safe to re-run over a whole corpus. It replaces a *trailing* section only: a report
that discusses reference validation in its body and then continues with another section
keeps everything.

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

## How long it takes

Measured against four live reports produced for this purpose - `claude_code` and `falcon`,
a short prompt and a long templated one - on an ordinary broadband connection. Numbers will
move with network conditions and upstream load, but the ratios are stable.

| Report | References | Mix | Research call | Validation, cold | Validation, warm |
|--------|-----------:|-----|--------------:|-----------------:|-----------------:|
| Achondroplasia, `claude_code` | 31 | 25 PMID, 4 DOI, 2 PMC | 283s | 70s | 1.1s |
| Achondroplasia, `falcon` | 18 | 6 PMID, 12 DOI | 356s | 76s | 2.0s |
| Marfan template, `claude_code` | 33 | 11 PMID, 5 DOI, 17 PMC | 743s | 74s | 1.2s |
| Marfan template, `falcon` | 13 | 13 DOI | 742s | 68s | 1.0s |

Two things follow from this. Validation costs roughly **10-25% of the research call** the
first time, which is a small addition to a run that already takes minutes. And on a warm
cache it costs **essentially nothing** - a second pass over the same corpus is one or two
seconds, a 35-70x speed-up.

### Cost is driven by identifier type, not reference count

The four cold timings above barely track reference count: thirteen references took 68s
while thirty-three took 74s. What separates them is what those references are.

| Identifier | Resolved via | Cold cost |
|------------|--------------|----------:|
| PMID | NCBI Entrez | 1.6s each |
| PMC | NCBI ID converter, batched, then Entrez | 1.5s each |
| DOI | Crossref, falling back to DataCite | 6.5s each |

A DOI costs about four times what a PMID does, so budget per DOI rather than per
reference. PMC accessions are as cheap as PMIDs despite needing an extra hop, because the
conversion is batched into one request for the whole report.

The two tables agree: the first report's 25 PMID + 4 DOI + 2 PMC predicts 69s against 70s
measured, and the last report's 13 DOI predicts 85s against 68s. That is also why thirteen
references took longer than thirty-three - they were all DOIs.

The delay applies per upstream request, not per reference, and a single reference can take
more than one request: removing the 0.5s delay from ten PMIDs saved 7.7s, implying about
fifteen requests for ten references.

### What each option is worth

Every option below exists on both commands. On `research` they take a `--validation-`
prefix - `--validation-rate-limit-delay`, `--validation-skip-prefix` and so on - so that
they cannot be confused with the research options they sit beside.

| Change | Effect on 10 references, cold |
|--------|------------------------------|
| Default | 15.6s |
| `--rate-limit-delay 0` | 7.9s |
| `--rate-limit-delay 2` | 44.3s |
| `--full-text` | 354.5s |

- **Cache, above all.** Fetched references are written to `./references_cache` by default
  and reused on later runs. Point `--cache-dir` at a shared directory - or commit it - to
  make repeated validation across a corpus nearly free. This is worth more than every other
  option combined. Two caveats: PMC accessions still need one request to NCBI's ID
  converter on every run, which is not cached, so a warm run is fast but not offline; and a
  cached abstract-only record *is* upgraded on a later `--full-text` run, so that first
  full-text pass over a warm corpus pays the full cost.
- **Full text is the expensive one.** `--full-text` took **23x** longer on the same ten
  references. It is off by default for that reason. Turn it on when quote checking matters:
  without it, a quote drawn from the body of a paper is reported as not found, because only
  the title and abstract were searched.
- **Rate limiting is about half the default cost, and lowering it risks false results.**
  `--rate-limit-delay` sets the pause between lookups, 0.5s by default. Setting it to 0
  roughly halves cold runtime - but NCBI caps unkeyed clients at 3 requests per second, and
  a rate-limited lookup comes back from the underlying library indistinguishable from a
  record that does not exist. It is then reported as an unresolved reference under
  *"may be fabricated"*. So this option trades speed against false accusations, not merely
  against politeness. Lower it for a one-off small run; raise it if an API starts
  rejecting requests. See [what the outcomes mean](#what-the-outcomes-mean).
- **Cap the work.** `--max-references 20` stops after the first twenty; the report says so
  explicitly rather than implying it covered everything.
- **Skip a whole identifier type.** `--skip-prefix DOI` marks those references unverifiable
  without looking them up, which given the table above is the single biggest saving after
  caching - a DOI-heavy report validates in seconds rather than minutes. Extraction produces `PMID:`, `DOI:`, `PMC:` and `GEO:` identifiers, so those are
  the prefixes this can skip; other prefixes apply to reports validated through the Python
  API, which accepts any identifier the underlying library can resolve.
- **`--no-check-quotes` saves little.** Quote checking reuses references already fetched
  for the existence check, so it adds no requests. Use it to change what is reported, not
  to go faster. Combining it with `--full-text` would be pure waste, so full text is not
  retrieved at all when no quote will be checked.

Resolution is sequential, so a bibliography of a hundred DOIs takes about eleven minutes on
a cold cache, against under three for a hundred PMIDs. Validating as part of `research --validate-references` adds that time to a run
that is already slow; validating afterwards with `validate-references` lets you do it once
across a whole corpus against a shared cache.

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
paraphrases rather than quotes gets existence checking only.

A quote is matched against the abstract, any full text retrieved, and the reference's
title. The title check exists because reports habitually quote a paper's title before
citing it, and a title never appears in its own abstract; it matches from the start of the
title, so quoting one without its subtitle is accepted. If your reports use a
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

## PMC accessions

`linkml-reference-validator` has no metadata source registered for PMC, so
`PMC12345678` would otherwise go unchecked - and long reports cite PMC heavily. A Marfan
report from `claude_code` carried 17 distinct PMC accessions against 18 PMIDs.

Accessions are therefore resolved to a PMID (or DOI) through NCBI's ID converter in one
batched call, after which the ordinary PubMed path applies, full text and quote checking
included. Europe PMC was measured as an alternative and rejected: it returned no hit for 1
of 17 real accessions taken from that report, and a lookup gap in a tool that accuses
citations of being fabricated is worse than the coverage gap it closes.

If the converter cannot be reached, those references are reported as unverifiable, never as
missing.

## Dataset accessions

GEO series and dataset accessions (`GSE68086`, `GDS1234`) are extracted and resolved
through Entrez, bare or in the `acc.cgi` URL form. Reports cite datasets as confidently as
they cite papers, and an invented accession misleads just as much as an invented PMID.

`BIOPROJECT` and `BIOSAMPLE` are deliberately **not** extracted, even though
`linkml-reference-validator` registers sources for them. Their Entrez esummary responses
currently fail to parse, so real accessions - `PRJNA31257`, `PRJEB1787`, `PRJNA13830`,
`PRJDB1234` were all checked - resolve to nothing. Extracting them would report every
BioProject citation as a possible fabrication. An integration test asserts the breakage, so
it will start failing once upstream is fixed and the accessions can be added.

`SRA`, `OMIM`, `MGNIFY`, `GTEX` and similar have no resolver at all. If you need one, the
library takes custom JSON API sources through `.linkml-reference-validator-sources.yaml`,
but this integration will not extract those identifiers from report text.

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
