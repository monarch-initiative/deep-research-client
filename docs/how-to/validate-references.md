# Validate References

Deep research providers confabulate. A report will happily attach `PMID:41258632` to a
claim when no such record exists, or quote a sentence that never appeared in the paper it
cites. Shipping a report with an unverified list of PMIDs and DOIs pushes that problem
downstream.

`deep-research-client` integrates
[linkml-reference-validator](https://pypi.org/project/linkml-reference-validator/) so a
report's references can be checked as soon as it is produced.

Three things are checked:

1. **Existence.** Every PMID, DOI, PMC accession and GEO accession in the report is
   resolved against PubMed, Crossref, DataCite and Entrez. Identifiers that do not resolve are flagged as suspect -
   see [what the outcomes mean](#what-the-outcomes-mean) for how much weight that carries.
2. **Supporting text.** Any quote directly attributed to a reference - written as
   `"quoted text" (PMID:12345678)` - is checked against the abstract or full text of that
   reference using deterministic substring matching.
3. **Topical relevance.** Every record that *did* resolve is weighed against the report's
   own vocabulary, because a real identifier attached to a paper about something else
   passes an existence check without complaint. See
   [topical relevance](#topical-relevance).

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
| Quoted claims **not** found in source | 0 |
| Quoted claims with nothing to check against | 1 |
| References weighed for topical relevance | 1 |
| On topic | 1 |
| Off topic | 0 |

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
  quotes_unsupported: 1
  unsupported_quote_references:
  - PMID:12345678
  quotes_not_checkable: 1
  relevance_assessed: 40
  on_topic: 38
  off_topic: 1
  off_topic_references:
  - PMID:37958553
  unresolved_references:
  - PMID:99999998
  - DOI:10.9999/totally.made.up
  needs_review: true
  validator_version: 0.2.1
```

`confabulation_rate` counts identifier resolution and nothing else. A report can resolve
every identifier it cites, score `0.0`, and still have quotes that do not match their
source - that is exactly what happened to a CHILD syndrome report which resolved 36 of 36
references while 6 of 33 quotes failed. So `quotes_unsupported`, `off_topic` and
`needs_review` are written out whenever they apply, rather than left to be worked out by
subtracting `quotes_valid` from `quotes_checked`. If `needs_review` is absent, nothing
needed reviewing.

`needs_review` is deliberately wider than the thing that fails a build. An off-topic
reference sets it, but does not make `--fail-on-unresolved` exit non-zero: it is a
go-and-look, not a failure.

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

The model is good for mixed reports and over-predicts for all-DOI ones: the first report's
25 PMID + 4 DOI + 2 PMC predicts 69s against 70s measured, while the last report's 13 DOI
predicts 85s against 68s - Crossref is quicker in bulk than the isolated measurement
suggests. Treat 6.5s per DOI as an upper bound. It is also why thirteen references took
longer than thirty-three: they were all DOIs.

The delay applies per upstream request, not per reference, and a single reference can take
more than one request: removing the 0.5s delay from ten PMIDs saved 7.7s, implying about
fifteen requests for ten references.

### What each option is worth

Every option below except `--no-check-quotes` exists on both commands. On `research` they
take a `--validation-` prefix - `--validation-rate-limit-delay`,
`--validation-skip-prefix` and so on - so that they cannot be confused with the research
options they sit beside. `--no-check-quotes` is `validate-references` only: it changes what
is reported rather than what is fetched, and the run it would speed up is already fast.

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
- **`--no-check-relevance` saves nothing at all.** The relevance check reads metadata that
  has already been fetched and does no network work of its own. Turn it off only if you do
  not want the extra section.

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
for check in report.off_topic_references:
    print(check.reference_id, check.title, check.relevance_score, check.matched_keywords)
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

## Topical relevance

An existence check answers "is this a real record?". It cannot answer "is this record about
what the report says it is about?", and it cannot tell you which question it just answered.
A CHILD syndrome report - a human sterol-biosynthesis disorder - cited an *Arabidopsis*
pollen-development paper among its sources. Real PMID, real paper, fetchable abstract,
clean bill of health. On inspection that one is a defensible citation of the plant
orthologue, but the validator had no way to distinguish it from a citation that was simply
wrong, and a reader skimming the validation block had none either.

So every record that resolves is also weighed against the report's own vocabulary:

1. The prompt is set aside - both the `## Question` section and whatever the provider
   echoed back into its own answer - so that a template's boilerplate cannot pass for the
   report's subject. See [why that matters](#the-prompt-is-not-the-subject).
2. The report's most characteristic terms are read off what remains, scored by frequency
   and by how many of the report's sections they turn up in.
3. Each reference's own metadata - title, journal, MeSH keywords, abstract - is searched
   for those terms.
4. The share of keyword weight that turns up decides the verdict.

| Verdict | Meaning |
|---------|---------|
| On topic | The record carries a substantial share of the report's vocabulary |
| Uncertain | Some overlap, or too little metadata to judge from |
| Off topic | An abstract's worth of text carrying almost none of it |
| Not assessed | Relevance checking was off, the reference did not resolve, or the report yielded no keywords |

This costs **nothing extra**: the record has already been fetched to check that it
resolves, so no new requests are made. It is on by default, and `--no-check-relevance`
(`--validation-no-relevance` on `research`) turns it off.

### Where the thresholds come from

They were measured rather than guessed, against two sets that answer different questions.

**Can it tell on topic from off?** Two real report-and-bibliography pairs - a CHILD syndrome
report with 34 fetched references and an antiphospholipid syndrome report with 47 - giving
**72 positives** (each abstract against its own report) and **144 negatives** (each abstract
against a different report's keywords, including a third report on the Parkinson's gut
microbiome). Every positive scored at or above 0.35; negatives ran from 0.00 to 0.41.

**How often would it accuse a good citation?** 2,561 references from **400 curated Falcon
disease reports**, 2,025 of them with enough text to judge. These are presumptively on
topic, so every flag there is a cost.

| | Result |
|---|---|
| `on topic` at ≥ 0.35 | confirms 72/72 positives, wrongly confirms 4/144 negatives |
| `off topic` at ≤ 0.08 | catches 27/144 negatives, flags **1.5%** of the Falcon set |
| `off topic` at ≤ 0.15 (rejected) | catches 69/144, but flags **5.6%** of the Falcon set |

That second set is what set the threshold. At 0.15 the check was flagging a good citation
roughly every other report - three were inspected and all three were plainly on topic,
including a paper on neurofibromin signalling in a neurofibromatosis report. Recall was
traded away for that: the two errors are not equally bad, because calling a good citation
off topic is a false accusation printed in a user-facing report, while missing a bad one
only leaves things where they were before this check existed.

A record without an abstract is never called off topic. Over a fifth of those 2,561
references resolved to under 300 characters of text, they score a median 0.16 with a tenth
below 0.03, and without that gate they would have filled the flagged list while saying
nothing.

The gate measures the **abstract alone**, not everything that gets searched. Title, journal
and MeSH headings are all searched and all count towards the score - a professional
indexer's subject terms are good evidence when they match. But they are controlled
vocabulary, and a paper can be squarely on topic while its MeSH headings share little with
a report's prose, so a long heading list must not by itself license an accusation. A record
with twenty headings and no abstract is reported as uncertain.

### When the whole bibliography matches nothing

One more guard, on the same principle as the existing outage hint. If **no** reference in a
report comes out on topic, the off-topic verdicts are withheld and reported as uncertain: a
keyword set that matches nothing in its own bibliography is far more likely to be a bad
keyword set than a researcher who cited nothing relevant. This is not hypothetical either -
a Falcon report on homocystinuria whose provider echoed its prompt yielded `provide`,
`comprehensive`, `claim` and `authoritative` as keywords, matched none of its eight
references, and flagged a paper titled *"Hyperhomocysteinemia in Adult Patients"*. The
reason is recorded on each reference rather than silently dropped.

A flagged reference is rendered with what the verdict rested on. This is real output, from
a Parkinson's gut-microbiome report with three genuine CHILD syndrome papers spliced into
its bibliography - every identifier resolves, and none of the report's own seventeen
citations was flagged:

```markdown
### References that may not be about this subject

- `PMID:16776722` (1 mention) - Abnormal lamellar granules in a case of CHILD syndrome.
  - shared terms: change
- `PMID:10710235` (1 mention) - Mutations in the NSDHL gene, encoding a 3beta-hydroxysteroid
  dehydrogenase, cause CHILD syndrome.
  - shared terms: patient, model, gene

Weighed against this report's own most characteristic terms: `gut`, `microbiome`,
`parkinson`, `disease`, `alpha-synuclein`, `patient`, `bacteria`, `symptom`, `brain`, ...
```

The shared-terms line is the point: `patient, model, gene` is what an unrelated biomedical
paper shares with any other, and seeing that is what lets you dismiss or confirm the flag.

**It is a clue, not a verdict, and the code is built to hedge that way.** A low score is
only treated as evidence when there was an abstract for a match to have been possible in,
so a record that resolved without one is never called off topic however little it shares.
An off-topic reference is reported in its own section, listed in the frontmatter, and
deliberately kept out of `confabulation_rate` and `--fail-on-unresolved`: the citation is
not fabricated, and a paper can be relevant in ways its abstract does not spell out. Read
the flagged references before acting on the flag.

The terms the verdict rested on are printed under the flagged list and stored in the report
as `report_keywords`, so a wrong call can be diagnosed rather than merely disbelieved.

### Why not TF-IDF

The obvious implementation is TF-IDF, and it was tried. It is the wrong tool here, and
measurably so. TF-IDF exists to find the terms that distinguish one document from a corpus;
a deep research report is a single document about a single subject, and its subject words
are precisely the ones that recur in every section - the ones an inverse document frequency
penalises hardest. Run against the 122-section CHILD syndrome report, inverting the
document-frequency term returned `reproduced`, `kegg`, `rhea`, `uspstf` and `atlas`, each a
single-section aside, while burying `nsdhl`, `cholesterol` and `ichthyosis`.

Weighting by document frequency in the *forward* direction returns `child`, `disease`,
`syndrome`, `nsdhl`, `skin`. Coverage across sections is the signal: a term that appears
throughout a report is what the report is about. It also handles a case no stoplist can
anticipate - a report that documents which databases it searched will say `search` more
often than it says its own subject, but only inside one section, so coverage demotes it.

### The prompt is not the subject

Coverage has one blind spot, and templated research pipelines walk straight into it. A
templated prompt is word-for-word identical across every report made from it, and it is
spread evenly over every section by construction - the exact profile the weighting rewards.
Measured on 1,188 real Edison/Falcon reports built from one disease template, the keywords
for a neurofibromatosis report came out as:

```
page, disease, genetic, type, clinical, nf1, gene, tumor, guideline, model,
omim, variant, peduto2023neurofibromatosistype1, treatment, applicable, ...
```

Two separate problems there. `page`, `omim`, `applicable` and `guideline` are the prompt's
"Search first: OMIM, Orphanet, ICD-10/ICD-11, MeSH" scaffolding.
`peduto2023neurofibromatosistype1` is Falcon's inline citation key, which no abstract can
ever contain - so it permanently held keyword weight that no reference could win back,
depressing every score in the report. Between them they pushed a paper titled *"The
therapeutic potential of neurofibromin signalling pathways and binding partners"* below the
off-topic line in a report about neurofibromatosis.

Three things now happen before keywords are read:

- The `## Question` section is dropped.
- Any line appearing **twice** in the report is dropped, in every copy. Providers restate
  the whole prompt inside their own answer, so cutting at `## Output` removes only the
  first of two copies; a sentence of findings repeated word for word is vanishingly rare,
  and short lines are exempt so that list items and table rules are unaffected.
- Tokens shaped like an author-year-title citation key are dropped: a surname, a plausible
  publication year, then title words. Matching by shape rather than by "contains four
  digits" is deliberate - that was the first attempt, and it silently ate the `KIAA####`
  gene family (`KIAA0319` in dyslexia, `KIAA1109` in Alkuraya-Kučinskas, `KIAA0586` and
  `KIAA0753` in Joubert). Dropping a report's most characteristic term depresses the score
  of every reference that discusses it, which is the failure this rule exists to prevent,
  running backwards.

Afterwards the same report yields `nf1, tumor, genetic, mpnst, plexiform, neurofibromas,
neurofibromatosis, optic`, and across a sample of twelve reports the share of references
recognised as on topic rose from 52/125 to 77/125.

## What the outcomes mean

| Outcome | Meaning |
|---------|---------|
| Resolved | The identifier corresponds to a real record |
| Unresolved | The identifier did not resolve; treat it as suspect until shown otherwise |
| Unverifiable | The prefix was skipped, or no resolver exists for it — nothing was learned either way |

A resolved reference means the paper is real. It does not mean the paper supports the
claim attached to it - that is what quote checking, [topical
relevance](#topical-relevance), and the LLM-judged scorers in
`deep_research_client.evaluation`, are for.

`confabulation_rate` is computed over resolved plus unresolved references only.
Unverifiable ones are left out of the denominator, so skipping a prefix cannot dilute the
figure. It says nothing at all about quotes or relevance: read `needs_review`, not the
rate, to know whether anything wants a second look.

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

**A quote that fails against an abstract has failed a weaker test.** Without `--full-text`,
only the title and abstract are searched, so a quote taken faithfully from the body of the
paper lands under "not found in the cited source" all the same. This is not hypothetical:
four of the six quote failures in the CHILD syndrome report that prompted this check were
quotes lifted correctly from the body of a GeneReviews chapter, of which PubMed serves only
the summary. Each such quote is now annotated with the fact that only the abstract was
searched, and `source_content_type` records it in the structured output. Re-run with
`--full-text` before treating one of those as invented.

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

Two DOI-bearing URL shapes are also not reached: bioRxiv's version suffix
(`biorxiv.org/content/10.1101/…v1`, where the `v1` is part of no path segment) and any
form that neither names the DOI after a path segment nor as an `id=` parameter.

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
