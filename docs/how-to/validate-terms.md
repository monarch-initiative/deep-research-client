# Validate Ontology Terms

Deep research providers cite ontology terms with the same confidence they cite papers, and
with the same reliability. A report will attach `HP:9999999` to a phenotype that has no
such identifier - and, worse, will attach a *real* identifier to the wrong thing.

That second failure is why this check exists. `NCIT:C16814` is a genuine NCIT term. It
resolves. It has a label, a definition and a URL. Every check that asks "does this term
exist?" passes it. It means **Malaysia**, and a report that writes it beside the words
"Echocardiography Test" is wrong in a way nothing but a label lookup will show.

`deep-research-client` integrates
[linkml-term-validator](https://pypi.org/project/linkml-term-validator/) so a report's
terms can be checked as soon as it is produced.

Three things are checked:

1. **Existence.** Every CURIE in the report is resolved through
   [OAK](https://incatools.github.io/ontology-access-kit/). Identifiers that do not resolve
   in an ontology that resolved other terms from the same prefix are flagged as suspect.
2. **Label agreement.** The label the report writes beside a CURIE is compared with that
   term's own label. This is the check that catches `NCIT:C16814`.
3. **Obsolescence.** Terms that resolve but have been deprecated are reported separately,
   with their replacement where the ontology states one. Citing an obsolete term is not a
   fabrication; it does mean the report names something the ontology has retired.

## Install

Term validation pulls in OAK and its ontology tooling, which most users do not need, so it
is an optional extra:

```bash
pip install "deep_research_client[terms]"
```

Or, in a uv project:

```bash
uv add "deep_research_client[terms]"
```

## Validate a report you already have

```bash
deep-research-client validate-terms report.md
```

```markdown
## Term Validation

Checked with `linkml-term-validator` 0.4.5, through the `ols:` adapter.

| Outcome | Count |
| --- | --- |
| Terms checked | 10 |
| Resolved | 7 |
| Unresolved (possible confabulation) | 1 |
| Obsolete | 1 |
| Unverifiable | 1 |
| Labels checked | 7 |
| Labels matching the term | 2 |
| Labels naming a **different** term | 3 |
| Labels worth a second look | 2 |

### Terms the report names something else

These identifiers resolve, so nothing about them looks wrong, and the ontology calls them
something unrelated to what the report calls them. That usually means the identifier is not
the one the sentence needs:

- `NCIT:C16814` (1 mention) - the report calls it "Echocardiography Test"; NCIT calls it **Malaysia**
- `NCIT:C38048` (1 mention) - the report calls it "Ophthalmologic examination"; NCIT calls it **Vasovagal**

### Unresolved terms

These identifiers do not exist in an ontology that resolved other terms from the same
prefix, so they were most likely invented:

- `HP:9999999` (1 mention), reported as "Invented phenotype" - HP does not contain this term

### Obsolete terms

These terms are real but deprecated. Citing one is not a fabrication; it does mean the
report is naming something the ontology has retired:

- `GO:0008022` (obsolete protein C-terminus binding) (1 mention) - replaced by `GO:0005515`
```

Use `--in-place` to write that section into the report itself, or `--json` to get the same
result as structured data. `--in-place` replaces any section a previous run left behind, so
it is safe to re-run over a whole corpus.

## Validate as part of the research run

```bash
deep-research-client research "Marfan syndrome surveillance" \
  --output marfan.md \
  --validate-terms
```

The saved report gains both a `## Term Validation` section and a frontmatter summary:

```yaml
term_validation:
  total_terms: 10
  verified: 7
  not_found: 1
  obsolete: 1
  unverifiable: 1
  confabulation_rate: 0.111
  labels_checked: 7
  labels_matching: 2
  labels_mismatched: 3
  mislabelled_terms:
  - NCIT:C16814
  labels_variant: 2
  unresolved_terms:
  - HP:9999999
  obsolete_terms:
  - GO:0008022
  unresolvable_prefixes:
  - FAKEONT
  needs_review: true
  adapter: 'ols:'
  validator_version: 0.4.5
```

`--validate-terms` and `--validate-references` compose: passing both appends both sections
and writes both summaries.

`confabulation_rate` counts identifier resolution and nothing else. A report can resolve
every CURIE it cites, score `0.0`, and still call half of them by the names of other terms.
So `labels_mismatched`, `mislabelled_terms` and `needs_review` are written out whenever
they apply, rather than left to be worked out. If `needs_review` is absent, nothing needed
reviewing.

`needs_review` is deliberately wider than the thing that fails a build. An obsolete term or
a variant label sets it, but does not make `--fail-on-unresolved` exit non-zero: both are
go-and-looks, not failures.

## What the outcomes mean

### Term status

| Status | Meaning |
|--------|---------|
| `VERIFIED` | The CURIE resolves to a current term. |
| `NOT_FOUND` | The CURIE did not resolve, in an ontology that resolved other terms from the same prefix. Likely invented. |
| `OBSOLETE` | The CURIE resolves, but the term has been deprecated. |
| `UNVERIFIABLE` | No resolver covered this prefix, or it was skipped. Nothing was learned either way, so this is never evidence of fabrication. |

The line between `NOT_FOUND` and `UNVERIFIABLE` is drawn from what the run actually learned.
A prefix counts as resolvable if it is an OBO-library prefix, if it is configured in an
`oak_config.yaml`, or if some other term carrying it resolved during this run. A failure
under such a prefix is reported as a failure. A failure under any other prefix is reported
as unverifiable, because an unrecognised prefix may name an ontology this run could not
reach as easily as one that does not exist.

When *every* resolvable term fails at once, the report says so and hedges: that is far more
often an unreachable ontology service than a report in which every identifier is invented.

### Label agreement

| Agreement | Meaning |
|-----------|---------|
| `MATCH` | The report's label is the term's label, up to case, punctuation, word order and plurals. `seizures` matches `Seizure`. |
| `VARIANT` | Recognisably related without being the same. `Long QT syndrome` against `Long QT syndrome 1`, or a British spelling. Listed, not judged. |
| `MISMATCH` | Almost nothing in common. The report is calling the identifier something the ontology does not. |
| `NOT_ASSESSED` | The report wrote no label beside the CURIE, the term did not resolve, or `--no-check-labels` was passed. |

The threshold between `VARIANT` and `MISMATCH` was set against real pairs rather than
guessed. `NCIT:C16814` written as "Echocardiography Test" against its actual label
"Malaysia" scores 0.21; the closest genuine near-miss in the same sample, "Type 2 diabetes"
against "type 2 diabetes mellitus", scores 0.77. The threshold sits at 0.5, with margin on
both sides.

## Where labels are read from

Deciding which nearby words are a label and which are prose is the hard part, and getting
it wrong produces a confident accusation about a term that was cited correctly. So labels
are only read out of positions where a label is the only thing the text can reasonably be:

| Written as | Read as |
|------------|---------|
| `\| Seizure \| HP:0001250 \|` | the label column of a table |
| `\| HP:0001250 \| Seizure \|` | the same, identifier first |
| `HP:0001250 (Seizure)` | a bracket immediately after the CURIE |
| `HP:0001250: Seizure`, `HP:0001250 - Seizure` | a separator immediately after the CURIE |
| `**Seizure** (HP:0001250)`, `"Seizure" (HP:0001250)` | an emphasised or quoted run before it |
| `- Seizure (HP:0001250)` | a list item that names the term |

A term mentioned in flowing prose - "patients with aortic root dilation (HP:0002616) were
followed" - gets **no** reported label, and so no label check. It is still resolved and
still checked for existence. That undercounts rather than invents, which is the right way
round: reading "patients with aortic root dilation" as a label would flag a term that was
cited perfectly well.

Synonyms are not consulted, because the default OLS adapter does not expose them through
OAK. A report that calls `NCIT:C12727` "heart" rather than "Cardiac" will show a
`MISMATCH`. Read the section, do not just count it.

## Fail a pipeline on bad terms

Both commands accept `--fail-on-unresolved`, which exits with code `2` when any term fails
to resolve or is named as a different term. Obsolete terms, variant labels and unverifiable
prefixes do not trip it:

```bash
deep-research-client validate-terms report.md --fail-on-unresolved
```

## How long it takes

Measured against a 20-term list on an ordinary broadband connection, through the default
`ols:` adapter. Numbers will move with network conditions and upstream load.

| Run | Time | Per term |
|-----|-----:|---------:|
| Cold cache | 33.6s | 1.7s |
| Warm cache | 16.0s | 0.8s |

A warm cache halves the cost rather than eliminating it, because only labels are cached to
disk. Obsolescence is asked per term on every run, and over OLS that is a round trip. For a
report's worth of terms this is a small addition to a research call that already takes
minutes; for a corpus it is the thing to plan around.

For bulk work, `--adapter sqlite:obo:` downloads each ontology once and answers locally
afterwards, which turns both checks into local lookups - obsolescence included, since a
whole-ontology scan is cheap against a local database and is done once per prefix. The
first run pays for the downloads, which are large.

## Options

Every option below exists on both commands. On `research` they take a `--term-` prefix -
`--term-adapter`, `--term-skip-prefix` and so on - so that they cannot be confused with the
research options they sit beside.

| Option | What it is for |
|--------|----------------|
| `--adapter` | The OAK adapter terms are resolved through. Defaults to `ols:`, which is right for a report's worth of terms. `sqlite:obo:` is right for a corpus. |
| `--oak-config` | An `oak_config.yaml` mapping prefixes to adapters, for ontologies the default adapter does not serve. When given it is authoritative: a prefix absent from it gets no adapter at all. |
| `--cache-dir` | Where resolved labels are cached between runs. Defaults to `./terms_cache`. Point several runs at one directory. |
| `--offline` | Resolve only from the label cache, never reaching the network. Every uncached term comes back unverifiable. Useful in CI, where a network failure would otherwise look like a fabricated term. |
| `--skip-prefix` | Report a prefix as unverifiable instead of resolving it. Repeatable. |
| `--max-terms` | Stop after this many terms. The report says it was truncated. |
| `--no-check-labels` | Resolve terms but do not compare labels. Costs nothing to leave on, since the label has already been fetched to check that the term resolves. |

## What this does not check

That a term is *the right term for the sentence*. A report can name `HP:0001250` "Seizure",
pass every check here, and still be citing seizures in a paragraph about cardiac
arrhythmia. Label agreement tells you the report knows what the identifier means; it does
not tell you the identifier belongs there.

## See also

- [Validate References](validate-references.md) - the same treatment for PMIDs, DOIs and
  accessions.
