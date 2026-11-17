# Report vs curated record: Atelosteogenesis Type II

A real DisMech deep-research report compared against its curated disease record,
run through the whole claim substrate and summarised with the enriched rubric
(issue #43). This is the canonical "report vs curated disease YAML" workflow the
issue names, on real data.

## Source

- **Candidate**: `Atelosteogenesis_Type_II-deep-research-openai.md` — a DisMech
  deep-research report (provider `openai`, model `o3-deep-research`), from
  [monarch-initiative/dismech](https://github.com/monarch-initiative/dismech/tree/main/research).
- **Reference**: `Atelosteogenesis_Type_II.yaml` — the curated DisMech disorder
  record for the same disease (`MONDO:0009727`), from `kb/disorders/`.

Atelosteogenesis type II is an autosomal-recessive `SLC26A2` chondrodysplasia, so
the report and record share a dense, focused vocabulary.

## What was run

1. **Extract** both sources to claim sets (`claims extract`).
2. **Ground** the report's entities to ontology CURIEs (`claims ground`, OAK +
   model): the disease to `MONDO:0009727`, `cleft palate` to `HP:0000175`, and so
   on.
3. **Verify** the report (`claims verify --check-citations`): triage flags plus
   citation resolution against the sources.
4. **Align** the report against the record (`claims align`).
5. **Summarise** with the enriched rubric
   (`summarize_alignment(..., verification=...)`), folding grounding and
   verification in.

## Files

| File | What |
|------|------|
| `atelosteogenesis_type_ii.report.grounded.json` | the report's claims, grounded |
| `atelosteogenesis_type_ii.gold.claims.json` | the curated record's claims |
| `atelosteogenesis_type_ii.verification.json` | the report's verification (triage + citations) |
| `atelosteogenesis_type_ii.align.json` | the alignment of report (A) vs record (B) |
| `atelosteogenesis_type_ii.rubric.json` | the enriched `RubricSummary` |

The rubric (candidate = the report):

```
relevance 1.00 · informativeness 0.00 · support 0.97 · completeness 1.00
citation_quality 1.00 · grounding 0.53 · citation_accuracy 1.00 · review_burden 0.83
```

`grounding`, `citation_accuracy` and `review_burden` are the folded metrics: the
report's entities are half-grounded, every checked citation resolves to a real
work, and a sixth of the claims (the contradictory pairs' candidate side) want a
human's eye.

## A note on scale

These files hold a **representative slice** — 12 of the report's 88 claims against
10 of the record's 40 — not the whole. Aligning two *focused single-disease*
documents in full blocks near-cartesian: the disease CURIE and a handful of shared
PMIDs sit on most claims, so candidate generation proposes ~2,600 pairs for the
full report against the record (and ~440 even ungrounded, from shared citations
alone). That is intractable to classify and unwieldy to commit, so this example is
bounded to a coherent slice chosen to exercise every metric — 7 citation-checked
claims, 5 grounded-only. The full-scale live path (extract → ground → verify →
align → enriched rubric, end to end through a model) is exercised in
`tests/test_rubric_integration.py`. That candidate generation over-proposes on a
ubiquitous entity is a real limitation worth a later look, not fixed here.
