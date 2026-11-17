# Worked example: comparing a report's claims with itself

This is a real run of within-report alignment (see [Align Claims](../how-to/align-claims.md))
on one deep research report, to show what the substrate finds on real output
rather than on fixtures. The claim set and the alignment it produced are committed
alongside this page, so the numbers below can be inspected without re-running any
model:

- [`examples/within-report-alignment/acne_vulgaris.claims.json`](https://github.com/monarch-initiative/deep-research-client/blob/main/examples/within-report-alignment/acne_vulgaris.claims.json)
- [`examples/within-report-alignment/acne_vulgaris.self-align.json`](https://github.com/monarch-initiative/deep-research-client/blob/main/examples/within-report-alignment/acne_vulgaris.self-align.json)

## The source

One report from the dismech corpus: `Acne_Vulgaris-deep-research-falcon.md`
(provider `falcon`, ~27 KB), from
<https://github.com/monarch-initiative/dismech/tree/main/research>. Acne vulgaris
is a common, heavily studied condition, so a long report tends to state the same
facts in more than one section — exactly the internal structure within-report
alignment is meant to surface. The claims' `source.content_sha256` pins the exact
text they were extracted from.

## Reproducing it

```bash
# 1. extract the report's claims (Claude Code, no API key)
deep-research-client claims extract Acne_Vulgaris-deep-research-falcon.md \
  --llm-backend claude-code --llm-max-tokens 16000 -o acne_vulgaris.claims.json

# 2. align the report against itself; a higher --min-overlap keeps candidate
#    pairs meaningful within one single-topic document
deep-research-client claims align acne_vulgaris.claims.json \
  --llm-backend claude-code --min-overlap 0.6 -o acne_vulgaris.self-align.json
```

Both steps used `claude-sonnet-5-5` (extraction prompt version 5, alignment prompt
version 1). Extraction produced **267 claims**, all anchored exactly in the source.
Candidate generation at `--min-overlap 0.6` proposed **114 pairs** to classify (all
by lexical overlap: the report's entities carry no CURIEs and it uses no inline
citation markers, so the entity and citation signals did not fire).

## What the 114 pairs were

| `semantic_relation` | count | what it means here |
|---|---:|---|
| `CONSISTENT` | 86 | different, compatible facts about acne — most lexically-similar pairs are not the same claim |
| `COMPLEMENTARY` | 9 | different facts about one subject that together say more |
| `IDENTICAL` | 7 | the report states the same claim twice (or more) |
| `CLOSE` | 7 | the same fact at a different precision or scope |
| `SUBSUMES` / `SUBSUMED_BY` | 2 / 1 | one statement is a general or specific case of another |
| `UNRELATED` | 2 | lexical look-alikes with no real relation |
| `CONTRADICTORY` | 0 | the report does not contradict itself |

On the evidential axis, 105 pairs are `BOTH_SECONDARY` (both claims attributed to
other work), 6 `BOTH_BACKGROUND`, and 3 `NOT_APPLICABLE` (a claim about a
publication). The report makes almost all of its claims as secondary reporting, so
within it there is little observation-vs-background contrast to find; that contrast
is where the evidential axis earns its place when a report is compared against a
*different* source.

### Internal redundancy (14 pairs, 12 distinct claims)

The report states acne's adult-female prevalence three times, in three sections:

> `c18` "Acne has a prevalence of 15–20% among adult women." · `IDENTICAL` ·
> `c63` "Adult female acne has a prevalence of 15–20%." · `IDENTICAL` ·
> `c166` "The prevalence of acne in adult women is 15–20%."

and that hyperandrogenism occurs in ~50% of adult female acne twice (`c19`/`c64`).
`CLOSE`, not `IDENTICAL`, is reserved for the same fact at a different scope —
*C. acnes* phylotype IA1 is reported at 84.4% "overall" but 95.6% "on back lesions"
(`c58`/`c59`), which the model keeps distinct rather than flattening or calling a
contradiction.

### Refinement (3 pairs)

A vague claim and its precise instance are linked by subsumption, not treated as
unrelated:

> `c20` "Hyperandrogenism in adult female acne is commonly linked to PCOS."
> **subsumes** `c168` "70% of hyperandrogenism cases in adult female acne feature PCOS."

and ontology generality is captured too: "regulation of cell differentiation"
subsumes "Th17 cell differentiation" (`c117`/`c122`).

### The classifier correcting the blocker (2 pairs)

Lexical blocking proposes any two sentences that share enough words, including ones
that only share a *template*. The model rejected both as `UNRELATED`:

> `c231` "Linoleic acid has the ChEBI identifier CHEBI:15956." vs
> `c233` "Succinic acid has the ChEBI identifier CHEBI:16707."

and two "poor quality or predatory journal" judgements about different articles
(`c263`/`c265`). This is why candidate generation is cheap-and-permissive and the
model is the arbiter: the blocker's false positives are caught, not kept.

## Reading it

The headline is that a long, careful report restates **12 of its claims** and
refines a few more, with no internal contradiction. That is a useful signal on its
own — redundancy and over-claiming are quality signals a curator can act on — and
it is the same machinery that, across two sources, drives the report-level rubrics.
A model-judged relation over untrusted text is advisory, as always; the committed
alignment carries each pair's one-sentence rationale so a reader can check any call.
