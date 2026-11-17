# Align Claims

Compare the claims of one source against another and classify how each matched
pair relates. This is the second step of comparing sources
([issue #43](https://github.com/monarch-initiative/deep-research-client/issues/43)),
after [extracting claims](extract-claims.md): given two claim sets A and B, it
finds the pairs that are about the same thing and labels each on two axes.

Alignment does **not** judge whether a claim is true. It says how two claims
relate, and leaves checking either against the literature to verification.

## The two axes

A single "identical / contradictory / ..." label hides a distinction that
matters: two sources can state the **same conclusion** while standing in a
**different relationship to the evidence**. One may report it as its own
experimental finding; the other may assume it as background knowledge, or
attribute it to a third work. So every aligned pair carries two labels:

- **`semantic_relation`** — how the two propositions relate. Judged by a model
  from the claim texts: `IDENTICAL`, `SUBSUMES` (A is more general), `SUBSUMED_BY`
  (A is more specific), `CLOSE` (same fact, different wording/precision/scope),
  `CONSISTENT` (different but compatible), `COMPLEMENTARY` (consistent and jointly
  more informative), `CONTRADICTORY`, `UNRELATED`.
- **`evidential_relation`** — how the two claims' evidential standing relates.
  **Derived**, not judged, from each claim's `about` and `basis` (recorded during
  extraction): for example `BOTH_OBSERVATION`, `A_OBSERVATION_B_BACKGROUND`,
  `A_SECONDARY_B_OBSERVATION`, `NOT_APPLICABLE` (a claim about a work),
  `UNDETERMINED` (a basis the extractor did not give). It is read from A to B, so
  swapping the sources swaps an asymmetric label.

A pair also records **`shares_citation`**: whether both claims cite a work that
resolves to the same identifier. Two claims that agree but rest on the same cited
source are not independent corroboration; two that *disagree* while citing the
same source are worth a closer look.

A claim with no aligned partner is not a relation; it is *coverage*, read off the
alignment (`unaligned_a_ids`, `unaligned_b_ids`) and summarised by the rubrics.

## Quick start

```bash
# Extract claims from each source first (see the extract-claims guide)
deep-research-client claims extract report.md    -o report.claims.json --llm-backend claude-code
deep-research-client claims extract gold.yaml     -o gold.claims.json   --llm-backend claude-code

# Align A against B, and print a rubric summary
deep-research-client claims align report.claims.json gold.claims.json \
    --llm-backend claude-code --rubric --reference b -o alignment.json
```

From Python:

```python
from openai import AsyncOpenAI
from deep_research_client.claims import ClaimSet, align_claim_sets, summarize_alignment

set_a = ClaimSet.model_validate_json(open("report.claims.json").read())
set_b = ClaimSet.model_validate_json(open("gold.claims.json").read())

alignment = align_claim_sets(set_a, set_b, llm_client=AsyncOpenAI(), model="gpt-4o-mini")
for pair in alignment.pair_list:
    print(pair.claim_a_id, pair.claim_b_id, pair.semantic_relation, pair.evidential_relation)

summary = summarize_alignment(alignment, set_a, set_b, reference="b")
```

`align_claim_sets` is synchronous and runs its own event loop, so it raises
`RuntimeError` where a loop is already running, as in Jupyter. There, use
`await aalign_claim_sets(...)`. The client can be any OpenAI-compatible async
client, or `ClaudeCodeChatClient` for a logged-in Claude Code with no API key
(the same backend the [extract guide](extract-claims.md) describes).

To compare a single pair of claims directly, without candidate generation:

```python
from deep_research_client.claims import compare_claims

pair = compare_claims(claim_a, claim_b, llm_client=AsyncOpenAI(), model="gpt-4o-mini")
```

## How pairs are proposed

Asking a model about every A×B pair does not scale (two reports of a few hundred
claims each would be a hundred thousand calls). A cheap pass first proposes only
the pairs worth classifying, and each pair records which signal proposed it:

- **`ENTITY`** — the two claims share a grounded entity (the same CURIE).
- **`CITATION`** — the two claims cite a work with the same resolved `reference_id`.
- **`LEXICAL`** — the two claims' content words overlap at or above `--min-overlap`
  (Jaccard over stop-word-filtered, singularised terms).
- **`EMBEDDING`** — a caller-supplied embedder scored the claim texts as similar.

The first three need no extra dependency. Embeddings are an **optional hook**: pass
an `embedder` to `align_claim_sets` and pairs above the cosine threshold are added.

```python
def embedder(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=list(texts))
    return [d.embedding for d in resp.data]

alignment = align_claim_sets(set_a, set_b, llm_client=..., model=..., embedder=embedder)
```

The CLI uses the no-dependency path; the embedder hook is Python-only for now.

## Report-level rubrics

`summarize_alignment` aggregates the pairs into count-based, auditable metrics
(issue #43, item 4), with one source treated as the reference the other is judged
against. The standard metrics are `relevance`, `informativeness`, `support`,
`completeness` and `citation_quality`. Each is a `numerator / denominator` with the
ids behind it, so any number can be checked and a downstream project can recompute
or extend the set. For the example above, with the report as A and the curated file
as B (`--reference b`):

```
relevance: 0.75 (3/4)          -- 1 candidate claim aligns to nothing in the reference
informativeness: 0.25 (1/4)    -- 1 candidate claim adds information the reference lacks
support: 0.67 (2/3)            -- 1 of 3 aligned pairs is contradictory
completeness: 1.00 (3/3)       -- 0 reference claims the candidate does not cover
citation_quality: 0.25 (1/4)   -- 3 of 4 candidate domain claims are uncited
```

## What each backend records

The model that classified the relations is recorded on `alignment.aligner`
(`model`, `prompt_version`, `min_overlap`, `embedder_used`), exactly as extraction
records its extractor, so two alignments are never compared across different
prompts or models unknowingly. An alias such as `sonnet` is stored as the full id
the replies reported.
