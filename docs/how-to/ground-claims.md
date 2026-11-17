# Ground Claims

Resolve a claim set's entities to ontology CURIEs: `Marfan syndrome` →
`MONDO:0007947`, `hyperandrogenism` → `HP:0030348`, `sebaceous gland` →
`UBERON:0001821`. Grounding fills each `EntityMention.id` while leaving the written
label alone, so claims become comparable on shared terms and
[alignment's](align-claims.md) `ENTITY` candidate signal can fire. This is the
third step of comparing sources ([issue #43](https://github.com/monarch-initiative/deep-research-client/issues/43)),
after [extraction](extract-claims.md).

Grounding is **opt-in**: the claims from `extract` are usable without it, and it is
a separate step you run when you want it. It needs the `terms` extra for OAK:

```bash
pip install "deep_research_client[terms]"
```

## Quick start

```bash
deep-research-client claims ground report.claims.json --ontologies biomedical \
    --llm-backend claude-code -o report.grounded.json
```

```python
from openai import AsyncOpenAI
from deep_research_client.claims import (
    ClaimSet, OakGrounder, biomedical_ontology_config, ground_claims)

claim_set = ClaimSet.model_validate_json(open("report.claims.json").read())
grounder = OakGrounder(
    biomedical_ontology_config(), llm_client=AsyncOpenAI(), model="gpt-4o-mini")
grounded = ground_claims(claim_set, grounder)
```

`ground_claims` is synchronous and runs its own event loop (use
`await aground_claims(...)` where one is already running, as in Jupyter). The
client is any OpenAI-compatible async client, or `ClaudeCodeChatClient` for a
logged-in Claude Code with no API key.

## How it works

1. **Collect and rank.** Every ungrounded entity label (across each claim's
   `entities`, `subject` and `object`) is collected and deduplicated, and the
   distinct labels are ordered **most frequent first**. Ontology search is the slow
   step, so the terms that matter most are grounded first, and each distinct label
   is grounded once — a report that says `acne` eighty times is one search.
2. **Search.** Each label is searched against the configured ontologies with OAK
   (`basic_search`), gathering candidate terms with their labels and definitions.
3. **Choose.** A model is shown the label, a sentence it appears in, and the
   candidates, and picks the one that denotes the same concept, or none. It can
   only choose a CURIE the search offered — an id it was not shown is refused, so
   it cannot invent one.
4. **Propagate.** The chosen CURIE is written to every mention of that label.

`--max-terms N` grounds only the N most frequent labels, bounding the work when a
report has a very long tail of one-off mentions. A `grounder` record is written to
the claim set (the ontologies, the model, the prompt version, and how many of how
many labels were grounded), as extraction and alignment record their own steps.

## Which ontologies

`--ontologies biomedical` is a built-in set: MONDO, HP, GO, CHEBI, CL and UBERON
(diseases, phenotypes, processes and functions, chemicals, cell types, anatomy),
reached over OLS. Genes and proteins (HGNC, UniProt, PR) are **not** in it; name
the ontologies you want in an `oak_config.yaml` and pass `--ontology-config`:

```yaml
ontology_adapters:
  MONDO: sqlite:obo:mondo     # a local SQLite build, downloaded once, for bulk work
  HP: sqlite:obo:hp
  HGNC: ols:hgnc              # over the network
```

The config is the same `oak_config.yaml` term validation uses. `--adapter` sets the
default for an entry with no value (`ols:` over the network, `sqlite:obo:` for bulk
local lookups); `--offline` resolves only from the cache, so with an empty cache it
grounds nothing — populate the cache with an online run first, or use a local
`sqlite:obo:` adapter. `--include-definitions` shows each candidate's definition to
the model — a help for disambiguation, but extra lookups over a networked adapter,
so it is off by default and best paired with a local `sqlite:obo:` adapter.

## Limitations

- **One concept per label across the set.** A label is grounded once and the CURIE
  is applied to every mention of it, using one representative sentence as context.
  A term that genuinely means two things in one report is grounded for the first
  sense only.
- **The `grounder` record describes the last run.** Running `claims ground` again
  keeps the ids already present and only re-grounds the rest, but overwrites the
  `grounder` provenance with the second run's counts. Ground in one pass when the
  record needs to describe the whole set.
- **Ids from the source and ids added here are not distinguished afterwards.**
  Both live in `EntityMention.id`. The `grounder` record says how many were added,
  not which.
- **A total outage fails fast.** If enough searches fail with none succeeding (a
  bad adapter, OLS down from the start), grounding stops early and exits non-zero
  rather than making the same failing calls for every label; a single transient
  error on a one-ontology config is tolerated. An outage that begins *mid-run*,
  after some labels have grounded, is not detected early — the remaining labels
  are tried, left ungrounded, and reported in the end-of-run "N of M searches
  failed" warning. A *partial* outage (one ontology of several down) is likewise
  warned about, and the labels it would have grounded are left ungrounded.

## Then align

Grounding and alignment compose: ground two reports (or a report and a curated
file), then `claims align` them, and pairs that share a grounded entity are
proposed by the `ENTITY` signal, not only by lexical overlap — which finds matches
that worded the same fact differently.

A model-judged grounding over untrusted text is advisory, as any model judgement
is; the `reason` the model gave for each choice is not stored, but the search is
reproducible and the chosen CURIE is always one the ontology actually offered.
