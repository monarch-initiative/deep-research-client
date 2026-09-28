---
name: review-claim-set
description: Read, audit and filter a *.claims.json/yaml from `claims extract`: what it says, which claims to trust, why some failed to anchor, and cited or domain-only subsets.
---

# review-claim-set

Read a claim set and say what in it can be trusted, what cannot, and why. A
claim set records what a source **claims**, never whether the claims are true.

The full reference is `docs/how-to/extract-claims.md`. To make a claim set,
use `extract-claims`.

## When to Use

- "What does report.claims.json say?" / "Summarise these claims"
- "Can I trust these claims?" / "Which claims didn't anchor, and why?"
- "Give me only the cited claims" / "only the claims about the disease, not the papers"
- Before comparing two claim sets, to check each one against its source

## Step 1: Summarise, checking against the source

```bash
uv run python .claude/skills/review-claim-set/scripts/summarize_claims.py \
  report.claims.json --source report.md
```

Always pass `--source` when the source file is available. The script:

- refuses a set whose slots contradict each other (the claim model's own
  consistency rules run on load)
- prints the extractor, with the model that answered and the prompt version
- counts claims by `anchor_status`, `citation_status` and `about` / `basis`,
  and citations by `scope`
- compares the source's SHA-256 with the one recorded at extraction, and for a
  markdown report checks that every span still reads as the source does
- lists unanchored claims with their nearest passage, cited background claims,
  and uncited domain claims, by section

## Step 2: Interpret it

**Source check first.** If the SHA-256 differs, the source changed after
extraction and no offset can be trusted. Say so and offer to re-extract. If
any spans differ while the hash matches, that is a bug; report it.

**What the code checked, and what the model judged.**

| Field | Set by | Trust |
|---|---|---|
| `source_span`, `anchor_status` | the code, by finding the model's quote in the source | Checked |
| `citations`, `citation_status` | the code: markers must be in the claim's own sentence, or the section must be headed by the cited work | Checked |
| `about` (`WORK` / `DOMAIN`) | the model | Judgement |
| `basis` (`OBSERVATION` / `SECONDARY_SOURCE` / `BACKGROUND_KNOWLEDGE`) | the model, except under a `### [n] Title` heading, where the code sets `SECONDARY_SOURCE` | Mostly judgement |
| `claim_text`, subject / predicate / object, `negated` | the model | Judgement; the span shows the words it came from |

**Each finding, and what to say about it.**

| Finding | Meaning |
|---|---|
| `UNANCHORED` | The model's quote is not in the source. Not the source's claim. `citation_status` is `UNKNOWN` |
| `nearest_passage` on an unanchored claim | The closest sentence to the failed quote, with a 0 to 100 score. Read it against `claim_text`: a misquote ("recessive" where the source says "dominant") means the claim misstates the source |
| Unanchored with no nearest passage | Nothing close in the section. Often a claim taken from a heading (headings are context, not text), or one the model invented |
| `NORMALIZED` | Found once whitespace, emphasis, links, markers and case are ignored. As good as `EXACT` for provenance |
| Cited background claims | The model called it uncited knowledge but a citation is attached. One of the two is wrong |
| Uncited domain claims | Located, and nothing cites them. In a report this is the report's own unsupported statement, or a marker the model did not attach |
| `WORK` claims | About a publication (authors, venue, identifiers, scope). Asta-style paper listings produce hundreds; usually filter them out |
| `- / -` in the about / basis table | The model classified neither. Such claims are left unclassified, even when a paper's section cites them, because the code cannot tell a claim about the paper from one about its subject |

Citation `scope` says how each citation was attached: `SENTENCE` (a marker in
the claim's sentence), `SECTION` (the section is headed by that paper), or
`RECORD` (a curated record's evidence).

## Step 3: Filter

```python
from collections import defaultdict
from pathlib import Path

from deep_research_client.claims import ClaimSet

claims = ClaimSet.model_validate_json(Path("report.claims.json").read_text())

# Claims about the subject matter, located in the source
domain = [c for c in claims.claim_list if c.about == "DOMAIN" and c.anchor_status != "UNANCHORED"]

# Of those, the ones the source cites, with the identifiers
cited = [(c.claim_text, [h.reference_id for h in c.citations or []])
         for c in domain if c.citation_status == "CITED"]

# Claims by section
by_section = defaultdict(list)
for c in claims.claim_list:
    by_section[c.section or c.source_path].append(c)
```

For a YAML claim set, load with
`ClaimSet.model_validate(yaml.safe_load(Path(p).read_text()))`.

Enum fields read back as plain strings (`"DOMAIN"`, `"CITED"`), so compare
with strings or with the enum members; both work.

## Step 4: Report

Lead with what can be trusted: the number of anchored claims, how many are
cited, and to which identifiers. Then the gaps: unanchored claims (with the
near misses that explain them), and contradictions. Quote `claim_text` and,
for anchored claims, the source's own words from `source_span.text`.

## Rules

- Never say a claim is true, supported or verified. Say that the source makes
  it, and cites X for it. Checking against cited works is not built yet.
- Never quote an unanchored claim as something the source said.
- A `nearest_passage` is a diagnostic, never the claim's location.
- Do not edit a claim set by hand to fix a problem. Re-extract, so provenance
  stays true.
