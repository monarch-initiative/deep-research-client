# Extract Claims

Turn a source into a structured list of the specific claims it makes: a deep
research report, a curated knowledge file, or any YAML/JSON document with
prose in it. Each claim is one atomic assertion, with enough provenance for a
person to find it in the source again.

Extraction does **not** judge whether a claim is true. It is the first step of
comparing sources ([issue #43](https://github.com/monarch-initiative/deep-research-client/issues/43)):
claims from two sources are what alignment and verification will work on.

Two Claude Code skills, in `.claude/skills/`, follow this page:
`extract-claims` runs an extraction, and `review-claim-set` audits and filters
the result.

## Quick start

```bash
# A deep research report (needs a model)
export OPENAI_API_KEY=...
deep-research-client claims extract report.md -o report.claims.json

# The same, through a logged-in Claude Code instead of an API key
deep-research-client claims extract report.md -o report.claims.json --llm-backend claude-code

# A curated dismech disease file (no model)
deep-research-client claims extract Marfan_Syndrome.yaml -o marfan.claims.yaml
```

From Python:

```python
from openai import AsyncOpenAI
from deep_research_client.claims import extract_claims

claims = extract_claims("report.md", llm_client=AsyncOpenAI())
for claim in claims.claim_list:
    print(claim.id, claim.anchor_status, claim.claim_text)
```

`extract_claims` is synchronous and runs its own event loop, so it raises
`RuntimeError` where a loop is already running, as in Jupyter. There, use
`await aextract_claims(...)`.
The client can be any OpenAI-compatible async client: OpenAI, CBORG, or a local
server via `AsyncOpenAI(base_url=...)`.

### Through Claude Code, with no API key

`ClaudeCodeChatClient` answers the same calls through the local `claude` CLI,
so a machine where Claude Code is logged in can extract with Claude models and
no key. Each call is one `claude --print` run with no tools, no settings files
and no MCP servers, so your hooks, output style and CLAUDE.md files do not
reach the prompt. Billing is the Claude Code login's.

```python
from deep_research_client.claims import extract_claims
from deep_research_client.claude_code_chat import ClaudeCodeChatClient

claims = extract_claims("report.md", llm_client=ClaudeCodeChatClient(), model="sonnet")
```

What to know about this backend:

- **Temperature cannot be set.** Replies are not deterministic.
- **Thinking is off**, so `max_tokens` counts only the reply, as it does for an
  OpenAI chat model.
- **An alias is recorded as the model it resolved to.** `sonnet` goes into
  `extractor.model` as the full id the replies report, such as
  `claude-sonnet-5`. (This holds for any backend: an OpenAI endpoint reports
  a dated id such as `gpt-4o-mini-2024-07-18`.)

The default is `sonnet`. Pass `--llm-model opus`, `haiku`, or a full model id.
Each call starts a process, so small sections cost a few seconds each; a
64 KB report of 64 sections took about four minutes at the default
concurrency of 4.

### Long sections

A reply cut off at `max_tokens` (default 4096) is an error, never a shorter
claim list. A section that makes many claims can need more:
`--llm-max-tokens 16000`.

## How each kind of source is read

| Source | Format | How claims are found | Needs a model |
|--------|--------|----------------------|---------------|
| Markdown report | `markdown` | Each section is decomposed into atomic claims by a model | Yes |
| dismech disease file | `dismech` | Each curated pathophysiology, phenotype, treatment and inheritance record is one claim | No |
| ai-gene-review file | `gene-review` | Each kept annotation and core function is one claim | No |
| Any other YAML/JSON | `structured` | Each prose field is decomposed like a report section | Yes |

`--format auto` (the default) decides from the extension and, for YAML/JSON,
from the keys: a `disease_term` or `pathophysiology` key means dismech, and
`existing_annotations` or `core_functions` means gene-review.

### Reports

A report written by `deep-research-client research` is read from its
`## Output` section. The question, the numbered `## Citations` list and any
generated validation sections are not claims the report makes, so they are
skipped. So is a reference list the provider wrote inside its answer (a
`References`, `Sources` or `Bibliography` heading, and anything under it). Plain
markdown with no `## Output` is read whole. Each heading starts a new unit, and
long sections are split at paragraph breaks. A `#` line inside a code fence is
code, not a heading.

For each unit the model returns, per claim:

- **`claim`**: the assertion as a standalone sentence. Pronouns are resolved,
  and a table's headers are folded into claims made by its cells.
- **`quote`**: the passage of the text that states the claim, copied verbatim.
- the parts of the claim, when it has them: **subject, predicate, object**, a
  **negation** flag and **qualifiers** such as "may" or "in adults"
- the **entities** mentioned, and the **citation markers** attached to the
  passage
- what the claim is **about**, and for a claim about the domain, what its
  **basis** is (see [What a claim is about, and what it rests on](#what-a-claim-is-about-and-what-it-rests-on))

The quote is what makes a claim auditable. It is located in the report and
becomes the claim's `source_span`: character offsets into the file, end
exclusive, so `text[start:end] == span.text`. `anchor_status` records how the
quote was found:

| `anchor_status` | Meaning |
|-----------------|---------|
| `EXACT` | The quote occurs verbatim |
| `NORMALIZED` | It occurs once whitespace, markdown emphasis, link targets, citation markers like `[3]`, quote and dash styles, and case are ignored. The span still covers the report's own characters |
| `UNANCHORED` | The quote is not in the report. The claim is kept, with no span, so the gap is visible; don't treat it as coming from the report |
| `NOT_APPLICABLE` | A structured record, located by `source_path` instead |

`claims extract` warns on stderr when any claims are `UNANCHORED`.

An `UNANCHORED` claim can carry a `nearest_passage`: the sentence of its
section closest to the quote the model gave, with a similarity score from 0 to
100 and, when it can be located, a span. It is found by
linkml-reference-validator's fuzzy matcher, and kept only at a score of 70 or
more and when it shares at least half the quote's content words. It shows why
the quote failed: a misquote ("autosomal recessive" where the report says
"dominant"), a quote joined from two sentences, a loose paraphrase. It is a
diagnostic and never the claim's location. A quote under five words gets none.

The source text goes into the prompt as it is, so a source can carry text that
tries to steer the model, such as instructions or a fake end-of-text marker. This
matters for third-party sources. Anchoring limits what such text can do: a claim
it makes the model invent has no quote in the source, so it comes back
`UNANCHORED` and gets no citations. But a steered model can still skip claims
the source does make. Treat a claim set from an untrusted source as that
source's claims only where they are anchored, and don't read it as complete.

Citations are checked the same way. A marker the model reports is kept only if
it is in the claim's own sentence, so a marker from elsewhere in the section is
not attached on the model's say-so:

- The sentence runs from the previous sentence end to the next one after the
  claim's span. A marker written after the full stop (`.[1]` or `. [1]`)
  belongs to the sentence it closes.
- The full stop of `et al.`, `e.g.`, `vs.`, `Fig.` and similar doesn't end a
  sentence. A decimal point doesn't either.
- A sentence that ends inside a bracket or quotation (`in adults.)`, `"A is
  B."`) ends where it closes.
- A table row, a list item, or a paragraph ends it too. A line break inside a
  hard-wrapped paragraph doesn't.
- `[3]` is found in `[2, 3]`, `[2-5]` and the linked `[3](https://...)`, but not
  in `[Figure 3]`.
- A claim with no span (`UNANCHORED`) gets no citations.

A numbered marker such as `[2]` is resolved through the report's own citation
list to a PMID or DOI where it names one. When the report has no `## Citations`
list, or an empty one, the provider's own numbered reference list inside its
answer (`References`, `Sources`...) is used instead: the longest, if there are
several, and never a `Further reading` list. Entries may be written `1. ...`,
`[1] ...`, `[1]: ...`, or any of them as a bulleted item. Footnotes (`[^1]: ...`)
are not read. Nor is numbering that restarts in each section's own `Sources`
list: no single list serves such an answer.

Some reports list papers, one section each, headed `### [n] Title`, with the
paper's metadata and abstract under it. Asta's reports do. There the source of
every claim in the section is paper `n`, though no sentence carries a marker.
So a section headed by `[n]` gives each located claim in it, subsections
included, a citation of paper `n`. Its identifier comes from the citation list,
or else from the section's own `PMID:` or `DOI:` lines. Every citation records
how it was attached, in `scope`:

| `scope` | Meaning |
|---------|---------|
| `SENTENCE` | A marker in the claim's own sentence |
| `SECTION` | The claim's section is headed by the cited work |
| `RECORD` | A curated record's own evidence |

### Curated files

A curated record already is one claim, so no model is needed. Each claim has:

- a `source_path` such as `phenotypes[1]`. The index counts every entry in the
  list, so it points at the record even when others were skipped.
- the file's disease or gene as its **subject**, grounded (for example
  `MONDO:0007947`, or `UniProtKB:P35555` for a gene review's accession)
- the curated ontology terms as grounded **entities**
- the evidence references as **citations**
- `about` `DOMAIN`. A record with evidence is `CITED` with basis
  `SECONDARY_SOURCE`; one with none is `UNCITED` with no basis, since a bare
  curated record does not say what it rests on.

Phenotype, treatment and inheritance records also get a **predicate** and
**object** from their section ("has phenotype" `HP:0001083`). A record with no
description is given one built from them: "Marfan syndrome has phenotype
Ectopia lentis."

Only the sections the evaluation loaders already read are covered. Other dismech
sections, such as diagnosis, prevalence and genetics, are not yet claims.

## What a claim is about, and what it rests on

Three slots classify a claim. The first is checked by the code; the other two
are the model's judgement.

**`citation_status`** says whether the source cites the claim. It makes the
lack of a citation explicit, since an empty `citations` list alone cannot tell
"cites nothing" from "not checked":

| `citation_status` | Meaning |
|-------------------|---------|
| `CITED` | At least one citation is attached, by sentence or by section, and all are in `citations` |
| `UNCITED` | The claim was located and nothing is attached to it. Its sentence may still cite a source for another claim. For a curated record: no evidence |
| `UNKNOWN` | The claim is `UNANCHORED`, so there was no sentence to check |

**`about`** is `WORK` for a claim about a publication as an object (its
authors, venue, date, identifiers, what it covers) and `DOMAIN` for a claim
about the subject matter. Reports that list papers, as Asta's do, make many
`WORK` claims: "The paper X was published in 2021."

**`basis`**, for `DOMAIN` claims only, is how the source presents the claim:

| `basis` | Meaning |
|---------|---------|
| `OBSERVATION` | The source document's own finding: an experiment, analysis, dataset or case it made or ran. A finding it reports from another work, such as an abstract under a `[n]` heading, is `SECONDARY_SOURCE` |
| `SECONDARY_SOURCE` | Attributed to another work, by a citation marker or by naming it or its authors |
| `BACKGROUND_KNOWLEDGE` | Stated with no citation or attribution, as known in the field |

Only a `DOMAIN` claim has a basis, so when the model gives a basis and no
readable `about`, the claim is read as `DOMAIN`; a basis on a claim it called
`WORK` is dropped.

When the model gives neither, the claim has no `about` and no `basis`, even in
a section headed by a cited work, where it is still `CITED`: the code cannot
tell a claim about the paper from a claim about its subject.

A deep research report observes little itself, so most of its domain claims
rest on a secondary source or on background knowledge. In a section headed by a
cited work (`### [n] Title`), a located domain claim's basis is set to
`SECONDARY_SOURCE` by the code, whatever the model said: the section's
structure already says where the claim comes from.

A claim set is refused, when built or loaded, if these contradict what the
code knows: a `CITED` claim with no citations, an `UNCITED` one with some, an
`UNKNOWN` one that was anchored, or a `basis` on a claim that is not `DOMAIN`.
One contradiction is only reported: a claim the model calls
`BACKGROUND_KNOWLEDGE` that carries a citation. Either the basis or the marker
it attached is wrong, and the code cannot tell which.
`ClaimSet.cited_background_claims` lists these, and `claims extract` warns.

## The claim set

Output is one `ClaimSet` per source, defined in LinkML
(`src/deep_research_client/claims/claims.yaml`). A curated example, from the
test fixtures:

```yaml
source:
  id: tests/input/claims/marfan_dismech.yaml
  source_type: STRUCTURED_DOCUMENT
  title: Marfan syndrome
  content_sha256: c6a9101b0de51804e12f3bce88372c72ecb66e97084601d1472a01cca6336fb7
extractor:
  name: dismech
claims:
- id: phenotypes[2]
  claim_text: Marfan syndrome has phenotype Ectopia lentis.
  source_path: phenotypes[2]
  anchor_status: NOT_APPLICABLE
  subject:
    label: Marfan syndrome
    id: MONDO:0007947
  predicate:
    label: has phenotype
  object:
    label: Ectopia lentis
    id: HP:0001083
  entities:
  - label: Ectopia lentis
    id: HP:0001083
  citation_status: UNCITED
  about: DOMAIN
```

A claim from a report has the same fields, plus a span. `source.content_sha256`
is the hash of the exact text the offsets index into. To check a saved set
against its source, compare the hash, then:

```python
from pathlib import Path
from deep_research_client.claims import ClaimSet

claims = ClaimSet.model_validate_json(Path("report.claims.json").read_text())
assert claims.mismatched_spans(Path("report.md").read_text(encoding="utf-8")) == []
```

`extractor` records the extractor, and for model-based extraction the model and
prompt version, so two sets made differently are never compared unknowingly.

### Relation to linkml-reference-validator

Slots that mean a standard term declare it: a span's `text`, `start` and `end`
are the Web Annotation `oa:exact`, `oa:start` and `oa:end`, a citation's
`reference_id` is `dcterms:references`, and a source's `title` is
`dcterms:title`. These are the URIs linkml-reference-validator finds excerpt,
reference and title fields by. It checks an excerpt against a reference only
when one class holds both, and no class here does: a span is the source's own
words, not a quote from the work it cites, so it must not be validated as one.

### Relation to OntoGPT

`subject`, `predicate`, `object`, `qualifier`, `subject_qualifier`,
`object_qualifier` and `negated` are named as in OntoGPT's `Triple` and
`ScientificClaim`, and `EntityMention` mirrors its `NamedEntity`, so claims
convert between the two directly. Two differences are deliberate:

- **Every claim has its own span.** It is established by locating the quote,
  not by searching for an entity label after the fact.
- **Offsets are end-exclusive integers.** OntoGPT uses inclusive `"start:end"`
  strings.

## What is not here yet

- **Alignment and relation labels** across two claim sets (`IDENTICAL`,
  `SUBSUMING`, `CONTRADICTORY`...), rubric aggregation and verification. These
  are the later parts of issue #43, and they consume these sets.
- **Checking a claim against the work it cites.**
  `deep_research_client.claims.verification.verify_claims` is a stub that
  raises `NotImplementedError`; its module docstring has the plan, which reuses
  linkml-reference-validator's reference fetching and caching.
- **Ontology grounding** of entities in reports beyond CURIEs written in the
  text. An entity is grounded when the report or the model gives an identifier;
  a pluggable annotator is future work.
- **Publication identifier normalisation.** Citations carry the marker as
  written, plus a PMID or DOI where one is readable; fuller normalisation is
  issue #41.
