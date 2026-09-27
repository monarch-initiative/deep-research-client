# Extract Claims

Turn a source into a structured list of the specific claims it makes: a deep
research report, a curated knowledge file, or any YAML/JSON document with
prose in it. Each claim is one atomic assertion, with enough provenance for a
person to find it in the source again.

Extraction does **not** judge whether a claim is true. It is the first step of
comparing sources ([issue #43](https://github.com/monarch-initiative/deep-research-client/issues/43)):
claims from two sources are what alignment and verification will work on.

## Quick start

```bash
# A deep research report (needs a model)
export OPENAI_API_KEY=...
deep-research-client claims extract report.md -o report.claims.json

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

`extract_claims` is synchronous; use `aextract_claims` inside an event loop.
The client can be any OpenAI-compatible async client: OpenAI, CBORG, or a local
server via `AsyncOpenAI(base_url=...)`.

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
skipped. Plain markdown with no `## Output` is read whole. Each heading starts a
new unit, and long sections are split at paragraph breaks.

For each unit the model returns, per claim:

- **`claim`**: the assertion as a standalone sentence. Pronouns are resolved,
  and a table's headers are folded into claims made by its cells.
- **`quote`**: the passage of the text that states the claim, copied verbatim.
- the parts of the claim, when it has them: **subject, predicate, object**, a
  **negation** flag and **qualifiers** such as "may" or "in adults"
- the **entities** mentioned, and the **citation markers** attached to the
  passage

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

Citations are checked the same way. A marker the model reports but the text
does not contain is dropped. A numbered marker such as `[2]` is resolved
through the report's own citation list to a PMID or DOI where it names one.

### Curated files

A curated record already is one claim, so no model is needed. Each claim has:

- a `source_path` such as `phenotypes[1]`. The index counts every entry in the
  list, so it points at the record even when others were skipped.
- the file's disease or gene as its **subject**, grounded (for example
  `MONDO:0007947`, or `UniProtKB:P35555` for a gene review's accession)
- the curated ontology terms as grounded **entities**
- the evidence references as **citations**

Phenotype, treatment and inheritance records also get a **predicate** and
**object** from their section ("has phenotype" `HP:0001083`). A record with no
description is given one built from them: "Marfan syndrome has phenotype
Ectopia lentis."

Only the sections the evaluation loaders already read are covered. Other dismech
sections, such as diagnosis, prevalence and genetics, are not yet claims.

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
- **Ontology grounding** of entities in reports beyond CURIEs written in the
  text. An entity is grounded when the report or the model gives an identifier;
  a pluggable annotator is future work.
- **Publication identifier normalisation.** Citations carry the marker as
  written, plus a PMID or DOI where one is readable; fuller normalisation is
  issue #41.
