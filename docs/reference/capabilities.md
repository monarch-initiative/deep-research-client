# Capabilities, Resources & Archetypes

Every provider model card is annotated with three controlled vocabularies that
describe *what a provider is*:

- **`archetype`** — where it sits on the spectrum from pure retrieval to an
  autonomous co-scientist.
- **`capabilities`** — the functional things it can *do* (verbs).
- **`resources`** — the data sources / knowledge bases it *wraps* (nouns).

These vocabularies are defined once, authoritatively, in the LinkML schema at
`src/deep_research_client/schema/deep_research_client.yaml` and generated into
Pydantic enums under `src/deep_research_client/datamodel/`. To change them, edit
the schema and run `just gen-datamodel` — never hand-edit the generated file.

## Deep research is a subset of the co-scientist case

It is useful to see the providers as points on a single spectrum. A conventional
"deep research" tool searches a corpus and writes a cited report. An autonomous
**co-scientist** does that *and more*: it forms hypotheses, executes code against
biomedical data, and can design experiments. In other words, a deep-research run
is essentially a **subset** of what a co-scientist does — sometimes traded off
against more sophisticated literature-scanning algorithms on the retrieval side.

The `ProviderArchetype` enum makes this ordering explicit (each archetype is
broadly a superset of the previous one):

| Rank | Archetype | What it does | Examples |
|------|-----------|--------------|----------|
| 1 | `retriever` | Returns retrieved evidence (papers, passages, snippets) with no authored synthesis. | Asta |
| 2 | `synthesizer` | Searches one or more corpora and writes a cited narrative report — the classic "deep research" tool. | OpenAI Deep Research, Perplexity, Consensus, Edison/Falcon |
| 3 | `agentic_researcher` | A multi-step agent that plans, browses, and may run tools/code, still oriented around producing a written answer. | Claude Code, Cyberian, DeepER-Med |
| 4 | `co_scientist` | An autonomous scientific agent that forms and tests hypotheses, runs code against biomedical data, and can design experiments. | OpenScientist, Biomni |

```
retriever  ⊂  synthesizer  ⊂  agentic_researcher  ⊂  co_scientist
                (deep research)                       (hypothesis-driven,
                                                        runs code)
```

## Capabilities

`ResearchCapability` — the functional capabilities a provider exposes. The
historical `ModelCapability` set (`web_search`, `academic_search`,
`scientific_literature`, `citation_tracking`, `real_time_data`,
`code_interpretation`, `visual_analysis`, `multi_language`) is retained: both
`ModelCapability` and its `UPPER_CASE` member names (e.g.
`ModelCapability.WEB_SEARCH`) still resolve to the same members. The remaining
values below were added to describe co-scientist behavior.

| Capability | Meaning |
|------------|---------|
| `web_search` | Searches the open web for current information. |
| `academic_search` | Searches curated academic / scholarly corpora. |
| `scientific_literature` | Specialised search / analysis over scientific literature. |
| `citation_tracking` | Produces and tracks structured citations. |
| `real_time_data` | Can access recent / real-time information. |
| `code_interpretation` | Executes code (analysis, computation, plotting). |
| `visual_analysis` | Interprets or generates figures and charts. |
| `multi_language` | Operates across multiple natural languages. |
| `retrieval_only` | Returns retrieved evidence verbatim, no synthesis. |
| `evidence_synthesis` | Synthesises sources into an authored report. |
| `hypothesis_generation` | Proposes testable scientific hypotheses. |
| `experiment_design` | Designs experiments / analysis plans. |
| `data_analysis` | Analyses structured biomedical / experimental datasets. |

## Resources

`ResearchResource` — the data sources and knowledge bases a provider wraps. A
capability is a verb (*what it does*); a resource is a noun (*what it can
reach*). `pubmed` is a resource; `code_interpretation` is a capability.

| Resource | Examples |
|----------|----------|
| `general_web` | The open web. |
| `pubmed` | PubMed / MEDLINE. |
| `semantic_scholar` | Semantic Scholar corpus. |
| `arxiv` | arXiv preprints. |
| `preprint_servers` | bioRxiv / medRxiv. |
| `clinical_trials` | ClinicalTrials.gov and similar registries. |
| `biomedical_databases` | Curated disease / pathway / drug knowledge bases. |
| `genomic_databases` | Ensembl, ClinVar, etc. |
| `chemical_databases` | PubChem, ChEMBL, etc. |
| `protein_structure_databases` | UniProt, PDB, etc. |

## Querying by capability, resource, or archetype

The vocabularies are queryable across all registered providers:

```python
from deep_research_client import (
    find_models_by_capability,
    find_models_by_resource,
    find_models_by_archetype,
    ResearchCapability,
    ResearchResource,
    ProviderArchetype,
)

# Which providers run code?
find_models_by_capability(ResearchCapability.code_interpretation)

# Which providers reach PubMed?
find_models_by_resource(ResearchResource.pubmed)

# Which providers are full co-scientists?
find_models_by_archetype(ProviderArchetype.co_scientist)   # -> openscientist, biomni
```

Each returns a `{provider_name: [ModelCard, ...]}` mapping.
