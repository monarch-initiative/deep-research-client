# DARMA: Deep Agentic Research Metadata Archive

**Status:** Brainstorming / Proposal
**Date:** 2025-12-14
**Authors:** Chris Mungall, Claude

## Executive Summary

DARMA (Deep Agentic Research Metadata Archive) is a proposed community-driven archive for deep research outputs that serves dual purposes:

1. **Shared Cache**: A centralized repository of deep research results, avoiding duplicate expensive queries across labs and institutions
2. **Prototype AI Research Journal**: A transparent, evaluation-driven publication venue for AI-generated research with full provenance

The infrastructure is intentionally lightweight: GitHub repositories, static pages, and issue-based workflows. It leverages the existing `deep-research-client` (drc) data model and can ingest from pre-existing research repositories.

## Background and Motivation

### The Problem

AI-generated research content is rapidly emerging as a significant contributor to scientific knowledge. Tools like OpenAI Deep Research, Perplexity, Edison Scientific, and others can synthesize vast bodies of literature into coherent summaries. However:

- **Duplication**: Multiple researchers run identical or similar queries, wasting compute and API costs
- **No shared infrastructure**: Results live in personal caches, not discoverable by others
- **Quality uncertainty**: No systematic way to evaluate or compare AI research outputs
- **Provenance gaps**: Traditional publishing infrastructure assumes human authorship

### The Case for Dedicated AI Research Archives

As Martin Monperrus argues in ["Preprint Servers for AI Generated Papers"](https://www.monperrus.net/martin/preprint-servers-of-AI-generated-papers), the scientific community faces a critical infrastructure gap. Current preprint servers actively reject AI as a legitimate research contributor:

> "We need an AI preprint server that explicitly accepts AI-authored work... without proper venues, valuable AI-generated insights may be dismissed or lost."

Major platforms enforce restrictive policies:
- **arXiv** now requires peer-review documentation for review articles due to AI generation concerns
- **bioRxiv/medRxiv** prohibit listing AI systems as authors
- **SSRN** rejects content "completely or mostly generated" by LLMs
- **OSF Preprints** has similar restrictions

The consequence is a binary choice: either AI-generated research is lost entirely, or it infiltrates traditional servers without proper disclosure, "eroding the current human-to-human scholarly communication system."

Monperrus identifies the key requirements for AI research infrastructure:
1. **Detailed metadata**: AI system specifications, model details, degree of human involvement
2. **Clear separation**: Distinguish AI-generated from human-written research
3. **Transparency**: Treat AI as a legitimate contributor while ensuring complete provenance
4. **Trusted operation**: Ideally operated by established organizations for credibility

### Existing Platforms

Several platforms have emerged to address this gap (surveyed by [Monperrus 2025](https://www.monperrus.net/martin/preprint-servers-of-AI-generated-papers), with updates):

| Platform | Launch | Description | Status |
|----------|--------|-------------|--------|
| **[aiXiv](https://aixiv.science)** | 2025 | "Preprint server for AI Scientists and Robot Scientists" | Live; open source ([GitHub](https://github.com/aixiv-org/aiXiv)) |
| **[ai.vixra.org](http://ai.vixra.org)** | March 2025 | Archives AI-assisted articles with minimal gatekeeping | Live; operated by Scientific God Inc. |
| **[rxiVerse](https://rxiverse.com)** | July 2025 | Similar to ai.vixra.org; $19 submission fee | Live; operated by Scientific God Inc. |
| **[AI Archive](https://ai-archive.io)** | Fall 2025 | Publishes AI-generated papers and reviews; API-first | Live; custom platform |
| **AgentRxiv** | 2025 (proposed) | Framework for agent-to-agent scientific communication | Conceptual |

#### aiXiv Deep Dive

[aiXiv](https://github.com/aixiv-org/aiXiv) is the most technically mature open-source option, worth examining in detail:

**Architecture:**
- Frontend: React 18 + Tailwind CSS + Radix UI
- Backend: FastAPI + PostgreSQL + AWS S3 ([aixiv-core](https://github.com/aixiv-org/aixiv-core))
- Auth: Clerk
- License: MIT

**Data Model** (from `aixiv-core`):
```sql
submissions (
  id SERIAL PRIMARY KEY,
  title VARCHAR(220),
  agent_authors TEXT[],           -- Array of AI agent identifiers
  corresponding_author VARCHAR(120),
  category VARCHAR(100)[],
  keywords VARCHAR(100)[],
  license VARCHAR(50),
  abstract TEXT,
  s3_url TEXT,                    -- PDF storage
  uploaded_by VARCHAR(64),
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)
```

**Key Features:**
- **Dual-track review**: Papers (editor-led peer review) vs Proposals (community validation with mentors)
- **Auto-agents ecosystem**: Custom AI agents for discovery, quality assessment, statistical analysis
- **API-first**: Bearer token auth, endpoints for search, submit, reviews, agent registration
- **DOI assignment** and version control

**Reported Stats:** 15,420+ papers, 3,280+ proposals, 892+ active agents, 8,650+ reviews

**Limitations for our use case:**
- Custom platform infrastructure (not Git-native)
- Focused on full papers, not granular research queries
- No structured evaluation criteria schema
- Limited integration with research tools like drc

**Limitations of existing platforms generally:**
- Custom infrastructure with platform risk (except aiXiv which is open source)
- Minimal structured evaluation frameworks
- Limited integration with existing research tools
- Varying levels of metadata rigor
- Most lack community governance models

### Related Publishing Models

| Model | Description | Relevance to DARMA |
|-------|-------------|-------------------|
| **[Micropublications](https://www.micropublication.org/)** | Single-observation peer-reviewed papers | Granularity model; one result = one publication |
| **[Nanopublications](https://nanopub.net/)** | Machine-readable RDF assertions with provenance | Structured claims; but complex infrastructure |
| **Overlay journals** | Curation layer over preprints | Model for "highlights" feature |

### Why DARMA?

DARMA aims to fill the gap with a pragmatic, GitHub-native approach that:

1. **Starts immediately** with minimal infrastructure overhead
2. **Leverages existing tools** (drc data model, GitHub Actions, static sites)
3. **Emphasizes evaluation** via community and agent review
4. **Maintains full provenance** per Monperrus's requirements
5. **Operates transparently** with open governance
6. **Avoids platform lock-in** through standard formats and Git-based storage

### Design Principles

Drawing from the evolving landscape of AI in scientific publishing:

| Principle | Implementation |
|-----------|----------------|
| **Transparency** | Full provenance: query, provider, model, params, timestamps |
| **Reproducibility** | Store enough metadata to re-run queries |
| **Human accountability** | Every submission has a human supervisor/submitter |
| **Machine-native** | Structured YAML/JSON, not just PDFs |
| **Granularity** | One query = one result = one citable unit |
| **Immutability** | Results don't change; evaluations accumulate separately |
| **Community evaluation** | Quality assessments by humans and agents |

## Architecture

### Repository Structure

```
darma/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── submit-result.yml      # Submit new DR result
│   │   └── submit-evaluation.yml  # Evaluate an existing result
│   └── workflows/
│       ├── validate-submission.yml
│       ├── sync-feeders.yml       # Pull from ai-gene-review, dismech, etc.
│       └── build-site.yml         # Rebuild static pages
│
├── results/                       # ResearchResult objects (immutable)
│   └── YYYY/MM/
│       ├── {provider}-{slug}-{hash8}.yaml
│       └── {provider}-{slug}-{hash8}.md
│
├── submissions/                   # Ingestion metadata (who submitted, how)
│   └── YYYY/MM/
│       └── {provider}-{slug}-{hash8}.yaml
│
├── evaluations/                   # Evals pointing to results
│   └── YYYY/MM/
│       └── eval-{result-id}-{NNN}.yaml
│
├── schema/                        # Validation schemas
│   ├── research_result.yaml       # Synced with drc
│   ├── submission.yaml
│   └── evaluation.yaml
│
├── feeders.yaml                   # Config for auto-sync sources
│
└── site/                          # Generated static site
    ├── index.html
    ├── browse/
    ├── search/
    └── result/{id}/
```

### ID Format

Reuses the drc cache key format:

```
{provider}-{slugified-query}-{hash8}

Examples:
  perplexity-crispr-mechanisms-a1b2c3d4
  openai-tp53-gene-function-f7e8d9c0
  edison-autophagy-cancer-therapy-12345678
```

Full canonical reference includes the date path:

```
results/2025/12/perplexity-crispr-mechanisms-a1b2c3d4
```

For citations:

```
DARMA:2025/12/perplexity-crispr-mechanisms-a1b2c3d4
```

## Data Model

### Result Object

The archived result is exactly the drc `ResearchResult` model - no DARMA-specific extensions. This keeps the two projects in sync and allows seamless ingestion from drc caches.

```yaml
# results/2025/12/perplexity-crispr-mechanisms-a1b2c3d4.yaml

# Core fields (required)
markdown: |
  # CRISPR Mechanisms in Human Cells
  ...
citations:
  - https://doi.org/10.1038/...
  - https://pubmed.ncbi.nlm.nih.gov/...
provider: perplexity
query: "Research CRISPR mechanisms in human cells"
cached: false

# Publication-style metadata
title: "CRISPR Mechanisms in Human Cells"
abstract: "This report synthesizes current understanding of..."
keywords:
  - CRISPR
  - Cas9
  - gene editing
  - human cells
query_metadata:
  author: Chris Mungall
  contributors: []

# Timing information
start_time: 2025-12-14T10:25:00
end_time: 2025-12-14T10:27:25
duration_seconds: 145.2

# Provider details
model: sonar-deep-research
provider_config:
  timeout: 600
  max_retries: 3

# Template info (if used)
template_file: null
template_variables: null
```

### Submission Metadata

Tracks provenance of how the result entered DARMA, without modifying the result itself:

```yaml
# submissions/2025/12/perplexity-crispr-mechanisms-a1b2c3d4.yaml

result_id: perplexity-crispr-mechanisms-a1b2c3d4
submitted: 2025-12-14T12:00:00Z

submitter:
  name: Chris Mungall
  orcid: 0000-0002-6601-2165
  github: cmungall

source:
  type: drc_cache          # drc_cache | feeder | manual | issue
  feeder_repo: null        # e.g., monarch-initiative/ai-gene-review
  original_path: ~/.deep_research_cache/perplexity-crispr-mechanisms-a1b2c3d4.json
  issue_number: null       # if submitted via GitHub issue

content_hash: sha256:abc123...  # for deduplication
license: CC-BY-4.0
```

### Evaluation Object

Evaluations point to results (not vice versa), keeping results immutable:

```yaml
# evaluations/2025/12/eval-perplexity-crispr-mechanisms-a1b2c3d4-001.yaml

id: eval-perplexity-crispr-mechanisms-a1b2c3d4-001
result_id: perplexity-crispr-mechanisms-a1b2c3d4
result_path: results/2025/12/perplexity-crispr-mechanisms-a1b2c3d4.yaml

submitted: 2025-12-14T14:00:00Z

evaluator:
  type: human              # human | agent
  name: Jane Doe
  orcid: 0000-0001-2345-6789
  github: janedoe
  # For agent evaluators:
  # agent_type: claude
  # agent_model: claude-3-opus
  # agent_version: 2025-01-01

criteria:
  factual_accuracy:
    score: 0.85
    evidence: "Verified 17/20 claims against primary sources"

  citation_quality:
    score: 0.90
    evidence: "Citations are relevant and accessible; 22/23 URLs resolve"

  comprehensiveness:
    score: 0.75
    evidence: "Missing recent work on base editing (2024)"

  hallucination_check:
    score: 0.95
    evidence: "One minor hallucinated statistic identified in section 3.2"

  bias_assessment:
    score: 0.80
    evidence: "Slight over-representation of US-based research"

overall_recommendation: accept_with_notes  # accept | accept_with_notes | revise | reject
notes: |
  High quality synthesis. Would benefit from coverage of
  base editing advances. Minor factual correction needed
  in section 3.2.
```

## Evaluation Criteria

Inspired by DeepEval and LLM-as-judge patterns:

| Criterion | Description | Evaluator Types |
|-----------|-------------|-----------------|
| **Factual Accuracy** | Claims match cited sources | Human, LLM-as-judge |
| **Citation Quality** | Sources are relevant, accessible, reputable | Human, automated URL checks |
| **Comprehensiveness** | Covers key aspects of the topic | Human, domain expert |
| **Hallucination Check** | No fabricated facts or citations | LLM-as-judge, human |
| **Bias Assessment** | Balanced coverage, no systematic skew | Human |
| **Recency** | Uses recent literature where relevant | Automated date extraction |
| **Clarity** | Well-organized, readable | Human, LLM |

### Aggregate Scores

Computed at query/build time by scanning evaluations:

```python
def get_result_with_evals(result_id: str) -> dict:
    result = load_yaml(f"results/{year}/{month}/{result_id}.yaml")

    # Find all evals pointing to this result
    evals = glob(f"evaluations/**/*{result_id}*.yaml")

    # Compute aggregates
    scores = defaultdict(list)
    for eval_file in evals:
        eval_data = load_yaml(eval_file)
        for criterion, data in eval_data['criteria'].items():
            scores[criterion].append(data['score'])

    # Inject for display (not persisted to result file)
    result['_computed'] = {
        'evaluation_count': len(evals),
        'aggregate_scores': {k: mean(v) for k, v in scores.items()},
        'evaluations': [load_yaml(e) for e in evals]
    }
    return result
```

Benefits of this approach:
- Results are **immutable** after submission
- Evaluations can accumulate without touching result files
- No merge conflicts or churn on popular results
- Easy to regenerate aggregates

## Workflows

### 1. Individual Submission via GitHub Issue

```
User opens issue using "Submit Result" template
  │
  ├─ Pastes: YAML content or path to drc cache file
  ├─ Fills in: submitter info, license
  │
  ▼
GitHub Action validates against schema
  │
  ▼
Creates files:
  ├─ results/YYYY/MM/{id}.yaml
  ├─ results/YYYY/MM/{id}.md
  └─ submissions/YYYY/MM/{id}.yaml
  │
  ▼
Closes issue with link to result page
```

### 2. Command-Line Submission

```bash
# User runs drc, result cached locally
deep-research-client research "CRISPR mechanisms" --provider perplexity

# Submit to DARMA
darma submit ~/.deep_research_cache/perplexity-crispr-mechanisms-a1b2c3d4.json \
  --submitter "Chris Mungall" \
  --orcid "0000-0002-6601-2165"
```

### 3. Bulk Ingest from Feeders

```yaml
# feeders.yaml
feeders:
  - name: ai-gene-review
    repo: monarch-initiative/ai-gene-review
    path: results/
    format: drc

  - name: dismech
    repo: monarch-initiative/dismech
    path: research_outputs/
    format: drc
```

Sync workflow:
```
GitHub Action runs daily (or on-demand)
  │
  ▼
For each feeder repo:
  ├─ Fetch new results not already in DARMA
  ├─ Check content_hash for deduplication
  └─ Create result + submission files
      └─ source.type = feeder name
```

### 4. Evaluation Submission

```bash
darma evaluate perplexity-crispr-mechanisms-a1b2c3d4 \
  --evaluator "Jane Doe" \
  --orcid "0000-0001-2345-6789" \
  --scores factual_accuracy=0.85 citation_quality=0.90 \
  --notes "Good synthesis, minor issues in section 3"
```

Or via GitHub issue template.

## Static Site Features

Built with mkdocs-material or similar:

### Browse
- By date, domain, provider, model
- Filter by aggregate scores
- Show evaluation status (unevaluated, evaluated, contested)

### Search
- Full-text search of content
- Filter by metadata fields
- (Future: semantic search)

### Result Page
- Full markdown render
- Provenance sidebar (provider, model, timing)
- Citations list with link validation status
- Evaluation scores with expandable evidence
- "Cite this" button (BibTeX, RIS)
- "Re-run with drc" command snippet

### Leaderboards
- Top contributors
- Most evaluated results
- Highest quality by domain
- Provider/model comparisons

## Schema Synchronization with drc

Options for keeping schemas in sync:

**Option A: DARMA imports from drc**
```yaml
# darma/schema/research_result.yaml
$ref: https://raw.githubusercontent.com/.../deep-research-client/main/schema/research_result.yaml
```

**Option B: Shared schema repository**
```
monarch-initiative/research-schemas/
├── research_result.yaml
├── evaluation.yaml
└── ...
```

**Option C: Periodic copy with drift detection** (simplest to start)
- GitHub Action periodically copies schema from drc
- Alerts on changes for manual review

## Comparison with Other Platforms

Based on the landscape surveyed by [Monperrus (2025)](https://www.monperrus.net/martin/preprint-servers-of-AI-generated-papers), with additional research:

| Feature | aiXiv | ai.vixra.org | AI Archive | arXiv | DARMA |
|---------|-------|--------------|------------|-------|-------|
| Accepts AI research | Yes | Yes | Yes | Limited | Yes |
| Infrastructure | FastAPI+S3 | Custom | Custom | Custom | GitHub-native |
| Submission cost | Free | Free | Free | Free | Free |
| Barrier to entry | Account | Low | Account | Endorsement | GitHub account |
| AI metadata depth | Basic (`agent_authors[]`) | Basic | Detailed | N/A | Full provenance |
| Evaluation system | Auto-agents | None | AI + human | External | Community + agent |
| Quality assurance | Dual-track review | Minimal | Automated | Peer review | Schema + evals |
| Open source | **Yes (MIT)** | No | No | No | Yes |
| Content granularity | Full papers | Papers | Papers | Papers | Query-level results |
| Integration w/ tools | API | None | API | None | drc + Git |
| Governance | OPENAGS org | Private | Private | Cornell | Community |

### Key Differentiators

**DARMA vs aiXiv:**
| Aspect | aiXiv | DARMA |
|--------|-------|-------|
| Unit of publication | Full papers/proposals | Individual research queries |
| Data model | Generic submissions | drc `ResearchResult` with full provenance |
| Storage | PostgreSQL + S3 | Git (YAML/MD files) |
| Reproducibility | PDF storage | Query + params = re-runnable |
| Evaluation schema | Unstructured | DeepEval-inspired criteria |
| Integration | Standalone platform | drc ecosystem, feeder repos |

**Per Monperrus's requirements:**

| Requirement | How DARMA Addresses It |
|-------------|----------------------|
| **Detailed metadata** | Full drc `ResearchResult`: provider, model, params, timing, template info |
| **Clear separation** | Dedicated archive; no mixing with human-only research |
| **Transparency** | Open repo, immutable results, evaluation history |
| **Trusted operation** | Community governance; potential for institutional adoption |

### Potential Synergies

DARMA and aiXiv could be complementary:
- **aiXiv**: Full AI-generated papers and proposals
- **DARMA**: Granular deep research queries that might feed into larger papers
- Cross-posting: DARMA results could be cited in aiXiv papers
- Shared evaluation: Agent evaluators could work across both platforms

## Export and Integration

### HuggingFace Datasets

DARMA's structured format (results + evaluations) makes it ideal for export to HuggingFace as training/evaluation datasets for research AI systems.

**Use cases:**
- **LLM fine-tuning**: High-quality research syntheses as training data
- **Evaluation benchmarks**: Results with human/agent evals become ground truth for assessing new models
- **RAG corpora**: Curated, evaluated research for retrieval-augmented generation

**Export format:**
```python
# Example HuggingFace dataset structure
{
    "id": "perplexity-crispr-mechanisms-a1b2c3d4",
    "query": "Research CRISPR mechanisms in human cells",
    "provider": "perplexity",
    "model": "sonar-deep-research",
    "markdown": "# CRISPR Mechanisms...",
    "citations": ["https://doi.org/...", ...],

    # Evaluation data (key for ML training)
    "eval_factual_accuracy": 0.85,
    "eval_citation_quality": 0.90,
    "eval_comprehensiveness": 0.75,
    "eval_count": 3,
    "human_eval_count": 2,
    "agent_eval_count": 1,

    # Metadata
    "domain": "molecular_biology",
    "keywords": ["CRISPR", "Cas9", "gene editing"],
    "timestamp": "2025-12-14T10:25:00Z"
}
```

**Potential dataset variants:**
| Dataset | Description | Use Case |
|---------|-------------|----------|
| `darma/full` | All results with evals | General training |
| `darma/high-quality` | Results with avg eval score > 0.8 | Fine-tuning on quality |
| `darma/human-evaluated` | Only human-evaluated results | Ground truth benchmarks |
| `darma/by-domain` | Domain-specific subsets | Specialized models |
| `darma/eval-pairs` | (result, evaluation) pairs | Training evaluation models |

**Automation:**
```yaml
# .github/workflows/export-huggingface.yml
# Runs weekly or on-demand
- Aggregate all results + evaluations
- Compute aggregate scores
- Filter by quality thresholds
- Push to huggingface.co/datasets/darma/research-results
```

### Zenodo Integration

For long-term archival and DOI minting, DARMA can export collections to Zenodo.

**Export triggers:**
- **Periodic snapshots**: Monthly/quarterly archive of all new results
- **Curated collections**: Thematic collections (e.g., "AI in Drug Discovery 2025")
- **Milestone releases**: Version-tagged releases (v1.0, v2.0)

**Zenodo metadata mapping:**
```yaml
# Example Zenodo deposit
title: "DARMA Collection: Molecular Biology Q4 2025"
description: "Curated deep research results on molecular biology topics"
creators:
  - name: "DARMA Community"
    affiliation: "DARMA Archive"
upload_type: "dataset"
publication_date: "2025-12-31"
keywords: ["AI research", "deep research", "molecular biology"]
license: "cc-by-4.0"
communities:
  - identifier: "darma"
related_identifiers:
  - identifier: "https://github.com/darma-archive/darma"
    relation: "isSupplementTo"
    scheme: "url"
```

**Collection types:**
| Collection | Trigger | DOI Scope |
|------------|---------|-----------|
| **Snapshots** | Quarterly | One DOI per quarter |
| **Thematic** | Manual curation | One DOI per collection |
| **Individual** | High-impact results | One DOI per result (optional) |

**Benefits:**
- **Citability**: DOIs for academic citation
- **Preservation**: Long-term archival independent of GitHub
- **Discoverability**: Indexed in OpenAIRE, DataCite, Google Dataset Search
- **Versioning**: Zenodo handles version DOIs automatically

### Export Pipeline

```
DARMA (GitHub)
     │
     ├──→ HuggingFace (weekly)
     │      └── Training datasets with evals
     │
     ├──→ Zenodo (quarterly + manual)
     │      └── Archived collections with DOIs
     │
     └──→ Static site (on push)
            └── Browse/search interface
```

### Future Integrations

| Platform | Purpose | Priority |
|----------|---------|----------|
| **HuggingFace** | ML datasets, model training | High |
| **Zenodo** | DOIs, long-term archival | High |
| **ORCID** | Contributor attribution | Medium |
| **OpenAlex** | Academic graph integration | Medium |
| **Wikidata** | Linked data, knowledge graph | Low |
| **IPFS** | Decentralized backup | Low |

## Roadmap

### Phase 1: MVP (GitHub-only)
- [ ] Repository setup with issue templates
- [ ] YAML schema definitions (synced with drc `ResearchResult`)
- [ ] GitHub Actions for validation
- [ ] Basic mkdocs site
- [ ] Manual submission workflow

### Phase 2: Automation
- [ ] `darma` CLI tool
- [ ] drc integration: `--submit-to-darma` flag
- [ ] Feeder sync automation (ai-gene-review, dismech)
- [ ] Automated citation URL validation
- [ ] HuggingFace export pipeline (weekly sync)

### Phase 3: Community & Evaluation
- [ ] Agent evaluation bots (LLM-as-judge)
- [ ] Human evaluation via GitHub issues
- [ ] Reputation/contribution tracking
- [ ] Domain-specific collections
- [ ] Cross-referencing between results
- [ ] HuggingFace dataset variants (high-quality, by-domain, eval-pairs)

### Phase 4: Archival & Journal-like Features
- [ ] Zenodo integration for quarterly snapshots
- [ ] DOI minting for curated collections
- [ ] Editorial curation for "highlights"
- [ ] Periodic digests/newsletters
- [ ] Integration with ORCID
- [ ] OpenAlex integration for academic graph

## Open Questions

1. **Versioning**: How to handle updates to results?
   - Option A: Immutable - new query = new result
   - Option B: Versioned with history

2. **Quality threshold**: Auto-accept all submissions, or require minimum evaluation?

3. **Agent evaluators**: Which LLM-as-judge patterns? How to distinguish agent vs human evals in aggregates?

4. **Near-duplicates**: Same query, different models - treat as separate or linked?

5. **Negative results**: Include "query failed" or "no useful results" entries?

6. **Licensing**: Default CC-BY-4.0, or submitter choice?

7. **Governance**: Who can merge PRs? Community maintainers model?

## References

### Primary Sources

- **Monperrus, M. (2025).** ["Preprint Servers for AI Generated Papers"](https://www.monperrus.net/martin/preprint-servers-of-AI-generated-papers). Personal blog. *Key reference for the landscape of AI research archives and requirements.*

- **Moons, P. et al. (2025).** "Google's AI co-scientist and OpenAI's deep research: new partners in health research?" [Eur. J. Cardiovasc. Nurs. 24(5): 800-807](https://academic.oup.com/eurjcn/article/24/5/800/8205996). *Overview of AI research tools and publishing implications.*

### Platforms

- [aiXiv](https://aixiv.science) - Open-source preprint server for AI scientists ([GitHub](https://github.com/aixiv-org/aiXiv), [backend](https://github.com/aixiv-org/aixiv-core))
- [AI Archive](https://ai-archive.io) - Publishing platform for AI agents (Fall 2025)
- [ai.vixra.org](http://ai.vixra.org) - AI-assisted article archive (March 2025)
- [rxiVerse](https://rxiverse.com) - AI preprint server with submission fees (July 2025)

### Related Infrastructure

- [deep-research-client](https://github.com/monarch-initiative/deep-research-client) - The underlying research client; provides data model
- [Micropublications](https://www.micropublication.org/) - Single-observation peer-reviewed publishing
- [Nanopublications](https://nanopub.net/) - Machine-readable scientific assertions
- [DeepEval](https://github.com/confident-ai/deepeval) - LLM evaluation framework

### Further Reading

- Lee, J. & Cho, K. (2023). "Defining the Boundaries of AI Use in Scientific Writing: A Comparative Review of Editorial Policies." [J. Korean Med. Sci. 40(e187)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12170296/)
- COPE, ICMJE, WAME guidelines on AI in scholarly publishing
