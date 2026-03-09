# Paper Plan: Comparative Evaluation of Deep Research Tools

**Status:** Planning
**Date:** 2026-03-09
**Authors:** TBD

## Working Title

"Benchmarking Agentic Deep Research: A Comparative Evaluation of AI-Powered Literature Synthesis Tools"

Alternative titles:
- "How Deep is Deep Research? Evaluating AI Research Agents Across Multiple Dimensions"
- "deep-research-client: A Unified Framework for Comparing AI Research Providers"

## Motivation

AI-powered "deep research" tools have rapidly emerged as alternatives to traditional literature
search and synthesis. Tools like OpenAI Deep Research, Perplexity, FutureHouse Edison/PaperQA3,
and Consensus each take different approaches — from web-augmented LLM reasoning to specialized
scientific literature agents. Yet there is **no standardized evaluation framework** for comparing
these tools, and **no published head-to-head comparisons** across multiple dimensions.

The `deep-research-client` (drc) provides a unique vantage point: a unified API wrapper that can
send identical queries to multiple providers and collect structured results with timing, citation,
and metadata information. This makes systematic comparison feasible.

## Research Questions

1. **How do deep research providers differ in output quality** across dimensions of accuracy,
   completeness, citation quality, and report structure?
2. **What are the cost/latency/quality trade-offs** between providers and model tiers?
3. **Does domain matter?** How do providers compare on biomedical vs. general science vs.
   non-scientific topics?
4. **Can we define a reusable evaluation framework** for benchmarking deep research tools?

## Providers Under Comparison

| Provider | API Model(s) | Source Type | Depth Tier | Cost Tier |
|----------|-------------|-------------|------------|-----------|
| **OpenAI** | o3-deep-research, o4-mini-deep-research | Web + code interpreter | Deep / Medium | Very High / Medium |
| **Perplexity** | sonar-deep-research, sonar-pro, sonar | Real-time web | Deep / Medium / Fast | High / Medium / Low |
| **Edison (FutureHouse)** | PaperQA3 | Scientific literature (full text) | Deep | High |
| **Consensus** | Academic Search API | Peer-reviewed papers only | Fast | Low |
| **Cyberian** | Agent-based (Claude/Aider) | Multi-source, iterative | Very Deep | Variable |

## Evaluation Framework

### Dimension 1: Citation Quality

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Citation count** | Number of unique references | Automated count from `ResearchResult.citations` |
| **Citation accuracy** | Do citations actually support claims? | Human evaluation (spot-check) + LLM-as-judge |
| **Source diversity** | Mix of source types (journals, preprints, web, etc.) | Automated classification |
| **Source recency** | Publication dates of cited works | Automated date extraction |
| **URL validity** | Do citation URLs resolve? | Automated HTTP checks |
| **DOI coverage** | Fraction of citations with DOIs | Automated regex extraction |
| **Hallucinated citations** | Citations that don't exist | Manual verification + CrossRef/PubMed lookup |

### Dimension 2: Depth

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Report length** | Word count, section count | Automated |
| **Subtopic coverage** | Number of distinct subtopics addressed | Human evaluation against reference outline |
| **Mechanistic detail** | Level of mechanistic/technical detail | Human Likert scale (1-5) |
| **Evidence synthesis** | Integration across sources vs. serial summarizing | Human evaluation |
| **Contradiction handling** | Does the report note conflicting evidence? | Human evaluation |

### Dimension 3: Breadth

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Perspective diversity** | Multiple viewpoints, schools of thought | Human evaluation |
| **Cross-disciplinary reach** | References from multiple fields | Automated journal classification |
| **Geographic diversity** | Research from different countries/regions | Metadata analysis |
| **Temporal breadth** | Coverage of historical vs. recent work | Citation date range analysis |

### Dimension 4: Accuracy

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Factual correctness** | Claims verified against sources | Human spot-check (sample of claims) |
| **Hallucination rate** | Fabricated facts, statistics, or citations | Human + LLM-as-judge |
| **Numerical accuracy** | Correct reporting of statistics/numbers | Human verification |
| **Attribution accuracy** | Claims correctly attributed to sources | Human verification |

### Dimension 5: Completeness

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Key finding recall** | Fraction of known key findings captured | Human evaluation against gold standard |
| **Gap identification** | Does the report note knowledge gaps? | Human evaluation |
| **Methodological coverage** | Are methods/approaches described? | Human evaluation |

### Dimension 6: Report Quality

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Structure** | Logical organization, headings, sections | Human + automated markdown analysis |
| **Coherence** | Logical flow between sections | Human Likert scale |
| **Readability** | Clarity of writing | Automated readability scores + human |
| **Actionability** | Can a researcher act on the information? | Human evaluation |

### Dimension 7: Operational

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Latency** | Time to completion | `ResearchResult.duration_seconds` |
| **Cost per query** | API cost | Provider billing data |
| **Reliability** | Success rate, error rate | Automated tracking |
| **Reproducibility** | Consistency across runs of same query | Multiple runs, similarity scoring |

## Proposed Study Design

### Query Set Design

Design a benchmark set of queries across domains and difficulty levels:

| Domain | Example Queries | Count |
|--------|----------------|-------|
| **Biomedical** | "Role of TP53 in cancer", "CRISPR off-target effects" | 10 |
| **General science** | "Climate change tipping points", "Quantum error correction" | 10 |
| **Current events** | "AI regulation in EU 2025", "mRNA vaccine developments" | 5 |
| **Interdisciplinary** | "Microbiome and mental health", "AI in drug discovery" | 5 |
| **Niche/specialized** | "Phenotype ontology interoperability", "Rare disease diagnostics" | 5 |

**Total: ~35 queries x N providers x M models = evaluation matrix**

### Query Complexity Levels

For a subset of topics, vary complexity:
- **Simple factual**: "What is X?"
- **Comparative**: "Compare X and Y approaches"
- **Synthetic**: "What is the evidence for X causing Y?"
- **Prospective**: "What are the open questions in X?"
- **Methodological**: "What methods are used to study X?"

### Evaluation Protocol

1. **Automated metrics** (run first):
   - Citation count, URL validity, DOI coverage
   - Report length, section count
   - Latency, cost
   - Readability scores
   - Citation date distribution

2. **LLM-as-judge evaluation**:
   - Factual accuracy spot-check (claim extraction + verification)
   - Hallucination detection
   - Completeness against reference key findings
   - Structure and coherence scoring

3. **Human expert evaluation** (subset):
   - Domain experts score a subset of reports
   - Inter-annotator agreement measured
   - Focus on dimensions where LLM-as-judge is less reliable

### Statistical Analysis

- **Per-dimension comparison**: ANOVA or Kruskal-Wallis across providers
- **Radar/spider plots**: Multi-dimensional profiles per provider
- **Cost-quality Pareto frontiers**: Identify optimal cost/quality trade-offs
- **Domain interaction effects**: Do provider rankings change by domain?

## Data Collection Infrastructure

The `deep-research-client` already provides most of what we need:

```bash
# Run same query across all providers
for provider in openai perplexity falcon consensus; do
  deep-research-client research \
    --template benchmarks/query_template.md \
    --var "query=..." \
    --provider $provider \
    --output "results/${provider}/${query_id}.md"
done
```

### What We Already Have (from drc)

- `ResearchResult.citations` — citation list
- `ResearchResult.duration_seconds` — latency
- `ResearchResult.provider` / `.model` — provenance
- `ResearchResult.markdown` — full report content
- Caching infrastructure — avoid re-running identical queries

### What We Need to Build

- **Evaluation harness**: Automated metrics computation on `ResearchResult` objects
- **LLM-as-judge pipeline**: Prompt templates for each evaluation dimension
- **Gold standard annotations**: Reference key findings for benchmark queries
- **Visualization**: Comparison plots, radar charts, tables
- **Statistical tests**: Provider comparison analysis

## Related Work (Literature Review from Web Search)

### Key Benchmarks and Evaluation Frameworks

#### 1. DeepResearch Bench (DRB / DRB II)
- **Source**: [Ayanami0730/deep_research_bench](https://github.com/Ayanami0730/deep_research_bench)
- **Scope**: 100 PhD-level research tasks across 22 fields
- **DRB II** (Feb 2026): 9,430 fine-grained binary rubrics covering information recall, analysis, and presentation
- **Two evaluation methodologies**:
  - **RACE** (Reference-based Adaptive Criteria-driven Evaluation): Evaluates report quality across
    Comprehensiveness, Insight/Depth, Instruction-Following, Readability with dynamic weights
  - **FACT** (Framework for Factual Abundance and Citation Trustworthiness): Extracts factual claims,
    verifies source support, calculates Citation Accuracy and Effective Citations
- **Leaderboard** (Feb 2026): Qianfan-DeepResearch Pro (#1), Tavily Research, CellCog.ai, Salesforce Enterprise
- **Relevance**: Most directly comparable to our evaluation goals; we should align with or extend RACE/FACT

#### 2. DRACO Benchmark (Perplexity Research)
- **Source**: [Perplexity Research](https://research.perplexity.ai/articles/evaluating-deep-research-performance-in-the-wild-with-the-draco-benchmark)
- **Scope**: 100 anonymized, open-ended, objectively evaluable tasks
- **Dimensions**: Factual accuracy, breadth/depth/completeness, presentation quality/objectivity, citation quality
- **Systems evaluated**: Perplexity Deep Research, OpenAI Deep Research (o3, o4-mini), Gemini Deep Research, Claude Opus
- **Note**: Created by Perplexity — potential conflict of interest, but methodology is useful

#### 3. PaperQA / PaperQA2 (FutureHouse)
- **Paper**: Skarlinski et al. (2024) "Language agents achieve superhuman synthesis of scientific knowledge" [arXiv:2409.13740](https://arxiv.org/abs/2409.13740)
- **Benchmarks introduced**: LitQA, LitQA2 — requiring retrieval and synthesis from full-text papers
- **Key result**: PaperQA2 matches or exceeds subject matter experts on LitQA2 (precision 85.2%, accuracy 66.0%)
- **ContraCrow**: Automated contradiction detection (2.34 contradictions/paper, 70% validated by experts)
- **Relevance**: Edison/Falcon in our client is based on this; LitQA2 is a gold standard for scientific QA

#### 4. PaperArena
- **Paper**: Wang et al. (2025) [arXiv:2510.10909](https://arxiv.org/html/2510.10909v2)
- **Scope**: 784 questions evaluating tool-augmented agentic reasoning on scientific literature
- **Metrics**: Correctness (LLM-as-judge), Reasoning Steps (tool invocations), Reasoning Efficiency
- **Capabilities tested**: Multi-step reasoning, multi-modal understanding, cross-document integration, database interfacing
- **Key finding**: Even Gemini 2.5 Pro achieves only 38.78% accuracy vs 83.5% for human PhD experts

#### 5. SciArena
- **Paper**: Zhao et al. (2025) [arXiv:2507.01001](https://arxiv.org/abs/2507.01001) — NeurIPS 2025 Spotlight
- **Design**: Community-driven evaluation via researcher voting (like Chatbot Arena for science)
- **Scope**: 47 foundation models, 20,000+ human votes
- **Focus**: Open-ended scientific tasks requiring literature-grounded, long-form responses
- **Also provides**: SciArena-Eval meta-benchmark for evaluating automated assessment systems

#### 6. Search Arena / LM Arena
- Perplexity Sonar models rank #1-4 in Search Arena (10,000+ human preference votes, March-April 2025)
- Focus on current events with longer, more complex prompts

#### 7. Standard Benchmarks Used for Deep Research
- **Humanity's Last Exam**: Perplexity Deep Research 21.1%, OpenAI 17.8% (26.6% in other reports)
- **SimpleQA**: Perplexity Deep Research 93.9% factual accuracy
- **LitQA2**: PaperQA2 66.0% accuracy, 85.2% precision

### Evaluation Frameworks for LLM Systems (General)

- **DeepEval** (Confident AI): Open-source LLM evaluation framework — faithfulness, answer relevancy,
  hallucination, contextual recall/precision
- **RAGAS**: RAG evaluation framework (precision, recall, faithfulness)
- **LLM-as-judge**: Using LLMs to evaluate LLM outputs (Zheng et al. 2023)

### Published Head-to-Head Comparisons

| Source | Tools Compared | Evaluation Method | Key Finding |
|--------|---------------|-------------------|-------------|
| DRACO (Perplexity) | Perplexity, OpenAI, Gemini, Claude | 100 tasks, 4 dimensions | Perplexity competitive |
| DRB II | 10+ DRAs | RACE + FACT | Qianfan Pro leads |
| Gradient Flow | OpenAI, Gemini, Grok, Perplexity | Qualitative | OpenAI leads capability |
| Moons et al. (2025) | OpenAI DR, Google AI Co-Scientist | Descriptive | Overview of AI in health research |
| AIMultiple | Claude, ChatGPT, Grok | Side-by-side | Qualitative comparison |
| God of Prompt | Perplexity vs OpenAI | Direct comparison | Perplexity cheaper/faster, OpenAI deeper |

#### 8. OpenScholar + ScholarQABench (Asai et al., Nov 2024)
- **Paper**: [arXiv:2411.14199](https://arxiv.org/abs/2411.14199) — 70 citations
- **System**: RAG over 45M open-access papers
- **Key finding**: GPT-4o hallucinates citations 78-90%; OpenScholar matches expert citation accuracy
- **Expert preference**: OpenScholar-GPT4o preferred over expert-written responses 70% of the time

#### 9. Answer Engine Evaluation / AEE (Venkit et al., Oct 2024)
- **Paper**: [arXiv:2410.22349](https://arxiv.org/abs/2410.22349) — 19 citations
- **Systems**: Perplexity, Bing Copilot, You.com
- **Key findings**: Citation accuracy: You 68.3%, Bing 65.8%, **Perplexity 49.0%**; 23-32% unsupported statements
- **Novel**: Includes UX/interaction metrics (hovering, clicking under contradictions)

#### 10. DeepScholar-Bench (Patel et al., Aug 2025)
- **Paper**: [arXiv:2508.20033](https://arxiv.org/abs/2508.20033) — 12 citations
- **7 metrics**: Organization, Nugget Coverage, Relevance Rate, Document Importance, Reference Coverage, Citation Precision, Claim Coverage
- **Key finding**: No system scored >0.19 across all metrics

#### 11. LiveDRBench (Java et al., Aug 2025)
- **Paper**: [arXiv:2508.04183](https://arxiv.org/abs/2508.04183) — 16 citations
- **Evaluated**: OpenAI/Perplexity/Google deep research
- **Metrics**: Claim-level modified precision/recall; OpenAI highest average F1: 0.55

#### 12. Evaluating Verifiability in Generative Search Engines (Liu et al., 2023)
- **Paper**: [arXiv:2304.09848](https://arxiv.org/abs/2304.09848) — **345 citations** (seminal work)
- Foundational paper on citation precision/recall in generative search

### Known Cost/Speed Trade-offs (from literature)

| Provider | Typical Latency | Citations per Report | Monthly Cost |
|----------|----------------|---------------------|-------------|
| Perplexity Deep Research | ~3 min | ~50 | $20/mo |
| OpenAI Deep Research | 7-20 min | ~20 | $200/mo |
| Perplexity sonar-pro | seconds | ~10-15 | API pricing |
| Edison/PaperQA | 5-25 min | ~14.5 papers/query | API pricing |

### Three-Tier Benchmark Framework (from Edison research)

**Tier A: Literature-grounded QA/synthesis (scientific PDFs)**
- LitQA/LitQA2 (PaperQA): correctness + DOI recall + refusal option
- ScholarQABench (OpenScholar): multi-domain synthesis + citation accuracy + expert preference

**Tier B: Web deep-research (multi-step browsing + synthesis)**
- DRB (Bosse et al.): frozen web corpus (RetroSearch), trace analysis
- DeepResearch Bench (Du et al.): RACE/FACT metrics
- LiveDRBench (Java et al.): claim-level F1

**Tier C: Generative search engines ("answer engines")**
- AEE (Venkit et al.): user-perceived limitations → measurable metrics

### Gaps Our Paper Can Fill

1. **No unified framework** combining RACE, FACT, and domain-specific evaluation with cost analysis
2. **No open-source evaluation harness** that runs across commercial providers via unified API
3. **Limited biomedical-specific evaluation** — most benchmarks are general science
4. **No comparison including Edison/Falcon** alongside commercial deep research tools
5. **No analysis using identical queries** sent simultaneously to all providers
6. **drc provides unique infrastructure**: same query → multiple providers → structured comparison
7. **No paper evaluates the same tool (PaperQA) alongside the web-scale tools** it was designed to surpass

## Paper Structure (Draft Outline)

1. **Introduction**
   - Rise of AI deep research tools
   - The evaluation gap
   - Contribution: framework + empirical comparison

2. **Background**
   - Landscape of deep research tools (Table 1)
   - Related evaluation frameworks
   - The deep-research-client as evaluation infrastructure

3. **Evaluation Framework**
   - Seven dimensions defined above
   - Metric definitions and measurement protocols
   - Automated vs. human evaluation components

4. **Methods**
   - Query set design
   - Provider configurations
   - Data collection protocol
   - Statistical analysis plan

5. **Results**
   - Automated metrics comparison (Table/Figure)
   - LLM-as-judge results
   - Human expert evaluation results
   - Domain-specific analysis
   - Cost-quality trade-off analysis

6. **Discussion**
   - Provider strengths and weaknesses
   - When to use which tool
   - Limitations of our evaluation
   - Toward a community benchmark (DARMA connection)

7. **Conclusion**

## Data Requirements

### Data We Need to Collect (not in repo yet)

| Data Type | Source | Format | Notes |
|-----------|--------|--------|-------|
| Benchmark query results | Run drc across providers | ResearchResult YAML/JSON | ~35 queries x 4-6 providers |
| Automated metrics | Computed from results | CSV/JSON | Citation counts, lengths, etc. |
| LLM-as-judge scores | Evaluation pipeline | CSV/JSON | Per-dimension scores |
| Human expert scores | Manual annotation | CSV/JSON | Subset of queries |
| Cost data | Provider billing | CSV | Per-query costs |
| Gold standard key findings | Expert curation | Markdown/YAML | Per benchmark query |

### Connection to DARMA

The DARMA proposal (see `docs/ideas/deep-research-archive.md`) already defines an evaluation
object schema that aligns well with this paper's framework. The benchmark results could become
the seed dataset for DARMA.

## Target Venues

- **Bioinformatics / ISMB** — if biomedical focus is strong
- **JCDL / SIGIR** — digital libraries / information retrieval
- **NeurIPS / ICML workshop** — AI tools track
- **arXiv preprint** — for speed
- **JOSS** — if the evaluation framework becomes a standalone tool

## Next Steps

1. [ ] Finalize benchmark query set (35 queries across domains)
2. [ ] Run queries across all available providers (collect results)
3. [ ] Build automated metrics computation module
4. [ ] Design LLM-as-judge evaluation prompts
5. [ ] Curate gold standard key findings for subset of queries
6. [ ] Run evaluations and generate comparison tables/figures
7. [ ] Recruit domain experts for human evaluation subset
8. [ ] Statistical analysis and visualization
9. [ ] Draft manuscript
10. [ ] Review deep research output for additional related work
