---
provider: falcon
model: Edison Scientific Literature
cached: false
start_time: '2026-03-09T01:14:28.712397'
end_time: '2026-03-09T01:39:57.399094'
duration_seconds: 1528.69
provider_config:
  timeout: null
  max_retries: 3
  parameters:
    allowed_domains: []
    temperature: 0.1
citation_count: 41
---

## Question

What published papers or benchmarks evaluate AI deep research and literature synthesis tools? Include PaperQA, OpenAI deep research, Perplexity, and similar agentic research systems. Focus on evaluation metrics and comparison frameworks.

## Output

Question: You are an expert researcher providing comprehensive, well-cited information.

Provide detailed information focusing on:
1. Key concepts and definitions with current understanding
2. Recent developments and latest research (prioritize 2023-2024 sources)
3. Current applications and real-world implementations
4. Expert opinions and analysis from authoritative sources
5. Relevant statistics and data from recent studies

Format as a comprehensive research report with proper citations. Include URLs and publication dates where available.
Always prioritize recent, authoritative sources and provide specific citations for all major claims.

What published papers or benchmarks evaluate AI deep research and literature synthesis tools? Include PaperQA, OpenAI deep research, Perplexity, and similar agentic research systems. Focus on evaluation metrics and comparison frameworks.

## Research report: Benchmarks and papers evaluating AI deep-research & literature-synthesis systems (2023–2024 priority)

### Executive summary
Evaluation of “deep research” and literature-synthesis agents has converged on a few recurring measurement axes: (i) *task correctness* on research questions; (ii) *retrieval quality and coverage*; (iii) *verifiability and citation quality* (citation precision/recall, supported-sentence rates); (iv) *synthesis quality* (organization and nugget/claim coverage); (v) *human factors* (preference, effort, interaction behaviors); and (vi) *process/trace* properties (hallucination, tool use, forgetting). Across 2023–2024, the best-publicly-documented agentic systems (notably PaperQA2 and OpenScholar) report strong performance on literature-specific benchmarks, while commercial answer engines frequently show weaker citation accuracy and substantial unsupported content under controlled audits. (skarlinski2024languageagentsachieve pages 3-5, asai2024openscholarsynthesizingscientific pages 2-4, venkit2024searchenginesin pages 14-15)

| Item | Date | URL/DOI | What is evaluated | Key metrics | Example quantitative results |
|---|---|---|---|---|---|
| PaperQA (LitQA + biomed QA) | 2023-12 | https://doi.org/10.48550/arxiv.2312.07559 | RAG agent answering scientific questions vs. GPT-4/agents (lala2023paperqaretrievalaugmentedgenerative pages 9-11) | Accuracy on MedQA/BioASQ/PubMedQA; citation hallucination rate | MedQA 68.0, BioASQ 89.0, PubMedQAblind 86.3; 0% hallucinated citations (lala2023paperqaretrievalaugmentedgenerative pages 9-11) |
| PaperQA2 / LitQA2 | 2024-09 | https://doi.org/10.48550/arxiv.2409.13740 | Hard full-text literature QA + synthesis; human vs agent on retrieval/summarization/contradiction (skarlinski2024languageagentsachieve pages 1-3, skarlinski2024languageagentsachieve pages 3-5) | Precision (answered-correct), Accuracy (overall), DOI Recall; contradiction detection/validation | Precision 85.2%±1.1, Accuracy 66.0%±1.2; Humans: Precision 73.8%±9.6, Accuracy 67.7%±11.9; 2.34±1.99 contradictions/paper with 70% validated (skarlinski2024languageagentsachieve pages 3-5, skarlinski2024languageagentsachieve pages 1-3) |
| OpenScholar / ScholarQABench | 2024-11 | https://doi.org/10.48550/arxiv.2411.14199 | Retrieval-augmented literature synthesis over 45M papers; multi-domain benchmark with expert long-form answers (asai2024openscholarsynthesizingscientific pages 2-4) | Correctness, citation accuracy, coverage/coherence, human preference | +7% correctness vs PaperQA2; GPT-4o citation hallucination 78–90%; expert preference: 51% (OpenScholar-8B), 70% (OpenScholar-GPT4o) vs expert refs (asai2024openscholarsynthesizingscientific pages 2-4) |
| Venkit et al., Answer Engine Evaluation (AEE) | 2024-10 | https://doi.org/10.48550/arxiv.2410.22349 | Generative search engines (Perplexity, Bing Copilot, You.com): source-cited answer quality + UX (venkit2024searchenginesin pages 8-10, venkit2024searchenginesin pages 14-15) | Citation accuracy/thoroughness; unsupported statements; one-sidedness/overconfidence; interaction metrics | Citation accuracy: You 68.3%, Bing 65.8%, Perplexity 49.0; Unsupported statements 23–32%; significant increases in source hovering/clicking under contradictions (venkit2024searchenginesin pages 14-15, venkit2024searchenginesin pages 8-10) |
| Knollmeyer et al., RAG eval SLR | 2024-01 | https://doi.org/10.5220/0013065700003838 | Systematic review of RAG evaluation dimensions/metrics/datasets (knollmeyer2024benchmarkingofretrieval pages 8-9, knollmeyer2024benchmarkingofretrieval pages 9-10) | Correctness, faithfulness, answer/context relevance; citation precision/recall; Recall@k/MRR; LLM-as-judge (with pros/cons) | Framework/metrics synthesis (no single benchmark score); notes limits of BERTScore vs recall for correctness (knollmeyer2024benchmarkingofretrieval pages 8-9, knollmeyer2024benchmarkingofretrieval pages 9-10) |
| Bosse et al., Deep Research Bench (DRB) | 2025-05 | https://doi.org/10.48550/arxiv.2506.06287 | 89 multi-step web research tasks; offline RetroSearch; evaluates commercial deep-research products incl. OpenAI Deep Research (bosse2025deepresearchbench pages 1-2) | Automated trace analysis: hallucination, tool use, forgetting; task success; leaderboard | Public leaderboard; comparable offline vs live-web performance; no per-system numbers in paper text excerpt (bosse2025deepresearchbench pages 1-2) |
| Du et al., DeepResearch Bench | 2025-06 | https://doi.org/10.48550/arxiv.2506.11763 | 100 expert-crafted deep-research tasks across 22 fields; new eval rubrics RACE/FACT (du2506deepresearchbencha pages 1-3) | RACE: report quality via adaptive criteria; FACT: effective citation count, citation accuracy (with human validation) | Reports agent citation accuracy and effective-citation counts; methods validated with human studies (du2506deepresearchbencha pages 1-3) |
| Patel et al., DeepScholar‑Bench | 2025-08 | https://doi.org/10.48550/arxiv.2508.20033 | Live benchmark for related‑work synthesis; automated 7-metric evaluation across synthesis/retrieval/verifiability (patel2025deepscholarbenchalive pages 2-3, patel2025deepscholarbenchalive pages 1-2) | Organization, Nugget Coverage, Relevance Rate, Document Importance, Reference Coverage, Citation Precision, Claim Coverage | No system >0.19 across all metrics; baseline pipeline provided; strong human‑agreement for metrics (patel2025deepscholarbenchalive pages 2-3, patel2025deepscholarbenchalive pages 1-2) |
| Gwon et al., SR literature search | 2024-05 | https://doi.org/10.2196/51187 | ChatGPT & Bing AI vs human SR search (Peyronie’s disease) using human query formula (gwon2024theuseof pages 1-2) | Relevance rate; bibliographic grading (A/B/C/F) | ChatGPT: 7/1287 relevant (0.5%); Bing AI: 19/48 relevant (40%); output grade distributions reported (gwon2024theuseof pages 1-2) |
| Meliante et al., AI vs PRISMA (glaucoma SLRs) | 2025-09 | https://doi.org/10.2196/68592 | Elicit/Connected Papers (identification), Elicit/ChatPDF (extraction), Jenni AI (composition) vs PRISMA (meliante2025evaluationofai pages 1-2, meliante2025evaluationofai pages 4-6) | Extraction accuracy/precision/missing/incorrect; retrieval coverage vs PRISMA; composition quality scores | Elicit extraction: 51.4% accurate (±31.45), 22.4% missing, 12.5% incorrect; ChatPDF: 60.3% accurate (±30.7); AI tools failed to retrieve all PRISMA results (meliante2025evaluationofai pages 1-2, meliante2025evaluationofai pages 4-6) |


*Table: Compact overview of published evaluations and benchmarks for AI deep-research and literature-synthesis systems, highlighting tasks, metrics, and representative results. Citations link to source papers; the table helps compare systems like PaperQA, OpenAI Deep Research, Perplexity, and others across consistent evaluation dimensions.*

---

## 1) Key concepts and definitions (current understanding)

### 1.1 “Deep research” / “agentic research” systems
In the 2023–2025 literature, deep-research agents are typically defined operationally as LLM-based systems that iteratively plan, retrieve (web or paper corpora), and synthesize *long-form, citation-backed* reports rather than answering single-shot factoids. Core behaviors include multi-step exploration, targeted retrieval, and report-style synthesis. Benchmarks such as Deep Research Bench explicitly target this multi-step web exploration + synthesis setting and emphasize controlling the changing web via frozen corpora (RetroSearch). (bosse2025deepresearchbench pages 1-2)

### 1.2 Literature-synthesis agents (scientific RAG agents)
Scientific literature agents are a narrower class: they retrieve and read *full-text papers*, then synthesize answers with provenance. PaperQA and PaperQA2 frame the task as answering research questions over scientific PDFs, emphasizing reduced hallucination and transparent citations. (lala2023paperqaretrievalaugmentedgenerative pages 9-11, skarlinski2024languageagentsachieve pages 1-3)

### 1.3 Verifiability and citation metrics
A widely used conceptual split is:
- **Citation precision**: fraction of provided citations that truly support the associated claim/sentence.
- **Citation recall / thoroughness**: degree to which *all* verification-worthy claims are supported by citations.
These are used in audits of answer engines (e.g., AEE) and in RAG evaluation frameworks and reviews. (venkit2024searchenginesin pages 14-15, knollmeyer2024benchmarkingofretrieval pages 8-9)

A critical nuance in recent work is that *correctness of a statement* is not the same as *faithfulness of attribution* (whether the model truly relied on the cited evidence), motivating separate measurement of correctness vs faithfulness. (knollmeyer2024benchmarkingofretrieval pages 8-9)

---

## 2) Recent developments and latest research (prioritizing 2023–2024)

### 2.1 PaperQA (Dec 2023): LitQA + biomedical QA benchmarks
**PaperQA** introduced an agent that retrieves across full-text papers and answers with citations, and it introduced **LitQA** as a harder full-text retrieval benchmark (50 questions). PaperQA also evaluated on PubMedQA (context withheld), MedQA, and BioASQ. Reported scores (100 sampled questions per dataset) showed PaperQA outperforming GPT-4 on these science/biomedical QA settings (e.g., PubMedQAblind 86.3 vs GPT-4 57.9) and reporting *no hallucinated citations* vs literature-reported high citation-hallucination rates for non-grounded LLMs. (lala2023paperqaretrievalaugmentedgenerative pages 9-11)

**Why it matters for evaluation**: PaperQA combines (a) standard benchmark accuracy and (b) explicit citation-hallucination auditing as a trust metric. (lala2023paperqaretrievalaugmentedgenerative pages 9-11)

### 2.2 PaperQA2 + LitQA2 (Sep 2024): expert-level literature QA and human comparisons
Skarlinski et al. introduced **LitQA2** (248 expert-generated multiple-choice questions) with constraints that answers appear in the *main body* (not abstracts) and are ideally unique in the literature. They define three key metrics:
- **Precision**: fraction correct *given the system answered*.
- **Accuracy**: fraction correct overall.
- **DOI recall**: fraction of answers attributed to the correct source DOI (i.e., source-level retrieval attribution). (skarlinski2024languageagentsachieve pages 1-3)

They report **PaperQA2** performance and a direct human comparison:
- PaperQA2: precision **85.2% ± 1.1**, accuracy **66.0% ± 1.2**, and it chose “insufficient information” **21.9% ± 0.9**; it parsed ~**14.5 ± 0.6** papers per question. (skarlinski2024languageagentsachieve pages 3-5)
- Human experts: precision **73.8% ± 9.6**, accuracy **67.7% ± 11.9**. PaperQA2 had significantly higher precision than humans and similar accuracy. (skarlinski2024languageagentsachieve pages 3-5)

Beyond QA, they evaluate contradiction detection: PaperQA2 found **2.34 ± 1.99** contradictions per paper in a sample of **93** biology papers, with **70%** of identified contradictions validated by human experts. (skarlinski2024languageagentsachieve pages 1-3)

**Inclusion of Perplexity**: The same work reports running **Perplexity Pro (GPT-4o)** manually over all 248 LitQA2 questions and scoring it with the same metrics as PaperQA2 (accuracy/precision/DOI-recall framework), though the provided excerpt does not include Perplexity’s final numeric score table. (skarlinski2024languageagentsachieve pages 16-18)

### 2.3 OpenScholar + ScholarQABench (Nov 2024): multi-domain literature synthesis with citation accuracy emphasis
Asai et al. introduced **OpenScholar**, a retrieval-augmented LM over an open-access datastore (reported as **45M** papers) and a new benchmark **ScholarQABench** for multi-domain literature search/synthesis with expert-authored long-form answers. Evaluation emphasizes **correctness** and **citation accuracy** and includes expert preference tests.

Key reported results include:
- OpenScholar correctness improvements vs baselines: OpenScholar outperforms GPT-4o and reports improvements relative to PaperQA2 and Perplexity Pro (as summarized in the excerpt). (asai2024openscholarsynthesizingscientific pages 2-4)
- Citation behavior: the paper reports GPT-4o “hallucinates citations” at **78–90%** in their setting, while OpenScholar achieves citation accuracy comparable to human experts (as stated in the excerpt). (asai2024openscholarsynthesizingscientific pages 2-4)
- Expert preference: OpenScholar-8B and OpenScholar-GPT4o are preferred over expert-written responses **51%** and **70%** of the time, respectively, while GPT-4o alone is preferred **31%** of the time. (asai2024openscholarsynthesizingscientific pages 2-4)

**Why it matters**: ScholarQABench is explicitly designed as a literature-synthesis benchmark rather than a standard QA benchmark, and it couples correctness with citation accuracy and human preference. (asai2024openscholarsynthesizingscientific pages 2-4)

### 2.4 Answer engines and web research audits (2024)
#### Answer Engine Evaluation (AEE) benchmark (Oct 2024)
Venkit et al. introduce an **Answer Engine Evaluation (AEE)** with **303** queries and compute metrics that connect user-reported limitations to measurable indicators. They evaluate **You.com, BingChat/Copilot, and Perplexity**.

Representative metrics and results include:
- **Unsupported statements**: ~**23–32%** across engines. (venkit2024searchenginesin pages 14-15)
- **Citation accuracy**: You **68.3%**, Bing **65.8%**, Perplexity **49.0%**. (venkit2024searchenginesin pages 14-15)
- **Citation thoroughness**: ~**20–24%**. (venkit2024searchenginesin pages 14-15)
- Also tracked: one-sidedness, overconfidence, sources per answer, citations per statement. (venkit2024searchenginesin pages 14-15)

#### UX/interaction measures for answer engines
The same paper reports user-study-derived interaction measures (e.g., users hovered and clicked more sources under contradictory questions than aligned questions, with statistically significant differences), tying evaluation to *verification effort* rather than only output quality. (venkit2024searchenginesin pages 8-10)

---

## 3) Current applications and real-world implementations

### 3.1 Scientific literature Q&A and synthesis
- **PaperQA/PaperQA2**: positioned as scientific research assistants for literature Q&A, synthesis, and contradiction discovery, with explicit citations and benchmark-driven development (LitQA/LitQA2). (lala2023paperqaretrievalaugmentedgenerative pages 9-11, skarlinski2024languageagentsachieve pages 1-3)
- **OpenScholar**: an open retrieval-augmented approach over very large scholarly corpora, evaluated on a dedicated literature-synthesis benchmark, with emphasis on citation accuracy and expert preference. (asai2024openscholarsynthesizingscientific pages 2-4)

### 3.2 Generative search / “answer engines” in the wild
- Commercial “answer engines” (Perplexity, Bing Copilot, You.com) are evaluated as consumer-facing web research products where *citations exist but are often incomplete or inaccurate*, and users must expend additional verification effort. (venkit2024searchenginesin pages 14-15, venkit2024searchenginesin pages 8-10)

### 3.3 Systematic review workflows (screening, extraction, and writing)
Peer-reviewed evaluations in healthcare show that present-day AI tools can assist but often fail to match PRISMA rigor.
- **Meliante et al. (Sep 2025)**: tested **Connected Papers** and **Elicit** for identification; **Elicit** and **ChatPDF** for extraction; and **Jenni AI** for composition. They report Elicit extraction averaged **51.40%** accurate (SD 31.45%), **22.37%** missing, **12.51%** incorrect; ChatPDF averaged **60.33%** accurate (SD 30.72%). Tools did not retrieve the full PRISMA set. (meliante2025evaluationofai pages 1-2, meliante2025evaluationofai pages 4-6)
- **Gwon et al. (May 2024)**: compared ChatGPT vs Bing AI against a human systematic review benchmark; ChatGPT returned **7/1287** directly relevant studies (**0.5%**) vs Bing AI **19/48** (**40%**), using a bibliographic grading scheme. (gwon2024theuseof pages 1-2)

These studies illustrate “real-world implementations” where evaluation is grounded in *replicating a published systematic review*, but they also underscore that deep-research-style tools still require expert oversight for high-stakes evidence synthesis. (meliante2025evaluationofai pages 1-2, gwon2024theuseof pages 1-2)

---

## 4) Expert opinions and analysis from authoritative sources

### 4.1 RAG evaluation dimensions and metric standardization
Knollmeyer et al. (Jan 2024) synthesize evaluation practice for RAG systems and highlight that robust evaluation requires separating retrieval vs generation and measuring multiple dimensions:
- **Correctness**, **faithfulness**, **context relevance**, **answer relevance**, and **citation quality**.
- Citation quality is commonly operationalized with **citation precision** and **citation recall**; retrieval quality via **Recall@k** and **MRR@k**.
- The review notes that LLM-as-a-judge approaches can better capture multi-hop, multi-context cases but introduce latency/cost and dependence on the judge model. (knollmeyer2024benchmarkingofretrieval pages 9-10, knollmeyer2024benchmarkingofretrieval pages 8-9)

This aligns with how dedicated literature-synthesis benchmarks (LitQA2, ScholarQABench, DeepScholar-Bench) increasingly use multi-metric and often human-validated evaluation. (skarlinski2024languageagentsachieve pages 3-5, asai2024openscholarsynthesizingscientific pages 2-4, patel2025deepscholarbenchalive pages 2-3)

### 4.2 “Citations are not enough” in answer engines
Venkit et al. argue that merely providing citations does not ensure factual/verifiable responses; their AEE audit finds substantial unsupported content and low citation thoroughness, and their user study identifies limitations such as limited user control over sources and extra verification work. (venkit2024searchenginesin pages 14-15, venkit2024searchenginesin pages 8-10)

---

## 5) Relevant statistics and data (recent studies)

### 5.1 Literature QA / synthesis
- PaperQA (Dec 2023): PubMedQAblind **86.3** vs GPT-4 **57.9** (100-question sample); also reports **0** hallucinated citations in its outputs. (lala2023paperqaretrievalaugmentedgenerative pages 9-11)
- PaperQA2 (Sep 2024): precision **85.2% ± 1.1** vs human **73.8% ± 9.6**; accuracy similar to humans (**66.0% ± 1.2** vs **67.7% ± 11.9**). (skarlinski2024languageagentsachieve pages 3-5)
- PaperQA2 contradiction detection: **2.34 ± 1.99** contradictions per paper; **70%** validated by experts. (skarlinski2024languageagentsachieve pages 1-3)
- OpenScholar (Nov 2024): GPT-4o citation hallucination **78–90%** (in their study setting), while OpenScholar reports citation accuracy on par with human experts; expert preference wins up to **70%** in pairwise comparisons (depending on system variant). (asai2024openscholarsynthesizingscientific pages 2-4)

### 5.2 Answer engine audits
- AEE (Oct 2024): citation accuracy You **68.3%**, Bing **65.8%**, Perplexity **49.0%**; citation thoroughness ~**20–24%**; unsupported statements ~**23–32%**. (venkit2024searchenginesin pages 14-15)

### 5.3 Systematic-review workflow tasks
- Gwon et al. (May 2024): ChatGPT **0.5%** directly relevant (7/1287) vs Bing AI **40%** relevant (19/48) on a benchmark systematic-review query formula. (gwon2024theuseof pages 1-2)
- Meliante et al. (Sep 2025): Elicit extraction **51.4%** accurate vs ChatPDF **60.3%** accurate over 1,274 extraction items; substantial missing/incorrect fractions. (meliante2025evaluationofai pages 1-2, meliante2025evaluationofai pages 4-6)

---

## 6) Benchmarks and comparison frameworks: what to use and how to compare tools

### 6.1 Three-tier comparison framework (recommended)
**Tier A: Literature-grounded QA/synthesis (scientific PDFs / scholarly corpora)**
- **LitQA/LitQA2** (PaperQA): measures correctness plus source attribution via DOI recall; includes refusal option for insufficient information. (skarlinski2024languageagentsachieve pages 1-3)
- **ScholarQABench** (OpenScholar): multi-domain literature search + long-form synthesis; emphasizes correctness and citation accuracy; uses expert preference. (asai2024openscholarsynthesizingscientific pages 2-4)

**Tier B: Web deep-research (dynamic web; multi-step browsing and synthesis)**
- **Deep Research Bench (DRB)**: evaluates research products such as OpenAI Deep Research using a frozen web corpus (RetroSearch) and analyzes agent traces for hallucination/tool-use/forgetting. (bosse2025deepresearchbench pages 1-2)
- **DeepResearch Bench (Du et al.)**: proposes RACE (report-quality rubric) and FACT (effective citation count, citation accuracy) for 100 expert-crafted tasks. (du2506deepresearchbencha pages 1-3)
- **LiveDRBench**: evaluates OpenAI/Perplexity/Google deep-research systems with claim-level modified precision/recall; reports OpenAI highest average F1 **0.55** in the excerpt. (java2025characterizingdeepresearch pages 3-5)

**Tier C: Generative search engines (“answer engines”)**
- **AEE**: aligns user-perceived limitations to measurable metrics (unsupported statements, citation accuracy/thoroughness, one-sidedness, overconfidence) on curated query sets. (venkit2024searchenginesin pages 14-15)

### 6.2 Metrics that best discriminate deep-research quality
Drawing from benchmark definitions and RAG evaluation surveys, the following metric families are most diagnostic:
1. **Correctness / task success**: accuracy, rubric-based correctness, or claim-level scoring. (skarlinski2024languageagentsachieve pages 3-5, java2025characterizingdeepresearch pages 3-5)
2. **Attribution quality**: citation precision, citation recall/thoroughness, “supported sentence” rates, DOI recall. (skarlinski2024languageagentsachieve pages 1-3, venkit2024searchenginesin pages 14-15, knollmeyer2024benchmarkingofretrieval pages 8-9)
3. **Coverage / completeness**: nugget coverage, claim coverage, reference coverage, document importance (notability). (patel2025deepscholarbenchalive pages 2-3)
4. **Human evaluation**: pairwise preference, expert grading; also time/effort proxies such as click/hover behavior under contradictions. (asai2024openscholarsynthesizingscientific pages 2-4, venkit2024searchenginesin pages 8-10)
5. **Agent process robustness**: hallucination incidence, tool-use quality, forgetting, stability under offline-vs-live web. (bosse2025deepresearchbench pages 1-2)

---

## 7) Notable gaps and cautions
- **Liu et al. (Apr 2023) verifiability audit**: the available excerpt here documents dataset construction and annotation workflow but does not include the numeric citation precision/recall results; use the full paper for the reported supported-sentence and citation-precision statistics. (liu2023evaluatingverifiabilityin pages 11-13)
- **Perplexity on LitQA2**: PaperQA2 reports running Perplexity Pro across all questions with identical scoring, but the numeric results are not present in the provided excerpt (tables are referenced but not included). (skarlinski2024languageagentsachieve pages 16-18)
- **DRB (Bosse et al.)**: confirms OpenAI Deep Research is included and describes evaluation design, but per-system numeric scores are not in the excerpted pages; the public leaderboard is the likely source for detailed numbers. (bosse2025deepresearchbench pages 1-2)

---

## Primary source bibliography (URLs + dates)
- L’ala et al. *PaperQA: Retrieval-Augmented Generative Agent for Scientific Research*. arXiv (2023-12). https://doi.org/10.48550/arxiv.2312.07559 (lala2023paperqaretrievalaugmentedgenerative pages 9-11)
- Skarlinski et al. *Language agents achieve superhuman synthesis of scientific knowledge*. arXiv (2024-09). https://doi.org/10.48550/arxiv.2409.13740 (skarlinski2024languageagentsachieve pages 3-5)
- Asai et al. *OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs*. arXiv (2024-11). https://doi.org/10.48550/arxiv.2411.14199 (asai2024openscholarsynthesizingscientific pages 2-4)
- Venkit et al. *Search Engines in an AI Era: The False Promise of Factual and Verifiable Source-Cited Responses*. arXiv (2024-10). https://doi.org/10.48550/arxiv.2410.22349 (venkit2024searchenginesin pages 14-15)
- Liu et al. *Evaluating Verifiability in Generative Search Engines*. arXiv (2023-04). https://doi.org/10.48550/arxiv.2304.09848 (liu2023evaluatingverifiabilityin pages 11-13)
- Knollmeyer et al. *Benchmarking of Retrieval Augmented Generation: A Systematic Literature Review on Evaluation Dimensions, Metrics and Datasets*. KEOD (2024-01). https://doi.org/10.5220/0013065700003838 (knollmeyer2024benchmarkingofretrieval pages 9-10)
- Bosse et al. *Deep Research Bench: Evaluating AI Web Research Agents*. arXiv (2025-05). https://doi.org/10.48550/arxiv.2506.06287 (bosse2025deepresearchbench pages 1-2)
- Du et al. *DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents*. arXiv (2025-06). https://doi.org/10.48550/arxiv.2506.11763 (du2506deepresearchbencha pages 1-3)
- Patel et al. *DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis*. arXiv (2025-08). https://doi.org/10.48550/arxiv.2508.20033 (patel2025deepscholarbenchalive pages 2-3)
- Gwon et al. *The Use of Generative AI for Scientific Literature Searches for Systematic Reviews: ChatGPT and Microsoft Bing AI Performance Evaluation*. JMIR Med Inform (2024-05). https://doi.org/10.2196/51187 (gwon2024theuseof pages 1-2)
- Meliante et al. *Evaluation of AI Tools Versus the PRISMA Method for Literature Search, Data Extraction, and Study Composition in Glaucoma Systematic Reviews*. JMIR AI (2025-09). https://doi.org/10.2196/68592 (meliante2025evaluationofai pages 1-2)


References

1. (skarlinski2024languageagentsachieve pages 3-5): Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, and Andrew D. White. Language agents achieve superhuman synthesis of scientific knowledge. ArXiv, Sep 2024. URL: https://doi.org/10.48550/arxiv.2409.13740, doi:10.48550/arxiv.2409.13740. This article has 132 citations.

2. (asai2024openscholarsynthesizingscientific pages 2-4): Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D'arcy, David Wadden, Matt Latzke, Minyang Tian, Pan Ji, Shengyan Liu, Hao Tong, Bohao Wu, Yanyu Xiong, Luke Zettlemoyer, Graham Neubig, Dan Weld, Doug Downey, Wen-tau Yih, Pang Wei Koh, and Hannaneh Hajishirzi. Openscholar: synthesizing scientific literature with retrieval-augmented lms. ArXiv, Nov 2024. URL: https://doi.org/10.48550/arxiv.2411.14199, doi:10.48550/arxiv.2411.14199. This article has 70 citations.

3. (venkit2024searchenginesin pages 14-15): Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Yixin Mao, and Chien-Sheng Wu. Search engines in an ai era: the false promise of factual and verifiable source-cited responses. ArXiv, Oct 2024. URL: https://doi.org/10.48550/arxiv.2410.22349, doi:10.48550/arxiv.2410.22349. This article has 19 citations.

4. (lala2023paperqaretrievalaugmentedgenerative pages 9-11): Jakub L'ala, Odhran O'Donoghue, Aleksandar Shtedritski, Sam Cox, Samuel G. Rodriques, and Andrew D. White. Paperqa: retrieval-augmented generative agent for scientific research. ArXiv, Dec 2023. URL: https://doi.org/10.48550/arxiv.2312.07559, doi:10.48550/arxiv.2312.07559. This article has 219 citations.

5. (skarlinski2024languageagentsachieve pages 1-3): Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, and Andrew D. White. Language agents achieve superhuman synthesis of scientific knowledge. ArXiv, Sep 2024. URL: https://doi.org/10.48550/arxiv.2409.13740, doi:10.48550/arxiv.2409.13740. This article has 132 citations.

6. (venkit2024searchenginesin pages 8-10): Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Yixin Mao, and Chien-Sheng Wu. Search engines in an ai era: the false promise of factual and verifiable source-cited responses. ArXiv, Oct 2024. URL: https://doi.org/10.48550/arxiv.2410.22349, doi:10.48550/arxiv.2410.22349. This article has 19 citations.

7. (knollmeyer2024benchmarkingofretrieval pages 8-9): Simon Knollmeyer, Oğuz Caymazer, Leonid Koval, Muhammad Uzair Akmal, Saara Asif, Selvine G. Mathias, and Daniel Grossmann. Benchmarking of retrieval augmented generation: a comprehensive systematic literature review on evaluation dimensions, evaluation metrics and datasets. Proceedings of the 16th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management, pages 137-148, Jan 2024. URL: https://doi.org/10.5220/0013065700003838, doi:10.5220/0013065700003838. This article has 10 citations.

8. (knollmeyer2024benchmarkingofretrieval pages 9-10): Simon Knollmeyer, Oğuz Caymazer, Leonid Koval, Muhammad Uzair Akmal, Saara Asif, Selvine G. Mathias, and Daniel Grossmann. Benchmarking of retrieval augmented generation: a comprehensive systematic literature review on evaluation dimensions, evaluation metrics and datasets. Proceedings of the 16th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management, pages 137-148, Jan 2024. URL: https://doi.org/10.5220/0013065700003838, doi:10.5220/0013065700003838. This article has 10 citations.

9. (bosse2025deepresearchbench pages 1-2): Nikos I. Bosse, Jon Evans, Robert G. Gambee, Daniel Hnyk, Peter Mühlbacher, Lawrence Phillips, Dan Schwarz, and Jack Wildman. Deep research bench: evaluating ai web research agents. ArXiv, May 2025. URL: https://doi.org/10.48550/arxiv.2506.06287, doi:10.48550/arxiv.2506.06287. This article has 12 citations.

10. (du2506deepresearchbencha pages 1-3): Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. Deepresearch bench: a comprehensive benchmark for deep research agents. ArXiv, Jun 2506. URL: https://doi.org/10.48550/arxiv.2506.11763, doi:10.48550/arxiv.2506.11763. This article has 81 citations.

11. (patel2025deepscholarbenchalive pages 2-3): Liana Patel, Negar Arabzadeh, Harshit Gupta, Ankita Sundar, Ion Stoica, Matei Zaharia, and Carlos Guestrin. Deepscholar-bench: a live benchmark and automated evaluation for generative research synthesis. ArXiv, Aug 2025. URL: https://doi.org/10.48550/arxiv.2508.20033, doi:10.48550/arxiv.2508.20033. This article has 12 citations.

12. (patel2025deepscholarbenchalive pages 1-2): Liana Patel, Negar Arabzadeh, Harshit Gupta, Ankita Sundar, Ion Stoica, Matei Zaharia, and Carlos Guestrin. Deepscholar-bench: a live benchmark and automated evaluation for generative research synthesis. ArXiv, Aug 2025. URL: https://doi.org/10.48550/arxiv.2508.20033, doi:10.48550/arxiv.2508.20033. This article has 12 citations.

13. (gwon2024theuseof pages 1-2): Yong Nam Gwon, Jae Heon Kim, Hyun Soo Chung, Eun Jee Jung, Joey Chun, Serin Lee, and Sung Ryul Shim. The use of generative ai for scientific literature searches for systematic reviews: chatgpt and microsoft bing ai performance evaluation. JMIR Medical Informatics, 12:e51187-e51187, May 2024. URL: https://doi.org/10.2196/51187, doi:10.2196/51187. This article has 66 citations and is from a peer-reviewed journal.

14. (meliante2025evaluationofai pages 1-2): Laura Antonia Meliante, Giulia Coco, Alessandro Rabiolo, Stefano De Cillà, and Gianluca Manni. Evaluation of ai tools versus the prisma method for literature search, data extraction, and study composition in glaucoma systematic reviews: content analysis. JMIR AI, 4:e68592-e68592, Sep 2025. URL: https://doi.org/10.2196/68592, doi:10.2196/68592. This article has 4 citations and is from a peer-reviewed journal.

15. (meliante2025evaluationofai pages 4-6): Laura Antonia Meliante, Giulia Coco, Alessandro Rabiolo, Stefano De Cillà, and Gianluca Manni. Evaluation of ai tools versus the prisma method for literature search, data extraction, and study composition in glaucoma systematic reviews: content analysis. JMIR AI, 4:e68592-e68592, Sep 2025. URL: https://doi.org/10.2196/68592, doi:10.2196/68592. This article has 4 citations and is from a peer-reviewed journal.

16. (skarlinski2024languageagentsachieve pages 16-18): Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, and Andrew D. White. Language agents achieve superhuman synthesis of scientific knowledge. ArXiv, Sep 2024. URL: https://doi.org/10.48550/arxiv.2409.13740, doi:10.48550/arxiv.2409.13740. This article has 132 citations.

17. (java2025characterizingdeepresearch pages 3-5): Abhinav Java, Ashmit Khandelwal, S. Midigeshi, Aaron Halfaker, Amit Deshpande, Navin Goyal, Ankur Gupta, Nagarajan Natarajan, and Amit Sharma. Characterizing deep research: a benchmark and formal definition. ArXiv, Aug 2025. URL: https://doi.org/10.48550/arxiv.2508.04183, doi:10.48550/arxiv.2508.04183. This article has 16 citations.

18. (liu2023evaluatingverifiabilityin pages 11-13): Nelson F. Liu, Tianyi Zhang, and Percy Liang. Evaluating verifiability in generative search engines. ArXiv, Apr 2023. URL: https://doi.org/10.48550/arxiv.2304.09848, doi:10.48550/arxiv.2304.09848. This article has 345 citations.

## Citations

1. lala2023paperqaretrievalaugmentedgenerative pages 9-11
2. asai2024openscholarsynthesizingscientific pages 2-4
3. bosse2025deepresearchbench pages 1-2
4. du2506deepresearchbencha pages 1-3
5. gwon2024theuseof pages 1-2
6. knollmeyer2024benchmarkingofretrieval pages 8-9
7. skarlinski2024languageagentsachieve pages 1-3
8. skarlinski2024languageagentsachieve pages 3-5
9. skarlinski2024languageagentsachieve pages 16-18
10. venkit2024searchenginesin pages 14-15
11. venkit2024searchenginesin pages 8-10
12. java2025characterizingdeepresearch pages 3-5
13. patel2025deepscholarbenchalive pages 2-3
14. liu2023evaluatingverifiabilityin pages 11-13
15. knollmeyer2024benchmarkingofretrieval pages 9-10
16. meliante2025evaluationofai pages 1-2
17. patel2025deepscholarbenchalive pages 1-2
18. meliante2025evaluationofai pages 4-6
19. https://doi.org/10.48550/arxiv.2312.07559
20. https://doi.org/10.48550/arxiv.2409.13740
21. https://doi.org/10.48550/arxiv.2411.14199
22. https://doi.org/10.48550/arxiv.2410.22349
23. https://doi.org/10.5220/0013065700003838
24. https://doi.org/10.48550/arxiv.2506.06287
25. https://doi.org/10.48550/arxiv.2506.11763
26. https://doi.org/10.48550/arxiv.2508.20033
27. https://doi.org/10.2196/51187
28. https://doi.org/10.2196/68592
29. https://doi.org/10.48550/arxiv.2304.09848
30. https://doi.org/10.48550/arxiv.2409.13740,
31. https://doi.org/10.48550/arxiv.2411.14199,
32. https://doi.org/10.48550/arxiv.2410.22349,
33. https://doi.org/10.48550/arxiv.2312.07559,
34. https://doi.org/10.5220/0013065700003838,
35. https://doi.org/10.48550/arxiv.2506.06287,
36. https://doi.org/10.48550/arxiv.2506.11763,
37. https://doi.org/10.48550/arxiv.2508.20033,
38. https://doi.org/10.2196/51187,
39. https://doi.org/10.2196/68592,
40. https://doi.org/10.48550/arxiv.2508.04183,
41. https://doi.org/10.48550/arxiv.2304.09848,