# Evaluate providers against a benchmark

The `eval` commands run research tools against a set of questions and score what
comes back. A benchmark is data here, not code: you point an *adapter* at a
source, and it produces an eval set of tasks that the scorers understand.

## Bring your own questions

The plainest eval set is a list of questions in YAML:

```yaml
# questions.yaml
name: coscientist-v1
description: Mechanism questions for the co-scientist comparison
tasks:
  - id: fgfr3_mech
    prompt: What are the pathophysiological mechanisms of achondroplasia?
    tags: [biomedical, mechanism]

  - id: fgfr3_residue
    prompt: Which FGFR3 residue is most commonly mutated in achondroplasia?
    ideal: G380R
    distractors: [G375C, R248C, K650E]
    abstention_option: Insufficient information to answer this question.
```

Check that it parses before spending money on providers:

```bash
deep-research-client eval load questions.yaml
```

```
Eval set: coscientist-v1
  Tasks:    2
  Shapes:   MULTIPLE_CHOICE=1, REPORT=1
```

The two tasks have different shapes, and that determines how each is scored. A
task with distractors is multiple choice, scored by matching the option chosen.
A task with neither an ideal answer nor distractors is a report task, scored
against a rubric. You do not have to say which — it is inferred — but you can
state `answer_type` explicitly to override.

TSV works too, for questions that came out of a spreadsheet:

```
id	question	ideal	distractors	tags
q_mcq	Which base pairs with adenine?	Thymine	Guanine|Cytosine	biology,basics
q_open	What are the mechanisms of achondroplasia?			biomedical
```

```bash
deep-research-client eval load questions.tsv --adapter tsv
```

Columns the model does not name are kept as task metadata rather than dropped,
so your bookkeeping columns survive into the results.

## Use a published benchmark

```bash
deep-research-client eval adapters
```

```
yaml             Eval set written as YAML, with optional distractors and provenance
tsv              Eval set written as TSV/CSV with a header row
lab-bench        LAB-Bench biology multiple-choice benchmark (FutureHouse) (downloads data)
dismech          Curated disease mechanisms (Monarch dismech)
ai-gene-review   Curated gene function annotations (Monarch ai-gene-review)
```

### LAB-Bench

[LAB-Bench](https://arxiv.org/abs/2407.10362) (Laurent et al. 2024) is a
multiple-choice benchmark of biology research tasks. Download a subset first:

```bash
deep-research-client eval fetch LitQA2
```

```
LitQA2: 199 rows at revision 5c77cec64843
```

Then load it like any other eval set:

```bash
deep-research-client eval load LitQA2 --adapter lab-bench
```

The data is downloaded rather than shipped with this package, for three reasons:
LAB-Bench carries a canary string so that contamination can be detected if its
questions turn up in a training corpus, and committing it to a public repository
would feed it to scrapers; it is CC-BY-SA-4.0 where this project is
BSD-3-Clause; and it is large. The revision that was downloaded is recorded in
the eval set, which is what makes a score reproducible — more so than a vendored
copy, which cannot tell you which version produced a given number.

Two constraints are worth knowing before you read any LAB-Bench number:

- **Scores are not leaderboard-comparable.** The publisher withholds roughly 20%
  of the benchmark privately for contamination monitoring, so only the public
  portion is scored here. Loading the set prints this, and the eval set carries
  it as `is_partial`.
- **FigQA and TableQA are refused.** Their questions are about figures and tables
  supplied as images, which a text-only research client cannot present. Scoring
  them would measure the harness rather than the provider.

Text-only subsets: `LitQA2`, `SuppQA`, `DbQA`, `ProtocolQA`, `SeqQA`,
`CloningScenarios`. Pass `all` to fetch every one of them.

### Monarch knowledge bases

`dismech` and `ai-gene-review` produce report tasks from curated YAML, with the
curated claims attached as reference claims to score recall against:

```bash
deep-research-client eval load /path/to/dismech/kb/disorders --adapter dismech
deep-research-client eval load /path/to/ai-gene-review/genes/human --adapter ai-gene-review
```

Both accept a single YAML file or a directory.

## Score a saved report

For report-shaped tasks, score a markdown file you already have:

```bash
deep-research-client eval score report.md \
  --source questions.yaml \
  --task-id fgfr3_mech \
  --provider falcon
```

Four groups of scores are available, and they differ sharply in what they cost:

| Score | What it measures | Needs |
|---|---|---|
| Claim recall | Fraction of reference claims the report covers | LLM judge |
| FACT | Whether each citation supports the claim it is attached to | LLM judge + PubMed |
| RACE | Report quality across four dimensions | LLM judge |
| Intrinsic | Citation existence, title/claim alignment, spot checks, topic coverage | PubMed only |

The intrinsic scores need no LLM judge at all, so they are the cheap ones to
run first:

```bash
deep-research-client eval score report.md --source questions.yaml \
  --no-fact --no-recall --no-race
```

## Rubrics

Report tasks are scored against a rubric — reference claims to recall, facts to
spot-check, topics to cover. The `dismech` and `ai-gene-review` adapters attach
bundled rubrics from `src/deep_research_client/evaluation/rubrics/`, and an eval
set can supply its own instead.

A spot check with an `expected` value and a capturing group is an accuracy
check: the captured text has to match. One without an `expected` only asks
whether the pattern appears at all. Presence and accuracy are reported as
separate rates, because "the report never mentioned it" and "the report got it
wrong" are different failures.

## Adding a benchmark

Write one adapter class. Subclass `EvalSetAdapter`, implement `load()` to return
an `EvalSet`, and register it in `ADAPTER_CLASSES`. Nothing in the runner or the
scorers needs to change, because scorers dispatch on each task's `answer_type`
rather than on its subject matter.

The eval-set model itself is the LinkML schema at
`src/deep_research_client/evaluation/evaluation.yaml`; regenerate its Pydantic
classes with `just gen-datamodel-eval` after any change.

## What is not here yet

There is no command that runs a whole matrix of questions across several
providers and writes the results into a predictable directory tree. Today you
loop over `deep-research-client research` yourself and score the outputs
afterwards.

The groundwork is in place: benchmarks are data, tasks declare their answer
shape, and eval sets carry the provenance a published score needs. What a matrix
runner still needs is its own vocabulary in the schema — the provider/model/param
combinations to sweep, and a per-cell record of what each produced — plus the
executor and output layout on top.
