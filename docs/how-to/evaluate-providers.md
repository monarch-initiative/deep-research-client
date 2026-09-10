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

## Run the matrix

`eval run` sends every task to every arm. An *arm* is one configuration under
test — a provider, optionally a model, optionally provider parameters. It is
called an arm rather than a provider because the same provider usually appears
more than once in a useful comparison:

```yaml
# arms.yaml
arms:
  - id: edison
    provider: falcon

  - id: agent-web
    provider: claude_code
    description: Plain agent with web search, as a control

  - id: agent-noweb
    provider: claude_code
    description: Closed-book control, to probe contamination
    params:
      allowed_tools: []
```

Price the run before committing to it — `--dry-run` shows the grid and the exact
prompt one cell would receive, without calling anything:

```bash
deep-research-client eval run LitQA2 --adapter lab-bench \
  --arms arms.yaml --limit 20 --dry-run
```

Then run it:

```bash
deep-research-client eval run LitQA2 --adapter lab-bench \
  --arms arms.yaml --limit 20 --concurrency 4
```

Simple arms need no file: `--arm falcon --arm openai:o3-deep-research --arm
baseline=claude_code`.

### What a run writes

```
runs/2026-09-10T14-22Z/
  manifest.json        the run: arms, dataset revision, every cell
  results.tsv          one row per cell
  scores.tsv           per-arm aggregates, for multiple-choice runs
  <task_id>/
    <arm_id>/
      prompt.md        exactly what the provider was sent
      output.md        exactly what it returned
      cell.json        the cell record
      answer.json      the graded answer, for multiple-choice tasks
```

`prompt.md` matters more than it looks: for a multiple-choice task the prompt
carries the lettered options in the order that provider actually saw, and
without it a score cannot be audited.

Cells are written as they finish, so a run can be inspected while it is going
and resumed if it is interrupted — point `--output-dir` at the same directory
and completed cells are skipped. Failed cells are always retried, so a transient
error never becomes permanent.

A failing arm does not take the run down with it. If one provider is out of
quota you lose that arm's cells and keep everything else.

### Test the harness before paying for it

The mock provider can answer multiple-choice questions by position, which makes
the whole pipeline verifiable at zero cost. It has no idea which option is
correct, and that is exactly why this works: because option order is
deterministic for a given task, the score each policy deserves can be worked out
in advance and checked against what the harness reports.

```yaml
# mock-arms.yaml
arms:
  - id: always-a
    provider: mock
    params: {answer_policy: first}    # picks option A every time
  - id: echoing
    provider: mock
    params: {answer_policy: echo}     # restates every option, then answers A
  - id: decliner
    provider: mock
    params: {answer_policy: last}     # picks the last option: the abstention
  - id: silent
    provider: mock
    params: {answer_policy: none}     # never states an answer
```

```bash
ENABLE_MOCK_PROVIDER=true deep-research-client eval run LitQA2 \
  --adapter lab-bench --arms mock-arms.yaml --limit 40
```

```
  arm                      acc     cov    prec       n
  always-a               0.325   1.000   0.325   13/40
  decliner               0.000   0.000   0.000    0/40
  echoing                0.325   1.000   0.325   13/40
  silent                 0.000   0.000   0.000    0/40
```

Four things worth reading off that table:

- **`always-a` is your chance baseline.** On these 40 LitQA2 questions, picking
  by position scores 0.325 — the options-per-question vary, so this is not
  1-in-4. Any real provider has to beat this to have shown anything at all.
- **`echoing` scores identically to `always-a`.** It must: they choose the same
  option, one of them just quotes the question first. A gap between those two
  rows means the extractor is being fooled by restated options.
- **`decliner` has zero coverage, not zero accuracy alone.** Abstentions are not
  wrong answers.
- **`silent` has zero coverage too.** A provider that never answers must record
  extraction failures, never a fabricated choice.

Run this against a new benchmark before spending anything on it. If these four
rows do not come out as above, the harness is misreading that benchmark's
answers, and every real number you then collect would be wrong in the same way.

### Reading the scores

Multiple-choice tasks are graded during the run, because grading them costs
nothing. Report tasks are saved but not scored — that needs an LLM judge, so it
is a separate deliberate pass with `eval score`.

```
  arm                      acc     cov    prec       n
  edison                 0.550   0.900   0.611   11/20
  agent-web              0.400   1.000   0.400    8/20
  agent-noweb            0.150   0.950   0.158    3/20
```

Never read accuracy without coverage beside it. A low accuracy because the
provider answered wrongly and a low accuracy because it declined to answer are
different results, and only coverage separates them.

Watch `extraction_failures` in `scores.tsv` too. That column counts responses
no answer could be recovered from — a limitation of this harness, not a result
about the provider. It is reported separately precisely so it can be driven down
rather than quietly deflating someone's score.

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

Report-shaped tasks are still scored one at a time: `eval run` saves the reports
but `eval score` takes a single file. A batch scoring pass over a whole run
directory is the obvious next step.
