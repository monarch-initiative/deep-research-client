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

There is a third shape, and it is worth knowing about before you write an eval
set: a task with an ideal answer and **no** distractors infers `SHORT_ANSWER`,
and nothing in this client scores that yet — not `eval run`, not `eval score`.
Such tasks run and their responses are saved like any other; they simply cannot
be graded. `eval load` and `eval run` both say so up front — before any provider
is called, and in `--dry-run` too — rather than leaving it to be discovered once
the run is paid for. Add distractors to make it multiple choice, or drop the
ideal answer to make it a report task.

A multiple-choice task whose options cannot pose an answerable question is
refused when the eval set loads — by `eval load` and `eval run` alike, before
any provider is called. The commonest case is `answer_type: MULTIPLE_CHOICE`
with no distractors: one option and a right answer is not a question, and every
arm would score 1.000 on it. Duplicated options are refused for the mirror-image
reason, since two lettered options reading identically mark a correct answer
wrong half the time.

The full list of refused shapes lives in the docstring of `degenerate_reason`
in `deep_research_client.evaluation.mcq`, which is the canonical statement and
carries a worked example of each. It is deliberately not re-enumerated here —
this list has drifted behind that one before.

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

### Grading is deliberately not part of a run

A run materialises results and stops there. Every response is on disk beside the
prompt that produced it, so how they get scored is a decision you can make — and
change — later, without paying any provider a second time.

`--grade` will additionally score multiple-choice answers, but read the next
section before trusting what it prints.

### The multiple-choice grader is provisional

It reads a provider's answer out of its prose with regular expressions. That is
a stopgap, not the design. Three defects surfaced within a day of first use, and
what makes them worth recording is that none of them looked like a failure:

| What happened | What the table showed |
|---|---|
| A restated option list read as choosing the last option | Every arm abstaining on nearly everything |
| A bare quantity (`6%`) appearing anywhere read as choosing that option | An answer the provider never gave |
| `**Answer: D**` read as no answer at all | Every correct answer discarded, for any provider that bolds its verdict |

Each was found by running the thing rather than reading it, and each produced a
plausible table. The next defect of this kind is equally likely to look like a
score rather than a bug — which is the argument against the approach, not a list
of things now fixed.

**The intended replacement is an LLM judge**, which is what the report scorers
already use. Deciding which option a report settled on is reading comprehension,
and a model asked for a structured answer can both do it more reliably and say
when it is unsure. Until that lands, `--grade` is fine for a quick look and
should not be the basis of a published number.

Presentation and aggregation are unaffected by this and will survive the change:
option order is deterministic per task, and accuracy/coverage/precision are
arithmetic over dispositions regardless of how the dispositions were obtained.

### Checking the plumbing for free

The mock provider can answer multiple-choice questions by position, which makes
the run machinery verifiable at zero cost — it has no idea which option is
correct, so the score each policy deserves is computable in advance:

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
  --adapter lab-bench --arms mock-arms.yaml --limit 40 --grade
```

```
  arm                      acc     cov    prec       n
  always-a               0.325   1.000   0.325   13/40
  decliner               0.000   0.000   0.000    0/40
  echoing                0.325   1.000   0.325   13/40
  silent                 0.000   0.000   0.000    0/40
```

`always-a` gives a chance baseline — 0.325 on these questions, since the number
of options varies. `echoing` must score identically to `always-a`; a gap means
the extractor is being fooled by restated options. `decliner` and `silent` must
both show zero coverage, for different reasons: declining is not answering, and
neither is saying nothing.

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

Scoring is the open piece, by choice. The runner materialises results; how to
grade them is deferred.

When it is picked up, the direction is LLM-as-judge throughout, replacing the
provisional regex extractor described above. Report scoring already works that
way (`score_fact`, `score_claim_recall`, `score_race`), so the pattern to follow
is in the codebase. Batch scoring over a whole run directory — rather than one
report at a time — is the other missing piece.
