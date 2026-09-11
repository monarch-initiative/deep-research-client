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

## Responses are cached, and a replayed cell is not a measurement

The client caches responses in `~/.deep_research_cache`, keyed on the prompt,
provider, model and parameters. That makes a resumed or repeated run cheap, and
it is on by default — but it means a cell can be served from a response
recorded weeks ago, by a model version that has since changed. The cache
re-stamps the timings for the current run, so nothing downstream distinguishes
a replay from a live call.

Two consequences worth knowing before you publish a number:

- `--no-resume` re-runs a cell; it does not re-call the provider, and on a run
  directory that already has results `--no-cache` alone cannot reach those
  cells either — resume skips them before the client is consulted. To force
  real calls on an existing run directory, pass `--no-resume --no-cache`.
- Two arms sharing a provider, model and parameters — the way you ask what a
  provider's run-to-run spread looks like — share a cache key, so one may
  replay another instead of calling the provider. Whether it happens depends on
  scheduling: at `-j 1` the second arm replays the first, while at the default
  concurrency both usually reach the provider before either writes the cache.
  That means the number of independent samples behind a reported spread varies
  between identical invocations, which is why `eval run` warns about identically
  configured arms before it starts rather than reporting it afterwards.

At the end of a run `eval run` says how many cells were measured, how many were
replayed from the cache and how many were resumed from a previous run in the
same directory — three different things, only the first of which describes the
provider as it is now. `results.tsv` carries `resumed` and `cached` columns
marking which rows were which. Use `--no-cache` for a calibration run, and
`--cache-dir` to keep a benchmark's cache separate from your ad-hoc queries.

```
deep-research-client eval run questions.yaml --arm a=falcon --arm b=falcon --no-cache
```

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
  decliner               0.000   0.000       —    0/40
  echoing                0.325   1.000   0.325   13/40
  silent                 0.000   0.000       —    0/40

  Some responses had no recoverable answer. Those count against coverage but
  are a harness limitation, not a provider result; see the extraction_failures
  column in scores.tsv.

  These come from a provisional regex extractor, not an LLM judge. It has
  produced plausible-looking but wrong numbers before; treat them as a quick
  look, not as a result.
```

`always-a` gives a chance baseline — 0.325 on these questions, since the number
of options varies. `echoing` must score identically to `always-a`; a gap means
the extractor is being fooled by restated options. `decliner` and `silent` must
both show zero coverage, for different reasons: declining is not answering, and
neither is saying nothing.

Neither has a precision. It is over the answers attempted whose correctness
was actually established, and these two attempted nothing, so the column shows
an em dash rather than `0.000` — which in a comparison would read as
"answered and got them all wrong". `silent` is why this matters most: its
responses are ones the provisional extractor could not
read, which is a limitation of this harness rather than a result from the
provider, and the first of the two notes below the table says so. The second
is printed after every graded run and is the one that qualifies the whole
table: these numbers come from the provisional extractor, not a judge.

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

Citation existence resolves PMIDs against PubMed and DOIs against CrossRef.
Anything else the report cites — a PMC accession, a GEO series — is reported as
*not checked* rather than counted against the report, and both citation lines
say how many. The two counts are not the same count: verifiability's `N not
checked` is citations it did not resolve, while alignment's `N with nothing to
align against` also covers a citation that resolved to a record carrying no
title and one with no identifier at all. On an accession-only report they
agree by coincidence. That matters for a genomics benchmark, whose reference lists are
often accessions: `Citation Verifiability: not measured, 12 not checked` is a
report this client cannot judge, not a report that invented twelve references.
The rate is absent rather than zero whenever nothing was checkable, on this
line and on every other score line, so a measured zero always means a measured
zero. That holds across commands: `eval run`'s precision column is an em dash
whenever there is nothing to take a rate over — that is, whenever no
attempted answer had its correctness established. An arm gets there by
attempting nothing (every question declined, the endpoint down all run,
every response unreadable by the extractor, the pair skipped, or any mixture
of those), or by attempting and having every attempt come back with no
recorded correctness.

Do not read the `cov` beside the dash as saying which happened. Those causes
compose, so coverage there can be anything: `cov 0.000` when nothing was
attempted, `cov 1.000` when everything was and none of it was usable — not a
contradiction, the provider answered and the harness cannot say whether it was
right — and anything in between for a mixture. Five questions declined and
five answered with no recorded correctness, out of ten, prints `cov 0.500`
beside the dash.

Answers with no recorded correctness have a count of their own. `scores.tsv`
carries an `unusable` column, and a graded run with any such answers prints a
note under the table saying so — the same treatment `extraction_failures`
gets, and for the same reason: a harness record gap that moves a published
rate has to say so where the rate is printed. Only a hand-edited or
older-format run produces them.

The intrinsic scores need no LLM judge at all, so they are the cheap ones to
run first:

```bash
deep-research-client eval score report.md --source questions.yaml \
  --no-fact --no-recall --no-race
```

The three judge-backed scores need an API key, and the command refuses to start
without one rather than running everything and failing at the judge. Three ways
past it: set `OPENAI_API_KEY`, point `--llm-api-key-env` at a variable that is
set, or turn the judge-backed scores off with the flags above. A local
OpenAI-compatible endpoint needs no key at all — pass `--llm-base-url` and the
check is skipped:

Note that *any* `--llm-base-url` skips it, not only a local one. The command
cannot tell a keyless endpoint from one that checks keys, so it sends a
placeholder and prints a line saying so; point it at a proxy that does check
and every judge call will answer 401.

```bash
deep-research-client eval score report.md --source questions.yaml \
  --llm-base-url http://localhost:8000/v1 --llm-model my-local-model
```

## Rubrics

Report tasks are scored against a rubric — reference claims to recall, facts to
spot-check, topics to cover. The `dismech` and `ai-gene-review` adapters attach
bundled rubrics from `src/deep_research_client/evaluation/rubrics/`, and an eval
set can supply its own instead.

### Writing a rubric in your own eval set

Attach a `rubric` block to any report task:

```yaml
# questions.yaml
tasks:
  - id: brca1_function
    prompt: What does BRCA1 do, and where is it?
    answer_type: REPORT
    rubric:
      spot_checks:
        - name: chromosome
          pattern: 'chromosome\s+(17(?:[pq]\d+(?:\.\d+)?)?)\b'
          expected: 17q21.31
          match: prefix
        - name: ring_domain
          pattern: '\bRING\s*(?:finger\s*)?domain\b'
      expected_topics:
        - name: dna_repair
          keywords: [homologous recombination, double-strand break]
      reference_claims:
        - name: e3_ligase
          category: molecular_function
          description: BRCA1 is an E3 ubiquitin ligase in complex with BARD1.
```

A spot check with an `expected` value and a capturing group is an accuracy
check: the captured text has to match. One without an `expected` only asks
whether the pattern appears at all. Presence and accuracy are reported as
separate rates, because "the report never mentioned it" and "the report got it
wrong" are different failures.

`match` says how a captured value is compared:

| `match`  | Accepts                                                     |
| -------- | ----------------------------------------------------------- |
| `exact`  | Equality, ignoring case, surrounding whitespace and thousands separators. The default. |
| `prefix` | The above, plus a captured value that is a leading part of `expected`. |

Use `prefix` for hierarchical facts — a cytogenetic locus, an ontology
identifier, a version — where a shorter answer is less precise rather than
wrong. A report saying "chromosome 17" where the answer is 17q21.31 is the
commonest phrasing in the literature; `prefix` accepts it, while a report
saying 17p13.1 is still scored wrong, which a presence-only check could not
tell apart from silence.

A rubric is checked when the eval set loads, not when a report is scored, so a
typo is reported by `eval load` before any provider is paid. The full list of
refusals is below.

A check whose pattern matches but captures nothing (any group that can match
the empty string) counts as present and is left out of the accuracy rate. It
is not evidence either way, and the two match styles would otherwise disagree
about the same report.

Every occurrence of a pattern is considered, not just the first. A report on
BRCA1 that mentions TP53's locus before stating BRCA1's own would otherwise be
marked wrong for a fact it got right two sentences later. A check is correct if
any occurrence compares correctly — except that under `prefix`, an occurrence
that disagreed with a *strictly more specific* value wins instead. "Genes on
chromosome 17 include BRCA1" captures `17`, which is a valid prefix, and
without that rule it would excuse a report that went on to place BRCA1 at
17p13.1. Either way the detail reports the occurrence that settled it.

`eval load` refuses:

| Refused | Because |
| --- | --- |
| A pattern that is not a valid regular expression | It would raise mid-scoring and cost the report its other intrinsic scores too |
| A pattern that matches the empty string | It reports itself present in a report that says nothing |
| `match: prefix` with no capturing group, or no `expected` | The comparison it asks for cannot happen |
| A topic with no keywords | No report can ever cover it |
| A topic with a blank keyword | Every report covers it, including an empty one |
| A reference claim with a blank description | It asks the judge to look for nothing |
| A spot check with a blank name | It is how a failed check is identified in the results |
| An empty `rubric:` block | It scores exactly what no rubric scores |

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
