# Verify Claims

Surface the claims and pairs in a report that need review, and say who can settle
each: a domain finding that cites nothing, a citation that resolves to no
identifier, a point framed as settled consensus, a contradiction. This is the
fourth step of comparing sources ([issue #43](https://github.com/monarch-initiative/deep-research-client/issues/43)),
after [extraction](extract-claims.md), [grounding](ground-claims.md) and
[alignment](align-claims.md).

Verification has two halves. **Triage** is pure and always runs: it reads the claim
set and raises a flag wherever something needs a look, with no model and no
network. **Verifiers** are pluggable hooks that reach out to the world — the one
shipped here resolves a claim's cited identifiers to catch fabricated citations.
The point is not to decide the hard questions automatically. It is to route them:
most of what matters is a human's call, and the report says so.

## Quick start

```bash
# Triage only: no network, nothing to install
deep-research-client claims verify report.claims.json \
    --alignment report.self-align.json -o report.verified.json
```

```python
from deep_research_client.claims import ClaimSet, ClaimAlignment, verify_claims

claims = ClaimSet.model_validate_json(open("report.claims.json").read())
alignment = ClaimAlignment.model_validate_json(open("report.self-align.json").read())
report = verify_claims(claims, alignment=alignment)

for flag in report.needs_human:        # the flags only a person can settle
    print(flag.kind, flag.claim_ids, flag.message)
```

`verify_claims` is synchronous and runs its own event loop (use
`await averify_claims(...)` where one is already running, as in Jupyter). With no
verifiers it is triage-only and does no I/O.

## What triage flags

Each flag carries a **decidability** — who can settle the question it raises. Triage
*detects*; it does not adjudicate.

| Flag | Raised when | Who can settle it |
|------|-------------|-------------------|
| `UNSUPPORTED_CLAIM` | A domain finding (observation or attributed) cites nothing. Background-knowledge claims are expected to be uncited and are not flagged. | **agent** — retrieval could find support |
| `UNRESOLVED_CITATION` | A cited claim whose citations resolve to no identifier; the marker points nowhere checkable. | **agent** — ties to [#41](https://github.com/monarch-initiative/deep-research-client/issues/41) |
| `CONSENSUS_OVERSTATED` | A claim frames a point as settled (`well established`, `consensus`, `it is well known`, …) while either uncited or taking part in a contradiction. | **human** — a lexical heuristic, always a person's call |
| `CONTRADICTION` / `SELF_CONTRADICTION` | A contradictory alignment pair — across two sources, or within one. | **human** — which claim holds is a judgement |
| `CONFABULATED_CITATION` | *(citation check)* A cited identifier resolves to no real record. | **machine** — a resolver settled it |
| `WEAK_CITATION` | *(citation check)* A citation resolves, but the work looks off topic for the report. | **human** |
| `UNCHECKED_CITATION` | A resolvable citation a verifier ran over but did not confirm either way. | **agent** |

`--alignment` loads an alignment from [`claims align`](align-claims.md) so that
contradictions become flags. Without it, triage still runs the claim-level rules.

## Checking citations

`--check-citations` adds the one verifier shipped here. It resolves each cited
identifier against its source (reusing the reference validator) and returns a
verdict per identifier: a fabricated identifier (resolves to nothing) becomes a
machine-decidable `CONFABULATED_CITATION`; a real but off-topic work becomes a
`WEAK_CITATION` for a human; a resolved, on-topic work is `SUPPORTED`. Relevance is
weighed against **the report's own vocabulary** (all of the set's claim text, which
the CLI assembles), so `WEAK_CITATION` means the work looks off topic *for the
report*, not that it fails to support this one claim. A single claim is too little
text for the keyword heuristic, which is why the whole report is used.

```bash
deep-research-client claims verify report.claims.json --check-citations
```

It needs the `validation` extra for the resolver:

```bash
pip install "deep_research_client[validation]"
```

`--email` passes a contact address to the NCBI API (it asks for one to raise the
rate limit); `--cache` reuses resolved references between runs; `--max-claims`
bounds how many claims are resolved — triage still sees every claim, and any
resolvable citation past the cap is reported `UNCHECKED_CITATION`. Citation checks
run **one at a time** (the resolver is not assumed thread-safe), so `--concurrency`
does not speed a run that uses only this verifier.

`SUPPORTED` means the cited work resolves and is on topic, **not** that it entails
the claim. A claim citing several works gets one verdict each; a report's
`verifications_for(claim_id)` returns them all, so a later `UNSUPPORTED` is not
hidden behind an earlier `SUPPORTED`. A hard failure (a resolver error the engine
does not turn into `UNVERIFIABLE`) aborts the run rather than returning partial
results.

## Normalising citations (issue #41)

Grounding and alignment both get noisier when the same paper is cited a dozen ways.
`normalize_publication_id` is the deterministic core of
[#41](https://github.com/monarch-initiative/deep-research-client/issues/41): it
turns one raw citation into a canonical id with its kind and a confidence.

```python
from deep_research_client.claims import normalize_publication_id

normalize_publication_id("https://pubmed.ncbi.nlm.nih.gov/7913883").canonical_id
# 'PMID:7913883'  -- the same id a written "PMID:7913883" gives
normalize_publication_id("Smith et al. 2020").resolved
# False  -- kept, not dropped, with confidence 0.0 for a later resolver
```

The canonical form matches `CitationHandle.reference_id`, so the two are
interchangeable (which is why a clinical-trial id reads `NCT:NCT00000419` — the
`PREFIX:value` shape the rest of the substrate uses, prefix and value alike).
Online resolution of a paper's title, authors and year from an identifier — the
rest of #41 — is a hook left for a following PR.

## Limitations

- **Triage routes; it does not adjudicate.** A flag means "a human (or an agent, or
  a machine) should look", not "this claim is wrong". The decidability says who.
- **`UNSUPPORTED_CLAIM` is about the text, not the world.** It flags a claim that
  *cites* nothing, which a true claim can still be. It is a prompt to check, found
  without reading any cited work.
- **`CONSENSUS_OVERSTATED` is a lexical heuristic.** It keys on a small, fixed list
  of phrases (overridable), so it both misses paraphrased overclaiming and can fire
  on a phrase used carefully. Always a human's call, never automated.
- **Citation checking confirms existence and topicality, not support.**
  `SUPPORTED` here means the cited work resolves and is on topic — not that it
  entails the claim. A semantic claim-vs-work judge (a model reading both) is the
  agent hook deliberately deferred.
- **Relevance needs enough report text to judge.** The keyword heuristic keys on
  terms the report repeats, so for a very small set (a claim or two) it finds none,
  relevance is left unassessed, and a resolved citation is `SUPPORTED` on existence
  alone — `WEAK_CITATION` simply will not fire.
- **The report describes one run.** Running `claims verify` again overwrites the
  report; it is a snapshot of the flags and verdicts at that moment.

## Then review

The report is built to be read by a person, most-urgent-first: start with
`report.needs_human`, the flags no rule or tool can close. A downstream agent loop
can take `report.needs_agent` (unsupported claims, unresolved citations) and try to
resolve them, escalating what it cannot. The machine-decidable flags
(`CONFABULATED_CITATION`) are already settled — those citations are fabricated.

The report also feeds the report-level rubric: pass it to
[`summarize_alignment(..., verification=report)`](align-claims.md#folding-in-grounding-and-verification)
and its citation verdicts and review flags fold into `citation_accuracy` and
`review_burden` beside the alignment metrics.
