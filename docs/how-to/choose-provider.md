# How to Choose a Provider

This guide helps you select the right provider for your research needs.

## Quick Decision Guide

| Need | Best Provider | Command |
|------|---------------|---------|
| Quick summary | Perplexity (sonar) | `--provider perplexity --model sonar` |
| General research | Perplexity (sonar-pro) | `--provider perplexity --model pro` |
| Comprehensive report | OpenAI (o3) | `--provider openai` |
| Scientific literature | Edison | `--provider edison` |
| Academic papers only | Consensus | `--provider consensus` |

## Provider Comparison

### Perplexity AI

Best for: **General web research with real-time data**

```bash
# Lighter research task
deep-research-client research "Summarize recent developments in room-temperature superconductor claims" \
  --provider perplexity --model sonar

# Comprehensive web research
deep-research-client research "Analyze the current landscape of AI chip startups and their differentiation strategies" \
  --provider perplexity --model sonar-deep-research
```

Strengths:

- Real-time web search
- Multiple speed/quality options
- Good citation quality
- Reasonable pricing

### OpenAI Deep Research

Best for: **Comprehensive, in-depth research reports**

```bash
deep-research-client research "Comprehensive analysis of CRISPR base editing versus prime editing: mechanisms, applications, limitations, and recent clinical developments" \
  --provider openai
```

Strengths:

- Most thorough analysis
- Excellent synthesis
- Code interpretation capability

Trade-offs:

- Slowest (can take several minutes)
- Most expensive

### Edison Scientific (Formerly Falcon)

Best for: **Scientific literature review**

```bash
deep-research-client research "Review molecular mechanisms of autophagy regulation and its role in cancer progression" \
  --provider edison
```

Strengths:

- Powered by PaperQA3
- Focus on peer-reviewed papers
- Good for biomedical research

### Consensus

Best for: **Academic paper search**

```bash
deep-research-client research "Synthesize clinical evidence on intermittent fasting effects on metabolic health markers" \
  --provider consensus
```

Strengths:

- Only peer-reviewed sources
- Evidence-based summaries
- Good for scientific claims

## Budget Considerations

From cheapest to most expensive:

1. **Perplexity sonar** - Best value for quick queries
2. **Consensus** - Good value for academic research
3. **Perplexity sonar-pro** - Mid-range, good quality
4. **Edison** - Mid-range, scientific focus
5. **Perplexity sonar-deep-research** - Higher cost, comprehensive
6. **OpenAI o4-mini** - Higher cost, good quality
7. **OpenAI o3** - Highest cost, most comprehensive

Example for budget-conscious usage:

```bash
# Use cheapest for initial exploration
deep-research-client research "Overview of CAR-T cell therapy landscape" \
  --provider perplexity --model sonar

# Then use comprehensive for final deep dive
deep-research-client research "Detailed analysis of CAR-T manufacturing challenges and novel approaches to reduce vein-to-vein time" \
  --provider openai
```

## Speed Considerations

From fastest to slowest:

1. **Perplexity sonar** - Seconds
2. **Consensus** - Seconds
3. **Perplexity sonar-pro** - ~30 seconds
4. **Perplexity sonar-deep-research** - 1-3 minutes
5. **Edison** - 2-5 minutes
6. **OpenAI o4-mini** - 2-5 minutes
7. **OpenAI o3** - 5-15 minutes

## Use Multiple Providers

For thorough research, combine providers:

```bash
# Quick landscape overview
deep-research-client research "Overview of alpha-synuclein aggregation in Parkinson's disease" \
  --provider perplexity --model sonar \
  --output synuclein-overview.md

# Scientific literature deep-dive
deep-research-client research "Molecular mechanisms of alpha-synuclein fibril formation and propagation" \
  --provider edison \
  --output synuclein-mechanisms.md

# Clinical evidence synthesis
deep-research-client research "Clinical trial evidence for alpha-synuclein targeting therapies" \
  --provider consensus \
  --output synuclein-trials.md
```

## Fall Back to Another Provider

A run can fail for reasons no retry fixes: an account out of credits, a spent
plan allowance, a rejected or missing key. `--fallback` lets a different
provider take the work instead.

```bash
deep-research-client research "Statins and myopathy" \
  --provider falcon --fallback --output report.md
```

It is **off by default, and deliberately so.** A report records the provider
that produced it, and downstream curation reads that field: silently swapping
providers would make those records wrong. So you have to ask for it.

`--fallback` tries the other configured providers in **registration order**,
which is a fixed sequence filtered down to whichever you have configured — not
a preference ranking, and not dependent on how your environment is set up:

    openai, falcon, asta, perplexity, consensus, openscientist, cyberian, claude_code

Each candidate is a paid account, so when it matters which one you spend money
on, name them yourself rather than relying on that order:

```bash
deep-research-client research "CRISPR delivery mechanisms" \
  --provider falcon \
  --fallback-provider openai \
  --fallback-provider perplexity \
  --output report.md
```

Naming providers *replaces* the automatic ordering rather than adding to it, so
a configured provider you leave out is never tried. For a single fallback the
bare name works too — `fallback="openai"` from Python is the same as
`fallback=["openai"]`.

### What gets recorded

A report produced by a fallback says so in its frontmatter, and says why:

```yaml
provider: openai
fell_back: true
requested_provider: falcon
provider_attempts:
- provider: falcon
  succeeded: false
  error_type: ProviderBillingError
  status_code: 402
  remedy: the account is out of credits
  retryable: false
- provider: openai
  succeeded: true
```

A report that came from the provider you asked for gains none of these fields:
their presence *is* the finding.

`remedy` is our reading of the failure, not the provider's own error text.
That text can quote whatever the provider chose to put in a response body —
including, for a rejected key, the key — and these reports get committed, so
the raw version stays on the result object and in the logs rather than on disk.

Every key in `provider_attempts` is one the client produces, which is what
makes the rule checkable by reading the list. A quota error's reset time is
withheld for the same reason even though it is harmless in itself: it is parsed
out of provider text, so keeping it would make the rule "ours, except one
field". The console still prints the provider's own words — an operator
diagnosing a failed provider needs them, and they are reading their own
terminal rather than a file someone else committed.

Two other places behave differently, and it is worth being precise about how:

- **Cache entries** do carry `requested_provider` and `provider_attempts`, as
  `null` and `[]`. Nothing from a run is stored in them — the entry is written
  before the provenance is stamped, so a cached answer can never be replayed as
  a fallback that never happened — but the keys themselves are present.
- **The result object** always populates both, so a library caller serialising
  a result sees them on every run, recording the single provider that answered.
  `fell_back` is a property rather than a stored field, so it is *not* in
  `model_dump()`; read it off the object, or derive it from
  `provider_attempts`.

### Which failures are followed

| Failure | Falls back? | Why |
|---------|-------------|-----|
| No credits (402), spent quota, rejected key (401/403), not configured | Yes | This provider cannot do the work; another might |
| Rate limited (429) | No | The same provider will take it shortly -- wait and retry |
| Server error (5xx) | No | A temporary fault, not a reason to change who answers |
| Anything unrecognised | No | An unexplained failure is not evidence another provider would do better |

The last row is the conservative default: a fallback changes who produced your
report, so it is taken only where the failure type is evidence it should be.

### Two things to know

**`--model` and `--param` apply to the first provider only.** They were chosen
for the provider you named -- a Perplexity model name means nothing to OpenAI,
and an unknown parameter is a hard error. A fallback provider therefore runs on
its own defaults, and the run says so on stderr when it happens.

**Nothing is written until a provider succeeds.** If every candidate fails, the
run raises with the last failure and no report, no cache entry, and no partial
file is left behind.

### Trying it without an outage

The mock provider can simulate any of these failures, so you can see the
behaviour before you rely on it. With a second provider configured, this
produces a real report whose frontmatter has the shape shown above, with
`requested_provider: mock` and whichever provider answered:

```bash
ENABLE_MOCK_PROVIDER=true deep-research-client research "test" \
  --provider mock --param error_type=billing --fallback \
  --output report.md
```

If you have no other provider configured, the run has nowhere to go and stops
with the mock's simulated billing failure -- which is the fail-closed behaviour,
not a bug. To watch the switch happen without configuring anything, name
`deeper_med` as the fallback: it is a permanently unavailable stub, so the run
still ends in an error and writes no report, but the log shows the fallback
being taken and each provider explaining itself:

```bash
ENABLE_MOCK_PROVIDER=true deep-research-client research "test" \
  --provider mock --param error_type=billing \
  --fallback-provider deeper_med
```

Swap `error_type=billing` for `error_type=quota` in the **first** command
above — the one that writes `report.md` — to watch the withholding happen. A
quota failure is the only kind whose remedy embeds a provider-supplied reset
time, so the console prints `renews at 3pm, pool quota_pool_7f21` while the
saved report keeps only `the plan's usage limit is spent`. Running it against
the `deeper_med` command just above shows you the console half and no report,
since that one is fail-closed by design.

Note that `mock` is reached by `--fallback-provider mock` but never by
`--fallback` on its own: a provider that invents its reports is excluded from
the automatic ordering, so a real run that runs out of credits fails rather
than quietly handing you a fabricated report.

## Check Available Models

List all available models and their characteristics:

```bash
# All models
deep-research-client models

# Filter by provider
deep-research-client models --provider perplexity

# Show detailed info
deep-research-client models --detailed

# Filter by cost
deep-research-client models --cost low
```

## See Also

- [Provider Reference](../reference/providers.md) - Full provider documentation
- [Model Reference](../reference/models.md) - Complete model specifications
