---
name: run-deep-research
description: Run cited research and literature reviews with deep-research-client (DRC). Use for research reports, scientific literature reviews, provider/model selection, and template-based research with saved citations and provenance.
---

# Run deep research

Use DRC to produce a research artifact, then assess its evidence. A generated
report is a starting point for synthesis, not proof that its claims are supported.

## Choose the runtime and research scope

The skill supplies instructions; it does not install DRC or configure credentials.
Examples use `uvx deep-research-client`. In a DRC checkout, use
`uv run deep-research-client` instead. Check command help before using options
that may not exist in an older installed release.

1. Establish the question, organism or population, date range, and desired output
   from the request. Carry forward any chosen provider, budget, and depth.
2. Inspect configured providers and model cards without launching research:

   ```bash
   uvx deep-research-client providers
   uvx deep-research-client models --detailed
   uvx deep-research-client research --help
   ```

3. Choose a configured provider and model suitable for the requested depth.
   Model cards describe capabilities and cost categories, not a guarantee of
   live availability or a current price quote. Use `providers --show-params`
   for accepted provider parameters. `providers --check` makes live probe calls.
4. If the request leaves a consequential cost/depth decision unresolved, clarify
   it before starting. An explicit deep-research request with an established
   provider/budget does not need another confirmation. Keep credentials in the
   environment or the provider's supported configuration; never print keys.

## Run and preserve the result

Replace `PROVIDER` and `MODEL` with the selected values. Prefer an input file for
long prompts and an explicit output path for the report:

```bash
uvx deep-research-client research --input-file question.md \
  --provider PROVIDER --model MODEL \
  --output report.md --separate-citations citations.md
```

For reusable questions, use a template with `{variable}` placeholders or a Jinja2
template supported by DRC:

```bash
uvx deep-research-client research --template gene_research.md \
  --var gene=TP53 --var organism=human \
  --provider PROVIDER --model MODEL --output tp53-report.md
```

Keep caching enabled for retries and repeated questions. The default cache is
`~/.deep_research_cache`; use `--cache-dir PATH` for project-specific storage.
Inspect cached work with `list-cache` before repeating a costly run. Use
`--no-cache` only when freshness or an explicit rerun requires it; do not clear
the entire cache to retry one question.

Provider fallback is opt-in. Add `--fallback-provider NAME` (repeatable) only when
the alternatives fit the user's authorization and research needs. Record the
provider and model that actually produced the result. A provider failure or
quota error is not an empty literature result.

## Check evidence and deliver

Read the saved report, metadata, and citations before synthesizing an answer.
Verify identifiers resolve, quoted passages occur in the retrieved source, and
the sources support the specific claims, organism, and context. Separate
unresolved or inaccessible sources from contradicted claims.

DRC offers optional reference and ontology-term checks:

```bash
uvx --from 'deep-research-client[validation,terms]' deep-research-client \
  research --input-file question.md --provider PROVIDER --model MODEL \
  --output report.md --validate-references --validate-terms --fail-on-unresolved
```

These checks fetch references and ontology data and may add substantial runtime.
Use them on the initial run when wanted; for an existing report inspect
`validate-references --help` and `validate-terms --help` to validate without
repeating research. Identifier resolution and quotation matching do not
establish scientific truth. Report unchecked citations or terms explicitly.

Return the report path, citation path if separate, actual provider/model, cache
status, and a short synthesis with evidence gaps. Preserve metadata and source
links when editing the final report.

See the [DRC documentation](https://monarch-initiative.github.io/deep-research-client/)
for provider setup and advanced workflows.
