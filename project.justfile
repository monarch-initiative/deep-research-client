## Add your own just recipes here. This is imported by the main justfile.

# Regenerate every Pydantic datamodel from its LinkML schema (source of truth).
[group('model development')]
gen-datamodel: gen-datamodel-validation gen-term-datamodel gen-datamodel-vocabulary gen-datamodel-eval gen-datamodel-claims gen-datamodel-alignment gen-datamodel-verification

# Each recipe below generates to a temporary file first: a redirect straight onto
# the target would truncate it before gen-pydantic runs, so a schema typo would
# leave an empty datamodel and break `import deep_research_client` until it was
# restored. The `rm` on failure keeps a stray .tmp out of the tree, and the
# trailing `false` keeps the exit status non-zero so an error is not reported as
# success.

# Regenerate the reference-validation datamodel from reference_validation.yaml.
[group('model development')]
gen-datamodel-validation:
  uv run gen-pydantic src/deep_research_client/validation/reference_validation.yaml \
    > src/deep_research_client/validation/datamodel.py.tmp \
    && mv src/deep_research_client/validation/datamodel.py.tmp \
      src/deep_research_client/validation/datamodel.py \
    || { rm -f src/deep_research_client/validation/datamodel.py.tmp; false; }

# Regenerate the term-validation datamodel from term_validation.yaml.
[group('model development')]
gen-term-datamodel:
  uv run gen-pydantic src/deep_research_client/validation/term_validation.yaml \
    > src/deep_research_client/validation/term_datamodel.py.tmp \
    && mv src/deep_research_client/validation/term_datamodel.py.tmp \
      src/deep_research_client/validation/term_datamodel.py \
    || { rm -f src/deep_research_client/validation/term_datamodel.py.tmp; false; }

# Regenerate the capability/resource/archetype vocabularies from deep_research_client.yaml.
[group('model development')]
gen-datamodel-vocabulary:
  uv run gen-pydantic src/deep_research_client/schema/deep_research_client.yaml \
    > src/deep_research_client/datamodel/deep_research_client_pydantic.py.tmp \
    && mv src/deep_research_client/datamodel/deep_research_client_pydantic.py.tmp \
      src/deep_research_client/datamodel/deep_research_client_pydantic.py \
    || { rm -f src/deep_research_client/datamodel/deep_research_client_pydantic.py.tmp; false; }

# Regenerate the evaluation datamodel from evaluation.yaml.
[group('model development')]
gen-datamodel-eval:
  uv run gen-pydantic src/deep_research_client/evaluation/evaluation.yaml \
    > src/deep_research_client/evaluation/datamodel.py.tmp \
    && mv src/deep_research_client/evaluation/datamodel.py.tmp \
      src/deep_research_client/evaluation/datamodel.py \
    || { rm -f src/deep_research_client/evaluation/datamodel.py.tmp; false; }

# Regenerate the claims datamodel from claims.yaml.
[group('model development')]
gen-datamodel-claims:
  uv run gen-pydantic src/deep_research_client/claims/claims.yaml \
    > src/deep_research_client/claims/datamodel.py.tmp \
    && mv src/deep_research_client/claims/datamodel.py.tmp \
      src/deep_research_client/claims/datamodel.py \
    || { rm -f src/deep_research_client/claims/datamodel.py.tmp; false; }

# Regenerate the claim-alignment datamodel from alignment.yaml.
gen-datamodel-alignment:
  uv run gen-pydantic src/deep_research_client/claims/alignment.yaml \
    > src/deep_research_client/claims/alignment_datamodel.py.tmp \
    && mv src/deep_research_client/claims/alignment_datamodel.py.tmp \
      src/deep_research_client/claims/alignment_datamodel.py \
    || { rm -f src/deep_research_client/claims/alignment_datamodel.py.tmp; false; }

# Regenerate the claim-verification datamodel from verification.yaml.
gen-datamodel-verification:
  uv run gen-pydantic src/deep_research_client/claims/verification.yaml \
    > src/deep_research_client/claims/verification_datamodel.py.tmp \
    && mv src/deep_research_client/claims/verification_datamodel.py.tmp \
      src/deep_research_client/claims/verification_datamodel.py \
    || { rm -f src/deep_research_client/claims/verification_datamodel.py.tmp; false; }
