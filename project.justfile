## Add your own just recipes here. This is imported by the main justfile.

# Regenerate the Pydantic datamodel from the LinkML schema (source of truth).
# Generates to a temporary file first: a redirect straight onto the target would
# truncate it before gen-pydantic runs, so a schema typo would leave an empty
# datamodel.py and break `import deep_research_client` until it was restored.
gen-datamodel:
  uv run gen-pydantic src/deep_research_client/validation/reference_validation.yaml \
    > src/deep_research_client/validation/datamodel.py.tmp
  mv src/deep_research_client/validation/datamodel.py.tmp \
    src/deep_research_client/validation/datamodel.py

# Regenerate the Pydantic datamodel for term validation from its LinkML schema.
# Same two-step as gen-datamodel: a redirect onto the target would truncate it
# before gen-pydantic ran, leaving an empty module behind a schema typo.
gen-term-datamodel:
  uv run gen-pydantic src/deep_research_client/validation/term_validation.yaml \
    > src/deep_research_client/validation/term_datamodel.py.tmp
  mv src/deep_research_client/validation/term_datamodel.py.tmp \
    src/deep_research_client/validation/term_datamodel.py
