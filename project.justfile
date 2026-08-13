## Add your own just recipes here. This is imported by the main justfile.

# Regenerate the Pydantic datamodel from the LinkML schema (source of truth)
gen-datamodel:
  uv run gen-pydantic src/deep_research_client/validation/reference_validation.yaml \
    > src/deep_research_client/validation/datamodel.py
