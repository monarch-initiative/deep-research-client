"""Tests for the capability / resource / archetype controlled vocabularies."""

import pytest

from deep_research_client import (
    ModelCapability,
    ProviderArchetype,
    ResearchCapability,
    ResearchResource,
    find_models_by_archetype,
    find_models_by_capability,
    find_models_by_resource,
)
from deep_research_client.model_cards import (
    PROVIDER_MODEL_CARDS,
    ModelCard,
    ProviderModelCards,
    get_provider_model_cards,
)


def test_model_capability_is_research_capability_alias():
    """The legacy name stays importable and points at the canonical enum."""
    assert ModelCapability is ResearchCapability


@pytest.mark.parametrize(
    "value",
    [
        "web_search",
        "academic_search",
        "scientific_literature",
        "citation_tracking",
        "real_time_data",
        "code_interpretation",
        "visual_analysis",
        "multi_language",
    ],
)
def test_legacy_capability_values_preserved(value):
    """All historical ModelCapability string values still resolve by value."""
    assert ResearchCapability(value).value == value


@pytest.mark.parametrize(
    "value",
    [
        "hypothesis_generation",
        "experiment_design",
        "data_analysis",
        "evidence_synthesis",
        "retrieval_only",
    ],
)
def test_new_capabilities_present(value):
    """Co-scientist-oriented capabilities were added."""
    assert ResearchCapability(value).value == value


@pytest.mark.parametrize(
    "value",
    [
        "general_web",
        "pubmed",
        "semantic_scholar",
        "arxiv",
        "preprint_servers",
        "clinical_trials",
        "biomedical_databases",
        "genomic_databases",
        "chemical_databases",
        "protein_structure_databases",
    ],
)
def test_research_resource_values(value):
    assert ResearchResource(value).value == value


def test_provider_archetype_spectrum_ordered():
    """Archetypes form the retrieval -> co-scientist spectrum."""
    assert [a.value for a in ProviderArchetype] == [
        "retriever",
        "synthesizer",
        "agentic_researcher",
        "co_scientist",
    ]


def test_model_card_has_capability_fields():
    """ModelCard exposes capabilities, resources, and archetype."""
    card = ModelCard(
        name="x",
        display_name="X",
        description="d",
        cost_level="low",
        time_estimate="fast",
        capabilities=[ResearchCapability.code_interpretation],
        resources=[ResearchResource.pubmed],
        archetype=ProviderArchetype.co_scientist,
    )
    assert card.archetype == ProviderArchetype.co_scientist
    assert ResearchResource.pubmed in card.resources


def test_openscientist_and_biomni_are_co_scientists():
    """OpenScientist and Biomni are co-scientists (others may be added later)."""
    providers = set(find_models_by_archetype(ProviderArchetype.co_scientist))
    assert providers >= {"openscientist", "biomni"}


def test_cyberian_registered_as_agentic_researcher():
    """Cyberian is registered and discoverable via its archetype."""
    providers = set(find_models_by_archetype(ProviderArchetype.agentic_researcher))
    assert "cyberian" in providers


def test_unique_cards_dedupes_a_multi_key_provider():
    """_unique_cards collapses one card aliased under two keys to a single hit.

    Uses a synthetic provider because no shipped provider currently maps one
    ModelCard object to two keys; this keeps the defensive helper covered.
    """
    card = ModelCard(
        name="dup",
        display_name="Dup",
        description="d",
        cost_level="low",
        time_estimate="fast",
        archetype=ProviderArchetype.retriever,
    )
    cards = ProviderModelCards(
        provider_name="synthetic",
        default_model="dup",
        models={"dup": card, "dup-alias": card},
    )
    assert cards.get_models_by_archetype(ProviderArchetype.retriever) == [card]
    assert cards.get_models_by_cost("low") == [card]


def test_cyberian_lists_one_card_but_still_resolves_deep_research():
    """`deep-research` is an alias, not a second card, so list_models has no dup."""
    cards = get_provider_model_cards("cyberian")
    assert cards.list_models() == ["Cyberian Deep Research"]
    # The workflow-name alias must still resolve to the canonical card.
    assert cards.resolve_model_name("deep-research") == "Cyberian Deep Research"


def test_model_capability_uppercase_aliases_preserved():
    """Legacy UPPER_CASE attribute access still resolves to the same members."""
    assert ModelCapability.WEB_SEARCH is ResearchCapability.web_search
    assert ModelCapability.CODE_INTERPRETATION is ResearchCapability.code_interpretation


def test_find_by_resource_and_capability():
    """Cross-provider lookups return provider -> cards mappings."""
    code_runners = find_models_by_capability(ResearchCapability.code_interpretation)
    assert "biomni" in code_runners

    pubmed_backed = find_models_by_resource(ResearchResource.pubmed)
    assert "biomni" in pubmed_backed
    assert "openscientist" in pubmed_backed


def test_asta_is_retriever():
    """Retrieval-only providers carry the retriever archetype."""
    card = get_provider_model_cards("asta").get_model_card(
        "Asta Scientific Corpus Retrieval"
    )
    assert card.archetype == ProviderArchetype.retriever


def test_vocabulary_datamodel_matches_linkml_schema() -> None:
    """The generated vocabularies are pinned to their schema, as validation's are.

    deep_research_client_pydantic.py is generated; regenerate it with
    `just gen-datamodel`. Without this, a linkml upgrade silently leaves the
    checked-in enums behind the schema that is their source of truth.
    """
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    schema = Path("src/deep_research_client/schema/deep_research_client.yaml")
    generated = repo_root / "src/deep_research_client/datamodel/deep_research_client_pydantic.py"

    # Resolve the generator next to the running interpreter so the comparison
    # uses the linkml version this environment pins, not whatever is on PATH.
    gen_pydantic = shutil.which("gen-pydantic", path=str(Path(sys.executable).parent))
    if not gen_pydantic:
        pytest.skip("linkml is not installed; install the dev dependency group to check drift")

    completed = subprocess.run(
        [gen_pydantic, str(schema)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"gen-pydantic failed:\n{completed.stderr}"

    assert completed.stdout == generated.read_text(encoding="utf-8"), (
        "deep_research_client_pydantic.py does not match deep_research_client.yaml. "
        "Either the schema changed or linkml was upgraded; run `just gen-datamodel` "
        "and review the diff."
    )


def test_every_provider_default_model_resolves_to_a_card():
    """The name a provider calls its default has to name a card it owns.

    Derived from the registry rather than listed, so a provider added later is
    covered the day it lands. Two names must resolve: the one the cards
    advertise as their default, and the one the provider class returns from
    get_default_model() -- which is what lands in `provider.model`, and so what
    a caller writing `cards.get_model_card_by_alias(provider.model)` looks up.
    """
    import importlib

    from deep_research_client.client import PROVIDER_CLASS_PATHS
    from deep_research_client.model_cards import PROVIDER_MODEL_CARDS
    from deep_research_client.models import ProviderConfig

    unresolved = []
    for name, cards in sorted(PROVIDER_MODEL_CARDS.items()):
        if cards.get_model_card_by_alias(cards.default_model) is None:
            unresolved.append(f"{name}: cards.default_model={cards.default_model!r}")

        module_name, class_name = PROVIDER_CLASS_PATHS[name]
        provider_class = getattr(importlib.import_module(module_name), class_name)
        default = provider_class(ProviderConfig(name=name)).get_default_model()
        if cards.get_model_card_by_alias(default) is None:
            unresolved.append(f"{name}: get_default_model()={default!r}")

    assert not unresolved, "default model names that match no card or alias: " + "; ".join(unresolved)


def test_the_default_model_guard_covers_every_registered_provider():
    """A derived guard that iterated nothing would pass without checking anything."""
    from deep_research_client.client import PROVIDER_CLASS_PATHS
    from deep_research_client.model_cards import PROVIDER_MODEL_CARDS

    assert len(PROVIDER_MODEL_CARDS) >= 10
    # Every carded provider must be constructible through the same registry the
    # guard above walks, or that provider is silently skipped.
    assert not set(PROVIDER_MODEL_CARDS) - set(PROVIDER_CLASS_PATHS)


def _all_registered_cards() -> list:
    """Every (provider, card) pair in the registry, deduplicated by identity.

    Derived rather than listed, like the default-model guard: a card added to a
    provider is covered the day it lands. Dedup reuses the finders' own helper
    so this walks the registry exactly as they do -- a provider that aliases one
    card under two keys yields it once, here and there alike.

    Returns:
        (provider name, ModelCard) pairs
    """
    return [
        (provider, card)
        for provider, cards in sorted(PROVIDER_MODEL_CARDS.items())
        for card in cards.unique_models()
    ]


#: Computed once: parametrize and its ids both need it, and the walk is not free.
_ALL_CARDS = _all_registered_cards()


@pytest.mark.parametrize(
    "provider,card",
    _ALL_CARDS,
    ids=[f"{provider}/{card.name}" for provider, card in _ALL_CARDS],
)
def test_archetype_and_capabilities_agree(provider, card):
    """The two axes describe one provider, so they must not contradict.

    Deliberately pins only what the schema states, not the tidier biconditional:

    - `retrieval_only` "pairs with the ``retriever`` archetype" -- stated
      outright, so a retriever carries it and a non-retriever does not.
    - `synthesizer` is "searches one or more corpora and writes a cited
      narrative report"; `evidence_synthesis` is "synthesises multiple sources
      into an authored narrative report". The same sentence, so a synthesizer
      carries it.

    Nothing in the schema says an `agentic_researcher` or `co_scientist` must
    author a narrative report -- a co-scientist is defined by hypotheses and
    experiments. Every one we ship happens to write a report and is annotated
    accordingly, but requiring it here would force a future co-scientist that
    returns a ranked hypothesis list to claim a capability it does not have.
    """
    assert card.archetype is not None, (
        f"{provider}/{card.name} declares no archetype, so there is nothing to "
        "check its capabilities against"
    )
    capabilities = set(card.capabilities)

    if card.archetype == ProviderArchetype.retriever:
        assert ResearchCapability.retrieval_only in capabilities, (
            f"{provider}/{card.name} is a retriever but is not marked retrieval_only"
        )
        assert ResearchCapability.evidence_synthesis not in capabilities, (
            f"{provider}/{card.name} retrieves; it does not author a synthesis"
        )
    else:
        assert ResearchCapability.retrieval_only not in capabilities, (
            f"{provider}/{card.name} is a {card.archetype}, not retrieval-only"
        )

    if card.archetype == ProviderArchetype.synthesizer:
        assert ResearchCapability.evidence_synthesis in capabilities, (
            f"{provider}/{card.name} is a synthesizer -- the archetype whose "
            "definition is evidence_synthesis -- but is not marked with it"
        )


def test_both_coherence_terms_are_actually_used():
    """A vocabulary term on zero cards is the defect this pair of terms had."""
    by_capability = {
        term: find_models_by_capability(term)
        for term in (ResearchCapability.retrieval_only, ResearchCapability.evidence_synthesis)
    }

    for term, providers in by_capability.items():
        assert providers, f"{term} is defined in the schema but annotated on no card"

    # The case that motivated the check: a user asking who writes them a report
    # must get the deep-research tools, not only the co-scientists. Subset, not
    # equality -- which providers hold each term is for the tests above, and a
    # second retriever should not fail a test about the terms being used at all.
    assert {"openai", "perplexity", "consensus", "falcon"} <= set(
        by_capability[ResearchCapability.evidence_synthesis]
    )
    assert "asta" in by_capability[ResearchCapability.retrieval_only]


#: Vocabulary terms defined in the schema but deliberately on no card yet, with
#: the reason. Anything not listed here must be annotated somewhere -- the CLI
#: offers every permissible value in its help and errors, so an unused term is a
#: filter a user can run that returns nothing.
RESERVED_TERMS = {
    # Every provider that reads preprints reaches arXiv through a broader
    # source (semantic_scholar, preprint_servers) rather than as a distinct
    # one. Kept for a provider that queries arXiv directly.
    ResearchResource.arxiv: "subsumed by preprint_servers / semantic_scholar today",
}


@pytest.mark.parametrize(
    "term",
    [
        *ResearchCapability,
        *ResearchResource,
        *ProviderArchetype,
    ],
    ids=lambda term: f"{type(term).__name__}.{term.value}",
)
def test_every_vocabulary_term_is_annotated_or_explicitly_reserved(term):
    """A term the CLI offers must match something, or say why it does not.

    `models --capability <term>` accepts every permissible value, so a term on
    zero cards is a documented filter that returns an empty list. Either a card
    claims it or RESERVED_TERMS records the intent.
    """
    finders = {
        ResearchCapability: find_models_by_capability,
        ResearchResource: find_models_by_resource,
        ProviderArchetype: find_models_by_archetype,
    }
    matches = finders[type(term)](term)

    if term in RESERVED_TERMS:
        assert not matches, (
            f"{term} is listed in RESERVED_TERMS ({RESERVED_TERMS[term]}) but is "
            f"now annotated on {sorted(matches)}; remove it from the reserved list"
        )
        return

    assert matches, (
        f"{type(term).__name__}.{term.value} is defined in the schema and offered "
        "by the CLI but annotated on no card. Annotate it, or add it to "
        "RESERVED_TERMS with the reason."
    )


def test_every_registered_provider_has_a_params_class():
    """`providers` renders its listing from the params registry, not the paths.

    A provider added to PROVIDER_CLASS_PATHS without a params class would be
    constructible and invisible there. The model-cards side is checked by
    test_every_provider_default_model_resolves_to_a_card; this closes the other
    registry.
    """
    from deep_research_client.client import PROVIDER_CLASS_PATHS
    from deep_research_client.provider_params import PROVIDER_PARAMS_REGISTRY

    assert set(PROVIDER_CLASS_PATHS) == set(PROVIDER_PARAMS_REGISTRY), (
        "PROVIDER_CLASS_PATHS and PROVIDER_PARAMS_REGISTRY have drifted: "
        f"only in class paths {set(PROVIDER_CLASS_PATHS) - set(PROVIDER_PARAMS_REGISTRY)}, "
        f"only in params {set(PROVIDER_PARAMS_REGISTRY) - set(PROVIDER_CLASS_PATHS)}"
    )
