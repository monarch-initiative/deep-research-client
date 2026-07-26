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
from deep_research_client.model_cards import ModelCard, get_provider_model_cards


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


def test_finders_deduplicate_multi_key_cards():
    """Cyberian aliases one card under two keys; finders must not double-count."""
    cards = find_models_by_archetype(ProviderArchetype.agentic_researcher)["cyberian"]
    names = [c.name for c in cards]
    assert len(names) == len(set(names))


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
