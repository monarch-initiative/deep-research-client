"""Tests for the Biomni provider.

These unit tests deliberately run *without* the optional ``biomni`` package
installed: the provider imports ``biomni`` lazily, so its scaffolding (params,
model card, availability, output parsing) is testable on its own. The actual
agent run is exercised by the integration test at the bottom, which is skipped
unless ``biomni`` is importable.
"""

import importlib.util

import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.model_cards import (
    ProviderArchetype,
    ResearchCapability,
    ResearchResource,
    create_biomni_model_cards,
)
from deep_research_client.provider_params import BiomniParams, create_provider_params
from deep_research_client.providers.biomni import BiomniProvider

BIOMNI_INSTALLED = importlib.util.find_spec("biomni") is not None


def make_provider(**params) -> BiomniProvider:
    config = ProviderConfig(name="biomni", api_key=None, enabled=True)
    return BiomniProvider(config, BiomniParams(**params))


def test_default_model_is_card_identity():
    assert make_provider().get_default_model() == "biomni-a1"
    assert make_provider().model == "biomni-a1"


@pytest.mark.parametrize(
    "kwargs,env,expected",
    [
        ({"path": "/data/custom"}, {}, "/data/custom"),
        ({}, {"BIOMNI_DATA_PATH": "/data/env"}, "/data/env"),
        ({}, {}, "./biomni_data"),
    ],
)
def test_data_path_resolution(kwargs, env, expected, monkeypatch):
    monkeypatch.delenv("BIOMNI_DATA_PATH", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert make_provider(**kwargs).data_path == expected


def test_params_llm_distinct_from_model():
    """The underlying LLM is a separate param from the research model card."""
    params = create_provider_params(
        "biomni",
        model=None,
        provider_params={"llm": "claude-sonnet-4-20250514", "source": "Anthropic"},
    )
    assert params.llm == "claude-sonnet-4-20250514"
    assert params.source == "Anthropic"


def test_params_reject_unknown_field():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BiomniParams(not_a_real_field=True)


def test_is_available_tracks_package_presence():
    provider = make_provider()
    assert provider.is_available() is BIOMNI_INSTALLED


def test_disabled_config_is_unavailable():
    config = ProviderConfig(name="biomni", enabled=False)
    assert BiomniProvider(config).is_available() is False


@pytest.mark.asyncio
async def test_research_requires_available_provider():
    if BIOMNI_INSTALLED:
        pytest.skip("biomni installed; availability guard not exercised")
    with pytest.raises(ValueError, match="not available"):
        await make_provider().research("some biomedical question")


@pytest.mark.asyncio
async def test_research_rejects_empty_query():
    # The empty-query guard runs before the availability check, so it is
    # exercised whether or not the optional package is installed.
    with pytest.raises(ValueError, match="must not be empty"):
        await make_provider().research("   ")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("final answer", "final answer"),
        (("verbose log", "final answer"), "final answer"),
        (["a", "b", "c"], "c"),
        ([], ""),
    ],
)
def test_result_to_markdown(raw, expected):
    assert BiomniProvider._result_to_markdown(raw) == expected


def test_result_to_markdown_object_with_content():
    class Result:
        content = "the report"

    assert BiomniProvider._result_to_markdown(Result()) == "the report"


def test_extract_citations_pmids_and_dois():
    text = "Evidence PMID: 12345678 and PMID:12345678 plus doi:10.1000/abc.def."
    assert BiomniProvider._extract_citations(text) == [
        "PMID:12345678",
        "doi:10.1000/abc.def",
    ]


def test_biomni_model_card_shape():
    cards = create_biomni_model_cards()
    card = cards.get_model_card("biomni-a1")
    assert card.archetype == ProviderArchetype.co_scientist
    assert ResearchResource.pubmed in card.resources
    assert ResearchCapability.code_interpretation in card.capabilities
    assert ResearchCapability.hypothesis_generation in card.capabilities
    # Aliases resolve to the canonical model name
    assert cards.resolve_model_name("biomni") == "biomni-a1"


def test_build_agent_kwargs_accepted_by_a1_signature():
    """Every kwarg _build_agent can emit must be accepted by A1.__init__.

    Guards against silent signature drift (e.g. `timeout` vs `timeout_seconds`)
    without mocking or constructing the agent (which would download the data
    lake). Skipped when the optional package is not installed.
    """
    import inspect

    pytest.importorskip("biomni")
    from biomni.agent import A1  # type: ignore[import-not-found, import-untyped]

    accepted = set(inspect.signature(A1.__init__).parameters)
    # Superset of keys _build_agent may produce across all param combinations.
    possible_kwargs = {
        "path",
        "use_tool_retriever",
        "llm",
        "source",
        "base_url",
        "api_key",
        "timeout_seconds",
        "expected_data_lake_files",
    }
    missing = possible_kwargs - accepted
    assert not missing, f"A1.__init__ does not accept: {sorted(missing)}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_biomni_end_to_end():
    """Run a real (small) Biomni task. Requires the biomni package + LLM key."""
    pytest.importorskip("biomni")
    provider = make_provider(skip_data_lake=True)
    result = await provider.research(
        "In one sentence, what does the tumor suppressor gene TP53 do?"
    )
    assert result.provider == "biomni"
    assert result.markdown.strip()
