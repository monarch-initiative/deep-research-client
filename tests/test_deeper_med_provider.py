"""Tests for the DeepER-Med stub provider."""

import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import (
    DeeperMedParams,
    create_provider_params,
    get_provider_params_class,
)
from deep_research_client.providers.deeper_med import (
    DEEPER_MED_ARXIV_URL,
    DeeperMedProvider,
)
from deep_research_client.model_cards import (
    PROVIDER_MODEL_CARDS,
    get_provider_model_cards,
)


def test_provider_is_unavailable_by_default():
    provider = DeeperMedProvider(ProviderConfig(name="deeper_med"))
    assert provider.is_available() is False


def test_provider_is_unavailable_even_with_api_key():
    """Stub stays disabled until an upstream endpoint exists."""
    provider = DeeperMedProvider(
        ProviderConfig(name="deeper_med", api_key="placeholder")
    )
    assert provider.is_available() is False


def test_default_model_name():
    provider = DeeperMedProvider(ProviderConfig(name="deeper_med"))
    assert provider.model == "deeper-med-agentic"


@pytest.mark.parametrize("alias", ["deeper-med", "deeper_med", "deepermed"])
def test_model_aliases_resolve(alias):
    provider = DeeperMedProvider(
        ProviderConfig(name="deeper_med"), DeeperMedParams(model=alias)
    )
    assert provider.model == "deeper-med-agentic"


@pytest.mark.asyncio
async def test_research_raises_with_arxiv_pointer():
    provider = DeeperMedProvider(ProviderConfig(name="deeper_med"))
    with pytest.raises(NotImplementedError) as excinfo:
        await provider.research("What causes glioblastoma?")
    assert DEEPER_MED_ARXIV_URL in str(excinfo.value)


def test_provider_params_registered():
    assert get_provider_params_class("deeper_med") is DeeperMedParams
    params = create_provider_params("deeper_med")
    assert isinstance(params, DeeperMedParams)


def test_model_card_registered():
    assert "deeper_med" in PROVIDER_MODEL_CARDS
    cards = get_provider_model_cards("deeper_med")
    assert cards is not None
    assert cards.default_model == "deeper-med-agentic"
