"""Tests for the DeepER-Med stub provider."""

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.client import DeepResearchClient
from deep_research_client.models import CacheConfig, ProviderConfig
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


@pytest.mark.parametrize("alias", ["deeper-med", "deepermed"])
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


def test_model_card_marks_cost_and_speed_as_unmeasured():
    """Cost/time are transcribed from the paper, so the card must say so."""
    cards = get_provider_model_cards("deeper_med")
    card = cards.models["deeper-med-agentic"]
    assert any("not measured" in limit for limit in card.limitations)
    assert any(DEEPER_MED_ARXIV_URL in limit for limit in card.limitations)


def test_unavailable_reason_points_at_arxiv():
    """The stub explains itself rather than falling back to the generic message."""
    provider = DeeperMedProvider(ProviderConfig(name="deeper_med"))
    assert DEEPER_MED_ARXIV_URL in provider.unavailable_reason()


def test_base_provider_has_generic_unavailable_reason():
    """Other providers keep the default wording."""
    from deep_research_client.providers.mock import MockProvider

    provider = MockProvider(ProviderConfig(name="mock", enabled=False))
    assert provider.unavailable_reason() == "Provider 'mock' is not available"


# --- Client-level integration of the stub ---------------------------------


def test_client_registers_stub_without_credentials(monkeypatch):
    """The stub is registered from a bare environment so it can be asked for."""
    monkeypatch.delenv("ENABLE_MOCK_PROVIDER", raising=False)
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))
    assert client.registry.get_provider("deeper_med") is not None


def test_client_never_auto_selects_the_stub(monkeypatch):
    """Registering the stub must not make it selectable as a default provider."""
    monkeypatch.delenv("ENABLE_MOCK_PROVIDER", raising=False)
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))
    assert "deeper_med" not in client.get_available_providers()
    assert client.registry.get_first_available() is None or (
        client.registry.get_first_available().name != "deeper_med"
    )


def test_client_research_surfaces_arxiv_pointer():
    """Requesting the stub through the client explains why, with the citation.

    This is the user-facing path the provider exists to serve: without it the
    caller would only see a bare "provider not found" / "is not available".
    """
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))
    with pytest.raises(ValueError) as excinfo:
        client.research("What causes glioblastoma?", provider="deeper_med")
    assert DEEPER_MED_ARXIV_URL in str(excinfo.value)


# --- CLI discoverability --------------------------------------------------


def test_cli_lists_stub_provider():
    """`providers` must surface the stub; you shouldn't need to know the name."""
    result = CliRunner().invoke(app, ["providers"])
    assert result.exit_code == 0
    assert "deeper_med" in result.stdout
    assert "Stub providers" in result.stdout


def test_cli_stub_detail_does_not_claim_missing_api_key():
    """A stub is not credential-blocked, so it must not ask for a key."""
    result = CliRunner().invoke(app, ["providers", "--provider", "deeper_med"])
    assert result.exit_code == 0
    assert "stub - no upstream API yet" in result.stdout
    assert "missing API key" not in result.stdout
