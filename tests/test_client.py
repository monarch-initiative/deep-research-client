"""Tests for the deep research client."""

import pytest
from unittest.mock import patch
import os

from deep_research_client import DeepResearchClient, ResearchResult, ProviderConfig, CacheConfig
from deep_research_client.providers import ResearchProvider


class MockProvider(ResearchProvider):
    """Mock provider for testing."""

    def __init__(self, config: ProviderConfig, should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail

    async def research(self, query: str) -> ResearchResult:
        """Mock research method."""
        if self.should_fail:
            raise ValueError("Mock provider failure")

        return ResearchResult(
            markdown=f"# Research Result\n\nThis is a mock result for: {query}",
            citations=["Mock Citation 1", "Mock Citation 2"],
            provider=self.name,
            query=query
        )


def test_client_initialization():
    """Test client initialization."""
    # Mock environment to avoid auto-registering providers
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient()
        assert client is not None
        assert client.cache is not None
        assert client.registry is not None


def test_client_with_custom_config():
    """Test client with custom cache configuration."""
    cache_config = CacheConfig(enabled=False, directory="/tmp/test_cache")
    # Mock environment to avoid auto-registering providers
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)
        assert client.cache_config.enabled is False
        assert client.cache_config.directory == "/tmp/test_cache"


def test_provider_setup_from_config():
    """Test setting up providers from configuration."""
    provider_configs = {
        "mock": ProviderConfig(name="mock", api_key="test-key", enabled=True)
    }

    # Mock the provider creation to avoid import issues
    original_init = DeepResearchClient._setup_providers_from_config

    def mock_setup(self, configs):
        mock_provider = MockProvider(configs["mock"])
        self.registry.register(mock_provider)

    DeepResearchClient._setup_providers_from_config = mock_setup

    client = DeepResearchClient(provider_configs=provider_configs)
    assert len(client.get_available_providers()) == 1
    assert "mock" in client.get_available_providers()

    # Restore original method
    DeepResearchClient._setup_providers_from_config = original_init


async def test_research_with_mock_provider():
    """Test research functionality with mock provider."""
    # Create client with disabled cache to avoid file system interactions
    cache_config = CacheConfig(enabled=False)
    # Mock environment to avoid auto-registering providers with delays
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

        # Add mock provider
        provider_config = ProviderConfig(name="mock", api_key="test-key", enabled=True)
        mock_provider = MockProvider(provider_config)
        client.registry.register(mock_provider)

        # Test research
        result = await client.aresearch("What is CRISPR?")

        assert result is not None
        assert result.markdown.startswith("# Research Result")
        assert "CRISPR" in result.markdown
        assert len(result.citations) == 2
        assert result.provider == "mock"
        assert result.query == "What is CRISPR?"
        assert result.cached is False


def test_research_sync():
    """Test synchronous research method."""
    cache_config = CacheConfig(enabled=False)
    # Mock environment to avoid auto-registering providers with delays
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

        # Add mock provider
        provider_config = ProviderConfig(name="mock", api_key="test-key", enabled=True)
        mock_provider = MockProvider(provider_config)
        client.registry.register(mock_provider)

        # Test research
        result = client.research("What is machine learning?")

        assert result is not None
        assert "machine learning" in result.markdown
        assert result.provider == "mock"


def test_research_no_providers():
    """Test research when no providers are available."""
    cache_config = CacheConfig(enabled=False)
    # Mock environment to avoid auto-registering providers
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

        with pytest.raises(ValueError, match="No research providers available"):
            client.research("test query")


def test_research_specific_provider():
    """Test research with specific provider."""
    cache_config = CacheConfig(enabled=False)
    # Mock environment to avoid auto-registering providers
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

        # Add mock providers
        provider1_config = ProviderConfig(name="provider1", api_key="key1", enabled=True)
        provider2_config = ProviderConfig(name="provider2", api_key="key2", enabled=True)

        mock_provider1 = MockProvider(provider1_config)
        mock_provider2 = MockProvider(provider2_config)

        client.registry.register(mock_provider1)
        client.registry.register(mock_provider2)

        # Test with specific provider
        result = client.research("test query", provider="provider2")
        assert result.provider == "provider2"


def test_research_invalid_provider():
    """Test research with invalid provider name."""
    cache_config = CacheConfig(enabled=False)
    # Mock environment to avoid auto-registering providers
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

        # Add one mock provider
        provider_config = ProviderConfig(name="mock", api_key="test-key", enabled=True)
        mock_provider = MockProvider(provider_config)
        client.registry.register(mock_provider)

        with pytest.raises(ValueError, match="Provider 'invalid' not found"):
            client.research("test query", provider="invalid")


def test_get_available_providers():
    """Test getting available providers."""
    cache_config = CacheConfig(enabled=False)
    # Mock environment to avoid auto-registering providers
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

        # Initially no providers
        assert len(client.get_available_providers()) == 0

        # Add providers
        provider1_config = ProviderConfig(name="provider1", api_key="key1", enabled=True)
        provider2_config = ProviderConfig(name="provider2", api_key=None, enabled=True)  # No API key

        mock_provider1 = MockProvider(provider1_config)
        mock_provider2 = MockProvider(provider2_config)

        client.registry.register(mock_provider1)
        client.registry.register(mock_provider2)

        # Should only return provider1 (has API key)
        available = client.get_available_providers()
        assert len(available) == 1
        assert "provider1" in available


# Integration tests that may use actual API calls
@pytest.mark.integration
def test_client_with_real_providers():
    """Test client with real provider setup from environment."""
    # This test allows auto-registration from environment and may make API calls
    client = DeepResearchClient()
    # Just test basic initialization works with real environment
    assert client is not None
    assert client.cache is not None
    assert client.registry is not None


@pytest.mark.integration
async def test_research_with_environment_providers():
    """Test research with providers from environment (may make real API calls)."""
    client = DeepResearchClient()
    providers = client.get_available_providers()

    if providers:
        # Only run if we have available providers
        # This may make real API calls depending on environment
        result = await client.aresearch("What is 2+2?")
        assert result is not None
        assert result.markdown
        assert result.provider in providers