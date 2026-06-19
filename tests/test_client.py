"""Tests for the deep research client."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_research_client import DeepResearchClient, ResearchResult, ProviderConfig, CacheConfig
from deep_research_client.providers import ResearchProvider
from deep_research_client.providers.asta import AstaProvider
from deep_research_client.providers.falcon import FalconProvider
from deep_research_client.providers.openscientist import OpenScientistProvider


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


def test_asta_provider_setup_from_env():
    """Asta should auto-register when its API key is present."""
    cache_config = CacheConfig(enabled=False)
    with patch.dict(
        os.environ,
        {"ASTA_API_KEY": "asta-key"},
        clear=True,
    ):
        client = DeepResearchClient(cache_config=cache_config)
        assert "asta" in client.get_available_providers()


def test_openscientist_provider_setup_from_env():
    """OpenScientist should auto-register when its API key is present."""
    cache_config = CacheConfig(enabled=False)
    with patch.dict(
        os.environ,
        {
            "OPENSCIENTIST_API_KEY": "openscientist-key",
            "OPENSCIENTIST_URL": "https://openscientist.internal",
        },
        clear=True,
    ):
        client = DeepResearchClient(cache_config=cache_config)
        assert "openscientist" in client.get_available_providers()

        provider = client.registry.get_provider("openscientist")
        assert provider is not None
        assert provider.config.api_key == "openscientist-key"
        assert provider.config.base_url == "https://openscientist.internal"
        assert provider.config.timeout == 3600


def test_client_import_does_not_eagerly_import_falcon_for_asta(tmp_path: Path):
    """Importing the client for Asta should not require Falcon/Edison dependencies."""
    fake_edison_module = tmp_path / "edison_client.py"
    fake_edison_module.write_text(
        "raise ImportError('simulated broken edison_client dependency')\n",
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(tmp_path),
            str(repo_root / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["ASTA_API_KEY"] = "asta-key"
    env.pop("EDISON_API_KEY", None)
    env.pop("FUTUREHOUSE_API_KEY", None)

    script = textwrap.dedent(
        """
        from deep_research_client import CacheConfig, DeepResearchClient

        client = DeepResearchClient(cache_config=CacheConfig(enabled=False))
        assert "asta" in client.get_available_providers()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_asta_cache_params_include_version_tag():
    """Asta cache keys should include a version tag to invalidate stale entries."""
    cache_config = CacheConfig(enabled=False)
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

    provider = AstaProvider(ProviderConfig(
        name="asta", api_key="asta-key", enabled=True))
    cache_params = client._get_cache_provider_params(
        provider, {"paper_limit": 20})

    assert cache_params == {
        "paper_limit": 20,
        "_cache_version": "snippet-v5",
    }


def test_falcon_cache_params_include_artifact_version_tag():
    """Falcon cache keys should invalidate stale no-artifact Edison entries."""
    cache_config = CacheConfig(enabled=False)
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

    provider = FalconProvider(ProviderConfig(
        name="falcon", api_key="edison-key", enabled=True))
    cache_params = client._get_cache_provider_params(provider)

    assert cache_params == {"_cache_version": "artifacts-v1"}


def test_openscientist_cache_params_include_artifact_version_tag():
    """OpenScientist cache keys should invalidate stale no-artifact entries."""
    cache_config = CacheConfig(enabled=False)
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

    provider = OpenScientistProvider(
        ProviderConfig(name="openscientist", api_key="opensci-key", enabled=True)
    )
    cache_params = client._get_cache_provider_params(provider)

    assert cache_params == {"_cache_version": "artifacts-v1"}


async def test_research_with_mock_provider():
    """Test research functionality with mock provider."""
    # Create client with disabled cache to avoid file system interactions
    cache_config = CacheConfig(enabled=False)
    # Mock environment to avoid auto-registering providers with delays
    with patch.dict(os.environ, {}, clear=True):
        client = DeepResearchClient(cache_config=cache_config)

        # Add mock provider
        provider_config = ProviderConfig(
            name="mock", api_key="test-key", enabled=True)
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
        provider_config = ProviderConfig(
            name="mock", api_key="test-key", enabled=True)
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
        provider1_config = ProviderConfig(
            name="provider1", api_key="key1", enabled=True)
        provider2_config = ProviderConfig(
            name="provider2", api_key="key2", enabled=True)

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
        provider_config = ProviderConfig(
            name="mock", api_key="test-key", enabled=True)
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
        provider1_config = ProviderConfig(
            name="provider1", api_key="key1", enabled=True)
        provider2_config = ProviderConfig(
            name="provider2", api_key=None, enabled=True)  # No API key

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
