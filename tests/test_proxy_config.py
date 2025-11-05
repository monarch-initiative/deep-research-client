"""Tests for proxy configuration."""

from deep_research_client.models import ProviderConfig
from deep_research_client.providers.openai import OpenAIProvider


class TestProxyConfiguration:
    """Test proxy configuration features."""

    def test_provider_config_with_base_url(self):
        """Test ProviderConfig can be created with base_url."""
        config = ProviderConfig(
            name="openai",
            api_key="test-key",
            base_url="https://api.example.com"
        )

        assert config.name == "openai"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com"

    def test_provider_config_without_base_url(self):
        """Test ProviderConfig works without base_url (backward compatibility)."""
        config = ProviderConfig(
            name="openai",
            api_key="test-key"
        )

        assert config.name == "openai"
        assert config.api_key == "test-key"
        assert config.base_url is None

    def test_openai_provider_with_base_url(self):
        """Test OpenAI provider can be initialized with base_url in config."""
        config = ProviderConfig(
            name="openai",
            api_key="test-key",
            base_url="https://api.cborg.lbl.gov"
        )

        provider = OpenAIProvider(config)

        assert provider.name == "openai"
        assert provider.config.base_url == "https://api.cborg.lbl.gov"
        assert provider.is_available()

    def test_openai_provider_without_base_url(self):
        """Test OpenAI provider works without base_url (default behavior)."""
        config = ProviderConfig(
            name="openai",
            api_key="test-key"
        )

        provider = OpenAIProvider(config)

        assert provider.name == "openai"
        assert provider.config.base_url is None
        assert provider.is_available()

    def test_cborg_config_scenario(self):
        """Test typical CBORG configuration scenario."""
        config = ProviderConfig(
            name="openai",
            api_key="cborg-test-key",
            base_url="https://api.cborg.lbl.gov",
            enabled=True
        )

        provider = OpenAIProvider(config)

        assert provider.config.api_key == "cborg-test-key"
        assert provider.config.base_url == "https://api.cborg.lbl.gov"
        assert provider.is_available()
        assert provider.model == "o3-deep-research-2025-06-26"
