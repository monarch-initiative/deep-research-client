"""Test cache hashing behavior with provider parameters, especially search_domain_filter."""

import pytest
import tempfile
from pathlib import Path

from deep_research_client.cache import CacheManager
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.providers.mock import MockProvider
from deep_research_client.provider_params import PerplexityParams


@pytest.fixture
def cache_manager():
    """Create a temporary cache manager for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CacheConfig(enabled=True, directory=temp_dir)
        yield CacheManager(config)


@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    config = ProviderConfig(name="mock", api_key="test")
    return MockProvider(config)


class TestCacheHashing:
    """Test cache hashing with various provider parameters."""

    def test_basic_cache_filename_generation(self, cache_manager):
        """Test basic cache filename generation without parameters."""
        query = "What is machine learning?"
        provider = "mock"

        filename = cache_manager._get_cache_filename(query, provider)

        assert filename.startswith("mock-what-is-machine-learning-")
        assert filename.endswith(".json")
        assert len(filename.split("-")[-1].replace(".json", "")) == 8  # Hash suffix

    def test_cache_filename_with_model(self, cache_manager):
        """Test cache filename generation with model parameter."""
        query = "AI research"
        provider = "mock"
        model = "mock-model-v2"

        filename = cache_manager._get_cache_filename(query, provider, model)

        assert filename.startswith("mock-ai-research-")
        assert filename.endswith(".json")

    def test_cache_filename_with_search_domain_filter(self, cache_manager):
        """Test cache filename generation with search_domain_filter parameter."""
        query = "AI research"
        provider = "perplexity"
        provider_params = {
            "search_domain_filter": ["wikipedia.org", "github.com"]
        }

        filename1 = cache_manager._get_cache_filename(query, provider, provider_params=provider_params)

        # Test with different domain filter order - should produce different hash
        provider_params2 = {
            "search_domain_filter": ["github.com", "wikipedia.org"]
        }
        filename2 = cache_manager._get_cache_filename(query, provider, provider_params=provider_params2)

        # Different parameter values should produce different hashes
        assert filename1 != filename2
        assert filename1.startswith("perplexity-ai-research-")
        assert filename2.startswith("perplexity-ai-research-")

    def test_cache_filename_with_denylist_domains(self, cache_manager):
        """Test cache filename generation with denylist domain filters."""
        query = "research topic"
        provider = "perplexity"

        # Test allowlist
        allowlist_params = {
            "search_domain_filter": ["wikipedia.org", "github.com"]
        }
        filename_allowlist = cache_manager._get_cache_filename(query, provider, provider_params=allowlist_params)

        # Test denylist (with - prefix)
        denylist_params = {
            "search_domain_filter": ["-reddit.com", "-quora.com"]
        }
        filename_denylist = cache_manager._get_cache_filename(query, provider, provider_params=denylist_params)

        # Should produce different hashes
        assert filename_allowlist != filename_denylist

    def test_cache_filename_with_multiple_perplexity_params(self, cache_manager):
        """Test cache filename generation with multiple Perplexity parameters."""
        query = "complex query"
        provider = "perplexity"
        provider_params = {
            "search_domain_filter": ["arxiv.org", "scholar.google.com"],
            "search_recency_filter": "week",
            "temperature": 0.2,
            "reasoning_effort": "high"
        }

        filename = cache_manager._get_cache_filename(query, provider, provider_params=provider_params)

        assert filename.startswith("perplexity-complex-query-")
        assert filename.endswith(".json")

    def test_parameter_sorting_consistency(self, cache_manager):
        """Test that parameter order doesn't affect hash (parameters are sorted)."""
        query = "test query"
        provider = "perplexity"

        params1 = {
            "search_domain_filter": ["example.com"],
            "temperature": 0.1,
            "search_recency_filter": "month"
        }

        params2 = {
            "temperature": 0.1,
            "search_recency_filter": "month",
            "search_domain_filter": ["example.com"]
        }

        filename1 = cache_manager._get_cache_filename(query, provider, provider_params=params1)
        filename2 = cache_manager._get_cache_filename(query, provider, provider_params=params2)

        # Same parameters in different order should produce same filename
        assert filename1 == filename2

    def test_empty_vs_populated_domain_filter(self, cache_manager):
        """Test that empty domain filter produces different hash than populated one."""
        query = "test query"
        provider = "perplexity"

        # Empty domain filter
        params_empty = {
            "search_domain_filter": []
        }

        # Populated domain filter
        params_populated = {
            "search_domain_filter": ["example.com"]
        }

        # No params at all
        filename_none = cache_manager._get_cache_filename(query, provider)
        filename_empty = cache_manager._get_cache_filename(query, provider, provider_params=params_empty)
        filename_populated = cache_manager._get_cache_filename(query, provider, provider_params=params_populated)

        # All should produce different hashes
        assert filename_none != filename_empty
        assert filename_empty != filename_populated
        assert filename_none != filename_populated

    @pytest.mark.asyncio
    async def test_end_to_end_caching_with_domain_filter(self, cache_manager, mock_provider):
        """Test end-to-end caching with search_domain_filter parameter."""
        query = "AI research"
        provider_params = {
            "search_domain_filter": ["wikipedia.org", "github.com"],
            "search_recency_filter": "week"
        }

        # First request - should not be cached
        result1 = await cache_manager.get(query, "mock", provider_params=provider_params)
        assert result1 is None

        # Generate mock result
        mock_result = await mock_provider.research(query)

        # Cache the result
        await cache_manager.set(query, "mock", mock_result, provider_params=provider_params)

        # Second request - should be cached
        result2 = await cache_manager.get(query, "mock", provider_params=provider_params)
        assert result2 is not None
        assert result2.cached is True
        assert result2.query == query

        # Different parameters should not hit cache
        different_params = {
            "search_domain_filter": ["example.com"]
        }
        result3 = await cache_manager.get(query, "mock", provider_params=different_params)
        assert result3 is None

    def test_special_characters_in_domain_filter(self, cache_manager):
        """Test that special characters in domain filter are handled properly."""
        query = "special test"
        provider = "perplexity"

        # Domain with special characters
        provider_params = {
            "search_domain_filter": ["sub-domain.example.com", "test_site.org"]
        }

        filename = cache_manager._get_cache_filename(query, provider, provider_params=provider_params)

        # Should generate valid filename
        assert filename.startswith("perplexity-special-test-")
        assert filename.endswith(".json")

        # Verify filename is valid (no invalid characters)
        path = Path(filename)
        assert path.name == filename  # Valid filename

    def test_long_domain_list_hashing(self, cache_manager):
        """Test hashing with long list of domains (up to the 20 domain limit)."""
        query = "comprehensive search"
        provider = "perplexity"

        # Create a list of 20 domains (max allowed)
        domains = [f"site{i}.com" for i in range(20)]
        provider_params = {
            "search_domain_filter": domains
        }

        filename = cache_manager._get_cache_filename(query, provider, provider_params=provider_params)

        assert filename.startswith("perplexity-comprehensive-search-")
        assert filename.endswith(".json")

        # Verify consistent hashing with same long list
        filename2 = cache_manager._get_cache_filename(query, provider, provider_params=provider_params)
        assert filename == filename2

    def test_pydantic_model_params_hashing(self, cache_manager):
        """Test hashing when using Pydantic model for parameters."""
        query = "pydantic test"
        provider = "perplexity"

        # Create PerplexityParams instance
        params = PerplexityParams(
            search_domain_filter=["example.com", "test.org"],
            search_recency_filter="day",
            temperature=0.3
        )

        # Convert to dict for hashing
        params_dict = params.model_dump(exclude_unset=True)

        filename = cache_manager._get_cache_filename(query, provider, provider_params=params_dict)

        assert filename.startswith("perplexity-pydantic-test-")
        assert filename.endswith(".json")