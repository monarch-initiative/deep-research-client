"""Tests for provider-specific parameters and harmonization."""

import pytest

from deep_research_client.provider_params import (
    BaseProviderParams,
    OpenAIParams,
    PerplexityParams,
    FalconParams,
    create_provider_params,
)


def test_base_params_allowed_domains():
    """Test that BaseProviderParams includes allowed_domains."""
    params = BaseProviderParams(allowed_domains=["wikipedia.org", "github.com"])

    assert params.allowed_domains == ["wikipedia.org", "github.com"]


def test_base_params_allowed_domains_default():
    """Test that allowed_domains defaults to empty list."""
    params = BaseProviderParams()

    assert params.allowed_domains == []


def test_openai_params_allowed_domains():
    """Test that OpenAI params supports allowed_domains."""
    params = OpenAIParams(
        allowed_domains=["pubmed.ncbi.nlm.nih.gov", "clinicaltrials.gov"]
    )

    assert params.allowed_domains == ["pubmed.ncbi.nlm.nih.gov", "clinicaltrials.gov"]
    assert params.temperature == 0.1  # Default value


def test_perplexity_params_allowed_domains_sync():
    """Test that Perplexity syncs allowed_domains to search_domain_filter."""
    params = PerplexityParams(
        allowed_domains=["wikipedia.org", "github.com"]
    )

    # Both should be populated after validation
    assert params.allowed_domains == ["wikipedia.org", "github.com"]
    assert params.search_domain_filter == ["wikipedia.org", "github.com"]


def test_perplexity_params_search_domain_filter_only():
    """Test that Perplexity can use search_domain_filter alone."""
    params = PerplexityParams(
        search_domain_filter=["wikipedia.org", "-reddit.com"]
    )

    # search_domain_filter should be set, allowed_domains empty
    assert params.search_domain_filter == ["wikipedia.org", "-reddit.com"]
    assert params.allowed_domains == []


def test_perplexity_params_both_filters():
    """Test that search_domain_filter takes precedence when both are provided."""
    params = PerplexityParams(
        allowed_domains=["wikipedia.org"],
        search_domain_filter=["github.com", "-reddit.com"]
    )

    # search_domain_filter should be used as-is when explicitly provided
    assert params.search_domain_filter == ["github.com", "-reddit.com"]
    assert params.allowed_domains == ["wikipedia.org"]


def test_perplexity_params_denylist():
    """Test that Perplexity supports denylist with - prefix."""
    params = PerplexityParams(
        search_domain_filter=["-reddit.com", "-quora.com"]
    )

    assert params.search_domain_filter == ["-reddit.com", "-quora.com"]


def test_perplexity_params_mixed_allowdeny():
    """Test that Perplexity supports mixed allow/deny lists."""
    params = PerplexityParams(
        search_domain_filter=["github.com", "stackoverflow.com", "-reddit.com"]
    )

    assert len(params.search_domain_filter) == 3
    assert "github.com" in params.search_domain_filter
    assert "-reddit.com" in params.search_domain_filter


def test_create_provider_params_openai():
    """Test creating OpenAI params via factory function."""
    params = create_provider_params(
        "openai",
        model="o3-deep-research-2025-06-26",
        provider_params={"allowed_domains": ["wikipedia.org"]}
    )

    assert isinstance(params, OpenAIParams)
    assert params.model == "o3-deep-research-2025-06-26"
    assert params.allowed_domains == ["wikipedia.org"]


def test_create_provider_params_perplexity():
    """Test creating Perplexity params via factory function."""
    params = create_provider_params(
        "perplexity",
        provider_params={
            "allowed_domains": ["github.com"],
            "reasoning_effort": "high"
        }
    )

    assert isinstance(params, PerplexityParams)
    assert params.allowed_domains == ["github.com"]
    assert params.search_domain_filter == ["github.com"]  # Should be synced
    assert params.reasoning_effort == "high"


def test_create_provider_params_invalid():
    """Test that invalid parameters are rejected."""
    with pytest.raises(ValueError, match="Invalid parameters"):
        create_provider_params(
            "openai",
            provider_params={"invalid_param": "value"}
        )


def test_falcon_params_allowed_domains():
    """Test that Falcon params inherits allowed_domains (though not used by API)."""
    params = FalconParams(
        allowed_domains=["wikipedia.org"]
    )

    # Should be accepted even if Falcon doesn't support it yet
    assert params.allowed_domains == ["wikipedia.org"]


def test_allowed_domains_max_limit_documented():
    """Test that documentation mentions 20 domain limit."""
    # Check that the field description mentions the limit
    field_info = BaseProviderParams.model_fields["allowed_domains"]
    assert "20" in field_info.description or "max" in field_info.description.lower()


def test_perplexity_sync_preserves_denylist():
    """Test that syncing doesn't corrupt denylist entries."""
    # If only allowed_domains is provided (no deny entries)
    params = PerplexityParams(allowed_domains=["github.com"])

    # search_domain_filter should be pure allowlist
    assert all(not domain.startswith("-") for domain in params.search_domain_filter)
