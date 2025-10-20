"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from deep_research_client.models import ResearchResult, ProviderConfig, CacheConfig


def test_research_result_creation():
    """Test creating a ResearchResult."""
    result = ResearchResult(
        markdown="# Test Result",
        citations=["Citation 1", "Citation 2"],
        provider="test-provider",
        query="test query"
    )

    assert result.markdown == "# Test Result"
    assert len(result.citations) == 2
    assert result.provider == "test-provider"
    assert result.query == "test query"
    assert result.cached is False  # Default value


def test_research_result_with_defaults():
    """Test ResearchResult with default values."""
    result = ResearchResult(
        markdown="# Test",
        provider="test",
        query="test"
    )

    assert result.citations == []  # Default empty list
    assert result.cached is False


def test_research_result_cached():
    """Test ResearchResult with cached flag."""
    result = ResearchResult(
        markdown="# Cached Result",
        provider="test",
        query="test",
        cached=True
    )

    assert result.cached is True


def test_research_result_validation():
    """Test ResearchResult validation."""
    # Missing required fields should raise ValidationError
    with pytest.raises(ValidationError):
        ResearchResult(markdown="test")  # Missing provider and query


def test_provider_config_creation():
    """Test creating a ProviderConfig."""
    config = ProviderConfig(
        name="test-provider",
        api_key="test-key",
        enabled=True,
        timeout=300,
        max_retries=5
    )

    assert config.name == "test-provider"
    assert config.api_key == "test-key"
    assert config.enabled is True
    assert config.timeout == 300
    assert config.max_retries == 5


def test_provider_config_defaults():
    """Test ProviderConfig with default values."""
    config = ProviderConfig(name="test")

    assert config.name == "test"
    assert config.api_key is None
    assert config.enabled is True
    assert config.timeout == 600
    assert config.max_retries == 3


def test_cache_config_creation():
    """Test creating a CacheConfig."""
    config = CacheConfig(
        enabled=False,
        directory="/custom/cache"
    )

    assert config.enabled is False
    assert config.directory == "/custom/cache"


def test_cache_config_defaults():
    """Test CacheConfig with default values."""
    config = CacheConfig()

    assert config.enabled is True
    assert config.directory is None  # Defaults to None, actual default path handled by cache implementation


def test_model_serialization():
    """Test that models can be serialized to JSON."""
    result = ResearchResult(
        markdown="# Test",
        citations=["ref1", "ref2"],
        provider="test",
        query="query"
    )

    # Should be able to convert to dict
    data = result.model_dump()
    assert isinstance(data, dict)
    assert data["markdown"] == "# Test"

    # Should be able to convert to JSON
    json_str = result.model_dump_json()
    assert isinstance(json_str, str)
    assert "Test" in json_str


def test_model_deserialization():
    """Test that models can be created from JSON."""
    data = {
        "markdown": "# Test Result",
        "citations": ["Citation 1"],
        "provider": "test-provider",
        "query": "test query",
        "cached": True
    }

    result = ResearchResult(**data)
    assert result.markdown == "# Test Result"
    assert result.cached is True