"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError
import yaml

import deep_research_client.models as model_module
from deep_research_client.formatter import ResultFormatter as LegacyResultFormatter
from deep_research_client.models import (
    CacheConfig,
    ProviderConfig,
    ResearchArtifact,
    ResearchResult,
)
from deep_research_client.processing import ResearchProcessor, ResultFormatter


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
    assert result.artifacts == []  # Default empty list
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
    assert config.timeout is None  # Provider-specific default
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


def test_research_artifact_image_detection():
    """Research artifacts should identify embeddable image media types."""
    artifact = ResearchArtifact(
        filename="figure.png",
        content_base64="ZmFrZQ==",
        media_type="image/png",
    )

    assert artifact.is_image is True


def test_research_artifact_sanitizes_filename():
    """Artifact filenames should be sanitized at model construction time."""
    artifact = ResearchArtifact(
        filename=r"..\..\figure.png",
        content_base64="ZmFrZQ==",
    )

    assert artifact.filename == "_figure.png"


def test_research_artifact_rejects_oversized_content(monkeypatch):
    """Artifact model validation should reject payloads above the size limit."""
    monkeypatch.setattr(model_module, "MAX_ARTIFACT_BYTES", 10)

    with pytest.raises(ValueError, match="Artifact too large"):
        ResearchArtifact(
            filename="large.bin",
            content_base64="A" * 16,
        )


def test_format_research_result_includes_image_artifacts():
    """Formatted markdown should embed image artifacts."""
    result = ResearchResult(
        markdown="# Report",
        provider="falcon",
        query="query",
        artifacts=[
            ResearchArtifact(
                filename="figure.png",
                content_base64="ZmFrZQ==",
                media_type="image/png",
                path="report_artifacts/figure.png",
                description="Figure 1",
            )
        ],
    )

    formatted = ResearchProcessor().format_research_result(result)

    assert "artifact_count: 1" in formatted
    assert "## Artifacts" in formatted
    assert "![Figure 1](report_artifacts/figure.png)" in formatted


@pytest.mark.parametrize("formatter_class", [ResultFormatter, LegacyResultFormatter])
def test_format_research_result_includes_artifact_frontmatter(formatter_class):
    """Formatter frontmatter should preserve structured artifact metadata."""
    result = ResearchResult(
        markdown="# Report",
        provider="falcon",
        query="query",
        provider_config={"trajectory_id": "784d73d5-da42-402e-9701-6c5b44beab14"},
        artifacts=[
            ResearchArtifact(
                filename="figure.png",
                content_base64="ZmFrZQ==",
                media_type="image/png",
                path="report_artifacts/figure.png",
                source="edison_output_data",
                data_storage_id="11111111-1111-1111-1111-111111111111",
                description="Figure 1",
            )
        ],
    )

    formatted = formatter_class().format_full_markdown(result)
    frontmatter = yaml.safe_load(formatted.split("---", 2)[1])

    assert frontmatter["artifact_count"] == 1
    assert frontmatter["trajectory_id"] == "784d73d5-da42-402e-9701-6c5b44beab14"
    assert frontmatter["artifact_sources"] == {"edison_output_data": 1}
    assert frontmatter["artifacts"] == [
        {
            "filename": "figure.png",
            "path": "report_artifacts/figure.png",
            "media_type": "image/png",
            "source": "edison_output_data",
            "data_storage_id": "11111111-1111-1111-1111-111111111111",
            "description": "Figure 1",
        }
    ]


@pytest.mark.parametrize("formatter_class", [ResultFormatter, LegacyResultFormatter])
def test_format_research_result_handles_mixed_artifacts(formatter_class):
    """Formatters should embed images, link files, and preserve artifact order."""
    long_description = "Supplementary table " + "with detailed evidence " * 20
    result = ResearchResult(
        markdown="# Report",
        provider="falcon",
        query="query",
        artifacts=[
            ResearchArtifact(
                filename="figure.png",
                content_base64="ZmFrZQ==",
                media_type="image/png",
                path="report_artifacts/figure.png",
                description="Figure 1",
            ),
            ResearchArtifact(
                filename="supplement.md",
                content_base64="ZmFrZQ==",
                media_type="text/markdown",
                path="report_artifacts/supplement.md",
                description=long_description,
            ),
            ResearchArtifact(
                filename="raw.json",
                content_base64="e30=",
                media_type="application/json",
            ),
        ],
    )

    formatted = formatter_class().format_full_markdown(result)

    image_line = "![Figure 1](report_artifacts/figure.png)"
    supplement_line = f"- [{long_description}](report_artifacts/supplement.md)"
    fallback_line = "- [raw.json](raw.json)"
    assert image_line in formatted
    assert supplement_line in formatted
    assert fallback_line in formatted
    assert formatted.index(image_line) < formatted.index(supplement_line)
    assert formatted.index(supplement_line) < formatted.index(fallback_line)
