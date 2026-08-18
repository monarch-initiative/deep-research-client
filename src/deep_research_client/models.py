"""Pydantic models for deep research client."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import re

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
_UNSAFE_ARTIFACT_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_REPEATED_UNDERSCORES = re.compile(r"_+")


def sanitize_artifact_filename(raw_name: str, fallback: str = "artifact") -> str:
    """Return a filesystem-safe basename for an artifact filename."""
    safe_name = _UNSAFE_ARTIFACT_FILENAME_CHARS.sub("_", str(raw_name or fallback))
    safe_name = safe_name.replace("..", "_").strip(". ")
    safe_name = _REPEATED_UNDERSCORES.sub("_", safe_name)
    return safe_name or fallback


class EditHistoryEntry(BaseModel):
    """An entry in the edit history of a research result."""

    author: str = Field(..., description="Author or contributor who made this edit")
    date: datetime = Field(default_factory=datetime.now, description="When the edit was made")
    summary: str = Field(..., description="Summary of what was changed")


class QueryMetadata(BaseModel):
    """Metadata about the research query, similar to publication metadata."""

    author: Optional[str] = Field(default=None, description="Primary author of the research")
    contributors: List[str] = Field(default_factory=list, description="List of contributors")


class ResearchArtifact(BaseModel):
    """A non-text artifact produced alongside a research report."""

    filename: str = Field(..., description="Artifact filename")
    content_base64: str = Field(..., description="Base64-encoded artifact content")
    media_type: Optional[str] = Field(default=None, description="Artifact MIME/media type")
    path: Optional[str] = Field(default=None, description="Relative path used in formatted markdown")
    source: Optional[str] = Field(default=None, description="Provider-specific artifact source")
    data_storage_id: Optional[str] = Field(default=None, description="Provider data storage identifier")
    description: Optional[str] = Field(default=None, description="Human-readable artifact description")

    @field_validator("filename")
    @classmethod
    def sanitize_filename(cls, value: str) -> str:
        """Sanitize artifact filenames before they reach filesystem output."""
        return sanitize_artifact_filename(value)

    @model_validator(mode="after")
    def validate_artifact_size(self) -> "ResearchArtifact":
        """Reject artifacts that are too large to safely hold in memory."""
        estimated_size = len(self.content_base64) * 3 // 4
        if estimated_size > MAX_ARTIFACT_BYTES:
            raise ValueError(
                f"Artifact too large: estimated {estimated_size} bytes exceeds "
                f"{MAX_ARTIFACT_BYTES} byte limit"
            )
        return self

    @property
    def is_image(self) -> bool:
        """Return whether this artifact can be embedded as a Markdown image."""
        return (self.media_type or "").startswith("image/")


class ResearchResult(BaseModel):
    """Result from a deep research query."""

    markdown: str = Field(..., description="Research report in markdown format")
    citations: List[str] = Field(default_factory=list, description="List of citations/references")
    artifacts: List[ResearchArtifact] = Field(default_factory=list, description="Non-text artifacts produced with the report")
    provider: str = Field(..., description="Name of the research provider used")
    cached: bool = Field(default=False, description="Whether result was retrieved from cache")
    query: str = Field(..., description="Original query that generated this result")

    # Publication-style metadata
    title: Optional[str] = Field(default=None, description="Title for the research report")
    abstract: Optional[str] = Field(default=None, description="Abstract or summary of the research")
    keywords: List[str] = Field(default_factory=list, description="Keywords or tags for the research")
    query_metadata: Optional[QueryMetadata] = Field(default=None, description="Author and contributor metadata")
    edit_history: List[EditHistoryEntry] = Field(default_factory=list, description="History of edits to this research")

    # Timing information
    start_time: Optional[datetime] = Field(default=None, description="When research started")
    end_time: Optional[datetime] = Field(default=None, description="When research completed")
    duration_seconds: Optional[float] = Field(default=None, description="Duration in seconds")

    # Template information
    template_variables: Optional[Dict[str, Any]] = Field(default=None, description="Template variables used")
    template_file: Optional[str] = Field(default=None, description="Template file used")

    # Provider configuration
    model: Optional[str] = Field(default=None, description="Model used by provider")
    provider_config: Optional[Dict[str, Any]] = Field(default=None, description="Provider configuration")
    run_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Provider-reported provenance about the actual run (e.g. model(s) used, "
            "token/cost usage, number of turns). Distinct from the requested "
            "configuration in provider_config."
        ),
    )


class ProviderConfig(BaseModel):
    """Configuration for a research provider."""

    name: str = Field(..., description="Provider name")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for API endpoint (e.g., proxy or OpenAI-compatible service)")
    enabled: bool = Field(default=True, description="Whether provider is enabled")
    timeout: Optional[int] = Field(default=None, description="Request timeout in seconds (provider-specific default if not set)")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class CacheConfig(BaseModel):
    """Configuration for caching."""

    enabled: bool = Field(default=True, description="Whether caching is enabled")
    directory: Optional[str] = Field(default=None, description="Cache directory path (defaults to ~/.deep_research_cache)")


class ProviderHealth(BaseModel):
    """Result of asking a provider whether it can actually take work.

    Distinct from configuration: a provider with an API key set is *configured*,
    which is not the same as reachable, credentialed, or in credit.

    >>> ProviderHealth(provider="falcon", configured=True, reachable=False,
    ...                detail="402 no credits").summary()
    'falcon: UNREACHABLE - 402 no credits'
    """

    provider: str = Field(description="Provider name")
    configured: bool = Field(description="Whether credentials are present and the provider is enabled")
    reachable: Optional[bool] = Field(
        default=None,
        description="Whether a live probe succeeded; None when no probe was performed",
    )
    detail: Optional[str] = Field(default=None, description="Explanation of the outcome")

    def summary(self) -> str:
        """Render a single-line status suitable for CLI output.

        Returns:
            Human-readable status line for this provider.

        >>> ProviderHealth(provider="openai", configured=True, reachable=True).summary()
        'openai: OK'
        >>> ProviderHealth(provider="asta", configured=False, detail="no API key").summary()
        'asta: NOT CONFIGURED - no API key'
        """
        if not self.configured:
            status = "NOT CONFIGURED"
        elif self.reachable is None:
            status = "UNKNOWN (no probe available)"
        elif self.reachable:
            status = "OK"
        else:
            status = "UNREACHABLE"
        suffix = f" - {self.detail}" if self.detail else ""
        return f"{self.provider}: {status}{suffix}"
