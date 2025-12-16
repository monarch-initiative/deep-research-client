"""Pydantic models for deep research client."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class EditHistoryEntry(BaseModel):
    """An entry in the edit history of a research result."""

    author: str = Field(..., description="Author or contributor who made this edit")
    date: datetime = Field(default_factory=datetime.now, description="When the edit was made")
    summary: str = Field(..., description="Summary of what was changed")


class QueryMetadata(BaseModel):
    """Metadata about the research query, similar to publication metadata."""

    author: Optional[str] = Field(default=None, description="Primary author of the research")
    contributors: List[str] = Field(default_factory=list, description="List of contributors")


class ResearchResult(BaseModel):
    """Result from a deep research query."""

    markdown: str = Field(..., description="Research report in markdown format")
    citations: List[str] = Field(default_factory=list, description="List of citations/references")
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