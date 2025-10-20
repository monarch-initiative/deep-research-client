"""Pydantic models for deep research client."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ResearchResult(BaseModel):
    """Result from a deep research query."""

    markdown: str = Field(..., description="Research report in markdown format")
    citations: List[str] = Field(default_factory=list, description="List of citations/references")
    provider: str = Field(..., description="Name of the research provider used")
    cached: bool = Field(default=False, description="Whether result was retrieved from cache")
    query: str = Field(..., description="Original query that generated this result")

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
    enabled: bool = Field(default=True, description="Whether provider is enabled")
    timeout: int = Field(default=600, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")


class CacheConfig(BaseModel):
    """Configuration for caching."""

    enabled: bool = Field(default=True, description="Whether caching is enabled")
    directory: Optional[str] = Field(default=None, description="Cache directory path (defaults to ~/.deep_research_cache)")