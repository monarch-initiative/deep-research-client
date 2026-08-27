"""Pydantic models for deep research client."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import re

from pydantic import BaseModel, Field, field_validator, model_validator

from .exceptions import MAX_DETAIL_CHARS, ProviderError, truncate_detail


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


class ProviderAttempt(BaseModel):
    """One provider asked to do a research run, and how it answered.

    A fallback changes who produced a report, which is exactly the kind of
    thing a curator must be able to see afterwards rather than infer. One of
    these is recorded per provider tried, in the order they were tried, so the
    trail reads as a sequence rather than a single verdict.

    >>> ProviderAttempt(provider="openai", succeeded=True).summary()
    'openai: produced the report'
    """

    provider: str = Field(..., description="Name of the provider that was tried")
    succeeded: bool = Field(..., description="Whether this provider produced the report")
    error_type: Optional[str] = Field(
        default=None, description="Class name of the failure, when this provider failed"
    )
    reason: Optional[str] = Field(
        default=None, description="What the provider said, and what it means"
    )
    retryable: Optional[bool] = Field(
        default=None,
        description=(
            "Whether calling this same provider again could have helped; None "
            "when the failure was not one we classify"
        ),
    )

    @classmethod
    def from_exception(cls, provider: str, exc: BaseException) -> "ProviderAttempt":
        """Record a failed attempt from the exception that ended it.

        Args:
            provider: Name of the provider that failed.
            exc: The exception it raised.

        Returns:
            A failed attempt carrying whatever the failure could tell us.

        >>> from .exceptions import ProviderBillingError
        >>> attempt = ProviderAttempt.from_exception(
        ...     "falcon", ProviderBillingError("falcon", "no credits", 402))
        >>> attempt.error_type, attempt.retryable
        ('ProviderBillingError', False)
        >>> attempt.reason
        '402 no credits -- the account is out of credits'

        An unclassified failure records what it can and leaves ``retryable``
        unset rather than guessing:

        >>> plain = ProviderAttempt.from_exception("mock", RuntimeError("boom"))
        >>> plain.error_type, plain.reason, plain.retryable
        ('RuntimeError', 'boom', None)
        """
        if isinstance(exc, ProviderError):
            return cls(
                provider=provider,
                succeeded=False,
                error_type=type(exc).__name__,
                reason=exc.diagnosis,
                retryable=exc.retryable,
            )
        return cls(
            provider=provider,
            succeeded=False,
            error_type=type(exc).__name__,
            reason=truncate_detail(str(exc), MAX_DETAIL_CHARS),
        )

    def summary(self) -> str:
        """Render this attempt as a single human-readable line.

        Returns:
            One line naming the provider and what happened.

        >>> from .exceptions import ProviderQuotaError
        >>> print(ProviderAttempt.from_exception(
        ...     "claude_code", ProviderQuotaError("claude_code", "limit reached")).summary())
        claude_code: limit reached -- the plan's usage limit is spent
        """
        if self.succeeded:
            return f"{self.provider}: produced the report"
        return f"{self.provider}: {self.reason or self.error_type or 'failed'}"


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

    # Which provider actually did the work. ``provider`` above already names
    # the one that produced this report; these say what was *asked for* and
    # what was tried on the way, so a report produced by a fallback can never
    # read as one produced by the provider the caller named.
    requested_provider: Optional[str] = Field(
        default=None,
        description=(
            "Provider the caller asked for, when they named one. Differs from "
            "provider only when a fallback was taken; None when the caller let "
            "the client choose."
        ),
    )
    provider_attempts: List[ProviderAttempt] = Field(
        default_factory=list,
        description=(
            "Providers tried for this run, in order, and why each failed. "
            "Describes this run only, so it is never read back from cache."
        ),
    )

    @property
    def fell_back(self) -> bool:
        """Report whether a provider other than the first choice produced this.

        Returns:
            True when at least one provider was tried and failed first.

        >>> ResearchResult(markdown="x", provider="openai", query="q").fell_back
        False
        >>> ResearchResult(markdown="x", provider="openai", query="q", provider_attempts=[
        ...     ProviderAttempt(provider="falcon", succeeded=False, reason="402"),
        ...     ProviderAttempt(provider="openai", succeeded=True),
        ... ]).fell_back
        True
        """
        return any(not attempt.succeeded for attempt in self.provider_attempts)

    def fallback_frontmatter(self) -> Dict[str, Any]:
        """Render the fallback provenance for a report's frontmatter.

        Lives here rather than in a formatter because there is more than one
        formatter, and a fallback that only some reports admit to would be
        worse than one none of them mention.

        Returns:
            Frontmatter keys describing the fallback, or an empty dict when no
            fallback happened -- the presence of these keys is the finding.

        >>> ResearchResult(markdown="x", provider="openai", query="q").fallback_frontmatter()
        {}
        >>> result = ResearchResult(
        ...     markdown="x", provider="openai", query="q", requested_provider="falcon",
        ...     provider_attempts=[
        ...         ProviderAttempt(provider="falcon", succeeded=False, reason="402"),
        ...         ProviderAttempt(provider="openai", succeeded=True),
        ...     ])
        >>> result.fallback_frontmatter()["fell_back"]
        True
        >>> result.fallback_frontmatter()["requested_provider"]
        'falcon'
        >>> result.fallback_frontmatter()["provider_attempts"]
        [{'provider': 'falcon', 'succeeded': False, 'reason': '402'}, {'provider': 'openai', 'succeeded': True}]
        """
        if not self.fell_back:
            return {}
        metadata: Dict[str, Any] = {"fell_back": True}
        if self.requested_provider:
            metadata["requested_provider"] = self.requested_provider
        metadata["provider_attempts"] = [
            attempt.model_dump(exclude_none=True) for attempt in self.provider_attempts
        ]
        return metadata


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
        description=(
            "False when the provider cannot take work -- either a probe failed or "
            "it is not configured; None when no probe was possible"
        ),
    )
    detail: Optional[str] = Field(default=None, description="Explanation of the outcome")

    @field_validator("detail")
    @classmethod
    def _cap_detail(cls, value: Optional[str]) -> Optional[str]:
        """Keep a raw stack trace or error page from becoming the whole report.

        This is the guard for *unclassified* text -- a bare ``str(e)`` or raw
        stderr. A composed ``diagnosis`` is already within the cap by
        construction, so this never truncates the remedy off one.

        Args:
            value: The proposed detail text.

        Returns:
            The detail, truncated to a readable length.
        """
        return truncate_detail(value, MAX_DETAIL_CHARS) if value else value

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
