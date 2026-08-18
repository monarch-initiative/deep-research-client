"""OpenAI Deep Research provider."""

import asyncio
import logging
from typing import List, Optional, Any, Dict, cast

import httpx
from openai import OpenAI
from openai.types.responses import WebSearchPreviewToolParam

from . import ResearchProvider
from ..exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderError,
    ProviderNotConfiguredError,
    classify_exception,
)
from ..models import ResearchResult, ProviderConfig, ProviderHealth
from ..provider_params import OpenAIParams
from ..model_cards import ProviderModelCards, create_openai_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# OpenAI reports a spent quota as 429 -- the same status it uses for ordinary
# throttling -- and distinguishes the two only by the body's error code. Reading
# the status alone would mark a spent quota retryable and loop on it forever.
_ERROR_CODE_TYPES: dict[str, type[ProviderError]] = {
    "insufficient_quota": ProviderBillingError,
    "billing_hard_limit_reached": ProviderBillingError,
    "invalid_api_key": ProviderAuthError,
    "account_deactivated": ProviderAuthError,
}


def _classify_openai_error(provider: str, exc: BaseException) -> Optional[ProviderError]:
    """Classify an OpenAI SDK exception, preferring its error code to its status.

    Args:
        provider: Name of the provider that raised.
        exc: The exception raised by the OpenAI SDK.

    Returns:
        A typed error to raise in its place, or None to leave the original alone.
    """
    code = getattr(exc, "code", None)
    error_class = _ERROR_CODE_TYPES.get(code) if isinstance(code, str) else None
    if error_class is not None:
        status_code = getattr(exc, "status_code", None)
        return error_class(provider, getattr(exc, "message", None) or str(exc), status_code)
    # No recognised code: fall back to reading the HTTP status.
    return classify_exception(provider, exc)


class OpenAIProvider(ResearchProvider):
    """Provider for OpenAI Deep Research API."""

    credential_label = "OpenAI"
    credential_env_var = "OPENAI_API_KEY"

    def __init__(self, config: ProviderConfig, params: Optional[OpenAIParams] = None):
        """Initialize OpenAI provider."""
        self.params = params or OpenAIParams()
        super().__init__(config, self.params.model)

        # Log initialization details
        logger.debug(f"Initializing OpenAI provider with model: {self.model}")
        if config.base_url:
            logger.info(f"OpenAI provider configured with custom endpoint: {config.base_url}")
        else:
            logger.debug("OpenAI provider using default endpoint (api.openai.com)")

        # Log API key presence (not the actual key!)
        if config.api_key:
            # Show first 8 chars to help identify which key is being used
            key_preview = config.api_key[:8] + "..." if len(config.api_key) > 8 else "***"
            logger.debug(f"API key configured (starts with: {key_preview})")

    def get_default_model(self) -> str:
        """Get default OpenAI model."""
        return "o3-deep-research-2025-06-26"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for OpenAI provider."""
        return create_openai_model_cards()

    async def check_health(self) -> ProviderHealth:
        """Probe OpenAI by listing models, which is authenticated but not billed.

        Like Edison, this proves the key is accepted and nothing more: listing
        models does not consume quota, so a key whose quota is spent still
        passes. OpenAI exposes no balance endpoint to read instead -- the spent
        quota only announces itself as a ``429`` on the run.

        Returns:
            Health record for this provider
        """
        if not self.is_available():
            return ProviderHealth(
                provider=self.name,
                configured=False,
                reachable=False,
                detail=self.unavailable_reason(),
            )

        client_kwargs: Dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        client = None
        try:
            # Construction can fail too (a malformed base_url, an SDK-level
            # problem), and this method promises a record either way.
            client = OpenAI(**client_kwargs)
            await asyncio.to_thread(client.models.list)
        except Exception as e:
            classified = _classify_openai_error(self.name, e)
            detail = classified.diagnosis if classified else str(e)
            logger.debug("OpenAI health probe failed:", exc_info=True)
            return ProviderHealth(
                provider=self.name, configured=True, reachable=False, detail=detail
            )
        finally:
            if client is not None:
                client.close()
        return ProviderHealth(
            provider=self.name,
            configured=True,
            reachable=True,
            detail="credentials accepted (quota not readable until a run)",
        )

    async def research(self, query: str) -> ResearchResult:
        """Perform research using OpenAI Deep Research API."""
        logger.info(f"Starting OpenAI research query (model: {self.model})")
        logger.debug(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        if not self.is_available():
            raise ProviderNotConfiguredError(self.name, self.unavailable_reason())

        # Create HTTP client with timeout
        http_client = httpx.Client(
            timeout=httpx.Timeout(
                connect=30.0,
                read=self.config.timeout or 600,
                write=30.0,
                pool=30.0,
            )
        )
        logger.debug(f"HTTP client configured with timeout: {self.config.timeout or 600}s")

        # Build OpenAI client kwargs
        client_kwargs: Dict[str, Any] = {
            "api_key": self.config.api_key,
            "http_client": http_client
        }

        # Add base_url if configured (for proxies/OpenAI-compatible endpoints)
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
            logger.info(f"Using custom endpoint: {self.config.base_url}")

        client = OpenAI(**client_kwargs)

        # Use custom system prompt or default
        system_prompt = self.params.system_prompt or DEFAULT_RESEARCH_SYSTEM_PROMPT

        # OpenAI uses developer role for system prompts in reasoning models
        input_messages: List[Any] = []
        if system_prompt:
            input_messages.append({
                "role": "developer",
                "content": [{"type": "input_text", "text": system_prompt}]
            })
        input_messages.append({
            "role": "user",
            "content": [{"type": "input_text", "text": query}]
        })

        try:
            # Configure web search tool with optional domain filtering
            web_search_tool: Dict[str, Any] = {"type": "web_search_preview"}
            if self.params.allowed_domains:
                # Add domain filtering if specified
                web_search_tool["filters"] = {
                    "allowed_domains": self.params.allowed_domains
                }
                logger.info(f"Domain filtering enabled: {len(self.params.allowed_domains)} domains")
                logger.debug(f"Allowed domains: {', '.join(self.params.allowed_domains)}")

            logger.debug(f"Making API request to OpenAI (model: {self.model})")
            response = client.responses.create(
                model=self.model,
                input=input_messages,
                tools=[cast(WebSearchPreviewToolParam, web_search_tool)],
            )
            logger.info("OpenAI API request completed successfully")

            # Extract the final report
            final_output = response.output[-1]
            markdown_content = self._extract_text_content(final_output)
            logger.debug(f"Extracted markdown content: {len(markdown_content)} characters")

            # Extract citations
            citations = self._extract_citations(final_output)
            logger.info(f"Extracted {len(citations)} citations from response")
            if citations:
                logger.debug(f"Citation sources: {citations[:3]}{'...' if len(citations) > 3 else ''}")

            return ResearchResult(
                markdown=markdown_content,
                citations=citations,
                provider=self.name,
                query=query
            )

        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            logger.debug("Error details:", exc_info=True)
            classified = _classify_openai_error(self.name, e)
            if classified is not None:
                logger.error(classified.actionable_message())
                raise classified from e
            raise
        finally:
            http_client.close()
            logger.debug("HTTP client closed")

    def _extract_text_content(self, output) -> str:
        """Extract text content from OpenAI response output."""
        if hasattr(output, "content") and output.content:
            content = output.content
            if isinstance(content, list) and len(content) > 0:
                first_content = content[0]
                if hasattr(first_content, "text"):
                    return first_content.text
                else:
                    return str(first_content)
            else:
                return str(content)
        else:
            return str(output)

    def _extract_citations(self, output) -> List[str]:
        """Extract citations from OpenAI response."""
        citations = []

        # Try to get annotations from content
        if hasattr(output, "content") and output.content:
            content = output.content
            if isinstance(content, list) and len(content) > 0:
                first_content = content[0]
                if hasattr(first_content, "annotations"):
                    annotations = first_content.annotations
                    if annotations:
                        citations.extend([str(annotation) for annotation in annotations])

        return citations