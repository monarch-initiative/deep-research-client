"""Perplexity AI provider."""

import logging
import re
from typing import List, Optional, Dict, Any

import httpx

from . import ResearchProvider
from ..models import ResearchResult, ProviderConfig
from ..provider_params import PerplexityParams
from ..model_cards import ProviderModelCards, create_perplexity_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class PerplexityProvider(ResearchProvider):
    """Provider for Perplexity AI API."""

    def __init__(self, config: ProviderConfig, params: Optional[PerplexityParams] = None):
        """Initialize Perplexity provider.

        Args:
            config: Provider configuration
            params: Perplexity-specific parameters
        """
        self.params = params or PerplexityParams()
        super().__init__(config, self.params.model)
        self.api_url = "https://api.perplexity.ai/chat/completions"

        logger.debug(f"Initializing Perplexity provider with model: {self.model}")
        if config.api_key:
            key_preview = config.api_key[:8] + "..." if len(config.api_key) > 8 else "***"
            logger.debug(f"API key configured (starts with: {key_preview})")

    def get_default_model(self) -> str:
        """Get default Perplexity model."""
        return "sonar-deep-research"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for Perplexity provider."""
        return create_perplexity_model_cards()

    async def research(self, query: str) -> ResearchResult:
        """Perform research using Perplexity AI API."""
        logger.info(f"Starting Perplexity research query (model: {self.model})")
        logger.debug(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        if not self.is_available():
            raise ValueError(f"Perplexity provider not available (API key: {bool(self.config.api_key)})")

        # Use custom system prompt or default
        system_prompt = self.params.system_prompt or DEFAULT_RESEARCH_SYSTEM_PROMPT

        # Prepare request payload with all Perplexity-specific parameters
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": self.params.temperature,
        }

        # Add Perplexity-specific parameters
        if self.params.reasoning_effort != "medium":
            payload["reasoning_effort"] = self.params.reasoning_effort
            logger.debug(f"Reasoning effort: {self.params.reasoning_effort}")

        if self.params.search_recency_filter:
            payload["search_recency_filter"] = self.params.search_recency_filter
            logger.debug(f"Search recency filter: {self.params.search_recency_filter}")

        if self.params.search_domain_filter:
            payload["search_domain_filter"] = self.params.search_domain_filter
            logger.info(f"Domain filtering enabled: {len(self.params.search_domain_filter)} domains")
            logger.debug(f"Domain filter: {', '.join(self.params.search_domain_filter)}")

        # Always pass return_citations since it affects response format
        payload["return_citations"] = self.params.return_citations

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        try:
            logger.debug(f"Making API request to Perplexity (model: {self.model})")
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()

                data = response.json()

            logger.info("Perplexity API request completed successfully")

            # Extract the response content
            if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                markdown_content = data["choices"][0]["message"].get("content", "")
                logger.debug(f"Extracted markdown content: {len(markdown_content)} characters")
            else:
                raise ValueError("No response content received from Perplexity")

            # Extract citations from the response (both from content and metadata)
            citations = self._extract_citations(markdown_content, data)
            logger.info(f"Extracted {len(citations)} citations from response")

            return ResearchResult(
                markdown=markdown_content,
                citations=citations,
                provider=self.name,
                query=query
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Perplexity API HTTP error: {e.response.status_code}")
            logger.debug("Error details:", exc_info=True)
            raise ValueError(f"Perplexity API HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"Perplexity API request failed: {e}")
            logger.debug("Error details:", exc_info=True)
            raise ValueError(f"Perplexity API error: {e}")

    def _extract_citations(self, content: str, response_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Extract citations from Perplexity response content and metadata."""
        citations = []

        # First, try to extract citations from response metadata (if available)
        if response_data and "citations" in response_data:
            metadata_citations = response_data["citations"]
            for citation in metadata_citations:
                if isinstance(citation, dict):
                    # Handle structured citation objects
                    if "url" in citation:
                        title = citation.get("title", "")
                        url = citation["url"]
                        if title:
                            citations.append(f"{title} - {url}")
                        else:
                            citations.append(url)
                elif isinstance(citation, str):
                    citations.append(citation)

        # Also check for citations in message metadata
        if response_data and "choices" in response_data:
            for choice in response_data["choices"]:
                if "message" in choice and "citations" in choice["message"]:
                    for citation in choice["message"]["citations"]:
                        if isinstance(citation, str):
                            citations.append(citation)

        # Extract from content as fallback
        # Look for URLs
        url_pattern = re.compile(r'https?://[^\s\)\]]+')
        urls = url_pattern.findall(content)
        citations.extend(urls)

        # Look for parenthetical citations like (Source: ...)
        source_refs = re.findall(r'\(Source: ([^)]+)\)', content)
        citations.extend(source_refs)

        # Look for "according to" patterns
        according_to_refs = re.findall(r'according to ([^,.]+)', content, re.IGNORECASE)
        citations.extend(according_to_refs)

        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            citation = citation.strip()
            if citation and citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)

        return unique_citations