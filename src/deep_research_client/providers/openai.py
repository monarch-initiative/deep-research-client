"""OpenAI Deep Research provider."""

from typing import List, Optional, Any, Dict, cast

import httpx
from openai import OpenAI
from openai.types.responses import WebSearchPreviewToolParam

from . import ResearchProvider
from ..models import ResearchResult, ProviderConfig
from ..provider_params import OpenAIParams
from ..model_cards import ProviderModelCards, create_openai_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT


class OpenAIProvider(ResearchProvider):
    """Provider for OpenAI Deep Research API."""

    def __init__(self, config: ProviderConfig, params: Optional[OpenAIParams] = None):
        """Initialize OpenAI provider."""
        self.params = params or OpenAIParams()
        super().__init__(config, self.params.model)

    def get_default_model(self) -> str:
        """Get default OpenAI model."""
        return "o3-deep-research-2025-06-26"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for OpenAI provider."""
        return create_openai_model_cards()

    async def research(self, query: str) -> ResearchResult:
        """Perform research using OpenAI Deep Research API."""
        if not self.is_available():
            raise ValueError(f"OpenAI provider not available (API key: {bool(self.config.api_key)})")

        # Create HTTP client with timeout
        http_client = httpx.Client(
            timeout=httpx.Timeout(
                connect=30.0,
                read=self.config.timeout,
                write=30.0,
                pool=30.0,
            )
        )

        client = OpenAI(api_key=self.config.api_key, http_client=http_client)

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

            response = client.responses.create(
                model=self.model,
                input=input_messages,
                tools=[cast(WebSearchPreviewToolParam, web_search_tool)],
            )

            # Extract the final report
            final_output = response.output[-1]
            markdown_content = self._extract_text_content(final_output)

            # Extract citations
            citations = self._extract_citations(final_output)

            return ResearchResult(
                markdown=markdown_content,
                citations=citations,
                provider=self.name,
                query=query
            )

        finally:
            http_client.close()

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