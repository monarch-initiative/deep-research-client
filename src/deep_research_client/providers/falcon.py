"""FutureHouse Falcon provider."""

import re
from typing import List, Optional

from futurehouse_client import FutureHouseClient, JobNames
from futurehouse_client.models.app import PQATaskResponse

from . import ResearchProvider
from ..models import ResearchResult, ProviderConfig
from ..provider_params import FalconParams
from ..model_cards import ProviderModelCards, create_falcon_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT


class FalconProvider(ResearchProvider):
    """Provider for FutureHouse Falcon API."""

    def __init__(self, config: ProviderConfig, params: Optional[FalconParams] = None):
        """Initialize Falcon provider."""
        self.params = params or FalconParams()
        super().__init__(config, self.params.model)

    def get_default_model(self) -> str:
        """Get default Falcon model."""
        return "FutureHouse Falcon API"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for Falcon provider."""
        return create_falcon_model_cards()

    async def research(self, query: str) -> ResearchResult:
        """Perform research using FutureHouse Falcon API."""
        if not self.is_available():
            raise ValueError(f"Falcon provider not available (API key: {bool(self.config.api_key)})")

        client = FutureHouseClient(api_key=self.config.api_key)

        # Use custom system prompt or default
        system_prompt = self.params.system_prompt or DEFAULT_RESEARCH_SYSTEM_PROMPT

        # Falcon combines system prompt and user query
        full_query = f"{system_prompt}\n\n{query}"

        # Use Falcon API for deep research
        task_data = {
            "name": JobNames.FALCON,
            "query": full_query
        }

        response = client.run_tasks_until_done(task_data)

        # Extract the report text and citations
        markdown_content = self._extract_text_content(response)
        citations = self._extract_citations(response, markdown_content)

        return ResearchResult(
            markdown=markdown_content,
            citations=citations,
            provider=self.name,
            query=query
        )

    def _extract_text_content(self, response) -> str:
        """Extract text content from Falcon response.

        Falcon returns PQATaskResponse objects with 'formatted_answer' (preferred)
        and 'answer' fields.
        """
        if not isinstance(response, list) or len(response) == 0:
            raise ValueError(f"Unexpected Falcon response structure: {type(response)}")

        task_response = response[0]

        # Falcon always returns PQATaskResponse - fail fast if it doesn't
        if not isinstance(task_response, PQATaskResponse):
            raise ValueError(
                f"Expected PQATaskResponse, got {type(task_response)}. "
                f"This indicates an API change in futurehouse-client."
            )

        # Prefer formatted_answer as it includes references
        if task_response.formatted_answer:
            return task_response.formatted_answer
        elif task_response.answer:
            return task_response.answer
        else:
            raise ValueError(
                f"PQATaskResponse has no answer. Status: {task_response.status}, "
                f"has_successful_answer: {task_response.has_successful_answer}"
            )

    def _extract_citations(self, response, report_text: str) -> List[str]:
        """Extract citations from Falcon response.

        Citations are embedded in the formatted_answer text using various patterns.
        """
        # Extract inline citations from the formatted answer text
        # Look for PaperQA-style citations like (Author2020Title pages 6-8)
        paperqa_citations = re.findall(r'\(([a-z]+\d{4}[a-z\s]+pages?\s+[\d\-]+)\)', report_text, re.IGNORECASE)

        # Look for standard reference patterns like [PMID:12345678], [DOI:10.xxx], [1]
        standard_refs = re.findall(r'\[([^\]]+)\]', report_text)

        # Look for URL citations
        url_citations = re.findall(r'https?://[^\s\)]+', report_text)

        # Combine all citation sources
        all_citations = paperqa_citations + standard_refs + url_citations

        # Remove duplicates while preserving order
        if all_citations:
            seen = set()
            unique_citations = []
            for citation in all_citations:
                citation_str = str(citation).strip()
                if citation_str and citation_str not in seen:
                    seen.add(citation_str)
                    unique_citations.append(citation_str)
            return unique_citations

        return []