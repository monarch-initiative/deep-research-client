"""FutureHouse Falcon provider."""

import re
from typing import List, Optional

from futurehouse_client import FutureHouseClient, JobNames

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
        """Extract text content from Falcon response."""
        report_text = None

        if isinstance(response, list) and len(response) > 0:
            # Get the first (and usually only) response
            task_response = response[0]

            # Extract formatted_answer if available (preferred)
            if hasattr(task_response, 'formatted_answer') and task_response.formatted_answer:
                report_text = task_response.formatted_answer
            elif hasattr(task_response, 'answer') and task_response.answer:
                report_text = task_response.answer
            else:
                # Fallback to string representation
                report_text = str(task_response)

        elif hasattr(response, 'formatted_answer'):
            report_text = response.formatted_answer
        elif hasattr(response, 'answer'):
            report_text = response.answer
        elif hasattr(response, 'result'):
            report_text = response.result
        elif hasattr(response, 'content'):
            report_text = response.content
        elif isinstance(response, dict):
            # Try various possible keys
            if 'formatted_answer' in response:
                report_text = response['formatted_answer']
            elif 'answer' in response:
                report_text = response['answer']
            elif 'result' in response:
                report_text = response['result']
            elif 'content' in response:
                report_text = response['content']
            else:
                report_text = str(response)
        elif isinstance(response, str):
            report_text = response
        else:
            report_text = str(response)

        # Validate that we got meaningful content
        if not report_text or len(report_text.strip()) < 100:
            report_text = str(response)

        return report_text or ""

    def _extract_citations(self, response, report_text: str) -> List[str]:
        """Extract citations from Falcon response."""
        citations = []

        # Try to extract citations from the response object
        task_response = None
        if isinstance(response, list) and len(response) > 0:
            task_response = response[0]
        else:
            task_response = response

        # Try to extract citations from the response
        if task_response and hasattr(task_response, 'sources'):
            citations = task_response.sources
        elif task_response and hasattr(task_response, 'citations'):
            citations = task_response.citations
        elif task_response and hasattr(task_response, 'references'):
            citations = task_response.references
        elif isinstance(task_response, dict):
            if 'sources' in task_response:
                citations = task_response['sources']
            elif 'citations' in task_response:
                citations = task_response['citations']
            elif 'references' in task_response:
                citations = task_response['references']

        # Extract inline citations from the text using regex patterns
        # Look for Falcon-style citations like (elsen2005ppsramultifaceted pages 6-8)
        falcon_citations = re.findall(r'\(([^)]+pages?[^)]+)\)', report_text)

        # Look for standard patterns like [PMID:12345678], [1], etc.
        standard_refs = re.findall(r'\[([^\]]+)\]', report_text)

        # Combine all citation sources
        all_inline_citations = falcon_citations + standard_refs

        if all_inline_citations and not citations:
            citations = all_inline_citations
        elif all_inline_citations and citations:
            # Merge both sources
            if isinstance(citations, list):
                citations.extend(all_inline_citations)

        # Convert to list of strings and remove duplicates
        if citations:
            # Remove duplicates while preserving order
            seen = set()
            unique_citations = []
            for citation in citations:
                citation_str = str(citation)
                if citation_str not in seen:
                    seen.add(citation_str)
                    unique_citations.append(citation_str)
            return unique_citations

        return []