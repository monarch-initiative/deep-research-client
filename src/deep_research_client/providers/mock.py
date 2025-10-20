"""Mock provider for testing and development."""

import asyncio
from datetime import datetime
from typing import List, Optional

from . import ResearchProvider
from ..models import ResearchResult, ProviderConfig
from ..provider_params import MockParams


class MockProvider(ResearchProvider):
    """Mock provider that returns fake responses for testing."""

    def __init__(self, config: ProviderConfig, params: Optional[MockParams] = None):
        """Initialize Mock provider.

        Args:
            config: Provider configuration (API key not required for mock)
            params: Mock-specific parameters
        """
        self.params = params or MockParams()
        super().__init__(config, self.params.model)

    def get_default_model(self) -> str:
        """Get default Mock model."""
        return "mock-model-v1"

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True

    async def research(self, query: str) -> ResearchResult:
        """Perform mock research with fake response.

        Args:
            query: The research question

        Returns:
            ResearchResult with mock content and citations

        Raises:
            ValueError: If include_error parameter is True
        """
        # Simulate API delay
        await asyncio.sleep(self.params.response_delay)

        # Simulate error if requested
        if self.params.include_error:
            raise ValueError("Mock error: This is a simulated API error for testing")

        # Generate mock content
        if self.params.custom_response:
            markdown_content = self.params.custom_response
        else:
            markdown_content = self._generate_mock_response(query)

        # Generate mock citations
        citations = self._generate_mock_citations(query)

        return ResearchResult(
            markdown=markdown_content,
            citations=citations,
            provider=self.name,
            query=query,
            model=self.model,
            start_time=datetime.now(),
            end_time=datetime.now()
        )

    def _generate_mock_response(self, query: str) -> str:
        """Generate mock markdown response based on query and parameters."""
        base_response = f"# Mock Research Response\n\nThe user asked: **{query}**\n\n"

        if self.params.response_length == "short":
            content = (
                "This is a short mock response for testing purposes. "
                "The mock provider has simulated a research query and is returning "
                "this fake content to verify the system works correctly."
            )
        elif self.params.response_length == "long":
            content = (
                "This is an extended mock response designed to test how the system "
                "handles longer content. The mock provider is simulating a comprehensive "
                "research response that might include multiple sections, detailed analysis, "
                "and extensive information.\n\n"
                "## Background Information\n\n"
                "Mock providers are essential for testing research systems without making "
                "actual API calls. They allow developers to verify functionality, test "
                "error handling, and ensure proper data flow through the application.\n\n"
                "## Key Findings\n\n"
                "1. Mock responses enable rapid development and testing\n"
                "2. Parameter validation can be tested without external dependencies\n"
                "3. Caching behavior can be verified with consistent mock data\n\n"
                "## Methodology\n\n"
                "The mock provider generates responses based on configurable parameters "
                "including response length, delay simulation, and error injection capabilities. "
                "This approach provides comprehensive testing coverage while maintaining "
                "deterministic behavior for reliable testing scenarios."
            )
        else:  # medium
            content = (
                "This is a medium-length mock response for testing the research system. "
                "The mock provider generates fake but realistic-looking content to simulate "
                "what a real research API might return. This includes structured information, "
                "proper formatting, and citations to verify all components work correctly.\n\n"
                "## Mock Provider Features\n\n"
                "- Configurable response length and delay\n"
                "- Error simulation capabilities\n"
                "- Custom response text support\n"
                "- Proper citation generation"
            )

        return base_response + content

    def _generate_mock_citations(self, query: str) -> List[str]:
        """Generate mock citations for testing."""
        # Always include one basic citation as requested
        citations = [
            f"Mock Citation: Research on '{query}' - https://example.com/mock-citation-1"
        ]

        # Add more citations based on response length
        if self.params.response_length == "medium":
            citations.append("Mock Journal Article - https://example.com/mock-journal-2024")
        elif self.params.response_length == "long":
            citations.extend([
                "Mock Journal Article - https://example.com/mock-journal-2024",
                "Mock Research Database - https://example.com/mock-database",
                "Mock Academic Paper - https://example.com/mock-paper-doi"
            ])

        return citations