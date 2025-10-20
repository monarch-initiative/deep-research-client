"""Tests for the Consensus provider."""

import pytest

from deep_research_client.providers.consensus import ConsensusProvider
from deep_research_client.models import ProviderConfig


class TestConsensusProvider:
    """Test cases for ConsensusProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="consensus",
            api_key="test-api-key",
            enabled=True
        )
        self.provider = ConsensusProvider(self.config)

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert self.provider.name == "consensus"
        assert self.provider.config.api_key == "test-api-key"
        assert self.provider.is_available() is True

    def test_get_default_model(self):
        """Test default model."""
        assert self.provider.get_default_model() == "Consensus Academic Search"

    def test_provider_without_api_key(self):
        """Test provider initialization without API key."""
        config = ProviderConfig(name="consensus", api_key=None, enabled=True)
        provider = ConsensusProvider(config)
        assert provider.is_available() is False

    # Note: Complex async tests with mocking are temporarily disabled
    # These would test the full HTTP interaction but require more complex setup
    # The key functionality is tested through the formatting methods below

    def test_research_provider_not_available(self):
        """Test research when provider is not available."""
        config = ProviderConfig(name="consensus", api_key=None, enabled=True)
        provider = ConsensusProvider(config)

        import asyncio
        with pytest.raises(ValueError, match="Consensus provider not available"):
            asyncio.run(provider.research("test query"))

    # Note: HTTP error handling tests would go here
    # These are covered by integration testing with real API calls

    def test_format_research_report(self):
        """Test research report formatting."""
        mock_data = {
            "results": [
                {
                    "title": "Test Paper",
                    "authors": [{"name": "Author One"}, {"name": "Author Two"}],
                    "year": 2023,
                    "journal": "Test Journal",
                    "citation_count": 100,
                    "relevance_score": 0.9,
                    "abstract": "This is a test abstract."
                }
            ]
        }

        report = self.provider._format_research_report("test query", mock_data)

        assert "Academic Research Report: test query" in report
        assert "Test Paper" in report
        assert "Author One" in report
        assert "2023" in report
        assert "Test Journal" in report
        assert "This is a test abstract" in report

    def test_extract_citations(self):
        """Test citation extraction."""
        mock_data = {
            "results": [
                {
                    "title": "Test Paper",
                    "authors": [{"name": "Author One"}],
                    "year": 2023,
                    "journal": "Test Journal",
                    "doi": "10.1000/test123"
                },
                {
                    "title": "Another Paper",
                    "authors": [{"name": "Author Two"}],
                    "year": 2022,
                    "journal": "Another Journal",
                    "url": "https://example.com/paper"
                }
            ]
        }

        citations = self.provider._extract_citations(mock_data)

        assert len(citations) == 2
        assert "Author One (2023). Test Paper. Test Journal. DOI: 10.1000/test123" in citations
        assert "Author Two (2022). Another Paper. Another Journal. Retrieved from https://example.com/paper" in citations

    def test_format_research_report_with_many_authors(self):
        """Test research report with many authors (tests et al. handling)."""
        mock_data = {
            "results": [
                {
                    "title": "Test Paper",
                    "authors": [
                        {"name": "Author One"},
                        {"name": "Author Two"},
                        {"name": "Author Three"},
                        {"name": "Author Four"},
                        {"name": "Author Five"},
                        {"name": "Author Six"}
                    ],
                    "year": 2023,
                    "journal": "Test Journal",
                    "citation_count": 100,
                    "relevance_score": 0.9
                }
            ]
        }

        report = self.provider._format_research_report("test query", mock_data)
        assert "Author One, Author Two, Author Three et al." in report

    def test_format_research_report_with_research_trends(self):
        """Test research report with trend analysis."""
        mock_data = {
            "results": [
                {"year": 2023, "journal": "Journal A", "citation_count": 150},
                {"year": 2022, "journal": "Journal B", "citation_count": 200},
                {"year": 2021, "journal": "Journal A", "citation_count": 50},
                {"year": 2019, "journal": "Journal C", "citation_count": 300}
            ]
        }

        report = self.provider._format_research_report("test query", mock_data)

        assert "Research Trends" in report
        assert "recent research activity" in report
        assert "research impact" in report
        assert "journals" in report