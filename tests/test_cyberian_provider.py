"""Tests for the Cyberian provider."""

import pytest
import tempfile
from pathlib import Path

# Skip all tests in this module if cyberian is not installed
pytest.importorskip("cyberian")

from deep_research_client.providers.cyberian import CyberianProvider
from deep_research_client.models import ProviderConfig


class TestCyberianProvider:
    """Test cases for CyberianProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="cyberian",
            api_key=None,  # Cyberian doesn't need API key
            enabled=True,
            timeout=600
        )

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = CyberianProvider(self.config)
        assert provider.name == "cyberian"
        assert provider.config.api_key is None
        assert provider.agent_type == "claude"  # default
        assert provider.skip_permissions is True  # default

    def test_get_default_model(self):
        """Test default model."""
        provider = CyberianProvider(self.config)
        assert provider.get_default_model() == "deep-research"

    def test_provider_availability_with_cyberian_installed(self):
        """Test provider availability when cyberian is installed."""
        # Since cyberian is actually installed in the test environment,
        # this should return True
        provider = CyberianProvider(self.config)
        result = provider.is_available()
        assert result is True

    def test_provider_availability_without_cyberian(self):
        """Test provider availability when cyberian is not installed."""
        # Disable the provider to test unavailability
        config = ProviderConfig(
            name="cyberian",
            api_key=None,
            enabled=False  # Disabled
        )
        provider = CyberianProvider(config)
        result = provider.is_available()
        assert result is False

    def test_read_report_success(self):
        """Test reading REPORT.md from workdir."""
        provider = CyberianProvider(self.config)

        with tempfile.TemporaryDirectory() as workdir:
            # Create a test REPORT.md
            report_path = Path(workdir) / "REPORT.md"
            report_content = "# Test Report\n\nThis is a test research report."
            report_path.write_text(report_content)

            # Read the report
            result = provider._read_report(workdir)
            assert result == report_content

    def test_read_report_missing(self):
        """Test reading REPORT.md when it doesn't exist."""
        provider = CyberianProvider(self.config)

        with tempfile.TemporaryDirectory() as workdir:
            # Don't create REPORT.md
            with pytest.raises(FileNotFoundError, match="REPORT.md not found"):
                provider._read_report(workdir)

    def test_extract_citations_success(self):
        """Test extracting citations from citations directory."""
        provider = CyberianProvider(self.config)

        with tempfile.TemporaryDirectory() as workdir:
            # Create citations directory with test files
            citations_dir = Path(workdir) / "citations"
            citations_dir.mkdir()

            # Create test citation files
            (citations_dir / "smith-2002-autophagy-abstract.md").write_text("Abstract...")
            (citations_dir / "jones-2003-regulation-fulltext.txt").write_text("Full text...")
            (citations_dir / "doe-2024-review-summary.md").write_text("Summary...")

            # Extract citations
            citations = provider._extract_citations(workdir)

            assert len(citations) == 3
            assert "smith-2002-autophagy-abstract.md" in citations
            assert "jones-2003-regulation-fulltext.txt" in citations
            assert "doe-2024-review-summary.md" in citations

    def test_extract_citations_no_directory(self):
        """Test extracting citations when citations directory doesn't exist."""
        provider = CyberianProvider(self.config)

        with tempfile.TemporaryDirectory() as workdir:
            # Don't create citations directory
            citations = provider._extract_citations(workdir)
            assert citations == []

    def test_extract_citations_empty_directory(self):
        """Test extracting citations from empty directory."""
        provider = CyberianProvider(self.config)

        with tempfile.TemporaryDirectory() as workdir:
            # Create empty citations directory
            citations_dir = Path(workdir) / "citations"
            citations_dir.mkdir()

            citations = provider._extract_citations(workdir)
            assert citations == []

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_research_integration(self):
        """Integration test for full research workflow.

        This test requires:
        - cyberian to be installed
        - agentapi to be running
        - An actual agent (Claude, etc.) available

        Mark as integration test to skip in normal test runs.
        """
        pytest.importorskip("cyberian")

        provider = CyberianProvider(self.config)

        if not provider.is_available():
            pytest.skip("Cyberian not available")

        # Run a simple research query
        result = await provider.research("What is autophagy?")

        # Verify result structure
        assert result.provider == "cyberian"
        assert result.query == "What is autophagy?"
        assert result.markdown  # Should have content
        assert result.model == "deep-research"
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0

    def test_model_cards(self):
        """Test that model cards are available."""
        provider = CyberianProvider(self.config)
        cards = provider.model_cards()

        assert cards is not None
        assert cards.provider_name == "cyberian"
        assert cards.default_model == "Cyberian Deep Research"
        assert len(cards.models) > 0
        assert "Cyberian Deep Research" in cards.models

    def test_custom_params(self):
        """Test provider initialization with custom parameters."""
        from deep_research_client.provider_params import CyberianParams

        params = CyberianParams(
            agent_type="aider",
            port=4000,
            skip_permissions=False,
            sources="academic papers only"
        )

        provider = CyberianProvider(self.config, params)

        assert provider.agent_type == "aider"
        assert provider.params.port == 4000
        assert provider.skip_permissions is False
        assert provider.params.sources == "academic papers only"

    def test_default_workflow_path(self):
        """Test default workflow path resolution."""
        provider = CyberianProvider(self.config)

        # The method should return a path to the bundled workflow
        workflow_path = provider._default_workflow_path()
        assert isinstance(workflow_path, str)
        assert "deep-research.yaml" in workflow_path
        assert "workflows" in workflow_path

        # Verify the file actually exists and is valid
        import os
        import yaml
        assert os.path.exists(workflow_path)

        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)
            assert workflow["name"] == "deep-research"
            assert "subtasks" in workflow
