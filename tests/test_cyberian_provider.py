"""Tests for the Cyberian provider."""

from pathlib import Path
import shutil
import socket
import tempfile

import pytest

# Skip all tests in this module if cyberian is not installed
pytest.importorskip("cyberian")

from deep_research_client.providers.cyberian import CyberianProvider
from deep_research_client.models import ProviderConfig


def get_free_port() -> int:
    """Get an available local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class TestCyberianProvider:
    """Test cases for CyberianProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="cyberian",
            api_key=None,  # Cyberian doesn't need API key
            enabled=True,
            timeout=600,  # 10 min timeout for integration tests
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
    @pytest.mark.parametrize("agent_type", ["claude", "codex"])
    async def test_research_integration(self, agent_type):
        """Integration test for research workflow with a managed agent.

        This test requires:
        - cyberian to be installed
        - agentapi in PATH
        - The agent binary in PATH (e.g., claude or codex)

        Uses quick-test workflow for faster execution.
        Mark as integration test to skip in normal test runs.
        """
        pytest.importorskip("cyberian")
        from deep_research_client.provider_params import CyberianParams

        if shutil.which(agent_type) is None:
            pytest.skip(f"{agent_type} not found in PATH")

        port = get_free_port()
        workdir_base = Path.cwd() / ".cyberian-test-workdir" / agent_type
        workdir_base.mkdir(parents=True, exist_ok=True)

        # Use quick-test workflow for faster testing
        quick_workflow = Path(__file__).parent.parent / "src" / "deep_research_client" / "workflows" / "quick-test.yaml"

        params = CyberianParams(
            agent_type=agent_type,
            port=port,
            manage_server=True,
            skip_permissions=True,
            workdir_base=str(workdir_base),
            max_iterations=2,
            workflow_file=str(quick_workflow),
        )
        provider = CyberianProvider(self.config, params)

        if not provider.is_available():
            pytest.skip("Cyberian not available")

        # Run a simple research query with quick-test workflow
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

    def test_codex_agent_type_initialization(self):
        """Test provider initialization with codex agent type."""
        from deep_research_client.provider_params import CyberianParams

        params = CyberianParams(
            agent_type="codex",
            port=3299,
            skip_permissions=True,
            manage_server=True,
        )

        provider = CyberianProvider(self.config, params)

        assert provider.agent_type == "codex"
        assert provider.params.port == 3299
        assert provider.skip_permissions is True
        assert provider.manage_server is True

    def test_codex_startup_warning_logged(self, caplog):
        """Test that codex agent logs a startup warning about longer wait times."""
        import logging
        from deep_research_client.provider_params import CyberianParams

        caplog.set_level(logging.WARNING)

        params = CyberianParams(
            agent_type="codex",
            manage_server=True,
        )

        CyberianProvider(self.config, params)

        # Check that the warning about codex startup time was logged
        assert any("Codex startup can take longer" in record.message for record in caplog.records)

    def test_codex_no_warning_when_not_managing_server(self, caplog):
        """Test that codex startup warning is NOT logged when manage_server=False."""
        import logging
        from deep_research_client.provider_params import CyberianParams

        caplog.set_level(logging.WARNING)

        params = CyberianParams(
            agent_type="codex",
            manage_server=False,  # Not managing server, so no warning expected
        )

        CyberianProvider(self.config, params)

        # Should NOT have the startup warning
        assert not any("Codex startup can take longer" in record.message for record in caplog.records)

    def test_max_iterations_param(self):
        """Test that max_iterations parameter is accepted."""
        from deep_research_client.provider_params import CyberianParams

        params = CyberianParams(
            agent_type="claude",
            max_iterations=3,
        )

        provider = CyberianProvider(self.config, params)
        assert provider.params.max_iterations == 3

    def test_max_iterations_validation(self):
        """Test that max_iterations must be positive."""
        from deep_research_client.provider_params import CyberianParams
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            CyberianParams(max_iterations=0)

        with pytest.raises(pydantic.ValidationError):
            CyberianParams(max_iterations=-1)


class TestCyberianCodexIntegration:
    """Integration tests specifically for Cyberian + Codex."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="cyberian",
            api_key=None,
            enabled=True,
            timeout=600,  # 10 min timeout for integration tests (with max_iterations)
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_codex_research_with_workdir_base(self):
        """Integration test for Codex research with workdir_base for trusted workspace.

        This test validates the v0.2.2rc1 features:
        - Codex agent type with proper permission bypass
        - workdir_base to place workspace in a trusted location
        - Longer startup timeout handling

        Requires:
        - cyberian >= 0.2.2rc1
        - agentapi in PATH
        - codex in PATH
        """
        pytest.importorskip("cyberian")
        from deep_research_client.provider_params import CyberianParams

        if shutil.which("codex") is None:
            pytest.skip("codex not found in PATH")

        port = get_free_port()
        workdir_base = Path.cwd() / ".cyberian-test-workdir" / "codex-integration"
        workdir_base.mkdir(parents=True, exist_ok=True)

        # Use quick-test workflow for faster testing
        quick_workflow = Path(__file__).parent.parent / "src" / "deep_research_client" / "workflows" / "quick-test.yaml"

        params = CyberianParams(
            agent_type="codex",
            port=port,
            manage_server=True,
            skip_permissions=True,
            workdir_base=str(workdir_base),
            max_iterations=2,  # Limit iterations for faster testing
            workflow_file=str(quick_workflow),  # Use quick test workflow
        )
        provider = CyberianProvider(self.config, params)

        if not provider.is_available():
            pytest.skip("Cyberian not available")

        # Run a simple test query with quick-test workflow
        result = await provider.research("What is the function of the p53 gene?")

        # Verify result structure
        assert result.provider == "cyberian"
        assert result.query == "What is the function of the p53 gene?"
        assert result.markdown  # Should have content
        assert result.model == "deep-research"
        assert result.start_time is not None
        assert result.end_time is not None
        assert result.duration_seconds > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_codex_research_external_server(self):
        """Integration test for Codex using an externally managed agentapi server.

        This tests the manage_server=False path where the user runs agentapi manually.
        Useful for debugging or when custom agentapi flags are needed.

        To run this test, first start the server manually:
            agentapi server codex --port 3299 -- --dangerously-bypass-approvals-and-sandbox
        """
        pytest.importorskip("cyberian")
        from deep_research_client.provider_params import CyberianParams

        if shutil.which("codex") is None:
            pytest.skip("codex not found in PATH")

        # Use a fixed port for external server
        external_port = 3299

        # Check if server is already running
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', external_port))
        sock.close()

        if result != 0:
            pytest.skip(
                f"External agentapi server not running on port {external_port}. "
                "Start it with: agentapi server codex --port 3299 -- --dangerously-bypass-approvals-and-sandbox"
            )

        workdir_base = Path.cwd() / ".cyberian-test-workdir" / "codex-external"
        workdir_base.mkdir(parents=True, exist_ok=True)

        params = CyberianParams(
            agent_type="codex",
            port=external_port,
            manage_server=False,  # Use external server
            skip_permissions=True,
            workdir_base=str(workdir_base),
        )
        provider = CyberianProvider(self.config, params)

        if not provider.is_available():
            pytest.skip("Cyberian not available")

        # Run a simple research query
        result = await provider.research("What is autophagy?")

        assert result.provider == "cyberian"
        assert result.markdown
        assert result.duration_seconds > 0


class TestCyberianParallelExecution:
    """Integration tests for parallel cyberian execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="cyberian",
            api_key=None,
            enabled=True,
            timeout=600,
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_research_queries(self):
        """Integration test for running multiple research queries in parallel.

        This test validates that:
        - Multiple cyberian providers can run concurrently
        - Each uses its own port and workdir to avoid conflicts
        - Both queries complete successfully

        Requires:
        - cyberian >= 0.2.2
        - agentapi in PATH
        - codex in PATH
        """
        import asyncio

        pytest.importorskip("cyberian")
        from deep_research_client.provider_params import CyberianParams

        if shutil.which("codex") is None:
            pytest.skip("codex not found in PATH")

        # Use quick-test workflow for faster testing
        quick_workflow = Path(__file__).parent.parent / "src" / "deep_research_client" / "workflows" / "quick-test.yaml"

        # Set up two independent providers with different ports and workdirs
        port1 = get_free_port()
        port2 = get_free_port()

        workdir_base1 = Path.cwd() / ".cyberian-test-workdir" / "parallel-1"
        workdir_base2 = Path.cwd() / ".cyberian-test-workdir" / "parallel-2"
        workdir_base1.mkdir(parents=True, exist_ok=True)
        workdir_base2.mkdir(parents=True, exist_ok=True)

        params1 = CyberianParams(
            agent_type="codex",
            port=port1,
            manage_server=True,
            skip_permissions=True,
            workdir_base=str(workdir_base1),
            max_iterations=2,
            workflow_file=str(quick_workflow),
        )

        params2 = CyberianParams(
            agent_type="codex",
            port=port2,
            manage_server=True,
            skip_permissions=True,
            workdir_base=str(workdir_base2),
            max_iterations=2,
            workflow_file=str(quick_workflow),
        )

        provider1 = CyberianProvider(self.config, params1)
        provider2 = CyberianProvider(self.config, params2)

        if not provider1.is_available() or not provider2.is_available():
            pytest.skip("Cyberian not available")

        # Run two queries in parallel
        query1 = "What is the function of the TP53 gene?"
        query2 = "What is the role of BRCA1 in DNA repair?"

        results = await asyncio.gather(
            provider1.research(query1),
            provider2.research(query2),
        )

        result1, result2 = results

        # Verify both completed successfully
        assert result1.provider == "cyberian"
        assert result1.query == query1
        assert result1.markdown
        assert result1.duration_seconds > 0

        assert result2.provider == "cyberian"
        assert result2.query == query2
        assert result2.markdown
        assert result2.duration_seconds > 0

        # Both should have completed (not necessarily at the same time)
        assert result1.end_time is not None
        assert result2.end_time is not None
