"""Cyberian agent-based research provider.

This provider wraps the cyberian workflow system, using AI agents to
perform iterative deep research.
"""

import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from . import ResearchProvider
from ..models import ResearchResult, ProviderConfig
from ..provider_params import CyberianParams
from ..model_cards import ProviderModelCards, create_cyberian_model_cards

logger = logging.getLogger(__name__)


class CyberianProvider(ResearchProvider):
    """Provider that uses cyberian agent workflows for research.

    Unlike API-based providers, this provider runs a multi-step agent workflow
    that performs iterative research, citation management, and synthesis.
    """

    def __init__(self, config: ProviderConfig, params: Optional[CyberianParams] = None):
        """Initialize Cyberian provider.

        Args:
            config: Provider configuration
            params: Cyberian-specific parameters
        """
        self.params = params or CyberianParams()
        super().__init__(config, self.params.model)

        # Workflow configuration
        self.workflow_file = self.params.workflow_file or self._default_workflow_path()
        self.agent_type = self.params.agent_type or "claude"
        self.skip_permissions = self.params.skip_permissions

        logger.debug(f"Initializing Cyberian provider with workflow: {self.workflow_file}")
        logger.debug(f"Agent type: {self.agent_type}")

    def _default_workflow_path(self) -> str:
        """Get default path to deep-research.yaml workflow.

        The workflow is bundled with deep-research-client in the workflows/ directory.
        """
        # Get path to bundled workflow file
        workflow_path = Path(__file__).parent.parent / "workflows" / "deep-research.yaml"

        if workflow_path.exists():
            return str(workflow_path)

        # Fallback for development: try cyberian's example location
        try:
            import cyberian  # type: ignore[import-not-found, import-untyped]
            cyberian_path = Path(cyberian.__file__).parent.parent.parent
            fallback_path = cyberian_path / "tests" / "examples" / "deep-research.yaml"
            if fallback_path.exists():
                logger.warning(f"Using fallback workflow from cyberian repository: {fallback_path}")
                return str(fallback_path)
        except (ImportError, AttributeError):
            pass

        raise FileNotFoundError(
            f"Could not find deep-research.yaml workflow. "
            f"Expected at: {workflow_path}"
        )

    def get_default_model(self) -> str:
        """Get default model (workflow configuration)."""
        return "deep-research"  # The workflow name

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for Cyberian provider."""
        return create_cyberian_model_cards()

    def is_available(self) -> bool:
        """Check if cyberian is available."""
        if not self.config.enabled:
            return False

        # Check if cyberian can be imported
        try:
            import cyberian  # noqa
        except ImportError:
            logger.warning("cyberian not installed")
            return False

        # Check if agentapi is available in PATH
        import shutil
        if not shutil.which("agentapi"):
            logger.warning("agentapi not found in PATH")
            return False

        return True

    async def research(self, query: str) -> ResearchResult:
        """Perform research using cyberian agent workflow.

        Args:
            query: Research question

        Returns:
            ResearchResult with markdown report and citations
        """
        start_time = datetime.now()
        logger.info(f"Starting Cyberian research workflow (agent: {self.agent_type})")
        logger.debug(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        if not self.is_available():
            raise ValueError("Cyberian provider not available (cyberian not installed)")

        # Create temporary working directory
        with tempfile.TemporaryDirectory(prefix="cyberian_research_") as workdir:
            logger.debug(f"Created workdir: {workdir}")

            try:
                # Run the workflow in a thread (since TaskRunner is synchronous)
                await asyncio.to_thread(
                    self._run_workflow,
                    query=query,
                    workdir=workdir
                )

                # Parse results
                markdown_content = self._read_report(workdir)
                citations = self._extract_citations(workdir)

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.info(f"Cyberian research completed in {duration:.1f}s")
                logger.info(f"Report length: {len(markdown_content)} chars, Citations: {len(citations)}")

                return ResearchResult(
                    markdown=markdown_content,
                    citations=citations,
                    provider=self.name,
                    query=query,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    model=self.model
                )

            except Exception as e:
                logger.error(f"Cyberian workflow failed: {e}")
                logger.debug("Error details:", exc_info=True)
                raise ValueError(f"Cyberian workflow error: {e}")

    def _run_workflow(self, query: str, workdir: str) -> None:
        """Run cyberian workflow synchronously.

        Args:
            query: Research question
            workdir: Working directory for output
        """
        from cyberian.models import Task  # type: ignore[import-not-found, import-untyped]
        from cyberian.runner import TaskRunner  # type: ignore[import-not-found, import-untyped]

        # Load workflow definition
        logger.debug(f"Loading workflow from: {self.workflow_file}")
        with open(self.workflow_file, 'r') as f:
            workflow_data = yaml.safe_load(f)

        task = Task(**workflow_data)

        # Prepare context with parameters
        context = {
            "query": query,
            "workdir": workdir,
            "sources": self.params.sources or "all available sources"
        }

        logger.debug(f"Workflow context: {context}")

        # Create TaskRunner with lifecycle_mode='reuse' to manage server
        runner = TaskRunner(
            host="localhost",
            port=self.params.port or 3284,
            timeout=self.config.timeout or 600,
            lifecycle_mode="reuse",  # Keep server running for workflow
            agent_type=self.agent_type,
            skip_permissions=self.skip_permissions,
            directory=workdir
        )

        # Start the agent server
        logger.info(f"Starting {self.agent_type} agent server on port {self.params.port or 3284}...")
        runner._start_server()

        try:
            # Run the workflow
            logger.info("Executing cyberian workflow...")
            runner.run_task(task, context)
            logger.info("Workflow execution completed")
        finally:
            # Always stop the server when done
            logger.info("Stopping agent server...")
            runner._stop_server()

    def _read_report(self, workdir: str) -> str:
        """Read the REPORT.md file from workdir.

        Args:
            workdir: Working directory

        Returns:
            Markdown content of the report

        Raises:
            FileNotFoundError: If REPORT.md doesn't exist
        """
        report_path = Path(workdir) / "REPORT.md"

        if not report_path.exists():
            raise FileNotFoundError(
                f"REPORT.md not found in {workdir}. "
                "Workflow may have failed to complete."
            )

        with open(report_path, 'r') as f:
            content = f.read()

        logger.debug(f"Read report: {len(content)} characters")
        return content

    def _extract_citations(self, workdir: str) -> List[str]:
        """Extract citations from the citations/ directory.

        Args:
            workdir: Working directory

        Returns:
            List of citation strings
        """
        citations: List[str] = []
        citations_dir = Path(workdir) / "citations"

        if not citations_dir.exists():
            logger.warning(f"Citations directory not found: {citations_dir}")
            return citations

        # Collect all files in citations directory
        for citation_file in sorted(citations_dir.glob("*")):
            if citation_file.is_file():
                # Use filename as citation reference
                filename = citation_file.name
                citations.append(filename)

                logger.debug(f"Found citation file: {filename}")

        logger.debug(f"Extracted {len(citations)} citations from {citations_dir}")
        return citations
