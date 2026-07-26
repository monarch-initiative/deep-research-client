"""Biomni biomedical co-scientist provider.

`Biomni <https://github.com/snap-stanford/Biomni>`_ is a general-purpose
biomedical AI agent from Stanford SNAP. It wraps a large toolbox of biomedical
software and curated databases and executes generated code to plan and carry
out research tasks. Unlike a classic "deep research" provider that searches the
literature and writes a report, Biomni is a *co-scientist*: it forms and tests
hypotheses, runs code against biomedical data, and can design experiments. A
conventional literature-synthesis run is essentially a subset of what it does.

This provider wraps the local ``biomni`` Python package (``biomni.agent.A1``),
an optional dependency installed via ``pip install deep-research-client[biomni]``.
Because Biomni executes generated code locally and downloads a large (~11GB)
data lake, run it only in a trusted/sandboxed environment.
"""

import asyncio
from datetime import datetime
import importlib.util
import logging
import os
import re
from typing import Any, List, Optional

from . import ResearchProvider
from ..models import ProviderConfig, ResearchResult
from ..provider_params import BiomniParams
from ..model_cards import ProviderModelCards, create_biomni_model_cards

logger = logging.getLogger(__name__)

# Biomni agent runs are long: multi-step planning plus local code execution.
BIOMNI_DEFAULT_TIMEOUT = 3600
DEFAULT_DATA_PATH = "./biomni_data"


class BiomniProvider(ResearchProvider):
    """Provider that runs the local Biomni ``A1`` biomedical agent.

    Requires the optional ``biomni`` package. The underlying LLM (Claude by
    default) is configured through Biomni itself and authenticated via the
    usual provider environment variables (e.g. ``ANTHROPIC_API_KEY``).
    """

    def __init__(self, config: ProviderConfig, params: Optional[BiomniParams] = None):
        """Initialize the Biomni provider.

        Args:
            config: Provider configuration.
            params: Biomni-specific parameters.
        """
        self.params = params or BiomniParams()
        super().__init__(config, self.params.model)

        self.data_path = (
            self.params.path
            or os.getenv("BIOMNI_DATA_PATH")
            or DEFAULT_DATA_PATH
        )

    def get_default_model(self) -> str:
        """Return the default research model card identifier."""
        return "biomni-a1"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for the Biomni provider."""
        return create_biomni_model_cards()

    def is_available(self) -> bool:
        """Check whether the optional ``biomni`` package is importable."""
        if not self.config.enabled:
            return False
        if importlib.util.find_spec("biomni") is None:
            logger.warning("biomni not installed (pip install deep-research-client[biomni])")
            return False
        return True

    async def research(self, query: str) -> ResearchResult:
        """Run a Biomni agent task and return the result as a research report.

        Args:
            query: The biomedical task or research question.

        Returns:
            ResearchResult with the agent's final answer as markdown and any
            PMID/DOI citations extracted from it.
        """
        # Validate the query first so a caller error surfaces regardless of
        # whether the optional package happens to be installed.
        if not query or not query.strip():
            raise ValueError("Research query must not be empty.")
        if not self.is_available():
            raise ValueError(
                "Biomni provider not available. "
                "Install it with: pip install deep-research-client[biomni]"
            )

        start_time = datetime.now()
        logger.info("Starting Biomni agent run (data path: %s)", self.data_path)
        logger.debug("Query: %s%s", query[:100], "..." if len(query) > 100 else "")

        try:
            raw = await asyncio.to_thread(self._run_agent, query)
        except Exception as e:  # noqa: BLE001 - surface a clean provider error
            logger.error("Biomni agent run failed: %s", e)
            logger.debug("Error details:", exc_info=True)
            raise ValueError(f"Biomni agent error: {e}") from e

        markdown = self._result_to_markdown(raw)
        citations = self._extract_citations(markdown)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            "Biomni run complete in %.1fs: %d chars, %d citations",
            duration, len(markdown), len(citations),
        )

        return ResearchResult(
            markdown=markdown,
            citations=citations,
            provider=self.name,
            query=query,
            model=self.model,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )

    def _build_agent(self) -> Any:
        """Construct a Biomni ``A1`` agent from the configured parameters."""
        from biomni.agent import A1  # type: ignore[import-not-found, import-untyped]

        kwargs: dict[str, Any] = {
            "path": self.data_path,
            "use_tool_retriever": self.params.use_tool_retriever,
        }
        if self.params.llm:
            kwargs["llm"] = self.params.llm
        if self.params.source:
            kwargs["source"] = self.params.source
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        # A1's parameter is `timeout_seconds` (not `timeout`). Let an explicit
        # BiomniParams.timeout win over the config default, then the module
        # default.
        timeout = self.params.timeout or self.config.timeout or BIOMNI_DEFAULT_TIMEOUT
        kwargs["timeout_seconds"] = timeout
        if self.params.skip_data_lake:
            # An empty expected-files list tells A1 not to load/download the lake.
            kwargs["expected_data_lake_files"] = []

        logger.debug("Constructing Biomni A1 with: %s", sorted(kwargs))
        return A1(**kwargs)

    def _run_agent(self, query: str) -> Any:
        """Run the Biomni agent synchronously and return its raw output."""
        agent = self._build_agent()
        return agent.go(query)

    @staticmethod
    def _result_to_markdown(result: Any) -> str:
        """Normalize Biomni's ``go()`` return value into markdown text.

        ``A1.go`` may return the final answer string, or a ``(log, output)``
        pair, or a list of messages. This coerces those shapes into a single
        markdown string, preferring the final answer.
        """
        if isinstance(result, str):
            return result
        if isinstance(result, tuple) or isinstance(result, list):
            if len(result) == 2 and isinstance(result[1], str):
                return result[1]
            parts = [p for p in result if isinstance(p, str)]
            if parts:
                return parts[-1]
            if result:
                return str(result[-1])
            return ""
        for attr in ("content", "output", "final_answer", "answer"):
            value = getattr(result, attr, None)
            if isinstance(value, str):
                return value
        return str(result)

    @staticmethod
    def _extract_citations(markdown: str) -> List[str]:
        """Extract PMID and DOI citations from the report text.

        >>> BiomniProvider._extract_citations("See PMID: 12345678 and doi:10.1/xyz")
        ['PMID:12345678', 'doi:10.1/xyz']
        >>> BiomniProvider._extract_citations("PMID:11111111 again PMID: 11111111")
        ['PMID:11111111']
        >>> BiomniProvider._extract_citations("PMID: 123456789")
        ['PMID:123456789']
        """
        citations: List[str] = []
        seen: set[str] = set()

        # Bound the match with (?!\d) so a longer numeric string is not silently
        # truncated to a wrong PMID (PubMed IDs are 7-9 digits).
        for pmid in re.findall(r"PMID:\s*(\d{7,9})(?!\d)", markdown):
            ref = f"PMID:{pmid}"
            if ref not in seen:
                seen.add(ref)
                citations.append(ref)

        for doi in re.findall(r"\bdoi:\s*(10\.\S+)", markdown, flags=re.IGNORECASE):
            ref = f"doi:{doi.rstrip('.,);]')}"
            if ref not in seen:
                seen.add(ref)
                citations.append(ref)

        return citations
