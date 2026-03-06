"""Asta provider using Allen AI's scientific corpus retrieval tools."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import httpx

from . import ResearchProvider
from ..model_cards import ProviderModelCards, create_asta_model_cards
from ..models import ProviderConfig, ResearchResult
from ..provider_params import AstaParams

logger = logging.getLogger(__name__)

ASTA_MCP_URL = "https://asta-tools.allen.ai/mcp/v1"


@dataclass
class AstaPaper:
    """Normalized Asta paper search result."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    journal: str = ""
    url: str = ""
    publication_date: str = ""
    tldr: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AstaSnippet:
    """Normalized Asta snippet search result."""

    snippet: str
    paper_id: str = ""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    url: str = ""
    score: Optional[float] = None
    raw: dict[str, Any] = field(default_factory=dict)


class AstaProvider(ResearchProvider):
    """Provider for Asta paper and snippet retrieval."""

    def __init__(self, config: ProviderConfig, params: Optional[AstaParams] = None):
        """Initialize Asta provider."""
        self.params = params or AstaParams()
        super().__init__(config, self.params.model)

        logger.debug(f"Initializing Asta provider with mode: {self.model}")
        if config.api_key:
            key_preview = config.api_key[:8] + "..." if len(config.api_key) > 8 else "***"
            logger.debug(f"Asta API key configured (starts with: {key_preview})")

    def get_default_model(self) -> str:
        """Get the fixed retrieval mode label."""
        return "Asta Scientific Corpus Retrieval"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for Asta provider."""
        return create_asta_model_cards()

    async def research(self, query: str) -> ResearchResult:
        """Retrieve papers and snippets from Asta and format them as markdown."""
        logger.info(f"Starting Asta retrieval query (mode: {self.model})")
        logger.debug(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        if not self.is_available():
            raise ValueError(f"Asta provider not available (API key: {bool(self.config.api_key)})")

        timeout_seconds = self.config.timeout or 180
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0,
                read=timeout_seconds,
                write=30.0,
                pool=30.0,
            )
        ) as http_client:
            papers = await self._search_papers(http_client, query)
            snippets = await self._search_snippets(http_client, query, papers)

        logger.info(
            "Asta retrieval completed with %s papers and %s snippets",
            len(papers),
            len(snippets),
        )

        return ResearchResult(
            markdown=self._format_research_report(query, papers, snippets),
            citations=self._format_citations(papers),
            provider=self.name,
            query=query,
        )

    async def _search_papers(self, http_client: httpx.AsyncClient, query: str) -> list[AstaPaper]:
        """Retrieve relevant papers from Asta."""
        payload = await self._call_tool(
            http_client,
            "search_papers_by_relevance",
            {
                "keyword": query,
                "fields": self.params.paper_fields,
                "limit": self.params.paper_limit,
                "publication_date_range": self.params.publication_date_range,
                "venues": self.params.venues,
            },
        )
        return self._normalize_papers(payload)

    async def _search_snippets(
        self,
        http_client: httpx.AsyncClient,
        query: str,
        papers: list[AstaPaper],
    ) -> list[AstaSnippet]:
        """Retrieve evidence snippets from Asta."""
        paper_ids = ",".join(
            paper.paper_id
            for paper in papers[: self.params.snippet_paper_limit]
            if paper.paper_id
        )
        payload = await self._call_tool(
            http_client,
            "snippet_search",
            {
                "query": query,
                "limit": self.params.snippet_limit,
                "venues": self.params.venues,
                "paper_ids": paper_ids,
                "inserted_before": self.params.inserted_before,
            },
        )
        return self._normalize_snippets(payload)

    async def _call_tool(
        self,
        http_client: httpx.AsyncClient,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call an Asta MCP tool and return the normalized payload."""
        logger.debug("Calling Asta tool %s with arguments: %s", tool_name, arguments)

        response = await http_client.post(
            self.config.base_url or ASTA_MCP_URL,
            json={
                "jsonrpc": "2.0",
                "id": str(uuid4()),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key or "",
            },
        )
        response.raise_for_status()

        payload = self._extract_rpc_payload(response.text)
        if "error" in payload:
            raise ValueError(f"Asta MCP error calling {tool_name}: {payload['error']}")

        result = payload.get("result", {})
        if result.get("isError"):
            raise ValueError(self._extract_tool_error(tool_name, result))

        return self._extract_tool_payload(result)

    def _extract_rpc_payload(self, response_text: str) -> dict[str, Any]:
        """Extract the JSON-RPC payload from an SSE or JSON response.

        >>> provider = AstaProvider(ProviderConfig(name="asta", api_key="test"))
        >>> sse = 'event: message\\ndata: {"jsonrpc":"2.0","id":"1","result":{"isError":false}}\\n\\n'
        >>> provider._extract_rpc_payload(sse)["result"]["isError"]
        False
        """
        response_text = response_text.strip()
        if not response_text:
            raise ValueError("Empty response from Asta MCP endpoint")

        if response_text.startswith("{"):
            return json.loads(response_text)

        payloads: list[dict[str, Any]] = []
        for line in response_text.splitlines():
            if not line.startswith("data: "):
                continue
            candidate = line[6:].strip()
            if not candidate:
                continue
            payloads.append(json.loads(candidate))

        if not payloads:
            raise ValueError(f"Unable to parse Asta MCP response: {response_text[:200]}")
        return payloads[-1]

    def _extract_tool_payload(self, result: dict[str, Any]) -> Any:
        """Extract structured content from a tool call result."""
        structured_content = result.get("structuredContent")
        if structured_content is not None:
            return structured_content

        for block in result.get("content", []):
            if block.get("type") != "text":
                continue
            text = block.get("text", "").strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}

        return {}

    def _extract_tool_error(self, tool_name: str, result: dict[str, Any]) -> str:
        """Extract a readable error from a failed tool call."""
        for block in result.get("content", []):
            if block.get("type") == "text" and block.get("text"):
                return f"Asta tool {tool_name} failed: {block['text']}"
        return f"Asta tool {tool_name} failed without an error message"

    def _normalize_papers(self, payload: Any) -> list[AstaPaper]:
        """Normalize Asta paper results into a consistent structure."""
        papers: list[AstaPaper] = []
        for paper_data in self._coerce_result_list(payload):
            paper_id = str(
                paper_data.get("paperId")
                or paper_data.get("paper_id")
                or paper_data.get("corpusId")
                or ""
            )
            title = str(paper_data.get("title") or "Untitled")
            papers.append(
                AstaPaper(
                    paper_id=paper_id,
                    title=title,
                    authors=self._extract_author_names(paper_data.get("authors")),
                    abstract=str(paper_data.get("abstract") or ""),
                    year=self._coerce_int(paper_data.get("year")),
                    venue=str(paper_data.get("venue") or ""),
                    journal=self._extract_journal_name(paper_data.get("journal")),
                    url=str(paper_data.get("url") or ""),
                    publication_date=str(paper_data.get("publicationDate") or ""),
                    tldr=self._extract_tldr_text(paper_data.get("tldr")),
                    raw=paper_data,
                )
            )
        return papers

    def _normalize_snippets(self, payload: Any) -> list[AstaSnippet]:
        """Normalize Asta snippet results into a consistent structure."""
        snippets: list[AstaSnippet] = []
        for snippet_data in self._coerce_result_list(payload):
            paper_info = snippet_data.get("paper")
            if not isinstance(paper_info, dict):
                paper_info = snippet_data

            snippet_text = (
                snippet_data.get("snippet")
                or snippet_data.get("text")
                or snippet_data.get("passage")
                or snippet_data.get("content")
                or ""
            )
            snippets.append(
                AstaSnippet(
                    snippet=str(snippet_text),
                    paper_id=str(
                        snippet_data.get("paperId")
                        or snippet_data.get("paper_id")
                        or paper_info.get("paperId")
                        or ""
                    ),
                    title=str(snippet_data.get("title") or paper_info.get("title") or "Untitled"),
                    authors=self._extract_author_names(
                        snippet_data.get("authors") or paper_info.get("authors")
                    ),
                    year=self._coerce_int(snippet_data.get("year") or paper_info.get("year")),
                    url=str(snippet_data.get("url") or paper_info.get("url") or ""),
                    score=self._coerce_float(snippet_data.get("score") or snippet_data.get("relevance")),
                    raw=snippet_data,
                )
            )

        return [snippet for snippet in snippets if snippet.snippet]

    def _coerce_result_list(self, payload: Any) -> list[dict[str, Any]]:
        """Coerce Asta tool output into a list of dicts."""
        if isinstance(payload, dict):
            if isinstance(payload.get("result"), list):
                return [item for item in payload["result"] if isinstance(item, dict)]
            if isinstance(payload.get("results"), list):
                return [item for item in payload["results"] if isinstance(item, dict)]
            if isinstance(payload.get("snippets"), list):
                return [item for item in payload["snippets"] if isinstance(item, dict)]
            if isinstance(payload.get("items"), list):
                return [item for item in payload["items"] if isinstance(item, dict)]
            if payload:
                return [payload]
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _format_research_report(
        self,
        query: str,
        papers: list[AstaPaper],
        snippets: list[AstaSnippet],
    ) -> str:
        """Format Asta retrieval output as a structured markdown report."""
        lines = [
            f"# Asta Literature Retrieval: {query}",
            "",
            "This report is retrieval-only and is generated directly from Asta results.",
            "",
            f"- Papers retrieved: {len(papers)}",
            f"- Snippets retrieved: {len(snippets)}",
            "",
        ]

        lines.extend(["## Relevant Papers", ""])
        if not papers:
            lines.append("No papers were retrieved for this query.")
        else:
            for index, paper in enumerate(papers, start=1):
                author_str = ", ".join(paper.authors[:5]) if paper.authors else "Unknown authors"
                if len(paper.authors) > 5:
                    author_str += " et al."
                venue = paper.journal or paper.venue or "Unknown venue"
                summary = paper.tldr or paper.abstract or "No summary available."
                lines.extend(
                    [
                        f"### [{index}] {paper.title}",
                        f"- Authors: {author_str}",
                        f"- Year: {paper.year or 'Unknown'}",
                        f"- Venue: {venue}",
                        f"- Paper ID: {paper.paper_id or 'Unknown'}",
                        f"- URL: {paper.url or 'N/A'}",
                        f"- Summary: {self._truncate(summary, 500)}",
                        "",
                    ]
                )

        lines.extend(["## Evidence Snippets", ""])
        if not snippets:
            lines.append("No snippets were retrieved for this query.")
        else:
            for index, snippet in enumerate(snippets, start=1):
                score_line = f" (score: {snippet.score:.3f})" if snippet.score is not None else ""
                lines.extend(
                    [
                        f"### Snippet {index}: {snippet.title or 'Untitled'}{score_line}",
                        f"- Paper ID: {snippet.paper_id or 'Unknown'}",
                        f"- Year: {snippet.year or 'Unknown'}",
                        f"- URL: {snippet.url or 'N/A'}",
                        "",
                        f"> {self._quote_block(snippet.snippet)}",
                        "",
                    ]
                )

        lines.extend(
            [
                "## Notes",
                "",
                "- This provider combines `search_papers_by_relevance` with `snippet_search`.",
                "- No synthesis or second-stage model call is performed.",
            ]
        )
        return "\n".join(lines)

    def _format_citations(self, papers: list[AstaPaper]) -> list[str]:
        """Format normalized papers into citation strings."""
        citations: list[str] = []
        seen: set[str] = set()

        for paper in papers:
            author_str = ", ".join(paper.authors[:5]) if paper.authors else "Unknown authors"
            if len(paper.authors) > 5:
                author_str += " et al."

            parts = [f"{author_str} ({paper.year or 'n.d.'}). {paper.title}."]
            venue = paper.journal or paper.venue
            if venue:
                parts.append(f"{venue}.")
            if paper.url:
                parts.append(paper.url)
            elif paper.paper_id:
                parts.append(f"Asta paperId: {paper.paper_id}")

            citation = " ".join(parts).strip()
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)

        return citations

    def _extract_author_names(self, authors: Any) -> list[str]:
        """Extract author names from Asta response payloads."""
        if not isinstance(authors, list):
            return []

        names: list[str] = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("name")
                if name:
                    names.append(str(name))
            elif isinstance(author, str):
                names.append(author)
        return names

    def _extract_journal_name(self, journal: Any) -> str:
        """Extract a journal name from a string or object."""
        if isinstance(journal, dict):
            return str(journal.get("name") or journal.get("title") or "")
        if isinstance(journal, str):
            return journal
        return ""

    def _extract_tldr_text(self, tldr: Any) -> str:
        """Extract TLDR text from a string or object."""
        if isinstance(tldr, dict):
            return str(tldr.get("text") or "")
        if isinstance(tldr, str):
            return tldr
        return ""

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate long text for markdown rendering."""
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _quote_block(self, text: str) -> str:
        """Convert multi-line text to a markdown quote block."""
        return "\n> ".join(line.strip() for line in text.splitlines() if line.strip())

    def _coerce_int(self, value: Any) -> Optional[int]:
        """Coerce a value to an integer when possible."""
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        """Coerce a value to a float when possible."""
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
