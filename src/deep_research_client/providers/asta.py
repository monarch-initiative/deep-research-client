"""Asta provider using Allen AI's scientific corpus retrieval tools."""

import json
import logging
import math
import re
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
MAX_COERCED_INT_ABS = (2 ** 63) - 1
ASTA_REPORT_QUERY_MAX_CHARS = 120

# Keep Markdown image alt text while dropping the linked asset target.
MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
# Keep Markdown link labels while dropping their destinations.
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# Preserve inline code contents but remove the surrounding backticks.
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
# Strip one leading Markdown structural prefix at a time so nested forms such as
# "> - item" collapse fully into plain text.
LEADING_MARKDOWN_PREFIX_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*|>\s?|(?:[-*+]|\d+\.)\s+)"
)
# Remove simple emphasis markers after links/code have been normalized.
INLINE_MARKER_RE = re.compile(r"\*\*|__|`|\*")
# Collapse repeated whitespace introduced by Markdown cleanup.
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class AstaPaper:
    """Normalized Asta paper search result."""

    paper_id: str
    title: str
    corpus_id: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    journal: str = ""
    url: str = ""
    publication_date: str = ""
    tldr: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
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
            key_preview = config.api_key[:8] + \
                "..." if len(config.api_key) > 8 else "***"
            logger.debug(
                f"Asta API key configured (starts with: {key_preview})")

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
        logger.debug(
            f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        if not self.is_available():
            raise ValueError(
                f"Asta provider not available (API key: {bool(self.config.api_key)})")

        search_query, display_query = self._prepare_query_text(query)
        if search_query != query.strip():
            logger.debug("Query sanitized: %d -> %d chars",
                         len(query), len(search_query))

        timeout_seconds = self.config.timeout or 180
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0,
                read=timeout_seconds,
                write=30.0,
                pool=30.0,
            )
        ) as http_client:
            papers = await self._search_papers(http_client, search_query)
            snippets = await self._search_snippets(
                http_client, search_query, papers)
            papers = await self._enrich_snippet_source_papers(http_client, papers, snippets)
            papers = self._merge_snippet_source_papers(papers, snippets)

        logger.info(
            "Asta retrieval completed with %s papers and %s snippets",
            len(papers),
            len(snippets),
        )

        return ResearchResult(
            markdown=self._format_research_report(
                display_query, papers, snippets),
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
        payload = await self._call_tool(
            http_client,
            "snippet_search",
            self._build_snippet_search_arguments(query, papers),
        )
        return self._normalize_snippets(payload)

    async def _enrich_snippet_source_papers(
        self,
        http_client: httpx.AsyncClient,
        papers: list[AstaPaper],
        snippets: list[AstaSnippet],
    ) -> list[AstaPaper]:
        """Fetch fuller paper metadata for snippet-only corpus IDs."""
        batch_identifiers = self._build_snippet_source_batch_identifiers(
            papers, snippets)
        if not batch_identifiers:
            return papers

        payload = await self._call_tool(
            http_client,
            "get_paper_batch",
            {
                "ids": batch_identifiers,
                "fields": self.params.paper_fields,
            },
        )
        enriched_papers = self._normalize_papers(payload)
        return self._merge_paper_lists(papers, enriched_papers)

    def _build_snippet_search_arguments(
        self,
        query: str,
        papers: list[AstaPaper],
    ) -> dict[str, Any]:
        """Build snippet search arguments."""
        arguments: dict[str, Any] = {
            "query": query,
            "limit": self.params.snippet_limit,
            "venues": self.params.venues,
            "inserted_before": self.params.inserted_before,
        }

        if self.params.restrict_snippets_to_papers:
            paper_ids = ",".join(
                paper.paper_id
                for paper in papers[: self.params.snippet_paper_limit]
                if paper.paper_id
            )
            if paper_ids:
                arguments["paper_ids"] = paper_ids

        return arguments

    def _prepare_query_text(self, query: str) -> tuple[str, str]:
        """Convert user input into a retrieval-friendly query and display label."""
        search_query = self._sanitize_query_text(query)
        display_query = self._truncate(
            search_query, ASTA_REPORT_QUERY_MAX_CHARS)
        return search_query, display_query

    def _sanitize_query_text(self, query: str) -> str:
        """Strip Markdown-heavy prompt formatting into plain text for Asta.

        >>> provider = AstaProvider(ProviderConfig(name="asta", api_key="test"))
        >>> provider._sanitize_query_text("# Query\\n- **Disease Name:** Contact Dermatitis")
        'Query Disease Name: Contact Dermatitis'
        """
        cleaned_lines: list[str] = []
        seen_lines: set[str] = set()
        for raw_line in self._strip_frontmatter(query).splitlines():
            cleaned = self._sanitize_query_line(raw_line)
            if not cleaned:
                continue
            normalized = cleaned.casefold()
            if normalized in seen_lines:
                continue
            seen_lines.add(normalized)
            cleaned_lines.append(cleaned)

        cleaned_query = " ".join(cleaned_lines)
        cleaned_query = WHITESPACE_RE.sub(" ", cleaned_query).strip()
        if not cleaned_query:
            logger.warning(
                "Query became empty after sanitization, using original")
            cleaned_query = WHITESPACE_RE.sub(" ", query).strip()
        if not cleaned_query:
            raise ValueError("Asta query is empty after sanitization")

        return self._truncate_at_word_boundary(
            cleaned_query,
            self.params.query_char_limit,
        )

    def _sanitize_query_line(self, line: str) -> str:
        """Normalize a single Markdown line into plain text."""
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            return ""

        stripped = MARKDOWN_IMAGE_RE.sub(r"\1", stripped)
        stripped = MARKDOWN_LINK_RE.sub(r"\1", stripped)
        stripped = INLINE_CODE_RE.sub(r"\1", stripped)
        while True:
            updated = LEADING_MARKDOWN_PREFIX_RE.sub("", stripped)
            if updated == stripped:
                break
            stripped = updated
        stripped = INLINE_MARKER_RE.sub("", stripped)
        stripped = stripped.replace("|", " ")
        stripped = WHITESPACE_RE.sub(" ", stripped)
        return stripped.strip(" \t:-")

    def _strip_frontmatter(self, text: str) -> str:
        """Remove leading Markdown frontmatter from a query if present."""
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return text

        closing_index = 1
        while closing_index < len(lines) and lines[closing_index].strip() != "---":
            closing_index += 1
        if closing_index >= len(lines):
            return text
        return "\n".join(lines[closing_index + 1:])

    def _build_snippet_source_batch_identifiers(
        self,
        papers: list[AstaPaper],
        snippets: list[AstaSnippet],
    ) -> list[str]:
        """Build batch lookup identifiers for snippet-only sources with corpus IDs."""
        known_source_ids = self._paper_source_identifiers(papers)
        identifiers: list[str] = []
        seen_identifiers: set[str] = set()

        for snippet in snippets:
            if snippet.paper_id and snippet.paper_id in known_source_ids:
                continue

            identifier = self._format_batch_paper_identifier(snippet.paper_id)
            if not identifier or identifier in seen_identifiers:
                continue

            seen_identifiers.add(identifier)
            identifiers.append(identifier)

        return identifiers

    async def _call_tool(
        self,
        http_client: httpx.AsyncClient,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call an Asta MCP tool and return the normalized payload."""
        logger.debug("Calling Asta tool %s with arguments: %s",
                     tool_name, arguments)

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
            raise ValueError(
                f"Asta MCP error calling {tool_name}: {payload['error']}")

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
            logger.debug(
                "Full unparseable Asta MCP response: %s", response_text)
            raise ValueError(
                f"Unable to parse Asta MCP response: {response_text[:200]}")
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
            external_ids = paper_data.get("externalIds")
            paper_id = str(
                paper_data.get("paperId")
                or paper_data.get("paper_id")
                or paper_data.get("corpusId")
                or ""
            )
            corpus_id = self._extract_external_id(external_ids, "CorpusId") or str(
                paper_data.get("corpusId") or ""
            )
            title = str(paper_data.get("title") or "Untitled")
            papers.append(
                AstaPaper(
                    paper_id=paper_id,
                    title=title,
                    corpus_id=corpus_id,
                    authors=self._extract_author_names(
                        paper_data.get("authors")),
                    abstract=str(paper_data.get("abstract") or ""),
                    year=self._coerce_int(paper_data.get("year")),
                    venue=str(paper_data.get("venue") or ""),
                    journal=self._extract_journal_name(
                        paper_data.get("journal")),
                    url=str(paper_data.get("url") or ""),
                    publication_date=str(
                        paper_data.get("publicationDate") or ""),
                    tldr=self._extract_tldr_text(paper_data.get("tldr")),
                    doi=self._extract_external_id(external_ids, "DOI"),
                    pmid=self._extract_external_id(
                        external_ids, "PubMed", "PMID"),
                    pmcid=self._extract_external_id(
                        external_ids, "PMCID", "PubMedCentral"),
                    citation_count=self._coerce_int(
                        paper_data.get("citationCount")
                        or paper_data.get("citation_count")
                    ),
                    influential_citation_count=self._coerce_int(
                        paper_data.get("influentialCitationCount")
                        or paper_data.get("influential_citation_count")
                    ),
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

            snippet_info = snippet_data.get("snippet")
            if not isinstance(snippet_info, dict):
                snippet_info = snippet_data

            snippet_text = (
                snippet_info.get("text")
                or snippet_data.get("snippet")
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
                        or paper_info.get("corpusId")
                        or ""
                    ),
                    title=str(snippet_data.get("title")
                              or paper_info.get("title") or "Untitled"),
                    authors=self._extract_author_names(
                        snippet_data.get(
                            "authors") or paper_info.get("authors")
                    ),
                    year=self._coerce_int(snippet_data.get(
                        "year") or paper_info.get("year")),
                    url=str(snippet_data.get("url")
                            or paper_info.get("url") or ""),
                    score=self._coerce_float(snippet_data.get(
                        "score") or snippet_data.get("relevance")),
                    raw=snippet_data,
                )
            )

        return [snippet for snippet in snippets if snippet.snippet]

    def _coerce_result_list(self, payload: Any) -> list[dict[str, Any]]:
        """Coerce Asta tool output into a list of dicts."""
        if isinstance(payload, dict):
            nested_result = payload.get("result")
            if isinstance(nested_result, dict):
                if isinstance(nested_result.get("data"), list):
                    return [item for item in nested_result["data"] if isinstance(item, dict)]
                if isinstance(nested_result.get("items"), list):
                    return [item for item in nested_result["items"] if isinstance(item, dict)]
            if isinstance(payload.get("result"), list):
                return [item for item in payload["result"] if isinstance(item, dict)]
            if isinstance(payload.get("data"), list):
                return [item for item in payload["data"] if isinstance(item, dict)]
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
            grouped_snippets, unmatched_snippets = self._group_snippets_by_paper(
                papers, snippets)
            for index, paper in enumerate(papers, start=1):
                author_str = ", ".join(
                    paper.authors[:5]) if paper.authors else "Unknown authors"
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
                        f"- URL: {paper.url or 'N/A'}",
                    ]
                )
                if paper.doi:
                    lines.append(f"- DOI: {paper.doi}")
                if paper.pmid:
                    lines.append(f"- PMID: {paper.pmid}")
                if paper.pmcid:
                    lines.append(f"- PMCID: {paper.pmcid}")
                if paper.citation_count is not None:
                    lines.append(f"- Citations: {paper.citation_count}")
                if paper.influential_citation_count is not None:
                    lines.append(
                        f"- Influential citations: {paper.influential_citation_count}"
                    )
                lines.append(f"- Summary: {self._truncate(summary, 500)}")
                paper_snippets = grouped_snippets.get(index - 1, [])
                if paper_snippets:
                    lines.extend(["- Evidence snippets:"])
                    for snippet_index, snippet in enumerate(paper_snippets, start=1):
                        score_line = f" (score: {snippet.score:.3f})" if snippet.score is not None else ""
                        lines.extend(
                            [
                                f"  - Snippet {snippet_index}{score_line}",
                                self._quote_block(
                                    snippet.snippet, indent="    "),
                            ]
                        )
                lines.append("")

            if unmatched_snippets:
                lines.extend(["## Additional Snippet Sources", ""])
                for index, snippet in enumerate(unmatched_snippets, start=1):
                    score_line = f" (score: {snippet.score:.3f})" if snippet.score is not None else ""
                    lines.extend(
                        [
                            f"### Snippet Source {index}: {snippet.title or 'Untitled'}{score_line}",
                            f"- Year: {snippet.year or 'Unknown'}",
                            f"- URL: {snippet.url or 'N/A'}",
                            "",
                            self._quote_block(snippet.snippet),
                            "",
                        ]
                    )
        if not snippets:
            lines.extend(["## Evidence Snippets", "",
                         "No snippets were retrieved for this query."])

        lines.extend(
            [
                "## Notes",
                "",
                "- This provider combines `search_papers_by_relevance` with `snippet_search`.",
                "- No synthesis or second-stage model call is performed.",
            ]
        )
        return "\n".join(lines)

    def _group_snippets_by_paper(
        self,
        papers: list[AstaPaper],
        snippets: list[AstaSnippet],
    ) -> tuple[dict[int, list[AstaSnippet]], list[AstaSnippet]]:
        """Group snippets under their most likely source paper."""
        grouped: dict[int, list[AstaSnippet]] = {
            index: [] for index in range(len(papers))}
        unmatched: list[AstaSnippet] = []

        title_to_indexes: dict[str, list[int]] = {}
        id_to_index: dict[str, int] = {}
        for index, paper in enumerate(papers):
            for identifier in self._paper_identifiers(paper):
                id_to_index[identifier] = index
            normalized_title = self._normalize_title(paper.title)
            if normalized_title:
                title_to_indexes.setdefault(normalized_title, []).append(index)

        for snippet in snippets:
            matched_index: Optional[int] = None
            if snippet.paper_id and snippet.paper_id in id_to_index:
                matched_index = id_to_index[snippet.paper_id]
            elif not snippet.paper_id:
                normalized_title = self._normalize_title(snippet.title)
                title_matches = title_to_indexes.get(normalized_title, [])
                if len(title_matches) == 1:
                    matched_index = title_matches[0]

            if matched_index is None:
                unmatched.append(snippet)
            else:
                grouped[matched_index].append(snippet)

        return grouped, unmatched

    def _merge_snippet_source_papers(
        self,
        papers: list[AstaPaper],
        snippets: list[AstaSnippet],
    ) -> list[AstaPaper]:
        """Add snippet-only source papers so their snippets can be grouped under them."""
        merged = list(papers)
        known_source_ids = self._paper_source_identifiers(merged)
        known_titles = self._untethered_titles(merged)

        for snippet in snippets:
            normalized_title = self._normalize_title(snippet.title)
            if snippet.paper_id:
                if snippet.paper_id in known_source_ids:
                    continue
            elif normalized_title in known_titles:
                continue

            merged.append(
                AstaPaper(
                    paper_id=snippet.paper_id,
                    title=snippet.title or "Untitled",
                    corpus_id=snippet.paper_id,
                    authors=snippet.authors,
                    year=snippet.year,
                    url=snippet.url,
                    raw=snippet.raw,
                )
            )
            if snippet.paper_id:
                known_source_ids.add(snippet.paper_id)
            elif normalized_title:
                known_titles.add(normalized_title)

        return merged

    def _merge_paper_lists(
        self,
        papers: list[AstaPaper],
        additional_papers: list[AstaPaper],
    ) -> list[AstaPaper]:
        """Merge paper lists without duplicating the same source."""
        merged = list(papers)
        known_identifiers = self._paper_source_identifiers(merged)
        known_titles = self._untethered_titles(merged)

        for paper in additional_papers:
            paper_identifiers = self._paper_identifiers(paper)
            normalized_title = self._normalize_title(paper.title)
            if paper_identifiers and any(
                identifier in known_identifiers for identifier in paper_identifiers
            ):
                continue

            if not paper_identifiers and normalized_title in known_titles:
                continue

            merged.append(paper)
            known_identifiers.update(paper_identifiers)
            if not paper_identifiers and normalized_title:
                known_titles.add(normalized_title)

        return merged

    def _format_citations(self, papers: list[AstaPaper]) -> list[str]:
        """Format normalized papers into citation strings."""
        citations: list[str] = []
        seen: set[str] = set()

        for paper in papers:
            author_str = ", ".join(
                paper.authors[:5]) if paper.authors else "Unknown authors"
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

    def _extract_external_id(self, external_ids: Any, *keys: str) -> str:
        """Extract a preferred identifier from Asta externalIds."""
        if not isinstance(external_ids, dict):
            return ""
        for key in keys:
            value = external_ids.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    def _truncate(self, text: str, limit: int) -> str:
        """Truncate long text for markdown rendering."""
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _truncate_at_word_boundary(self, text: str, limit: int) -> str:
        """Truncate text near a word boundary while preserving search signal."""
        if len(text) <= limit:
            return text

        truncated = text[:limit].rstrip()
        last_space = truncated.rfind(" ")
        if last_space >= limit // 2:
            truncated = truncated[:last_space]
        return truncated.rstrip(" ,;:")

    def _quote_block(self, text: str, indent: str = "") -> str:
        """Convert multi-line text to a markdown quote block."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(f"{indent}> {line}" for line in lines)

    def _normalize_title(self, title: str) -> str:
        """Normalize titles for grouping snippets with source papers."""
        return re.sub(r"[^a-z0-9]+", " ", title.casefold()).strip()

    def _paper_identifiers(self, paper: AstaPaper) -> set[str]:
        """Return all stable identifiers known for a paper."""
        identifiers: set[str] = set()
        if paper.paper_id:
            identifiers.add(paper.paper_id)
        if paper.corpus_id:
            identifiers.add(paper.corpus_id)
        return identifiers

    def _paper_source_identifiers(self, papers: list[AstaPaper]) -> set[str]:
        """Return stable identifiers across a paper list."""
        identifiers: set[str] = set()
        for paper in papers:
            identifiers.update(self._paper_identifiers(paper))
        return identifiers

    def _untethered_titles(self, papers: list[AstaPaper]) -> set[str]:
        """Return normalized titles only for papers without stable identifiers."""
        return {
            self._normalize_title(paper.title)
            for paper in papers
            if paper.title and not self._paper_identifiers(paper)
        }

    def _format_batch_paper_identifier(self, paper_id: str) -> str:
        """Convert a snippet paper identifier into a get_paper_batch identifier."""
        if not paper_id:
            return ""
        if ":" in paper_id:
            return paper_id
        if paper_id.isdigit():
            return f"CorpusId:{paper_id}"
        return ""

    def _coerce_int(self, value: Any) -> Optional[int]:
        """Coerce a value to an integer when possible."""
        if value in (None, ""):
            return None
        try:
            coerced = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if abs(coerced) > MAX_COERCED_INT_ABS:
            return None
        return coerced

    def _coerce_float(self, value: Any) -> Optional[float]:
        """Coerce a value to a float when possible."""
        if value in (None, ""):
            return None
        try:
            coerced = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(coerced):
            return None
        return coerced
