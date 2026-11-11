"""Perplexity AI provider."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional, Dict, Any

import httpx

from . import ResearchProvider
from ..exceptions import ProviderError, ProviderNotConfiguredError, ProviderTransientError
from ..models import ResearchResult, ProviderConfig
from ..provider_params import PerplexityParams
from ..model_cards import ProviderModelCards, create_perplexity_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"\A\s*<think>.*?</think>\s*", re.DOTALL)


@dataclass
class _StreamState:
    """What has been gathered from a Perplexity event stream so far."""

    content: str = ""
    citations: List[Any] = field(default_factory=list)
    search_results: List[Any] = field(default_factory=list)
    finish_reason: Optional[str] = None

    def add_line(self, line: str) -> None:
        """Fold one server-sent-event line into the state.

        Perplexity is inconsistent about chunk shape: some streams repeat the
        whole answer so far in every chunk, others send true deltas. So keep
        the longest full ``message.content`` seen, and append ``delta.content``
        only when it is not itself the accumulated text plus more.

        >>> state = _StreamState()
        >>> for text in ["Hel", "lo"]:
        ...     state.add_line('data: {"choices": [{"delta": {"content": "%s"}}]}' % text)
        >>> state.content
        'Hello'
        >>> state = _StreamState()
        >>> for text in ["Hel", "Hello"]:
        ...     state.add_line('data: {"choices": [{"delta": {"content": "%s"}}]}' % text)
        >>> state.content
        'Hello'
        >>> state.add_line("data: [DONE]")
        >>> state.content
        'Hello'
        >>> state.finish_reason is None
        True
        >>> state.add_line('data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}')
        >>> state.finish_reason
        'stop'
        """
        if not line.startswith("data:"):
            return
        body = line[len("data:"):].strip()
        if not body or body == "[DONE]":
            return
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError as e:
            raise ValueError(f"Unreadable Perplexity stream line: {body[:200]!r}") from e

        # Citations and search results arrive on intermediate chunks, and the
        # final chunk does not always repeat them.
        if chunk.get("citations"):
            self.citations = chunk["citations"]
        if chunk.get("search_results"):
            self.search_results = chunk["search_results"]

        choices = chunk.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        if choice.get("finish_reason"):
            self.finish_reason = choice["finish_reason"]
        message = (choice.get("message") or {}).get("content")
        if message:
            if len(message) > len(self.content):
                self.content = message
            return
        delta = (choice.get("delta") or {}).get("content")
        if delta:
            # Deliberate trade-off: in a true-delta stream, a delta that happens
            # to begin with all the text so far is read as cumulative and the
            # earlier text is replaced. That needs the accumulated text to be a
            # prefix of the next piece, which is only plausible in the first
            # chunk or two (e.g. content "#" then delta "# Report").
            if self.content and delta.startswith(self.content):
                self.content = delta
            else:
                self.content += delta


async def _read_stream(lines: AsyncIterator[str]) -> _StreamState:
    """Consume a Perplexity event stream line by line.

    The body is never buffered whole: cumulative streams can run to hundreds
    of megabytes for a report of a few tens of kilobytes. The server sometimes
    resets the connection just after the report is complete, so a dropped
    connection is tolerated once the final chunk (the one carrying a
    ``finish_reason``) has arrived. A drop before that means the report is
    incomplete, and the error is raised rather than returning a partial report.
    The same holds for a stream that ends cleanly without its final chunk --
    a proxy closing gracefully, say -- which raises too. Both are raised as
    :class:`ProviderTransientError`, since running the request again may well
    succeed, and say how much had arrived. A final chunk whose
    ``finish_reason`` is anything but ``"stop"`` (``"length"`` when the model
    ran out of output tokens) also means the report was cut short.
    """
    state = _StreamState()
    try:
        async for line in lines:
            state.add_line(line)
    except (httpx.RemoteProtocolError, httpx.ReadError) as e:
        if state.finish_reason is None:
            raise ProviderTransientError(
                "perplexity",
                f"connection dropped after {len(state.content)} characters, "
                f"before the final chunk ({e})",
            ) from e
        logger.warning(
            f"Perplexity stream ended abruptly after the final chunk "
            f"(finish_reason={state.finish_reason!r}, {len(state.content)} characters); "
            f"keeping the report ({e})"
        )
    if state.finish_reason is None:
        raise ProviderTransientError(
            "perplexity",
            f"stream ended after {len(state.content)} characters without its "
            f"final chunk; the report is incomplete",
        )
    if state.finish_reason != "stop":
        raise ValueError(
            f"Perplexity stopped with finish_reason={state.finish_reason!r} after "
            f"{len(state.content)} characters; the report is incomplete"
        )
    return state


def _strip_think(content: str) -> str:
    """Remove a leading ``<think>...</think>`` reasoning block.

    >>> _strip_think("<think>\\nplanning\\n</think>\\n\\n# Report")
    '# Report'
    >>> _strip_think("# Report mentioning <think>tags</think>")
    '# Report mentioning <think>tags</think>'
    """
    return _THINK_BLOCK.sub("", content, count=1)


def _report_from(content: str) -> str:
    """Turn streamed content into the report, refusing reasoning-only output.

    A reasoning model writes a ``<think>`` block before the report. If the
    stream stops inside that block (a token limit hit while reasoning, say),
    or nothing follows it, there is no report, and returning the reasoning
    as one would pass off an unfinished run as a finished one.

    >>> _report_from("<think>plan</think>\\n# Report")
    '# Report'
    >>> _report_from("<think>still planning")
    Traceback (most recent call last):
    ...
    ValueError: Perplexity response ended inside its <think> reasoning block; no report was written
    >>> _report_from("<think>plan</think>\\n")
    Traceback (most recent call last):
    ...
    ValueError: Perplexity response contained only a <think> reasoning block; no report was written
    """
    report = _strip_think(content)
    if report.lstrip().startswith("<think>") and "</think>" not in report:
        raise ValueError(
            "Perplexity response ended inside its <think> reasoning block; no report was written"
        )
    if not report.strip():
        raise ValueError(
            "Perplexity response contained only a <think> reasoning block; no report was written"
        )
    return report


def _titled_citations(citations: List[Any], search_results: List[Any]) -> List[Any]:
    """Attach titles from ``search_results`` to the URLs in ``citations``.

    ``citations`` keeps its order, because the report's ``[n]`` markers
    number into it; ``search_results`` only supplies titles.

    >>> _titled_citations(["https://a.example", "https://b.example"],
    ...                   [{"url": "https://b.example", "title": "B"}])
    ['https://a.example', {'url': 'https://b.example', 'title': 'B'}]
    >>> _titled_citations([], [{"url": "https://b.example", "title": "B"}])
    [{'url': 'https://b.example', 'title': 'B'}]
    """
    if not citations:
        return list(search_results)
    titles = {
        r["url"]: r["title"]
        for r in search_results
        if isinstance(r, dict) and r.get("url") and r.get("title")
    }
    return [
        {"url": c, "title": titles[c]} if isinstance(c, str) and c in titles else c
        for c in citations
    ]


class PerplexityProvider(ResearchProvider):
    """Provider for Perplexity AI API."""

    credential_label = "Perplexity AI"
    credential_env_var = "PERPLEXITY_API_KEY"

    def __init__(self, config: ProviderConfig, params: Optional[PerplexityParams] = None):
        """Initialize Perplexity provider.

        Args:
            config: Provider configuration
            params: Perplexity-specific parameters
        """
        self.params = params or PerplexityParams()
        super().__init__(config, self.params.model)
        self.api_url = "https://api.perplexity.ai/chat/completions"

        logger.debug(f"Initializing Perplexity provider with model: {self.model}")
        if config.api_key:
            key_preview = config.api_key[:8] + "..." if len(config.api_key) > 8 else "***"
            logger.debug(f"API key configured (starts with: {key_preview})")

    def get_default_model(self) -> str:
        """Get default Perplexity model."""
        return "sonar-deep-research"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for Perplexity provider."""
        return create_perplexity_model_cards()

    async def research(self, query: str) -> ResearchResult:
        """Perform research using Perplexity AI API."""
        logger.info(f"Starting Perplexity research query (model: {self.model})")
        logger.debug(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        if not self.is_available():
            raise ProviderNotConfiguredError(self.name, self.unavailable_reason())

        # Use custom system prompt or default
        system_prompt = self.params.system_prompt or DEFAULT_RESEARCH_SYSTEM_PROMPT

        # Prepare request payload with all Perplexity-specific parameters
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": self.params.temperature,
        }

        # Add Perplexity-specific parameters
        if self.params.reasoning_effort != "medium":
            payload["reasoning_effort"] = self.params.reasoning_effort
            logger.debug(f"Reasoning effort: {self.params.reasoning_effort}")

        if self.params.search_recency_filter:
            payload["search_recency_filter"] = self.params.search_recency_filter
            logger.debug(f"Search recency filter: {self.params.search_recency_filter}")

        if self.params.search_domain_filter:
            payload["search_domain_filter"] = self.params.search_domain_filter
            logger.info(f"Domain filtering enabled: {len(self.params.search_domain_filter)} domains")
            logger.debug(f"Domain filter: {', '.join(self.params.search_domain_filter)}")

        # Always pass return_citations since it affects response format
        payload["return_citations"] = self.params.return_citations

        # Stream the response: sonar-deep-research can take several minutes and
        # sends nothing until it is done, so a single blocking POST gets cut off
        # by any proxy with an idle timeout. Streamed chunks keep the line busy.
        payload["stream"] = True

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }

        try:
            logger.debug(f"Making streaming API request to Perplexity (model: {self.model})")
            # The read timeout is the longest gap allowed between chunks; connect
            # gets its own short limit so a bad network fails fast.
            timeout = httpx.Timeout(self.config.timeout or 600, connect=30)
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    self.api_url,
                    json=payload,
                    headers=headers
                ) as response:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    stream = await _read_stream(response.aiter_lines())

            logger.info("Perplexity API request completed successfully")

            if not stream.content:
                raise ValueError("No response content received from Perplexity")
            markdown_content = _report_from(stream.content)
            logger.debug(f"Extracted markdown content: {len(markdown_content)} characters")
            data = {"citations": _titled_citations(stream.citations, stream.search_results)}

            # Extract citations from the response (both from content and metadata)
            citations = self._extract_citations(markdown_content, data)
            logger.info(f"Extracted {len(citations)} citations from response")

            return ResearchResult(
                markdown=markdown_content,
                citations=citations,
                provider=self.name,
                query=query
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Perplexity API HTTP error: {e.response.status_code}")
            logger.debug("Error details:", exc_info=True)
            raise ValueError(f"Perplexity API HTTP error: {e.response.status_code} - {e.response.text}")
        except ProviderError:
            # Already typed (e.g. a dropped stream is transient); wrapping it in
            # a bare ValueError would hide that from retry and fallback logic.
            raise
        except Exception as e:
            logger.error(f"Perplexity API request failed: {e}")
            logger.debug("Error details:", exc_info=True)
            raise ValueError(f"Perplexity API error: {e}")

    def _extract_citations(self, content: str, response_data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Extract citations from Perplexity response content and metadata."""
        citations = []

        # First, try to extract citations from response metadata (if available)
        if response_data and "citations" in response_data:
            metadata_citations = response_data["citations"]
            for citation in metadata_citations:
                if isinstance(citation, dict):
                    # Handle structured citation objects
                    if "url" in citation:
                        title = citation.get("title", "")
                        url = citation["url"]
                        if title:
                            citations.append(f"{title} - {url}")
                        else:
                            citations.append(url)
                elif isinstance(citation, str):
                    citations.append(citation)

        # Extract from content as fallback
        # Look for URLs
        url_pattern = re.compile(r'https?://[^\s\)\]]+')
        urls = url_pattern.findall(content)
        citations.extend(urls)

        # Look for parenthetical citations like (Source: ...)
        source_refs = re.findall(r'\(Source: ([^)]+)\)', content)
        citations.extend(source_refs)

        # Look for "according to" patterns
        according_to_refs = re.findall(r'according to ([^,.]+)', content, re.IGNORECASE)
        citations.extend(according_to_refs)

        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            citation = citation.strip()
            if citation and citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)

        return unique_citations