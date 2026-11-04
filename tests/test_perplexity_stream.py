"""Tests for reading Perplexity's streamed responses.

The provider streams so that long sonar-deep-research runs are not dropped by
idle-timeout proxies. These cover the chunk shapes seen in practice: cumulative
chunks, true deltas, citations on intermediate chunks, and a connection reset
just after the report is complete.
"""

import asyncio
import json
import os
from typing import AsyncIterator, List

import httpx
import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import PerplexityParams
from deep_research_client.providers.perplexity import (
    PerplexityProvider,
    _read_stream,
    _StreamState,
    _strip_think,
)


def _sse(chunk: dict) -> str:
    """Render one chunk as a server-sent-event data line."""
    return f"data: {json.dumps(chunk)}"


def _delta(text: str, **extra) -> str:
    return _sse({"choices": [{"delta": {"content": text}}], **extra})


def _message(text: str, **extra) -> str:
    return _sse({"choices": [{"delta": {}, "message": {"content": text}}], **extra})


async def _lines(lines: List[str], then: Exception | None = None) -> AsyncIterator[str]:
    """Yield lines as a response body would, optionally failing at the end."""
    for line in lines:
        yield line
    if then is not None:
        raise then


def _fold(lines: List[str]) -> _StreamState:
    state = _StreamState()
    for line in lines:
        state.add_line(line)
    return state


@pytest.mark.parametrize(
    "lines",
    [
        [_delta("# Re"), _delta("port"), "", "data: [DONE]"],
        [_delta("# Re"), _delta("# Report"), "data: [DONE]"],
        [_message("# Re"), _message("# Report"), "data: [DONE]"],
        [_message("# Report"), _message("# Rep")],
        [": keep-alive", _delta("# Re"), "", _delta("port")],
    ],
    ids=["deltas", "cumulative-deltas", "cumulative-messages", "shorter-message-ignored", "comments-and-blanks"],
)
def test_chunk_shapes_accumulate_to_the_same_report(lines):
    """Whatever shape the server picks, the report comes out once, whole."""
    assert _fold(lines).content == "# Report"


def test_citations_from_intermediate_chunks_are_kept():
    """The final chunk may not repeat citations; the last non-empty set wins."""
    state = _fold([
        _delta("# Re", citations=["https://a.example"]),
        _delta("port", citations=["https://a.example", "https://b.example"],
               search_results=[{"url": "https://b.example", "title": "B"}]),
        _delta("", citations=[], search_results=[]),
    ])

    assert state.citations == ["https://a.example", "https://b.example"]
    assert state.search_results == [{"url": "https://b.example", "title": "B"}]


def test_reset_after_content_keeps_the_report():
    """The server sometimes drops the connection right after the last chunk."""
    lines = _lines([_delta("# Report")], then=httpx.RemoteProtocolError("peer closed"))

    state = asyncio.run(_read_stream(lines))

    assert state.content == "# Report"


@pytest.mark.parametrize("error", [httpx.RemoteProtocolError("peer closed"), httpx.ReadError("reset")])
def test_reset_before_content_is_still_an_error(error):
    """Nothing arrived, so there is nothing to keep -- fail loudly."""
    with pytest.raises(type(error)):
        asyncio.run(_read_stream(_lines([": keep-alive"], then=error)))


def test_leading_think_block_is_stripped():
    assert _strip_think("<think>\nplan the search\n</think>\n\n# Report\n") == "# Report\n"


def test_unclosed_think_block_is_left_alone():
    """A stream cut off mid-reasoning should not silently become empty."""
    assert _strip_think("<think>still planning") == "<think>still planning"


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("PERPLEXITY_API_KEY"), reason="PERPLEXITY_API_KEY not set")
def test_streamed_research_returns_report_and_citations():
    """A short live query comes back through the streaming path intact."""
    provider = PerplexityProvider(
        ProviderConfig(name="perplexity", api_key=os.environ["PERPLEXITY_API_KEY"], enabled=True),
        PerplexityParams(model="sonar"),
    )

    result = asyncio.run(provider.research("What does the CFTR gene encode? One paragraph."))

    assert result.markdown
    assert not result.markdown.lstrip().startswith("<think>")
    assert result.citations
