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

from deep_research_client.exceptions import ProviderTransientError
from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import PerplexityParams
from deep_research_client.providers.perplexity import (
    PerplexityProvider,
    _read_stream,
    _report_from,
    _StreamState,
    _strip_think,
    _titled_citations,
)


def _sse(chunk: dict) -> str:
    """Render one chunk as a server-sent-event data line."""
    return f"data: {json.dumps(chunk)}"


def _delta(text: str, finish_reason: str | None = None, **extra) -> str:
    """Render a true-delta chunk carrying ``text``, optionally the final one."""
    return _sse({"choices": [{"delta": {"content": text}, "finish_reason": finish_reason}], **extra})


def _message(text: str, **extra) -> str:
    """Render a cumulative chunk whose ``message.content`` is ``text``."""
    return _sse({"choices": [{"delta": {}, "message": {"content": text}}], **extra})


async def _lines(lines: List[str], then: Exception | None = None) -> AsyncIterator[str]:
    """Yield lines as a response body would, optionally failing at the end."""
    for line in lines:
        yield line
    if then is not None:
        raise then


def _fold(lines: List[str]) -> _StreamState:
    """Feed ``lines`` into a fresh stream state and return it."""
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


RESETS = [httpx.RemoteProtocolError("peer closed"), httpx.ReadError("reset")]


@pytest.mark.parametrize("error", RESETS)
def test_reset_after_the_final_chunk_keeps_the_report(error):
    """The server sometimes drops the connection right after the last chunk."""
    lines = _lines([_delta("# Rep"), _delta("ort", finish_reason="stop")], then=error)

    state = asyncio.run(_read_stream(lines))

    assert state.content == "# Report"
    assert state.finish_reason == "stop"


@pytest.mark.parametrize("error", RESETS)
@pytest.mark.parametrize(
    "lines",
    [[": keep-alive"], [_delta("# Rep")], [_delta("<think>planning the search")]],
    ids=["before-content", "mid-report", "mid-reasoning"],
)
def test_reset_before_the_final_chunk_is_an_error(lines, error):
    """Without a finish_reason the report is incomplete -- fail loudly, keep nothing.

    It is transient (a retry may succeed), keeps the network error as its
    cause, and says how far the stream got.
    """
    with pytest.raises(ProviderTransientError, match="connection dropped after") as raised:
        asyncio.run(_read_stream(_lines(lines, then=error)))

    assert raised.value.retryable
    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    "lines",
    [[], [_delta("# Rep")], [_delta("# Rep"), "data: [DONE]"]],
    ids=["empty", "content-only", "done-without-finish"],
)
def test_clean_end_without_the_final_chunk_is_an_error(lines):
    """A graceful close before the final chunk is as incomplete as a reset."""
    with pytest.raises(ProviderTransientError, match="without its final chunk"):
        asyncio.run(_read_stream(_lines(lines)))


@pytest.mark.parametrize("then", [None, httpx.RemoteProtocolError("peer closed")], ids=["clean-end", "reset"])
@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_final_chunk_other_than_stop_is_an_error(finish_reason, then):
    """Running out of tokens mid-report is a truncated report, not a finished one."""
    lines = _lines([_delta("# Rep", finish_reason=finish_reason)], then=then)

    with pytest.raises(ValueError, match=f"finish_reason='{finish_reason}'"):
        asyncio.run(_read_stream(lines))


@pytest.mark.parametrize(
    "error",
    [{"message": "quota exceeded", "type": "insufficient_quota"}, "quota exceeded"],
    ids=["object", "string"],
)
def test_error_event_is_raised_with_its_message(error):
    """Not 'No response content received' -- the server said what went wrong."""
    with pytest.raises(ValueError, match="error event: quota exceeded"):
        _fold([_delta("# Rep"), _sse({"error": error})])


def test_unreadable_line_names_the_line():
    """A format change should say what arrived, not just that JSON failed."""
    with pytest.raises(ValueError, match="not json"):
        _fold(["data: {not json"])


def test_search_result_titles_label_citations_in_citation_order():
    """[n] markers number into citations, so its order wins; titles come from search results."""
    citations = _titled_citations(
        ["https://a.example", "https://b.example"],
        [{"url": "https://b.example", "title": "B"}, {"url": "https://a.example", "title": "A"}],
    )

    assert citations == [
        {"url": "https://a.example", "title": "A"},
        {"url": "https://b.example", "title": "B"},
    ]


def test_extracted_citations_carry_search_result_titles():
    """The titles reach the result's citation strings."""
    provider = PerplexityProvider(ProviderConfig(name="perplexity", api_key="unused", enabled=True))
    state = _fold([
        _delta("# Report", citations=["https://a.example"],
               search_results=[{"url": "https://a.example", "title": "A"}]),
    ])

    citations = provider._extract_citations(
        state.content, {"citations": _titled_citations(state.citations, state.search_results)}
    )

    assert citations == ["A - https://a.example"]


def test_leading_think_block_is_stripped():
    assert _strip_think("<think>\nplan the search\n</think>\n\n# Report\n") == "# Report\n"


def test_unclosed_think_block_is_left_alone():
    """``_strip_think`` only removes a closed block; it leaves this for ``_report_from`` to reject."""
    assert _strip_think("<think>still planning") == "<think>still planning"


@pytest.mark.parametrize(
    "content",
    ["<think>still planning", "  <think>\nstill planning\n", "<think>planned</think>\n\n"],
    ids=["unclosed", "unclosed-after-whitespace", "reasoning-only"],
)
def test_reasoning_without_a_report_is_an_error(content):
    """The model's reasoning must never be returned as the report."""
    with pytest.raises(ValueError, match="<think>"):
        _report_from(content)


def test_report_after_reasoning_is_returned():
    assert _report_from("<think>planned</think>\n\n# Report\n") == "# Report\n"


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
