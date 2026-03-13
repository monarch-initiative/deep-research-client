"""Tests for the Asta provider."""

import os

import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.providers.asta import AstaProvider


def test_asta_provider_availability_uses_api_key_only():
    """Asta should be available when its API key is configured."""
    provider = AstaProvider(ProviderConfig(
        name="asta", api_key="asta-key", enabled=True))
    assert provider.is_available() is True

    unavailable = AstaProvider(ProviderConfig(
        name="asta", api_key=None, enabled=True))
    assert unavailable.is_available() is False


def test_extract_rpc_payload_from_sse():
    """SSE-wrapped MCP responses should parse into a JSON-RPC payload."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    payload = provider._extract_rpc_payload(
        'event: message\n'
        'data: {"jsonrpc":"2.0","id":"abc","result":{"structuredContent":{"result":[{"title":"Paper"}]}}}\n\n'
    )

    assert payload["id"] == "abc"
    assert payload["result"]["structuredContent"]["result"][0]["title"] == "Paper"


def test_extract_tool_payload_prefers_structured_content():
    """Structured MCP tool payloads should be returned directly."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    payload = provider._extract_tool_payload(
        {
            "structuredContent": {"result": [{"paperId": "p1", "title": "Paper 1"}]},
            "content": [{"type": "text", "text": '{"ignored": true}'}],
        }
    )

    assert payload == {"result": [{"paperId": "p1", "title": "Paper 1"}]}


def test_normalize_papers_and_citations():
    """Paper normalization should preserve metadata needed for citations."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    papers = provider._normalize_papers(
        {
            "result": [
                {
                    "paperId": "paper-1",
                    "title": "A Paper",
                    "authors": [{"name": "Author One"}, {"name": "Author Two"}],
                    "year": 2024,
                    "venue": "Nature",
                    "url": "https://example.org/paper-1",
                    "tldr": {"text": "Important finding."},
                }
            ]
        }
    )

    assert len(papers) == 1
    assert papers[0].paper_id == "paper-1"
    assert papers[0].authors == ["Author One", "Author Two"]
    assert papers[0].tldr == "Important finding."

    citations = provider._format_citations(papers)
    assert citations == [
        "Author One, Author Two (2024). A Paper. Nature. https://example.org/paper-1"
    ]


def test_normalize_snippets_handles_nested_paper_records():
    """Snippet normalization should handle nested paper metadata."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    snippets = provider._normalize_snippets(
        {
            "result": [
                {
                    "text": "Evidence excerpt",
                    "score": 0.98,
                    "paper": {
                        "paperId": "paper-1",
                        "title": "A Paper",
                        "authors": [{"name": "Author One"}],
                        "year": 2024,
                        "url": "https://example.org/paper-1",
                    },
                }
            ]
        }
    )

    assert len(snippets) == 1
    assert snippets[0].snippet == "Evidence excerpt"
    assert snippets[0].paper_id == "paper-1"
    assert snippets[0].title == "A Paper"
    assert snippets[0].authors == ["Author One"]


def test_format_research_report_lists_papers_and_snippets():
    """The retrieval report should expose numbered papers and snippet evidence."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    papers = provider._normalize_papers(
        {"result": [{"paperId": "paper-1", "title": "A Paper", "year": 2024}]}
    )
    snippets = provider._normalize_snippets(
        {"result": [{"snippet": "Evidence excerpt",
                     "paperId": "paper-1", "title": "A Paper"}]}
    )

    report = provider._format_research_report("test query", papers, snippets)

    assert "# Asta Literature Retrieval: test query" in report
    assert "This report is retrieval-only" in report
    assert "### [1] A Paper" in report
    assert "### Snippet 1: A Paper" in report
    assert "Evidence excerpt" in report


@pytest.mark.integration
async def test_asta_research_integration():
    """Asta integration test against the live MCP endpoint."""
    api_key = os.getenv("ASTA_API_KEY")
    if not api_key:
        pytest.skip("ASTA_API_KEY not set")

    provider = AstaProvider(
        ProviderConfig(
            name="asta",
            api_key=api_key,
            enabled=True,
            timeout=300,
        )
    )

    result = await provider.research("hyperlipidemia")

    assert result.provider == "asta"
    assert result.query == "hyperlipidemia"
    assert result.markdown.startswith(
        "# Asta Literature Retrieval: hyperlipidemia")
    assert "## Relevant Papers" in result.markdown
    assert "## Evidence Snippets" in result.markdown
    assert len(result.citations) > 0
