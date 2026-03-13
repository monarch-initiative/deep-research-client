"""Tests for the Asta provider."""

import os

import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import AstaParams
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
                    "externalIds": {
                        "DOI": "10.1000/test-doi",
                        "PubMed": "12345678",
                        "PMCID": "PMC123456",
                    },
                    "tldr": {"text": "Important finding."},
                    "citationCount": 42,
                    "influentialCitationCount": 7,
                }
            ]
        }
    )

    assert len(papers) == 1
    assert papers[0].paper_id == "paper-1"
    assert papers[0].authors == ["Author One", "Author Two"]
    assert papers[0].tldr == "Important finding."
    assert papers[0].doi == "10.1000/test-doi"
    assert papers[0].pmid == "12345678"
    assert papers[0].pmcid == "PMC123456"
    assert papers[0].citation_count == 42
    assert papers[0].influential_citation_count == 7

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


def test_normalize_snippets_handles_live_asta_payload_shape():
    """Snippet normalization should handle Asta's structuredContent.result.data shape."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    snippets = provider._normalize_snippets(
        {
            "result": {
                "data": [
                    {
                        "score": 1.23,
                        "paper": {
                            "corpusId": "5163820",
                            "title": "Pathophysiology of the antiphospholipid antibody syndrome",
                            "authors": ["R. Willis", "S. Pierangeli"],
                        },
                        "snippet": {
                            "text": "Pathophysiology of the antiphospholipid antibody syndrome",
                            "snippetKind": "title",
                        },
                    }
                ]
            }
        }
    )

    assert len(snippets) == 1
    assert snippets[0].snippet == "Pathophysiology of the antiphospholipid antibody syndrome"
    assert snippets[0].paper_id == "5163820"
    assert snippets[0].title == "Pathophysiology of the antiphospholipid antibody syndrome"
    assert snippets[0].authors == ["R. Willis", "S. Pierangeli"]
    assert snippets[0].score == 1.23


def test_format_research_report_lists_papers_and_snippets():
    """The retrieval report should expose numbered papers and snippet evidence."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    papers = provider._normalize_papers(
        {
            "result": [
                {
                    "paperId": "paper-1",
                    "title": "A Paper",
                    "year": 2024,
                    "externalIds": {
                        "DOI": "10.1000/test-doi",
                        "PubMed": "12345678",
                        "PMCID": "PMC123456",
                    },
                    "citationCount": 42,
                    "influentialCitationCount": 7,
                }
            ]
        }
    )
    snippets = provider._normalize_snippets(
        {"result": [{"snippet": "Evidence excerpt",
                     "paperId": "paper-1", "title": "A Paper"}]}
    )

    report = provider._format_research_report("test query", papers, snippets)

    assert "# Asta Literature Retrieval: test query" in report
    assert "This report is retrieval-only" in report
    assert "### [1] A Paper" in report
    assert "- DOI: 10.1000/test-doi" in report
    assert "- PMID: 12345678" in report
    assert "- PMCID: PMC123456" in report
    assert "- Citations: 42" in report
    assert "- Influential citations: 7" in report
    assert "- Paper ID:" not in report
    assert "- Evidence snippets:" in report
    assert "Snippet 1" in report
    assert "Evidence excerpt" in report
    assert "## Evidence Snippets" not in report


def test_group_snippets_by_title_when_ids_differ():
    """Snippets should still group with papers when title matches but IDs differ."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    papers = provider._normalize_papers(
        {"result": [{"paperId": "paper-1",
                     "title": "Pathophysiology of Antiphospholipid Syndrome"}]}
    )
    snippets = provider._normalize_snippets(
        {
            "result": {
                "data": [
                    {
                        "score": 1.2,
                        "paper": {
                            "corpusId": "244402730",
                            "title": "Pathophysiology of Antiphospholipid Syndrome",
                            "authors": ["D. Green"],
                        },
                        "snippet": {
                            "text": "Pathophysiology of Antiphospholipid Syndrome",
                        },
                    }
                ]
            }
        }
    )

    grouped, unmatched = provider._group_snippets_by_paper(papers, snippets)

    assert len(grouped[0]) == 1
    assert grouped[0][0].paper_id == "244402730"
    assert unmatched == []


def test_build_snippet_search_arguments_defaults_to_global_search():
    """Snippet search should not be constrained to the initial paper set by default."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    papers = provider._normalize_papers(
        {"result": [{"paperId": "paper-1", "title": "A Paper"}]}
    )

    arguments = provider._build_snippet_search_arguments("test query", papers)

    assert arguments["query"] == "test query"
    assert "paper_ids" not in arguments


def test_build_snippet_search_arguments_can_restrict_to_papers():
    """Snippet search can optionally be constrained to the initial paper set."""
    provider = AstaProvider(
        ProviderConfig(name="asta", api_key="asta-key"),
        AstaParams(restrict_snippets_to_papers=True),
    )
    papers = provider._normalize_papers(
        {"result": [{"paperId": "paper-1", "title": "A Paper"}]}
    )

    arguments = provider._build_snippet_search_arguments("test query", papers)

    assert arguments["paper_ids"] == "paper-1"


def test_format_batch_paper_identifier_supports_corpus_ids():
    """Batch paper lookup should prefix raw corpus IDs and preserve explicit prefixes."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))

    assert provider._format_batch_paper_identifier(
        "244402730") == "CorpusId:244402730"
    assert provider._format_batch_paper_identifier(
        "CorpusId:244402730") == "CorpusId:244402730"
    assert provider._format_batch_paper_identifier("paper-1") == ""


def test_build_snippet_source_batch_identifiers_targets_snippet_only_corpus_ids():
    """Only snippet-only corpus IDs should be queued for batch enrichment."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    papers = provider._normalize_papers(
        {"result": [{"paperId": "paper-1", "title": "Known Paper"}]}
    )
    snippets = provider._normalize_snippets(
        {
            "result": {
                "data": [
                    {
                        "paper": {
                            "paperId": "paper-1",
                            "title": "Known Paper",
                        },
                        "snippet": {"text": "Known snippet"},
                    },
                    {
                        "paper": {
                            "corpusId": "244402730",
                            "title": "Snippet Only Paper",
                        },
                        "snippet": {"text": "Snippet-only evidence"},
                    },
                    {
                        "paper": {
                            "corpusId": "244402730",
                            "title": "Snippet Only Paper",
                        },
                        "snippet": {"text": "Repeated snippet-only evidence"},
                    },
                ]
            }
        }
    )

    identifiers = provider._build_snippet_source_batch_identifiers(
        papers, snippets)

    assert identifiers == ["CorpusId:244402730"]


def test_merge_snippet_source_papers_adds_snippet_only_sources():
    """Snippet-only sources should be added to the paper list for grouping."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    papers = provider._normalize_papers(
        {"result": [{"paperId": "paper-1", "title": "A Paper"}]}
    )
    snippets = provider._normalize_snippets(
        {
            "result": {
                "data": [
                    {
                        "score": 0.9,
                        "paper": {
                            "corpusId": "paper-2",
                            "title": "B Paper",
                            "authors": ["Author Two"],
                        },
                        "snippet": {
                            "text": "B snippet",
                        },
                    }
                ]
            }
        }
    )

    merged = provider._merge_snippet_source_papers(papers, snippets)

    assert len(merged) == 2
    assert merged[1].paper_id == "paper-2"
    assert merged[1].title == "B Paper"


def test_merge_paper_lists_skips_duplicate_titles():
    """Enriched papers should not duplicate sources already present by title."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))
    papers = provider._normalize_papers(
        {"result": [{"paperId": "paper-1",
                     "title": "Pathophysiology of Antiphospholipid Syndrome"}]}
    )
    additional = provider._normalize_papers(
        {"result": [{"paperId": "paper-2",
                     "title": "Pathophysiology of Antiphospholipid Syndrome"}]}
    )

    merged = provider._merge_paper_lists(papers, additional)

    assert len(merged) == 1


def test_quote_block_preserves_indentation_for_multiline_snippets():
    """Multiline snippets should stay nested under their source bullet."""
    provider = AstaProvider(ProviderConfig(name="asta", api_key="asta-key"))

    quoted = provider._quote_block("First line\nSecond line", indent="    ")

    assert quoted == "    > First line\n    > Second line"


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
    assert "Evidence snippets:" in result.markdown or "## Additional Snippet Sources" in result.markdown
    assert len(result.citations) > 0
