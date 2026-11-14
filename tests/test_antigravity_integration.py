"""Live integration tests for the Antigravity (agy) provider.

Marked ``integration``: they run only under ``just test-full`` /
``just pytest-integration`` and are skipped when the ``agy`` CLI is not
installed. The research test drives the real agent, so it is slow and spends a
little Antigravity credit; the health probe is free (no model tokens).
"""

import asyncio

import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import AntigravityParams
from deep_research_client.providers.antigravity import AntigravityProvider


def _provider(params: AntigravityParams | None = None) -> AntigravityProvider:
    return AntigravityProvider(ProviderConfig(name="antigravity"), params)


@pytest.mark.integration
def test_live_models_probe():
    """`agy models` answers the health probe without spending model tokens."""
    provider = _provider()
    if not provider.is_available():
        pytest.skip("agy CLI not installed")

    health = asyncio.run(provider.check_health())

    assert health.configured is True
    assert health.reachable is not None


@pytest.mark.integration
def test_live_research_returns_a_cited_report():
    """A real headless run retrieves from the web and returns a cited report.

    Uses a fast model and a short query to keep cost and time down. The default
    read-only permission mode is relied on (no --dangerously-skip-permissions).
    """
    provider = _provider(AntigravityParams(model="gemini-3.8-flash-medium"))
    if not provider.is_available():
        pytest.skip("agy CLI not installed")

    result = asyncio.run(provider.research(
        "In two sentences, what is CRISPR? Use one web search and cite one source URL."
    ))

    assert result.provider == "antigravity"
    assert result.markdown.strip()
    # The run should have retrieved something, so it is not a from-memory answer.
    assert result.run_metadata is not None
    assert result.run_metadata.get("retrieval_calls", 0) >= 1
    assert result.citations, "a researched report should cite at least one URL"
