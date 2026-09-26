"""A provider that says it could not answer must not pass for a report (issue #52).

Driven by the packaged mock provider's `unsuccessful_answer` parameter, which
returns the shape falcon produces when retrieval finds nothing: an explanation
instead of a report, no citations, and `answer_successful=False`.
"""

from pathlib import Path

import pytest

from deep_research_client.client import DeepResearchClient
from deep_research_client.models import CacheConfig, ProviderConfig


def _client(cache_dir: Path) -> DeepResearchClient:
    """A client with only the mock provider and an on-disk cache in cache_dir."""
    return DeepResearchClient(
        cache_config=CacheConfig(enabled=True, directory=str(cache_dir)),
        provider_configs={"mock": ProviderConfig(name="mock", api_key="mock-key")},
    )


@pytest.mark.parametrize(
    "unsuccessful,cached_after",
    [(True, False), (False, True)],
    ids=["unsuccessful-is-not-cached", "control-successful-is-cached"],
)
def test_an_unsuccessful_answer_is_not_cached(tmp_path: Path, unsuccessful, cached_after):
    """A cached non-answer would be served to every rerun of the same query."""
    client = _client(tmp_path)
    params = {"unsuccessful_answer": unsuccessful, "response_delay": 0.0}

    first = client.research("What does hadA do?", provider="mock", provider_params=params)
    second = client.research("What does hadA do?", provider="mock", provider_params=params)

    assert first.answer_successful is (False if unsuccessful else None)
    assert second.cached is cached_after
    assert bool(list(tmp_path.glob("*.json"))) is cached_after
