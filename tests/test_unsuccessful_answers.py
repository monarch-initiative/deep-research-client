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


def _run_research(*extra: str):
    """Invoke `research` with the mock provider and no cache."""
    from typer.testing import CliRunner

    from deep_research_client.cli import app

    return CliRunner().invoke(
        app,
        ["research", "What does hadA do?", "--provider", "mock", "--no-cache",
         "--param", "response_delay=0", *extra],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )


def test_the_cli_writes_the_file_marks_it_and_exits_4(tmp_path: Path):
    """Kept, because the run's output has value; flagged three ways, because it is not a report."""
    from deep_research_client.cli import EXIT_UNSUCCESSFUL_ANSWER

    output = tmp_path / "hadA.md"

    result = _run_research("--param", "unsuccessful_answer=true", "--output", str(output))

    assert result.exit_code == EXIT_UNSUCCESSFUL_ANSWER == 4
    assert "Warning: mock could not answer this question" in result.stderr
    content = output.read_text(encoding="utf-8")
    assert "answer_status: unsuccessful" in content
    assert "citation_count: 0" in content
    assert "> **Warning:** mock reported that it could not answer" in content


def test_printed_to_stdout_the_warning_line_travels_with_the_report():
    """Redirected, the report still carries its own warning; the run's goes to stderr."""
    result = _run_research("--param", "unsuccessful_answer=true")

    assert result.exit_code == 4
    assert "> **Warning:** mock reported that it could not answer" in result.stdout
    assert "Warning: mock could not answer this question" in result.stderr
    assert "Warning: mock could not answer this question" not in result.stdout


def test_control_an_ordinary_run_still_exits_0():
    """The mock says nothing about success by default, so nothing changes for it."""
    result = _run_research()

    assert result.exit_code == 0, result.output
    assert "answer_status" not in result.stdout
    assert "could not answer" not in result.output
