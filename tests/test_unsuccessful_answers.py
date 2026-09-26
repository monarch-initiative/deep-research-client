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


def test_with_validation_requested_the_run_exits_4_before_validating():
    """A non-answer is not validated: nothing in it is worth checking.

    Validation of a report printed to stdout appends its own section there,
    so its absence shows validation never ran -- which is also what keeps a
    validation failure (exit 3) from masking this one.
    """
    result = _run_research(
        "--param", "unsuccessful_answer=true",
        "--validate-references", "--fail-on-unresolved",
    )

    assert result.exit_code == 4, result.output
    assert "Warning: mock could not answer this question" in result.stderr
    assert "Reference Validation" not in result.stdout


def test_an_unsuccessful_answer_does_not_fall_back(tmp_path: Path):
    """The run completed, so the next candidate is not billed for the same question."""
    from deep_research_client.provider_params import MockParams
    from deep_research_client.providers.mock import MockProvider

    client = _client(tmp_path)
    client.registry.register(
        MockProvider(ProviderConfig(name="mock_backup"), MockParams(response_delay=0.0))
    )

    result = client.research(
        "What does hadA do?",
        provider="mock",
        provider_params={"unsuccessful_answer": True, "response_delay": 0.0},
        fallback=["mock_backup"],
    )

    assert result.provider == "mock"
    assert result.answer_successful is False
    assert not result.fell_back


def test_edison_trajectory_writes_the_file_and_exits_4(tmp_path: Path, monkeypatch):
    """The path where typer.Exit would have been swallowed by the catch-all.

    Substitutes the Edison client at its source, as the falcon provider tests
    do: retrieving a real failed trajectory needs a live key and a task id.
    """
    pytest.importorskip("edison_client")
    from datetime import datetime

    from edison_client.models.app import TaskResponseVerbose
    from typer.testing import CliRunner

    from deep_research_client.cli import app

    failed = TaskResponseVerbose.model_construct(
        status="success", query="What does hadA do?", user=None,
        created_at=datetime.now(), job_name="job-futurehouse-paperqa3",
        share_status="private", permitted_accessors=None, build_owner=None,
        environment_name=None, agent_name=None, task_id=None, project_id=None,
        agent_state=None, metadata=None, deployment_config=None,
        environment_frame={"state": {"state": {"response": {"answer": {
            "formatted_answer": "Question: What does hadA do?\n\nNo papers were found.",
            "has_successful_answer": False,
        }}}}},
    )

    class FailedTrajectoryClient:
        """Returns one finished, unsuccessful trajectory."""

        def __init__(self, api_key: str):
            self.api_key = api_key

        def get_task(self, task_id: str, verbose: bool):
            return failed

        def close(self) -> None:
            pass

    monkeypatch.setattr("edison_client.EdisonClient", FailedTrajectoryClient)
    output = tmp_path / "trajectory.md"

    result = CliRunner().invoke(
        app,
        ["edison-trajectory", "784d73d5-da42-402e-9701-6c5b44beab14", "--output", str(output)],
        env={"EDISON_API_KEY": "test-key"},
    )

    assert result.exit_code == 4, result.output
    assert "Warning: falcon could not answer this question" in result.stderr
    content = output.read_text(encoding="utf-8")
    assert "answer_status: unsuccessful" in content
    assert "Question: What does hadA do?" not in content, "the echo is stripped here too"
