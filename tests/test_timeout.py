"""Tests for the caller-set provider timeout (``--timeout``, issue #95).

Nothing here is mocked. The shipped :class:`MockProvider` honours
``ProviderConfig.timeout`` for real -- its simulated API delay is bounded by
it -- so the client is exercised through its ordinary code path.
"""

import logging

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.client import (
    PROVIDER_CLASS_PATHS,
    DeepResearchClient,
    load_provider_class,
)
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.provider_params import MockParams
from deep_research_client.providers.mock import MockProvider

runner = CliRunner()

PRIMARY = "mock"
BACKUP = "mock_backup"


class StandInProvider(MockProvider):
    """A mock that may be fallen back to automatically."""

    produces_real_reports = True


class UntimedProvider(MockProvider):
    """A provider that, like ToolUniverse, honours no caller-set timeout."""

    timeout_meaning = None


def _client(*extra: MockProvider, primary_config: ProviderConfig | None = None) -> DeepResearchClient:
    """Build a client holding the mock provider plus any extra providers.

    Args:
        extra: Further provider instances to register, in order.
        primary_config: Config for the mock provider; a bare one by default.

    Returns:
        A client with no cache and nothing from the environment.
    """
    client = DeepResearchClient(
        cache_config=CacheConfig(enabled=False),
        provider_configs={PRIMARY: primary_config or ProviderConfig(name=PRIMARY)},
    )
    for provider in extra:
        client.registry.register(provider)
    return client


@pytest.mark.parametrize(
    "provider_name, honours_timeout",
    [
        ("openai", True),
        ("falcon", True),
        ("asta", True),
        ("perplexity", True),
        ("consensus", True),
        ("cyberian", True),
        ("openscientist", True),
        ("claude_code", True),
        ("biomni", True),
        ("toxindex", True),
        ("mock", True),
        # Rejects a whole-run timeout outright; it has request_timeout instead.
        ("tooluniverse", False),
        # A stub with no upstream service, so there is nothing to time out.
        ("deeper_med", False),
    ],
)
def test_each_provider_declares_what_its_timeout_limits(provider_name, honours_timeout):
    """Every provider says what the number means, or that it takes none."""
    meaning = load_provider_class(provider_name).timeout_meaning
    assert (meaning is not None) is honours_timeout
    if honours_timeout:
        assert meaning.strip()


def test_every_registered_provider_is_in_the_declaration_table():
    """A new provider must be added to the table above, not slip past it."""
    table = {
        "openai", "falcon", "asta", "perplexity", "consensus", "cyberian",
        "openscientist", "claude_code", "biomni", "toxindex", "mock",
        "tooluniverse", "deeper_med",
    }
    assert set(PROVIDER_CLASS_PATHS) == table


def test_timeout_reaches_the_provider_and_the_report():
    """The override is what the provider runs with and what the report records."""
    client = _client()
    result = client.research(
        "q", provider=PRIMARY, provider_params={"response_delay": 0.0}, timeout=7
    )
    assert result.provider_config["timeout"] == 7


def test_timeout_alone_is_enough_to_apply_it():
    """No model or parameters are needed for the timeout to take effect."""
    client = _client()
    result = client.research("q", provider=PRIMARY, timeout=7)
    assert result.provider_config["timeout"] == 7


def test_timeout_leaves_the_registered_provider_untouched():
    """One run's override must not become the next run's default."""
    client = _client(primary_config=ProviderConfig(name=PRIMARY, timeout=30))
    client.research("q", provider=PRIMARY, timeout=7)
    assert client.registry.get_provider(PRIMARY).config.timeout == 30
    result = client.research("q", provider=PRIMARY)
    assert result.provider_config["timeout"] == 30


def test_timeout_is_enforced():
    """A provider that takes longer than the timeout fails with TimeoutError."""
    client = _client()
    with pytest.raises(TimeoutError):
        client.research(
            "q", provider=PRIMARY, provider_params={"response_delay": 2.0}, timeout=1
        )


@pytest.mark.parametrize("timeout", [0, -5])
def test_non_positive_timeout_is_refused(timeout):
    """Zero or a negative number is a caller error, not "no limit"."""
    client = _client()
    with pytest.raises(ValueError, match="positive"):
        client.research("q", provider=PRIMARY, timeout=timeout)


def test_timeout_is_refused_by_a_provider_that_does_not_honour_one():
    """Recording a limit nothing enforced would mislead the report's reader."""
    untimed = UntimedProvider(ProviderConfig(name="untimed"), MockParams(response_delay=0.0))
    client = _client(untimed)
    with pytest.raises(ValueError, match="does not support a caller-set timeout"):
        client.research("q", provider="untimed", timeout=7)


def test_untimed_provider_still_runs_without_a_timeout():
    """The refusal is about the override, not the provider."""
    untimed = UntimedProvider(ProviderConfig(name="untimed"), MockParams(response_delay=0.0))
    client = _client(untimed)
    result = client.research("q", provider="untimed")
    assert result.provider_config["timeout"] is None


def test_timeout_is_not_carried_to_a_fallback(caplog):
    """A fallback runs on its own default, and the log says the override was dropped."""
    backup = StandInProvider(ProviderConfig(name=BACKUP, timeout=30), MockParams(response_delay=0.0))
    client = _client(backup)
    with caplog.at_level(logging.WARNING, logger="deep_research_client.client"):
        result = client.research(
            "q",
            provider=PRIMARY,
            provider_params={"error_type": "billing"},
            fallback=[BACKUP],
            timeout=7,
        )
    assert result.provider == BACKUP
    assert result.provider_config["timeout"] == 30
    assert "--timeout" in caplog.text


def test_cli_timeout_runs():
    """``--timeout`` is accepted and the run completes."""
    result = runner.invoke(
        app,
        ["research", "What is CRISPR?", "--provider", "mock", "--no-cache", "--timeout", "30"],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )
    assert result.exit_code == 0, result.output


def test_cli_timeout_is_enforced():
    """The CLI's number reaches the provider: too short a timeout fails the run."""
    result = runner.invoke(
        app,
        [
            "research", "What is CRISPR?", "--provider", "mock", "--no-cache",
            "--param", "response_delay=2", "--timeout", "1",
        ],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )
    assert result.exit_code == 1
    # TimeoutError subclasses OSError; it must not be called a filesystem error.
    assert "Research timed out" in result.stderr
    assert "Filesystem error" not in result.stderr


def test_cli_rejects_a_non_positive_timeout():
    """typer's own range check catches zero before any client is built."""
    result = runner.invoke(
        app,
        ["research", "What is CRISPR?", "--provider", "mock", "--timeout", "0"],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )
    assert result.exit_code == 2
