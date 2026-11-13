"""Tests for the caller-set provider timeout (``--timeout``, issue #95).

Nothing here is mocked. The shipped :class:`MockProvider` honours
``ProviderConfig.timeout`` for real -- its simulated API delay is bounded by
it -- so the client is exercised through its ordinary code path.
"""

import logging
from typing import Optional

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.client import (
    PROVIDER_CLASS_PATHS,
    DeepResearchClient,
    load_provider_class,
)
from deep_research_client.exceptions import ProviderTimeoutError, is_fallback_worthy
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.provider_params import MockParams
from deep_research_client.providers.claude_code import ClaudeCodeProvider
from deep_research_client.providers.mock import MockProvider
from deep_research_client.providers.openscientist import OpenScientistProvider

runner = CliRunner()

PRIMARY = "mock"
BACKUP = "mock_backup"


class StandInProvider(MockProvider):
    """A mock that may be fallen back to automatically."""

    produces_real_reports = True


class UntimedProvider(MockProvider):
    """A provider that, like ToolUniverse, honours no caller-set timeout."""

    timeout_meaning = None


class ParamTimeoutProvider(MockProvider):
    """A provider whose ``timeout`` parameter overrides the config timeout.

    Stands in for Biomni and OpenScientist, where passing both ``--timeout``
    and ``--param timeout=`` is ambiguous and rejected.
    """

    timeout_param = "timeout"


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


# Whether each registered provider honours a caller-set timeout. One list, so
# the parametrize below and the completeness check cannot drift, and adding a
# provider forces an entry here. False = the provider refuses or has nothing to
# time out (tooluniverse rejects a whole-run timeout and uses request_timeout
# instead; deeper_med is a stub with no upstream service).
TIMEOUT_TABLE: dict[str, bool] = {
    "openai": True,
    "falcon": True,
    "asta": True,
    "perplexity": True,
    "consensus": True,
    "cyberian": True,
    "openscientist": True,
    "claude_code": True,
    "biomni": True,
    "toxindex": True,
    "mock": True,
    "tooluniverse": False,
    "deeper_med": False,
}


@pytest.mark.parametrize("provider_name, honours_timeout", TIMEOUT_TABLE.items())
def test_each_provider_declares_what_its_timeout_limits(provider_name, honours_timeout):
    """Every provider says what the number means, or that it takes none."""
    meaning = load_provider_class(provider_name).timeout_meaning
    assert (meaning is not None) is honours_timeout
    if honours_timeout:
        assert meaning.strip()


def test_every_registered_provider_is_in_the_declaration_table():
    """A new provider must be added to TIMEOUT_TABLE, not slip past it."""
    assert set(PROVIDER_CLASS_PATHS) == set(TIMEOUT_TABLE)


# Which providers expose a ``timeout`` provider-parameter that sets the same
# limit as ``--timeout`` -- so giving both is ambiguous and rejected. Only
# these two; every other provider must declare no ``timeout_param`` (None).
# Pinning it here means removing ``timeout_param = "timeout"`` from a provider,
# or adding a stray one elsewhere, fails this test instead of silently letting
# ``--timeout`` and ``--param timeout=`` through together.
TIMEOUT_PARAM_TABLE: dict[str, Optional[str]] = {
    "biomni": "timeout",
    "openscientist": "timeout",
}


@pytest.mark.parametrize("provider_name", sorted(PROVIDER_CLASS_PATHS))
def test_each_provider_declares_its_real_timeout_param(provider_name):
    """Each provider's declared ``timeout_param`` matches the table exactly."""
    expected = TIMEOUT_PARAM_TABLE.get(provider_name)
    assert load_provider_class(provider_name).timeout_param == expected


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


# --------------------------------------------------------------------------
# ProviderTimeoutError: a timeout reaches the CLI's timeout path, not ValueError
# --------------------------------------------------------------------------


def test_provider_timeout_error_is_a_timeout_not_fallback_worthy():
    """Both a TimeoutError (for the CLI) and a ValueError (to carry the trail),
    and never a reason to switch provider."""
    err = ProviderTimeoutError("mock", "no response within 600s")
    assert isinstance(err, TimeoutError)
    assert isinstance(err, ValueError)
    assert not is_fallback_worthy(err)


def test_timeout_error_type_reaches_the_cli_timeout_message():
    """A provider that times out is reported as a timeout, pointing at --timeout.

    The mock's ``error_type=timeout`` raises ProviderTimeoutError, which is both
    a ValueError and a TimeoutError -- so this also pins that the CLI's
    ``except TimeoutError`` sits before its ``except ValueError``.
    """
    result = runner.invoke(
        app,
        [
            "research", "What is CRISPR?", "--provider", "mock", "--no-cache",
            "--param", "error_type=timeout",
        ],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )
    assert result.exit_code == 1
    assert "Research timed out" in result.stderr
    assert "--timeout" in result.stderr
    assert "Filesystem error" not in result.stderr
    # ProviderTimeoutError names --timeout in its own remedy, so the CLI must
    # not append a second hint: the user should see it exactly once.
    assert result.stderr.count("--timeout") == 1
    # And no provider-switch suggestion, which would contradict the reason a
    # timeout is deliberately not fallback-worthy (another provider cannot beat
    # the caller's own deadline).
    assert "re-run with --provider" not in result.stderr


def test_cli_refuses_timeout_for_an_unsupported_provider_cleanly(monkeypatch):
    """--timeout against a provider that honours none exits 1 with a clear error,
    not a traceback."""
    # Make the shipped mock declare that it honours no caller-set timeout, the
    # way ToolUniverse does, so the CLI reaches the client's refusal path.
    monkeypatch.setattr(MockProvider, "timeout_meaning", None)
    result = runner.invoke(
        app,
        ["research", "What is CRISPR?", "--provider", "mock", "--no-cache", "--timeout", "5"],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )
    assert result.exit_code == 1
    assert "does not support a caller-set timeout" in result.stderr
    assert "Traceback" not in result.output


def test_cli_v_logs_the_timeout_meaning():
    """`-v` names what the number limits for the chosen provider."""
    result = runner.invoke(
        app,
        ["-v", "research", "What is CRISPR?", "--provider", "mock", "--no-cache", "--timeout", "30"],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )
    assert result.exit_code == 0, result.output
    assert "Timeout 30s for mock" in result.output


# --------------------------------------------------------------------------
# Real providers wire ProviderConfig.timeout through (no network)
# --------------------------------------------------------------------------


def test_claude_code_passes_config_timeout_through():
    """ClaudeCodeProvider uses ProviderConfig.timeout over its param default."""
    provider = ClaudeCodeProvider(ProviderConfig(name="claude_code", timeout=7))
    assert provider.timeout == 7


def test_openscientist_config_timeout_reaches_its_param():
    """OpenScientist copies ProviderConfig.timeout onto its validated params."""
    provider = OpenScientistProvider(
        ProviderConfig(name="openscientist", api_key="x", timeout=120)
    )
    assert provider.params.timeout == 120


@pytest.mark.parametrize("bad", [10, 100000])
def test_openscientist_out_of_range_config_timeout_fails_fast(bad):
    """A --timeout outside OpenScientistParams' ge=60/le=7200 is rejected on
    construction, not smuggled past validation by model_copy."""
    with pytest.raises(ValidationError):
        OpenScientistProvider(ProviderConfig(name="openscientist", api_key="x", timeout=bad))


# --------------------------------------------------------------------------
# Passing both --timeout and the overriding param is rejected
# --------------------------------------------------------------------------


def test_both_timeout_and_overriding_param_is_rejected():
    """For a provider whose param wins, giving both would record a timeout that
    did not apply, so the client refuses the combination."""
    provider = ParamTimeoutProvider(ProviderConfig(name="untimed"), MockParams(response_delay=0.0))
    client = _client(provider)
    with pytest.raises(ValueError, match="ambiguous"):
        client.research(
            "q", provider="untimed", timeout=5, provider_params={"timeout": 10}
        )
