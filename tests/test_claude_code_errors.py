"""Tests for Claude Code failure classification and its auth probe.

The CLI reports failures as prose rather than status codes, so these cover the
wordings that mean "stop" versus the ones that mean "try again".
"""

import asyncio
import json

import pytest

from deep_research_client.exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderNotInstalledError,
    ProviderQuotaError,
    ProviderTransientError,
)
from deep_research_client.models import ProviderConfig
from deep_research_client.providers.claude_code import (
    ClaudeCodeProvider,
    _classify_cli_failure,
)


def _provider(executable: str = "claude") -> ClaudeCodeProvider:
    """Build a provider pointed at the given executable."""
    from deep_research_client.provider_params import ClaudeCodeParams

    return ClaudeCodeProvider(
        ProviderConfig(name="claude_code", enabled=True),
        ClaudeCodeParams(claude_executable=executable),
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Claude usage limit reached. Your limit will reset at 3pm.", ProviderQuotaError),
        ("Your limit will reset at 11:00 (UTC)", ProviderQuotaError),
        ("Your credit balance is too low to access the API", ProviderBillingError),
        ("Invalid API key. Please run /login", ProviderAuthError),
        ("OAuth token has expired", ProviderAuthError),
        ("You are not logged in", ProviderAuthError),
        ("Model opus is not available on your plan", ProviderAuthError),
        ("This feature requires you to upgrade your plan", ProviderAuthError),
        ("API Error 529: Overloaded", ProviderTransientError),
    ],
)
def test_cli_wordings_map_to_types(text, expected):
    """Each CLI failure wording resolves to the type that implies the remedy."""
    assert isinstance(_classify_cli_failure("claude_code", text), expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Tool 'WebSearch' returned no results",
        "error_max_turns reached the maximum number of turns",
    ],
)
def test_unrecognised_output_is_left_alone(text):
    """Failures we cannot explain must not be given a confident wrong label."""
    assert _classify_cli_failure("claude_code", text) is None


def test_usage_limit_is_not_retryable_and_reports_its_reset():
    """The quota case is the one that carries a bounded wait, so surface it."""
    error = _classify_cli_failure(
        "claude_code", "Claude usage limit reached. Your limit will reset at 3pm (PST)."
    )

    assert error.retryable is False
    assert error.resets_at == "3pm (PST)"
    assert "renews at 3pm (PST)" in str(error)


def test_rate_limit_and_usage_limit_are_told_apart():
    """Both say "limit"; only one clears in seconds."""
    quota = _classify_cli_failure("claude_code", "Claude usage limit reached")
    overloaded = _classify_cli_failure("claude_code", "API Error 529: Overloaded")

    assert isinstance(quota, ProviderQuotaError)
    assert quota.retryable is False
    assert overloaded.retryable is True


def test_missing_cli_is_not_an_auth_failure():
    """No credential installs a binary, so this needs its own type."""
    provider = _provider(executable="definitely-not-a-real-binary")

    assert provider.is_available() is False
    with pytest.raises(ProviderNotInstalledError) as excinfo:
        asyncio.run(provider.research("anything"))

    assert "not found on PATH" in str(excinfo.value)


def test_error_result_event_is_classified():
    """A terminal error event carries the reason; classify it, don't flatten it."""
    stdout = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "subtype": "error_during_execution",
            "result": "Claude usage limit reached. Your limit will reset at 9pm.",
        }
    )

    with pytest.raises(ProviderQuotaError) as excinfo:
        ClaudeCodeProvider._parse_stream(stdout, "claude_code")

    assert excinfo.value.resets_at == "9pm"


def test_error_result_event_without_a_known_reason_still_raises():
    """An unrecognised error event keeps its original, informative message."""
    stdout = json.dumps(
        {"type": "result", "is_error": True, "subtype": "error_max_turns", "result": "gave up"}
    )

    with pytest.raises(ValueError, match="error_max_turns"):
        ClaudeCodeProvider._parse_stream(stdout, "claude_code")


@pytest.mark.parametrize(
    "status,reachable,expected_detail",
    [
        (
            {"loggedIn": True, "authMethod": "claude.ai", "subscriptionType": "max"},
            True,
            "auth: claude.ai, plan: max",
        ),
        ({"loggedIn": True}, True, "logged in"),
        ({"loggedIn": False}, False, "claude auth login"),
    ],
)
def test_auth_status_becomes_health(status, reachable, expected_detail):
    """The probe reports the plan, which is more than any HTTP provider offers."""
    health = _provider()._health_from_auth_status(json.dumps(status), "", 0)

    assert health.configured is True
    assert health.reachable is reachable
    assert expected_detail in health.detail


def test_failed_auth_status_is_classified():
    """A probe that exits non-zero is read the same way a run failure is."""
    health = _provider()._health_from_auth_status("", "Invalid API key. Please run /login", 1)

    assert health.reachable is False
    assert "API key" in health.detail


def test_unparseable_auth_status_is_unknown_not_unreachable():
    """Output we cannot read is not evidence the provider is down."""
    health = _provider()._health_from_auth_status("not json at all", "", 0)

    assert health.reachable is None
    assert "could not parse" in health.detail


@pytest.mark.integration
def test_live_auth_status_probe():
    """The real CLI answers the probe without spending tokens."""
    provider = _provider()
    if not provider.is_available():
        pytest.skip("claude CLI not installed")

    health = asyncio.run(provider.check_health())

    assert health.configured is True
    assert health.reachable is not None
