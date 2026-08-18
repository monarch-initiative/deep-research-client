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


def test_the_models_own_report_is_never_evidence(monkeypatch):
    """A report that discusses usage limits is not a usage limit.

    The stream carries the model's prose, which contains the same phrases a
    real failure does. Only the CLI's own terminal event counts.
    """
    provider = _provider()
    monkeypatch.setattr(provider, "is_available", lambda: True)

    stream = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "# Report\n\nWhen an account is out of credits the API "
                                    "returns 'Claude usage limit reached' and the credit "
                                    "balance is too low to continue."
                                ),
                            }
                        ]
                    },
                }
            ),
            json.dumps({"type": "result", "subtype": "success", "result": "done"}),
        ]
    )

    async def _fake_run(command, query):
        return stream, "", 137  # killed, for a reason the CLI never explained

    monkeypatch.setattr(provider, "_run_process", _fake_run)

    with pytest.raises(ValueError) as excinfo:
        asyncio.run(provider.research("how does API billing work"))

    assert not isinstance(excinfo.value, ProviderQuotaError)
    assert not isinstance(excinfo.value, ProviderBillingError)
    assert "137" in str(excinfo.value)


def test_a_real_failure_on_stderr_is_still_caught(monkeypatch):
    """Ignoring the report must not cost us the CLI's own words."""
    provider = _provider()
    monkeypatch.setattr(provider, "is_available", lambda: True)

    async def _fake_run(command, query):
        return "", "Claude usage limit reached. Your limit will reset at 5pm.", 1

    monkeypatch.setattr(provider, "_run_process", _fake_run)

    with pytest.raises(ProviderQuotaError) as excinfo:
        asyncio.run(provider.research("anything"))

    assert excinfo.value.resets_at == "5pm"


@pytest.mark.parametrize(
    "stdout,expected",
    [
        ('{"type": "result", "subtype": "err", "result": "boom"}', "err boom"),
        ('{"type": "assistant"}\n{"type": "result", "result": "late"}', "late"),
        ("garbage\n{not json}", ""),
        ("", ""),
    ],
)
def test_terminal_result_text_reads_only_the_terminal_event(stdout, expected):
    """Malformed or truncated streams must not throw on the failure path."""
    from deep_research_client.providers.claude_code import _terminal_result_text

    assert _terminal_result_text(stdout) == expected


def test_a_cli_too_old_to_probe_is_unknown_not_unreachable():
    """Missing a subcommand says nothing about whether the provider works."""
    health = _provider()._health_from_auth_status(
        "", "error: unknown command 'auth'\nUsage: claude [options]", 1
    )

    assert health.reachable is None
    assert health.configured is True
    assert "no `auth status` subcommand" in health.detail


def test_a_probe_that_cannot_exec_returns_a_record_not_an_exception(monkeypatch):
    """`check_health` promises a health record, so a PATH race must not raise.

    `is_available()` checks PATH a moment earlier, but the file can vanish or
    lose its exec bit in between -- and a programmatic caller gets a raw OSError
    instead of the record the signature promises.
    """
    provider = _provider()
    monkeypatch.setattr(provider, "is_available", lambda: True)

    async def _no_such_binary(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "claude")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _no_such_binary)

    health = asyncio.run(provider.check_health())

    assert health.reachable is False
    assert "could not run" in health.detail


def test_a_long_stderr_does_not_become_the_whole_message():
    """A stack trace on stderr is a hint, not a payload to reprint."""
    from deep_research_client.exceptions import MAX_DETAIL_CHARS

    noise = "at Object.<anonymous> (/usr/lib/node_modules/claude/cli.js:1:1)\n" * 50
    error = _classify_cli_failure("claude_code", f"Claude usage limit reached\n{noise}")

    assert error is not None
    assert len(error.detail) <= MAX_DETAIL_CHARS


@pytest.mark.parametrize(
    "stderr",
    [
        "error: unknown command 'auth'",
        "See 'claude --help' for more information",
        "Usage: claude [options] [command]",
        "unrecognized subcommand",
    ],
)
def test_old_cli_wordings_are_read_as_unknown(stderr):
    """Every way a CLI says "no such subcommand" means UNKNOWN, not UNREACHABLE."""
    health = _provider()._health_from_auth_status("", stderr, 1)

    assert health.reachable is None
