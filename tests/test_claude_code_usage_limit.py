"""A spent usage limit is classified, and an unexplained exit says what it saw (#127).

The CLI is replaced by a small executable that replays a recorded stream on a
real subprocess, so the provider's own process handling, classification and
error message are all exercised with no patching. The stream shapes follow a
real failing run: with an invalid API key, ``claude --print --output-format
stream-json --verbose`` exits 1 with empty stderr after writing a synthetic
assistant event marked ``is_api_error_message`` and then a ``result`` event with
``is_error: true``, both carrying the CLI's message. A spent limit was not
reproduced; its wordings are as quoted in monarch-initiative/dismech#10735.
"""

import asyncio
import json
import stat
import sys
from pathlib import Path

import pytest

from deep_research_client.exceptions import (
    ProviderBillingError,
    ProviderQuotaError,
    is_fallback_worthy,
)
from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import ClaudeCodeParams
from deep_research_client.providers.claude_code import (
    ClaudeCodeProvider,
    classify_cli_failure,
)

LIMIT = "You've hit your limit · resets 8pm (UTC)"


def _api_error_event(text: str, error: str = "rate_limit") -> dict:
    """The synthetic assistant event the CLI writes for a failed API call."""
    return {
        "type": "assistant",
        "is_api_error_message": True,
        "error": error,
        "message": {"model": "<synthetic>", "content": [{"type": "text", "text": text}]},
    }


def _error_result(text: str) -> dict:
    """The terminal event that normally follows it."""
    return {"type": "result", "subtype": "success", "is_error": True, "result": text}


def _report_event(text: str) -> dict:
    """An ordinary assistant event: the model's own words."""
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def _fake_cli(tmp_path: Path, stdout_lines: list[str], returncode: int = 1) -> Path:
    """An executable that ignores its arguments, writes ``stdout_lines`` and exits."""
    recorded = tmp_path / "stdout.txt"
    recorded.write_text("".join(f"{line}\n" for line in stdout_lines), encoding="utf-8")
    script = tmp_path / "claude"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stdin.read()\n"
        f"sys.stdout.write(open({str(recorded)!r}, encoding='utf-8').read())\n"
        f"sys.exit({returncode})\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _run(executable: Path) -> None:
    """Run a research call against the fake CLI."""
    provider = ClaudeCodeProvider(
        ProviderConfig(name="claude_code", enabled=True),
        ClaudeCodeParams(claude_executable=str(executable)),
    )
    asyncio.run(provider.research("anything"))


@pytest.mark.parametrize(
    ("text", "resets_at"),
    [
        ("You've hit your limit, resets 8:50pm", "8:50pm"),
        ("You've hit your limit · resets 8pm (UTC)", "8pm (UTC)"),
        ("5-hour limit reached ∙ resets 8:50pm", "8:50pm"),
        ("Claude usage limit reached. Your limit will reset at 3pm.", "3pm"),
    ],
)
def test_the_issues_wordings_are_a_spent_allowance(text: str, resets_at: str) -> None:
    """The three current wordings from the report, and the old one still."""
    error = classify_cli_failure("claude_code", text)

    assert isinstance(error, ProviderQuotaError)
    assert error.resets_at == resets_at
    assert is_fallback_worthy(error), "--fallback must apply to a spent allowance"


@pytest.mark.parametrize(
    "text",
    [
        "Context limit reached · /compact or /clear to continue",
        "Subagent nesting limit reached (depth 4)",
        "Concurrent export limit reached",
        "You've hit your rate limit, try again shortly",
        "You've hit your Opus rate limit",
        "You've hit your context window limit",
    ],
)
def test_limits_that_are_not_the_plans_allowance_stay_unclassified(text: str) -> None:
    """Other limits the CLI reports must not send a run to --fallback."""
    assert not isinstance(classify_cli_failure("claude_code", text), ProviderQuotaError)


def test_a_spent_limit_in_a_complete_stream_is_a_quota_error(tmp_path: Path) -> None:
    """The usual shape: an API-error event, then a failing result event."""
    cli = _fake_cli(tmp_path, [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps(_api_error_event(LIMIT)),
        json.dumps(_error_result(LIMIT)),
    ])

    with pytest.raises(ProviderQuotaError) as excinfo:
        _run(cli)

    assert excinfo.value.resets_at == "8pm (UTC)"


def test_a_spent_limit_with_no_result_event_is_a_quota_error(tmp_path: Path) -> None:
    """The reported case: exit 1, empty stderr, and the stream stops early."""
    cli = _fake_cli(tmp_path, [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps(_api_error_event(LIMIT)),
    ])

    with pytest.raises(ProviderQuotaError) as excinfo:
        _run(cli)

    assert excinfo.value.resets_at == "8pm (UTC)"


def test_an_unrecognised_result_event_is_quoted_not_dropped(tmp_path: Path) -> None:
    """A failing result event in wording no pattern knows still reaches the reader."""
    cli = _fake_cli(tmp_path, [json.dumps(_error_result("Something new went wrong"))])

    with pytest.raises(ValueError) as excinfo:
        _run(cli)

    # The "success" subtype a failed API call carries is not repeated.
    assert str(excinfo.value) == "Claude Code exited with code 1: Something new went wrong"


def test_a_plain_line_on_stdout_is_quoted(tmp_path: Path) -> None:
    """Empty stderr and no failure event: the CLI's own prose on stdout is the clue."""
    cli = _fake_cli(tmp_path, [
        json.dumps({"type": "system", "subtype": "init"}),
        "Unexpected failure while loading settings",
    ])

    with pytest.raises(ValueError, match="Unexpected failure while loading settings"):
        _run(cli)


def test_a_report_about_usage_limits_is_still_not_a_usage_limit(tmp_path: Path) -> None:
    """Widening the patterns must not make the model's prose count as evidence.

    The message still shows where the model had got to, so the reader is not
    left with nothing.
    """
    cli = _fake_cli(tmp_path, [
        json.dumps(_report_event(
            "Accounts that have hit your limit see 'usage limit reached'; "
            "spend limit reached means billing. Now checking PubMed"
        )),
    ], returncode=137)

    with pytest.raises(ValueError) as excinfo:
        _run(cli)

    assert not isinstance(excinfo.value, (ProviderQuotaError, ProviderBillingError))
    assert "137" in str(excinfo.value)
    assert "last text written:" in str(excinfo.value)
    assert "Now checking PubMed" in str(excinfo.value)


def test_an_api_error_the_run_recovered_from_is_not_blamed(tmp_path: Path) -> None:
    """A retried overload earlier in the stream did not end the run."""
    cli = _fake_cli(tmp_path, [
        json.dumps(_api_error_event("API Error: overloaded", error="server_error")),
        json.dumps(_report_event("Recovered and kept going")),
    ], returncode=137)

    with pytest.raises(ValueError) as excinfo:
        _run(cli)

    assert type(excinfo.value) is ValueError
    assert "last text written: Recovered and kept going" in str(excinfo.value)


def test_a_half_written_last_event_is_not_quoted(tmp_path: Path) -> None:
    """A killed run ends mid-event; the message skips the fragment."""
    cli = _fake_cli(tmp_path, [
        json.dumps(_report_event("Searching PubMed")),
        '{"type": "assistant", "message": {"content": [{"type": "te',
    ], returncode=137)

    with pytest.raises(ValueError) as excinfo:
        _run(cli)

    assert str(excinfo.value).endswith("last text written: Searching PubMed")


def test_a_silent_exit_says_so(tmp_path: Path) -> None:
    """Nothing at all on either stream is reported as such."""
    cli = _fake_cli(tmp_path, [])

    with pytest.raises(ValueError, match="exited with code 1: <no output>"):
        _run(cli)
