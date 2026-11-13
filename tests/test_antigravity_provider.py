"""Unit tests for the Google Antigravity (agy CLI) research provider.

These exercise the provider without the real ``agy`` binary by monkeypatching
``_run_process`` (the subprocess boundary), the same pattern the claude_code
tests use. A live run is covered by ``test_antigravity_integration.py``.
"""

import asyncio
import json

import pytest

from deep_research_client.exceptions import (
    ProviderAuthError,
    ProviderQuotaError,
    ProviderTimeoutError,
    ProviderTransientError,
)
from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import AntigravityParams
from deep_research_client.providers.antigravity import (
    AntigravityProvider,
    classify_agy_failure,
    parse_agy_error,
)


def _provider(params: AntigravityParams | None = None) -> AntigravityProvider:
    return AntigravityProvider(ProviderConfig(name="antigravity"), params)


def _stream(*, response: str, status: str = "SUCCESS", tools=(), model="gemini-3.8-flash-medium"):
    """Assemble an agy stream-json stdout from its parts.

    Each tool is emitted as an ACTIVE then a DONE event sharing a step_index,
    the way the real CLI does, so the parser's deduplication is exercised.
    """
    lines = [json.dumps({"event": "init", "init": {
        "cwd": "/x", "model": model, "permission_mode": "request-review",
        "tools": ["search_web", "read_url_content", "write_file"],
    }})]
    lines.append(json.dumps({"event": "step_update", "step_update": {
        "step_type": "user_input", "step_index": 0, "state": "DONE"}}))
    for i, tool in enumerate(tools, start=1):
        for state in ("ACTIVE", "DONE"):
            lines.append(json.dumps({"event": "step_update", "step_update": {
                "step_type": "tool", "tool_name": tool, "step_index": i,
                "state": state, "tool_info": {"name": tool, "parameters": {}}}}))
    lines.append(json.dumps({"event": "result", "result": {
        "status": status, "response": response, "num_turns": 1,
        "duration_seconds": 11.5, "conversation_id": "conv-123",
        "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}}}))
    return "\n".join(lines)


def _patch_run(monkeypatch, provider, stdout="", stderr="", returncode=0):
    async def _fake_run(command, stdin_input=None):
        return stdout, stderr, returncode
    monkeypatch.setattr(provider, "is_available", lambda: True)
    monkeypatch.setattr(provider, "_run_process", _fake_run)


# --------------------------------------------------------------------------- #
# Stream parsing
# --------------------------------------------------------------------------- #

def test_parse_stream_reads_report_tools_and_model():
    stdout = _stream(
        response="A cited report ([src](https://example.org/a)).",
        tools=("search_web", "read_url_content"),
    )
    parsed = AntigravityProvider._parse_stream(stdout)
    assert parsed.response.startswith("A cited report")
    assert parsed.model == "gemini-3.8-flash-medium"
    assert parsed.result["status"] == "SUCCESS"
    # Each tool's ACTIVE/DONE pair collapses to one call.
    assert parsed.tool_uses == ["search_web", "read_url_content"]


def test_parse_stream_empty_output_raises():
    with pytest.raises(ValueError, match="empty output"):
        AntigravityProvider._parse_stream("   ")


def test_parse_stream_without_result_event_raises():
    only_init = json.dumps({"event": "init", "init": {"model": "m"}})
    with pytest.raises(ValueError, match="no terminal 'result' event"):
        AntigravityProvider._parse_stream(only_init)


def test_parse_stream_skips_unparseable_lines():
    stdout = "not json\n" + _stream(response="x" * 10, tools=("search_web",))
    parsed = AntigravityProvider._parse_stream(stdout)
    assert parsed.tool_uses == ["search_web"]


# --------------------------------------------------------------------------- #
# Command building
# --------------------------------------------------------------------------- #

def test_build_command_defaults():
    cmd = _provider()._build_command("PROMPT")
    assert cmd[:5] == ["agy", "-p", "PROMPT", "--output-format", "stream-json"]
    assert "--disable-slash-commands" in cmd
    assert "--dangerously-skip-permissions" not in cmd
    assert "--model" not in cmd  # default sentinel -> let agy choose


def test_build_command_forwards_options():
    params = AntigravityParams(
        model="gemini-4-carbon-high", effort="high", skip_permissions=True,
        disable_slash_commands=False, print_timeout="15m",
        settings_override='{"a":1}', add_dirs=["/tmp/x"], extra_args=["--sandbox"],
    )
    cmd = _provider(params)._build_command("P")
    assert ["--model", "gemini-4-carbon-high"] == cmd[cmd.index("--model"):cmd.index("--model") + 2]
    assert ["--effort", "high"] == cmd[cmd.index("--effort"):cmd.index("--effort") + 2]
    assert "--dangerously-skip-permissions" in cmd
    assert "--disable-slash-commands" not in cmd
    assert ["--print-timeout", "15m"] == cmd[cmd.index("--print-timeout"):cmd.index("--print-timeout") + 2]
    assert ["--settings", '{"a":1}'] == cmd[cmd.index("--settings"):cmd.index("--settings") + 2]
    assert ["--add-dir", "/tmp/x"] == cmd[cmd.index("--add-dir"):cmd.index("--add-dir") + 2]
    assert cmd[-1] == "--sandbox"


@pytest.mark.parametrize("sentinel", ["antigravity", "agy", "default", "ANTIGRAVITY"])
def test_default_sentinels_do_not_forward_model(sentinel):
    provider = _provider(AntigravityParams(model=sentinel))
    assert provider._resolve_cli_model() is None


def test_real_model_slug_is_forwarded():
    provider = _provider(AntigravityParams(model="gemini-3.1-pro-high"))
    assert provider._resolve_cli_model() == "gemini-3.1-pro-high"


def test_compose_prompt_includes_directive_and_task():
    prompt = _provider()._compose_prompt("What is CRISPR?")
    assert "Research task:\nWhat is CRISPR?" in prompt
    assert "write the report inline" in prompt  # the inline-report directive


# --------------------------------------------------------------------------- #
# Retrieval classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tool,expected", [
    ("search_web", True),
    ("read_url_content", True),
    ("open_browser_url", True),
    ("write_file", False),
    ("run_command", False),
    ("finish", False),
])
def test_is_retrieval_tool(tool, expected):
    assert AntigravityProvider._is_retrieval_tool(tool) is expected


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #

def test_parse_agy_error_extracts_structured_object():
    stderr = 'noise\nAGY_ERROR: {"status":"DEADLINE_EXCEEDED","reason":"print_timeout"}\n'
    assert parse_agy_error(stderr)["reason"] == "print_timeout"


def test_parse_agy_error_none_when_absent():
    assert parse_agy_error("just prose, no structured error") == {}


@pytest.mark.parametrize("status,error_type", [
    ("UNAUTHENTICATED", ProviderAuthError),
    ("PERMISSION_DENIED", ProviderAuthError),
    ("RESOURCE_EXHAUSTED", ProviderQuotaError),
    ("DEADLINE_EXCEEDED", ProviderTimeoutError),
    ("UNAVAILABLE", ProviderTransientError),
])
def test_classify_agy_failure_by_status(status, error_type):
    stderr = f'AGY_ERROR: {{"status":"{status}","short_error":"boom"}}'
    err = classify_agy_failure("antigravity", stderr)
    assert isinstance(err, error_type)


def test_classify_agy_failure_text_fallback():
    err = classify_agy_failure("antigravity", "authentication required: run agy to log in")
    assert isinstance(err, ProviderAuthError)


def test_classify_agy_failure_unrecognised_is_none():
    assert classify_agy_failure("antigravity", "some tool returned nothing useful") is None


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

def test_health_from_models_success():
    provider = _provider()
    stdout = "Fetching available models...\ngemini-4-carbon-high\tGemini 4 Carbon (High)\ngemini-3.8-flash-medium\tGemini 3.8 Flash (Medium)\n"
    health = provider._health_from_models(stdout, "", 0)
    assert health.reachable is True
    assert "2 models" in health.detail


def test_health_from_models_no_models_is_unreachable():
    provider = _provider()
    health = provider._health_from_models("Fetching available models...\n", "", 0)
    assert health.reachable is False


def test_health_from_models_nonzero_classifies():
    provider = _provider()
    health = provider._health_from_models("", 'AGY_ERROR: {"status":"UNAUTHENTICATED"}', 1)
    assert health.reachable is False


# --------------------------------------------------------------------------- #
# research() end to end (subprocess stubbed)
# --------------------------------------------------------------------------- #

def test_research_success(monkeypatch):
    provider = _provider()
    stdout = _stream(
        response="A thorough, cited research report about the topic. " * 5
        + "See https://example.org/evidence.",
        tools=("search_web", "read_url_content"),
    )
    _patch_run(monkeypatch, provider, stdout=stdout)

    result = asyncio.run(provider.research("explain the topic"))
    assert result.provider == "antigravity"
    assert result.answer_successful is None  # a real researched report
    assert "https://example.org/evidence" in result.citations
    assert result.model == "gemini-3.8-flash-medium"
    assert result.run_metadata["retrieval_calls"] == 2
    assert result.run_metadata["conversation_id"] == "conv-123"


def test_research_error_status_raises(monkeypatch):
    provider = _provider()
    stdout = _stream(response="", status="ERROR")
    stderr = 'AGY_ERROR: {"status":"UNAUTHENTICATED","short_error":"log in first"}'
    _patch_run(monkeypatch, provider, stdout=stdout, stderr=stderr, returncode=1)

    with pytest.raises(ProviderAuthError):
        asyncio.run(provider.research("anything"))


def test_research_empty_response_is_rejected_by_length_guard(monkeypatch):
    """A soft-denied final tool can leave a SUCCESS run with an empty response."""
    provider = _provider()
    stdout = _stream(response="", tools=("search_web",))
    _patch_run(monkeypatch, provider, stdout=stdout,
               stderr='... the "write_file" permission ... was auto-denied.')

    with pytest.raises(ValueError, match="min_report_chars"):
        asyncio.run(provider.research("anything"))


def test_research_from_memory_is_flagged_unsuccessful(monkeypatch):
    """No retrieval tool calls: the text was written from memory, not researched."""
    provider = _provider()
    stdout = _stream(
        response="A long answer written entirely from the model's own memory, "
        "with no sources consulted and nothing fetched during the run at all. " * 3,
        tools=(),  # nothing retrieved
    )
    _patch_run(monkeypatch, provider, stdout=stdout)

    result = asyncio.run(provider.research("anything"))
    assert result.answer_successful is False
    assert "from memory" in result.run_metadata["unsuccessful_reason"]


def test_research_empty_query_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(_provider().research("   "))
