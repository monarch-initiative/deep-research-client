"""Tests for the Claude Code provider.

These focus on the pure, side-effect-free helpers (command building, output
parsing, citation extraction, availability) so they run without invoking the
real ``claude`` CLI. A single integration test exercises an actual run and is
skipped automatically when the CLI is unavailable.
"""

import json
import shutil

import pytest

from deep_research_client.providers.claude_code import (
    ClaudeCodeProvider,
    _INLINE_REPORT_DIRECTIVE,
)
from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import ClaudeCodeParams
from deep_research_client.system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT


def make_provider(params: ClaudeCodeParams | None = None) -> ClaudeCodeProvider:
    """Build a provider with a standard test config."""
    config = ProviderConfig(name="claude_code", api_key=None, enabled=True)
    return ClaudeCodeProvider(config, params)


def test_default_model():
    """The provider advertises its sentinel default model."""
    provider = make_provider()
    assert provider.get_default_model() == "claude-code-default"
    assert provider.model == "claude-code-default"


def test_is_available_false_when_executable_missing():
    """An unresolvable executable name means the provider is unavailable."""
    provider = make_provider(
        ClaudeCodeParams(claude_executable="definitely-not-a-real-binary-xyz")
    )
    assert provider.is_available() is False


def test_is_available_false_when_disabled():
    """A disabled config is never available, even if the CLI exists."""
    config = ProviderConfig(name="claude_code", api_key=None, enabled=False)
    provider = ClaudeCodeProvider(config)
    assert provider.is_available() is False


def make_stream(*events: dict) -> str:
    """Render events as the JSON Lines stream `claude` writes to stdout."""
    return "\n".join(json.dumps(event) for event in events)


def assistant_event(*texts: str) -> dict:
    """Build an assistant event carrying the given text blocks."""
    return {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text} for text in texts]},
    }


def test_build_command_defaults():
    """Default command runs print mode, streamed JSON output, and a read-only toolset."""
    provider = make_provider()
    command = provider._build_command()

    assert command[0] == "claude"
    assert "--print" in command
    # stream-json (and its required --verbose) keeps every assistant message;
    # the plain "json" format exposes only the final one.
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in command
    # Secure default: permissions are NOT skipped, and the agent is restricted to
    # a read-only research toolset via the allowlist.
    assert "--dangerously-skip-permissions" not in command
    assert command[command.index("--allowedTools") + 1] == "WebSearch,WebFetch"
    # The research system prompt plus the inline-report directive are appended.
    assert "--append-system-prompt" in command
    appended = command[command.index("--append-system-prompt") + 1]
    assert appended.startswith(DEFAULT_RESEARCH_SYSTEM_PROMPT)
    assert _INLINE_REPORT_DIRECTIVE in appended
    # No model is forwarded for the default sentinel.
    assert "--model" not in command


def test_build_command_skip_permissions_opt_in():
    """Opting into skip_permissions adds the dangerous flag (for sandboxes)."""
    provider = make_provider(ClaudeCodeParams(skip_permissions=True))
    command = provider._build_command()
    assert "--dangerously-skip-permissions" in command


def test_build_command_forwards_explicit_model():
    """A real model string is forwarded verbatim to --model."""
    provider = make_provider(ClaudeCodeParams(model="opus"))
    command = provider._build_command()
    assert command[command.index("--model") + 1] == "opus"


@pytest.mark.parametrize("alias", ["claude", "claude-code", "cc", "default", "claude-code-default"])
def test_build_command_omits_model_for_sentinel_aliases(alias):
    """Provider-internal aliases should not be forwarded as a real model."""
    provider = make_provider(ClaudeCodeParams(model=alias))
    assert "--model" not in provider._build_command()


def test_build_command_custom_system_prompt():
    """A custom system prompt overrides the default research prompt."""
    provider = make_provider(ClaudeCodeParams(system_prompt="Be terse."))
    command = provider._build_command()
    appended = command[command.index("--append-system-prompt") + 1]
    assert appended.startswith("Be terse.")
    assert _INLINE_REPORT_DIRECTIVE in appended


def test_build_command_permission_mode_when_not_skipping():
    """When not skipping permissions, the permission mode is passed instead."""
    provider = make_provider(
        ClaudeCodeParams(skip_permissions=False, permission_mode="plan")
    )
    command = provider._build_command()
    assert "--dangerously-skip-permissions" not in command
    assert command[command.index("--permission-mode") + 1] == "plan"


def test_build_command_allowed_tools_and_dirs_and_extra_args():
    """Allowed tools, add-dirs, and extra args are all forwarded."""
    provider = make_provider(
        ClaudeCodeParams(
            allowed_tools=["WebSearch", "WebFetch"],
            add_dirs=["/data/a", "/data/b"],
            extra_args=["--max-turns", "30"],
        )
    )
    command = provider._build_command()
    assert command[command.index("--allowedTools") + 1] == "WebSearch,WebFetch"
    # Two --add-dir flags, one per directory.
    assert command.count("--add-dir") == 2
    assert "/data/a" in command and "/data/b" in command
    # extra_args are appended verbatim at the end.
    assert command[-2:] == ["--max-turns", "30"]


def test_parse_stream_and_report_text_success():
    """A successful stream parses and yields the markdown report."""
    stdout = make_stream(
        assistant_event("# Report\n\nbody"),
        {"type": "result", "is_error": False, "result": "# Report\n\nbody"},
    )
    texts, data = ClaudeCodeProvider._parse_stream(stdout)
    assert ClaudeCodeProvider._report_text(texts, data) == "# Report\n\nbody"


def test_report_text_joins_all_assistant_messages():
    """Regression: a trailing message must not replace the report (#59).

    The plain ``json`` output format exposes only the agent's final message, so a
    run that wrote its report and then signed off used to yield just the sign-off.
    """
    stdout = make_stream(
        {"type": "system", "subtype": "init"},
        assistant_event("# Report\n\nThe substantive findings."),
        {"type": "user", "message": {"content": [{"type": "tool_result"}]}},
        assistant_event("One correction: nothing else changes."),
        # The terminal event's `result` carries only that final message.
        {"type": "result", "is_error": False, "result": "One correction: nothing else changes."},
    )
    texts, data = ClaudeCodeProvider._parse_stream(stdout)
    assert len(texts) == 2

    report = ClaudeCodeProvider._report_text(texts, data)
    assert "The substantive findings." in report
    assert "One correction: nothing else changes." in report


def test_parse_stream_ignores_non_text_blocks():
    """Thinking and tool_use blocks are not part of the report."""
    stdout = make_stream(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "internal reasoning"},
                    {"type": "text", "text": "visible answer"},
                    {"type": "tool_use", "name": "WebSearch", "input": {}},
                ]
            },
        },
        {"type": "result", "is_error": False, "result": "visible answer"},
    )
    texts, _ = ClaudeCodeProvider._parse_stream(stdout)
    assert texts == ["visible answer"]


def test_parse_stream_skips_unparseable_lines():
    """A stray non-JSON line does not sink an otherwise complete run."""
    stdout = "\n".join([
        "warning: something noisy",
        json.dumps(assistant_event("the report")),
        json.dumps({"type": "result", "is_error": False, "result": "the report"}),
    ])
    texts, data = ClaudeCodeProvider._parse_stream(stdout)
    assert texts == ["the report"]
    assert data["type"] == "result"


def test_parse_stream_raises_on_error_flag():
    """An is_error result raises a ValueError with the detail."""
    stdout = make_stream(
        {"type": "result", "is_error": True, "subtype": "error_max_turns", "result": "hit limit"}
    )
    with pytest.raises(ValueError, match="error_max_turns"):
        ClaudeCodeProvider._parse_stream(stdout)


def test_parse_stream_raises_on_invalid_json():
    """Non-JSON output raises a ValueError."""
    with pytest.raises(ValueError, match="parse Claude Code JSON"):
        ClaudeCodeProvider._parse_stream("not json at all")


def test_parse_stream_raises_when_run_did_not_complete():
    """A stream cut short before the terminal result event raises."""
    stdout = make_stream(assistant_event("partial output"))
    with pytest.raises(ValueError, match="did not complete"):
        ClaudeCodeProvider._parse_stream(stdout)


def test_parse_stream_raises_on_empty_output():
    """Empty stdout raises a ValueError."""
    with pytest.raises(ValueError, match="empty output"):
        ClaudeCodeProvider._parse_stream("   ")


def test_report_text_falls_back_to_result_field():
    """With no assistant text blocks, the terminal result field is used."""
    assert ClaudeCodeProvider._report_text([], {"result": "fallback"}) == "fallback"


def test_report_text_raises_when_nothing_available():
    """No assistant text and no result text raises a ValueError."""
    with pytest.raises(ValueError, match="no result text"):
        ClaudeCodeProvider._report_text([], {"is_error": False, "result": ""})


@pytest.mark.parametrize("report", ["", "   ", "Too short to be research."])
def test_check_report_length_rejects_implausible_reports(report):
    """An implausibly short report fails loudly instead of being written out."""
    with pytest.raises(ValueError, match="min_report_chars"):
        ClaudeCodeProvider._check_report_length(report, 200)


def test_check_report_length_accepts_a_real_report():
    """A report at or above the threshold passes."""
    ClaudeCodeProvider._check_report_length("x" * 200, 200)


def test_check_report_length_disabled_by_zero():
    """A zero threshold disables the check entirely."""
    ClaudeCodeProvider._check_report_length("", 0)


def test_extract_run_metadata_reads_actual_models_and_usage():
    """Run metadata should surface the real model(s), cost, turns, and searches."""
    data = {
        "result": "report",
        "num_turns": 12,
        "total_cost_usd": 0.34,
        "session_id": "sess-abc",
        "stop_reason": "end_turn",
        "modelUsage": {
            "claude-opus-4-8[1m]": {"outputTokens": 100, "webSearchRequests": 5},
        },
    }
    metadata = ClaudeCodeProvider._extract_run_metadata(data)
    assert metadata["models_used"] == ["claude-opus-4-8[1m]"]
    assert metadata["num_turns"] == 12
    assert metadata["total_cost_usd"] == 0.34
    assert metadata["web_search_requests"] == 5
    assert metadata["session_id"] == "sess-abc"
    assert metadata["stop_reason"] == "end_turn"
    # No denials in this payload, so the key is omitted rather than zero.
    assert "permission_denials" not in metadata


def test_extract_run_metadata_counts_permission_denials():
    """Denied tool calls are surfaced; they often explain a thin report."""
    data = {"permission_denials": [{"tool_name": "Read"}, {"tool_name": "Bash"}]}
    assert ClaudeCodeProvider._extract_run_metadata(data)["permission_denials"] == 2


def test_extract_run_metadata_handles_multiple_models():
    """If the run used more than one model, all are reported, sorted."""
    data = {
        "modelUsage": {
            "claude-sonnet-4-6": {"webSearchRequests": 1},
            "claude-opus-4-8[1m]": {"webSearchRequests": 2},
        }
    }
    metadata = ClaudeCodeProvider._extract_run_metadata(data)
    assert metadata["models_used"] == ["claude-opus-4-8[1m]", "claude-sonnet-4-6"]
    assert metadata["web_search_requests"] == 3


def test_extract_run_metadata_empty_when_absent():
    """A payload with no usage info yields empty metadata, not errors."""
    assert ClaudeCodeProvider._extract_run_metadata({"result": "x"}) == {}


def test_extract_citations_dedupes_and_strips_trailing_punctuation():
    """Citation extraction collects unique URLs in order, trimming punctuation."""
    markdown = (
        "See https://example.com/a, and also https://example.com/b. "
        "Again https://example.com/a for emphasis."
    )
    citations = ClaudeCodeProvider._extract_citations(markdown)
    assert citations == ["https://example.com/a", "https://example.com/b"]


def test_extract_citations_handles_no_urls():
    """Markdown without URLs yields no citations."""
    assert ClaudeCodeProvider._extract_citations("Plain text, no links.") == []


@pytest.mark.asyncio
async def test_research_raises_on_empty_query():
    """An empty query is rejected before invoking the CLI."""
    # Use a trivially-available executable so the check doesn't depend on `claude`.
    provider = make_provider(ClaudeCodeParams(claude_executable="true"))
    with pytest.raises(ValueError, match="must not be empty"):
        await provider.research("   ")


@pytest.mark.asyncio
async def test_research_raises_on_nonzero_exit():
    """A non-zero exit from the CLI surfaces as a ValueError."""
    if shutil.which("false") is None:
        pytest.skip("`false` not available")
    # `false` ignores stdin/args and exits 1 immediately.
    provider = make_provider(ClaudeCodeParams(claude_executable="false"))
    with pytest.raises(ValueError, match="exited with code"):
        await provider.research("anything")


@pytest.mark.asyncio
async def test_research_raises_on_timeout(tmp_path):
    """A run that exceeds the timeout is killed and surfaces as a ValueError."""
    script = tmp_path / "slowclaude"
    script.write_text("#!/bin/sh\nsleep 5\n")
    script.chmod(0o755)
    config = ProviderConfig(name="claude_code", api_key=None, enabled=True, timeout=1)
    provider = ClaudeCodeProvider(
        config, ClaudeCodeParams(claude_executable=str(script))
    )
    with pytest.raises(ValueError, match="timed out"):
        await provider.research("anything")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_end_to_end():
    """Run a tiny real research query through the local Claude Code CLI."""
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not available")

    provider = make_provider()
    result = await provider.research(
        "In one short paragraph, what is CRISPR? You may answer from prior knowledge."
    )
    assert result.markdown.strip()
    assert result.provider == "claude_code"
    assert result.duration_seconds is not None
    # Provenance: the real model id should be reported, not our sentinel default.
    assert result.run_metadata is not None
    assert result.run_metadata.get("models_used")
    assert result.run_metadata.get("assistant_text_blocks", 0) >= 1
    assert result.model != "claude-code-default"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_keeps_output_preceding_a_tool_call():
    """Regression (#59): content written before a later turn must survive.

    The prompt deliberately induces the shape that used to lose the report --
    substantive output, then a tool call, then a short closing message -- and
    asserts the early content is still present.
    """
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not available")

    provider = make_provider(ClaudeCodeParams(model="haiku"))
    result = await provider.research(
        "First, write a 150-word essay about granite. Then use WebSearch to search "
        "for 'granite countertop price'. Then respond with exactly the word: done"
    )
    assert "granite" in result.markdown.lower()
    # The essay dwarfs the "done" sign-off that the old parser would have kept alone.
    assert len(result.markdown) > 500
