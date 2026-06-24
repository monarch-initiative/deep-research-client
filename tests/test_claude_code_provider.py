"""Tests for the Claude Code provider.

These focus on the pure, side-effect-free helpers (command building, output
parsing, citation extraction, availability) so they run without invoking the
real ``claude`` CLI. A single integration test exercises an actual run and is
skipped automatically when the CLI is unavailable.
"""

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


def test_build_command_defaults():
    """Default command runs print mode, JSON output, and skips permissions."""
    provider = make_provider()
    command = provider._build_command()

    assert command[0] == "claude"
    assert "--print" in command
    assert command[command.index("--output-format") + 1] == "json"
    assert "--dangerously-skip-permissions" in command
    # The research system prompt plus the inline-report directive are appended.
    assert "--append-system-prompt" in command
    appended = command[command.index("--append-system-prompt") + 1]
    assert appended.startswith(DEFAULT_RESEARCH_SYSTEM_PROMPT)
    assert _INLINE_REPORT_DIRECTIVE in appended
    # No model is forwarded for the default sentinel.
    assert "--model" not in command


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


def test_parse_output_success():
    """A successful JSON result yields the markdown report."""
    stdout = '{"type": "result", "is_error": false, "result": "# Report\\n\\nbody"}'
    assert ClaudeCodeProvider._parse_output(stdout) == "# Report\n\nbody"


def test_parse_output_raises_on_error_flag():
    """An is_error result raises a ValueError with the detail."""
    stdout = '{"is_error": true, "subtype": "error_max_turns", "result": "hit limit"}'
    with pytest.raises(ValueError, match="error_max_turns"):
        ClaudeCodeProvider._parse_output(stdout)


def test_parse_output_raises_on_invalid_json():
    """Non-JSON output raises a ValueError."""
    with pytest.raises(ValueError, match="parse Claude Code JSON"):
        ClaudeCodeProvider._parse_output("not json at all")


def test_parse_output_raises_on_empty_output():
    """Empty stdout raises a ValueError."""
    with pytest.raises(ValueError, match="empty output"):
        ClaudeCodeProvider._parse_output("   ")


def test_parse_output_raises_on_missing_result():
    """A JSON object with no usable result text raises a ValueError."""
    with pytest.raises(ValueError, match="no result text"):
        ClaudeCodeProvider._parse_output('{"is_error": false, "result": ""}')


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
    provider = make_provider()
    if not provider.is_available():
        pytest.skip("claude CLI not available")
    with pytest.raises(ValueError, match="must not be empty"):
        await provider.research("   ")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_end_to_end():
    """Run a tiny real research query through the local Claude Code CLI."""
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not available")

    provider = make_provider()
    result = await provider.research(
        "In one sentence, what is CRISPR? You may answer from prior knowledge."
    )
    assert result.markdown.strip()
    assert result.provider == "claude_code"
    assert result.duration_seconds is not None
    # Provenance: the real model id should be reported, not our sentinel default.
    assert result.run_metadata is not None
    assert result.run_metadata.get("models_used")
    assert result.model != "claude-code-default"
