"""The chat-completions client that runs through the local Claude Code CLI.

Reading a run's output is a pure function, tested on output shapes the CLI
produced (2.1.283). The process itself runs under the ``llm`` marker, since
every call spends the logged-in account's allowance.
"""

import asyncio
import json
import shutil

import pytest

from deep_research_client.claude_code_chat import (
    ClaudeCodeChatClient,
    build_command,
    completion_from_output,
    split_messages,
)
from deep_research_client.exceptions import ProviderAuthError, ProviderQuotaError


def _result(**fields) -> str:
    """One ``--output-format json`` result object, as the CLI prints it."""
    return json.dumps({"type": "result", "subtype": "success", "is_error": False, **fields})


@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [("end_turn", "stop"), ("stop_sequence", "stop"), ("max_tokens", "length"), (None, "stop")],
)
def test_a_stop_reason_reads_as_the_openai_finish_reason(stop_reason, finish_reason):
    """Callers check ``finish_reason == "length"``, so the CLI's reason maps onto it."""
    reply = completion_from_output(_result(result="text", stop_reason=stop_reason), "", 0, "sonnet")

    assert reply.choices[0].finish_reason == finish_reason
    assert reply.choices[0].message.content == "text"


def test_the_reply_names_the_model_that_ran_not_the_alias():
    """``sonnet`` is an alias; the run's modelUsage has the id it resolved to."""
    stdout = _result(result="x", modelUsage={"claude-haiku-4-5-20251001": {"outputTokens": 3}})

    assert completion_from_output(stdout, "", 0, "haiku").model == "claude-haiku-4-5-20251001"
    assert completion_from_output(_result(result="x"), "", 0, "haiku").model == "haiku"


def test_a_reply_past_the_output_limit_is_empty_and_cut_off():
    """The CLI reports the limit as an error; callers need it as a truncation."""
    stdout = json.dumps({
        "type": "result", "subtype": "success", "is_error": True, "terminal_reason": "api_error",
        "result": "API Error: Claude's response exceeded the 60 output token maximum. To "
                  "configure this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.",
    })

    reply = completion_from_output(stdout, "", 0, "haiku")

    assert reply.choices[0].finish_reason == "length"
    assert reply.choices[0].message.content == ""


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode", "error", "text"),
    [
        (
            _result(is_error=True, result="", terminal_reason="api_error", modelUsage={}),
            '[claude-code:unrecognized_model] {"model":"no-such-model","query_source":"sdk"}',
            1, ValueError, "unrecognized_model",
        ),
        ("", "Invalid API key. Please run /login", 1, ProviderAuthError, "Invalid API key"),
        (
            _result(is_error=True, result="Claude usage limit reached. Your limit will reset at 3pm."),
            "", 1, ProviderQuotaError, "usage limit",
        ),
    ],
)
def test_a_failed_run_raises_with_the_cli_s_own_words(stdout, stderr, returncode, error, text):
    """A failure is never returned as a reply the parser would read as "no claims"."""
    with pytest.raises(error, match=text):
        completion_from_output(stdout, stderr, returncode, "sonnet")


def test_a_multi_turn_conversation_is_refused_rather_than_flattened():
    """One ``--print`` call takes one prompt."""
    with pytest.raises(ValueError, match="must be from the user"):
        split_messages([{"role": "system", "content": "S"}, {"role": "assistant", "content": "A"}])


def test_the_command_gives_the_model_no_tools_settings_or_mcp_servers():
    """Hooks, CLAUDE.md, output styles and tools must not reach an extraction call."""
    command = build_command("claude", "sonnet", "SYSTEM")

    assert command[command.index("--system-prompt") + 1] == "SYSTEM"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in command
    assert "--dangerously-skip-permissions" not in command


@pytest.mark.llm
async def test_a_real_call_answers_and_reports_its_model():
    """One short completion through the logged-in CLI."""
    if shutil.which("claude") is None:
        pytest.skip("needs the `claude` CLI on PATH")

    reply = await ClaudeCodeChatClient().chat.completions.create(
        model="haiku",
        messages=[
            {"role": "system", "content": "Reply with the single word the user asks for."},
            {"role": "user", "content": "Say: fibrillin"},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    assert "fibrillin" in reply.choices[0].message.content.lower()
    assert reply.choices[0].finish_reason == "stop"
    assert reply.model.startswith("claude-haiku")


@pytest.mark.llm
async def test_a_real_reply_past_max_tokens_reads_as_cut_off():
    """``max_tokens`` reaches the CLI, and running past it is a truncation."""
    if shutil.which("claude") is None:
        pytest.skip("needs the `claude` CLI on PATH")

    reply = await ClaudeCodeChatClient().chat.completions.create(
        model="haiku",
        messages=[{"role": "user", "content": "Count from 1 to 400, one number per line."}],
        max_tokens=60,
    )

    assert reply.choices[0].finish_reason == "length"


def test_a_continued_reply_reads_as_cut_off_not_as_its_tail():
    """Past the limit, Claude Code may continue in a second turn and report only that turn."""
    stdout = _result(num_turns=2, stop_reason="end_turn", result='"citations": []}]}')

    reply = completion_from_output(stdout, "", 0, "sonnet")

    assert reply.choices[0].finish_reason == "length"
    assert reply.choices[0].message.content == ""


@pytest.mark.llm
def test_one_cut_off_unit_raises_instead_of_hanging_the_others():
    """A failing unit must surface while its siblings' processes are still starting.

    This hung before decompose_units drained its own tasks: asyncio.run
    cancelled asyncio's pipe-connecting task for a process still spawning,
    and that process creation never finished.
    """
    if shutil.which("claude") is None:
        pytest.skip("needs the `claude` CLI on PATH")
    from deep_research_client.claims.llm import decompose_units
    from deep_research_client.claims.parsing import TextUnit, UnreadableReplyError

    texts = ["Fibrillin-1 is encoded by FBN1."] * 9
    texts[1] = " ".join(
        f"Gene G{i} causes disease D{i} and phenotype P{i} in adults [1]." for i in range(120)
    )
    units = [TextUnit(text=t, start=0, end=len(t), section=f"s{i}") for i, t in enumerate(texts)]

    with pytest.raises(UnreadableReplyError, match="s1 was cut off at max_tokens=300"):
        asyncio.run(asyncio.wait_for(
            decompose_units(units, ClaudeCodeChatClient(), "haiku", max_tokens=300), timeout=120,
        ))
