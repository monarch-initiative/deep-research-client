"""A chat-completions client that answers through the local Claude Code CLI.

The LLM-backed commands (``claims extract``, the ``eval score`` judges) call a
model through the one method of the OpenAI chat interface they use:
``await client.chat.completions.create(model=..., messages=..., ...)``, read
back as ``response.choices[0].message.content`` and ``.finish_reason``. This
client offers that same method and runs each call as one ``claude --print``
process, so a machine with Claude Code logged in can use Claude models with no
API key. Billing and authentication are the local installation's, as for the
``claude_code`` research provider.

Each call is a plain completion, not an agent run:

- ``--system-prompt`` replaces Claude Code's own system prompt with the
  caller's system message.
- ``--tools ""`` gives the model no tools.
- ``--setting-sources ""`` and ``--strict-mcp-config`` load no settings files
  and no MCP servers, so the user's hooks, output style and CLAUDE.md files do
  not reach the prompt. (``--bare`` would do this too, but it refuses the
  OAuth login a Claude subscription uses, which is the point of this client.)
- ``--no-session-persistence`` leaves no session behind.

Two parameters of the OpenAI interface cannot be passed the same way:

- ``temperature``: the CLI has no control for it. It is accepted and not
  applied, so replies are not deterministic even when the caller asks for 0.
- ``max_tokens``: applied through ``CLAUDE_CODE_MAX_OUTPUT_TOKENS``, with
  thinking turned off (``MAX_THINKING_TOKENS=0``), since thinking tokens count
  against that limit and an OpenAI chat model's ``max_tokens`` is all reply.

A reply that runs past the limit does not stop the way an OpenAI reply does.
Claude Code either reports an error naming the limit, or continues the reply
in a second turn, and then its ``result`` holds only that second turn: the
tail of the reply, with a stop reason of ``end_turn``. Both are returned as an
empty reply with ``finish_reason="length"``. With no tools, a continuation is
the only way a call takes more than one turn, so ``num_turns`` is the signal.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .providers.claude_code import classify_cli_failure, terminal_result_text

__all__ = [
    "DEFAULT_CLAUDE_CODE_MODEL",
    "ChatCompletion",
    "ClaudeCodeChatClient",
    "build_command",
    "completion_from_output",
    "split_messages",
]

#: Model used when the caller names none. An alias, resolved by the CLI to the
#: current model of that tier; the reply records the full id it resolved to.
DEFAULT_CLAUDE_CODE_MODEL = "sonnet"

#: Used when the caller sends no system message. Without one, the CLI would
#: fall back to its own coding-agent system prompt.
_NEUTRAL_SYSTEM_PROMPT = "Answer the user's message as it asks."

#: How the CLI says a reply ran past CLAUDE_CODE_MAX_OUTPUT_TOKENS.
_OUTPUT_LIMIT = re.compile(r"exceeded the \d+ output token maximum", re.IGNORECASE)

#: The CLI's stop reasons, as the OpenAI finish reasons the callers check.
_FINISH_REASONS = {"max_tokens": "length"}


@dataclass(frozen=True)
class ChatMessage:
    """The reply's message: ``choices[0].message``."""

    content: str
    role: str = "assistant"


@dataclass(frozen=True)
class ChatChoice:
    """One reply: ``choices[0]``."""

    message: ChatMessage
    finish_reason: str


@dataclass(frozen=True)
class ChatCompletion:
    """A reply shaped as the part of an OpenAI chat completion callers read."""

    choices: list[ChatChoice]
    model: str


def split_messages(messages: Sequence[dict[str, str]]) -> tuple[str, str]:
    """Split chat messages into a system prompt and the one user message.

    A ``claude --print`` call takes one prompt, so a conversation with earlier
    turns cannot be sent; that raises rather than being flattened into text
    the model would read differently.

    Args:
        messages: OpenAI-style messages.

    Returns:
        The system messages joined, or a neutral prompt if there are none,
        and the user message.

    Raises:
        ValueError: If there is not exactly one non-system message, or it is
            not from the user.

    >>> split_messages([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    ('S', 'U')
    >>> split_messages([{"role": "user", "content": "U"}])[0]
    "Answer the user's message as it asks."
    >>> split_messages([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
    ...                 {"role": "user", "content": "c"}])
    Traceback (most recent call last):
    ...
    ValueError: Claude Code takes one user message per call; got 3 non-system messages
    """
    system = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if len(rest) != 1:
        raise ValueError(
            f"Claude Code takes one user message per call; got {len(rest)} non-system messages"
        )
    if rest[0]["role"] != "user":
        raise ValueError(f"The message to Claude Code must be from the user, not {rest[0]['role']!r}")
    return "\n\n".join(system) or _NEUTRAL_SYSTEM_PROMPT, rest[0]["content"]


def build_command(executable: str, model: str, system_prompt: str) -> list[str]:
    """The ``claude`` invocation for one completion. The prompt goes to stdin.

    Args:
        executable: The ``claude`` binary.
        model: A model alias or full id, passed to ``--model``.
        system_prompt: Replaces Claude Code's own system prompt.

    Returns:
        The argument list for ``asyncio.create_subprocess_exec``.

    >>> command = build_command("claude", "haiku", "Extract claims.")
    >>> command[command.index("--tools") + 1], command[command.index("--model") + 1]
    ('', 'haiku')
    """
    return [
        executable,
        "--print",
        "--output-format", "json",
        "--model", model,
        "--system-prompt", system_prompt,
        "--tools", "",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]


def _main_model(usage: Any, requested_model: str) -> str:
    """The model that wrote the reply, from a run's ``modelUsage``.

    Claude Code can also call a smaller model for its own housekeeping, and
    that model is listed too, so the first entry need not be the one that
    answered. The one that wrote the most output tokens did.

    >>> _main_model({"claude-haiku-4-5": {"outputTokens": 12},
    ...              "claude-sonnet-5": {"outputTokens": 900}}, "sonnet")
    'claude-sonnet-5'
    >>> _main_model({}, "sonnet"), _main_model(None, "sonnet")
    ('sonnet', 'sonnet')
    """
    if not isinstance(usage, dict) or not usage:
        return requested_model

    def output_tokens(name: str) -> int:
        entry = usage[name]
        return (entry.get("outputTokens") or 0) if isinstance(entry, dict) else 0

    return max(usage, key=output_tokens)


def completion_from_output(
    stdout: str, stderr: str, returncode: int, requested_model: str,
) -> ChatCompletion:
    """Read one ``claude --print --output-format json`` run as a completion.

    Args:
        stdout: The run's stdout: one JSON result object.
        stderr: The run's stderr.
        returncode: The process exit code.
        requested_model: The model asked for, reported if the run names none.

    Returns:
        The reply. A reply cut off by the output limit is empty, with
        ``finish_reason="length"``.

    Raises:
        ProviderError: A subclass, when the failure is one the CLI's text
            identifies (not logged in, usage limit, overloaded...).
        ValueError: For any other failed run.

    >>> ok = json.dumps({"type": "result", "subtype": "success", "is_error": False,
    ...     "result": '{"claims": []}', "stop_reason": "end_turn",
    ...     "modelUsage": {"claude-sonnet-5": {}}})
    >>> reply = completion_from_output(ok, "", 0, "sonnet")
    >>> reply.choices[0].message.content, reply.choices[0].finish_reason, reply.model
    ('{"claims": []}', 'stop', 'claude-sonnet-5')
    >>> capped = json.dumps({"type": "result", "subtype": "success", "is_error": True,
    ...     "result": "API Error: Claude's response exceeded the 60 output token maximum."})
    >>> completion_from_output(capped, "", 0, "sonnet").choices[0].finish_reason
    'length'
    >>> continued = json.dumps({"type": "result", "subtype": "success", "is_error": False,
    ...     "num_turns": 2, "stop_reason": "end_turn", "result": 'ies": []}]}'})
    >>> completion_from_output(continued, "", 0, "sonnet").choices[0].message.content
    ''
    """
    try:
        data: Any = json.loads(stdout)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        classified = classify_cli_failure("claude_code", stderr)
        if classified is not None:
            raise classified
        raise ValueError(
            f"Claude Code exited with code {returncode} and no readable result: "
            f"{stderr.strip() or stdout.strip()[:200] or '<no output>'}"
        )

    result = data.get("result") or ""
    model = _main_model(data.get("modelUsage"), requested_model)
    cut_off = ChatCompletion(
        choices=[ChatChoice(message=ChatMessage(content=""), finish_reason="length")],
        model=model,
    )
    failed = returncode != 0 or data.get("is_error") or (data.get("subtype") or "success") != "success"
    if failed:
        if _OUTPUT_LIMIT.search(result):
            return cut_off
        classified = classify_cli_failure("claude_code", f"{stderr}\n{terminal_result_text(stdout)}")
        if classified is not None:
            raise classified
        detail = result or stderr.strip() or data.get("terminal_reason") or "<no detail>"
        raise ValueError(f"Claude Code exited with code {returncode}: {detail}")

    if (data.get("num_turns") or 1) > 1:
        return cut_off
    finish_reason = _FINISH_REASONS.get(data.get("stop_reason") or "", "stop")
    return ChatCompletion(
        choices=[ChatChoice(message=ChatMessage(content=result), finish_reason=finish_reason)],
        model=model,
    )


class _Completions:
    """``client.chat.completions``."""

    def __init__(self, client: "ClaudeCodeChatClient"):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletion:
        """Run one completion through ``claude --print``.

        Args:
            model: A Claude model alias (``sonnet``, ``opus``, ``haiku``) or id.
            messages: System messages and exactly one user message.
            temperature: Accepted for the OpenAI interface and not applied;
                the CLI has no control for it.
            max_tokens: Output limit, as ``CLAUDE_CODE_MAX_OUTPUT_TOKENS``.

        Returns:
            The reply.
        """
        del temperature  # The CLI cannot set it. See the module docstring.
        return await self._client.complete(model, messages, max_tokens)


class _Chat:
    """``client.chat``."""

    def __init__(self, client: "ClaudeCodeChatClient"):
        self.completions = _Completions(client)


class ClaudeCodeChatClient:
    """Answers ``chat.completions.create`` calls through the local ``claude`` CLI.

    Args:
        executable: The ``claude`` binary.
        timeout: Seconds one call may take before its process is killed.
    """

    def __init__(self, executable: str = "claude", timeout: float = 600):
        self.executable = executable
        self.timeout = timeout
        self.chat = _Chat(self)

    async def complete(
        self, model: str, messages: Sequence[dict[str, str]], max_tokens: Optional[int] = None,
    ) -> ChatCompletion:
        """Run one completion. ``chat.completions.create`` calls this.

        Args:
            model: A Claude model alias or id.
            messages: System messages and exactly one user message.
            max_tokens: Output limit, or None for the CLI's own.

        Returns:
            The reply.

        Raises:
            ValueError: If the run fails or does not finish within the timeout.
        """
        system_prompt, prompt = split_messages(messages)
        env = dict(os.environ, MAX_THINKING_TOKENS="0")
        if max_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        process = await asyncio.create_subprocess_exec(
            *build_command(self.executable, model, system_prompt),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        # A call can also end by cancellation: when one unit of a gather
        # fails, asyncio.run cancels its siblings. A child still running then
        # left the cancelled task waiting on its pipes forever, and the first
        # unit's error was never shown. So whatever ends the call, the child
        # is killed and reaped first.
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")), timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise ValueError(f"Claude Code did not answer within {self.timeout}s")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return completion_from_output(
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            -1 if process.returncode is None else process.returncode,
            model,
        )
