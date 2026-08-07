"""Claude Code research provider.

Unlike the API-based providers, Claude Code is a local command-line tool (the
``claude`` binary). This provider drives it in non-interactive "print" mode,
piping the research prompt to it and letting Claude Code's own agentic tools
(web search, web fetch, file reading, etc.) carry out a multi-step research
workflow. The full response is returned as a cited markdown report.

Output is read as a ``stream-json`` event stream rather than the simpler ``json``
format. The latter exposes only a ``result`` field holding the agent's *final*
message, so an agent that writes its report and then emits any closing remark
loses the entire report -- silently, since the run still exits 0 with valid
provenance metadata. Reading the stream lets us join every assistant text block
while still taking run metadata from the terminal ``result`` event.

No provider API key is required: billing and authentication are handled by the
local Claude Code installation.

Security: by default the provider does NOT pass ``--dangerously-skip-permissions``.
Instead it restricts the agent to a read-only research toolset
(``allowed_tools`` defaults to WebSearch + WebFetch) via ``--allowedTools``; in
non-interactive mode any tool not on that list is auto-denied without blocking.
This keeps the out-of-the-box behavior from mutating the filesystem or running
shell commands even on an untrusted query. Enabling ``skip_permissions`` bypasses
all permission checks and makes the allowlist a no-op, so it should only be used
in trusted, sandboxed environments.
"""

import asyncio
import json
import logging
import re
import shutil
from datetime import datetime
from typing import List, Optional

from . import ResearchProvider
from ..models import ResearchResult, ProviderConfig
from ..provider_params import ClaudeCodeParams
from ..model_cards import ProviderModelCards, create_claude_code_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Non-interactive ("print") mode has no "later": the process emits one response
# and exits. Local skills/workflows that defer work to a background task would be
# orphaned, leaving us with only a chatty preamble instead of a report. This
# directive forces Claude Code to do the work and emit the finished report inline.
_INLINE_REPORT_DIRECTIVE = (
    "IMPORTANT: You are running non-interactively and must produce the complete "
    "final research report directly in this single response. Do the research now "
    "using your available tools (web search, web fetch, etc.) and write the full "
    "report inline. Do NOT defer the work, do NOT hand it off to a background task, "
    "workflow, or sub-agent, do NOT promise to follow up later, and do NOT ask "
    "clarifying questions. Output only the report itself in Markdown."
)

# Aliases that point at the provider's own default model card rather than a real
# Claude model id. When the user "selects" one of these we let the local Claude
# Code installation pick its own default model instead of forwarding --model.
_DEFAULT_MODEL_SENTINELS = {
    "claude-code-default",
    "claude-code",
    "claude",
    "cc",
    "default",
}

# URL pattern for citation extraction (mirrors the other providers).
_URL_PATTERN = re.compile(r"https?://[^\s\)\]>]+")


class ClaudeCodeProvider(ResearchProvider):
    """Provider that runs the local Claude Code CLI to perform research."""

    def __init__(self, config: ProviderConfig, params: Optional[ClaudeCodeParams] = None):
        """Initialize the Claude Code provider.

        Args:
            config: Provider configuration (no API key required).
            params: Claude Code-specific parameters.
        """
        self.params = params or ClaudeCodeParams()
        super().__init__(config, self.params.model)

        self.claude_executable = self.params.claude_executable
        # ProviderConfig.timeout wins when set; otherwise use the params default.
        self.timeout = config.timeout or self.params.timeout

        logger.debug(
            "Initializing Claude Code provider (executable=%s, skip_permissions=%s)",
            self.claude_executable,
            self.params.skip_permissions,
        )

    def get_default_model(self) -> str:
        """Get the default model identifier for this provider."""
        return "claude-code-default"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for the Claude Code provider."""
        return create_claude_code_model_cards()

    def is_available(self) -> bool:
        """Check whether the Claude Code CLI is available.

        Returns:
            True if the provider is enabled and the ``claude`` executable is
            resolvable on the PATH (or as an absolute path).
        """
        if not self.config.enabled:
            return False

        if shutil.which(self.claude_executable) is None:
            logger.warning(
                "Claude Code executable %r not found in PATH", self.claude_executable
            )
            return False

        return True

    def _resolve_cli_model(self) -> Optional[str]:
        """Resolve the raw ``--model`` value to forward to the CLI, if any.

        We forward the user's raw model string (e.g. ``opus`` or a full model
        id) rather than the resolved model-card name. Provider-internal default
        aliases resolve to ``None`` so the local Claude Code installation picks
        its own default model.
        """
        raw_model = self.params.model
        if not raw_model:
            return None
        if raw_model.strip().lower() in _DEFAULT_MODEL_SENTINELS:
            return None
        return raw_model

    def _build_command(self) -> List[str]:
        """Build the ``claude`` command-line invocation (without the prompt).

        The research prompt is supplied separately via stdin, so it does not
        appear here. This method is pure and side-effect free to keep it easy to
        unit test.

        Returns:
            The argument list to pass to ``asyncio.create_subprocess_exec``.
        """
        # stream-json (which requires --verbose in print mode) preserves every
        # assistant message; the plain "json" format would give us only the last
        # one. See the module docstring.
        command: List[str] = [
            self.claude_executable,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
        ]

        if self.params.skip_permissions:
            command.append("--dangerously-skip-permissions")
        elif self.params.permission_mode:
            command.extend(["--permission-mode", self.params.permission_mode])

        cli_model = self._resolve_cli_model()
        if cli_model:
            command.extend(["--model", cli_model])

        system_prompt = self.params.system_prompt or DEFAULT_RESEARCH_SYSTEM_PROMPT
        # Always append the inline-report directive so print-mode runs produce the
        # report directly rather than deferring to a background workflow.
        system_prompt = f"{system_prompt}\n\n{_INLINE_REPORT_DIRECTIVE}"
        command.extend(["--append-system-prompt", system_prompt])

        if self.params.allowed_tools:
            command.extend(["--allowedTools", ",".join(self.params.allowed_tools)])

        for directory in self.params.add_dirs:
            command.extend(["--add-dir", directory])

        command.extend(self.params.extra_args)

        return command

    async def research(self, query: str) -> ResearchResult:
        """Perform research by running the Claude Code CLI.

        Args:
            query: The research question or topic.

        Returns:
            ResearchResult with the markdown report and extracted citations.

        Raises:
            ValueError: If the provider is unavailable, the query is empty, or
                the Claude Code run fails.
        """
        start_time = datetime.now()

        if not self.is_available():
            raise ValueError(
                "Claude Code provider not available. Ensure the 'claude' CLI is "
                "installed and on your PATH."
            )

        if not query or not query.strip():
            raise ValueError("Research query must not be empty.")

        command = self._build_command()
        logger.info("Running Claude Code research (timeout=%ss)", self.timeout)
        logger.debug("Claude Code command: %s", " ".join(command))

        stdout, stderr, returncode = await self._run_process(command, query)

        if returncode != 0:
            raise ValueError(
                f"Claude Code exited with code {returncode}: "
                f"{stderr.strip() or '<no stderr>'}"
            )

        assistant_texts, data = self._parse_stream(stdout)
        markdown = self._report_text(assistant_texts, data)
        self._check_report_length(markdown, self.params.min_report_chars)

        run_metadata = self._extract_run_metadata(data)
        # How many separate assistant messages the report was assembled from. A
        # value above 1 means the agent spoke again after its main output, the
        # shape that used to truncate the report entirely.
        run_metadata["assistant_text_blocks"] = len(assistant_texts)
        citations = self._extract_citations(markdown)

        # Prefer the model id(s) the run actually reported over our sentinel
        # default, so provenance reflects what really ran (even if unspecified or
        # switched mid-flight).
        models_used = run_metadata.get("models_used")
        model_label = ", ".join(models_used) if models_used else self.model

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            "Claude Code research completed in %.1fs (%d chars, %d citations, model=%s)",
            duration,
            len(markdown),
            len(citations),
            model_label,
        )

        return ResearchResult(
            markdown=markdown,
            citations=citations,
            provider=self.name,
            query=query,
            model=model_label,
            run_metadata=run_metadata or None,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )

    async def _run_process(self, command: List[str], query: str) -> tuple[str, str, int]:
        """Run the claude subprocess, piping the query to stdin.

        Args:
            command: The command argument list (without the prompt).
            query: The research prompt, supplied via stdin.

        Returns:
            Tuple of (stdout, stderr, return_code).

        Raises:
            ValueError: If the process does not finish within the timeout.
        """
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.params.working_dir,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=query.encode("utf-8")),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise ValueError(
                f"Claude Code run timed out after {self.timeout}s."
            )

        # returncode is set once communicate() has returned.
        returncode = -1 if process.returncode is None else process.returncode
        return (
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            returncode,
        )

    @staticmethod
    def _parse_stream(stdout: str) -> tuple[List[str], dict]:
        """Parse Claude Code's ``stream-json`` output into text blocks and metadata.

        The stream is JSON Lines: one event per line. ``assistant`` events carry
        the model's own output (text, thinking, and tool_use blocks, split across
        events); the terminal ``result`` event carries the run's provenance and
        error status.

        Args:
            stdout: Raw stdout from ``claude --print --output-format stream-json``.

        Returns:
            Tuple of (assistant text blocks in emission order, terminal result event).

        Raises:
            ValueError: If the output is empty, contains no terminal result event,
                or that event reports an error.
        """
        text = stdout.strip()
        if not text:
            raise ValueError("Claude Code returned empty output.")

        assistant_texts: List[str] = []
        result_event: Optional[dict] = None

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A stray non-JSON line (a warning, say) should not sink an
                # otherwise complete run; a genuinely unparseable stream is
                # caught below by the missing result event.
                logger.warning(
                    "Skipping unparseable line in Claude Code output: %.200s", line
                )
                continue

            if not isinstance(event, dict):
                continue

            if event.get("type") == "assistant":
                content = event.get("message", {}).get("content") or []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        block_text = str(block.get("text") or "")
                        if block_text.strip():
                            assistant_texts.append(block_text)
            elif event.get("type") == "result":
                result_event = event

        if result_event is None:
            raise ValueError(
                "Could not parse Claude Code JSON output: the stream contained no "
                "terminal 'result' event, so the run did not complete."
            )

        if result_event.get("is_error"):
            subtype = result_event.get("subtype", "unknown error")
            detail = result_event.get("result") or subtype
            raise ValueError(f"Claude Code reported an error ({subtype}): {detail}")

        return assistant_texts, result_event

    @staticmethod
    def _report_text(assistant_texts: List[str], data: dict) -> str:
        """Assemble the markdown report from every assistant text block.

        Falls back to the terminal event's ``result`` field only when the stream
        carried no assistant text at all. That field holds just the agent's final
        message, so preferring the joined blocks is what keeps a report from
        being truncated to its closing remark.

        Args:
            assistant_texts: Assistant text blocks in emission order.
            data: The terminal ``result`` event.

        Raises:
            ValueError: If neither source yields any text.
        """
        if assistant_texts:
            return "\n\n".join(block.strip() for block in assistant_texts)

        result = data.get("result")
        if not result or not str(result).strip():
            raise ValueError("Claude Code output contained no result text.")
        return str(result)

    @staticmethod
    def _check_report_length(markdown: str, min_chars: int) -> None:
        """Reject a report too short to be a plausible research result.

        Silent success on an empty run is the costly failure mode: the caller
        gets a well-formed file with real provenance and no content. Failing here
        turns that into a non-zero exit.

        Raises:
            ValueError: If ``markdown`` is shorter than ``min_chars``.
        """
        if min_chars <= 0:
            return

        length = len(markdown.strip())
        if length < min_chars:
            raise ValueError(
                f"Claude Code returned a {length}-character report, below the "
                f"min_report_chars threshold of {min_chars}. This usually means "
                "the run failed, was cut short, or produced no research. Lower or "
                "disable the threshold (min_report_chars=0) if short answers are "
                "expected."
            )

    @staticmethod
    def _extract_run_metadata(data: dict) -> dict:
        """Extract run provenance from Claude Code's JSON output.

        The ``modelUsage`` map is keyed by the model id(s) that actually ran, so
        we can report the real model even when none was requested or it changed
        mid-flight.

        Args:
            data: The terminal ``result`` event from the output stream.

        Returns:
            A metadata dict with any of: models_used, num_turns, total_cost_usd,
            web_search_requests, session_id, stop_reason, permission_denials.
        """
        metadata: dict = {}

        model_usage = data.get("modelUsage") or {}
        models_used = sorted(model_usage.keys())
        if models_used:
            metadata["models_used"] = models_used

        web_search = 0
        for usage in model_usage.values():
            if isinstance(usage, dict):
                web_search += usage.get("webSearchRequests") or 0
        if web_search:
            metadata["web_search_requests"] = web_search

        if data.get("num_turns") is not None:
            metadata["num_turns"] = data["num_turns"]
        if data.get("total_cost_usd") is not None:
            metadata["total_cost_usd"] = data["total_cost_usd"]
        if data.get("session_id"):
            metadata["session_id"] = data["session_id"]
        if data.get("stop_reason"):
            metadata["stop_reason"] = data["stop_reason"]

        # Denied tool calls are a common cause of a thin report: the agent asked
        # for a tool outside the allowlist, was refused, and gave up.
        denials = data.get("permission_denials")
        if denials:
            metadata["permission_denials"] = len(denials)

        return metadata

    @staticmethod
    def _extract_citations(markdown: str) -> List[str]:
        """Extract URL citations from the markdown report, preserving order.

        Args:
            markdown: The markdown research report.

        Returns:
            Deduplicated list of cited URLs.
        """
        citations: List[str] = []
        seen = set()
        for url in _URL_PATTERN.findall(markdown):
            cleaned = url.rstrip(".,;")
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                citations.append(cleaned)
        return citations
