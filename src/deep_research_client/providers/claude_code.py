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
from typing import List, NamedTuple, Optional

from . import ResearchProvider
from ..exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderError,
    ProviderNotInstalledError,
    ProviderQuotaError,
    ProviderTransientError,
)
from ..models import ResearchResult, ProviderConfig, ProviderHealth
from ..provider_params import ClaudeCodeParams
from ..model_cards import ProviderModelCards, create_claude_code_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# The CLI reports failures as prose on stderr or in a terminal result event, so
# these are the closest thing it has to a status code. Ordered most specific
# first: a spent usage allowance also contains the word "limit", and a plan
# without access to a model also mentions the model.
_CLI_FAILURE_PATTERNS: list[tuple[re.Pattern[str], type[ProviderError]]] = [
    (re.compile(r"usage limit reached|limit will reset", re.IGNORECASE), ProviderQuotaError),
    (re.compile(r"credit balance is too low|insufficient credit", re.IGNORECASE), ProviderBillingError),
    (
        re.compile(
            r"invalid api key|please run /login|not logged in|log ?in to|"
            r"authentication_error|oauth token (?:has )?expired|unauthorized",
            re.IGNORECASE,
        ),
        ProviderAuthError,
    ),
    (
        re.compile(
            r"(?:model|access) [^.\n]*not (?:available|permitted|supported|allowed)|"
            r"does not have access|upgrade your plan",
            re.IGNORECASE,
        ),
        ProviderAuthError,
    ),
    (re.compile(r"overloaded|\b529\b|service unavailable", re.IGNORECASE), ProviderTransientError),
]

# "Your limit will reset at 3pm (America/Los_Angeles)." -- the closest thing any
# provider here gives us to a remaining-allowance reading.
# The capture stops at a clause boundary, not just a sentence one: a wordy
# message continues past the time with advice ("...at 5pm Pacific Time; if you
# need capacity sooner..."), and everything after the semicolon is not a time.
# Commas are kept: "10am, Tuesday 19 August" is one time, not two clauses.
_LIMIT_RESET = re.compile(r"reset(?:s|ting)?\s+at\s+([^.;\n]+)", re.IGNORECASE)

# `claude auth status` reads local credentials and makes no model call, so it
# should answer immediately. This only guards against a wedged process.
_HEALTH_PROBE_TIMEOUT = 30

# A CLI too old to have `auth status` says so like this. That is evidence about
# the CLI's version, not about whether the provider works.
_NO_SUCH_SUBCOMMAND = re.compile(
    r"unknown (?:command|option|argument)|unrecognized|see .*--help|^usage:",
    re.IGNORECASE | re.MULTILINE,
)


def terminal_result_text(stdout: str) -> str:
    r"""Pull the terminal ``result`` event's own words out of a stream.

    The stream also carries the model's report, which is prose we must never
    classify against: a report *about* rate limits or credits contains the same
    phrases a failure does. Only the CLI's own account of a *failure* is
    evidence -- on a success event the ``result`` field is itself the report
    (see :meth:`ClaudeCodeProvider._report_text`), so it is skipped too.

    Args:
        stdout: Raw stdout from the CLI, which may be truncated or malformed.

    Returns:
        The failing event's subtype and result text, or "" if there is none.

    >>> terminal_result_text('{"type": "result", "is_error": true, "subtype": "e", "result": "boom"}')
    'e boom'
    >>> terminal_result_text('{"type": "result", "subtype": "success", "result": "the report"}')
    ''
    >>> terminal_result_text('not json at all')
    ''
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "result":
            continue
        # `.get(key, default)` only defaults an *absent* key, so an explicit
        # null subtype would read as a failure and hand back the report.
        if not event.get("is_error") and (event.get("subtype") or "success") == "success":
            return ""
        return f"{event.get('subtype') or ''} {event.get('result') or ''}".strip()
    return ""


def classify_cli_failure(provider: str, text: str) -> Optional[ProviderError]:
    """Classify a Claude Code failure from the text the CLI produced.

    Args:
        provider: Name of the provider that failed.
        text: Combined stderr and/or terminal result text from the CLI.

    Returns:
        A typed error, or None when the text matches nothing we recognise.

    >>> err = classify_cli_failure("claude_code", "Claude usage limit reached. Your limit will reset at 3pm.")
    >>> type(err).__name__, err.resets_at
    ('ProviderQuotaError', '3pm')
    >>> type(classify_cli_failure("claude_code", "Invalid API key. Please run /login")).__name__
    'ProviderAuthError'
    >>> classify_cli_failure("claude_code", "tool returned no results") is None
    True
    """
    if not text or not text.strip():
        return None

    for pattern, error_class in _CLI_FAILURE_PATTERNS:
        if not pattern.search(text):
            continue
        detail = text.strip()
        if error_class is ProviderQuotaError:
            reset_match = _LIMIT_RESET.search(text)
            return ProviderQuotaError(
                provider,
                detail,
                resets_at=reset_match.group(1).strip() if reset_match else None,
            )
        return error_class(provider, detail)

    return None


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

# How a run that declined to write the report talks about it: the shape of the
# reply saved as a "report" in issue #101 ("I'm not going to regenerate that
# comprehensive from-memory research narrative -- the previous attempt was
# halted by a safety classifier"). Only consulted when the text cites nothing,
# so a sourced report that merely discusses classifiers is never read as one.
#
# A refusal verb counts only when its object, within a few words of the same
# clause, is the report itself: "I cannot provide a definitive answer" is
# hedging, and "I cannot provide medical advice. This report ..." is a
# disclaimer followed by the report, so the gap between verb and noun never
# crosses . ; : ! ? or a newline. The noun must stand alone, so "research-backed"
# is a modifier, not the object. Bare "halted by" is not a marker either, since
# "the outbreak was halted by vector control" is ordinary epidemiology. Both
# apostrophes, because models write ’.
_REFUSAL_TAIL = (
    r" (?:regenerate|reproduce|produce|write|provide)\b"
    r"(?:[ \t,]+[\w'’-]+){0,5}?[ \t]+"
    r"(?:report|narrative|research|write-?up)(?![\w-])"
)
_REFUSAL_PATTERN = re.compile(
    r"safety classifier"
    rf"|\bI(?:['’]m| am) not going to{_REFUSAL_TAIL}"
    rf"|\bI (?:can['’]t|cannot|won['’]t|will not){_REFUSAL_TAIL}",
    re.IGNORECASE,
)

# Non-URL source identifiers. A run backed by literature tools (ToolUniverse,
# say) can cite PMIDs or DOIs and no URL, and that is still a sourced report.
_SOURCE_ID_PATTERN = re.compile(
    r"\bPMID:?\s*\d+|\bPMC\d+|\bdoi:?\s*10\.\d{4,9}/\S+|\b10\.\d{4,9}/[^\s,;)\]]+",
    re.IGNORECASE,
)

# Tools that organise the agent's own work rather than retrieve anything. A run
# whose only calls are these has still answered from memory.
_NON_RETRIEVAL_TOOLS = frozenset({"TodoWrite", "ExitPlanMode"})


class ParsedStream(NamedTuple):
    """What a ``stream-json`` run produced, as read by ``_parse_stream``."""

    #: The model's text blocks, in emission order.
    assistant_texts: List[str]
    #: The terminal ``result`` event: provenance and error status.
    result: dict
    #: Names of the tools the model called, in call order, repeats included.
    tool_uses: List[str]


class ClaudeCodeProvider(ResearchProvider):
    """Provider that runs the local Claude Code CLI to perform research."""

    #: Auto-detection probes the default ``claude`` binary and honours an
    #: opt-out, neither of which an instance's is_available() can see.
    registration_requirement = (
        "requires the local Claude Code CLI, with DISABLE_CLAUDE_CODE_PROVIDER unset"
    )

    #: Nothing to run is an install problem, not a missing key.
    unavailable_error = ProviderNotInstalledError

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

    def unavailable_reason(self) -> str:
        """Explain the missing CLI, which no API key would fix.

        Returns:
            Human-readable explanation suitable for an error message
        """
        if not self.config.enabled:
            return super().unavailable_reason()
        return (
            f"the {self.claude_executable!r} CLI was not found on PATH; "
            f"install Claude Code to use this provider"
        )

    async def check_health(self) -> ProviderHealth:
        """Ask the CLI whether it is logged in.

        ``claude auth status --json`` is the rare free probe: it reports the
        account, auth method and plan without spending a single token, so
        unlike the HTTP providers this one can say something about the
        allowance before a run rather than only after it fails.

        Returns:
            Health record for this provider
        """
        if not self.is_available():
            return ProviderHealth(
                provider=self.name,
                configured=False,
                reachable=False,
                detail=self.unavailable_reason(),
            )

        try:
            # is_available() checked PATH a moment ago, but a race, a
            # non-executable file, or a permission change lands here as an
            # OSError. A caller promised a health record must still get one.
            process = await asyncio.create_subprocess_exec(
                self.claude_executable, "auth", "status", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=False,
                detail=f"could not run {self.claude_executable!r}: {e}",
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=_HEALTH_PROBE_TIMEOUT
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=False,
                detail=f"`claude auth status` did not answer within {_HEALTH_PROBE_TIMEOUT}s",
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return self._health_from_auth_status(stdout, stderr, process.returncode or 0)

    def _health_from_auth_status(
        self, stdout: str, stderr: str, returncode: int
    ) -> ProviderHealth:
        """Turn ``claude auth status --json`` output into a health record.

        Args:
            stdout: Raw stdout from the probe.
            stderr: Raw stderr from the probe.
            returncode: Exit status of the probe.

        Returns:
            Health record for this provider
        """
        if returncode != 0:
            output = f"{stderr}\n{stdout}"
            # Classify first. _NO_SUCH_SUBCOMMAND is deliberately broad, and
            # plenty of real failures also tell you to try --help; letting the
            # version check win would report a logged-out CLI as UNKNOWN, and
            # UNKNOWN does not set reachable=False, so `--check` would exit 0
            # on a provider that cannot work.
            classified = classify_cli_failure(self.name, output)
            if classified is None and _NO_SUCH_SUBCOMMAND.search(output):
                return ProviderHealth(
                    provider=self.name,
                    configured=True,
                    detail="this CLI has no `auth status` subcommand to probe",
                )
            detail = classified.diagnosis if classified else (stderr.strip() or stdout.strip())
            return ProviderHealth(
                provider=self.name, configured=True, reachable=False, detail=detail
            )

        try:
            status = json.loads(stdout)
        except json.JSONDecodeError:
            status = None
        # Valid JSON of the wrong shape (null, a list, a bare string) parses
        # fine and then has no .get -- the same guard terminal_result_text
        # already applies to stream events.
        if not isinstance(status, dict):
            return ProviderHealth(
                provider=self.name,
                configured=True,
                detail="`claude auth status` returned output we could not parse",
            )

        if not status.get("loggedIn"):
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=False,
                detail="not logged in; run `claude auth login`",
            )

        described = ", ".join(
            f"{label}: {status[key]}"
            for key, label in (("authMethod", "auth"), ("subscriptionType", "plan"))
            if status.get(key)
        )
        return ProviderHealth(
            provider=self.name,
            configured=True,
            reachable=True,
            detail=described or "logged in",
        )

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

        allowed_tools = self.params.allowed_tools.copy()
        if self.params.tooluniverse:
            command.extend(["--mcp-config", json.dumps(self.params.tooluniverse.claude_mcp_config())])
            allowed_tools.extend(self.params.tooluniverse.claude_allowed_tools())
        if allowed_tools:
            command.extend(["--allowedTools", ",".join(allowed_tools)])

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

        self.require_available()

        if not query or not query.strip():
            raise ValueError("Research query must not be empty.")

        if self.params.tooluniverse:
            await asyncio.to_thread(self.params.tooluniverse.prepare, self.name)
        command = self._build_command()
        logger.info("Running Claude Code research (timeout=%ss)", self.timeout)
        logger.debug("Claude Code command: %s", " ".join(command))

        stdout, stderr, returncode = await self._run_process(command, query)

        if returncode != 0:
            # stderr is the CLI's own voice; from stdout take only the terminal
            # event. The rest of the stream is the model's report, and a report
            # that merely discusses usage limits is not a usage limit.
            classified = classify_cli_failure(
                self.name, f"{stderr}\n{terminal_result_text(stdout)}"
            )
            if classified is not None:
                logger.error(classified.actionable_message())
                raise classified
            raise ValueError(
                f"Claude Code exited with code {returncode}: "
                f"{stderr.strip() or '<no stderr>'}"
            )

        assistant_texts, data, tool_uses = self._parse_stream(stdout, self.name)
        markdown = self._report_text(assistant_texts, data)
        self._check_report_length(markdown, self.params.min_report_chars)

        run_metadata = self._extract_run_metadata(data)
        retrieval_calls = self._retrieval_calls(tool_uses, data.get("permission_denials"))
        run_metadata["tool_calls"] = len(tool_uses)
        run_metadata["retrieval_calls"] = retrieval_calls
        run_metadata.update(self.params.toolset_run_metadata())
        # How many separate assistant messages the report was assembled from.
        # More than one is normal for an agentic run (the model narrates between
        # tool calls); the count is provenance for how the report was assembled,
        # not a warning sign.
        run_metadata["assistant_text_blocks"] = len(assistant_texts)
        citations = self._extract_citations(markdown)
        unsuccessful_reason = self._unsuccessful_reason(
            markdown,
            citations,
            retrieval_calls=retrieval_calls,
            web_search_requests=run_metadata.get("web_search_requests", 0),
        )
        if unsuccessful_reason:
            run_metadata["unsuccessful_reason"] = unsuccessful_reason
            logger.warning("Claude Code did not produce a report: %s", unsuccessful_reason)

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
            answer_successful=False if unsuccessful_reason else None,
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
            # A spent usage allowance can stall a run rather than fail it, and
            # then it arrives here looking like slowness. Say so, so the reader
            # checks the allowance before chasing latency.
            raise ValueError(
                f"Claude Code run timed out after {self.timeout}s. If this repeats, "
                f"check the account with `deep-research-client providers --check "
                f"--provider {self.name}` -- a spent usage limit can stall a run "
                f"instead of failing it."
            )

        # returncode is set once communicate() has returned.
        returncode = -1 if process.returncode is None else process.returncode
        return (
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            returncode,
        )

    @staticmethod
    def _parse_stream(stdout: str, provider: str = "claude_code") -> ParsedStream:
        """Parse Claude Code's ``stream-json`` output into text blocks and metadata.

        The stream is JSON Lines: one event per line. ``assistant`` events carry
        the model's own output (text, thinking, and tool_use blocks, split across
        events); the terminal ``result`` event carries the run's provenance and
        error status.

        Args:
            stdout: Raw stdout from ``claude --print --output-format stream-json``.
            provider: Provider name to attribute a classified failure to.

        Returns:
            The assistant text blocks, the terminal result event, and the names
            of the tools the model called.

        Raises:
            ValueError: If the output is empty, contains no terminal result event,
                or that event reports an error.
        """
        text = stdout.strip()
        if not text:
            raise ValueError("Claude Code returned empty output.")

        assistant_texts: List[str] = []
        tool_uses: List[str] = []
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
                # `message` can be an explicit null, and `content` is a plain
                # string in the single-text-block form the Anthropic message
                # format permits. Neither is what the CLI emits today, but
                # iterating a string would silently yield characters that the
                # block guard below drops -- a truncated report with no signal.
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                elif not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        block_text = str(block.get("text") or "")
                        if block_text.strip():
                            assistant_texts.append(block_text)
                    elif block.get("type") == "tool_use":
                        tool_uses.append(str(block.get("name") or "<unnamed>"))
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
            classified = classify_cli_failure(provider, f"{subtype} {detail}")
            if classified is not None:
                raise classified
            raise ValueError(f"Claude Code reported an error ({subtype}): {detail}")

        return ParsedStream(assistant_texts, result_event, tool_uses)

    @staticmethod
    def _report_text(assistant_texts: List[str], data: dict) -> str:
        """Assemble the markdown report from every assistant text block.

        Falls back to the terminal event's ``result`` field only when the stream
        carried no assistant text at all. That field holds just the agent's final
        message, so preferring the joined blocks is what keeps a report from
        being truncated to its closing remark.

        Joining everything means any narration the model emits between tool calls
        ("Let me search for X...") lands in the report too. That is the deliberate
        trade: including narration is a cosmetic cost, whereas picking a single
        block risks dropping the research. Note that citation extraction runs over
        the joined text, so a URL mentioned only in narration is counted.

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
        return str(result).strip()

    @staticmethod
    def _check_report_length(markdown: str, min_chars: int) -> None:
        """Reject a report too short to be a plausible research result.

        Silent success on an empty run is the costly failure mode: the caller
        gets a well-formed file with real provenance and no content. Failing here
        turns that into a non-zero exit.

        This is an emptiness check, not a quality one: a short "I could not find
        sufficient information" reply passes it.

        Because the run that failed has already been paid for, the rejected text
        is logged in full and previewed in the exception, so the cause is
        diagnosable without a second run.

        Raises:
            ValueError: If ``markdown`` is shorter than ``min_chars``.
        """
        if min_chars <= 0:
            return

        report = markdown.strip()
        if len(report) >= min_chars:
            return

        logger.warning(
            "Claude Code report is below the min_report_chars threshold (%d < %d). "
            "Full text of the rejected report: %s",
            len(report),
            min_chars,
            report or "<empty>",
        )
        preview = report[:200] + ("..." if len(report) > 200 else "")
        raise ValueError(
            f"Claude Code returned a {len(report)}-character report, below the "
            f"min_report_chars threshold of {min_chars}. This usually means the "
            "run failed, was cut short, or produced no research. Lower or disable "
            "the threshold (min_report_chars=0) if short answers are expected. "
            f"Report text: {preview!r}"
        )

    @staticmethod
    def _retrieval_calls(tool_uses: List[str], permission_denials: Optional[list]) -> int:
        """Count the tool calls that could have retrieved something.

        A denied call retrieved nothing, and neither did a bookkeeping call such
        as ``TodoWrite``, so a run that tried ``Bash``, was refused and then
        answered from memory has no retrieval calls. Denials are matched to
        calls by tool name, one denial cancelling one call.

        Args:
            tool_uses: Names of the tools the model called, repeats included.
            permission_denials: The result event's ``permission_denials``.

        Returns:
            How many calls were neither denied nor bookkeeping.

        >>> ClaudeCodeProvider._retrieval_calls(["WebSearch", "WebFetch"], None)
        2
        >>> ClaudeCodeProvider._retrieval_calls(["Bash"], [{"tool_name": "Bash"}])
        0
        >>> ClaudeCodeProvider._retrieval_calls(["TodoWrite", "WebSearch", "WebSearch"],
        ...                                     [{"tool_name": "WebSearch"}])
        1
        """
        remaining = list(tool_uses)
        for denial in permission_denials or []:
            name = denial.get("tool_name") if isinstance(denial, dict) else None
            if name in remaining:
                remaining.remove(name)
        return sum(1 for name in remaining if name not in _NON_RETRIEVAL_TOOLS)

    @staticmethod
    def _unsuccessful_reason(
        markdown: str,
        citations: List[str],
        retrieval_calls: int,
        web_search_requests: int,
    ) -> Optional[str]:
        """Say why a completed run's text is not a research report, if it is not.

        The CLI exits 0 and reports success whatever the model wrote, so a
        refusal or a from-memory answer used to be written out as a valid
        report (issue #101). Two shapes are recognised, both requiring that the
        text cites nothing -- no URL, PMID or DOI. A report with even one source
        is left alone, since judging its quality is not this check's job:

        - no retrieval call and no web search at all: nothing was retrieved, so
          whatever the text says was written from memory (issue #102);
        - refusal phrasing aimed at the report itself: the model declined,
          typically after an output safety classifier stopped its first attempt.

        The text alone cannot prove the model refused, which is why the
        phrasing counts only alongside the missing sources, and why the result
        is marked unsuccessful (output kept, exit code 4) rather than raised.

        Args:
            markdown: The assembled report text.
            citations: URLs extracted from it.
            retrieval_calls: Tool calls that could have retrieved something
                (see :meth:`_retrieval_calls`).
            web_search_requests: Server-side web searches the run reported.

        Returns:
            A one-line reason when the text is not a report, otherwise None.

        >>> ClaudeCodeProvider._unsuccessful_reason("From memory.", [], 0, 0)
        'no retrieval calls, web searches or cited sources: the text was written from memory, not researched'
        >>> ClaudeCodeProvider._unsuccessful_reason(
        ...     "I'm not going to regenerate that narrative.", [], retrieval_calls=3,
        ...     web_search_requests=2,
        ... )
        'the model declined to write the report, and it cites no sources'
        >>> ClaudeCodeProvider._unsuccessful_reason(
        ...     "Mentions a safety classifier. https://example.org", ["https://example.org"], 0, 0,
        ... ) is None
        True
        >>> ClaudeCodeProvider._unsuccessful_reason("Researched, uncited.", [], 4, 0) is None
        True

        Hedging, disclaimers and ordinary prose in a researched report are not
        refusals, and a PMID is a source:

        >>> ClaudeCodeProvider._unsuccessful_reason(
        ...     "I cannot provide a definitive answer. The outbreak was halted by vector control.",
        ...     [], retrieval_calls=5, web_search_requests=3,
        ... ) is None
        True
        >>> ClaudeCodeProvider._unsuccessful_reason(
        ...     "I cannot provide medical advice. This report summarizes the literature.",
        ...     [], retrieval_calls=5, web_search_requests=0,
        ... ) is None
        True
        >>> ClaudeCodeProvider._unsuccessful_reason(
        ...     "Elongation is halted by a safety classifier-like checkpoint (PMID: 12345).",
        ...     [], retrieval_calls=2, web_search_requests=0,
        ... ) is None
        True
        """
        if citations or _SOURCE_ID_PATTERN.search(markdown):
            return None
        if retrieval_calls == 0 and web_search_requests == 0:
            return (
                "no retrieval calls, web searches or cited sources: the text was "
                "written from memory, not researched"
            )
        if _REFUSAL_PATTERN.search(markdown):
            return "the model declined to write the report, and it cites no sources"
        return None

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
        # for a tool outside the allowlist, was refused, and gave up. Which tool
        # was refused is the actionable half -- it names what to add to
        # allowed_tools -- so record the names alongside the count.
        denials = data.get("permission_denials")
        if denials:
            metadata["permission_denials"] = len(denials)
            denied_tools = sorted(
                {
                    denial["tool_name"]
                    for denial in denials
                    if isinstance(denial, dict) and denial.get("tool_name")
                }
            )
            if denied_tools:
                metadata["denied_tools"] = denied_tools

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
