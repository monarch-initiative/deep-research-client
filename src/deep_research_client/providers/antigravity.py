"""Google Antigravity research provider.

Like Claude Code, Google Antigravity is a local agentic command-line tool (the
``agy`` binary) rather than an HTTP API. Antigravity ships no dedicated "deep
research" product, but ``agy``'s headless ("print") mode is a close structural
twin of Claude Code's: one prompt in, one response out, with the agent's own
tools (``search_web``, ``read_url_content``, a browser family) driving a
multi-step research workflow. This provider drives it the same way the
``claude_code`` provider drives ``claude``.

Three things differ from Claude Code and shape this module:

1. **No system-prompt flag.** ``agy`` has no ``--append-system-prompt``, so the
   research system prompt and the inline-report directive are prepended to the
   prompt text itself, which is passed as the ``-p`` argument.

2. **Permissions, not an allowlist.** ``agy`` has no ``--allowedTools`` flag.
   Its default headless permission mode (``request-review``) already runs
   read-only retrieval tools freely while *soft-denying* mutating ones (file
   writes, shell): the denied call is skipped, a notice is printed to stderr,
   and the run still exits 0. So the out-of-the-box behaviour is read-only
   without any flag, and this provider never passes
   ``--dangerously-skip-permissions`` unless explicitly asked. (A soft-denied
   *final* action can leave an empty response on an otherwise-successful run;
   ``min_report_chars`` catches that.)

3. **Different output schema.** ``agy --output-format stream-json`` emits
   events keyed by an ``event`` field (``init`` / ``step_update`` / ``result``).
   The finished report is the terminal ``result`` event's ``response`` field
   (not streamed text deltas, which are absent in practice); the stream is read
   for the tool tally and provenance. Failures are classified from the
   ``result`` event's ``status`` plus a structured ``AGY_ERROR: {...}`` line on
   stderr, which carries a gRPC-style ``status`` and a ``reason``.

No provider API key is required: authentication and billing are handled by the
local ``agy`` installation (authenticate once with an interactive session).
"""

import asyncio
import json
import logging
import shutil
from datetime import datetime
from typing import List, NamedTuple, Optional

from ._cli_agent import CLIAgentResearchProvider, INLINE_REPORT_DIRECTIVE
from ..exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderError,
    ProviderNotInstalledError,
    ProviderQuotaError,
    ProviderTimeoutError,
    ProviderTransientError,
)
from ..models import ResearchResult, ProviderConfig, ProviderHealth
from ..provider_params import AntigravityParams
from ..model_cards import ProviderModelCards, create_antigravity_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# The tools ``agy`` exposes that actually retrieve something, for the from-memory
# guard. ``agy``'s tool surface is large and open-ended (~60 tools, many of them
# browser-control or bookkeeping), so retrieval is classified positively -- a
# small known set -- rather than by denylist. These are the read-only tools a
# research run legitimately uses; everything else (writes, shell, subagents,
# task management) does not count as retrieval.
_RETRIEVAL_TOOLS = frozenset(
    {
        "search_web",
        "read_url_content",
        "open_browser_url",
        "read_browser_page",
        "browser_get_dom",
        "browser_scroll_dom",
        "capture_browser_screenshot",
        "list_browser_pages",
        "read_resource",
        "list_resources",
        "grep_search",
        "view_file",
        "list_dir",
        "find_by_name",
        "call_mcp_tool",
    }
)

# The model a run uses when the caller names none. ``agy``'s own default is
# opaque -- it is whatever the CLI was last configured to use, is absent from
# the run's output (init event, result event), its settings file and
# ``agy models``, and so cannot be read back -- which left a default run's
# provenance showing only a meaningless placeholder. This provider therefore
# never relies on it: it always forwards an explicit ``--model``, defaulting to
# this slug, so the report always records the model that actually ran. Change
# the default here (and the model card) if agy's recommended model moves on.
DEFAULT_MODEL = "gemini-4-carbon-medium"

# Aliases that mean "use the provider default" rather than a concrete ``agy``
# slug. They resolve to :data:`DEFAULT_MODEL`, not to agy's opaque default.
_DEFAULT_MODEL_ALIASES = {
    "antigravity",
    "agy",
    "default",
    "antigravity-default",
    "gemini-4-carbon",
}

# ``agy`` writes a machine-readable failure line to stderr, e.g.
# ``AGY_ERROR: {"short_error":"...","status":"DEADLINE_EXCEEDED",...}``. The
# ``status`` is a gRPC-style code and is the most reliable classification
# signal; ``reason`` and ``short_error`` refine it.
_AGY_ERROR_PREFIX = "AGY_ERROR:"

# gRPC-style status codes ``agy`` reports, mapped to our typed errors. Codes not
# listed here fall through to text matching and then to a generic ValueError.
_STATUS_ERROR_CLASSES: dict[str, type[ProviderError]] = {
    "UNAUTHENTICATED": ProviderAuthError,
    "PERMISSION_DENIED": ProviderAuthError,
    "RESOURCE_EXHAUSTED": ProviderQuotaError,
    "DEADLINE_EXCEEDED": ProviderTimeoutError,
    "UNAVAILABLE": ProviderTransientError,
    "INTERNAL": ProviderTransientError,
    "ABORTED": ProviderTransientError,
}

# Best-effort text fallbacks for failures that arrive without a structured
# status (older CLIs, network errors printed as prose). Ordered most specific
# first. Grounded against observed ``agy`` output where possible and kept
# conservative otherwise.
_TEXT_ERROR_PATTERNS: list[tuple[tuple[str, ...], type[ProviderError]]] = [
    (("authentication required", "not authenticated", "unauthenticated",
      "please log in", "log in to", "run `agy`"), ProviderAuthError),
    (("insufficient credit", "credit balance", "billing", "payment required"),
     ProviderBillingError),
    (("resource exhausted", "quota", "usage limit", "rate limit",
      "too many requests"), ProviderQuotaError),
    (("overloaded", "service unavailable", "temporarily unavailable",
      "try again later", "503"), ProviderTransientError),
]

# Statuses the terminal ``result`` event reports for a run that produced a
# usable response. Anything else is a failure.
_SUCCESS_STATUS = "SUCCESS"


def parse_agy_error(stderr: str) -> dict:
    """Pull the structured ``AGY_ERROR: {...}`` object out of stderr, if any.

    Args:
        stderr: Raw stderr from the ``agy`` run, possibly multi-line.

    Returns:
        The parsed error object, or ``{}`` when there is none or it is
        unparseable.

    >>> parse_agy_error('note\\nAGY_ERROR: {"status":"DEADLINE_EXCEEDED","reason":"print_timeout"}')['status']
    'DEADLINE_EXCEEDED'
    >>> parse_agy_error('nothing structured here')
    {}
    """
    for line in stderr.splitlines():
        idx = line.find(_AGY_ERROR_PREFIX)
        if idx == -1:
            continue
        payload = line[idx + len(_AGY_ERROR_PREFIX):].strip()
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def classify_agy_failure(
    provider: str,
    stderr: str,
    result_status: Optional[str] = None,
    result_error: Optional[str] = None,
) -> Optional[ProviderError]:
    """Classify an ``agy`` failure from its stderr and terminal result event.

    The structured ``AGY_ERROR`` status is tried first, then the combined text
    of the error message and stderr. ``result_status``/``result_error`` come
    from the terminal ``result`` event; ``stderr`` is the CLI's own voice.

    Args:
        provider: Name of the provider that failed.
        stderr: Raw stderr from the run.
        result_status: The terminal event's ``status``, if parsed.
        result_error: The terminal event's ``error`` string, if any.

    Returns:
        A typed error, or None when nothing is recognised.

    >>> err = classify_agy_failure("antigravity",
    ...     'AGY_ERROR: {"status":"DEADLINE_EXCEEDED","reason":"print_timeout","short_error":"timed out"}')
    >>> type(err).__name__
    'ProviderTimeoutError'
    >>> type(classify_agy_failure("antigravity", 'AGY_ERROR: {"status":"UNAUTHENTICATED"}')).__name__
    'ProviderAuthError'
    >>> classify_agy_failure("antigravity", "some unrecognised message") is None
    True
    """
    structured = parse_agy_error(stderr)
    status = structured.get("status")
    detail = (
        structured.get("short_error")
        or result_error
        or (stderr.strip().splitlines()[0] if stderr.strip() else "")
        or (result_status or "")
    )

    if status and status in _STATUS_ERROR_CLASSES:
        return _STATUS_ERROR_CLASSES[status](provider, detail or status)

    haystack = f"{result_error or ''}\n{stderr}".lower()
    for needles, error_class in _TEXT_ERROR_PATTERNS:
        if any(needle in haystack for needle in needles):
            return error_class(provider, detail or needles[0])

    return None


class ParsedAgyStream(NamedTuple):
    """What a ``stream-json`` ``agy`` run produced, as read by ``_parse_stream``."""

    #: The finished report, from the terminal ``result`` event's ``response``.
    response: str
    #: The terminal ``result`` event payload: provenance and error status.
    result: dict
    #: Names of the tools the model called, one per distinct call.
    tool_uses: List[str]
    #: The model id reported by the ``init`` event, if any.
    model: Optional[str]


class AntigravityProvider(CLIAgentResearchProvider):
    """Provider that runs the local Google Antigravity ``agy`` CLI to research."""

    #: Retrieval is classified positively from a known read-only set (see
    #: :data:`_RETRIEVAL_TOOLS`), because ``agy``'s tool surface is too large
    #: and open-ended for a reliable denylist.
    _RETRIEVAL_TOOLS = _RETRIEVAL_TOOLS

    #: Auto-detection probes the default ``agy`` binary and honours an opt-out,
    #: neither of which an instance's is_available() can see.
    registration_requirement = (
        "requires the local Google Antigravity CLI (agy), with "
        "DISABLE_ANTIGRAVITY_PROVIDER unset"
    )

    #: Nothing to run is an install problem, not a missing key.
    unavailable_error = ProviderNotInstalledError
    timeout_meaning = (
        "how long the whole agy run may take before it is killed (default 1800 s)"
    )

    def __init__(
        self, config: ProviderConfig, params: Optional[AntigravityParams] = None
    ):
        """Initialize the Antigravity provider.

        Args:
            config: Provider configuration (no API key required).
            params: Antigravity-specific parameters.
        """
        self.params = params or AntigravityParams()
        super().__init__(config, self.params.model)

        self.agy_executable = self.params.agy_executable
        # ProviderConfig.timeout wins when set; otherwise use the params default.
        self.timeout = config.timeout or self.params.timeout

        logger.debug(
            "Initializing Antigravity provider (executable=%s, skip_permissions=%s)",
            self.agy_executable,
            self.params.skip_permissions,
        )

    @classmethod
    def _is_retrieval_tool(cls, name: str) -> bool:
        """Count only the known read-only retrieval tools (positive match)."""
        return name in cls._RETRIEVAL_TOOLS

    def get_default_model(self) -> str:
        """Get the default model identifier for this provider.

        A concrete ``agy`` slug, never a placeholder: the provider always sends
        an explicit ``--model`` (see :data:`DEFAULT_MODEL`), so provenance names
        the model that ran rather than agy's unreadable default.
        """
        return DEFAULT_MODEL

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for the Antigravity provider."""
        return create_antigravity_model_cards()

    def is_available(self) -> bool:
        """Check whether the Antigravity ``agy`` CLI is available.

        Returns:
            True if the provider is enabled and the ``agy`` executable is
            resolvable on the PATH (or as an absolute path).
        """
        if not self.config.enabled:
            return False

        if shutil.which(self.agy_executable) is None:
            logger.warning(
                "Antigravity executable %r not found in PATH", self.agy_executable
            )
            return False

        return True

    def unavailable_reason(self) -> str:
        """Explain the missing CLI, which no API key would fix.

        Returns:
            Human-readable explanation suitable for an error message.
        """
        if not self.config.enabled:
            return super().unavailable_reason()
        return (
            f"the {self.agy_executable!r} CLI was not found on PATH; "
            f"install Google Antigravity to use this provider"
        )

    async def check_health(self) -> ProviderHealth:
        """Ask the CLI to list models, as a cheap authenticated liveness probe.

        ``agy models`` fetches the available models over the network without
        spending model tokens, so -- like Claude Code's ``auth status`` -- it
        tells us whether the local install is logged in and can reach the
        service before a run rather than only after one fails. ``agy`` has no
        dedicated auth-status subcommand, so this is the closest free probe.

        Returns:
            Health record for this provider.
        """
        if not self.is_available():
            return ProviderHealth(
                provider=self.name,
                configured=False,
                reachable=False,
                detail=self.unavailable_reason(),
            )

        try:
            process = await asyncio.create_subprocess_exec(
                self.agy_executable, "models",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as e:
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=False,
                detail=f"could not run {self.agy_executable!r}: {e}",
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self.params.health_probe_timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=False,
                detail=(
                    f"`agy models` did not answer within "
                    f"{self.params.health_probe_timeout}s"
                ),
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return self._health_from_models(stdout, stderr, process.returncode or 0)

    def _health_from_models(
        self, stdout: str, stderr: str, returncode: int
    ) -> ProviderHealth:
        """Turn ``agy models`` output into a health record.

        Args:
            stdout: Raw stdout from the probe.
            stderr: Raw stderr from the probe.
            returncode: Exit status of the probe.

        Returns:
            Health record for this provider.
        """
        if returncode != 0:
            classified = classify_agy_failure(self.name, stderr)
            detail = (
                classified.diagnosis
                if classified
                else (stderr.strip() or stdout.strip() or "`agy models` failed")
            )
            return ProviderHealth(
                provider=self.name, configured=True, reachable=False, detail=detail
            )

        # Each model is a "slug\tDisplay Name" line; the header "Fetching..."
        # goes to stdout too, so count only lines that look like a slug row.
        model_lines = [
            line for line in stdout.splitlines()
            if "\t" in line and line.strip()
        ]
        if not model_lines:
            return ProviderHealth(
                provider=self.name,
                configured=True,
                reachable=False,
                detail="`agy models` returned no models; the install may not be logged in",
            )
        return ProviderHealth(
            provider=self.name,
            configured=True,
            reachable=True,
            detail=f"logged in; {len(model_lines)} models available",
        )

    def _resolve_cli_model(self) -> str:
        """Resolve the concrete ``--model`` slug to forward to the CLI.

        Always a concrete slug, never ``None``: an unset model or a
        provider-default alias resolves to :data:`DEFAULT_MODEL`, and any other
        value is forwarded as the user's raw slug (e.g.
        ``gemini-3.8-flash-medium``). This is what keeps agy from falling back
        to its own opaque default, so the report records the model that ran.
        """
        raw_model = (self.params.model or "").strip()
        if not raw_model or raw_model.lower() in _DEFAULT_MODEL_ALIASES:
            return DEFAULT_MODEL
        return raw_model

    def _compose_prompt(self, query: str) -> str:
        """Fold the system prompt and inline-report directive into the prompt.

        ``agy`` has no system-prompt flag, so the instructions ride in the
        prompt text itself, ahead of the research task.
        """
        system_prompt = self.params.system_prompt or DEFAULT_RESEARCH_SYSTEM_PROMPT
        return (
            f"{system_prompt}\n\n{INLINE_REPORT_DIRECTIVE}\n\n"
            f"Research task:\n{query}"
        )

    def _build_command(self, prompt: str) -> List[str]:
        """Build the ``agy`` command-line invocation, prompt included.

        Unlike Claude Code (which reads the prompt from stdin), ``agy`` takes
        the prompt as the ``-p`` argument, so the composed prompt is part of the
        command here.

        Args:
            prompt: The fully composed prompt (system + directive + task).

        Returns:
            The argument list to pass to ``asyncio.create_subprocess_exec``.
        """
        command: List[str] = [
            self.agy_executable,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
        ]

        if self.params.skip_permissions:
            # Bypasses the read-only default; only for trusted, sandboxed use.
            command.append("--dangerously-skip-permissions")

        if self.params.disable_slash_commands:
            # Keep a (possibly untrusted) query from triggering slash commands
            # or skill expansion.
            command.append("--disable-slash-commands")

        cli_model = self._resolve_cli_model()
        if cli_model:
            command.extend(["--model", cli_model])

        if self.params.effort:
            command.extend(["--effort", self.params.effort])

        # Default is agy's own 0s (no limit); we rely on the outer process
        # timeout instead, so a too-short cap cannot truncate a long run.
        if self.params.print_timeout:
            command.extend(["--print-timeout", self.params.print_timeout])

        if self.params.settings_override:
            command.extend(["--settings", self.params.settings_override])

        for directory in self.params.add_dirs:
            command.extend(["--add-dir", directory])

        command.extend(self.params.extra_args)

        return command

    async def research(self, query: str) -> ResearchResult:
        """Perform research by running the Antigravity ``agy`` CLI.

        Args:
            query: The research question or topic.

        Returns:
            ResearchResult with the markdown report and extracted citations.

        Raises:
            ValueError: If the provider is unavailable, the query is empty, or
                the ``agy`` run fails.
        """
        start_time = datetime.now()

        self.require_available()

        if not query or not query.strip():
            raise ValueError("Research query must not be empty.")

        prompt = self._compose_prompt(query)
        command = self._build_command(prompt)
        logger.info("Running Antigravity research (timeout=%ss)", self.timeout)
        logger.debug("Antigravity command: %s", " ".join(command[:2] + command[3:]))

        stdout, stderr, returncode = await self._run_process(command)

        response, result, tool_uses, init_model = self._parse_stream(stdout)
        status = result.get("status")

        if returncode != 0 or (status and status != _SUCCESS_STATUS):
            classified = classify_agy_failure(
                self.name, stderr, status, result.get("error")
            )
            if classified is not None:
                logger.error(classified.actionable_message())
                raise classified
            detail = result.get("error") or stderr.strip() or f"status {status}"
            raise ValueError(
                f"agy run failed (status={status}, exit={returncode}): {detail}"
            )

        markdown = response.strip()
        self._check_report_length(markdown, self.params.min_report_chars)

        retrieval_calls = self._retrieval_calls(tool_uses, None)
        run_metadata = self._extract_run_metadata(result, init_model)
        run_metadata["tool_calls"] = len(tool_uses)
        run_metadata["retrieval_calls"] = retrieval_calls

        citations = self._extract_citations(markdown)
        unsuccessful_reason = self._unsuccessful_reason(
            markdown,
            citations,
            retrieval_calls=retrieval_calls,
            web_search_requests=0,
        )
        if unsuccessful_reason:
            run_metadata["unsuccessful_reason"] = unsuccessful_reason
            logger.info(
                "Antigravity did not produce a report: %s", unsuccessful_reason
            )

        model_label = init_model or self.model

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            "Antigravity research completed in %.1fs (%d chars, %d citations, model=%s)",
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

    @staticmethod
    def _parse_stream(stdout: str) -> ParsedAgyStream:
        """Parse ``agy``'s ``stream-json`` output into report, tools and metadata.

        The stream is NDJSON: one event per line, each keyed by ``event``.
        ``init`` carries the model; ``step_update`` events of ``step_type``
        ``tool`` name the tools called (each call appears twice, as ``ACTIVE``
        then ``DONE`` sharing a ``step_index``, so calls are deduplicated by
        index); the terminal ``result`` event carries the report ``response``
        and provenance.

        Args:
            stdout: Raw stdout from ``agy -p --output-format stream-json``.

        Returns:
            The report text, terminal result payload, tool-call names, and the
            model id from the init event.

        Raises:
            ValueError: If the output is empty or carries no terminal ``result``
                event (so the run did not complete).
        """
        text = stdout.strip()
        if not text:
            raise ValueError("agy returned empty output.")

        result_payload: Optional[dict] = None
        init_model: Optional[str] = None
        # Keyed by step_index so the ACTIVE/DONE pair of one call counts once.
        tool_calls: dict[int, str] = {}
        fallback_tools: List[str] = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping unparseable line in agy output: %.200s", line
                )
                continue
            if not isinstance(event, dict):
                continue

            kind = event.get("event")
            if kind == "init":
                init = event.get("init")
                if isinstance(init, dict):
                    init_model = init.get("model") or init_model
            elif kind == "step_update":
                su = event.get("step_update")
                if not isinstance(su, dict) or su.get("step_type") != "tool":
                    continue
                name = str(su.get("tool_name") or "<unnamed>")
                index = su.get("step_index")
                if isinstance(index, int):
                    tool_calls[index] = name
                else:
                    fallback_tools.append(name)
            elif kind == "result":
                payload = event.get("result")
                if isinstance(payload, dict):
                    result_payload = payload

        if result_payload is None:
            raise ValueError(
                "Could not parse agy output: the stream contained no terminal "
                "'result' event, so the run did not complete."
            )

        tool_uses = list(tool_calls.values()) + fallback_tools
        response = str(result_payload.get("response") or "")
        return ParsedAgyStream(response, result_payload, tool_uses, init_model)

    @staticmethod
    def _extract_run_metadata(result: dict, init_model: Optional[str]) -> dict:
        """Extract run provenance from the terminal ``result`` event.

        Args:
            result: The terminal ``result`` event payload.
            init_model: The model id from the ``init`` event, if any.

        Returns:
            A metadata dict with any of: status, num_turns, duration_seconds,
            conversation_id, usage, model.
        """
        metadata: dict = {}

        if init_model:
            metadata["model"] = init_model
        if result.get("status"):
            metadata["status"] = result["status"]
        if result.get("num_turns") is not None:
            metadata["num_turns"] = result["num_turns"]
        if result.get("duration_seconds") is not None:
            metadata["duration_seconds"] = result["duration_seconds"]
        if result.get("conversation_id"):
            metadata["conversation_id"] = result["conversation_id"]

        usage = result.get("usage")
        if isinstance(usage, dict) and usage:
            metadata["usage"] = usage

        return metadata
