"""Summary statistics mined from provider agent transcripts.

A transcript is the step-by-step record of what an agent actually did: every
tool it called, every skill it invoked, every shell command, search, and file
write. Kept as an artifact it is a wall of JSON; what a reader usually wants
from it is the shape of the run — which tools were used and how often, which
skills fired, what failed, what the agent had available but never touched.

This module turns a transcript into that summary. It is deliberately
independent of any provider SDK: the input is plain decoded JSON, so a
transcript can be summarized without installing the agent framework that
produced it, and a shape this module does not recognize is counted rather than
dropped.

Entry shapes come from the OpenScientist transcript union, whose entries are
JSON objects discriminated by a ``type`` field: ``tool_call``, ``tool_result``,
``shell_execution``, ``file_change``, ``web_search``, ``session_init``, and so
on. Skills are ``tool_call`` entries whose tool is ``Skill``, carrying the skill
name in ``arguments.skill``.

Example:
    >>> entries = [
    ...     {"type": "tool_call", "id": "1", "tool": "Skill",
    ...      "arguments": {"skill": "gene-set-enrichment"}},
    ...     {"type": "tool_result", "call_id": "1", "output": "", "success": True},
    ...     {"type": "tool_call", "id": "2",
    ...      "tool": "mcp__openscientist__search_pubmed", "arguments": {}},
    ...     {"type": "tool_result", "call_id": "2", "output": "", "success": False},
    ... ]
    >>> stats = summarize_transcript(entries)
    >>> stats.distinct_tools
    ['Skill', 'search_pubmed']
    >>> stats.skills_used
    ['gene-set-enrichment']
    >>> stats.tool_calls, stats.failed_tool_calls
    (2, 1)
"""

from __future__ import annotations

import base64
import binascii
from collections import Counter
import json
from pathlib import Path
import shlex
from typing import Any, Iterable, Optional, Sequence, TypeGuard

from pydantic import BaseModel, Field

# A transcript artifact is recognized by name; OpenScientist writes
# ``provenance/iter<N>_transcript.json`` and ``provenance/report_transcript.json``.
TRANSCRIPT_NAME_FRAGMENT = "transcript"

# Tool-call entries whose tool is this invoke a named skill.
SKILL_TOOL_NAME = "Skill"

# Shell wrappers whose first word says nothing about what was run.
_SHELL_WRAPPERS = frozenset({"sudo", "env", "time", "nohup", "xargs", "nice"})


class ToolUsage(BaseModel):
    """How one tool was used across a transcript.

    Attributes:
        name: Short tool name, with any MCP server prefix stripped.
        qualified_names: Every fully qualified spelling seen for this tool.
        calls: Number of invocations.
        failures: Invocations whose paired result reported failure.
        server: MCP server the tool belongs to, when the entry records one.
            Two servers exposing the same short tool name collapse onto one
            entry and the last one seen wins; ``qualified_names`` still holds
            both spellings, so nothing is lost outright.
        total_duration_ms: Summed result durations, when reported.
    """

    name: str
    qualified_names: list[str] = Field(default_factory=list)
    calls: int = 0
    failures: int = 0
    server: Optional[str] = None
    total_duration_ms: Optional[int] = None

    @property
    def successes(self) -> int:
        """Invocations that did not report a failure."""
        return self.calls - self.failures


class TranscriptStats(BaseModel):
    """What an agent did, tallied from one or more transcripts.

    Every list field is sorted and deduplicated, so two runs of the same
    workload produce comparable output.
    """

    sources: list[str] = Field(
        default_factory=list, description="Transcript names this summary covers"
    )
    entries: int = Field(default=0, description="Total transcript entries read")
    entry_types: dict[str, int] = Field(
        default_factory=dict, description="Entry count by transcript ``type``"
    )

    tools: list[ToolUsage] = Field(
        default_factory=list, description="Per-tool usage, most-called first"
    )
    tool_calls: int = Field(default=0, description="Total tool invocations")
    failed_tool_calls: int = Field(
        default=0, description="Invocations whose result reported failure"
    )
    mcp_servers: list[str] = Field(
        default_factory=list, description="MCP servers whose tools were called"
    )

    skill_counts: dict[str, int] = Field(
        default_factory=dict, description="Invocation count per named skill"
    )

    shell_commands: dict[str, int] = Field(
        default_factory=dict, description="Shell invocations by program name"
    )
    failed_shell_commands: int = Field(
        default=0, description="Shell executions with a non-zero exit code"
    )
    web_searches: list[str] = Field(
        default_factory=list, description="Distinct web search queries issued"
    )
    files_changed: dict[str, list[str]] = Field(
        default_factory=dict, description="Paths touched, keyed by change kind"
    )

    models: list[str] = Field(
        default_factory=list, description="Models that produced output"
    )
    subagent_calls: int = Field(
        default=0, description="Collaborating-agent invocations"
    )
    tasks: dict[str, int] = Field(
        default_factory=dict, description="Task count by task type"
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict, description="Summed token counters, when reported"
    )

    available_tools: list[str] = Field(
        default_factory=list, description="Tools the session declared at init"
    )
    available_skills: list[str] = Field(
        default_factory=list, description="Slash commands/skills declared at init"
    )
    available_agents: list[str] = Field(
        default_factory=list, description="Subagents declared at init"
    )
    unknown_entries: int = Field(
        default=0,
        description="Entries the producing translator could not classify",
    )
    unrecognized_types: dict[str, int] = Field(
        default_factory=dict,
        description="Entry types this summarizer has no handling for",
    )

    @property
    def distinct_tools(self) -> list[str]:
        """Sorted names of every tool actually called."""
        return sorted(tool.name for tool in self.tools)

    @property
    def skills_used(self) -> list[str]:
        """Sorted names of every skill invoked."""
        return sorted(self.skill_counts)

    @property
    def unused_available_tools(self) -> list[str]:
        """Declared tools that were never called.

        The gap between what a session was given and what it reached for is
        usually more interesting than either list alone.
        """
        used = {tool.name for tool in self.tools} | {
            qualified for tool in self.tools for qualified in tool.qualified_names
        }
        return sorted(set(self.available_tools) - used)

    @property
    def tool_success_rate(self) -> Optional[float]:
        """Fraction of tool calls that did not fail, or None if none were made."""
        if not self.tool_calls:
            return None
        return (self.tool_calls - self.failed_tool_calls) / self.tool_calls

    def render_markdown(self) -> str:
        """Render the summary as a markdown section.

        Returns:
            Markdown suitable for appending to a report or committing beside
            the transcripts it describes.
        """
        return _render_markdown(self)


def summarize_transcript(
    entries: Sequence[dict[str, Any]],
    source: Optional[str] = None,
) -> TranscriptStats:
    """Summarize one decoded transcript.

    Args:
        entries: Decoded transcript entries, each a dict with a ``type`` key.
        source: Name recorded in the summary, typically the artifact filename.

    Returns:
        The summary for this transcript alone.
    """
    return summarize_transcripts({source or "transcript": entries})


def summarize_transcripts(
    transcripts: dict[str, Sequence[dict[str, Any]]],
) -> TranscriptStats:
    """Summarize several transcripts as one run.

    A job writes one transcript per iteration plus one for report generation;
    the interesting totals are across all of them.

    Args:
        transcripts: Decoded entries keyed by source name.

    Returns:
        The merged summary.
    """
    accumulator = _Accumulator()
    for source in sorted(transcripts):
        accumulator.add_source(source)
        for entry in transcripts[source]:
            accumulator.consume(entry)
    return accumulator.finish()


def summarize_artifacts(artifacts: Iterable[Any]) -> TranscriptStats:
    """Summarize every transcript artifact in a research result.

    Non-transcript artifacts are ignored, so this can be handed a whole
    ``ResearchResult.artifacts`` list.

    Args:
        artifacts: Objects exposing ``filename`` and ``content_base64``.

    Returns:
        The merged summary; empty when no transcript artifact is present.
    """
    transcripts: dict[str, Sequence[dict[str, Any]]] = {}
    for artifact in artifacts:
        filename = getattr(artifact, "filename", "")
        if not is_transcript_name(filename):
            continue
        raw = getattr(artifact, "content_base64", "")
        transcripts[filename] = _decode_entries(raw, filename)
    return summarize_transcripts(transcripts)


def summarize_paths(paths: Iterable[Path]) -> TranscriptStats:
    """Summarize transcripts read from disk.

    Args:
        paths: Transcript JSON files, or directories searched for them.

    Returns:
        The merged summary.

    Raises:
        FileNotFoundError: If a given path does not exist.
    """
    transcripts: dict[str, Sequence[dict[str, Any]]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"No such transcript path: {path}")
        for resolved in _iter_transcript_files(path):
            # Keyed by full path: two runs each holding
            # ``provenance/iter1_transcript.json`` would otherwise collapse
            # into one entry and a whole run would vanish unreported.
            transcripts[str(resolved)] = _as_entry_list(
                json.loads(resolved.read_text(encoding="utf-8")), str(resolved)
            )
    return summarize_transcripts(transcripts)


def is_transcript_name(name: str) -> bool:
    """Return whether a filename looks like an agent transcript.

    Args:
        name: Filename or path.

    Returns:
        Whether the name carries the transcript marker and a JSON suffix.

    Example:
        >>> is_transcript_name("provenance_iter1_transcript.json")
        True
        >>> is_transcript_name("evidence_matrix.json")
        False
    """
    lowered = name.lower()
    return TRANSCRIPT_NAME_FRAGMENT in lowered and lowered.endswith(".json")


def short_tool_name(tool_name: str, server: Optional[str] = None) -> str:
    """Strip an MCP server prefix from a tool name.

    Backends spell the same tool differently — one emits
    ``mcp__github__search_issues`` with no server field, another emits
    ``github.search_issues`` alongside ``namespace: github``. Normalizing both
    to the bare name is what lets a summary aggregate a tool across the agent
    backends a provider may have used.

    A dotted prefix is stripped only when it matches the server the entry
    itself reports, so a tool whose name genuinely contains a dot is left
    alone.

    Args:
        tool_name: Tool name, possibly ``mcp__server__tool`` or ``server.tool``.
        server: Server or namespace the entry recorded, when it recorded one.

    Returns:
        The bare tool name.

    Example:
        >>> short_tool_name("mcp__openscientist__search_pubmed")
        'search_pubmed'
        >>> short_tool_name("github.search_issues", server="github")
        'search_issues'
        >>> short_tool_name("github.search_issues")
        'github.search_issues'
        >>> short_tool_name("Bash")
        'Bash'
    """
    name = tool_name.split("__")[-1] if "__" in tool_name else tool_name
    if server:
        prefix = f"{server}."
        if name.startswith(prefix) and len(name) > len(prefix):
            return name[len(prefix) :]
    return name


def mcp_server_of(tool_name: str) -> Optional[str]:
    """Return the MCP server named in a qualified tool name, if any.

    Args:
        tool_name: Tool name, possibly ``mcp__server__tool``.

    Returns:
        The server segment, or None for an unqualified name.

    Example:
        >>> mcp_server_of("mcp__openscientist__search_pubmed")
        'openscientist'
        >>> mcp_server_of("Bash") is None
        True
    """
    parts = tool_name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return parts[1]
    return None


def program_of(command: str) -> str:
    """Return the program a shell command runs.

    Skips wrappers and leading environment assignments so ``sudo FOO=1 uv run
    pytest`` reports ``uv`` rather than ``sudo``.

    Args:
        command: The shell command line.

    Returns:
        The program name, or ``"(empty)"`` for a blank command.

    Example:
        >>> program_of("uv run pytest -q")
        'uv'
        >>> program_of("sudo FOO=1 apt-get install x")
        'apt-get'
        >>> program_of("cd /tmp && python analyze.py")
        'cd'
    """
    stripped = command.strip()
    if not stripped:
        return "(empty)"
    try:
        words = shlex.split(stripped)
    except ValueError:
        # An unbalanced quote is still a command someone ran; fall back to
        # whitespace splitting rather than losing the row.
        words = stripped.split()
    for word in words:
        if "=" in word and not word.startswith("-") and word.split("=")[0].isidentifier():
            continue
        if word in _SHELL_WRAPPERS:
            continue
        return word
    return words[0] if words else "(empty)"


class _Accumulator:
    """Mutable tallies behind :func:`summarize_transcripts`.

    Kept separate from the pydantic result so the summary stays an immutable
    report rather than a half-built collection of counters.
    """

    def __init__(self) -> None:
        self.sources: list[str] = []
        self.entries = 0
        self.entry_types: Counter[str] = Counter()
        self.unrecognized_types: Counter[str] = Counter()

        self.tool_calls_by_name: Counter[str] = Counter()
        self.qualified_by_name: dict[str, set[str]] = {}
        self.server_by_name: dict[str, str] = {}
        self.duration_by_name: Counter[str] = Counter()
        self.duration_seen: set[str] = set()
        self.name_by_call_id: dict[str, str] = {}
        self.failures_by_name: Counter[str] = Counter()

        self.skill_counts: Counter[str] = Counter()
        self.shell_commands: Counter[str] = Counter()
        self.failed_shell_commands = 0
        self.web_searches: set[str] = set()
        self.files_changed: dict[str, set[str]] = {}

        self.models: set[str] = set()
        self.subagent_calls = 0
        self.tasks: Counter[str] = Counter()
        self.token_usage: Counter[str] = Counter()

        self.available_tools: set[str] = set()
        self.available_skills: set[str] = set()
        self.available_agents: set[str] = set()
        self.unknown_entries = 0

        self._pending_results: list[dict[str, Any]] = []

    def add_source(self, source: str) -> None:
        """Close the previous transcript and record that another follows.

        Call ids are unique only *within* one transcript — two iterations of
        the same job both start numbering from scratch — so pairing has to be
        resolved and the id map dropped at each boundary. Left shared, a
        result from one iteration is attributed to whichever tool happened to
        reuse its id in another.
        """
        self._close_source()
        self.sources.append(source)

    def consume(self, entry: dict[str, Any]) -> None:
        """Fold one transcript entry into the tallies."""
        self.entries += 1
        entry_type = str(entry.get("type", "(missing type)"))
        self.entry_types[entry_type] += 1

        handler = self._HANDLERS.get(entry_type)
        if handler is None:
            self.unrecognized_types[entry_type] += 1
            return
        handler(self, entry)

    def _close_source(self) -> None:
        """Pair the current transcript's results, then forget its call ids."""
        for result in self._pending_results:
            self._apply_result(result)
        self._pending_results.clear()
        self.name_by_call_id.clear()

    def finish(self) -> TranscriptStats:
        """Resolve deferred pairing and build the immutable summary."""
        self._close_source()

        tools = [
            ToolUsage(
                name=name,
                qualified_names=sorted(self.qualified_by_name.get(name, {name})),
                calls=calls,
                failures=self.failures_by_name.get(name, 0),
                server=self.server_by_name.get(name),
                total_duration_ms=(
                    self.duration_by_name[name] if name in self.duration_seen else None
                ),
            )
            for name, calls in self.tool_calls_by_name.most_common()
        ]

        return TranscriptStats(
            sources=list(self.sources),
            entries=self.entries,
            entry_types=dict(self.entry_types.most_common()),
            tools=tools,
            tool_calls=sum(self.tool_calls_by_name.values()),
            failed_tool_calls=sum(self.failures_by_name.values()),
            mcp_servers=sorted(set(self.server_by_name.values())),
            skill_counts=dict(self.skill_counts.most_common()),
            shell_commands=dict(self.shell_commands.most_common()),
            failed_shell_commands=self.failed_shell_commands,
            web_searches=sorted(self.web_searches),
            files_changed={
                kind: sorted(paths) for kind, paths in sorted(self.files_changed.items())
            },
            models=sorted(self.models),
            subagent_calls=self.subagent_calls,
            tasks=dict(self.tasks.most_common()),
            token_usage=dict(sorted(self.token_usage.items())),
            available_tools=sorted(self.available_tools),
            available_skills=sorted(self.available_skills),
            available_agents=sorted(self.available_agents),
            unknown_entries=self.unknown_entries,
            unrecognized_types=dict(self.unrecognized_types.most_common()),
        )

    # -- per-entry handlers, dispatched by ``type`` ------------------------

    def _on_tool_call(self, entry: dict[str, Any]) -> None:
        qualified = str(entry.get("tool", ""))
        # ``namespace`` is where one backend records the server for a
        # dynamic tool call; ``server`` is where another records it.
        server = entry.get("server") or entry.get("namespace") or mcp_server_of(qualified)
        name = short_tool_name(qualified, str(server) if server else None)
        self.tool_calls_by_name[name] += 1
        self.qualified_by_name.setdefault(name, set()).add(qualified)

        if server:
            self.server_by_name[name] = str(server)

        call_id = entry.get("id")
        if call_id is not None:
            self.name_by_call_id[str(call_id)] = name

        # Compare the normalized name: a backend that namespaces the tool
        # still invoked a skill.
        if name == SKILL_TOOL_NAME:
            arguments = entry.get("arguments") or {}
            skill = arguments.get("skill") if isinstance(arguments, dict) else None
            if skill:
                self.skill_counts[str(skill)] += 1

    def _on_tool_result(self, entry: dict[str, Any]) -> None:
        # A result can precede its call in a merged multi-file read, so pair
        # at the end rather than assuming source order.
        self._pending_results.append(entry)

    def _apply_result(self, entry: dict[str, Any]) -> None:
        """Attribute one deferred result to the tool that produced it."""
        name = self.name_by_call_id.get(str(entry.get("call_id")))
        if name is None:
            return
        if entry.get("success") is False:
            self.failures_by_name[name] += 1
        duration = entry.get("duration_ms")
        if _is_number(duration):
            self.duration_by_name[name] += round(duration)
            self.duration_seen.add(name)

    def _on_shell_execution(self, entry: dict[str, Any]) -> None:
        self.shell_commands[program_of(str(entry.get("command", "")))] += 1
        exit_code = entry.get("exit_code")
        if isinstance(exit_code, int) and exit_code != 0:
            self.failed_shell_commands += 1

    def _on_file_change(self, entry: dict[str, Any]) -> None:
        kind = str(entry.get("kind", "unknown"))
        path = entry.get("path")
        if path:
            self.files_changed.setdefault(kind, set()).add(str(path))

    def _on_web_search(self, entry: dict[str, Any]) -> None:
        query = entry.get("query")
        if query:
            self.web_searches.add(str(query))

    def _on_collab_agent_tool_call(self, entry: dict[str, Any]) -> None:
        self.subagent_calls += 1
        model = entry.get("model")
        if model:
            self.models.add(str(model))

    def _on_assistant_text(self, entry: dict[str, Any]) -> None:
        model = entry.get("model")
        if model:
            self.models.add(str(model))

    def _on_session_init(self, entry: dict[str, Any]) -> None:
        self.available_tools.update(str(t) for t in entry.get("tools") or ())
        self.available_skills.update(str(c) for c in entry.get("slash_commands") or ())
        self.available_agents.update(str(a) for a in entry.get("agents") or ())
        model = entry.get("model")
        if model:
            self.models.add(str(model))

    def _on_task_started(self, entry: dict[str, Any]) -> None:
        self.tasks[str(entry.get("task_type") or "(untyped)")] += 1

    def _on_task_progress(self, entry: dict[str, Any]) -> None:
        self._add_usage(entry.get("usage"))

    def _on_task_notification(self, entry: dict[str, Any]) -> None:
        self._add_usage(entry.get("usage"))

    def _on_unknown_entry(self, entry: dict[str, Any]) -> None:
        self.unknown_entries += 1

    def _on_ignored(self, entry: dict[str, Any]) -> None:
        """Recognized shape contributing nothing beyond its type count.

        Registered so these types are not reported as unrecognized, which is
        reserved for producer drift.
        """
        return None

    # Explicit dispatch: an entry ``type`` selects a handler only if it is
    # named here. Attribute-name dispatch would make any future ``_on_*``
    # helper reachable from transcript content.
    _HANDLERS: dict[str, Any] = {}

    def _add_usage(self, usage: Any, prefix: str = "") -> None:
        """Sum numeric counters out of a reported usage mapping.

        One level of nesting is flattened into dotted keys, because a usage
        block reporting ``{"cache_creation": {"ephemeral_5m": 10}}`` otherwise
        contributes nothing and the reader cannot tell it was skipped.

        Args:
            usage: The reported usage mapping, or anything else (ignored).
            prefix: Dotted prefix applied to keys of a nested mapping.
        """
        if not isinstance(usage, dict):
            return
        for key, value in usage.items():
            name = f"{prefix}{key}"
            if _is_number(value):
                self.token_usage[name] += round(value)
            elif isinstance(value, dict) and not prefix:
                self._add_usage(value, prefix=f"{name}.")


_Accumulator._HANDLERS = {
    "assistant_text": _Accumulator._on_assistant_text,
    "collab_agent_tool_call": _Accumulator._on_collab_agent_tool_call,
    "file_change": _Accumulator._on_file_change,
    "hook_prompt": _Accumulator._on_ignored,
    "image_generation": _Accumulator._on_ignored,
    "image_view": _Accumulator._on_ignored,
    "plan": _Accumulator._on_ignored,
    "reasoning": _Accumulator._on_ignored,
    "review_mode_entered": _Accumulator._on_ignored,
    "review_mode_exited": _Accumulator._on_ignored,
    "session_init": _Accumulator._on_session_init,
    "shell_execution": _Accumulator._on_shell_execution,
    "task_notification": _Accumulator._on_task_notification,
    "task_progress": _Accumulator._on_task_progress,
    "task_started": _Accumulator._on_task_started,
    "tool_call": _Accumulator._on_tool_call,
    "tool_result": _Accumulator._on_tool_result,
    "unknown_entry": _Accumulator._on_unknown_entry,
    "user_prompt": _Accumulator._on_ignored,
    "web_search": _Accumulator._on_web_search,
}
"""Recognized entry types. A type absent here is counted as unrecognized."""


def _is_number(value: Any) -> TypeGuard[float]:
    """Return whether a value is a real number rather than a bool.

    ``bool`` is a subclass of ``int``, so an unguarded check would let
    ``True`` add 1 to a counter. A :class:`~typing.TypeGuard` so callers can
    round the value without a second cast.

    Example:
        >>> _is_number(5), _is_number(2.5), _is_number(True), _is_number("5")
        (True, True, False, False)
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _decode_entries(content_base64: str, source: str) -> list[dict[str, Any]]:
    """Decode one base64 artifact body into transcript entries.

    Args:
        content_base64: The artifact's base64 payload.
        source: Name used in error messages.

    Returns:
        The decoded entry list.

    Raises:
        ValueError: If the payload is not base64-encoded JSON.
    """
    try:
        raw = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Artifact {source} is not valid base64: {exc}") from exc
    return _as_entry_list(json.loads(raw.decode("utf-8")), source)


def _as_entry_list(payload: Any, source: str) -> list[dict[str, Any]]:
    """Validate that a decoded payload is a list of entry objects.

    Args:
        payload: Parsed JSON.
        source: Name used in error messages.

    Returns:
        The payload as a list of dicts.

    Raises:
        ValueError: If the payload is not a JSON list of objects.
    """
    if not isinstance(payload, list):
        raise ValueError(
            f"Transcript {source} is a {type(payload).__name__}, expected a JSON list"
        )
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Transcript {source} entry {index} is a {type(entry).__name__}, "
                "expected an object"
            )
    return payload


def _iter_transcript_files(path: Path) -> list[Path]:
    """Return the transcript files at or under a path.

    A file is taken as given; a directory is searched recursively for
    transcript-named JSON.
    """
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.json") if is_transcript_name(p.name))


def _md(value: str) -> str:
    r"""Escape agent-supplied text for a markdown table cell.

    Tool names, paths and search queries come from agent output. An
    unescaped pipe splits the cell and breaks the table for every reader
    downstream.

    Only pipes and newlines are escaped. Backslashes are left alone: most of
    these values render inside a code span, where an escape is shown
    literally, so doubling them would turn a Windows path into ``C:\\path``.

    Example:
        >>> _md("a|b")
        'a\\|b'
        >>> _md(r"C:\\runs\\out.csv")
        'C:\\\\runs\\\\out.csv'
    """
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_markdown(stats: TranscriptStats) -> str:
    """Build the markdown rendering for a summary."""
    lines: list[str] = ["## Agent run summary", ""]

    if not stats.entries:
        lines.append("No transcript entries found.")
        return "\n".join(lines) + "\n"

    lines.append(f"- Transcripts read: {len(stats.sources)}")
    lines.append(f"- Transcript entries: {stats.entries}")
    lines.append(f"- Tool calls: {stats.tool_calls} across {len(stats.tools)} distinct tools")
    rate = stats.tool_success_rate
    if rate is not None:
        lines.append(f"- Tool success rate: {rate:.1%} ({stats.failed_tool_calls} failed)")
    if stats.models:
        lines.append("- Models: " + ", ".join(_md(m) for m in stats.models))
    if stats.mcp_servers:
        lines.append(
            "- MCP servers used: " + ", ".join(_md(s) for s in stats.mcp_servers)
        )
    if stats.subagent_calls:
        lines.append(f"- Subagent invocations: {stats.subagent_calls}")
    if stats.unknown_entries:
        lines.append(
            f"- Unclassified entries in source transcript: {stats.unknown_entries}"
        )
    lines.append("")

    if stats.tools:
        lines.extend(["### Tools used", "", "| Tool | Calls | Failed | Server |", "|---|---:|---:|---|"])
        for tool in stats.tools:
            lines.append(
                f"| `{_md(tool.name)}` | {tool.calls} | {tool.failures} "
                f"| {_md(tool.server) if tool.server else '—'} |"
            )
        lines.append("")

    if stats.skill_counts:
        lines.extend(["### Skills invoked", ""])
        for skill, count in stats.skill_counts.items():
            lines.append(f"- `{_md(skill)}` ({count})")
        lines.append("")

    if stats.shell_commands:
        lines.extend(["### Shell programs", ""])
        summary = ", ".join(
            f"`{_md(program)}` ({count})"
            for program, count in stats.shell_commands.items()
        )
        lines.append(summary)
        if stats.failed_shell_commands:
            lines.append("")
            lines.append(f"{stats.failed_shell_commands} exited non-zero.")
        lines.append("")

    if stats.web_searches:
        lines.extend([f"### Web searches ({len(stats.web_searches)})", ""])
        lines.extend(f"- {_md(query)}" for query in stats.web_searches)
        lines.append("")

    if stats.files_changed:
        lines.extend(["### Files changed", ""])
        for kind, paths in stats.files_changed.items():
            lines.append(
                f"- **{_md(kind)}**: " + ", ".join(f"`{_md(p)}`" for p in paths)
            )
        lines.append("")

    if stats.token_usage:
        lines.extend(["### Reported token usage", ""])
        for key, value in stats.token_usage.items():
            lines.append(f"- {_md(key)}: {value:,}")
        lines.append("")

    unused = stats.unused_available_tools
    if unused:
        lines.extend(
            [
                "### Available but unused",
                "",
                f"{len(unused)} of {len(stats.available_tools)} declared tools were "
                "never called:",
                "",
                ", ".join(f"`{_md(name)}`" for name in unused),
                "",
            ]
        )

    if stats.unrecognized_types:
        lines.extend(["### Entry types not summarized", ""])
        for entry_type, count in stats.unrecognized_types.items():
            lines.append(f"- `{_md(entry_type)}` ({count})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
