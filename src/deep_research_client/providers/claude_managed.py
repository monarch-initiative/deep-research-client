"""Claude Managed Agents research provider.

Runs a research team in an Anthropic-hosted sandbox. A coordinator agent (Opus
by default) plans the review in ``PLAN.md``, then starts a workflow run that
hands each sub-question to a retriever agent (Haiku by default). Retrievers save
each source's text, verbatim, as an evidence file in ``citations/``, and return
a short digest. The coordinator writes ``REPORT.md`` from the evidence, citing
by identifier (``PMID:5568449``) and quoting it word for word, so the report
works with ``validate-references`` and claims extraction unchanged.

The whole task is sent as an *outcome*: a rubric that a separate grader, in its
own context window, checks against the report and evidence files after each
turn. The grader judges what a mechanical check cannot -- whether each source
supports the sentence it is attached to, and whether the sources are relevant
-- and sends the coordinator back until the rubric is met or
``max_iterations`` is reached.

The provider polls the session rather than holding a stream open, because a
run lasts tens of minutes and the event history is persisted server-side: the
history read once the session is idle is the complete record. When the run
ends, the session's files are listed through the Files API and downloaded.
``REPORT.md`` becomes the report, the evidence files become the citation list,
and the report is checked locally against the evidence: every cited identifier
should have an evidence file, and every attributed quotation should occur in
it. The check's result is recorded in ``run_metadata``.

The agents and environment are created on first use and reused afterwards.
Each is named after a fingerprint of its definition, so changing a model,
prompt or networking creates a new one instead of silently editing the old.

Billing is through ``ANTHROPIC_API_KEY`` (tokens, web searches and session
runtime), with a hard per-session cost cap from ``max_cost_usd`` that covers
the retrievers too.
"""

import asyncio
import base64
import hashlib
import json
import logging
import time
import unicodedata
from fnmatch import fnmatch
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from jinja2 import Template

from . import ResearchProvider
from ..artifact_selection import ArtifactSelectionPolicy
from ..exceptions import (
    ProviderBillingError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
    classify_exception,
)
from ..model_cards import ProviderModelCards, create_claude_managed_model_cards
from ..models import ProviderConfig, ProviderHealth, ResearchArtifact, ResearchResult
from ..provider_params import ClaudeManagedParams
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT
from ..validation.evidence import (
    EvidenceCheck,
    EvidenceFile,
    check_against_evidence,
    parse_evidence_file,
)

logger = logging.getLogger(__name__)

#: Beta header for every Managed Agents call. The SDK adds it on the
#: ``beta.agents``/``beta.sessions`` resources, but a ``scope_id`` filter on the
#: Files API needs it passed explicitly.
MANAGED_AGENTS_BETA = "managed-agents-2026-04-01"

#: Where the agents work. Files written here come back through the Files API.
OUTPUTS_DIR = "/mnt/session/outputs"

REPORT_FILE = "REPORT.md"

#: Outputs that are the team's own working files, not evidence.
WORKSPACE_FILES = frozenset({REPORT_FILE, "PLAN.md"})

#: Prefix of the agent and environment names; a fingerprint follows it.
RESOURCE_PREFIX = "deep-research-client"

#: ``span.outcome_evaluation_end`` results after which no evaluation follows.
TERMINAL_OUTCOME_RESULTS = frozenset(
    {"satisfied", "max_iterations_reached", "failed", "interrupted"}
)

_WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"
DEFAULT_TASK_FILE = _WORKFLOWS_DIR / "managed-task.md"
DEFAULT_RUBRIC_FILE = _WORKFLOWS_DIR / "managed-rubric.md"
COORDINATOR_PROMPT_FILE = _WORKFLOWS_DIR / "managed-coordinator.md"
RETRIEVER_PROMPT_FILE = _WORKFLOWS_DIR / "managed-retriever.md"

#: Output files can appear a few seconds after the session goes idle, so the
#: listing is retried this many times, this many seconds apart.
_OUTPUT_LISTING_ATTEMPTS = 12
_OUTPUT_LISTING_DELAY = 5.0

#: After an interrupt at the timeout, how long to wait for the session to settle
#: so the outputs written so far can still be collected.
_INTERRUPT_GRACE_SECONDS = 300

#: session.error types that have a closer match than the generic ProviderError.
_SESSION_ERROR_TYPES: dict[str, type[ProviderError]] = {
    "billing_error": ProviderBillingError,
    "model_rate_limited_error": ProviderRateLimitError,
    "model_overloaded_error": ProviderTransientError,
}


def fingerprint(definition: dict) -> str:
    """Hash a resource definition, independently of key order.

    Args:
        definition: JSON-serialisable agent or environment definition.

    Returns:
        A short hex digest identifying the definition.

    >>> fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})
    True
    >>> len(fingerprint({"a": 1}))
    16
    """
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:16]


def toolset(allowed_domains: List[str]) -> dict:
    """The full built-in toolset, with web search and fetch optionally limited.

    Bash is what lets an agent call the NCBI E-utilities or convert a PDF, and
    it is safe here because it runs in a throwaway sandbox holding no secrets.

    Args:
        allowed_domains: Domains web search and web fetch are limited to, or
            empty for no limit.

    Returns:
        The ``agent_toolset_20260401`` tool entry.

    >>> toolset([])
    {'type': 'agent_toolset_20260401'}
    >>> toolset(["pubmed.ncbi.nlm.nih.gov"])["configs"][1]
    {'name': 'web_fetch', 'allowed_domains': ['pubmed.ncbi.nlm.nih.gov']}
    """
    entry: dict[str, Any] = {"type": "agent_toolset_20260401"}
    if allowed_domains:
        entry["configs"] = [
            {"name": tool, "allowed_domains": list(allowed_domains)}
            for tool in ("web_search", "web_fetch")
        ]
    return entry


def retriever_definition(model: str, allowed_domains: List[str]) -> dict:
    """Build the retriever agent, which saves evidence files for one sub-question.

    Args:
        model: Claude model ID; a small model, since the work is mostly reading
            and copying text.
        allowed_domains: Domains web search and web fetch are limited to.

    Returns:
        Keyword arguments for ``beta.agents.create``, without the name.

    >>> retriever_definition("claude-haiku-5-5", [])["model"]
    'claude-haiku-5-5'
    """
    return {
        "model": model,
        "description": "Finds sources for one sub-question and saves their text as evidence files.",
        "system": RETRIEVER_PROMPT_FILE.read_text(encoding="utf-8"),
        "tools": [toolset(allowed_domains)],
    }


def coordinator_definition(
    model: str, system_prompt: str, allowed_domains: List[str], retriever_id: str
) -> dict:
    """Build the coordinator agent, which plans, fans out retrievers, and writes.

    Dynamic workflows are on, with the retriever as the only agent a workflow
    may use, so every fanned-out agent runs on the retriever's model. Inline
    agents (which would run on the coordinator's model) and subagents are off.

    Args:
        model: Claude model ID for planning and synthesis.
        system_prompt: Research system prompt, before the coordinator's phases.
        allowed_domains: Domains web search and web fetch are limited to.
        retriever_id: ID of the retriever agent.

    Returns:
        Keyword arguments for ``beta.agents.create``, without the name.

    >>> workflows = coordinator_definition("claude-opus-5-5", "p", [], "agent_1")["multiagent"]["workflows"]
    >>> workflows["predefined_agents"], workflows["inline_agents"]
    ([{'type': 'agent', 'id': 'agent_1'}], {'type': 'disabled'})
    """
    coordinator_prompt = COORDINATOR_PROMPT_FILE.read_text(encoding="utf-8")
    return {
        "model": model,
        "system": f"{system_prompt}\n\n{coordinator_prompt}",
        "tools": [toolset(allowed_domains)],
        "multiagent": {
            "type": "multiagent_20261001",
            "subagents": {"type": "disabled"},
            "workflows": {
                "type": "enabled",
                "inline_agents": {"type": "disabled"},
                "predefined_agents": [{"type": "agent", "id": retriever_id}],
            },
        },
    }


def environment_definition(allowed_hosts: List[str]) -> dict:
    """Build the cloud environment, with networking set explicitly.

    The API defaults an environment without ``networking`` to unrestricted, so
    the choice is always written out rather than left to that default.

    Args:
        allowed_hosts: Hosts the sandbox and web tools may reach, or empty for
            unrestricted networking.

    Returns:
        Keyword arguments for ``beta.environments.create``, without the name.

    >>> environment_definition([])["config"]["networking"]
    {'type': 'unrestricted'}
    >>> environment_definition(["pubmed.ncbi.nlm.nih.gov"])["config"]["networking"]["type"]
    'limited'
    """
    if allowed_hosts:
        networking: dict[str, Any] = {
            "type": "limited",
            "allowed_hosts": list(allowed_hosts),
            "allow_package_managers": True,
        }
    else:
        networking = {"type": "unrestricted"}
    return {"config": {"type": "cloud", "networking": networking}}


def session_title(query: str, limit: int = 100) -> str:
    """A one-line session title from the query, safe for the API.

    Queries rendered from templates span many lines, and the API rejects a
    title holding Unicode control or format characters, so whitespace runs are
    collapsed to single spaces and the remaining such characters dropped.

    Args:
        query: The research question.
        limit: Most characters of the query to keep.

    Returns:
        The title.

    >>> session_title("# Disease Research\\n\\n## Target\\n- **Name:** Birt-Hogg-Dube\\u200b")
    'deep-research-client: # Disease Research ## Target - **Name:** Birt-Hogg-Dube'
    """
    one_line = " ".join(query.split())
    printable = "".join(ch for ch in one_line if unicodedata.category(ch) not in ("Cc", "Cf"))
    return f"{RESOURCE_PREFIX}: {printable[:limit]}"


def render_task(task_file: Path, query: str) -> str:
    """Render the outcome's task description for a question.

    Args:
        task_file: Jinja template with a ``query`` variable.
        query: The research question.

    Returns:
        The task text.

    >>> render_task(DEFAULT_TASK_FILE, "What causes scurvy?").splitlines()[0]
    'Research question: What causes scurvy?'
    """
    return Template(task_file.read_text(encoding="utf-8")).render(query=query)


@dataclass
class SessionTrail:
    """What a session's event history says about the run so far."""

    #: Text of every agent message, in order.
    agent_texts: List[str] = field(default_factory=list)
    #: Names of the tools the agent called, in order, repeats included.
    tool_uses: List[str] = field(default_factory=list)
    #: Payload of each ``session.error`` event, in order.
    errors: List[dict] = field(default_factory=list)
    #: How many workflow runs the coordinator started.
    workflow_runs: int = 0
    #: The latest ``span.outcome_evaluation_end`` event, if any.
    outcome: Optional[dict] = None
    #: Why the session last stopped: an idle stop reason, "refusal" when the
    #: idle's stop details record one, or "terminated".
    stop_reason: Optional[str] = None
    #: The refusal category, such as "bio", when the session stopped on a refusal.
    refusal_category: Optional[str] = None
    #: Whether the session has stopped for good, so outputs can be collected.
    finished: bool = False
    #: Server time at which the repair message being waited on was processed.
    repair_started_at: Optional[str] = None

    @property
    def outcome_result(self) -> Optional[str]:
        """The latest grader verdict, if the grader has run."""
        return self.outcome["result"] if self.outcome else None


def read_trail(events: List[dict], after_event_id: Optional[str] = None) -> SessionTrail:
    """Read an event history, oldest first, into a :class:`SessionTrail`.

    The run is finished when the latest status event is an idle that follows a
    terminal grader verdict, an idle for any reason other than ``end_turn``
    (the cost cap, exhausted retries, a refusal, a pause for input), or a
    termination. An ``end_turn`` idle before the grader has settled is a pause
    between iterations, not the end.

    A repair turn, started by the message ``after_event_id``, has no outcome of
    its own: it is finished by any idle or termination that comes after that
    message. Until the message appears in the history, nothing is finished.

    Args:
        events: Events as dicts, in ``order="asc"``.
        after_event_id: ID of a repair message the run must get past.

    Returns:
        The trail.

    >>> repair = {"type": "user.message", "id": "sevt_9", "processed_at": "2026-11-07T10:00:00Z"}
    >>> stale = {"type": "session.status_idle", "stop_reason": {"type": "end_turn"}}
    >>> read_trail([stale, repair], after_event_id="sevt_9").finished
    False
    >>> trail = read_trail([stale, repair, {"type": "session.status_running"}, stale], after_event_id="sevt_9")
    >>> (trail.finished, trail.repair_started_at)
    (True, '2026-11-07T10:00:00Z')

    >>> idle = {"type": "session.status_idle", "stop_reason": {"type": "end_turn"}}
    >>> read_trail([{"type": "session.status_running"}, idle]).finished
    False
    >>> verdict = {"type": "span.outcome_evaluation_end", "result": "satisfied"}
    >>> read_trail([verdict, idle]).finished
    True
    >>> read_trail([{"type": "session.status_idle", "stop_reason": {"type": "budget_reached"}}]).stop_reason
    'budget_reached'
    >>> refused = {"type": "session.status_idle", "stop_reason": {"type": "end_turn"},
    ...            "stop_details": {"type": "refusal", "category": "bio"}}
    >>> trail = read_trail([refused])
    >>> (trail.stop_reason, trail.refusal_category, trail.finished)
    ('refusal', 'bio', True)
    """
    trail = SessionTrail()
    status: Optional[dict] = None
    status_index = outcome_index = repair_index = -1
    for index, event in enumerate(events):
        kind = event["type"]
        if after_event_id is not None and event.get("id") == after_event_id:
            repair_index, trail.repair_started_at = index, event["processed_at"]
        if kind == "agent.message":
            text = "".join(
                block["text"] for block in event["content"] if block["type"] == "text"
            )
            if text:
                trail.agent_texts.append(text)
        elif kind in ("agent.tool_use", "agent.mcp_tool_use"):
            trail.tool_uses.append(event["name"])
        elif kind == "session.error":
            trail.errors.append(event["error"])
        elif kind == "workflow_run.created":
            trail.workflow_runs += 1
        elif kind == "span.outcome_evaluation_end":
            trail.outcome, outcome_index = event, index
        elif kind.startswith("session.status_"):
            status, status_index = event, index

    if status is None or (after_event_id is not None and not 0 <= repair_index < status_index):
        return trail
    if status["type"] == "session.status_terminated":
        trail.stop_reason, trail.finished = "terminated", True
    elif status["type"] == "session.status_idle":
        trail.stop_reason = status["stop_reason"]["type"]
        # A refusal by a safety classifier ends the turn as end_turn and is
        # recorded only in the stop details, so it is read from there.
        details = status.get("stop_details") or {}
        if details.get("type") == "refusal":
            trail.stop_reason, trail.refusal_category = "refusal", details.get("category")
        settled = (
            trail.outcome_result in TERMINAL_OUTCOME_RESULTS and outcome_index < status_index
        )
        trail.finished = trail.stop_reason != "end_turn" or settled or after_event_id is not None
    return trail


def can_repair(trail: SessionTrail) -> bool:
    """Whether the session can take a repair turn.

    It cannot after the cost cap (the next request would stop again), a
    refusal, or an interrupt at the timeout.

    >>> verdict = {"type": "span.outcome_evaluation_end", "result": "satisfied"}
    >>> can_repair(SessionTrail(stop_reason="end_turn", outcome=verdict))
    True
    >>> can_repair(SessionTrail(stop_reason="budget_reached", outcome=verdict))
    False
    """
    return trail.stop_reason == "end_turn" and trail.outcome_result != "interrupted"


def repair_message(check: EvidenceCheck) -> str:
    """Tell the coordinator what DRC's evidence check found, and how to fix it.

    Args:
        check: A failed check.

    Returns:
        The text of the repair message.

    >>> from deep_research_client.validation.evidence import UnanchoredQuote
    >>> check = EvidenceCheck(evidence_files=3, cited_identifiers=2, cited_without_evidence=["PMID:31234567"],
    ...     unanchored_quotes=[UnanchoredQuote(quote="four treatments", reference_id="PMID:23288083",
    ...                                        reason="not found in the cited evidence")])
    >>> print(repair_message(check))  # doctest: +ELLIPSIS
    An automated check of REPORT.md against the evidence files in citations/ found problems...
    - "four treatments" (PMID:23288083): not found in the cited evidence
    ...
    - PMID:31234567
    ...
    """
    lines = [
        f"An automated check of REPORT.md against the evidence files in citations/ found "
        f"problems: {check.failure_summary()}. Fix every one, then end your turn.",
    ]
    if check.unanchored_quotes:
        lines += [
            "",
            "These quotations do not occur, word for word, in the body of the evidence file "
            "for the identifier they cite. For each, copy the passage exactly from the right "
            "file, cite the source it actually comes from, or rewrite the sentence without "
            "the quotation:",
        ]
        lines += [
            f'- "{quote.quote}" ({quote.reference_id}): {quote.reason}'
            for quote in check.unanchored_quotes
        ]
    if check.cited_without_evidence:
        lines += [
            "",
            "These identifiers are cited but appear in no evidence file's identifiers. Either "
            "retrieve the source with a retriever run, or remove the claims that rest on it:",
        ]
        lines += [f"- {identifier}" for identifier in check.cited_without_evidence]
    lines += [
        "",
        f"Edit {OUTPUTS_DIR}/REPORT.md in place, and keep its References section in step. "
        "Before ending your turn, check each quotation with `grep -F` against its evidence file.",
    ]
    return "\n".join(lines)


def session_failure(provider: str, session_id: str, trail: SessionTrail) -> Optional[ProviderError]:
    """The error to raise for a session that stopped without finishing its work.

    A cost cap, refusal or interrupt still leaves outputs worth returning, so
    only a termination, exhausted retries, or a pause for input DRC cannot give
    is a failure.

    Args:
        provider: Provider name for the error.
        session_id: Session ID, named so the session can be inspected.
        trail: The finished trail.

    Returns:
        The error, or None if the run's outputs should be collected.

    >>> trail = SessionTrail(stop_reason="retries_exhausted",
    ...     errors=[{"type": "billing_error", "message": "credit balance too low"}])
    >>> type(session_failure("claude_managed", "sesn_1", trail)).__name__
    'ProviderBillingError'
    >>> session_failure("claude_managed", "sesn_1", SessionTrail(stop_reason="budget_reached")) is None
    True
    """
    if trail.stop_reason not in ("terminated", "retries_exhausted", "requires_action"):
        return None
    if trail.stop_reason == "requires_action":
        return ProviderError(
            provider, f"session {session_id} paused for input this provider cannot give"
        )
    last = trail.errors[-1] if trail.errors else {}
    error_class = _SESSION_ERROR_TYPES.get(last.get("type", ""), ProviderError)
    message = last.get("message", "no error event recorded")
    return error_class(provider, f"session {session_id} {trail.stop_reason}: {message}")


def unsuccessful_reason(trail: SessionTrail) -> Optional[str]:
    """Say why a collected run is not a finished report, if it is not.

    ``max_iterations_reached`` is not flagged: the report is complete, the
    grader just wanted more, and its explanation is kept in ``run_metadata``.

    Args:
        trail: The finished trail.

    Returns:
        A one-line reason, or None for a run that finished normally.

    >>> unsuccessful_reason(SessionTrail(stop_reason="budget_reached"))
    'the session reached its cost cap (max_cost_usd) before the grader was satisfied'
    >>> verdict = {"type": "span.outcome_evaluation_end", "result": "satisfied"}
    >>> unsuccessful_reason(SessionTrail(stop_reason="end_turn", outcome=verdict)) is None
    True
    """
    if trail.stop_reason == "budget_reached":
        return "the session reached its cost cap (max_cost_usd) before the grader was satisfied"
    if trail.stop_reason == "refusal":
        category = f" ({trail.refusal_category})" if trail.refusal_category else ""
        return f"a safety classifier stopped the session{category}"
    if trail.outcome_result == "interrupted":
        return "the session was interrupted at the timeout before the grader was satisfied"
    if trail.outcome_result == "failed":
        # A failed outcome is not only a rubric mismatch: the platform also ends
        # an outcome as failed when the model refuses, and says so here.
        return f"the outcome failed: {trail.outcome['explanation']}" if trail.outcome else "the outcome failed"
    if trail.outcome_result is None:
        return "the grader never evaluated the outputs"
    return None


def newest_by_name(files: List[Any]) -> dict[str, Any]:
    """Keep one file per name: the first seen, which is the newest.

    The Files API lists newest first, so a file the agent rewrote is
    represented by its final version.

    Args:
        files: File metadata objects with a ``filename``, newest first.

    Returns:
        Files keyed by name.

    >>> from types import SimpleNamespace as File
    >>> picked = newest_by_name([File(filename="a.md", id=2), File(filename="a.md", id=1)])
    >>> picked["a.md"].id
    2
    """
    picked: dict[str, Any] = {}
    for file in files:
        picked.setdefault(file.filename, file)
    return picked


def is_evidence_candidate(name: str) -> bool:
    """Whether an output file could be an evidence file: markdown, not the workspace's own.

    Evidence files are told apart from stray notes by their frontmatter, which
    needs the file's text, so only candidates are downloaded to find out.

    Args:
        name: Output file name.

    Returns:
        True for a markdown file other than REPORT.md and PLAN.md.

    >>> [is_evidence_candidate(name) for name in ("hodges-1971.md", "REPORT.md", "hodges-1971.pdf")]
    [True, False, False]
    """
    return name.endswith(".md") and name not in WORKSPACE_FILES


def is_artifact_candidate(name: str, include_globs: List[str]) -> bool:
    """Whether an output file is offered to the artifact policy at all.

    The team's own working files are (PLAN.md; REPORT.md is the result body).
    The evidence files are not, unless a caller asks for them by glob: they are
    the provider's working store, already read for the evidence check and the
    citation list, and a run leaves around 200 of them. The session keeps them.

    Args:
        name: Output file name.
        include_globs: The caller's ``artifact_include_globs``, matched against
            the lowercased name.

    Returns:
        True to pass the file on to the artifact selection policy.

    >>> [is_artifact_candidate(n, []) for n in ("PLAN.md", "hodges-1971.md", "hodges-1971.pdf")]
    [True, False, False]
    >>> is_artifact_candidate("hodges-1971.md", ["*.md"])
    True
    """
    return name in WORKSPACE_FILES or any(fnmatch(name.lower(), glob.lower()) for glob in include_globs)


def missing_report_message(session_id: str, trail: SessionTrail, outputs: List[str]) -> str:
    """Explain a session that left no report, naming the likely fix.

    Args:
        session_id: The session, named so it can be inspected.
        trail: The finished trail.
        outputs: Names of the files the session did leave.

    Returns:
        The error message.

    >>> print(missing_report_message("sesn_1", SessionTrail(stop_reason="budget_reached"), ["PLAN.md"]))
    Claude Managed Agents session sesn_1 wrote no REPORT.md to /mnt/session/outputs before it stopped (budget_reached); raise max_cost_usd, or set it to None. Outputs: PLAN.md
    """
    hint = ""
    if trail.stop_reason == "budget_reached":
        hint = "; raise max_cost_usd, or set it to None"
    elif trail.outcome_result == "interrupted":
        hint = "; raise timeout"
    elif trail.stop_reason == "refusal":
        category = f" in category {trail.refusal_category}" if trail.refusal_category else ""
        hint = f"; a safety classifier refused the session's output{category}"
    elif trail.outcome_result == "failed" and trail.outcome:
        hint = f"; the outcome failed: {trail.outcome['explanation']}"
    return (
        f"Claude Managed Agents session {session_id} wrote no {REPORT_FILE} to {OUTPUTS_DIR} "
        f"before it stopped ({trail.stop_reason}){hint}. Outputs: {', '.join(outputs) or 'none'}"
    )


def run_metadata_for(session: Any, trail: SessionTrail, retriever_model: str) -> dict:
    """Provenance for the result: the session, the grader's verdict, and usage.

    Args:
        session: The retrieved session.
        trail: The finished trail.
        retriever_model: Model the retrievers ran on.

    Returns:
        A JSON-serialisable dict. Usage covers the whole session, retrievers
        and grader included.
    """
    metadata: dict[str, Any] = {
        "session_id": session.id,
        "agent_id": session.agent.id,
        "agent_version": session.agent.version,
        "environment_id": session.environment_id,
        "retriever_model": retriever_model,
        "stop_reason": trail.stop_reason,
        "outcome_result": trail.outcome_result,
        "workflow_runs": trail.workflow_runs,
        "tool_calls": len(trail.tool_uses),
    }
    if trail.outcome:
        metadata["outcome_iterations"] = trail.outcome["iteration"] + 1
        metadata["grader_explanation"] = trail.outcome["explanation"]
    usage = session.usage
    metadata["input_tokens"] = usage.input_tokens
    metadata["output_tokens"] = usage.output_tokens
    metadata["cache_read_input_tokens"] = usage.cache_read_input_tokens
    if usage.server_tool_use:
        metadata["web_search_requests"] = usage.server_tool_use.web_search_requests
        metadata["web_fetch_requests"] = usage.server_tool_use.web_fetch_requests
    if usage.list_cost:
        metadata["list_cost_usd"] = int(usage.list_cost.amount) / 100
    if session.stats:
        metadata["active_seconds"] = session.stats.active_seconds
    return metadata


class ClaudeManagedProvider(ResearchProvider):
    """Provider that runs a coordinator and retriever team on Claude Managed Agents."""

    credential_env_var = "ANTHROPIC_API_KEY"
    credential_label = "Claude Managed Agents"
    credential_noun = "Anthropic"

    def __init__(self, config: ProviderConfig, params: Optional[ClaudeManagedParams] = None):
        """Initialize the provider.

        Args:
            config: Provider configuration carrying the Anthropic API key.
            params: Managed Agents parameters.
        """
        self.params = params or ClaudeManagedParams()
        super().__init__(config, self.params.model)
        # ProviderConfig.timeout wins when set; otherwise use the params default.
        self.timeout = config.timeout or self.params.timeout

    def get_default_model(self) -> str:
        """The default card, which is a Claude model ID."""
        return self.model_cards().default_model

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for this provider."""
        return create_claude_managed_model_cards()

    def _client(self) -> Any:
        """Build an async Anthropic client from the configuration."""
        from anthropic import AsyncAnthropic

        kwargs: dict[str, Any] = {"api_key": self.config.api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        return AsyncAnthropic(**kwargs)

    async def check_health(self) -> ProviderHealth:
        """Probe the key and Managed Agents access by listing one agent.

        Returns:
            Health record for this provider
        """
        if not self.is_available():
            return ProviderHealth(
                provider=self.name, configured=False, reachable=False,
                detail=self.unavailable_reason(),
            )
        client = self._client()
        try:
            await client.beta.agents.list(limit=1)
        except Exception as e:
            classified = classify_exception(self.name, e)
            logger.debug("Managed Agents health probe failed:", exc_info=True)
            return ProviderHealth(
                provider=self.name, configured=True, reachable=False,
                detail=classified.diagnosis if classified else str(e),
            )
        finally:
            await client.close()
        return ProviderHealth(
            provider=self.name, configured=True, reachable=True,
            detail="credentials accepted (credit balance not exposed by the API)",
        )

    async def research(self, query: str) -> ResearchResult:
        """Run one research session and collect its outputs.

        Args:
            query: The research question.

        Returns:
            ResearchResult with REPORT.md as the markdown, each evidence file's
            identifier (or URL) as a citation, selected output files as
            artifacts, and the local evidence check in ``run_metadata``.

        Raises:
            ProviderError: If the session fails, or a typed subclass when the
                failure can be classified.
            ValueError: If the query is empty or the session writes no report.
        """
        self.require_available()
        if not query or not query.strip():
            raise ValueError("Research query must not be empty.")

        start_time = datetime.now()
        deadline = time.monotonic() + self.timeout
        client = self._client()
        try:
            if self.params.session_id:
                session_id = self.params.session_id
                logger.info("Reattaching to Claude Managed Agents session %s", session_id)
            else:
                session = await self._start_session(client, query)
                session_id = session.id
                logger.info("Claude Managed Agents session %s started (model=%s)", session_id, self.model)
            session, trail = await self._wait(client, session_id, deadline)
            session, trail, outputs, evidence_check, repair_history = await self._collect_checked(
                client, session_id, deadline, session, trail
            )
            markdown, evidence, artifacts = outputs
            if self.params.delete_session:
                await client.beta.sessions.delete(session_id)
        except Exception as e:
            classified = classify_exception(self.name, e)
            if classified is not None and classified is not e:
                logger.error(classified.actionable_message())
                raise classified from e
            raise
        finally:
            await client.close()

        run_metadata = run_metadata_for(session, trail, self.params.retriever_model)
        run_metadata["evidence_check"] = evidence_check.model_dump()
        run_metadata["repair_rounds"] = len(repair_history)
        if repair_history:
            run_metadata["repair_history"] = repair_history
        citations = sorted({source.citation for source in evidence})
        reason = unsuccessful_reason(trail)
        if reason is None and not evidence_check.passed:
            reason = (
                f"the report failed the evidence check after {len(repair_history)} repair "
                f"round(s): {evidence_check.failure_summary()}"
            )
        if reason:
            run_metadata["unsuccessful_reason"] = reason
            logger.info("Claude Managed Agents run is not a finished report: %s", reason)
        end_time = datetime.now()
        return ResearchResult(
            markdown=markdown,
            citations=citations,
            artifacts=artifacts,
            provider=self.name,
            query=query,
            model=self.model,
            run_metadata=run_metadata,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            answer_successful=False if reason else None,
        )

    async def _ensure(self, resource: Any, definition: dict) -> str:
        """Find the agent or environment for a definition, creating it if absent.

        Args:
            resource: ``client.beta.agents`` or ``client.beta.environments``.
            definition: The definition, without a name.

        Returns:
            The resource ID.
        """
        name = f"{RESOURCE_PREFIX}-{fingerprint(definition)}"
        async for existing in resource.list():
            if existing.name == name:
                return existing.id
        created = await resource.create(name=name, **definition)
        logger.info("Created %s", name)
        return created.id

    async def _start_session(self, client: Any, query: str) -> Any:
        """Create a session whose first event is the research outcome.

        Args:
            client: Async Anthropic client.
            query: The research question.

        Returns:
            The created session.
        """
        domains = self.params.allowed_domains
        retriever_id = await self._ensure(
            client.beta.agents, retriever_definition(self.params.retriever_model, domains)
        )
        system_prompt = self.params.system_prompt or DEFAULT_RESEARCH_SYSTEM_PROMPT
        agent_id = await self._ensure(
            client.beta.agents,
            coordinator_definition(self.model, system_prompt, domains, retriever_id),
        )
        environment_id = await self._ensure(
            client.beta.environments, environment_definition(self.params.allowed_hosts)
        )
        task_file = Path(self.params.task_file or DEFAULT_TASK_FILE)
        rubric_file = Path(self.params.rubric_file or DEFAULT_RUBRIC_FILE)
        outcome = {
            "type": "user.define_outcome",
            "description": render_task(task_file, query),
            "rubric": {"type": "text", "content": rubric_file.read_text(encoding="utf-8")},
            "max_iterations": self.params.max_iterations,
        }
        extra: dict[str, Any] = {}
        if self.params.max_cost_usd is not None:
            cents = str(round(self.params.max_cost_usd * 100))
            extra["budget"] = {"type": "limit", "max_list_cost": {"amount": cents, "currency": "USD"}}
        return await client.beta.sessions.create(
            agent=agent_id,
            environment_id=environment_id,
            title=session_title(query),
            initial_events=[outcome],
            **extra,
        )

    async def _wait(
        self, client: Any, session_id: str, deadline: float, after_event_id: Optional[str] = None
    ) -> tuple[Any, SessionTrail]:
        """Poll until the session has stopped for good.

        The event history is read only when the session is idle or terminated,
        since only then can the run be over. At the deadline the session is
        interrupted, and given a grace period to settle so its outputs so far
        can still be collected.

        Args:
            client: Async Anthropic client.
            session_id: The session to wait on.
            deadline: ``time.monotonic()`` value at which to interrupt, shared by
                the first turn and any repair turns.
            after_event_id: For a repair turn, the repair message's event ID.

        Returns:
            The final session and its trail.

        Raises:
            ProviderError: If the session stopped in a way that leaves nothing
                to collect (see :func:`session_failure`).
            TimeoutError: If the session does not settle after an interrupt.
        """
        from anthropic import APIConnectionError

        interrupted = False
        while True:
            try:
                session = await client.beta.sessions.retrieve(session_id)
                if session.status in ("idle", "terminated"):
                    events = [
                        event.model_dump(mode="json")
                        async for event in client.beta.sessions.events.list(session_id, order="asc")
                    ]
                    trail = read_trail(events, after_event_id)
                    if trail.finished:
                        failure = session_failure(self.name, session_id, trail)
                        if failure is not None:
                            raise failure
                        return session, trail
            except APIConnectionError as exc:
                # The session runs on regardless, so a client-side outage (a DNS
                # failure lost one hour-long run) only delays the next check. The
                # deadline still bounds the wait.
                logger.warning("Lost contact with session %s (%s); will retry", session_id, exc)
            if time.monotonic() > deadline:
                if interrupted:
                    raise TimeoutError(
                        f"Claude Managed Agents session {session_id} did not settle "
                        f"{_INTERRUPT_GRACE_SECONDS}s after being interrupted"
                    )
                logger.warning(
                    "Session %s still running after %ss; interrupting it", session_id, self.timeout
                )
                await client.beta.sessions.events.send(
                    session_id, events=[{"type": "user.interrupt"}]
                )
                interrupted = True
                deadline = time.monotonic() + _INTERRUPT_GRACE_SECONDS
            await asyncio.sleep(self.params.poll_interval)

    async def _collect_checked(
        self, client: Any, session_id: str, deadline: float, session: Any, trail: SessionTrail
    ) -> tuple[Any, SessionTrail, tuple[str, List[EvidenceFile], List[ResearchArtifact]], EvidenceCheck, List[dict]]:
        """Collect the outputs and check them, sending failures back for repair.

        While the evidence check fails, rounds remain and the session can take
        another turn, the failures are sent into the session as a message, the
        repair turn is waited out, and the outputs are collected and checked
        again.

        Args:
            client: Async Anthropic client.
            session_id: The finished session.
            deadline: ``time.monotonic()`` value at which to interrupt a repair turn.
            session: The session as last retrieved.
            trail: Its finished trail.

        Returns:
            The final session and trail, the outputs (report, evidence,
            artifacts), the last check, and a summary of each failed check that
            was sent back.
        """
        outputs = await self._collect_outputs(client, session_id, trail)
        evidence_check = check_against_evidence(outputs[0], outputs[1])
        repair_history: List[dict] = []
        while (
            not evidence_check.passed
            and len(repair_history) < self.params.max_repair_rounds
            and can_repair(trail)
        ):
            repair_history.append(evidence_check.model_dump(exclude={"unanchored_quotes"}))
            logger.info(
                "Evidence check failed (%s); asking the coordinator to repair, round %d",
                evidence_check.failure_summary(), len(repair_history),
            )
            sent = await client.beta.sessions.events.send(session_id, events=[
                {"type": "user.message",
                 "content": [{"type": "text", "text": repair_message(evidence_check)}]},
            ])
            session, trail = await self._wait(
                client, session_id, deadline, after_event_id=sent.data[0].id
            )
            outputs = await self._collect_outputs(
                client, session_id, trail, newer_than=trail.repair_started_at
            )
            evidence_check = check_against_evidence(outputs[0], outputs[1])
        return session, trail, outputs, evidence_check, repair_history

    async def _collect_outputs(
        self, client: Any, session_id: str, trail: SessionTrail, newer_than: Optional[str] = None
    ) -> tuple[str, List[EvidenceFile], List[ResearchArtifact]]:
        """Download the report, the evidence files and selected artifacts.

        Args:
            client: Async Anthropic client.
            session_id: The finished session.
            trail: The finished trail, which says why a report may be missing.
            newer_than: After a repair turn, the server time of the repair
                message: the listing is retried until a REPORT.md written after
                it appears. If none does, the coordinator left the report
                unchanged, and the latest one is used.

        Returns:
            The report markdown, the evidence files, and the artifacts.

        Raises:
            ValueError: If no REPORT.md appears in the session's files.
        """
        cutoff = datetime.fromisoformat(newer_than) if newer_than else None
        files: dict[str, Any] = {}
        for _ in range(_OUTPUT_LISTING_ATTEMPTS):
            listed = [
                file
                async for file in client.beta.files.list(
                    scope_id=session_id, betas=[MANAGED_AGENTS_BETA]
                )
            ]
            files = newest_by_name(listed)
            if REPORT_FILE in files and (cutoff is None or files[REPORT_FILE].created_at > cutoff):
                break
            await asyncio.sleep(_OUTPUT_LISTING_DELAY)
        else:
            if REPORT_FILE not in files:
                raise ValueError(missing_report_message(session_id, trail, sorted(files)))
            logger.warning("Session %s left REPORT.md unchanged after the repair request", session_id)

        contents: dict[str, bytes] = {}

        async def content_of(name: str) -> bytes:
            if name not in contents:
                contents[name] = await (await client.files.download(files[name].id)).read()
            return contents[name]

        markdown = (await content_of(REPORT_FILE)).decode("utf-8")
        evidence: List[EvidenceFile] = []
        for name in sorted(filter(is_evidence_candidate, files)):
            parsed = parse_evidence_file(name, (await content_of(name)).decode("utf-8"))
            if parsed is not None:
                evidence.append(parsed)

        artifacts: List[ResearchArtifact] = []
        if self.params.save_artifacts:
            policy = ArtifactSelectionPolicy.from_params(self.params)
            for name, file in sorted(files.items()):
                if not is_artifact_candidate(name, self.params.artifact_include_globs):
                    continue
                decision = policy.decide(name, file.size_bytes, provider_deny={REPORT_FILE})
                if not decision:
                    logger.debug("Not keeping %s as an artifact: %s", name, decision.reason)
                    continue
                artifacts.append(
                    ResearchArtifact(
                        filename=name,
                        content_base64=base64.b64encode(await content_of(name)).decode("ascii"),
                        media_type=file.mime_type,
                        source=self.name,
                        data_storage_id=file.id,
                    )
                )
        return markdown, evidence, artifacts
