"""Tests for the Claude Managed Agents provider.

The provider's decisions -- what agent and environment to define, when a
session is finished, what a stop means, which outputs are citations -- live in
pure functions over plain data, so they are tested here without the API. The
event dicts below have the shape ``event.model_dump(mode="json")`` produces.
One integration test runs a real, capped session when ANTHROPIC_API_KEY is set.
"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.exceptions import (
    ProviderBillingError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import ClaudeManagedParams
from deep_research_client.validation.evidence import EvidenceCheck, UnanchoredQuote
from deep_research_client.providers.claude_managed import (
    COORDINATOR_PROMPT_FILE,
    DEFAULT_RUBRIC_FILE,
    DEFAULT_TASK_FILE,
    OUTPUTS_DIR,
    RETRIEVER_PROMPT_FILE,
    ClaudeManagedProvider,
    SessionTrail,
    can_repair,
    coordinator_definition,
    environment_definition,
    fingerprint,
    is_artifact_candidate,
    is_evidence_candidate,
    newest_by_name,
    read_trail,
    render_task,
    repair_message,
    retriever_definition,
    session_failure,
    session_title,
    unsuccessful_reason,
)

RUNNING = {"type": "session.status_running"}
TERMINATED = {"type": "session.status_terminated"}


def idle(stop: str = "end_turn") -> dict:
    """A session.status_idle event with the given stop reason."""
    return {"type": "session.status_idle", "stop_reason": {"type": stop}}


def verdict(result: str, iteration: int = 0) -> dict:
    """A span.outcome_evaluation_end event."""
    return {
        "type": "span.outcome_evaluation_end",
        "result": result,
        "iteration": iteration,
        "explanation": f"grader says {result}",
    }


def make_provider(params: ClaudeManagedParams | None = None, api_key: str | None = "test-key"):
    """Build a provider with a standard test config."""
    config = ProviderConfig(name="claude_managed", api_key=api_key, enabled=True)
    return ClaudeManagedProvider(config, params)


# --- run state -----------------------------------------------------------------


@pytest.mark.parametrize(
    "events,finished,stop_reason",
    [
        ([], False, None),
        ([RUNNING], False, None),
        # An end_turn before any verdict is a pause between iterations.
        ([RUNNING, idle()], False, "end_turn"),
        ([verdict("needs_revision"), idle()], False, "end_turn"),
        # The verdict must come before the idle that follows it.
        ([idle(), verdict("satisfied")], False, "end_turn"),
        ([verdict("satisfied"), idle()], True, "end_turn"),
        ([verdict("needs_revision"), verdict("max_iterations_reached", 4), idle()], True, "end_turn"),
        ([verdict("interrupted"), idle()], True, "end_turn"),
        # A later status overrides an earlier idle.
        ([verdict("satisfied"), idle(), RUNNING], False, None),
        # Any stop other than end_turn is the end, whatever the grader said.
        ([verdict("needs_revision"), idle("budget_reached")], True, "budget_reached"),
        ([idle("retries_exhausted")], True, "retries_exhausted"),
        ([idle("refusal")], True, "refusal"),
        ([idle("requires_action")], True, "requires_action"),
        ([RUNNING, TERMINATED], True, "terminated"),
    ],
)
def test_read_trail_decides_when_the_run_is_over(events, finished, stop_reason):
    """Polling stops exactly when no further work can follow."""
    trail = read_trail(events)
    assert trail.finished is finished
    assert trail.stop_reason == stop_reason


PRIMARY = {"type": "session.thread_status_running", "session_thread_id": "sthr_primary"}
CHILD_RUNNING = {"type": "session.thread_status_running", "session_thread_id": "sthr_child"}


def thread_idle(thread: str, stop: str = "end_turn") -> dict:
    """A session.thread_status_idle event for one thread."""
    return {"type": "session.thread_status_idle", "session_thread_id": thread, "stop_reason": {"type": stop}}


@pytest.mark.parametrize(
    "events,finished,stop_reason",
    [
        # A real Marfan session hit its cap just after the grader was satisfied,
        # and only the primary thread's idle was recorded.
        ([PRIMARY, CHILD_RUNNING, verdict("satisfied"), thread_idle("sthr_primary", "budget_reached")],
         True, "budget_reached"),
        # A retriever thread going idle says nothing about the session.
        ([PRIMARY, CHILD_RUNNING, thread_idle("sthr_child")], False, None),
        ([PRIMARY, verdict("satisfied"), thread_idle("sthr_primary")], True, "end_turn"),
    ],
)
def test_the_primary_threads_status_stands_in_for_a_missing_session_status(events, finished, stop_reason):
    """The session-level idle is not always emitted; the primary thread's always is."""
    trail = read_trail(events)
    assert (trail.finished, trail.stop_reason) == (finished, stop_reason)


def test_a_cap_reached_after_the_grader_is_satisfied_is_a_finished_report():
    """The verdict came first, so the report is done; the cap only stopped what followed."""
    trail = read_trail([PRIMARY, verdict("satisfied"), thread_idle("sthr_primary", "budget_reached")])
    assert unsuccessful_reason(trail) is None
    assert can_repair(trail) is False


def test_read_trail_collects_messages_tools_and_errors():
    """Text blocks are joined per message; redacted blocks and empty messages are dropped."""
    events = [
        {"type": "agent.message", "content": [
            {"type": "text", "text": "Planning "}, {"type": "redacted"}, {"type": "text", "text": "now."},
        ]},
        {"type": "agent.message", "content": [{"type": "redacted"}]},
        {"type": "agent.tool_use", "name": "web_search", "input": {}},
        {"type": "agent.mcp_tool_use", "name": "pubmed_search", "input": {}},
        {"type": "agent.tool_result"},
        {"type": "session.error", "error": {"type": "model_overloaded_error", "message": "busy"}},
        {"type": "workflow_run.created", "workflow_run_id": "wfr_1"},
        {"type": "workflow_run.created", "workflow_run_id": "wfr_2"},
        verdict("satisfied", iteration=2),
        idle(),
    ]

    trail = read_trail(events)

    assert trail.agent_texts == ["Planning now."]
    assert trail.tool_uses == ["web_search", "pubmed_search"]
    assert trail.errors == [{"type": "model_overloaded_error", "message": "busy"}]
    assert trail.workflow_runs == 2
    assert trail.outcome_result == "satisfied"
    assert trail.outcome is not None and trail.outcome["iteration"] == 2


# --- repair turns ------------------------------------------------------------------

REPAIR = {"type": "user.message", "id": "sevt_repair", "processed_at": "2026-11-07T10:00:00Z"}


@pytest.mark.parametrize(
    "events,finished",
    [
        # The idle from before the repair message must not end the wait.
        ([verdict("satisfied"), idle()], False),
        ([verdict("satisfied"), idle(), REPAIR], False),
        ([verdict("satisfied"), idle(), REPAIR, RUNNING], False),
        # A repair turn has no outcome of its own: its end_turn is the end.
        ([verdict("satisfied"), idle(), REPAIR, RUNNING, idle()], True),
        ([verdict("satisfied"), idle(), REPAIR, RUNNING, idle("budget_reached")], True),
        ([verdict("satisfied"), idle(), REPAIR, TERMINATED], True),
    ],
)
def test_a_repair_turn_is_over_only_after_an_idle_that_follows_the_repair_message(events, finished):
    """Waiting starts from the repair message, identified by its event ID."""
    trail = read_trail(events, after_event_id="sevt_repair")
    assert trail.finished is finished


def test_a_repair_turn_keeps_the_grader_verdict_and_records_when_it_started():
    """The verdict is still the original outcome's; the start time is the server's."""
    trail = read_trail([verdict("satisfied"), idle(), REPAIR, RUNNING, idle()], after_event_id="sevt_repair")
    assert (trail.outcome_result, trail.repair_started_at) == ("satisfied", "2026-11-07T10:00:00Z")


@pytest.mark.parametrize(
    "stop_reason,result,repairable",
    [
        ("end_turn", "satisfied", True),
        ("end_turn", "max_iterations_reached", True),
        ("end_turn", "interrupted", False),
        ("budget_reached", "needs_revision", False),
        ("refusal", None, False),
    ],
)
def test_can_repair_only_a_session_that_can_take_another_turn(stop_reason, result, repairable):
    """After the cost cap, a refusal or the timeout, another turn would not run."""
    trail = SessionTrail(stop_reason=stop_reason, outcome=verdict(result) if result else None)
    assert can_repair(trail) is repairable


def test_repair_message_lists_every_failure_and_how_to_fix_it():
    """The coordinator gets the exact quotes and identifiers, not just counts."""
    check = EvidenceCheck(
        evidence_files=34, cited_identifiers=24, cited_without_evidence=["PMID:31234567"],
        unanchored_quotes=[
            UnanchoredQuote(quote="two men to each of four different daily treatments",
                            reference_id="PMID:23288083", reason="not found in the cited evidence"),
        ],
    )

    message = repair_message(check)

    assert '"two men to each of four different daily treatments" (PMID:23288083)' in message
    assert "- PMID:31234567" in message
    assert "grep -F" in message and f"{OUTPUTS_DIR}/REPORT.md" in message


@pytest.mark.parametrize("rounds", [-1, 6])
def test_max_repair_rounds_is_bounded(rounds):
    """Each round is a paid turn, so the number is capped."""
    with pytest.raises(ValidationError):
        ClaudeManagedParams(max_repair_rounds=rounds)


# --- stops ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "stop_reason,error_type,expected",
    [
        ("retries_exhausted", "billing_error", ProviderBillingError),
        ("retries_exhausted", "model_rate_limited_error", ProviderRateLimitError),
        ("retries_exhausted", "model_overloaded_error", ProviderTransientError),
        ("retries_exhausted", "unknown_error", ProviderError),
        ("terminated", None, ProviderError),
        ("requires_action", None, ProviderError),
    ],
)
def test_session_failure_raises_for_stops_that_leave_nothing_to_collect(stop_reason, error_type, expected):
    """The last session.error picks the error type, and the session ID is named."""
    errors = [{"type": error_type, "message": "it broke"}] if error_type else []
    trail = SessionTrail(stop_reason=stop_reason, errors=errors, finished=True)

    failure = session_failure("claude_managed", "sesn_abc", trail)

    assert type(failure) is expected
    assert "sesn_abc" in str(failure)


def test_a_refusal_recorded_only_in_stop_details_is_read_as_a_refusal():
    """A bio-classifier refusal ended a real session as end_turn, with the refusal in stop_details."""
    from deep_research_client.providers.claude_managed import missing_report_message

    refused = {**idle(), "stop_details": {"type": "refusal", "category": "bio", "explanation": None}}
    trail = read_trail([verdict("failed"), refused])

    assert (trail.stop_reason, trail.refusal_category, trail.finished) == ("refusal", "bio", True)
    assert can_repair(trail) is False
    assert session_failure("claude_managed", "sesn_1", trail) is None
    assert unsuccessful_reason(trail) == "a safety classifier stopped the session (bio)"
    assert "refused the session's output in category bio" in missing_report_message("sesn_1", trail, [])


def test_a_failed_outcome_reports_the_platforms_explanation():
    """The platform fails an outcome when the model refuses; "rubric does not apply" hid that."""
    from deep_research_client.providers.claude_managed import missing_report_message

    refused = {**verdict("failed"), "explanation": "Agent could not complete the task: the model response was refused (stop_reason: refusal)."}
    trail = SessionTrail(stop_reason="end_turn", outcome=refused, finished=True)

    assert "the model response was refused" in unsuccessful_reason(trail)
    assert "the model response was refused" in missing_report_message("sesn_1", trail, ["PLAN.md"])


@pytest.mark.parametrize("stop_reason", ["end_turn", "budget_reached", "refusal"])
def test_session_failure_lets_runs_with_outputs_through(stop_reason):
    """A capped, refused or completed run still has outputs worth returning."""
    trail = SessionTrail(stop_reason=stop_reason, finished=True)
    assert session_failure("claude_managed", "sesn_abc", trail) is None


@pytest.mark.parametrize(
    "stop_reason,result,flagged",
    [
        ("end_turn", "satisfied", False),
        # The grader wanted more, but the report is complete.
        ("end_turn", "max_iterations_reached", False),
        ("end_turn", "interrupted", True),
        ("end_turn", "failed", True),
        ("end_turn", None, True),
        ("budget_reached", "needs_revision", True),
        ("refusal", None, True),
    ],
)
def test_unsuccessful_reason_flags_runs_that_are_not_a_finished_report(stop_reason, result, flagged):
    """Flagged runs are written but not cached, as for claude_code."""
    trail = SessionTrail(stop_reason=stop_reason, outcome=verdict(result) if result else None)
    assert (unsuccessful_reason(trail) is not None) is flagged


# --- definitions -----------------------------------------------------------------


def test_coordinator_hands_fan_out_only_to_the_retriever():
    """Inline agents would run on the coordinator's model, so only the retriever is allowed."""
    definition = coordinator_definition("claude-opus-5-5", "Custom prompt.", [], "agent_retriever")

    assert definition["model"] == "claude-opus-5-5"
    assert definition["system"].startswith("Custom prompt.")
    assert COORDINATOR_PROMPT_FILE.read_text(encoding="utf-8") in definition["system"]
    assert definition["multiagent"] == {
        "type": "multiagent_20261001",
        "subagents": {"type": "disabled"},
        "workflows": {
            "type": "enabled",
            "inline_agents": {"type": "disabled"},
            "predefined_agents": [{"type": "agent", "id": "agent_retriever"}],
        },
    }


def test_retriever_runs_on_its_own_model_with_its_own_prompt():
    """The retriever cannot delegate further and is told the evidence file format."""
    definition = retriever_definition("claude-haiku-5-5", [])

    assert definition["model"] == "claude-haiku-5-5"
    assert "multiagent" not in definition
    assert definition["system"] == RETRIEVER_PROMPT_FILE.read_text(encoding="utf-8")


@pytest.mark.parametrize("make", [
    lambda domains: retriever_definition("m", domains),
    lambda domains: coordinator_definition("m", "p", domains, "agent_1"),
])
def test_both_agents_limit_both_web_tools_to_allowed_domains(make):
    """allowed_domains is the harmonized parameter; it restricts search and fetch alike."""
    toolset = make(["europepmc.org", "ncbi.nlm.nih.gov"])["tools"][0]

    assert {config["name"] for config in toolset["configs"]} == {"web_search", "web_fetch"}
    assert all(config["allowed_domains"] == ["europepmc.org", "ncbi.nlm.nih.gov"]
               for config in toolset["configs"])


def test_environment_definition_lets_package_managers_through_limited_networking():
    """A limited sandbox can still pip install the tools to convert a PDF."""
    networking = environment_definition(["europepmc.org"])["config"]["networking"]
    assert networking == {
        "type": "limited", "allowed_hosts": ["europepmc.org"], "allow_package_managers": True,
    }


@pytest.mark.parametrize(
    "first,second",
    [
        (retriever_definition("claude-haiku-5-5", []), retriever_definition("claude-sonnet-5-5", [])),
        (coordinator_definition("m", "p", [], "agent_1"), coordinator_definition("m", "p", [], "agent_2")),
        (coordinator_definition("m", "p", [], "a"), coordinator_definition("m", "other prompt", [], "a")),
        (environment_definition([]), environment_definition(["europepmc.org"])),
    ],
)
def test_a_changed_definition_gets_a_new_resource(first, second):
    """The name carries the fingerprint, so an edit never reuses a stale agent.

    A new retriever changes the coordinator's fingerprint too, because the
    coordinator names the retriever's ID.
    """
    assert fingerprint(first) != fingerprint(second)


def test_session_title_from_a_rendered_template_has_no_control_characters():
    """A dismech template renders to many lines; the API rejected the newlines in a title."""
    import unicodedata

    query = "# Disease Characteristics Research Template\n\n## Target Disease\n\t- **Disease Name:** BHD\u2060" * 5

    title = session_title(query)

    assert not [ch for ch in title if unicodedata.category(ch) in ("Cc", "Cf")]
    assert title.startswith("deep-research-client: # Disease Characteristics Research Template ## Target")
    assert len(title) <= len("deep-research-client: ") + 100


def test_render_task_uses_a_custom_template(tmp_path):
    """The task template is a parameter, with the question as its only variable."""
    template = tmp_path / "task.md"
    template.write_text("Investigate {{ query }} thoroughly.", encoding="utf-8")

    assert render_task(template, "scurvy") == "Investigate scurvy thoroughly."


def test_bundled_prompts_and_rubric_ship_with_the_package():
    """All four are read at run time, so a packaging slip would fail every run."""
    for path in (DEFAULT_TASK_FILE, COORDINATOR_PROMPT_FILE, RETRIEVER_PROMPT_FILE):
        assert path.is_file()
    rubric = DEFAULT_RUBRIC_FILE.read_text(encoding="utf-8")
    assert "evidence file" in rubric and "supports the sentence" in rubric


def test_the_prompts_agree_with_the_evidence_parser():
    """The retriever's example evidence file must be one DRC can read."""
    from deep_research_client.validation.evidence import parse_evidence_file

    prompt = RETRIEVER_PROMPT_FILE.read_text(encoding="utf-8")
    example = prompt.split("```\n---\n", 1)[1].split("```", 1)[0]
    evidence = parse_evidence_file("hodges-1971-clinical-manifestations.md", "---\n" + example)

    assert evidence is not None
    assert evidence.identifiers == ["PMID:5568449", "DOI:10.1093/ajcn/24.4.432"]
    assert OUTPUTS_DIR in prompt


# --- outputs ---------------------------------------------------------------------


def test_newest_by_name_keeps_the_final_version_of_a_rewritten_file():
    """The Files API lists newest first."""
    listed = [
        SimpleNamespace(filename="REPORT.md", id="file_new"),
        SimpleNamespace(filename="PLAN.md", id="file_plan"),
        SimpleNamespace(filename="REPORT.md", id="file_old"),
    ]
    picked = newest_by_name(listed)
    assert {name: file.id for name, file in picked.items()} == {
        "REPORT.md": "file_new", "PLAN.md": "file_plan",
    }


@pytest.mark.parametrize(
    "name,candidate",
    [("hodges-1971.md", True), ("REPORT.md", False), ("PLAN.md", False),
     ("hodges-1971.pdf", False), ("table1.csv", False)],
)
def test_only_markdown_outside_the_workspace_is_read_as_evidence(name, candidate):
    """Binary originals and the team's own files are never parsed as evidence."""
    assert is_evidence_candidate(name) is candidate


@pytest.mark.parametrize(
    "name,globs,candidate",
    [
        ("PLAN.md", [], True),
        # The report is the result body, not an artifact; the policy denies it.
        ("REPORT.md", [], True),
        ("hodges-1971-clinical-manifestations.md", [], False),
        ("hodges-1971-clinical-manifestations.pdf", [], False),
        ("hodges-1971-clinical-manifestations.md", ["*.md"], True),
        ("Hodges-1971.PDF", ["*.pdf"], True),
    ],
)
def test_only_the_plan_is_an_artifact_unless_evidence_is_asked_for(name, globs, candidate):
    """A run leaves ~200 evidence files; committing them made dismech PRs ~200 files each."""
    assert is_artifact_candidate(name, globs) is candidate


# --- provider ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,resolved",
    [(None, "claude-opus-5-5"), ("opus", "claude-opus-5-5"), ("sonnet", "claude-sonnet-5-5"),
     ("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001")],
)
def test_model_resolves_to_a_claude_model_id(model, resolved):
    """Card names are model IDs; an unknown ID passes through to the API."""
    assert make_provider(ClaudeManagedParams(model=model)).model == resolved


def test_provider_config_timeout_overrides_the_params_default():
    """As for claude_code, the client's timeout wins when set."""
    config = ProviderConfig(name="claude_managed", api_key="k", timeout=120)
    assert ClaudeManagedProvider(config).timeout == 120
    assert make_provider().timeout == 3600


@pytest.mark.parametrize(
    "field,value",
    [("max_iterations", 0), ("max_iterations", 21), ("max_cost_usd", 0), ("poll_interval", 0.5)],
)
def test_params_reject_values_the_api_or_the_loop_cannot_use(field, value):
    """Out-of-range values fail at construction, not mid-session."""
    with pytest.raises(ValidationError):
        ClaudeManagedParams(**{field: value})


def test_research_without_a_key_names_the_variable_to_set():
    """Fails before any network call."""
    provider = make_provider(api_key=None)
    with pytest.raises(ProviderNotConfiguredError, match="ANTHROPIC_API_KEY"):
        asyncio.run(provider.research("what causes scurvy"))


def test_research_rejects_an_empty_query():
    """An empty question would start a paid session with nothing to do."""
    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(make_provider().research("  "))


def test_models_cli_lists_the_managed_agents_cards():
    """Both model cards are reachable from the CLI."""
    result = CliRunner().invoke(app, ["models", "--provider", "claude_managed"])
    assert result.exit_code == 0, result.output
    assert "Default: claude-opus-5-5" in result.output
    assert "Claude Managed Agents (Sonnet 5.5)** (sonnet)" in result.output


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_a_capped_session_returns_a_report_backed_by_evidence(tmp_path: Path):
    """A real, small run: a Sonnet coordinator, Haiku retrievers, one grader turn, $5 cap.

    Either the grader gives its verdict, or the cap stops the session first; a
    capped run must still return the report, marked unsuccessful. Either way
    the coordinator fanned out, the retrievers saved evidence files DRC can
    read, and the report was checked against them.
    """
    params = ClaudeManagedParams(model="sonnet", max_iterations=1, max_cost_usd=5.0, timeout=1800)
    provider = ClaudeManagedProvider(
        ProviderConfig(name="claude_managed", api_key=os.environ["ANTHROPIC_API_KEY"]), params
    )

    result = asyncio.run(provider.research(
        "What is the evidence that vitamin C deficiency causes scurvy? Keep it to two sections."
    ))

    assert result.markdown.strip()
    metadata = result.run_metadata
    assert metadata is not None
    assert metadata["session_id"].startswith("sesn_")
    if metadata["stop_reason"] == "budget_reached":
        assert result.answer_successful is False
        assert "cost cap" in metadata["unsuccessful_reason"]
    else:
        assert metadata["outcome_result"] in ("satisfied", "max_iterations_reached")
    assert metadata["workflow_runs"] >= 1, "the coordinator never fanned out to retrievers"
    check = metadata["evidence_check"]
    assert check["evidence_files"] >= 1, "no evidence file had readable frontmatter"
    assert result.citations
