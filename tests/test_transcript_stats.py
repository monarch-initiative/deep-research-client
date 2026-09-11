"""Tests for transcript summary statistics.

Fixtures mirror the on-disk shape a provider writes: a JSON list of entries
discriminated by ``type``, with every optional field present (pydantic dumps
them as null) and a ``raw`` passthrough, so the parser is exercised against
the real serialization rather than a tidied-up version of it.
"""

import base64
import json
import re

import pytest

from deep_research_client.models import ResearchArtifact
from deep_research_client.transcript_stats import (
    is_transcript_name,
    mcp_server_of,
    program_of,
    short_tool_name,
    summarize_artifacts,
    summarize_paths,
    summarize_transcript,
    summarize_transcripts,
)


def tool_call(call_id, tool, arguments=None, server=None):
    """A ``tool_call`` entry as the provider serializes one."""
    return {
        "type": "tool_call",
        "id": call_id,
        "tool": tool,
        "arguments": arguments or {},
        "server": server,
        "namespace": None,
        "parent_tool_use_id": None,
        "uuid": None,
        "raw": {},
    }


def tool_result(call_id, success=True, duration_ms=None):
    """A ``tool_result`` entry as the provider serializes one."""
    return {
        "type": "tool_result",
        "call_id": call_id,
        "output": "ok" if success else "boom",
        "success": success,
        "status": None,
        "duration_ms": duration_ms,
        "structured_content": None,
        "content_items": None,
        "tool_use_result": None,
        "error_message": None if success else "tool failed",
        "mcp_app_resource_uri": None,
        "parent_tool_use_id": None,
        "uuid": None,
        "raw": {},
    }


@pytest.fixture
def run_transcript():
    """A transcript covering every entry type the summarizer reads."""
    return [
        {
            "type": "session_init",
            "session_id": "s1",
            "uuid": None,
            "cwd": "/job",
            "model": "claude-opus-5",
            "permission_mode": "default",
            "api_key_source": "env",
            "tools": ["Bash", "Read", "Write", "Skill", "WebSearch", "NotebookEdit"],
            "slash_commands": ["curate", "gene-set-enrichment"],
            "agents": ["literature-scout"],
            "mcp_servers": [{"name": "openscientist"}],
            "raw": {},
        },
        {
            "type": "user_prompt",
            "id": None,
            "text": "Investigate the mechanism",
            "parent_tool_use_id": None,
            "uuid": None,
            "raw": {},
        },
        {
            "type": "reasoning",
            "id": None,
            "text": "thinking",
            "summary": None,
            "signature": None,
            "raw": {},
        },
        tool_call("c1", "Skill", {"skill": "gene-set-enrichment"}),
        tool_result("c1", success=True, duration_ms=120),
        tool_call("c2", "mcp__openscientist__search_pubmed", {"query": "MYH7"}),
        tool_result("c2", success=True, duration_ms=800),
        tool_call("c3", "mcp__openscientist__search_pubmed", {"query": "TTN"}),
        tool_result("c3", success=False, duration_ms=200),
        tool_call("c4", "Read", {"file_path": "/job/data.csv"}),
        tool_result("c4", success=True),
        {
            "type": "shell_execution",
            "id": "s1",
            "command": "uv run python analyze.py",
            "output": "done",
            "exit_code": 0,
            "command_actions": None,
            "status": "completed",
            "raw": {},
        },
        {
            "type": "shell_execution",
            "id": "s2",
            "command": "pytest -q",
            "output": "fail",
            "exit_code": 1,
            "command_actions": None,
            "status": "completed",
            "raw": {},
        },
        {
            "type": "file_change",
            "id": "f1",
            "path": "results/table.csv",
            "kind": "create",
            "diff": None,
            "success": True,
            "status": None,
            "raw": {},
        },
        {
            "type": "file_change",
            "id": "f2",
            "path": "results/table.csv",
            "kind": "edit",
            "diff": None,
            "success": True,
            "status": None,
            "raw": {},
        },
        {
            "type": "web_search",
            "id": "w1",
            "query": "cardiomyopathy sarcomere",
            "action": None,
            "raw": {},
        },
        {
            "type": "collab_agent_tool_call",
            "id": "a1",
            "prompt": "check the literature",
            "model": "claude-haiku-4-5",
            "agents_states": None,
            "raw": {},
        },
        {
            "type": "task_started",
            "task_id": "t1",
            "description": "literature sweep",
            "task_type": "search",
            "parent_tool_use_id": None,
            "session_id": "s1",
            "uuid": None,
            "raw": {},
        },
        {
            "type": "task_notification",
            "task_id": "t1",
            "status": "completed",
            "summary": "found 12 papers",
            "output_file": "out.md",
            "usage": {"input_tokens": 1000, "output_tokens": 250, "cache_read": 40},
            "parent_tool_use_id": None,
            "session_id": "s1",
            "uuid": None,
            "raw": {},
        },
        {
            "type": "assistant_text",
            "id": None,
            "text": "Here is the finding",
            "model": "claude-opus-5",
            "error": None,
            "parent_tool_use_id": None,
            "uuid": None,
            "phase": None,
            "memory_citation": None,
            "raw": {},
        },
    ]


def test_tool_inventory_is_deduplicated_and_counted(run_transcript):
    """The headline answer: which distinct tools ran, and how often."""
    stats = summarize_transcript(run_transcript)

    assert stats.distinct_tools == ["Read", "Skill", "search_pubmed"]
    assert stats.tool_calls == 4
    assert {tool.name: tool.calls for tool in stats.tools}["search_pubmed"] == 2


def test_tools_are_ordered_most_called_first(run_transcript):
    """The table leads with what the agent leaned on."""
    stats = summarize_transcript(run_transcript)

    assert stats.tools[0].name == "search_pubmed"


def test_skills_are_named_from_the_skill_tool_argument(run_transcript):
    """A skill invocation is a Skill tool call carrying the skill name."""
    stats = summarize_transcript(run_transcript)

    assert stats.skills_used == ["gene-set-enrichment"]
    assert stats.skill_counts == {"gene-set-enrichment": 1}


def test_failures_are_attributed_to_the_calling_tool(run_transcript):
    """A failed result counts against the tool that produced it."""
    stats = summarize_transcript(run_transcript)
    by_name = {tool.name: tool for tool in stats.tools}

    assert stats.failed_tool_calls == 1
    assert by_name["search_pubmed"].failures == 1
    assert by_name["search_pubmed"].successes == 1
    assert by_name["Read"].failures == 0
    assert stats.tool_success_rate == 0.75


def test_durations_sum_only_where_reported(run_transcript):
    """A tool with no reported duration gets None, not a misleading zero."""
    by_name = {tool.name: tool for tool in summarize_transcript(run_transcript).tools}

    assert by_name["search_pubmed"].total_duration_ms == 1000
    assert by_name["Read"].total_duration_ms is None


def test_mcp_server_is_derived_from_the_qualified_name(run_transcript):
    """A tool's server is recoverable even when the entry omits the field."""
    stats = summarize_transcript(run_transcript)
    by_name = {tool.name: tool for tool in stats.tools}

    assert stats.mcp_servers == ["openscientist"]
    assert by_name["search_pubmed"].server == "openscientist"
    assert by_name["search_pubmed"].qualified_names == [
        "mcp__openscientist__search_pubmed"
    ]
    assert by_name["Read"].server is None


def test_shell_web_and_file_activity_is_tallied(run_transcript):
    """The non-tool activity types each get their own summary."""
    stats = summarize_transcript(run_transcript)

    assert stats.shell_commands == {"uv": 1, "pytest": 1}
    assert stats.failed_shell_commands == 1
    assert stats.web_searches == ["cardiomyopathy sarcomere"]
    assert stats.files_changed == {
        "create": ["results/table.csv"],
        "edit": ["results/table.csv"],
    }


def test_models_subagents_and_tasks_are_recorded(run_transcript):
    """A run's models come from several entry types and are merged."""
    stats = summarize_transcript(run_transcript)

    assert stats.models == ["claude-haiku-4-5", "claude-opus-5"]
    assert stats.subagent_calls == 1
    assert stats.tasks == {"search": 1}


def test_token_usage_sums_reported_counters(run_transcript):
    """Usage counters are summed without assuming a fixed set of keys."""
    stats = summarize_transcript(run_transcript)

    assert stats.token_usage == {
        "cache_read": 40,
        "input_tokens": 1000,
        "output_tokens": 250,
    }


def test_declared_but_unused_tools_are_reported(run_transcript):
    """The gap between what was offered and what was reached for."""
    stats = summarize_transcript(run_transcript)

    assert stats.available_tools == [
        "Bash",
        "NotebookEdit",
        "Read",
        "Skill",
        "WebSearch",
        "Write",
    ]
    assert stats.unused_available_tools == ["Bash", "NotebookEdit", "WebSearch", "Write"]
    assert stats.available_skills == ["curate", "gene-set-enrichment"]
    assert stats.available_agents == ["literature-scout"]


def test_every_entry_type_in_the_fixture_is_recognized(run_transcript):
    """An unrecognized type is a signal the producer drifted, not noise."""
    stats = summarize_transcript(run_transcript)

    assert stats.unrecognized_types == {}
    assert stats.entries == len(run_transcript)
    assert stats.entry_types["tool_call"] == 4


def test_unrecognized_entry_types_are_counted_not_dropped():
    """A shape from a newer producer is reported rather than silently lost."""
    stats = summarize_transcript([{"type": "quantum_teleport", "id": "x"}])

    assert stats.unrecognized_types == {"quantum_teleport": 1}
    assert stats.entries == 1


def test_unknown_entries_from_the_producer_are_surfaced():
    """The producer's own 'I could not classify this' marker is passed through."""
    stats = summarize_transcript(
        [{"type": "unknown_entry", "source": "codex", "raw": {}}]
    )

    assert stats.unknown_entries == 1
    assert stats.unrecognized_types == {}


def test_a_result_preceding_its_call_is_still_paired():
    """Merged multi-file reads can put a result before its call."""
    stats = summarize_transcript(
        [tool_result("c1", success=False), tool_call("c1", "Bash")]
    )

    assert stats.failed_tool_calls == 1
    assert stats.tools[0].failures == 1


def test_an_orphan_result_is_not_attributed_to_anything():
    """A result whose call is missing cannot be blamed on a tool."""
    stats = summarize_transcript([tool_result("nope", success=False)])

    assert stats.failed_tool_calls == 0
    assert stats.tools == []


def test_a_skill_call_with_no_skill_argument_counts_as_a_tool_only():
    """A malformed Skill call is still a tool call, just not a named skill."""
    stats = summarize_transcript([tool_call("c1", "Skill", {})])

    assert stats.tool_calls == 1
    assert stats.skills_used == []


def test_transcripts_merge_across_iterations(run_transcript):
    """A job writes one transcript per iteration; the totals span them."""
    stats = summarize_transcripts(
        {
            "iter1_transcript.json": [tool_call("a", "Bash"), tool_result("a")],
            "iter2_transcript.json": [tool_call("b", "Bash"), tool_result("b", False)],
        }
    )

    assert stats.sources == ["iter1_transcript.json", "iter2_transcript.json"]
    assert stats.tool_calls == 2
    assert stats.failed_tool_calls == 1
    assert stats.tools[0].calls == 2


def test_empty_input_summarizes_to_zero():
    """No transcripts is a valid answer, not an error."""
    stats = summarize_transcripts({})

    assert stats.entries == 0
    assert stats.tool_calls == 0
    assert stats.tool_success_rate is None
    assert "No transcript entries found." in stats.render_markdown()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("provenance_iter1_transcript.json", True),
        ("provenance_report_transcript.json", True),
        ("TRANSCRIPT.JSON", True),
        ("evidence_matrix.json", False),
        ("transcript.md", False),
        ("results_table.csv", False),
    ],
)
def test_transcript_names_are_recognized(name, expected):
    """Only transcript-shaped JSON is mined."""
    assert is_transcript_name(name) is expected


@pytest.mark.parametrize(
    "qualified,short,server",
    [
        ("mcp__openscientist__search_pubmed", "search_pubmed", "openscientist"),
        ("mcp__github__list_issues", "list_issues", "github"),
        ("Bash", "Bash", None),
        ("Skill", "Skill", None),
    ],
)
def test_tool_name_parsing(qualified, short, server):
    """MCP qualification is split off the tool name."""
    assert short_tool_name(qualified) == short
    assert mcp_server_of(qualified) == server


@pytest.mark.parametrize(
    "command,expected",
    [
        ("uv run pytest -q", "uv"),
        ("sudo apt-get install x", "apt-get"),
        ("FOO=1 BAR=2 python run.py", "python"),
        ("sudo FOO=1 apt-get install x", "apt-get"),
        ("  ", "(empty)"),
        ('echo "unbalanced', "echo"),
    ],
)
def test_program_extraction_skips_wrappers_and_assignments(command, expected):
    """The program name is what a reader wants, not the wrapper."""
    assert program_of(command) == expected


def test_summarize_artifacts_reads_only_transcript_artifacts(run_transcript):
    """A whole artifact list can be handed over; non-transcripts are skipped."""
    artifacts = [
        ResearchArtifact(
            filename="provenance_iter1_transcript.json",
            content_base64=base64.b64encode(
                json.dumps(run_transcript).encode("utf-8")
            ).decode("ascii"),
            media_type="application/json",
        ),
        ResearchArtifact(
            filename="results_table.csv",
            content_base64=base64.b64encode(b"gene,score\n").decode("ascii"),
            media_type="text/csv",
        ),
    ]

    stats = summarize_artifacts(artifacts)

    assert stats.sources == ["provenance_iter1_transcript.json"]
    assert stats.skills_used == ["gene-set-enrichment"]


def test_summarize_artifacts_is_empty_without_transcripts():
    """The default artifact policy drops transcripts, so this is the normal case."""
    artifacts = [
        ResearchArtifact(
            filename="results_table.csv",
            content_base64=base64.b64encode(b"gene,score\n").decode("ascii"),
        )
    ]

    assert summarize_artifacts(artifacts).entries == 0


def test_summarize_paths_reads_a_directory(tmp_path, run_transcript):
    """A directory is searched for transcript-named JSON."""
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "iter1_transcript.json").write_text(json.dumps(run_transcript))
    (provenance / "evidence_matrix.json").write_text(json.dumps([{"type": "nope"}]))

    stats = summarize_paths([tmp_path])

    assert stats.sources == [str(provenance / "iter1_transcript.json")]
    assert stats.unrecognized_types == {}


def test_same_named_transcripts_in_different_runs_are_both_read(tmp_path):
    """Two runs both holding provenance/iter1_transcript.json must not collide.

    Keying sources by basename dropped one run's entries entirely, with no
    warning — the summary just quietly described half the work.
    """
    for run, tool in (("run1", "Bash"), ("run2", "WebSearch")):
        provenance = tmp_path / run / "provenance"
        provenance.mkdir(parents=True)
        (provenance / "iter1_transcript.json").write_text(
            json.dumps([tool_call("a", tool)])
        )

    stats = summarize_paths([tmp_path])

    assert len(stats.sources) == 2
    assert stats.entries == 2
    assert stats.distinct_tools == ["Bash", "WebSearch"]


def test_summarize_paths_takes_a_named_file_as_given(tmp_path, run_transcript):
    """An explicitly named file is read whatever it is called."""
    path = tmp_path / "oddly-named.json"
    path.write_text(json.dumps(run_transcript))

    assert summarize_paths([path]).entries == len(run_transcript)


def test_summarize_paths_rejects_a_missing_path(tmp_path):
    """Fail fast rather than silently summarizing nothing."""
    with pytest.raises(FileNotFoundError):
        summarize_paths([tmp_path / "absent.json"])


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "tool_call"},
        ["not-an-object"],
        "a string",
    ],
)
def test_a_malformed_transcript_is_rejected(tmp_path, payload):
    """A transcript that is not a list of objects is an error, not empty stats."""
    path = tmp_path / "bad_transcript.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        summarize_paths([path])


def test_markdown_rendering_covers_the_headline_sections(run_transcript):
    """The rendered summary is what gets committed beside the transcripts."""
    markdown = summarize_transcript(run_transcript).render_markdown()

    assert "## Agent run summary" in markdown
    assert "### Tools used" in markdown
    assert "`search_pubmed`" in markdown
    assert "### Skills invoked" in markdown
    assert "gene-set-enrichment" in markdown
    assert "### Available but unused" in markdown
    assert "Tool success rate: 75.0%" in markdown


def test_a_dotted_tool_name_aggregates_with_its_mcp_spelling():
    """One backend writes 'github.search_issues', another 'mcp__github__…'.

    Both name the same tool, so a summary that split them would understate
    every cross-backend total.
    """
    stats = summarize_transcript(
        [
            {
                "type": "tool_call",
                "id": "a",
                "tool": "github.search_issues",
                "arguments": {},
                "namespace": "github",
            },
            tool_call("b", "mcp__github__search_issues"),
        ]
    )

    assert stats.distinct_tools == ["search_issues"]
    assert stats.tools[0].calls == 2
    assert stats.tools[0].qualified_names == [
        "github.search_issues",
        "mcp__github__search_issues",
    ]
    assert stats.mcp_servers == ["github"]


def test_a_dot_that_is_not_a_server_prefix_is_left_alone():
    """Only the server the entry itself reports may be stripped."""
    stats = summarize_transcript(
        [{"type": "tool_call", "id": "a", "tool": "alpha.beta", "arguments": {}}]
    )

    assert stats.distinct_tools == ["alpha.beta"]


def test_call_ids_are_scoped_to_their_own_transcript():
    """A call id is unique only within one transcript.

    Both iterations of a job number their calls from scratch, so a shared id
    map attributes iteration 1's failure to whichever tool reused the id in
    iteration 2 — and sums both durations onto it.
    """
    stats = summarize_transcripts(
        {
            "iter1_transcript.json": [
                tool_call("1", "Bash"),
                tool_result("1", success=False, duration_ms=500),
            ],
            "iter2_transcript.json": [
                tool_call("1", "WebSearch"),
                tool_result("1", success=True, duration_ms=10),
            ],
        }
    )
    by_name = {tool.name: tool for tool in stats.tools}

    assert by_name["Bash"].failures == 1
    assert by_name["Bash"].total_duration_ms == 500
    assert by_name["WebSearch"].failures == 0
    assert by_name["WebSearch"].total_duration_ms == 10
    assert stats.failed_tool_calls == 1


def test_no_tool_can_report_more_failures_than_calls():
    """The invariant the id collision broke, stated directly."""
    stats = summarize_transcripts(
        {
            f"iter{index}_transcript.json": [
                tool_call("1", f"Tool{index}"),
                tool_result("1", success=False),
            ]
            for index in range(5)
        }
    )

    assert all(tool.failures <= tool.calls for tool in stats.tools)
    assert all(tool.successes >= 0 for tool in stats.tools)


@pytest.mark.parametrize(
    "duration,expected",
    [
        (500, 500),
        (12.5, 12),
        (0, 0),
        (None, None),
        (True, None),
        ("500", None),
    ],
)
def test_duration_accepts_numbers_and_rejects_booleans(duration, expected):
    """``True`` is an int in Python; counting it as 1 ms would be a lie."""
    stats = summarize_transcript(
        [tool_call("a", "Bash"), tool_result("a", duration_ms=duration)]
    )

    assert stats.tools[0].total_duration_ms == expected


def test_a_namespaced_skill_tool_is_still_a_skill():
    """A backend that qualifies the Skill tool still invoked a skill."""
    stats = summarize_transcript(
        [tool_call("a", "mcp__agent__Skill", {"skill": "curate"})]
    )

    assert stats.skills_used == ["curate"]


def test_nested_usage_counters_are_flattened_not_dropped():
    """A usage block nesting its counters must not silently contribute zero."""
    stats = summarize_transcript(
        [
            {
                "type": "task_notification",
                "task_id": "t1",
                "status": "completed",
                "summary": "",
                "output_file": "",
                "usage": {
                    "input_tokens": 100,
                    "cache_creation": {"ephemeral_5m": 10, "ephemeral_1h": 5},
                },
            }
        ]
    )

    assert stats.token_usage == {
        "cache_creation.ephemeral_1h": 5,
        "cache_creation.ephemeral_5m": 10,
        "input_tokens": 100,
    }


def test_a_handler_named_attribute_is_not_reachable_from_transcript_content():
    """Dispatch is an explicit table, not attribute lookup on the entry type."""
    stats = summarize_transcript([{"type": "ignored", "id": "x"}])

    assert stats.unrecognized_types == {"ignored": 1}


def test_markdown_table_survives_a_pipe_in_a_tool_name():
    """Agent-supplied text lands in a markdown table and must not break it."""
    markdown = summarize_transcript(
        [tool_call("a", "we|ird"), {"type": "web_search", "id": "w", "query": "a|b"}]
    ).render_markdown()

    # Split on pipes that are not backslash-escaped: those are the real
    # cell boundaries, and a 4-column row has exactly 5 of them.
    table_rows = [line for line in markdown.splitlines() if line.startswith("| `")]
    assert table_rows
    for row in table_rows:
        assert len(re.split(r"(?<!\\)\|", row)) == 6, row
    assert "`we\\|ird`" in markdown


def test_two_artifacts_sharing_a_filename_are_both_summarized():
    """Merging two runs' artifacts must not drop one of them.

    A provider uniquifies filenames within one bundle, but nothing does across
    bundles — and merging is what summarize_artifacts is for. Keyed by name
    alone, the second transcript replaced the first and its entries vanished.
    """
    def artifact(tool):
        body = json.dumps([tool_call("a", tool)])
        return ResearchArtifact(
            filename="iter1_transcript.json",
            content_base64=base64.b64encode(body.encode("utf-8")).decode("ascii"),
        )

    stats = summarize_artifacts([artifact("Bash"), artifact("WebSearch")])

    assert stats.entries == 2
    assert stats.distinct_tools == ["Bash", "WebSearch"]
    assert stats.sources == [
        "iter1_transcript.json",
        "iter1_transcript.json#2",
    ]


def test_a_pipe_outside_a_table_is_left_alone():
    """Escaping is a table-row rule, not a markdown-wide one.

    A GFM table row is unescaped while the row is split, before inline
    parsing, so a backslash there never reaches the output. Outside a table
    there is no such pass — and inside a code span, which is where most of
    these values sit, the backslash renders literally. Escaping everywhere
    would show the reader a character the agent never produced.
    """
    markdown = summarize_transcript(
        [
            tool_call("a", "Skill", {"skill": "we|ird"}),
            {"type": "web_search", "id": "w", "query": "a|b"},
            {
                "type": "file_change",
                "id": "f",
                "path": "out|put.csv",
                "kind": "create",
                "success": True,
            },
        ]
    ).render_markdown()

    assert "`we|ird`" in markdown
    assert "- a|b" in markdown
    assert "`out|put.csv`" in markdown


def test_a_declared_tool_called_under_another_spelling_is_not_reported_unused():
    """One backend declares mcp__github__search_issues, another calls
    github.search_issues. Reporting it unused is a false statement in the one
    section with no count to look wrong.
    """
    stats = summarize_transcripts(
        {
            "iter1_transcript.json": [
                {
                    "type": "session_init",
                    "tools": ["mcp__github__search_issues", "Bash"],
                }
            ],
            "iter2_transcript.json": [
                {
                    "type": "tool_call",
                    "id": "a",
                    "tool": "github.search_issues",
                    "arguments": {},
                    "namespace": "github",
                }
            ],
        }
    )

    assert stats.distinct_tools == ["search_issues"]
    assert stats.unused_available_tools == ["Bash"]


def test_a_declared_tool_matching_the_call_exactly_is_still_not_unused():
    """The straightforward case must keep working."""
    stats = summarize_transcript(
        [
            {"type": "session_init", "tools": ["Bash", "Read"]},
            tool_call("a", "Bash"),
        ]
    )

    assert stats.unused_available_tools == ["Read"]


def test_one_file_named_two_ways_is_a_single_source(tmp_path, run_transcript):
    """`transcript-stats ./run/x.json run/` is one file, not two.

    Keyed on the spelling, the same transcript was read twice and every tally
    in it doubled.
    """
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    path = provenance / "iter1_transcript.json"
    path.write_text(json.dumps(run_transcript))

    # Two spellings of one file: as given, and via a redundant parent hop.
    detoured = provenance / ".." / "provenance" / "iter1_transcript.json"

    once = summarize_paths([path])
    twice = summarize_paths([path, detoured])

    assert len(twice.sources) == 1
    assert twice.entries == once.entries
    assert twice.tool_calls == once.tool_calls


def test_a_declared_server_tool_is_not_matched_by_an_unrelated_local_tool():
    """Matching on the bare name turns one false report into its mirror.

    `github.notify` and a local `notify` share a short name and nothing else.
    Treating them as the same tool hides a genuinely unused one, in the same
    list where a wrong entry is indistinguishable from a right one.
    """
    stats = summarize_transcript(
        [
            {
                "type": "session_init",
                "tools": ["github.notify", "mcp__github__search_issues"],
            },
            tool_call("a", "mcp__github__search_issues"),
            tool_call("b", "notify"),
        ]
    )

    assert stats.distinct_tools == ["notify", "search_issues"]
    assert stats.unused_available_tools == ["github.notify"]


def test_a_repeated_web_search_is_counted_not_collapsed():
    """A query retried four times is four searches, not one.

    Every neighbouring tally here is a count keyed by the thing; this was the
    one place a repeat vanished, under a heading that read as a total.
    """
    def search(query):
        return {"type": "web_search", "id": "w", "query": query, "raw": {}}

    stats = summarize_transcript([search("a"), search("a"), search("a"), search("b")])

    assert stats.web_searches == ["a", "b"]
    assert stats.web_search_counts == {"a": 3, "b": 1}

    markdown = stats.render_markdown()
    assert "### Web searches (4, 2 distinct)" in markdown
    assert "- a (x3)" in markdown
    assert "- b" in markdown


def test_a_single_web_search_reports_one_number():
    """No "N distinct" noise when nothing was repeated."""
    stats = summarize_transcript(
        [{"type": "web_search", "id": "w", "query": "only once", "raw": {}}]
    )

    assert "### Web searches (1)" in stats.render_markdown()


def test_a_web_search_with_no_query_still_counts_as_a_search():
    """The heading reports a total, so a dropped entry contradicts it.

    "Web searches (0)" beside three web_search entries in the type counts
    reads as a bug rather than a definition.
    """
    stats = summarize_transcript(
        [
            {"type": "web_search", "id": "w1", "query": "", "raw": {}},
            {"type": "web_search", "id": "w2", "raw": {}},
            {"type": "web_search", "id": "w3", "query": "real query", "raw": {}},
        ]
    )

    assert stats.entry_types["web_search"] == 3
    assert sum(stats.web_search_counts.values()) == 3
    assert stats.web_search_counts["(no query recorded)"] == 2


def test_the_distinct_query_list_is_derived_from_the_counts():
    """One source of truth, so the two cannot disagree."""
    stats = summarize_transcript(
        [{"type": "web_search", "id": "w", "query": q, "raw": {}} for q in "aab"]
    )

    assert stats.web_searches == ["a", "b"]
    assert json.loads(stats.model_dump_json())["web_searches"] == ["a", "b"]


@pytest.mark.parametrize(
    "declared,expected",
    [
        (["Bash", "Read"], ["Bash", "Read"]),
        ("Bash", []),
        (None, []),
        (42, []),
    ],
)
def test_a_malformed_tools_field_does_not_become_phantom_tools(declared, expected):
    """`"tools": "Bash"` would otherwise declare B, a, s and h.

    Transcript content is provider-produced JSON, so a malformed field is
    skipped rather than raised on — the rest of the summary is still worth
    having.
    """
    stats = summarize_transcript([{"type": "session_init", "tools": declared}])

    assert stats.available_tools == expected
