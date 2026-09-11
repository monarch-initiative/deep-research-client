"""Tests for the configurable artifact selection policy.

The defaults must keep reproducing the previously hardcoded curated set, and
each knob must widen or narrow exactly the layer it names.
"""

import io
import zipfile

import pytest

from deep_research_client.artifact_selection import (
    ARTIFACT_RULES,
    DEFAULT_ALLOWED_EXTENSIONS,
    DEFAULT_MAX_BYTES,
    DEFAULT_SCAFFOLDING_DIRECTORIES,
    DEFAULT_SCAFFOLDING_PREFIXES,
    RUNTIME_EXTENSIONS,
    ArtifactDecision,
    ArtifactSelectionPolicy,
    _with_trailing_slashes,
    is_under_directory,
    is_under_normalized_directory,
    normalize_member_path,
    split_name_list,
)
from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import OpenScientistParams
from deep_research_client.providers.openscientist import OpenScientistProvider

ONE_MB = 1024 * 1024


@pytest.fixture
def default_policy() -> ArtifactSelectionPolicy:
    """Policy built from stock OpenScientist params."""
    return ArtifactSelectionPolicy.from_params(OpenScientistParams())


@pytest.mark.parametrize(
    "name,expected",
    [
        ("results/table.csv", True),
        ("provenance/evidence_matrix.json", True),
        ("provenance/evidence_matrix.png", True),
        ("final_report.pdf", True),
        ("figures/plot.svg", True),
        (".claude/skills/runtime.md", False),
        ("__pycache__/module.pyc", False),
        ("agent-container.log", False),
        ("provenance/iter1_transcript.json", False),
        ("logs/agent_stdout.json", False),
        ("logs/agent_stderr.json", False),
        ("raw/archive.zip", False),
        ("scratch/notes.txt", False),
        ("scratch/session.jsonl", False),
    ],
)
def test_default_policy_reproduces_the_curated_set(default_policy, name, expected):
    """Stock params keep the same members the hardcoded filter kept."""
    assert bool(default_policy.decide(name, 100)) is expected


def test_size_cap_applies_before_any_allow(default_policy):
    """An oversized member is dropped whatever its extension."""
    decision = default_policy.decide("results/table.csv", 10 * ONE_MB)

    assert not decision.keep
    assert "artifact_max_bytes" in decision.reason


def test_size_cap_beats_an_explicit_include():
    """include_globs cannot be used to smuggle a huge file into memory."""
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(artifact_include_globs=["*"], artifact_max_bytes=1024)
    )

    assert not policy.decide("results/huge.csv", 4096).keep
    assert policy.decide("results/small.csv", 512).keep


def test_provider_deny_drops_the_consumed_report(default_policy):
    """The markdown already returned as the result body is not re-emitted."""
    assert not default_policy.decide(
        "final_report.md", 100, provider_deny={"final_report.md"}
    ).keep
    assert default_policy.decide("final_report.md", 100).keep


@pytest.mark.parametrize(
    "name",
    [
        "provenance/iter1_transcript.json",
        "provenance/report_transcript.json",
        "agent-container.log",
        "logs/agent_stdout.txt",
    ],
)
def test_keep_runtime_preserves_agent_records(name):
    """artifact_keep_runtime is enough on its own to get transcripts and logs.

    The switch has to extend the extension allowlist too, or a kept ``.log``
    would clear the noise filter and then fall at the extension check.
    """
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(artifact_keep_runtime=True)
    )

    assert policy.decide(name, 100).keep


def test_keep_runtime_does_not_open_up_scaffolding_or_archives():
    """Widening runtime records leaves the other default denies in place."""
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(artifact_keep_runtime=True)
    )

    assert not policy.decide(".claude/skills/runtime.md", 100).keep
    assert not policy.decide("raw/archive.zip", 100).keep
    assert not policy.decide("scratch/tempfile.tmp", 100).keep


def test_effective_extensions_only_grows_under_keep_runtime(default_policy):
    """The runtime extensions are added by the switch, not present by default."""
    assert default_policy.effective_extensions == DEFAULT_ALLOWED_EXTENSIONS
    assert default_policy.with_overrides(keep_runtime=True).effective_extensions == (
        DEFAULT_ALLOWED_EXTENSIONS | RUNTIME_EXTENSIONS
    )


@pytest.mark.parametrize(
    "extensions",
    [
        [".yaml"],
        ["yaml"],
        ["YAML"],
        " yaml , ",
    ],
)
def test_extra_extensions_accept_loose_spellings(extensions):
    """Extensions work with or without a dot, in any case, from a CLI string."""
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(artifact_extra_extensions=extensions)
    )

    assert policy.decide("config/run.yaml", 100).keep


def test_extra_extensions_extend_rather_than_replace():
    """Adding an extension keeps the built-in allowlist intact."""
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(artifact_extra_extensions=[".yaml"])
    )

    assert policy.decide("results/table.csv", 100).keep


def test_include_globs_bypass_the_default_denies():
    """An explicit include wins over noise, scaffolding, and the allowlist."""
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(
            artifact_include_globs=["provenance/*_transcript.json", ".claude/*"]
        )
    )

    assert policy.decide("provenance/iter1_transcript.json", 100).keep
    assert policy.decide(".claude/skills/runtime.md", 100).keep
    assert not policy.decide("provenance/other.log", 100).keep


def test_include_globs_cross_directory_separators():
    """'*' spans '/', so a bare pattern reaches nested members."""
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(artifact_include_globs=["*.jsonl"])
    )

    assert policy.decide("deep/nested/session.jsonl", 100).keep


def test_exclude_globs_beat_include_globs():
    """The explicit deny is the highest-precedence rule."""
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(
            artifact_include_globs=["provenance/*"],
            artifact_exclude_globs=["*_transcript.json"],
        )
    )

    assert policy.decide("provenance/evidence_matrix.json", 100).keep
    assert not policy.decide("provenance/iter1_transcript.json", 100).keep


def test_exclude_globs_narrow_the_default_set(default_policy):
    """A caller can drop something the defaults would have kept."""
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(artifact_exclude_globs=["*.pdf"])
    )

    assert default_policy.decide("final_report.pdf", 100).keep
    assert not policy.decide("final_report.pdf", 100).keep


def test_decision_reason_names_the_knob_to_turn(default_policy):
    """A skip explains itself, so the caller knows which field to set."""
    assert "artifact_keep_runtime" in default_policy.decide(
        "provenance/iter1_transcript.json", 100
    ).reason
    assert "artifact_extra_extensions" in default_policy.decide(
        "notes.rst", 100
    ).reason


def test_with_overrides_does_not_mutate_the_original(default_policy):
    """The policy is frozen; an override returns a separate instance."""
    widened = default_policy.with_overrides(keep_runtime=True)

    assert widened is not default_policy
    assert not default_policy.keep_runtime
    assert widened.keep_runtime


@pytest.mark.parametrize(
    "field,raw,expected",
    [
        ("artifact_include_globs", "a/*.json,b/*", ["a/*.json", "b/*"]),
        ("artifact_exclude_globs", "*.log", ["*.log"]),
        ("artifact_extra_extensions", ".yaml, .toml", [".yaml", ".toml"]),
        ("artifact_include_globs", "", []),
    ],
)
def test_list_params_accept_a_comma_separated_cli_string(field, raw, expected):
    """CLI '--param key=value' pairs arrive as strings and must still parse."""
    params = OpenScientistParams(**{field: raw})

    assert getattr(params, field) == expected


def test_list_params_still_accept_a_real_list():
    """String splitting does not break programmatic use."""
    params = OpenScientistParams(artifact_include_globs=["a/*", "b/*"])

    assert params.artifact_include_globs == ["a/*", "b/*"]


# --- Provider-level wiring -------------------------------------------------
#
# The policy above is only useful if the provider actually consults the params
# it was given, so these drive a real ZIP through the extraction path.


def _bundle() -> bytes:
    """A ZIP shaped like an OpenScientist job's artifacts archive."""
    buffer = io.BytesIO()
    files: dict[str, str | bytes] = {
        ".claude/skills/runtime.md": "runtime scaffolding",
        "agent-container.log": "log output",
        "final_report.md": "# Report",
        "final_report.pdf": b"%PDF-1.4 fake",
        "provenance/evidence_matrix.json": '{"rows":[]}',
        "provenance/iter1_transcript.json": '{"messages":[]}',
        "provenance/report_transcript.json": '{"messages":[]}',
        "raw/archive.zip": b"PK",
        "results/table.csv": "gene,score\nATP7B,0.9\n",
    }
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def _extract(params: OpenScientistParams | None = None) -> list[str]:
    """Return the artifact filenames the provider keeps for the test bundle."""
    config = ProviderConfig(
        name="openscientist",
        api_key="test-api-key",
        base_url="https://example.test/",
        enabled=True,
    )
    provider = OpenScientistProvider(config, params)
    artifacts = provider._extract_artifacts_from_artifact_zip(
        _bundle(), report_names={"final_report.md"}
    )
    return [artifact.filename for artifact in artifacts]


def test_provider_default_extraction_is_unchanged():
    """The refactor must not move the default set."""
    assert _extract() == [
        "final_report.pdf",
        "provenance_evidence_matrix.json",
        "results_table.csv",
    ]


def test_provider_keeps_transcripts_when_asked():
    """The dismech case: agent transcripts are provenance worth committing."""
    kept = _extract(OpenScientistParams(artifact_keep_runtime=True))

    assert "provenance_iter1_transcript.json" in kept
    assert "provenance_report_transcript.json" in kept
    assert "agent-container.log" in kept


def test_provider_honours_include_and_exclude_globs():
    """Narrow includes reach one family of runtime files without the rest."""
    kept = _extract(
        OpenScientistParams(
            artifact_include_globs=["provenance/*_transcript.json"],
            artifact_exclude_globs=["*report_transcript*"],
        )
    )

    assert "provenance_iter1_transcript.json" in kept
    assert "provenance_report_transcript.json" not in kept
    assert "agent-container.log" not in kept


def test_provider_never_re_emits_the_report_body():
    """Even the widest selection leaves the consumed markdown out."""
    kept = _extract(OpenScientistParams(artifact_include_globs=["*"]))

    assert "final_report.md" not in kept
    assert "raw_archive.zip" in kept


def test_scaffolding_is_matched_at_any_depth():
    """A nested .claude/ is the same scaffolding as one at the bundle root.

    These directories are frequently .json, so matching only at the root let
    a nested one clear the extension allowlist and become an artifact.
    """
    policy = ArtifactSelectionPolicy.from_params(OpenScientistParams())

    assert not policy.decide("workspace/.claude/settings.json", 100).keep
    assert not policy.decide("src/__pycache__/cache.json", 100).keep
    assert not policy.decide("a/b/node_modules/pkg/data.json", 100).keep


def test_a_directory_named_like_a_prefix_is_not_matched_by_accident():
    """Matching is on a path segment, not a substring."""
    policy = ArtifactSelectionPolicy.from_params(OpenScientistParams())

    assert policy.decide("mycache/data.csv", 100).keep
    assert policy.decide("results/cached_table.csv", 100).keep


def test_a_none_size_cap_falls_back_to_the_default():
    """from_params is duck-typed; a params object may declare the cap optional."""

    class LooseParams:
        artifact_max_bytes = None

    policy = ArtifactSelectionPolicy.from_params(LooseParams())

    assert policy.max_bytes == DEFAULT_MAX_BYTES
    assert policy.decide("results/table.csv", 100).keep


def test_the_root_report_body_is_still_not_re_emitted():
    """The main download path passes no report_names, so the root dedup holds."""
    kept = _extract(OpenScientistParams(artifact_include_globs=["*"]))

    assert "final_report.md" not in kept


def test_a_nested_report_markdown_is_reachable_by_an_include_glob():
    """Only the bundle root holds the report body.

    A nested analysis/report.md is a different document; dropping it as
    "already returned as the report body" was untrue, and sat above
    include_globs in precedence so nothing could recover it.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("final_report.md", "# Report")
        archive.writestr("analysis/subtopic/report.md", "# Subtopic")

    config = ProviderConfig(
        name="openscientist", api_key="k", base_url="https://example.test/", enabled=True
    )
    provider = OpenScientistProvider(
        config, OpenScientistParams(artifact_include_globs=["analysis/*"])
    )
    kept = [
        artifact.filename
        for artifact in provider._extract_artifacts_from_artifact_zip(buffer.getvalue())
    ]

    assert kept == ["analysis_subtopic_report.md"]


def test_a_nested_skill_report_is_not_chosen_as_the_report_body():
    """The report picker and the selection policy must agree on scaffolding.

    `_is_report_markdown_name` matches at any depth, so a root-anchored
    scaffolding check left a nested `.claude/skills/.../report.md` eligible —
    and the picker takes the first candidate in ZIP order, which is the order
    the server happened to write them. The whole result body became skill
    boilerplate.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("workspace/.claude/skills/writer/report.md", "# Boilerplate")
        archive.writestr("final_report.md", "# The actual report")

    config = ProviderConfig(
        name="openscientist", api_key="k", base_url="https://example.test/", enabled=True
    )
    provider = OpenScientistProvider(config)
    name, body = provider._extract_markdown_from_artifact_zip(
        buffer.getvalue(), "job-1"
    )

    assert name == "final_report.md"
    assert body == "# The actual report"


@pytest.mark.parametrize(
    "name,expected",
    [
        (".claude/skills/x.md", True),
        ("workspace/.claude/skills/x.md", True),
        ("a/b/__pycache__/x.json", True),
        ("run/logs_summary.csv", False),
        ("results/claude_notes.md", False),
    ],
)
def test_the_shared_scaffolding_rule_matches_whole_segments(name, expected):
    """The convenience wrapper, which normalizes both sides for a caller.

    Production goes through the normalized-input variant instead; that pair is
    covered by test_the_pair_production_actually_uses.
    """
    assert is_under_directory(name, DEFAULT_SCAFFOLDING_PREFIXES) is expected


def test_scaffolding_names_are_normalized_so_a_caller_cannot_break_matching():
    """A directory given without a trailing slash must not match a substring."""
    policy = ArtifactSelectionPolicy(
        max_bytes=1024, scaffolding_prefixes=("logs",)
    )

    assert policy.scaffolding_prefixes == ("logs/",)
    assert policy.decide("run/logs_summary.csv", 100).keep
    assert not policy.decide("run/logs/out.csv", 100).keep


def test_a_zero_size_cap_is_honoured_rather_than_replaced():
    """0 means "keep nothing"; only an absent cap falls back to the default."""

    class LooseParams:
        artifact_max_bytes = 0

    policy = ArtifactSelectionPolicy.from_params(LooseParams())

    assert policy.max_bytes == 0
    assert not policy.decide("results/table.csv", 1).keep


@pytest.mark.parametrize(
    "names,expected",
    [
        (["analysis/subtopic/report.md", "final_report.md"], "final_report.md"),
        (["final_report.md", "analysis/subtopic/report.md"], "final_report.md"),
        (["report.md", "final_report.md"], "final_report.md"),
        (["final_report.md", "report.md"], "final_report.md"),
        (["analysis/subtopic/report.md"], "analysis/subtopic/report.md"),
    ],
)
def test_the_report_body_is_chosen_independently_of_zip_order(names, expected):
    """The picker took the first match in server-controlled write order.

    A bundle holding analysis/subtopic/report.md before final_report.md
    returned the subtopic document as the entire research result, and two root
    candidates resolved by write order alone.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, f"# {name}")

    config = ProviderConfig(
        name="openscientist", api_key="k", base_url="https://example.test/", enabled=True
    )
    provider = OpenScientistProvider(config)
    picked, body = provider._extract_markdown_from_artifact_zip(buffer.getvalue(), "j")

    assert picked == expected
    assert body == f"# {expected}"


def test_a_mixed_case_scaffolding_name_still_matches():
    """Both sides of the comparison are normalized, as everywhere else here."""
    policy = ArtifactSelectionPolicy(max_bytes=1024, scaffolding_prefixes=(".Codex/",))

    assert not policy.decide("workspace/.codex/state.json", 100).keep
    assert is_under_directory("A/.CODEX/state.json", [".codex"])


def test_a_bare_string_of_directories_is_rejected():
    """Iterating a string into characters would match nothing, silently."""
    with pytest.raises(TypeError):
        is_under_directory("a/.claude/x.md", ".claude/")


def test_a_zero_cap_also_drops_a_zero_byte_member():
    """A cap of zero has to include an empty file in "keep nothing"."""

    class LooseParams:
        artifact_max_bytes = 0

    policy = ArtifactSelectionPolicy.from_params(LooseParams())

    assert not policy.decide("results/empty.csv", 0).keep
    assert not policy.decide("results/table.csv", 1).keep


@pytest.mark.parametrize(
    "names,expected",
    [
        # Tier 2: no canonical report name, but "report" in the path.
        (
            ["analysis/status_report.md", "weekly_report.md"],
            "weekly_report.md",
        ),
        (
            ["weekly_report.md", "analysis/status_report.md"],
            "weekly_report.md",
        ),
        # Tier 3: no "report" anywhere. A README is markdown at the root of
        # almost every bundle and should not win on depth alone.
        (["README.md", "analysis/writeup.md"], "analysis/writeup.md"),
        (["analysis/writeup.md", "README.md"], "analysis/writeup.md"),
        (["README.md"], "README.md"),
    ],
)
def test_the_fallback_tiers_are_ordered_too(names, expected):
    """Every earlier picker test had a tier-1 candidate, so the sort was
    only ever exercised on canonical report names."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, f"# {name}")

    config = ProviderConfig(
        name="openscientist", api_key="k", base_url="https://example.test/", enabled=True
    )
    provider = OpenScientistProvider(config)
    picked, _ = provider._extract_markdown_from_artifact_zip(buffer.getvalue(), "j")

    assert picked == expected


def test_the_zero_cap_reason_does_not_claim_a_size_comparison():
    """`reason` is shown to a user, so "0 bytes exceeds 0" would be a lie."""

    class LooseParams:
        artifact_max_bytes = 0

    decision = ArtifactSelectionPolicy.from_params(LooseParams()).decide(
        "results/empty.csv", 0
    )

    assert not decision.keep
    assert decision.rule == "size_cap"
    assert "exceeds" not in decision.reason


def test_a_member_exactly_at_the_cap_is_kept():
    """The cap is a maximum, not an exclusive bound."""
    policy = ArtifactSelectionPolicy(max_bytes=1024)

    assert policy.decide("results/table.csv", 1024).keep
    assert not policy.decide("results/table.csv", 1025).keep


def test_the_normalized_matcher_also_rejects_a_bare_string():
    """A bare string here matches wrongly rather than not at all.

    Iterating ".claude/" yields "." among others, and "/." is in any path with
    a dot-segment — so the first thing a caller tries looks correct and the
    bug only shows on paths without one.
    """
    with pytest.raises(TypeError):
        is_under_normalized_directory("a/.claude/x.md", ".claude/")


def test_the_default_directories_are_derived_from_the_shared_rule():
    """The constant and the policy must normalize by one rule, not two copies.

    Asserted on an input the rule actually has to transform. Every default is
    already lowercase and slash-terminated, so comparing the defaults to
    themselves passes even if the two sides normalize differently — which is
    the divergence this is supposed to catch.
    """
    policy = ArtifactSelectionPolicy(
        max_bytes=1024, scaffolding_prefixes=(".Codex", "Logs/")
    )

    # Against a literal, not against the same function the policy called:
    # `_with_trailing_slashes(x) == policy.scaffolding_prefixes` is f(x) == f(x).
    assert policy.scaffolding_prefixes == (".codex/", "logs/")
    # Goes red if the constant is ever re-inlined as a hand-written copy,
    # which is the round-6 regression.
    assert DEFAULT_SCAFFOLDING_DIRECTORIES == _with_trailing_slashes(
        DEFAULT_SCAFFOLDING_PREFIXES
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        (".claude/skills/x.md", True),
        ("workspace/.claude/skills/x.md", True),
        ("a/b/__pycache__/x.json", True),
        ("run/logs_summary.csv", False),
        ("results/claude_notes.md", False),
    ],
)
def test_the_pair_production_actually_uses(name, expected):
    """The provider runs is_under_normalized_directory against the constant.

    The sibling test covers is_under_directory, which no production caller
    uses any more.
    """
    assert (
        is_under_normalized_directory(
            normalize_member_path(name), DEFAULT_SCAFFOLDING_DIRECTORIES
        )
        is expected
    )


def test_a_negative_cap_reports_its_own_value():
    """Reporting "0" would be untrue, the defect this branch was split out for."""

    class LooseParams:
        artifact_max_bytes = -1

    decision = ArtifactSelectionPolicy.from_params(LooseParams()).decide("x.csv", 0)

    assert not decision.keep
    assert "-1" in decision.reason


def test_every_decision_carries_a_documented_rule():
    """The slug set and the returns must be one thing, not two.

    The docstring invites callers to branch on `rule`, so an omission is a
    silent miss for them — the enumeration written by hand left out
    `media_type`, the rule that keeps an image whose suffix is not in the
    allowlist.
    """
    policy = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(
            artifact_include_globs=["included/*"],
            artifact_exclude_globs=["excluded/*"],
        )
    )
    outcomes = [
        policy.decide("excluded/x.csv", 10),
        policy.decide("results/huge.csv", 99 * ONE_MB),
        policy.decide("final_report.md", 10, provider_deny={"final_report.md"}),
        policy.decide("included/anything.bin", 10),
        policy.decide(".claude/settings.json", 10),
        policy.decide("raw/archive.zip", 10),
        policy.decide("provenance/iter1_transcript.json", 10),
        policy.decide("results/table.csv", 10),
        policy.decide("figures/plot.bmp", 10),
        policy.decide("notes.rst", 10),
    ]

    seen = {decision.rule for decision in outcomes}
    assert "" not in seen, "a decision was returned without a rule slug"
    assert seen <= ARTIFACT_RULES, seen - ARTIFACT_RULES
    # Every documented slug is reachable, so the set has no dead entries.
    assert seen == ARTIFACT_RULES


def test_the_image_keep_is_the_one_the_enumeration_omitted():
    """A .bmp is not in the allowlist and is kept on its media type alone."""
    policy = ArtifactSelectionPolicy(max_bytes=1024)
    decision = policy.decide("figures/plot.bmp", 10)

    assert decision.keep
    assert decision.rule == "media_type"


def test_an_undocumented_rule_slug_is_rejected_at_construction():
    """The set and the returns are held together by the type, not by a list.

    A test that enumerates outcomes is a second hand-maintained copy of the
    same thing, which is how `media_type` came to be undocumented. Validating
    here means a new decide branch fails the moment any test reaches it.
    """
    with pytest.raises(ValueError, match="unknown decision rule"):
        ArtifactDecision(keep=False, reason="something", rule="not_a_real_rule")


def test_a_decision_may_still_carry_no_slug():
    """The empty default stays legal for a decision built outside decide."""
    assert ArtifactDecision(keep=True, reason="hand-built").rule == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("*.json", ("*.json",)),
        ("a/*, b/*", ("a/*", "b/*")),
        (["a/*", "b/*"], ("a/*", "b/*")),
        (None, ()),
        ((), ()),
    ],
)
def test_a_duck_typed_glob_setting_reads_the_same_as_the_pydantic_one(raw, expected):
    """A bare string is the trap: tuple("*.json") starts with '*'.

    include_globs sits above every default deny and '*' matches every path,
    so one stray character preserved the entire bundle — archives and
    transcripts included.
    """

    class LooseParams:
        artifact_include_globs = raw

    policy = ArtifactSelectionPolicy.from_params(LooseParams())

    assert policy.include_globs == expected


def test_a_bare_string_glob_no_longer_preserves_the_whole_bundle():
    """The concrete failure, stated as the outcome rather than the parse."""

    class LooseParams:
        artifact_include_globs = "*.json"

    policy = ArtifactSelectionPolicy.from_params(LooseParams())

    assert not policy.decide("raw/archive.zip", 100).keep
    assert policy.decide("results/table.json", 100).keep


def test_a_bare_string_extra_extension_is_not_split_into_characters():
    """"csv" became {".c", ".s", ".v"} before the shared splitter."""

    class LooseParams:
        artifact_extra_extensions = "csv,yaml"

    policy = ArtifactSelectionPolicy.from_params(LooseParams())

    assert policy.decide("config/run.yaml", 100).keep
    assert not policy.decide("notes.c", 100).keep


def test_both_doors_into_the_settings_read_a_string_the_same_way():
    """The pydantic validator and from_params share one rule."""
    typed = ArtifactSelectionPolicy.from_params(
        OpenScientistParams(artifact_include_globs="a/*, b/*")
    )

    class LooseParams:
        artifact_include_globs = "a/*, b/*"

    assert typed.include_globs == ArtifactSelectionPolicy.from_params(
        LooseParams()
    ).include_globs


def test_a_bare_string_exclude_glob_no_longer_empties_the_bundle():
    """exclude_globs sits at precedence 1, so a stray '*' dropped everything.

    Worse than the include direction, which over-preserves visibly: preserving
    nothing looks like a run that simply produced no files.
    """

    class LooseParams:
        artifact_exclude_globs = "*.log"

    policy = ArtifactSelectionPolicy.from_params(LooseParams())

    assert policy.exclude_globs == ("*.log",)
    assert policy.decide("results/table.csv", 100).keep
    assert not policy.decide("agent-container.log", 100).keep


@pytest.mark.parametrize(
    "value,expected",
    [
        (["a/*", "b/*"], ("a/*", "b/*")),
        (("a/*", "b/*"), ("a/*", "b/*")),
        ({"a/*": 1}.keys(), ("a/*",)),
        ({"b/*", "a/*"}, ("a/*", "b/*")),
        (frozenset({"a/*"}), ("a/*",)),
    ],
)
def test_any_re_readable_collection_of_names_is_accepted(value, expected):
    """dict_keys worked before the splitter existed and must still.

    An unordered collection is sorted, so two policies built from equal sets
    are equal — the dataclass compares by field.
    """
    assert split_name_list(value) == expected


@pytest.mark.parametrize("value", [42, 3.5, object(), {"a": 1}])
def test_a_setting_that_is_not_a_list_of_names_raises(value):
    """Silently yielding no names is the worst outcome available here.

    An empty exclude_globs keeps every log the caller asked to drop; an empty
    include_globs leaves denied the file they meant to rescue. `tuple(42)`
    used to raise, and the first version of the splitter swallowed it.
    """
    with pytest.raises(TypeError, match="expected a list of names"):
        split_name_list(value)


@pytest.mark.parametrize(
    "value", [b"*.json", bytearray(b"*.json"), memoryview(b"*.json")]
)
def test_bytes_are_not_read_as_a_list_of_names(value):
    """The one string-like that is also an iterable of something else.

    `tuple(b"*.json")` is six integers, so the bundle would be filtered
    against patterns named "42" and "46" — a deny list that denies nothing
    and an allow list that allows nothing.
    """
    with pytest.raises(TypeError, match="decode it to str first"):
        split_name_list(value)


@pytest.mark.parametrize(
    "make_value",
    [
        lambda: (name for name in ["a/*"]),
        lambda: iter(["a/*"]),
        lambda: map(str, ["a/*"]),
    ],
)
def test_a_one_shot_iterator_is_refused_rather_than_read_once(make_value):
    """It reads correctly once and empty after, which nothing would notice.

    A params object outlives the policy built from it — `artifact_policy` is
    a per-instance cached_property — so a second provider sharing the params
    would silently select against no patterns at all.
    """
    with pytest.raises(TypeError, match="re-readable collection"):
        split_name_list(make_value())


def test_two_policies_from_equal_sets_compare_equal():
    """tuple(some_set) has no defined order, and the policy is a dataclass.

    Passed raw rather than pre-split, because __post_init__ is what has to
    hold this — routing through the splitter by hand would test the caller.
    """
    first = ArtifactSelectionPolicy(max_bytes=1024, include_globs={"a", "b"})
    second = ArtifactSelectionPolicy(max_bytes=1024, include_globs={"b", "a"})

    assert first == second


def test_a_bare_string_exclude_glob_through_with_overrides_keeps_the_bundle():
    """with_overrides was one of two doors the params-side fix did not reach.

    `replace()` re-runs __post_init__, so normalizing there covers it; the
    `**changes: object` signature means the type checker never would.
    """
    policy = ArtifactSelectionPolicy(max_bytes=1024).with_overrides(
        exclude_globs="*.log"
    )

    assert policy.exclude_globs == ("*.log",)
    assert policy.decide("results/table.csv", 100).keep
    assert not policy.decide("agent-container.log", 100).keep


def test_a_bare_string_include_glob_through_the_constructor_is_split():
    """The other door. A stray "*" here force-keeps archives and transcripts."""
    policy = ArtifactSelectionPolicy(max_bytes=1024, include_globs="*.json")

    assert policy.include_globs == ("*.json",)
    assert not policy.decide("raw/archive.zip", 100).keep
    assert policy.decide("results/table.json", 100).keep


def test_the_constructor_refuses_a_glob_list_the_splitter_refuses():
    """One rule, one home: __post_init__ raises where from_params would."""
    with pytest.raises(TypeError, match="re-readable collection"):
        ArtifactSelectionPolicy(
            max_bytes=1024, exclude_globs=iter(["*.log"])
        )


def test_a_bare_string_archive_extension_drops_every_extensionless_member():
    """"suffix in self.archive_extensions" is a substring test on a string.

    PurePosixPath("results/LICENSE").suffix is "", and "" is a substring of
    every string, so the whole extensionless tail of a bundle was dropped as
    a nested archive — above the allowlist, so nothing downstream could
    rescue it.
    """
    policy = ArtifactSelectionPolicy(max_bytes=1024, archive_extensions=".zip")

    assert policy.archive_extensions == frozenset({".zip"})
    assert policy.decide("raw/bundle.zip", 100).rule == "archive"
    assert policy.decide("results/LICENSE", 100).rule != "archive"
    assert policy.decide("results/x.zi", 100).rule != "archive"


def test_a_bare_string_allowed_extension_keeps_every_extensionless_member():
    """The mirror of the archive case, failing in the opposite direction.

    Same mistake, one field over: there it dropped everything extensionless,
    here it kept everything extensionless as "allowed extension".
    """
    policy = ArtifactSelectionPolicy(max_bytes=1024, allowed_extensions=".json")

    assert policy.allowed_extensions == frozenset({".json"})
    assert policy.decide("results/table.json", 100).keep
    assert not policy.decide("results/LICENSE", 100).keep


def test_a_bare_string_allowed_extension_no_longer_breaks_the_runtime_union():
    """effective_extensions evaluated `str | frozenset` and raised.

    Loud with keep_runtime=True and silent without it — the same mistake
    reported differently depending on an unrelated knob.
    """
    policy = ArtifactSelectionPolicy(
        max_bytes=1024, allowed_extensions=".json", keep_runtime=True
    )

    assert ".json" in policy.effective_extensions
    assert ".log" in policy.effective_extensions


def test_a_bare_string_runtime_fragment_no_longer_matches_every_basename():
    """Iterated into characters, "t" in basename dropped nearly everything."""
    policy = ArtifactSelectionPolicy(
        max_bytes=1024, runtime_name_fragments="transcript"
    )

    assert policy.decide("results/table.csv", 100).keep
    assert not policy.decide("provenance/iter1_transcript.json", 100).keep


@pytest.mark.parametrize("spelling", ["csv", ".csv", ".CSV", "CSV"])
def test_an_extension_is_normalized_at_every_door_not_only_from_params(spelling):
    """_normalize_extensions ran only in from_params, so these matched nothing.

    normalize_member_path lowercases the path before any matcher sees it, so
    an uppercase spelling could never match, and a dotless one never acquired
    a dot — a setting that silently selects nothing, which is the failure this
    module keeps closing off.
    """
    policy = ArtifactSelectionPolicy(max_bytes=1024).with_overrides(
        allowed_extensions=frozenset({spelling})
    )

    assert policy.decide("results/table.csv", 100).keep


def test_a_runtime_suffix_is_lowercased_to_match_the_normalized_path():
    """Same silent-nothing-matches shape as an uppercase extension.

    Deliberately not spelled with "transcript": that is a default runtime
    *fragment*, so the member would be dropped by a different rule and the
    test would pass whether or not the suffix was lowercased.
    """
    policy = ArtifactSelectionPolicy(max_bytes=1024).with_overrides(
        runtime_suffixes=("_DUMP.JSON",)
    )

    dropped = policy.decide("provenance/iter1_dump.json", 100)

    assert not dropped.keep
    assert dropped.rule == "runtime"
    assert policy.decide("results/table.json", 100).keep


def test_every_unordered_collection_is_sorted_not_only_a_real_set():
    """The Args promise covers unordered collections, not `set` specifically.

    A KeysView and a custom Set are just as unordered as a set; sorting only
    the concrete types made the promise true of some of them and not others.
    """
    from collections.abc import Set as AbstractSet

    class OrderedLikeASet(AbstractSet):
        def __init__(self, items):
            self._items = list(items)

        def __contains__(self, item):
            return item in self._items

        def __iter__(self):
            return iter(self._items)

        def __len__(self):
            return len(self._items)

    assert split_name_list(OrderedLikeASet(["b/*", "a/*"])) == ("a/*", "b/*")
    assert split_name_list({"b/*": 1, "a/*": 2}.keys()) == ("a/*", "b/*")


def test_from_params_still_merges_extra_extensions_onto_the_defaults():
    """__post_init__ now normalizes, so from_params only has to merge."""

    class LooseParams:
        artifact_extra_extensions = "csv, YAML"

    policy = ArtifactSelectionPolicy.from_params(LooseParams())

    assert policy.decide("results/table.csv", 100).keep
    assert policy.decide("config/run.yaml", 100).keep
    assert policy.decide("results/table.json", 100).keep


@pytest.mark.parametrize(
    "field,setting",
    [
        ("runtime_name_fragments", ["stderr", ""]),
        ("runtime_suffixes", ["", ".log"]),
    ],
)
def test_an_empty_entry_in_a_runtime_list_does_not_drop_the_whole_bundle(
    field, setting
):
    """An empty name is inert in a glob and lethal in a substring test.

    `"" in basename` and `basename.endswith("")` are both true for every
    member, and both rules sit above the extension allowlist — so one blank
    entry dropped everything, and only when keep_runtime was False.

    `["a", ""]` is what `"a,".split(",")` produces, so a caller assembling
    the list from their own config is the way in. The string spelling
    `"stderr,"` was already handled; the list was the surprising one.
    """
    policy = ArtifactSelectionPolicy(max_bytes=1024, **{field: setting})

    assert policy.decide("results/table.csv", 100).keep


def test_the_runtime_names_that_are_not_empty_still_match():
    """The empty entry is dropped, not the whole setting.

    Asserts the *rule*, not just that the member was dropped: neither .txt
    nor .log is in the default allowlist, so both files are refused anyway
    and a keep/drop assertion would pass even if the setting were emptied
    entirely.
    """
    fragments = ArtifactSelectionPolicy(
        max_bytes=1024, runtime_name_fragments=["stderr", ""]
    )
    suffixes = ArtifactSelectionPolicy(max_bytes=1024, runtime_suffixes=["", ".log"])

    assert fragments.decide("logs/stderr.txt", 100).rule == "runtime"
    assert suffixes.decide("agent-container.log", 100).rule == "runtime"


def test_a_name_in_a_list_is_stripped_the_way_a_name_in_a_string_is():
    """The string branch stripped and the collection branches did not.

    `runtime_suffixes=" .log"` worked and `[" .log"]` silently matched
    nothing — one function answering the same question two ways.
    """
    policy = ArtifactSelectionPolicy(max_bytes=1024, runtime_suffixes=[" .log"])

    assert policy.runtime_suffixes == (".log",)
    # .rule, not .keep: .log is not in the default allowlist, so an unstripped
    # " .log" would fail to match here and the member would be dropped by
    # extension_not_allowed anyway.
    assert policy.decide("agent-container.log", 100).rule == "runtime"


@pytest.mark.parametrize(
    "value,expected_message",
    [
        ({".claude/": 1}, "its keys are unlikely"),
        (iter([".claude/"]), "re-readable collection"),
        (b".claude/", "decode it to str first"),
    ],
)
def test_scaffolding_refuses_what_every_other_setting_refuses(
    value, expected_message
):
    """It was the loosest field in a module built around being strict.

    A mapping and a one-shot iterator were accepted where every other setting
    raises, and bytes produced `AttributeError: 'int' object has no attribute
    'endswith'` rather than the corrective TypeError.
    """
    with pytest.raises(TypeError, match=expected_message):
        ArtifactSelectionPolicy(max_bytes=1024, scaffolding_prefixes=value)


def test_scaffolding_names_from_an_unordered_collection_are_sorted():
    """Asserted as an order, not as equality between two policies.

    Equality would pass either way: CPython iterates two equal small string
    sets alike, so the property would have held by coincidence rather than by
    construction — which is the failure mode this branch keeps catching in
    its own tests.

    The pair is chosen so the two candidate implementations disagree. The
    sort has to run on the *slash-terminated* form: "/" is 0x2F, above "."
    and every digit, so sorting the pre-slash names and appending afterwards
    yields ("logs/", "logs.old/"). An earlier version of this test used
    {"b", "a/"}, where both orders agree — the same coincidence its own
    docstring argues against.
    """
    policy = ArtifactSelectionPolicy(
        max_bytes=1024, scaffolding_prefixes={"logs", "logs.old"}
    )

    assert policy.scaffolding_prefixes == ("logs.old/", "logs/")


def test_a_bare_string_scaffolding_prefix_is_still_refused_not_comma_split():
    """The one place the shared rule is deliberately not applied.

    Scaffolding names are not a CLI knob, so there is no comma-separated
    spelling to honour, and iterating a string yields characters that match
    wrongly rather than not at all.
    """
    with pytest.raises(TypeError, match="not a single string"):
        ArtifactSelectionPolicy(max_bytes=1024, scaffolding_prefixes=".claude/")


def test_a_glob_is_stored_in_the_form_it_is_matched_in():
    """_matches_any lowercased per pattern per member; now __post_init__ does.

    Behaviourally identical, but `policy.include_globs` showed a form other
    than the one actually used.
    """
    policy = ArtifactSelectionPolicy(max_bytes=1024, include_globs="*.JSON")

    assert policy.include_globs == ("*.json",)
    assert policy.decide("results/table.json", 100).keep


@pytest.mark.parametrize(
    "field,setting,expected",
    [
        ("scaffolding_prefixes", {"a", "a/"}, ("a/",)),
        ("include_globs", "*.JSON,*.json", ("*.json",)),
        ("exclude_globs", ["a/*", "A/*"], ("a/*",)),
    ],
)
def test_normalization_does_not_leave_the_duplicates_it_creates(
    field, setting, expected
):
    """The caller wrote two names; normalization made them one name twice.

    Beyond the wasted match per member, it is the mirror of the equal-sets
    property this module establishes: {"a"} and {"a", "a/"} decide
    identically and compared unequal.
    """
    policy = ArtifactSelectionPolicy(max_bytes=1024, **{field: setting})

    assert getattr(policy, field) == expected


def test_two_policies_that_decide_identically_compare_equal():
    """The duplicate's visible consequence, stated as the property."""
    plain = ArtifactSelectionPolicy(max_bytes=1024, scaffolding_prefixes={"a"})
    redundant = ArtifactSelectionPolicy(
        max_bytes=1024, scaffolding_prefixes={"a", "a/"}
    )

    assert plain == redundant


def test_a_caller_supplied_list_order_survives_normalization():
    """Only an unordered collection is sorted; a list's order is the caller's."""
    policy = ArtifactSelectionPolicy(
        max_bytes=1024, scaffolding_prefixes=["z", "a"]
    )

    assert policy.scaffolding_prefixes == ("z/", "a/")


@pytest.mark.parametrize(
    "value,expected_message",
    [
        ({".claude/": 1}, "its keys are unlikely"),
        (iter([".claude/"]), "re-readable collection"),
        (b".claude/", "decode it to str first"),
    ],
)
def test_is_under_directory_refuses_what_the_shared_rule_refuses(
    value, expected_message
):
    """The public door behind _with_trailing_slashes, now documented as such.

    The one-shot case is the one that costs something: this function consumes
    its argument inside the call, so a generator would be safe here. It is
    refused so one rule holds at every door rather than a caller having to
    know which doors re-read — and the message now states the contract rather
    than a harm that does not apply here.
    """
    with pytest.raises(TypeError, match=expected_message):
        is_under_directory("a/.claude/x.md", value)


def test_is_under_directory_still_answers_for_an_ordinary_list():
    """The guard above narrows what is accepted, not what it decides."""
    assert is_under_directory("workspace/.claude/skills/x.md", [".claude/"])
    assert not is_under_directory("run/logs_summary.csv", ["logs"])
