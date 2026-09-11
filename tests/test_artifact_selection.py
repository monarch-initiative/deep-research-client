"""Tests for the configurable artifact selection policy.

The defaults must keep reproducing the previously hardcoded curated set, and
each knob must widen or narrow exactly the layer it names.
"""

import io
import zipfile

import pytest

from deep_research_client.artifact_selection import (
    DEFAULT_ALLOWED_EXTENSIONS,
    DEFAULT_MAX_BYTES,
    DEFAULT_SCAFFOLDING_DIRECTORIES,
    DEFAULT_SCAFFOLDING_PREFIXES,
    RUNTIME_EXTENSIONS,
    ArtifactSelectionPolicy,
    _with_trailing_slashes,
    is_under_directory,
    is_under_normalized_directory,
    normalize_member_path,
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
