"""Tests for CLI behaviors."""

import base64
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from deep_research_client.model_cards import (
    ProviderArchetype,
    ResearchCapability,
    ResearchResource,
)
from typer.testing import CliRunner

from deep_research_client.cli import (
    app,
    _collect_noop_research_option_warnings,
    _effective_research_options,
    _write_result_artifacts,
)
from deep_research_client.models import CacheConfig, ResearchArtifact, ResearchResult


runner = CliRunner()


def _load_asta_api_key() -> str:
    """Load the Asta API key from env or the repo-local helper file."""
    key = Path(__file__).resolve().parents[1] / "asta_key"
    if key.exists():
        return key.read_text(encoding="utf-8").strip()
    pytest.skip("asta_key file not present")


def test_research_uses_input_file(tmp_path):
    """CLI should read the query directly from a file when requested."""
    query_file = tmp_path / "query.md"
    query_file.write_text("What is synthetic biology?", encoding="utf-8")

    result = runner.invoke(
        app,
        ["research", "--input-file", str(query_file), "--provider", "mock"],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )

    assert result.exit_code == 0, result.output
    assert "What is synthetic biology?" in result.stdout


def test_research_rejects_conflicting_query_sources(tmp_path):
    """Passing both inline query and file should fail fast with a clear error."""
    query_file = tmp_path / "conflict.md"
    query_file.write_text("File-based query", encoding="utf-8")

    result = runner.invoke(
        app,
        ["research", "Inline query", "--input-file",
            str(query_file), "--provider", "mock"],
        env={"ENABLE_MOCK_PROVIDER": "true"},
    )

    assert result.exit_code == 1
    combined_output = result.output
    assert "Provide the query either as an argument or via --input-file" in combined_output


def test_providers_asta_shows_required_credentials():
    """CLI should explain that Asta needs its retrieval credential."""
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(
            app,
            ["providers", "--provider", "asta"],
            env={},
        )

    assert result.exit_code == 0, result.output
    assert "Provider: asta - Not available" in result.stdout
    assert "ASTA_API_KEY" in result.stdout


def test_providers_openscientist_shows_required_credentials():
    """CLI should explain that OpenScientist needs its API key."""
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(
            app,
            ["providers", "--provider", "openscientist"],
            env={},
        )

    assert result.exit_code == 0, result.output
    assert "Provider: openscientist - Not available" in result.stdout
    assert "OPENSCIENTIST_API_KEY" in result.stdout


def test_providers_no_available_providers_mentions_openscientist():
    """CLI should mention OpenScientist in missing-credential guidance."""
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(
            app,
            ["providers"],
            env={},
        )

    assert result.exit_code == 0, result.output
    assert "OPENSCIENTIST_API_KEY for OpenScientist" in result.stdout


def test_research_help_lists_openscientist_provider():
    """Research command help should list OpenScientist as a supported provider."""
    result = runner.invoke(
        app,
        ["research", "--help"],
        env={},
    )

    assert result.exit_code == 0, result.output
    assert "openscientist" in result.stdout


def test_edison_trajectory_requires_api_key():
    """Trajectory retrieval should fail fast when Edison credentials are missing."""
    with patch.dict(os.environ, {}, clear=True):
        result = runner.invoke(
            app,
            ["edison-trajectory", "784d73d5-da42-402e-9701-6c5b44beab14"],
        )

    assert result.exit_code == 1
    assert "EDISON_API_KEY is required" in result.output


def test_write_result_artifacts_sets_relative_paths(tmp_path):
    """CLI artifact writer should materialize sidecar files for markdown links."""
    result = ResearchResult(
        markdown="Report",
        provider="falcon",
        query="query",
        artifacts=[
            ResearchArtifact(
                filename="figure.png",
                content_base64="ZmFrZS1pbWFnZQ==",
                media_type="image/png",
            )
        ],
    )
    output_path = tmp_path / "report.md"

    _write_result_artifacts(result, output_path)

    artifact_path = tmp_path / "report_artifacts" / "figure.png"
    assert artifact_path.read_bytes() == b"fake-image"
    assert result.artifacts[0].path == "report_artifacts/figure.png"


def test_write_result_artifacts_handles_filename_collisions(tmp_path):
    """Artifacts with duplicate filenames should be preserved with unique paths."""
    result = ResearchResult(
        markdown="Report",
        provider="falcon",
        query="query",
        artifacts=[
            ResearchArtifact(filename="figure.png", content_base64="Zmlyc3Q="),
            ResearchArtifact(filename="figure.png", content_base64="c2Vjb25k"),
        ],
    )
    output_path = tmp_path / "report.md"

    _write_result_artifacts(result, output_path)

    artifact_dir = tmp_path / "report_artifacts"
    assert (artifact_dir / "figure.png").read_bytes() == b"first"
    assert (artifact_dir / "figure-2.png").read_bytes() == b"second"
    assert [artifact.path for artifact in result.artifacts] == [
        "report_artifacts/figure.png",
        "report_artifacts/figure-2.png",
    ]


def test_write_result_artifacts_sanitizes_path_traversal_names(tmp_path):
    """Artifact filenames should not escape the sidecar artifact directory."""
    result = ResearchResult(
        markdown="Report",
        provider="falcon",
        query="query",
        artifacts=[
            ResearchArtifact(
                filename="../../../etc/passwd",
                content_base64="c2FmZQ==",
            )
        ],
    )
    output_path = tmp_path / "report.md"

    _write_result_artifacts(result, output_path)

    artifact_path = tmp_path / result.artifacts[0].path
    assert result.artifacts[0].path.startswith("report_artifacts/")
    assert ".." not in result.artifacts[0].path
    assert artifact_path.read_bytes() == b"safe"
    assert not (tmp_path / "etc").exists()


def test_write_result_artifacts_sanitizes_windows_path_traversal_names(tmp_path):
    """Windows-style separators should not escape the sidecar artifact directory."""
    result = ResearchResult(
        markdown="Report",
        provider="falcon",
        query="query",
        artifacts=[
            ResearchArtifact(
                filename=r"..\..\evil.png",
                content_base64="c2FmZQ==",
            )
        ],
    )

    _write_result_artifacts(result, tmp_path / "report.md")

    artifact_path = tmp_path / result.artifacts[0].path
    assert result.artifacts[0].path == "report_artifacts/_evil.png"
    assert artifact_path.read_bytes() == b"safe"
    assert not (tmp_path / "evil.png").exists()


def test_write_result_artifacts_handles_unicode_filenames(tmp_path):
    """Unicode artifact names should be written and linked correctly."""
    result = ResearchResult(
        markdown="Report",
        provider="falcon",
        query="query",
        artifacts=[
            ResearchArtifact(
                filename="figuré-α.png",
                content_base64="aW1hZ2U=",
            )
        ],
    )
    output_path = tmp_path / "report.md"

    _write_result_artifacts(result, output_path)

    assert result.artifacts[0].path == "report_artifacts/figuré-α.png"
    assert (tmp_path / "report_artifacts" / "figuré-α.png").read_bytes() == b"image"


def test_write_result_artifacts_handles_large_artifact(tmp_path):
    """Large artifacts should be decoded and written without truncation."""
    payload = b"x" * (1024 * 1024)
    result = ResearchResult(
        markdown="Report",
        provider="falcon",
        query="query",
        artifacts=[
            ResearchArtifact(
                filename="large.bin",
                content_base64=base64.b64encode(payload).decode("ascii"),
            )
        ],
    )

    _write_result_artifacts(result, tmp_path / "report.md")

    assert (tmp_path / "report_artifacts" / "large.bin").read_bytes() == payload


def test_write_result_artifacts_rejects_invalid_base64(tmp_path):
    """Corrupted artifact payloads should fail with a clear error."""
    result = ResearchResult(
        markdown="Report",
        provider="falcon",
        query="query",
        artifacts=[
            ResearchArtifact(filename="broken.png", content_base64="not valid base64!")
        ],
    )

    with pytest.raises(ValueError, match="Invalid base64 content for artifact broken.png"):
        _write_result_artifacts(result, tmp_path / "report.md")


def test_write_result_artifacts_surfaces_directory_creation_failure(tmp_path):
    """Filesystem errors while creating artifact directories should not be swallowed."""
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")
    result = ResearchResult(
        markdown="Report",
        provider="falcon",
        query="query",
        artifacts=[ResearchArtifact(filename="figure.png", content_base64="ZGF0YQ==")],
    )

    with pytest.raises(OSError):
        _write_result_artifacts(result, blocked_parent / "report.md")


def test_collect_noop_research_option_warnings_for_asta():
    """Asta should warn on CLI options that do not affect retrieval."""
    warnings = _collect_noop_research_option_warnings(
        "asta",
        model="custom-model",
        base_url="https://example.org",
        use_cborg=True,
        api_key_env="CUSTOM_KEY",
    )

    assert len(warnings) == 4
    assert any("--model" in warning for warning in warnings)
    assert any("--base-url" in warning for warning in warnings)
    assert any("--use-cborg" in warning for warning in warnings)
    assert any("--api-key-env" in warning for warning in warnings)


def test_effective_research_options_discards_asta_noops_when_provider_is_explicit():
    """Explicit Asta provider selection should discard irrelevant CLI options."""
    options = _effective_research_options(
        provider="asta",
        model="custom-model",
        base_url="https://example.org",
        use_cborg=True,
        api_key_env="CUSTOM_KEY",
        cache_config=CacheConfig(enabled=False),
    )

    assert options.provider_hint == "asta"
    assert options.model is None
    assert options.base_url is None
    assert options.use_cborg is False
    assert options.api_key_env is None
    assert len(options.warnings) == 4


def test_effective_research_options_discards_asta_noops_when_asta_is_auto_selected():
    """If Asta would be auto-selected, no-op CLI options should still be pruned."""
    with patch.dict(
        os.environ,
        {
            "ASTA_API_KEY": "asta-key",
            "OPENAI_API_KEY": "",
            "EDISON_API_KEY": "",
            "PERPLEXITY_API_KEY": "",
            "CONSENSUS_API_KEY": "",
        },
        clear=True,
    ):
        options = _effective_research_options(
            provider=None,
            model="custom-model",
            base_url="https://example.org",
            use_cborg=True,
            api_key_env="CUSTOM_KEY",
            cache_config=CacheConfig(enabled=False),
        )

    assert options.provider_hint == "asta"
    assert options.model is None
    assert options.base_url is None
    assert options.use_cborg is False
    assert options.api_key_env is None
    assert len(options.warnings) == 4


@pytest.mark.integration
def test_research_asta_warns_on_noop_model_and_writes_separate_citations(tmp_path):
    """Asta CLI should warn on --model but still honor output formatting options."""
    output_path = tmp_path / "asta-report.md"
    citations_path = tmp_path / "asta-citations.md"

    result = runner.invoke(
        app,
        [
            "research",
            "hyperlipidemia",
            "--provider",
            "asta",
            "--model",
            "ignored-model",
            "--base-url",
            "https://example.org",
            "--use-cborg",
            "--api-key-env",
            "IGNORED_KEY",
            "--output",
            str(output_path),
            "--separate-citations",
            str(citations_path),
        ],
        env={"ASTA_API_KEY": _load_asta_api_key(), "IGNORED_KEY": ""},
    )

    assert result.exit_code == 0, result.output
    combined_output = result.output
    assert "ignores --model" in combined_output
    assert "ignores --base-url" in combined_output
    assert "ignores --use-cborg" in combined_output
    assert "ignores --api-key-env" in combined_output
    assert output_path.exists()
    assert citations_path.exists()
    assert "## Citations" not in output_path.read_text(encoding="utf-8")
    assert "# Citations for Research Query" in citations_path.read_text(
        encoding="utf-8")


# ---------------------------------------------------------------------------
# `models` vocabulary filters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,value,expected_provider,absent_provider",
    [
        # --cost included deliberately: it shares the parse/render path with the
        # three vocabulary axes, and a row pairing the wrong enum with the wrong
        # finder type-checks and prints "No models match" rather than crashing.
        ("--cost", "low", "ASTA", "BIOMNI"),
        ("--capability", "code_interpretation", "BIOMNI", "ASTA"),
        ("--capability", "retrieval_only", "ASTA", "BIOMNI"),
        ("--resource", "pubmed", "OPENSCIENTIST", "ASTA"),
        ("--archetype", "co_scientist", "BIOMNI", "ASTA"),
        ("--archetype", "retriever", "ASTA", "BIOMNI"),
    ],
)
def test_models_filters_by_each_vocabulary_axis(
    flag, value, expected_provider, absent_provider
):
    """All three axes are queryable, not just displayable.

    The negative matters as much as the positive: without it, a filter that
    fell through to the unfiltered listing would pass every case.
    """
    result = runner.invoke(app, ["models", flag, value])

    assert result.exit_code == 0, result.output
    assert expected_provider in result.output
    assert absent_provider not in result.output, "the filter did not narrow anything"


def test_models_intersects_filters_rather_than_honouring_only_the_first():
    """Two filters ask a conjunction; answering one of them silently is wrong."""
    both = runner.invoke(
        app, ["models", "--archetype", "co_scientist", "--resource", "pubmed"]
    )
    archetype_only = runner.invoke(app, ["models", "--archetype", "co_scientist"])

    assert both.exit_code == 0, both.output
    assert "BIOMNI" in both.output and "BIOMNI" in archetype_only.output
    # A conjunction with no overlap must say so rather than list one side.
    empty = runner.invoke(
        app, ["models", "--archetype", "retriever", "--resource", "pubmed"]
    )
    assert empty.exit_code == 0, empty.output
    assert "No models match" in empty.output
    assert "ASTA" not in empty.output, "the retriever half was answered alone"


def test_a_filter_matching_nothing_says_so_rather_than_printing_nothing():
    """Default verbosity is WARNING, so a logged-only message was invisible.

    Uses a conjunction rather than a term that happens to be unannotated: the
    day a provider claims `arxiv`, this test should not be the one that fails.
    """
    result = runner.invoke(app, ["models", "--archetype", "retriever", "--cost", "very_high"])

    assert result.exit_code == 0, result.output
    assert "No models match" in result.output


def test_provider_combined_with_a_filter_narrows_instead_of_winning():
    """--provider was the last flag that silently discarded the others."""
    matching = runner.invoke(
        app, ["models", "--provider", "biomni", "--capability", "code_interpretation"]
    )
    assert matching.exit_code == 0, matching.output
    assert "BIOMNI" in matching.output

    # biomni has no web_search capability, so the conjunction is empty. Before
    # this, --provider won and the capability was dropped without a word.
    empty = runner.invoke(
        app, ["models", "--provider", "biomni", "--capability", "web_search"]
    )
    assert empty.exit_code == 0, empty.output
    assert "No models match" in empty.output


def test_provider_named_alone_still_reports_its_default_model():
    """Narrowing must not cost the listing form its Default: header."""
    result = runner.invoke(app, ["models", "--provider", "biomni"])

    assert result.exit_code == 0, result.output
    assert "Default: biomni-a1" in result.output


def test_research_provider_help_names_every_registered_provider():
    """A provider added to the registry must not need a second list edited.

    biomni and cyberian were both absent from the hand-written help this
    replaced.
    """
    from deep_research_client.client import PROVIDER_CLASS_PATHS

    result = runner.invoke(app, ["research", "--help"])

    assert result.exit_code == 0, result.output
    rendered = " ".join(result.output.split())
    for name in PROVIDER_CLASS_PATHS:
        assert name in rendered, f"{name} is registered but absent from --provider help"


@pytest.mark.parametrize(
    "flag,enum_class",
    [
        ("--capability", ResearchCapability),
        ("--resource", ResearchResource),
        ("--archetype", ProviderArchetype),
    ],
    ids=lambda v: v if isinstance(v, str) else v.__name__,
)
def test_models_rejects_an_unknown_vocabulary_value_by_naming_the_whole_vocabulary(
    flag, enum_class
):
    """The error lists every permissible value, derived from the enum.

    Asserting the terms are present, not merely that "etc." is absent: the
    latter would also pass if the message vanished entirely.
    """
    result = runner.invoke(app, ["models", flag, "not-a-real-term"])

    assert result.exit_code == 1
    for member in enum_class:
        assert member.value in result.output, (
            f"{member.value} is a permissible value but the error does not list it"
        )
    assert "etc." not in result.output, "the truncated list this replaced"
