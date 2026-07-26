"""Tests for the Lightcone / ASTRA provider.

These focus on the pure, filesystem-level helpers (command building, project
resolution, artifact harvesting, report selection, citation extraction) so they
run without invoking the real ``lc`` CLI. A single integration test exercises
an actual run and is skipped automatically when the CLI is unavailable.
"""

import shutil
from pathlib import Path

import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import LightconeParams
from deep_research_client.providers.lightcone import LightconeProvider

FIXTURE_PROJECT = Path(__file__).parent / "input" / "astra_dismech"


def make_provider(params: LightconeParams | None = None) -> LightconeProvider:
    """Build a provider with a standard test config."""
    config = ProviderConfig(name="lightcone", api_key=None, enabled=True)
    return LightconeProvider(config, params)


def test_default_model():
    """The provider advertises its sentinel default model."""
    provider = make_provider()
    assert provider.get_default_model() == "lightcone-astra"
    assert provider.model == "lightcone-astra"


def test_is_available_false_when_executable_missing():
    """An unresolvable executable name means the provider is unavailable."""
    provider = make_provider(LightconeParams(lc_executable="definitely-not-a-real-lc-xyz"))
    assert provider.is_available() is False


def test_is_available_false_when_disabled():
    """A disabled config is never available, even if the CLI exists."""
    config = ProviderConfig(name="lightcone", api_key=None, enabled=False)
    provider = LightconeProvider(config)
    assert provider.is_available() is False


@pytest.mark.parametrize(
    "params, expected",
    [
        (LightconeParams(), ["lc", "run"]),
        (LightconeParams(universe="baseline"), ["lc", "run", "--universe", "baseline"]),
        (
            LightconeParams(materialize_args=["materialize"], extra_args=["--dry-run"]),
            ["lc", "materialize", "--dry-run"],
        ),
        (LightconeParams(lc_executable="/opt/lc"), ["/opt/lc", "run"]),
    ],
)
def test_build_command(params, expected):
    """The command reflects executable, subcommand, universe, and extra args."""
    assert make_provider(params)._build_command() == expected


def test_resolve_project_from_directory():
    """A directory containing astra.yaml resolves to itself."""
    provider = make_provider()
    assert provider._resolve_project(str(FIXTURE_PROJECT)) == FIXTURE_PROJECT


def test_resolve_project_from_spec_file():
    """An astra.yaml path resolves to its parent project directory."""
    provider = make_provider()
    spec = FIXTURE_PROJECT / "astra.yaml"
    assert provider._resolve_project(str(spec)) == FIXTURE_PROJECT


def test_resolve_project_missing_path_raises():
    """A non-existent path fails fast."""
    provider = make_provider()
    with pytest.raises(ValueError, match="not found"):
        provider._resolve_project("/no/such/astra/project")


def test_resolve_project_dir_without_spec_raises(tmp_path):
    """A directory without astra.yaml is not an ASTRA project."""
    provider = make_provider()
    with pytest.raises(ValueError, match="No astra.yaml"):
        provider._resolve_project(str(tmp_path))


def test_resolve_project_working_dir_overrides_query():
    """An explicit working_dir param takes precedence over the query."""
    provider = make_provider(LightconeParams(working_dir=str(FIXTURE_PROJECT)))
    assert provider._resolve_project("ignored") == FIXTURE_PROJECT


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_harvest_artifacts_filters(tmp_path):
    """Harvesting keeps allowed outputs and drops noise, logs, and oversize files."""
    out = tmp_path / "outputs"
    _write(out / "figures" / "fig1.png", "imagedata")
    _write(out / "tables" / "enrichment.tsv", "a\tb\n1\t2\n")
    _write(out / "run.log", "verbose log")
    _write(out / ".cache" / "junk.json", "{}")
    _write(out / "notes.txt", "keep me")
    _write(out / "big.csv", "x" * 100)

    provider = make_provider(LightconeParams(artifact_max_bytes=50))
    artifacts = provider._harvest_artifacts(out, skip_names=set())
    names = sorted(a.path for a in artifacts)

    assert names == ["figures/fig1.png", "notes.txt", "tables/enrichment.tsv"]
    # Sources and descriptions are populated for provenance.
    assert all(a.source == "lightcone_outputs" for a in artifacts)
    assert all(a.description for a in artifacts)


def test_harvest_artifacts_skips_named_report(tmp_path):
    """The already-captured report file is not re-harvested as an artifact."""
    out = tmp_path / "outputs"
    _write(out / "final_report.md", "# report")
    _write(out / "data.tsv", "a\tb\n")

    provider = make_provider()
    artifacts = provider._harvest_artifacts(out, skip_names={"final_report.md"})
    assert [a.path for a in artifacts] == ["data.tsv"]


def test_harvest_artifacts_missing_dir_returns_empty(tmp_path):
    """A missing output directory yields no artifacts rather than raising."""
    provider = make_provider()
    assert provider._harvest_artifacts(tmp_path / "nope", skip_names=set()) == []


@pytest.mark.parametrize(
    "files, expected_name",
    [
        (["report.md", "final_report.md", "notes.md"], "final_report.md"),
        (["notes.md", "de_report.md"], "de_report.md"),
        (["a.md", "b.md"], "a.md"),
    ],
)
def test_select_report_prefers_canonical(tmp_path, files, expected_name):
    """Report selection prefers canonical names, then report*, then any md."""
    out = tmp_path / "outputs"
    for name in files:
        _write(out / name, f"# {name}")
    provider = make_provider()
    name, markdown = provider._select_report(out)
    assert name == expected_name
    assert markdown == f"# {expected_name}"


def test_select_report_none_when_absent(tmp_path):
    """No markdown in the output tree yields (None, None)."""
    out = tmp_path / "outputs"
    _write(out / "data.tsv", "a\tb\n")
    provider = make_provider()
    assert make_provider()._select_report(out) == (None, None)
    assert provider._select_report(tmp_path / "missing") == (None, None)


def test_synthesize_report_uses_spec_name_and_lists_artifacts(tmp_path):
    """The fallback report reads the spec name and enumerates outputs."""
    provider = make_provider(LightconeParams(universe="baseline"))
    # Build one real artifact by harvesting a temp output.
    out = tmp_path / "outputs"
    _write(out / "tables" / "enrichment.tsv", "a\tb\n")
    artifacts = provider._harvest_artifacts(out, skip_names=set())

    report = provider._synthesize_report(FIXTURE_PROJECT, artifacts, "line1\nline2")
    assert "Mitochondrial dysfunction mediates neuronal loss in Disease-X" in report
    assert "baseline" in report
    assert "tables/enrichment.tsv" in report
    assert "line2" in report


def test_read_spec_name():
    """The top-level name is read from astra.yaml without a YAML dependency."""
    name = LightconeProvider._read_spec_name(FIXTURE_PROJECT)
    assert name == "Mitochondrial dysfunction mediates neuronal loss in Disease-X"


@pytest.mark.parametrize(
    "markdown, expected",
    [
        ("PMID: 12345678 and PMID:12345678", ["PMID:12345678"]),
        ("see https://example.org/a and https://example.org/b", ["https://example.org/a", "https://example.org/b"]),
        ("PMID: 12345678 at https://x.org/y", ["PMID:12345678", "https://x.org/y"]),
        ("no citations here", []),
    ],
)
def test_extract_citations(markdown, expected):
    """Citations extract PMIDs then URLs, de-duplicated, order-preserved."""
    assert LightconeProvider._extract_citations(markdown) == expected


@pytest.mark.integration
@pytest.mark.asyncio
async def test_research_against_fixture_project():
    """End-to-end run against the sample project; skipped when `lc` is absent."""
    if shutil.which("lc") is None:
        pytest.skip("lc CLI not installed")

    provider = make_provider()
    result = await provider.research(str(FIXTURE_PROJECT))
    assert result.provider == "lightcone"
    assert result.markdown
