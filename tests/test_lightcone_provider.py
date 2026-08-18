"""Tests for the Lightcone / ASTRA provider.

These focus on the pure, filesystem-level helpers (command building, project
resolution, manifest-driven output discovery, artifact harvesting, report
selection, provenance, citation extraction) so they run without invoking the
real ``lc`` CLI. A single integration test exercises an actual run and is
skipped automatically when the CLI is unavailable.
"""

import json
import shutil
from pathlib import Path

import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import LightconeParams
from deep_research_client.providers.lightcone import MANIFEST_FILENAME, LightconeProvider

FIXTURE_PROJECT = Path(__file__).parent / "input" / "astra_dismech"


def make_provider(params: LightconeParams | None = None) -> LightconeProvider:
    """Build a provider with a standard test config."""
    config = ProviderConfig(name="lightcone", api_key=None, enabled=True)
    return LightconeProvider(config, params)


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _materialize(out_dir: Path, files: dict[str, str], manifest: dict | None) -> None:
    """Create a materialized-output directory with files and a manifest sidecar."""
    for name, content in files.items():
        _write(out_dir / name, content)
    if manifest is not None:
        (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest))


# --- basics ----------------------------------------------------------------


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


# --- command building ------------------------------------------------------


@pytest.mark.parametrize(
    "params, expected",
    [
        (LightconeParams(), ["lc", "run"]),
        (LightconeParams(universe="baseline"), ["lc", "run", "--universe", "baseline"]),
        (LightconeParams(jobs=4, force=True), ["lc", "run", "--jobs", "4", "--force"]),
        (
            LightconeParams(universe="baseline", outputs=["accuracy", "precision"]),
            ["lc", "run", "--universe", "baseline", "accuracy", "precision"],
        ),
        (
            LightconeParams(materialize_args=["materialize"], extra_args=["--dry-run"]),
            ["lc", "materialize", "--dry-run"],
        ),
        (LightconeParams(lc_executable="/opt/lc"), ["/opt/lc", "run"]),
    ],
)
def test_build_command(params, expected):
    """The command reflects executable, subcommand, universe, flags, and outputs."""
    assert make_provider(params)._build_command() == expected


# --- project resolution ----------------------------------------------------


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


# --- manifest discovery ----------------------------------------------------


def test_find_manifest_dirs_parses_and_locates(tmp_path):
    """Directories with a manifest sidecar are found and their JSON parsed."""
    _materialize(
        tmp_path / "results" / "accuracy",
        {"accuracy.tsv": "a\tb\n"},
        {"output_id": "accuracy", "universe_id": "baseline"},
    )
    _write(tmp_path / "data" / "input.tsv", "not an output")  # no manifest -> ignored

    provider = make_provider()
    dirs = provider._find_manifest_dirs(tmp_path)
    assert set(dirs) == {tmp_path / "results" / "accuracy"}
    assert dirs[tmp_path / "results" / "accuracy"]["output_id"] == "accuracy"


def test_read_manifest_malformed_returns_none(tmp_path):
    """A malformed manifest is ignored (None) rather than raising."""
    bad = tmp_path / MANIFEST_FILENAME
    bad.write_text("{not valid json")
    assert LightconeProvider._read_manifest(bad) is None


def test_owned_files_excludes_non_materialized(tmp_path):
    """Only files under a manifest directory are considered materialized."""
    _materialize(
        tmp_path / "results" / "acc",
        {"acc.tsv": "a\tb\n", "fig.png": "img"},
        {"output_id": "acc"},
    )
    _write(tmp_path / "data" / "raw.tsv", "input")  # not owned

    provider = make_provider()
    manifest_dirs = provider._find_manifest_dirs(tmp_path)
    owned = provider._owned_files(tmp_path, set(manifest_dirs))
    names = sorted(p.name for p in owned)
    assert names == ["acc.tsv", "fig.png"]
    # The manifest sidecar itself is never an owned file.
    assert all(p.name != MANIFEST_FILENAME for p in owned)


# --- harvesting ------------------------------------------------------------


def test_harvest_artifacts_filters_and_tags(tmp_path):
    """Harvest keeps allowed outputs, drops noise/oversize, and tags provenance."""
    _materialize(
        tmp_path / "results" / "enrich",
        {
            "enrichment.tsv": "a\tb\n1\t2\n",
            "fig1.png": "imagedata",
            "run.log": "verbose log",  # noisy suffix -> dropped
            "big.csv": "x" * 100,  # oversize -> dropped
        },
        {"output_id": "mito_enrichment", "universe_id": "baseline"},
    )
    provider = make_provider(LightconeParams(artifact_max_bytes=50))
    manifest_dirs = provider._find_manifest_dirs(tmp_path)
    owned = provider._owned_files(tmp_path, set(manifest_dirs))

    artifacts = provider._harvest_artifacts(owned, tmp_path, manifest_dirs, skip=None)
    by_path = {a.path: a for a in artifacts}

    assert set(by_path) == {"results/enrich/enrichment.tsv", "results/enrich/fig1.png"}
    assert all(a.source == "lightcone_outputs" for a in artifacts)
    # Provenance from the manifest is woven into the description.
    assert "mito_enrichment" in by_path["results/enrich/enrichment.tsv"].description
    assert "baseline" in by_path["results/enrich/enrichment.tsv"].description


def test_harvest_skips_report_path(tmp_path):
    """The file chosen as the report is not also harvested as an artifact."""
    out = tmp_path / "results" / "call"
    _materialize(out, {"report.md": "# verdict", "data.tsv": "a\tb\n"}, {"output_id": "call"})
    provider = make_provider()
    manifest_dirs = provider._find_manifest_dirs(tmp_path)
    owned = provider._owned_files(tmp_path, set(manifest_dirs))

    artifacts = provider._harvest_artifacts(
        owned, tmp_path, manifest_dirs, skip=out / "report.md"
    )
    assert [a.path for a in artifacts] == ["results/call/data.tsv"]


def test_owned_files_empty_without_manifests(tmp_path):
    """No manifests anywhere means nothing is treated as a materialized output."""
    _write(tmp_path / "results" / "stray.tsv", "a\tb\n")
    provider = make_provider()
    assert provider._owned_files(tmp_path, set(provider._find_manifest_dirs(tmp_path))) == []


# --- report selection ------------------------------------------------------


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
    out = tmp_path / "results" / "r"
    _materialize(out, {name: f"# {name}" for name in files}, {"output_id": "r"})
    provider = make_provider()
    owned = provider._owned_files(tmp_path, set(provider._find_manifest_dirs(tmp_path)))

    name, markdown, path = provider._select_report(owned, tmp_path)
    assert name == f"results/r/{expected_name}"
    assert markdown == f"# {expected_name}"
    assert path == out / expected_name


def test_select_report_none_when_absent(tmp_path):
    """No markdown among materialized outputs yields (None, None, None)."""
    out = tmp_path / "results" / "r"
    _materialize(out, {"data.tsv": "a\tb\n"}, {"output_id": "r"})
    provider = make_provider()
    owned = provider._owned_files(tmp_path, set(provider._find_manifest_dirs(tmp_path)))
    assert provider._select_report(owned, tmp_path) == (None, None, None)


# --- provenance & synthesized report ---------------------------------------


def test_summarize_manifest_selects_provenance_fields():
    """Only the provenance-relevant manifest fields are surfaced."""
    manifest = {
        "schema_version": 1,
        "output_id": "accuracy",
        "universe_id": "baseline",
        "code_version": "sha256:abc",
        "decisions": {"scaling": "standard"},
        "recipe": "python scripts/eval.py",
        "host": "saul01",  # dropped: not provenance-relevant
        "slurm_job_id": "123",  # dropped
    }
    summary = LightconeProvider._summarize_manifest(manifest)
    assert summary["output_id"] == "accuracy"
    assert summary["decisions"] == {"scaling": "standard"}
    assert "host" not in summary
    assert "slurm_job_id" not in summary


def test_synthesize_report_uses_spec_name_and_provenance(tmp_path):
    """The fallback report reads the spec name and lists outputs with recipes."""
    provider = make_provider(LightconeParams(universe="baseline"))
    provenance = [
        {"output_id": "mito_enrichment", "universe_id": "baseline", "recipe": "python enrich.py"}
    ]
    report = provider._synthesize_report(FIXTURE_PROJECT, [], provenance, "line1\nline2")
    assert "Mitochondrial dysfunction mediates neuronal loss in Disease-X" in report
    assert "baseline" in report
    assert "mito_enrichment" in report
    assert "python enrich.py" in report
    assert "line2" in report


def test_read_spec_name():
    """The top-level name is read from astra.yaml without a YAML dependency."""
    name = LightconeProvider._read_spec_name(FIXTURE_PROJECT)
    assert name == "Mitochondrial dysfunction mediates neuronal loss in Disease-X"


# --- citations -------------------------------------------------------------


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


# --- integration -----------------------------------------------------------


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
