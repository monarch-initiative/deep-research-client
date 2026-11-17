"""The `claims ground` command: option validation and refusals before any call.

The OAK + model path is covered under the ``llm`` marker in
test_grounding_integration.py; here only the paths that need neither run.
"""

import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app

CLAIMS = str(Path(__file__).parent / "input" / "claims" / "alignment" / "report_a.claims.json")
runner = CliRunner()


def test_neither_ontology_option_is_refused() -> None:
    result = runner.invoke(app, ["claims", "ground", CLAIMS])
    assert result.exit_code == 1
    assert "exactly one of --ontology-config or --ontologies" in result.stderr


def test_both_ontology_options_are_refused(tmp_path) -> None:
    cfg = tmp_path / "oak_config.yaml"
    cfg.write_text("ontology_adapters:\n  HP:\n", encoding="utf-8")
    result = runner.invoke(
        app, ["claims", "ground", CLAIMS, "--ontologies", "biomedical", "--ontology-config", str(cfg)]
    )
    assert result.exit_code == 1
    assert "exactly one of --ontology-config or --ontologies" in result.stderr


def test_an_unknown_built_in_set_is_refused() -> None:
    result = runner.invoke(app, ["claims", "ground", CLAIMS, "--ontologies", "zoology"])
    assert result.exit_code == 1
    assert "Unknown --ontologies 'zoology'" in result.stderr


def test_a_missing_config_file_is_refused(tmp_path) -> None:
    result = runner.invoke(
        app, ["claims", "ground", CLAIMS, "--ontology-config", str(tmp_path / "nope.yaml")]
    )
    assert result.exit_code == 1
    assert "Ontology config not found" in result.stderr


def test_a_bad_backend_is_refused_before_any_search(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(
        app, ["claims", "ground", CLAIMS, "--ontologies", "biomedical", "--llm-backend", "bogus"]
    )
    assert result.exit_code == 1
    assert "Unknown --llm-backend 'bogus'" in result.stderr


@pytest.mark.skipif(
    importlib.util.find_spec("linkml_term_validator") is not None,
    reason="runs only when the optional `terms` extra is absent (the base-install CI job)",
)
def test_the_cli_names_the_terms_extra_when_it_is_missing() -> None:
    """With linkml-term-validator not installed, `claims ground` says how to install it."""
    result = runner.invoke(
        app, ["claims", "ground", CLAIMS, "--ontologies", "biomedical",
              "--llm-backend", "openai", "--llm-base-url", "http://localhost:9"],
    )
    assert result.exit_code == 1
    assert "deep_research_client[terms]" in result.stderr
