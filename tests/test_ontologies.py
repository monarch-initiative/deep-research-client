"""Tests for opt-in ontology terms: the config, the prompt section, and scoped validation.

The validator runs offline here, so a configured term comes back unverifiable
rather than resolved; what is under test is which terms are looked up at all,
and how the out-of-scope ones are reported.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.client import DeepResearchClient
from deep_research_client.models import CacheConfig
from deep_research_client.ontologies import DEFAULT_ONTOLOGY_ADAPTER, OntologyConfig
from deep_research_client.validation.term_datamodel import TermCheck, TermStatus
from deep_research_client.validation.term_models import TermValidationReport
from deep_research_client.validation.term_validator import (
    TermValidator,
    term_validator_is_available,
)

needs_terms_extra = pytest.mark.skipif(
    not term_validator_is_available(), reason="needs the terms extra"
)

#: Shaped like a real project oak_config.yaml: comments, an extra key, and
#: every kind of entry an ontology config can hold.
OAK_STYLE_CONFIG = """\
# Which ontologies reports may cite, and how each is looked up.
ontology_adapters:
  HP: ols:hp                  # online
  MONDO: sqlite:obo:mondo     # local build
  CHEBI:                      # default adapter
  NCIT: ""                    # named in the prompt, not checked
other_setting: kept but unused
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """An ontology config on disk."""
    path = tmp_path / "ontologies.yaml"
    path.write_text(OAK_STYLE_CONFIG, encoding="utf-8")
    return path


def test_an_oak_config_loads_as_an_ontology_config(config_file):
    """An existing oak_config.yaml is usable as it is, other keys included."""
    config = OntologyConfig.from_file(config_file)

    assert config.prefixes == ["HP", "MONDO", "CHEBI", "NCIT"]
    assert config.ontology_adapters["CHEBI"] is None
    assert config.ontology_adapters["NCIT"] == ""


@pytest.mark.parametrize(
    "default,expected_chebi",
    [(None, DEFAULT_ONTOLOGY_ADAPTER), ("sqlite:obo:", "sqlite:obo:")],
)
def test_an_entry_with_no_value_takes_the_default_adapter_and_an_empty_string_stays_a_skip(
    config_file, default, expected_chebi
):
    """OAK itself would skip CHEBI here; an ontology config gives it the default instead."""
    config = OntologyConfig.from_file(config_file)
    resolved = config.resolved_adapters(default) if default else config.resolved_adapters()

    assert resolved == {"HP": "ols:hp", "MONDO": "sqlite:obo:mondo", "CHEBI": expected_chebi, "NCIT": ""}


@pytest.mark.parametrize(
    "text",
    ["- HP\n- CHEBI\n", "prefixes: [HP, CHEBI]\n", "ontology_adapters: {}\n"],
)
def test_a_file_that_is_not_an_ontology_config_is_rejected(tmp_path, text):
    """A bare list or a misspelled key would otherwise configure nothing, silently."""
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        OntologyConfig.from_file(path)


def test_the_prompt_section_names_every_configured_ontology_and_no_example_term(config_file):
    """A concrete example term would name an ontology that may not be configured."""
    section = OntologyConfig.from_file(config_file).prompt_section()

    assert section.startswith("## Ontology terms")
    assert "HP, MONDO, CHEBI, NCIT" in section
    assert "`PREFIX:ID (term label)`" in section
    assert "Use no other ontologies" in section


def test_the_client_appends_the_section_to_the_query(config_file, tmp_path, monkeypatch):
    """Every provider sees it, and the recorded question is what was actually asked."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))
    config = OntologyConfig.from_file(config_file)

    result = client.research("What causes scurvy?", provider="mock", ontologies=config)

    assert result.query == f"What causes scurvy?\n\n{config.prompt_section()}"


def test_without_a_config_the_query_is_left_alone(tmp_path, monkeypatch):
    """Ontology terms are opt-in: no config, no section."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))

    assert client.research("What causes scurvy?", provider="mock").query == "What causes scurvy?"


@needs_terms_extra
def test_terms_outside_the_configured_ontologies_are_not_looked_up(config_file, tmp_path):
    """GO is not configured: reported as such, and not counted as unverifiable or invented."""
    validator = TermValidator(
        ontologies=OntologyConfig.from_file(config_file), offline=True, cache_dir=tmp_path
    )

    report = validator.validate_markdown(
        "Seizures HP:0001250 (Seizure) and autophagy GO:0006914 (autophagy) are discussed."
    )

    statuses = {check.term_id: check.status for check in report.checked_terms}
    assert statuses == {"HP:0001250": TermStatus.UNVERIFIABLE, "GO:0006914": TermStatus.NOT_CONFIGURED}
    assert report.not_configured_count == 1
    assert report.not_found_count == 0
    assert "GO" not in (report.unresolvable_prefixes or [])


@needs_terms_extra
def test_an_ontology_configured_with_an_empty_string_is_named_but_never_reported_invented(config_file, tmp_path):
    """NCIT is "" in the config: unverifiable, not NOT_FOUND. A real run reported 11 HGNC IDs as invented."""
    validator = TermValidator(ontologies=OntologyConfig.from_file(config_file), cache_dir=tmp_path)

    report = validator.validate_markdown("Echocardiography NCIT:C80404 (Echocardiography) was used.")

    assert [check.status for check in report.checked_terms] == [TermStatus.UNVERIFIABLE]
    assert report.not_found_count == 0


@needs_terms_extra
def test_the_validator_resolves_through_the_configs_adapters(config_file, tmp_path):
    """The config replaces oak_config, with empty entries already given the default."""
    validator = TermValidator(
        adapter="sqlite:obo:", ontologies=OntologyConfig.from_file(config_file),
        oak_config=tmp_path / "ignored.yaml", cache_dir=tmp_path,
    )

    access = validator._build_ontology_access()

    assert access.oak_config == {
        "HP": "ols:hp", "MONDO": "sqlite:obo:mondo", "CHEBI": "sqlite:obo:", "NCIT": "",
    }


def test_a_report_with_off_list_terms_needs_review_but_has_no_problems():
    """Reaching outside the set is worth a look, but it is not a fabrication and fails no build."""
    report = TermValidationReport(terms=[
        TermCheck(term_id="GO:0006914", prefix="GO", status=TermStatus.NOT_CONFIGURED,
                  occurrences=2, reported_labels=["autophagy"]),
        TermCheck(term_id="HP:0001250", prefix="HP", status=TermStatus.VERIFIED),
    ])

    summary = report.summary()
    markdown = report.to_markdown()

    assert report.has_problems is False
    assert summary["not_configured"] == 1
    assert summary["not_configured_terms"] == ["GO:0006914"]
    assert summary["needs_review"] is True
    assert "| Not in the configured ontologies | 1 |" in markdown
    assert '`GO:0006914` (2 mentions), reported as "autophagy"' in markdown


@needs_terms_extra
def test_cli_ontology_config_asks_for_and_checks_terms(config_file, tmp_path, monkeypatch):
    """One option: the section goes into the query, and term validation runs without --validate-terms."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")

    result = CliRunner().invoke(app, [
        "research", "Autophagy GO:0006914 (autophagy) and seizures HP:0001250 (Seizure)",
        "--provider", "mock", "--no-cache", "--no-validate-references",
        "--ontology-config", str(config_file), "--term-offline", "--term-cache-dir", str(tmp_path),
    ])

    assert result.exit_code == 0, result.output
    assert "## Ontology terms" in result.output
    assert "## Term Validation" in result.output
    assert "| Not in the configured ontologies | 1 |" in result.output


def test_cli_rejects_a_file_that_is_not_an_ontology_config(tmp_path, monkeypatch):
    """Before the provider runs, so a typo costs nothing."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    bad = tmp_path / "bad.yaml"
    bad.write_text("prefixes: [HP]\n", encoding="utf-8")

    result = CliRunner().invoke(app, [
        "research", "scurvy", "--provider", "mock", "--no-cache", "--ontology-config", str(bad),
    ])

    assert result.exit_code == 1
    assert "ontology_adapters" in result.output


@needs_terms_extra
def test_validate_terms_takes_the_same_config_for_a_saved_report(config_file, tmp_path):
    """A report researched with a config can be re-checked with it, after a network failure say."""
    report = tmp_path / "report.md"
    report.write_text("Autophagy GO:0006914 (autophagy) and seizures HP:0001250 (Seizure).\n", encoding="utf-8")

    result = CliRunner().invoke(app, [
        "validate-terms", str(report), "--ontology-config", str(config_file),
        "--offline", "--cache-dir", str(tmp_path / "terms"),
    ])

    assert result.exit_code == 0, result.output
    assert "| Not in the configured ontologies | 1 |" in result.output
