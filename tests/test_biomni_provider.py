"""Tests for the Biomni provider.

The ordinary suite runs without the optional ``biomni`` package, so provider
scaffolding (params, model card, availability, output parsing) stays testable on
the base install. A dedicated clean-extra CI job also imports and constructs the
real agent without downloading its data lake. The actual LLM-backed run remains
an integration test.
"""

import os
from pathlib import Path

import pytest

from deep_research_client.exceptions import ProviderNotInstalledError
from deep_research_client.models import ProviderConfig
from deep_research_client.model_cards import (
    ProviderArchetype,
    ResearchCapability,
    ResearchResource,
    create_biomni_model_cards,
)
from deep_research_client.provider_params import BiomniParams, create_provider_params
from deep_research_client.providers.biomni import (
    BIOMNI_DEFAULT_TIMEOUT,
    BiomniProvider,
    biomni_runtime_unavailable_reason,
    missing_biomni_runtime_modules,
)

BIOMNI_RUNTIME_AVAILABLE = not missing_biomni_runtime_modules()


def make_provider(**params) -> BiomniProvider:
    config = ProviderConfig(name="biomni", api_key=None, enabled=True)
    return BiomniProvider(config, BiomniParams(**params))


def require_biomni_runtime() -> None:
    """Skip on a base install, but fail closed in the clean-extra CI job."""
    missing = missing_biomni_runtime_modules()
    if not missing:
        return
    reason = f"complete Biomni runtime required; missing: {', '.join(missing)}"
    if os.getenv("REQUIRE_BIOMNI_RUNTIME") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def test_default_model_is_card_identity():
    assert make_provider().get_default_model() == "biomni-a1"
    assert make_provider().model == "biomni-a1"


@pytest.mark.parametrize(
    "kwargs,env,expected",
    [
        ({"path": "/data/custom"}, {}, "/data/custom"),
        ({}, {"BIOMNI_DATA_PATH": "/data/env"}, "/data/env"),
        ({}, {}, "./biomni_data"),
    ],
)
def test_data_path_resolution(kwargs, env, expected, monkeypatch):
    monkeypatch.delenv("BIOMNI_DATA_PATH", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert make_provider(**kwargs).data_path == expected


def test_params_llm_distinct_from_model():
    """The underlying LLM is a separate param from the research model card."""
    params = create_provider_params(
        "biomni",
        model=None,
        provider_params={"llm": "claude-sonnet-4-20250514", "source": "Anthropic"},
    )
    assert params.llm == "claude-sonnet-4-20250514"
    assert params.source == "Anthropic"


def test_params_reject_unknown_field():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BiomniParams(not_a_real_field=True)


def test_is_available_tracks_core_runtime_presence():
    provider = make_provider()
    assert provider.is_available() is BIOMNI_RUNTIME_AVAILABLE


def test_disabled_config_is_unavailable():
    config = ProviderConfig(name="biomni", enabled=False)
    assert BiomniProvider(config).is_available() is False


@pytest.mark.asyncio
async def test_research_requires_available_provider():
    if BIOMNI_RUNTIME_AVAILABLE:
        pytest.skip("Biomni runtime available; availability guard not exercised")
    # ProviderNotInstalledError, not a bare ValueError: no credential fixes a
    # missing local package, and callers branch on the type to say so.
    with pytest.raises(
        ProviderNotInstalledError, match=r"deep-research-client\[biomni\]"
    ):
        await make_provider().research("some biomedical question")


def test_unavailable_reason_names_the_package_not_a_key():
    """Biomni has no credential of its own, so the base wording would misdirect."""
    if BIOMNI_RUNTIME_AVAILABLE:
        pytest.skip("Biomni runtime available; the unavailable branch is not reachable")
    missing = missing_biomni_runtime_modules()
    reason = make_provider().unavailable_reason()

    if "biomni" in missing:
        assert "biomni package is not installed" in reason
    else:
        assert "Biomni Python runtime is incomplete" in reason
    assert "deep-research-client[biomni]" in reason
    assert "API key" not in reason


def test_unavailable_reason_reports_a_disabled_provider_as_disabled():
    """A disabled provider is not a missing install, whatever is on disk."""
    provider = BiomniProvider(ProviderConfig(name="biomni", enabled=False))

    assert provider.unavailable_reason() == "Provider 'biomni' is disabled"


@pytest.mark.parametrize(
    "source,module,package",
    [
        ("Ollama", "langchain_ollama", "langchain-ollama"),
        ("Bedrock", "langchain_aws", "langchain-aws"),
    ],
)
def test_external_backend_remedy_names_its_separate_adapter(
    source: str, module: str, package: str
) -> None:
    """Reinstalling the Biomni extra cannot supply opt-in backend adapters."""
    reason = biomni_runtime_unavailable_reason(source, [module])

    assert f"pip install {package}" in reason
    assert "reinstall deep-research-client[biomni]" not in reason


def test_fresh_ollama_install_names_both_required_install_steps() -> None:
    """A missing core must not hide the separately packaged source adapter."""
    reason = biomni_runtime_unavailable_reason(
        "Ollama", ["biomni", "langchain_ollama"]
    )

    assert "deep-research-client[biomni]" in reason
    assert "pip install langchain-ollama" in reason


@pytest.mark.parametrize(
    "missing,expected,unexpected",
    [
        (
            ["pandas"],
            "reinstall deep-research-client[biomni]",
            "install it inside the upstream Biomni environment",
        ),
        (
            ["rdkit"],
            "install it inside the upstream Biomni environment",
            "reinstall deep-research-client[biomni]",
        ),
    ],
)
def test_runtime_remedy_distinguishes_extra_from_scientific_environment(
    missing: list[str], expected: str, unexpected: str
) -> None:
    """Only dependencies actually owned by the extra recommend reinstalling it."""
    reason = biomni_runtime_unavailable_reason(None, missing)

    assert expected in reason
    assert unexpected not in reason


@pytest.mark.asyncio
async def test_research_rejects_empty_query():
    # The empty-query guard runs before the availability check, so it is
    # exercised whether or not the optional package is installed.
    with pytest.raises(ValueError, match="must not be empty"):
        await make_provider().research("   ")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("final answer", "final answer"),
        (("verbose log", "final answer"), "final answer"),
        (["a", "b", "c"], "c"),
        ([], ""),
    ],
)
def test_result_to_markdown(raw, expected):
    assert BiomniProvider._result_to_markdown(raw) == expected


def test_an_empty_report_is_flagged_rather_than_returned_in_silence(caplog):
    """The one fallback that returns a value rather than raising must still say so."""
    import logging

    with caplog.at_level(logging.DEBUG, logger="deep_research_client.providers.biomni"):
        assert BiomniProvider._result_to_markdown([]) == ""

    assert any("empty collection" in record.message for record in caplog.records)
    # research() owns the WARNING for every empty shape; this level must not
    # double it, or one empty run logs the same event twice.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_result_to_markdown_object_with_content():
    class Result:
        content = "the report"

    assert BiomniProvider._result_to_markdown(Result()) == "the report"


def test_extract_citations_pmids_and_dois():
    """Deduplicated, in the project's canonical form.

    Ordering is by identifier type (PMIDs, then DOIs, then accessions), and
    first-appearance only within a type -- find_reference_ids scans once per
    pattern, not once through the document.
    """
    text = "Evidence PMID: 12345678 and PMID:12345678 plus doi:10.1000/abc.def."
    assert BiomniProvider._extract_citations(text) == [
        "PMID:12345678",
        "DOI:10.1000/abc.def",
    ]


@pytest.mark.parametrize(
    "text,expected",
    [
        # A DOI written as a markdown link: the identifier, not the whole tail.
        (
            "[doi:10.1038/nature12373](https://doi.org/10.1038/nature12373)",
            ["DOI:10.1038/nature12373"],
        ),
        # A six-digit PMID is a 1970s paper, not a truncated modern one.
        ("Shown decades ago (PMID: 942051).", ["PMID:942051"]),
        # A bare PubMed URL is how agents most often cite.
        ("https://pubmed.ncbi.nlm.nih.gov/12345678", ["PMID:12345678"]),
        # Accessions a biomedical agent emits routinely.
        ("See PMC11000121 for details.", ["PMC:PMC11000121"]),
        ("Deposited under GSE68086.", ["GEO:GSE68086"]),
        # Parenthesised DOIs are real; the closing paren is not part of them.
        ("(doi:10.1016/0092-8674(94)90302-6)", ["DOI:10.1016/0092-8674(94)90302-6"]),
        ("nothing to cite here", []),
    ],
)
def test_extract_citations_handles_the_forms_reports_actually_use(text, expected):
    """Biomni reads references through the same patterns as reference validation.

    Each case below is one a hand-rolled `PMID:\\d{7,9}` / `doi:(10\\.\\S+)` pair
    got wrong: it dropped the pre-1990 PMID, the URL, and both accession types,
    and captured the markdown link's closing bracket as part of the DOI.
    """
    assert BiomniProvider._extract_citations(text) == expected


def test_biomni_model_card_shape():
    cards = create_biomni_model_cards()
    card = cards.get_model_card("biomni-a1")
    assert card.archetype == ProviderArchetype.co_scientist
    assert ResearchResource.pubmed in card.resources
    assert ResearchCapability.code_interpretation in card.capabilities
    assert ResearchCapability.hypothesis_generation in card.capabilities
    # Aliases resolve to the canonical model name
    assert cards.resolve_model_name("biomni") == "biomni-a1"


@pytest.mark.parametrize(
    "params_timeout,config_timeout,expected",
    [
        # An explicit param wins over the config default...
        (120, 900, 120),
        # ...the config is used when the param says nothing...
        (None, 900, 900),
        # ...and the module default is the floor when neither does.
        (None, None, BIOMNI_DEFAULT_TIMEOUT),
        (120, None, 120),
    ],
)
def test_agent_kwargs_timeout_precedence(params_timeout, config_timeout, expected):
    """A1's parameter is `timeout_seconds`, and params outrank the config.

    Covered here as well as in the clean-extra CI job so the precedence rule is
    still exercised by the ordinary base-install suite.
    """
    config = ProviderConfig(name="biomni", api_key=None, enabled=True, timeout=config_timeout)
    provider = BiomniProvider(config, BiomniParams(timeout=params_timeout))

    kwargs = provider._agent_kwargs()

    assert kwargs["timeout_seconds"] == expected
    assert "timeout" not in kwargs, "A1 takes timeout_seconds; a bare timeout is dropped"


def test_agent_kwargs_skip_data_lake_passes_an_empty_expected_file_list():
    """The empty list is what tells A1 not to load or download the ~11GB lake."""
    assert make_provider(skip_data_lake=True)._agent_kwargs()["expected_data_lake_files"] == []


def test_agent_kwargs_omits_the_data_lake_key_by_default():
    """Absent, not empty: an empty list would disable the lake for every run."""
    assert "expected_data_lake_files" not in make_provider()._agent_kwargs()


def test_agent_kwargs_omits_unset_optional_parameters():
    """A1 infers its own defaults, so passing None would override them with nothing."""
    kwargs = make_provider()._agent_kwargs()

    for unset in ("llm", "source", "base_url", "api_key"):
        assert unset not in kwargs

    assert kwargs["path"] == "./biomni_data"
    assert kwargs["use_tool_retriever"] is True


def test_agent_kwargs_forwards_the_configured_llm_and_credentials():
    """What the caller does set has to reach the agent."""
    config = ProviderConfig(
        name="biomni", api_key="sk-test", enabled=True, base_url="https://llm.example"
    )
    provider = BiomniProvider(
        config, BiomniParams(llm="claude-sonnet-4-20250514", source="Anthropic")
    )

    kwargs = provider._agent_kwargs()

    assert kwargs["llm"] == "claude-sonnet-4-20250514"
    assert kwargs["source"] == "Anthropic"
    assert kwargs["base_url"] == "https://llm.example"
    assert kwargs["api_key"] == "sk-test"


def _every_possible_agent_kwarg() -> set:
    """Every key _agent_kwargs can emit, derived by asking it rather than listing.

    A hand-written superset here would drift the moment a parameter is added --
    the same failure mode the registry-derived guards in test_provider_errors.py
    exist to avoid. Two configurations cover it: one setting every optional
    field, and one setting skip_data_lake, which is the only mutually exclusive
    branch.

    Returns:
        Union of the kwarg keys across those configurations
    """
    fully_configured = BiomniProvider(
        ProviderConfig(
            name="biomni", api_key="sk-test", enabled=True, base_url="https://llm.example", timeout=60
        ),
        BiomniParams(llm="claude-sonnet-4-20250514", source="Anthropic", timeout=120),
    )
    return set(fully_configured._agent_kwargs()) | set(
        make_provider(skip_data_lake=True)._agent_kwargs()
    )


def test_the_derived_kwarg_set_covers_every_documented_parameter():
    """A derived superset that silently lost a key would pass the test below."""
    assert _every_possible_agent_kwarg() == {
        "path",
        "use_tool_retriever",
        "llm",
        "source",
        "base_url",
        "api_key",
        "timeout_seconds",
        "expected_data_lake_files",
    }


def test_build_agent_kwargs_accepted_by_a1_signature():
    """Every kwarg _agent_kwargs can emit must be accepted by A1.__init__.

    Guards against silent signature drift (e.g. `timeout` vs `timeout_seconds`)
    without mocking or constructing the agent (which would download the data
    lake by default). Skipped in the ordinary base-install job and exercised by
    the dedicated clean-extra CI job.
    """
    import inspect

    require_biomni_runtime()
    from biomni.agent import A1  # type: ignore[import-not-found, import-untyped]

    accepted = set(inspect.signature(A1.__init__).parameters)
    missing = _every_possible_agent_kwarg() - accepted
    assert not missing, f"A1.__init__ does not accept: {sorted(missing)}"


def test_build_agent_constructs_with_clean_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented extra can construct the real default A1 without a download.

    This is intentionally not mocked. Biomni 0.0.8 under-declares pandas and
    its LangChain adapters, so an import-only signature test catches only the
    first missing package. Constructing A1 reaches the default Anthropic path;
    ``skip_data_lake`` and a temporary path keep the smoke test local and small.
    """
    require_biomni_runtime()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = make_provider(
        path=str(tmp_path),
        skip_data_lake=True,
        use_tool_retriever=False,
    )

    agent = provider._build_agent()

    assert agent.__class__.__name__ == "A1"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_biomni_end_to_end():
    """Run a real (small) Biomni task. Requires the biomni package + LLM key."""
    pytest.importorskip("biomni")
    provider = make_provider(skip_data_lake=True)
    result = await provider.research(
        "In one sentence, what does the tumor suppressor gene TP53 do?"
    )
    assert result.provider == "biomni"
    assert result.markdown.strip()


def test_cli_does_not_claim_a_missing_api_key_for_an_optional_package():
    """Biomni has no credential, so asking for one sends the reader nowhere.

    Mirrors the deeper_med precedent: `providers --provider X` used to report
    every non-stub as credential-blocked, disagreeing with what
    `providers --check` said about the same provider.
    """
    if BIOMNI_RUNTIME_AVAILABLE:
        pytest.skip("Biomni runtime available; the unavailable branch is not reachable")
    from typer.testing import CliRunner

    from deep_research_client.cli import app

    result = CliRunner().invoke(app, ["providers", "--provider", "biomni"])

    assert result.exit_code == 0, result.stdout
    assert "missing API key" not in result.stdout
    assert "deep-research-client[biomni]" in result.stdout


def test_the_two_provider_report_paths_agree():
    """`providers --provider X` and `providers --check` describe one provider.

    Both paths are invoked rather than the shared helper they call: asserting
    against `client.unregistered_reason` directly would keep passing if one
    path stopped using it, which is the drift this is here to catch.
    """
    if BIOMNI_RUNTIME_AVAILABLE:
        pytest.skip("Biomni runtime available; the unavailable branch is not reachable")
    from typer.testing import CliRunner

    from deep_research_client.cli import app

    detail = CliRunner().invoke(app, ["providers", "--provider", "biomni"])
    checked = CliRunner().invoke(app, ["providers", "--check", "--provider", "biomni"])

    explanation = make_provider().unavailable_reason()
    assert "deep-research-client[biomni]" in explanation
    assert explanation in detail.stdout, detail.stdout
    assert explanation in checked.stdout, checked.stdout


def test_an_unavailable_biomni_is_listed_rather_than_omitted():
    """It was in no section at all: not available, not unavailable, not a stub.

    A reader who had never heard of biomni could not learn from this command
    that it exists or what would enable it.
    """
    if BIOMNI_RUNTIME_AVAILABLE:
        pytest.skip("Biomni runtime available; it appears under available providers")
    from typer.testing import CliRunner

    from deep_research_client.cli import app

    result = CliRunner().invoke(app, ["providers"])

    assert result.exit_code == 0, result.stdout
    # Pins the section, not just the string: the message appearing anywhere in
    # the output would otherwise pass even if the section were removed.
    assert "Other unavailable providers:" in result.stdout
    section = result.stdout.split("Other unavailable providers:", 1)[1]
    assert "biomni" in section
    assert "deep-research-client[biomni]" in section
