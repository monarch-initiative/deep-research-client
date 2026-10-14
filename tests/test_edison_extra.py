"""Tests for the optional ``edison`` extra (issue #92).

The Edison SDK brings ldp, fhlmi and LiteLLM with it, so it is not part of the
base install. Which half of this module runs depends on the environment: CI
runs the base half in an install without the extra and the other half in one
with it. Nothing here is faked; each test reads the real environment.
"""

import asyncio
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.client import DeepResearchClient
from deep_research_client.exceptions import ProviderNotConfiguredError, ProviderNotInstalledError
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.providers.falcon import (
    EDISON_INSTALL_HINT,
    FalconProvider,
    edison_client_installed,
)

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"

#: Distributions that belong to the Edison stack and nowhere else.
EDISON_STACK = ("edison-client", "ldp", "fhlmi")

without_extra = pytest.mark.skipif(
    edison_client_installed(), reason="checks behaviour without the edison extra"
)
with_extra = pytest.mark.skipif(
    not edison_client_installed(), reason="needs the edison extra installed"
)


def _project() -> dict:
    """Read the ``[project]`` table of the repository's pyproject.toml."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _requirement_names(requirements: list[str]) -> set[str]:
    """Reduce requirement strings to their distribution names.

    >>> sorted(_requirement_names(["edison-client >= 0.11.0", "ldp>=0.45.1"]))
    ['edison-client', 'ldp']
    """
    return {
        requirement.split(">")[0].split("=")[0].split("[")[0].strip().lower()
        for requirement in requirements
    }


def _provider(api_key: str | None = "test-key") -> FalconProvider:
    """Build a Falcon provider with the given key."""
    return FalconProvider(ProviderConfig(name="falcon", api_key=api_key, enabled=True))


def test_the_edison_stack_is_not_a_base_dependency():
    """A base install must not pull in edison-client or what it brings."""
    assert not _requirement_names(_project()["dependencies"]) & set(EDISON_STACK)


@pytest.mark.parametrize("extra", ["edison", "falcon"])
def test_the_edison_extra_installs_the_stack(extra):
    """Both names install the stack: ``falcon`` by naming the ``edison`` extra."""
    extras = _project()["optional-dependencies"]

    if extra == "edison":
        assert _requirement_names(extras[extra]) == set(EDISON_STACK)
    else:
        assert extras[extra] == ["deep_research_client[edison]"]


@without_extra
def test_a_keyed_provider_without_the_extra_says_what_to_install():
    """The key is set, so the package is the only thing left to name."""
    provider = _provider()

    assert not provider.is_available()
    assert provider.unavailable_reason() == (
        f"the Edison client is not installed (install it with `{EDISON_INSTALL_HINT}`)"
    )


@without_extra
def test_without_key_or_extra_both_are_named():
    """Fixing one should not reveal the other on the next run."""
    reason = _provider(api_key=None).unavailable_reason()

    assert "EDISON_API_KEY" in reason
    assert EDISON_INSTALL_HINT in reason


@without_extra
def test_research_without_the_extra_raises_a_not_installed_error():
    """Not installed is its own error, still caught as not configured."""
    with pytest.raises(ProviderNotInstalledError, match=r"deep-research-client\[edison\]"):
        asyncio.run(_provider().research("what causes scurvy"))


@without_extra
def test_health_check_without_the_extra_names_the_extra():
    """The probe reports the missing package instead of trying to import it."""
    health = asyncio.run(_provider().check_health())

    assert health.configured is False
    assert health.reachable is False
    assert EDISON_INSTALL_HINT in (health.detail or "")


@without_extra
def test_the_client_reports_the_missing_extra(monkeypatch):
    """A key in the environment registers the provider; running it says why it cannot."""
    monkeypatch.setenv("EDISON_API_KEY", "test-key")
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))

    with pytest.raises(ProviderNotConfiguredError, match=r"deep-research-client\[edison\]"):
        client.research("what causes scurvy", provider="falcon")


@without_extra
def test_edison_trajectory_without_the_extra_names_the_extra(monkeypatch):
    """The trajectory command fails with the install hint, not an ImportError."""
    monkeypatch.setenv("EDISON_API_KEY", "test-key")

    result = CliRunner().invoke(app, ["edison-trajectory", "784d73d5-da42-402e-9701-6c5b44beab14"])

    assert result.exit_code == 1
    assert EDISON_INSTALL_HINT in result.output


@without_extra
@pytest.mark.parametrize("another_provider_available", [False, True])
def test_research_command_without_the_extra_names_the_extra(
    monkeypatch, bare_machine, another_provider_available
):
    """A known provider gets its fix, whether or not anything else could run.

    With nothing else available, the run used to stop at the list of keys to
    set, which never mentions the extra.
    """
    if another_provider_available:
        monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    monkeypatch.setenv("EDISON_API_KEY", "test-key")

    result = CliRunner().invoke(app, ["research", "what causes scurvy", "--provider", "falcon"])

    assert result.exit_code == 1
    assert EDISON_INSTALL_HINT in result.output


@without_extra
def test_providers_listing_does_not_ask_for_a_key_that_is_set(monkeypatch):
    """With the key set, the missing package is the only thing worth saying."""
    monkeypatch.setenv("EDISON_API_KEY", "test-key")

    result = CliRunner().invoke(app, ["providers"])

    assert result.exit_code == 0, result.output
    assert "EDISON_API_KEY" not in result.output
    assert f"falcon: the Edison client is not installed (install it with `{EDISON_INSTALL_HINT}`)" in (
        result.output
    )


@pytest.mark.parametrize("args", [["--help"], ["providers"]])
def test_cli_discovery_works_with_or_without_the_extra(args):
    """Help and the provider listing must not depend on optional packages."""
    result = CliRunner().invoke(app, args)

    assert result.exit_code == 0, result.output


@with_extra
def test_a_keyed_provider_with_the_extra_is_available():
    """Installing the extra is all it takes, once a key is set."""
    assert _provider().is_available()
