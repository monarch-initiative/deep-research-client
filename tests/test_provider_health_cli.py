"""Tests for the `providers --check` surface.

This is the command people run *because* something is already broken, so it
has to survive its own probes failing and still report on everything else.
"""

import pytest
import typer

from deep_research_client.cli import _check_provider_health
from deep_research_client.models import ProviderHealth


class _StubProvider:
    """A provider whose probe returns, or raises, whatever a test needs."""

    def __init__(
        self,
        name: str,
        health: ProviderHealth | None = None,
        error: Exception | None = None,
    ):
        """Record what this provider's probe should do."""
        self.name = name
        self._health = health
        self._error = error

    async def check_health(self) -> ProviderHealth:
        """Return the configured health, or raise the configured error."""
        if self._error is not None:
            raise self._error
        assert self._health is not None, "stub needs either a health or an error"
        return self._health


class _StubRegistry:
    """Just enough registry for the health command."""

    def __init__(self, providers: list[_StubProvider]):
        """Hold the providers this registry knows about."""
        self._providers = {p.name: p for p in providers}

    def get_provider(self, name: str):
        """Look a provider up by name."""
        return self._providers.get(name)

    def get_available_providers(self) -> list[_StubProvider]:
        """Every stub is treated as configured."""
        return list(self._providers.values())


class _StubClient:
    """A client that owns only a registry."""

    def __init__(self, providers: list[_StubProvider]):
        """Wrap the providers in a registry."""
        self.registry = _StubRegistry(providers)


def _ok(name: str) -> _StubProvider:
    """A provider that probes clean."""
    return _StubProvider(name, ProviderHealth(provider=name, configured=True, reachable=True))


def _down(name: str) -> _StubProvider:
    """A provider that probes unreachable."""
    return _StubProvider(
        name, ProviderHealth(provider=name, configured=True, reachable=False, detail="402 no credits")
    )


def test_all_healthy_exits_zero(capsys):
    """Nothing wrong means nothing to signal."""
    _check_provider_health(_StubClient([_ok("a"), _ok("b")]), None)

    out = capsys.readouterr().out
    assert "a: OK" in out and "b: OK" in out


def test_any_unreachable_exits_nonzero(capsys):
    """A broken provider has to be visible to a shell script, not just a reader."""
    with pytest.raises(typer.Exit) as excinfo:
        _check_provider_health(_StubClient([_ok("a"), _down("b")]), None)

    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "a: OK" in out
    assert "b: UNREACHABLE" in out


def test_a_probe_that_raises_does_not_cost_the_other_reports(capsys):
    """One provider blowing up must not take the whole report with it."""
    providers = [_ok("a"), _StubProvider("b", error=FileNotFoundError("claude")), _ok("c")]

    with pytest.raises(typer.Exit) as excinfo:
        _check_provider_health(_StubClient(providers), None)

    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "a: OK" in out and "c: OK" in out
    assert "b: UNREACHABLE" in out
    assert "the probe itself failed" in out


def test_unknown_provider_is_rejected():
    """A typo should not silently probe nothing and report success."""
    with pytest.raises(typer.Exit) as excinfo:
        _check_provider_health(_StubClient([_ok("a")]), "nosuchprovider")

    assert excinfo.value.exit_code == 1


def test_named_provider_is_the_only_one_probed(capsys):
    """--provider narrows the probe rather than filtering the output."""
    _check_provider_health(_StubClient([_ok("a"), _down("b")]), "a")

    out = capsys.readouterr().out
    assert "a: OK" in out
    assert "b:" not in out


def test_no_configured_providers_is_an_error(capsys):
    """Probing nothing is not the same as everything being fine."""
    with pytest.raises(typer.Exit) as excinfo:
        _check_provider_health(_StubClient([]), None)

    assert excinfo.value.exit_code == 1
    # The user is told what to set rather than just that nothing happened.
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_the_check_flag_is_wired_to_the_command(monkeypatch):
    """Cover the flag parsing and early return, not just the helper beneath."""
    from typer.testing import CliRunner

    import deep_research_client.cli as cli_module

    probed: list[str] = []

    def _fake_check(client, provider):
        probed.append(provider or "<all>")

    monkeypatch.setattr(cli_module, "_check_provider_health", _fake_check)

    result = CliRunner().invoke(cli_module.app, ["providers", "--check"])

    assert result.exit_code == 0
    assert probed == ["<all>"], "the command must delegate to the health check"
    # The early return means the normal listing never runs.
    assert "Available providers:" not in result.stdout


def test_a_missing_key_is_not_reported_as_a_typo(capsys):
    """Providers register only once their key is set, so absent != misspelled.

    Saying "unknown provider" would send the reader hunting for a spelling
    mistake instead of exporting a variable -- the wrong-remedy failure this
    whole change exists to remove.
    """
    with pytest.raises(typer.Exit) as excinfo:
        _check_provider_health(_StubClient([_ok("claude_code")]), "falcon")

    assert excinfo.value.exit_code == 1
    out = capsys.readouterr().out
    assert "NOT CONFIGURED" in out
    assert "EDISON_API_KEY" in out
    assert "Unknown provider" not in out


def test_a_genuine_typo_is_still_called_a_typo(capsys, caplog):
    """A name we do not recognise at all keeps the blunt message."""
    with pytest.raises(typer.Exit) as excinfo:
        _check_provider_health(_StubClient([_ok("claude_code")]), "flacon")

    assert excinfo.value.exit_code == 1
    assert "NOT CONFIGURED" not in capsys.readouterr().out
    assert "Unknown provider" in caplog.text, "the typo must actually be reported"


def test_an_unconfigured_provider_says_what_would_fix_it(capsys):
    """"NOT CONFIGURED" with no next step is the empty answer this replaces.

    cyberian needs an optional package rather than a credential, so there is no
    environment variable to name -- but there is still something to say.
    """
    with pytest.raises(typer.Exit):
        _check_provider_health(_StubClient([_ok("claude_code")]), "cyberian")

    out = capsys.readouterr().out
    assert "cyberian: NOT CONFIGURED" in out
    assert "pip install" in out, "a provider with no env var still needs a next step"


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("falcon", "set EDISON_API_KEY for Edison Scientific"),
        ("claude_code", "Claude Code requires the `claude` CLI on PATH"),
        ("cyberian", "pip install"),
    ],
)
def test_every_unconfigured_answer_carries_its_own_next_step(capsys, provider, expected):
    """A key, a binary and a package are three different fixes; say which.

    The claude_code case is the one that reads as nonsense if every hint is
    assumed to be an environment variable: "set the `claude` CLI on PATH".
    """
    with pytest.raises(typer.Exit):
        _check_provider_health(_StubClient([_ok("somethingelse")]), provider)

    out = capsys.readouterr().out
    assert f"{provider}: NOT CONFIGURED" in out
    assert expected in out
    assert "set the" not in out


def test_show_params_with_check_says_it_does_nothing(monkeypatch):
    """An accepted-but-ineffective flag should say so, not vanish."""
    from typer.testing import CliRunner

    import deep_research_client.cli as cli_module

    monkeypatch.setattr(cli_module, "_check_provider_health", lambda client, provider: None)

    result = CliRunner().invoke(cli_module.app, ["providers", "--check", "--show-params"])

    assert result.exit_code == 0
    assert "has no effect" in result.stdout
