"""Tests for the `providers --check` surface.

This is the command people run *because* something is already broken, so it
has to survive its own probes failing and still report on everything else.
"""

import pytest
import typer
from typer.testing import CliRunner

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
    """A client that owns only a registry, plus the one explanation hook.

    `unregistered_reason` defers to a real client so these tests exercise the
    branch users take. Without it the CLI falls back to its own tables, and
    these assertions would pin wording production can no longer produce -- as
    they did for cyberian, which reports its missing `agentapi` binary rather
    than a package it already has.
    """

    def __init__(self, providers: list[_StubProvider]):
        """Wrap the providers in a registry."""
        self.registry = _StubRegistry(providers)

    def unregistered_reason(self, provider: str) -> str:
        """Answer exactly as the real client does.

        Args:
            provider: Canonical provider name.

        Returns:
            The real client's explanation
        """
        from deep_research_client.client import DeepResearchClient
        from deep_research_client.models import CacheConfig

        return DeepResearchClient(
            cache_config=CacheConfig(enabled=False)
        ).unregistered_reason(provider)


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


def test_a_genuine_typo_is_still_called_a_typo(capsys):
    """A name we do not recognise at all keeps the blunt message.

    Two assertions for two properties, because neither covers the other. The
    command run proves the message survives the real wiring -- a `caplog`
    assertion would not, since the Typer callback runs
    `logging.basicConfig(force=True)` and drops the handler it relies on. The
    `capsys` run proves the message is on *stdout*: this click pins
    `mix_stderr=True`, so `result.stdout` is really both streams merged and
    would pass just as happily with the message back on stderr.
    """
    from typer.testing import CliRunner

    import deep_research_client.cli as cli_module

    result = CliRunner().invoke(cli_module.app, ["providers", "--check", "--provider", "flacon"])
    assert result.exit_code == 1
    assert "Unknown provider" in result.stdout, "the typo must actually be reported"
    assert "NOT CONFIGURED" not in result.stdout

    with pytest.raises(typer.Exit):
        _check_provider_health(_StubClient([_ok("claude_code")]), "flacon")
    assert "Unknown provider" in capsys.readouterr().out, "and on stdout, not stderr"


def test_an_unconfigured_provider_says_what_would_fix_it(capsys, no_local_binaries):
    """"NOT CONFIGURED" with no next step is the empty answer this replaces.

    cyberian needs an optional package rather than a credential, so there is no
    environment variable to name -- but there is still something to say.
    """
    with pytest.raises(typer.Exit):
        _check_provider_health(_StubClient([_ok("claude_code")]), "cyberian")

    out = capsys.readouterr().out
    assert "cyberian: NOT CONFIGURED" in out
    # What a user is actually told: cyberian ships as a base dependency, so the
    # missing piece is the `agentapi` binary, not a package to pip install.
    assert "agentapi" in out, "a provider with no env var still needs a next step"


@pytest.fixture
def no_local_binaries(monkeypatch):
    """Pin PATH so the explanations below do not depend on the machine.

    The CLI now asks the client, and the client asks the provider, so a runner
    that happens to have `claude` or `agentapi` installed gets a different --
    equally correct -- sentence. Fixing the environment is what makes these
    assertions about wording rather than about the box they run on.
    """
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: None)


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("falcon", "no Edison Scientific API key configured (set EDISON_API_KEY)"),
        ("claude_code", "was not found on PATH"),
        ("cyberian", "agentapi"),
    ],
)
def test_every_unconfigured_answer_carries_its_own_next_step(
    capsys, no_local_binaries, provider, expected
):
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


def test_the_research_hint_list_keeps_its_heading(capsys, monkeypatch):
    """A heading on one stream and its list on another is half a message.

    The list is indented variable names; without the sentence above it, a
    redirected file says nothing about what they are for.
    """
    import deep_research_client.cli as cli_module

    monkeypatch.setattr(
        cli_module.DeepResearchClient, "get_available_providers", lambda self: []
    )

    result = CliRunner().invoke(cli_module.app, ["research", "what causes scurvy"])
    assert "Please set API keys" in result.stdout, "the real command must emit it"

    # And on stdout specifically. CliRunner cannot show that -- this click
    # merges the two streams into one buffer -- so the helper is called direct.
    cli_module._echo_no_providers_message()
    captured = capsys.readouterr().out

    assert "Please set API keys" in captured
    heading_at = captured.index("Please set API keys")
    assert captured.index("OPENAI_API_KEY") > heading_at, "the list must follow its heading"


def test_the_key_list_offers_only_things_a_user_can_set(monkeypatch):
    """A CLI on PATH and a test double are not answers to "set API keys"."""
    import deep_research_client.cli as cli_module

    monkeypatch.setattr(
        cli_module.DeepResearchClient, "get_available_providers", lambda self: []
    )

    result = CliRunner().invoke(cli_module.app, ["research", "what causes scurvy"])

    assert "ENABLE_MOCK_PROVIDER" not in result.stdout, "a mock is not research"
    assert "CLI on PATH" not in result.stdout, "not something you set"
    assert "EDISON_API_KEY" in result.stdout


def test_the_cli_calls_a_method_the_client_actually_has():
    """The CLI asks by name, so a rename would degrade it to the tables silently.

    Duck-typed rather than isinstance-checked so a test double can stand in --
    which is exactly why the name needs pinning somewhere.
    """
    from deep_research_client.client import DeepResearchClient

    assert callable(getattr(DeepResearchClient, "unregistered_reason", None)), (
        "cli._why_unconfigured looks up 'unregistered_reason' on the client; "
        "renaming it here would silently fall back to the CLI's own tables"
    )
