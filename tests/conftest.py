"""Fixtures shared across test modules."""

import pytest

from deep_research_client.cli import _settable_credential_hints
from deep_research_client.client import load_provider_class


@pytest.fixture
def bare_machine(monkeypatch):
    """Pin PATH and credentials so the explanations below do not depend on the box.

    The CLI now asks the client, and the client asks the provider, so a runner
    that happens to have `claude` or `agentapi` installed -- or `EDISON_API_KEY`
    exported -- gets a different, equally correct sentence. A provider that is
    actually usable is told so rather than told what it is missing, which is the
    whole point of the client's availability guard. Fixing the environment is
    what makes these assertions about wording rather than about the machine.

    The variables are read from the provider classes rather than listed here,
    so a keyed provider added to the registry cannot leave a stale exception.
    """
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: None)
    for provider_name in _settable_credential_hints():
        monkeypatch.delenv(load_provider_class(provider_name).credential_env_var, raising=False)
    # Three gates no credential variable covers: the deprecated alias the client
    # still accepts for falcon, the mock's opt-in flag, and biomni, which is
    # gated on an import rather than a variable -- so `uv sync --extra biomni`
    # would otherwise register it.
    monkeypatch.delenv("FUTUREHOUSE_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_MOCK_PROVIDER", raising=False)
    monkeypatch.setenv("DISABLE_BIOMNI_PROVIDER", "true")

    # Assert the state rather than trust the list: a fourth gate should fail
    # here, naming what stayed available, not downstream as a missing heading.
    # Only a variable or a PATH probe can be closed from here, so a future
    # provider registered on a bare import with no opt-out will fail this --
    # correctly, and the fix is to give that provider an opt-out variable.
    from deep_research_client.client import DeepResearchClient
    from deep_research_client.models import CacheConfig

    still_up = DeepResearchClient(
        cache_config=CacheConfig(enabled=False)
    ).get_available_providers()
    assert not still_up, f"bare_machine left {still_up} registered"
