"""Tests for typed provider errors and the provider health probe.

Covers the failure mode from issue #65: a provider that is configured but
cannot take work, reported as a timeout instead of as a billing failure.
"""

import httpx
import pytest

from deep_research_client.client import DeepResearchClient
from deep_research_client.exceptions import (
    ProviderAuthError,
    ProviderNotConfiguredError,
    ProviderBillingError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
    classify_exception,
    classify_status,
    extract_status_code,
)
from deep_research_client.models import CacheConfig, ProviderConfig, ProviderHealth
from deep_research_client.providers.biomni import BIOMNI_EXTRA_RUNTIME_MODULES
from deep_research_client.providers.consensus import ConsensusProvider


@pytest.mark.parametrize(
    "status_code,expected,retryable",
    [
        (401, ProviderAuthError, False),
        (402, ProviderBillingError, False),
        (403, ProviderAuthError, False),
        (429, ProviderRateLimitError, True),
        (500, ProviderTransientError, True),
        (503, ProviderTransientError, True),
    ],
)
def test_classify_status_assigns_type_and_retryability(status_code, expected, retryable):
    """Each classified status maps to one type, carrying its own retry verdict."""
    error = classify_status("falcon", status_code, "boom")
    assert isinstance(error, expected)
    assert error.retryable is retryable
    assert error.status_code == status_code
    assert error.provider == "falcon"


@pytest.mark.parametrize("status_code", [200, 404, 418])
def test_unclassified_statuses_return_none(status_code):
    """Statuses we have no opinion about are left for the caller to handle."""
    assert classify_status("falcon", status_code, "boom") is None


def test_billing_error_is_not_retryable_and_says_so():
    """The 402 message must point at credits, not read like a transient blip."""
    error = classify_status("falcon", 402, "Payment Required")
    assert error.retryable is False
    message = str(error)
    assert "out of credits" in message
    assert "--provider" in message


def test_provider_errors_remain_value_errors():
    """Callers written against the pre-typed releases still catch these."""
    assert issubclass(ProviderError, ValueError)


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Client error '402 Payment Required' for url 'https://x/v0.1/crows'", 402),
        ("Client error '401 Unauthorized' for url 'https://x'", 401),
        ("HTTP 503 from upstream", 503),
        ("status code 429 returned", 429),
        ("connection reset by peer", None),
        ("finished in 402 seconds", None),
    ],
)
def test_extract_status_code_from_message(message, expected):
    """The status survives being rendered into a message by an SDK."""
    assert extract_status_code(RuntimeError(message)) == expected


def test_extract_status_code_prefers_response_object():
    """A real response object beats any digits in the message text."""
    request = httpx.Request("GET", "https://example.org")
    response = httpx.Response(402, request=request)
    error = httpx.HTTPStatusError("failed after 500 tries", request=request, response=response)
    assert extract_status_code(error) == 402


def test_extract_status_code_walks_exception_chain():
    """A wrapped failure is still classifiable from the outer exception."""
    inner = RuntimeError("Client error '402 Payment Required' for url 'x'")
    outer = RuntimeError("task failed")
    outer.__cause__ = inner
    assert extract_status_code(outer) == 402


def test_extract_status_code_unwraps_retry_wrapper():
    """A retry wrapper must not bury a permanent failure (issue #65)."""
    tenacity = pytest.importorskip("tenacity")

    attempts = {"n": 0}

    @tenacity.retry(stop=tenacity.stop_after_attempt(2), reraise=False)
    def always_402():
        attempts["n"] += 1
        request = httpx.Request("POST", "https://api.platform.edisonscientific.com/v0.1/crows")
        raise httpx.HTTPStatusError(
            "Client error '402 Payment Required'",
            request=request,
            response=httpx.Response(402, request=request),
        )

    with pytest.raises(tenacity.RetryError) as excinfo:
        always_402()

    assert attempts["n"] == 2
    assert extract_status_code(excinfo.value) == 402
    assert isinstance(classify_exception("falcon", excinfo.value), ProviderBillingError)


def test_classify_exception_leaves_unrecognised_errors_alone():
    """We only rewrite exceptions we can actually explain."""
    assert classify_exception("falcon", ValueError("malformed response")) is None


def test_classify_exception_passes_through_already_typed_errors():
    """Classifying twice must not re-wrap or lose the original type."""
    original = ProviderBillingError("falcon", "no credits", 402)
    assert classify_exception("falcon", original) is original


def test_consensus_maps_http_status_to_typed_error(monkeypatch):
    """The Consensus provider raises the shared type, not a bare ValueError."""
    provider = ConsensusProvider(
        ProviderConfig(name="consensus", api_key="test-key", enabled=True)
    )

    request = httpx.Request("POST", "https://api.consensus.app/x")

    class _FailingClient:
        async def get(self, *args, **kwargs):
            raise httpx.HTTPStatusError(
                "unauthorized",
                request=request,
                response=httpx.Response(401, request=request),
            )

        async def aclose(self):
            return None

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: _FailingClient())

    import asyncio

    with pytest.raises(ProviderAuthError) as excinfo:
        asyncio.run(provider.research("does aspirin help"))

    assert excinfo.value.status_code == 401
    assert "invalid API key" in str(excinfo.value)


@pytest.mark.parametrize(
    "health,expected",
    [
        (ProviderHealth(provider="p", configured=False, detail="no key"), "NOT CONFIGURED"),
        (ProviderHealth(provider="p", configured=True), "UNKNOWN"),
        (ProviderHealth(provider="p", configured=True, reachable=True), "OK"),
        (ProviderHealth(provider="p", configured=True, reachable=False), "UNREACHABLE"),
    ],
)
def test_provider_health_summary_distinguishes_configured_from_reachable(health, expected):
    """"Configured" and "reachable" must be visibly different states."""
    assert expected in health.summary()


def test_unconfigured_provider_reports_unreachable_without_probing():
    """No key means no network call is worth making."""
    import asyncio

    provider = ConsensusProvider(ProviderConfig(name="consensus", api_key=None, enabled=True))
    health = asyncio.run(provider.check_health())

    assert health.configured is False
    assert health.reachable is False


def test_provider_errors_survive_pickling():
    """These may cross a process boundary in a task queue; keep them rebuildable."""
    import pickle

    from deep_research_client.exceptions import ProviderQuotaError

    billing = pickle.loads(pickle.dumps(ProviderBillingError("falcon", "no credits", 402)))
    assert isinstance(billing, ProviderBillingError)
    assert (billing.provider, billing.detail, billing.status_code) == ("falcon", "no credits", 402)

    quota = pickle.loads(pickle.dumps(ProviderQuotaError("claude_code", "spent", resets_at="3pm")))
    assert quota.resets_at == "3pm"


def test_a_dead_end_in_the_retry_wrapper_does_not_abandon_the_search():
    """A retry whose own attempt says nothing must not end the search.

    The retry loop here runs inside an ``except`` block, so the RetryError is
    implicitly chained to the HTTP failure that preceded it -- the status is
    one link further along, and only reachable if the dead end doesn't return.
    """
    tenacity = pytest.importorskip("tenacity")

    @tenacity.retry(stop=tenacity.stop_after_attempt(1), reraise=False)
    def opaque_failure():
        raise RuntimeError("connection reset by peer")

    request = httpx.Request("GET", "https://api.example.org/v1/thing")
    try:
        raise httpx.HTTPStatusError(
            "boom", request=request, response=httpx.Response(503, request=request)
        )
    except httpx.HTTPStatusError:
        with pytest.raises(tenacity.RetryError) as excinfo:
            opaque_failure()

    assert extract_status_code(excinfo.value) == 503


def test_an_ip_address_is_not_a_status_code():
    """`extract_status_code` is exported, so its answer has to mean something."""
    assert extract_status_code(RuntimeError("cannot reach http://127.0.0.1/v1/crows")) is None
    assert extract_status_code(RuntimeError("proxy at HTTP://10.1.2.3:8080 refused")) is None


def test_not_configured_is_one_catchable_class():
    """A caller skipping unusable providers should need only one except clause.

    Imported from the package root on purpose: that is the import the provider
    docs tell callers to write, so the export list is part of the contract.
    """
    from deep_research_client import (
        ProviderNotConfiguredError,
        ProviderNotInstalledError,
        extract_status_code,
    )

    assert extract_status_code is not None

    assert issubclass(ProviderNotInstalledError, ProviderNotConfiguredError)
    assert not issubclass(ProviderNotConfiguredError, ProviderAuthError)


def test_a_long_body_never_costs_the_remedy():
    """Truncation must cut evidence, never the part that says what to do.

    The composed diagnosis is budgeted at construction so the outer
    ProviderHealth cap has nothing left to trim off the end.
    """
    from deep_research_client.exceptions import MAX_DETAIL_CHARS

    body = (
        "Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-proj-***. "
        "You can find your API key at https://platform.openai.com/account/api-keys.', "
        "'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}"
    )
    assert len(body) > MAX_DETAIL_CHARS, "the fixture must actually exceed the cap"

    error = ProviderAuthError("openai", body, 401)
    health = ProviderHealth(
        provider="openai", configured=True, reachable=False, detail=error.diagnosis
    )

    assert len(error.diagnosis) <= MAX_DETAIL_CHARS
    assert health.detail.endswith("lacks access to this endpoint")
    assert "…" in health.detail, "the cut should be visible, not silent"


def test_a_quota_error_keeps_its_reset_time_under_truncation():
    """The renews-at clause is the most actionable part; it must not be cut."""
    from deep_research_client.exceptions import ProviderQuotaError

    error = ProviderQuotaError("claude_code", "Claude usage limit reached. " + "x" * 300, resets_at="3pm")

    assert error.diagnosis.endswith("renews at 3pm")


def test_truncation_leaves_short_text_alone():
    """No marker on text that was never cut."""
    error = ProviderAuthError("openai", "bad key", 401)

    assert error.detail == "bad key"
    assert "…" not in error.diagnosis


def _keyed_provider_classes() -> list:
    """Every registered provider that needs a credential, from the registry.

    Derived rather than listed: the defect this guards was a provider being
    left out of a by-hand enumeration, and a hand-written list here would fail
    the same way. Anything added to PROVIDER_CLASS_PATHS is covered the day it
    lands, or fails this pin.

    Returns:
        (name, class) pairs for providers that declare a credential variable
    """
    import importlib

    from deep_research_client.client import PROVIDER_CLASS_PATHS

    classes = []
    for name, (module_name, class_name) in PROVIDER_CLASS_PATHS.items():
        provider_class = getattr(importlib.import_module(module_name), class_name)
        if provider_class.credential_env_var is not None:
            classes.append((name, provider_class))
    return classes


@pytest.mark.parametrize(
    "name,provider_class", _keyed_provider_classes(), ids=lambda v: v if isinstance(v, str) else None
)
def test_every_keyed_provider_reports_a_missing_key_the_same_way(name, provider_class):
    """One `except` must cover every provider, or the documented catch lies."""
    import asyncio

    from deep_research_client.exceptions import ProviderNotConfiguredError

    provider = provider_class(ProviderConfig(name=name, api_key=None, enabled=True))

    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        asyncio.run(provider.research("what causes scurvy"))

    # The message names the variable to set, not just that something is absent.
    assert provider.credential_env_var in str(excinfo.value)


@pytest.mark.parametrize(
    "name,provider_class", _keyed_provider_classes(), ids=lambda v: v if isinstance(v, str) else None
)
def test_the_cli_hint_table_agrees_with_the_provider_it_describes(name, provider_class):
    """Two places name each credential; they must not drift into two answers."""
    from deep_research_client.cli import PROVIDER_CREDENTIAL_HINTS

    assert name in PROVIDER_CREDENTIAL_HINTS, (
        f"{name} needs an entry in PROVIDER_CREDENTIAL_HINTS so the CLI can name its key"
    )
    assert PROVIDER_CREDENTIAL_HINTS[name] == (
        provider_class.credential_env_var,
        provider_class.credential_label,
    )


def test_the_hint_table_names_no_provider_that_no_longer_exists():
    """The cross-check above runs one way; this closes the other direction."""
    from deep_research_client.cli import PROVIDER_CREDENTIAL_HINTS
    from deep_research_client.client import PROVIDER_CLASS_PATHS

    stale = set(PROVIDER_CREDENTIAL_HINTS) - set(PROVIDER_CLASS_PATHS)
    assert not stale, f"hint entries naming providers that no longer exist: {stale}"


def test_the_registry_actually_yields_keyed_providers():
    """A derived parametrize that silently yields nothing would pass vacuously."""
    names = [name for name, _ in _keyed_provider_classes()]

    assert len(names) >= 6
    assert "openscientist" in names, "the provider whose omission prompted this test"
    assert "claude_code" not in names, "no credential variable, so not in scope"


@pytest.mark.parametrize(
    "message,expected",
    [
        ("HTTP/1.1 503 Service Unavailable", 503),
        ("HTTP/2 429 Too Many Requests", 429),
        ("GET http://127.0.0.1/v1/x failed", None),
    ],
)
def test_a_verbatim_status_line_is_read(message, expected):
    """A rendered status line is a common shape, and slashes must stay excluded.

    The separator class deliberately omits `/` so a URL cannot pass its first
    octet off as a status; this adds the version token as its own alternative
    rather than loosening that.
    """
    assert extract_status_code(RuntimeError(message)) == expected


def test_the_library_reports_a_missing_key_as_a_missing_key(monkeypatch):
    """The CLI learned this in round 5; the library entry point had not.

    Providers register only when their credential is present, so `not found`
    for a correctly-spelled provider is a spelling diagnosis for an unset key.
    """
    for variable in ("EDISON_API_KEY", "FUTUREHOUSE_API_KEY"):
        monkeypatch.delenv(variable, raising=False)

    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))

    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        client.research("what causes scurvy", provider="falcon")

    assert "EDISON_API_KEY" in str(excinfo.value)
    assert "not found" not in str(excinfo.value)


def test_the_library_still_calls_an_unknown_name_unknown():
    """A name that is no provider at all keeps the blunt answer."""
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))

    with pytest.raises(ValueError) as excinfo:
        client.research("what causes scurvy", provider="flacon")

    assert not isinstance(excinfo.value, ProviderNotConfiguredError)
    assert "not found" in str(excinfo.value)


def test_a_disabled_claude_code_is_not_told_to_install_the_cli(monkeypatch):
    """Registration and availability are different gates.

    With the CLI installed and DISABLE_CLAUDE_CODE_PROVIDER set, the provider
    considers itself available -- so asking it why it is unavailable yields its
    only other answer, "the CLI was not found", which is false and sends the
    reader to install software they already have.
    """
    import shutil

    # Patched rather than skipped: CI has no `claude` binary, so a skip here
    # meant the only test pinning this fix never ran where the counts come from.
    monkeypatch.setattr(
        shutil, "which", lambda name, *args, **kwargs: "/usr/bin/claude" if name == "claude" else None
    )
    monkeypatch.setenv("DISABLE_CLAUDE_CODE_PROVIDER", "true")
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))

    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        client.research("what causes scurvy", provider="claude_code")

    message = str(excinfo.value)
    assert "DISABLE_CLAUDE_CODE_PROVIDER" in message
    assert "not found on PATH" not in message, "the CLI is installed; do not say otherwise"


def test_a_disabled_biomni_is_not_told_to_install_the_package(monkeypatch):
    """The same two-gate confusion as claude_code, for the other local provider.

    With the core runtime importable and DISABLE_BIOMNI_PROVIDER set, the
    provider considers itself available, so its own unavailable_reason() would
    tell the reader to install something they already have.
    """
    import importlib.util

    # Patched rather than skipped: ordinary CI never installs the biomni extra,
    # so a skip here would mean this never runs in the base-install job.
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: (
            object()
            if name in BIOMNI_EXTRA_RUNTIME_MODULES
            else real_find_spec(name, *args, **kwargs)
        ),
    )
    monkeypatch.setenv("DISABLE_BIOMNI_PROVIDER", "true")
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))

    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        client.research("what causes scurvy", provider="biomni")

    message = str(excinfo.value)
    assert "DISABLE_BIOMNI_PROVIDER" in message
    assert "deep-research-client[biomni]" in message
    assert message.index("DISABLE_BIOMNI_PROVIDER") < message.index(
        "deep-research-client[biomni]"
    )
    assert "not installed" not in message, "the package is importable; do not say otherwise"


def test_the_mock_provider_names_the_variable_that_enables_it(monkeypatch):
    """MockProvider is always "available", so it cannot explain its own absence."""
    monkeypatch.delenv("ENABLE_MOCK_PROVIDER", raising=False)
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))

    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        client.research("what causes scurvy", provider="mock")

    assert "ENABLE_MOCK_PROVIDER" in str(excinfo.value)
    assert "is not available" not in str(excinfo.value), "the generic wording this PR removed"


def test_cyberian_names_the_binary_it_actually_needs(monkeypatch):
    """The package is a main dependency; the binary is the gate that shuts.

    Reporting "the cyberian package is not installed" would send the reader to
    reinstall something `pip` already gave them, while `agentapi` -- a separate
    Go binary -- is what is missing on a stock install.
    """
    import shutil

    from deep_research_client.providers.cyberian import CyberianProvider

    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: None)
    provider = CyberianProvider(ProviderConfig(name="cyberian"))

    reason = provider.unavailable_reason()

    assert "agentapi" in reason
    assert "package is not installed" not in reason, "the package is a main dependency"
    assert "is not available" not in reason, "the generic wording this branch removed"


def test_cyberian_research_reports_the_same_reason(monkeypatch):
    """The raise must not restate a guess the class can answer properly."""
    import asyncio
    import shutil

    from deep_research_client.exceptions import ProviderNotInstalledError
    from deep_research_client.providers.cyberian import CyberianProvider

    monkeypatch.setattr(shutil, "which", lambda name, *args, **kwargs: None)
    provider = CyberianProvider(ProviderConfig(name="cyberian"))

    with pytest.raises(ProviderNotInstalledError) as excinfo:
        asyncio.run(provider.research("what causes scurvy"))

    assert "agentapi" in str(excinfo.value)


def test_an_explicit_config_is_not_told_to_unset_a_variable_it_never_set(monkeypatch):
    """Absence from the registry has more than one cause.

    Building a client with explicit provider_configs skips environment
    detection entirely, so a message asserting *why* claude_code is missing
    would be false for a caller who never touched that variable.
    """
    import shutil

    # The branch under test is "registered nowhere, yet reports itself
    # available", which needs the CLI to look present. CI has no `claude`
    # binary, so without this the provider is simply unavailable and answers
    # with its own (correct, but different) reason.
    monkeypatch.setattr(
        shutil, "which", lambda name, *args, **kwargs: "/usr/bin/claude" if name == "claude" else None
    )
    monkeypatch.delenv("DISABLE_CLAUDE_CODE_PROVIDER", raising=False)
    client = DeepResearchClient(
        provider_configs={"openai": ProviderConfig(name="openai", api_key="k")},
        cache_config=CacheConfig(enabled=False),
    )

    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        client.research("what causes scurvy", provider="claude_code")

    message = str(excinfo.value)
    assert "DISABLE_CLAUDE_CODE_PROVIDER unset" in message, "state the requirement"
    assert "is set;" not in message, "do not assert a state that was never checked"
