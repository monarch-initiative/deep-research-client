"""Tests for Edison/Falcon failure classification.

Edison authenticates in the client constructor and retries inside the SDK, so
its failures arrive in two different places and one of them looks like a
timeout. Both paths are covered here.
"""

import asyncio

import httpx
import pytest

from deep_research_client.exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderError,
    ProviderNotConfiguredError,
)
from deep_research_client.models import ProviderConfig
from deep_research_client.providers.falcon import FalconProvider

EDISON = "deep_research_client.providers.falcon.EdisonClient"


def _provider(api_key: str | None = "test-key") -> FalconProvider:
    """Build a provider with the given key."""
    return FalconProvider(ProviderConfig(name="falcon", api_key=api_key, enabled=True))


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    """Build the error httpx raises for a status."""
    request = httpx.Request("POST", "https://api.platform.edisonscientific.com/v0.1/crows")
    return httpx.HTTPStatusError(
        f"Client error '{status_code}'",
        request=request,
        response=httpx.Response(status_code, request=request),
    )


def test_rejected_key_at_construction_is_an_auth_error(monkeypatch):
    """The constructor authenticates, so a bad key fails before the first call."""
    def _reject(**kwargs):
        raise _http_error(403)

    monkeypatch.setattr(EDISON, _reject)

    with pytest.raises(ProviderAuthError):
        asyncio.run(_provider().research("what causes scurvy"))


def test_a_constructor_failure_we_cannot_explain_is_left_alone(monkeypatch):
    """A DNS failure is not an auth problem, and must not be reported as one."""
    def _boom(**kwargs):
        raise ConnectionError("dns lookup failed for api.platform.edisonscientific.com")

    monkeypatch.setattr(EDISON, _boom)

    with pytest.raises(ConnectionError) as excinfo:
        asyncio.run(_provider().research("what causes scurvy"))

    # Specifically not relabelled as an auth failure.
    assert not isinstance(excinfo.value, ProviderError)


def test_an_sdk_signature_change_is_left_alone(monkeypatch):
    """A TypeError from the SDK says nothing about the credential either."""
    def _boom(**kwargs):
        raise TypeError("EdisonClient() got an unexpected keyword argument 'api_key'")

    monkeypatch.setattr(EDISON, _boom)

    with pytest.raises(TypeError):
        asyncio.run(_provider().research("what causes scurvy"))


def test_payment_required_mid_run_reaches_the_caller_as_billing(monkeypatch):
    """The failure this issue started with: a 402 raised during the run."""
    class _Client:
        def run_tasks_until_done(self, *args, **kwargs):
            raise _http_error(402)

        def close(self) -> None:
            """Match the real client, which the provider closes after use."""

    monkeypatch.setattr(EDISON, lambda **kwargs: _Client())

    with pytest.raises(ProviderBillingError) as excinfo:
        asyncio.run(_provider().research("what causes scurvy"))

    assert excinfo.value.retryable is False
    assert "out of credits" in str(excinfo.value)


def test_a_run_failure_we_cannot_explain_is_left_alone(monkeypatch):
    """Same discipline on the run path as on the constructor path."""
    class _Client:
        def run_tasks_until_done(self, *args, **kwargs):
            raise RuntimeError("the response was shaped unexpectedly")

        def close(self) -> None:
            """Match the real client, which the provider closes after use."""

    monkeypatch.setattr(EDISON, lambda **kwargs: _Client())

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(_provider().research("what causes scurvy"))

    assert not isinstance(excinfo.value, ProviderError)


def test_missing_key_names_the_variable_to_set():
    """"Not available" is useless; say which credential is missing."""
    with pytest.raises(ProviderNotConfiguredError, match="EDISON_API_KEY"):
        asyncio.run(_provider(api_key=None).research("what causes scurvy"))


class _ProbeClient:
    """A stand-in Edison client that records whether it was closed."""

    def __init__(self, error: Exception | None = None):
        """Fail the listing with the given error, or succeed."""
        self._error = error
        self.closed = False
        self.list_calls: list[int] = []

    def get_tasks(self, limit: int = 50):
        """Record the probe call, then answer or raise."""
        self.list_calls.append(limit)
        if self._error is not None:
            raise self._error
        return []

    def close(self) -> None:
        """Record that the session was released."""
        self.closed = True


def test_probe_lists_one_task_and_closes_the_session(monkeypatch):
    """The healthy path is the only place close() runs, so pin it."""
    client = _ProbeClient()
    monkeypatch.setattr(EDISON, lambda **kwargs: client)

    health = asyncio.run(_provider().check_health())

    assert health.reachable is True
    assert client.list_calls == [1], "the probe must list, not submit work"
    assert client.closed is True
    # A passing probe must not be read as "this account can pay for a run".
    assert "credit balance not exposed" in health.detail


def test_probe_reports_a_spent_account_as_unreachable(monkeypatch):
    """A 402 from the probe is reported, not raised, and still closes."""
    client = _ProbeClient(error=_http_error(402))
    monkeypatch.setattr(EDISON, lambda **kwargs: client)

    health = asyncio.run(_provider().check_health())

    assert health.reachable is False
    assert "out of credits" in health.detail
    assert client.closed is True


def test_probe_survives_a_constructor_failure(monkeypatch):
    """A rejected key fails before there is a client to close."""
    def _reject(**kwargs):
        raise _http_error(403)

    monkeypatch.setattr(EDISON, _reject)

    health = asyncio.run(_provider().check_health())

    assert health.reachable is False
    assert "403" in health.detail


def test_probe_on_an_unconfigured_provider_makes_no_call(monkeypatch):
    """No key means no client is built at all."""
    def _fail(**kwargs):
        raise AssertionError("an unconfigured provider must not be probed")

    monkeypatch.setattr(EDISON, _fail)

    health = asyncio.run(_provider(api_key=None).check_health())

    assert health.configured is False
    assert health.reachable is False
