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

    monkeypatch.setattr(EDISON, lambda **kwargs: _Client())

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(_provider().research("what causes scurvy"))

    assert not isinstance(excinfo.value, ProviderError)


def test_missing_key_names_the_variable_to_set():
    """"Not available" is useless; say which credential is missing."""
    with pytest.raises(ProviderNotConfiguredError, match="EDISON_API_KEY"):
        asyncio.run(_provider(api_key=None).research("what causes scurvy"))
