"""Tests for OpenAI failure classification and its health probe.

OpenAI reports a spent quota with the same status as ordinary throttling, so
these cover the case where reading the status alone gives the wrong answer.
"""

import asyncio

import httpx
import pytest
from openai import APIStatusError, AuthenticationError, RateLimitError

from deep_research_client.exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from deep_research_client.models import ProviderConfig
from deep_research_client.providers.openai import (
    OpenAIProvider,
    _classify_openai_error,
)


def _status_error(error_class, status_code: int, code: str | None, message: str = "boom"):
    """Build an OpenAI SDK error the way the SDK itself would."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code, request=request)
    body = {"message": message, "code": code, "type": "invalid_request_error"}
    return error_class(message, response=response, body=body)


def _provider(api_key: str | None = "test-key") -> OpenAIProvider:
    """Build a provider with the given key."""
    return OpenAIProvider(ProviderConfig(name="openai", api_key=api_key, enabled=True))


def test_spent_quota_is_billing_not_rate_limit():
    """429 + insufficient_quota is permanent; the bare status says otherwise."""
    error = _classify_openai_error(
        "openai", _status_error(RateLimitError, 429, "insufficient_quota")
    )

    assert isinstance(error, ProviderBillingError)
    assert error.retryable is False


def test_ordinary_throttling_stays_retryable():
    """A 429 without a quota code is the throttle it looks like."""
    error = _classify_openai_error(
        "openai", _status_error(RateLimitError, 429, "rate_limit_exceeded")
    )

    assert isinstance(error, ProviderRateLimitError)
    assert error.retryable is True


@pytest.mark.parametrize(
    "error_class,status_code,code,expected",
    [
        (RateLimitError, 429, "billing_hard_limit_reached", ProviderBillingError),
        (AuthenticationError, 401, "invalid_api_key", ProviderAuthError),
        (AuthenticationError, 401, "account_deactivated", ProviderAuthError),
        (AuthenticationError, 401, None, ProviderAuthError),
        (APIStatusError, 403, None, ProviderAuthError),
        (APIStatusError, 503, None, ProviderTransientError),
    ],
)
def test_error_codes_and_statuses_map_to_types(error_class, status_code, code, expected):
    """Codes win where present; the status carries the rest."""
    error = _classify_openai_error("openai", _status_error(error_class, status_code, code))

    assert isinstance(error, expected)


def test_unrecognised_failures_are_left_alone():
    """A model typo is a caller error, not a provider outage -- don't relabel it."""
    assert _classify_openai_error("openai", ValueError("bad response shape")) is None
    assert (
        _classify_openai_error("openai", _status_error(APIStatusError, 404, "model_not_found"))
        is None
    )


def test_research_raises_the_classified_error(monkeypatch):
    """The typed error reaches the caller instead of the raw SDK exception."""
    provider = _provider()

    class _FailingResponses:
        def create(self, **kwargs):
            raise _status_error(RateLimitError, 429, "insufficient_quota", "quota spent")

    class _FailingClient:
        responses = _FailingResponses()

    monkeypatch.setattr(
        "deep_research_client.providers.openai.OpenAI", lambda **kwargs: _FailingClient()
    )

    with pytest.raises(ProviderBillingError) as excinfo:
        asyncio.run(provider.research("what is a mitochondrion"))

    assert excinfo.value.retryable is False
    assert "out of credits" in str(excinfo.value)


def test_probe_reports_unreachable_on_bad_key(monkeypatch):
    """A rejected key is reported as unreachable, not as a crash."""
    provider = _provider()

    class _FailingModels:
        def list(self):
            raise _status_error(AuthenticationError, 401, "invalid_api_key")

    class _FailingClient:
        models = _FailingModels()

        def close(self):
            """Match the real client, which the probe closes."""

    monkeypatch.setattr(
        "deep_research_client.providers.openai.OpenAI", lambda **kwargs: _FailingClient()
    )

    health = asyncio.run(provider.check_health())

    assert health.reachable is False
    assert "401" in health.detail


def test_probe_says_what_it_cannot_prove(monkeypatch):
    """A passing probe must not be read as "the account has quota"."""
    provider = _provider()

    class _Models:
        def list(self):
            return []

    class _Client:
        models = _Models()

        def close(self):
            """Match the real client, which the probe closes."""

    monkeypatch.setattr(
        "deep_research_client.providers.openai.OpenAI", lambda **kwargs: _Client()
    )

    health = asyncio.run(provider.check_health())

    assert health.reachable is True
    assert "quota not readable" in health.detail


def test_unconfigured_provider_is_not_probed():
    """No key means no network call is worth making."""
    health = asyncio.run(_provider(api_key=None).check_health())

    assert health.configured is False
    assert health.reachable is False


def test_probe_returns_a_record_even_if_the_client_cannot_be_built(monkeypatch):
    """`check_health` promises a record, so construction failures return too."""
    def _bad_client(**kwargs):
        raise ValueError("Invalid base_url: 'not-a-url'")

    monkeypatch.setattr("deep_research_client.providers.openai.OpenAI", _bad_client)

    health = asyncio.run(_provider().check_health())

    assert health.reachable is False
    assert "base_url" in health.detail
