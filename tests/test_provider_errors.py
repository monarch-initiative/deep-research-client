"""Tests for typed provider errors and the provider health probe.

Covers the failure mode from issue #65: a provider that is configured but
cannot take work, reported as a timeout instead of as a billing failure.
"""

import httpx
import pytest

from deep_research_client.exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTransientError,
    classify_exception,
    classify_status,
    extract_status_code,
)
from deep_research_client.models import ProviderConfig, ProviderHealth
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
