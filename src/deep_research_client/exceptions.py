"""Typed provider errors.

Providers fail for very different reasons, and callers need to tell those
reasons apart. An expired key, an account with no credits, a rate limit and a
flaky connection all arrive as exceptions, but only some of them are worth
retrying, and only some of them mean "use a different provider".

Without a type to branch on, a caller can only match substrings in a log line.
Worse, an SDK that retries internally can bury a permanent failure -- a
``402 Payment Required`` retried until the clock runs out looks exactly like a
slow provider, which sends the reader off debugging the wrong thing.

The hierarchy here gives that decision a name:

>>> err = classify_status("falcon", 402, "no credits")
>>> type(err).__name__
'ProviderBillingError'
>>> err.retryable
False

See https://github.com/monarch-initiative/deep-research-client/issues/65
"""

import re
from typing import ClassVar, Optional

__all__ = [
    "ProviderError",
    "ProviderAuthError",
    "ProviderBillingError",
    "ProviderRateLimitError",
    "ProviderTransientError",
    "classify_status",
    "classify_exception",
    "extract_status_code",
]


class ProviderError(ValueError):
    """Base class for a failure reported by a research provider.

    Subclasses :class:`ValueError` so that callers written against earlier
    releases -- which raised bare ``ValueError`` for HTTP failures -- keep
    working while new callers branch on the specific type.

    Args:
        provider: Name of the provider that failed.
        detail: Provider-supplied description of the failure.
        status_code: HTTP status code, when the failure came from an HTTP call.
    """

    #: Whether retrying the same call against the same provider could succeed.
    retryable: ClassVar[bool] = False

    #: Short imperative hint appended to the message, e.g. "top up the account".
    remedy: ClassVar[str] = "check the provider configuration"

    def __init__(self, provider: str, detail: str, status_code: Optional[int] = None):
        """Build a provider error carrying its own remediation advice."""
        self.provider = provider
        self.detail = detail
        self.status_code = status_code
        super().__init__(self.actionable_message())

    @property
    def diagnosis(self) -> str:
        """Describe the failure without naming the provider or suggesting a fix.

        Useful where the provider is already named by the surrounding context,
        such as a health report line.

        Returns:
            Status, provider detail, and what it means

        >>> ProviderBillingError("falcon", "no credits", 402).diagnosis
        '402 no credits -- the account is out of credits'
        """
        status = f"{self.status_code} " if self.status_code else ""
        return f"{status}{self.detail} -- {self.remedy}"

    def actionable_message(self) -> str:
        """Render a message that tells the reader what to do next.

        Returns:
            Message naming the provider, the cause, and a suggested next step.

        >>> print(ProviderBillingError("falcon", "no credits", 402).actionable_message())
        falcon: 402 no credits -- the account is out of credits. Try: `deep-research-client providers --check`, or re-run with --provider <other>
        """
        return (
            f"{self.provider}: {self.diagnosis}. "
            f"Try: `deep-research-client providers --check`, or re-run with --provider <other>"
        )


class ProviderAuthError(ProviderError):
    """The credential was missing, malformed, expired, or lacks access (401/403)."""

    remedy = "the API key is missing, invalid, or lacks access to this endpoint"


class ProviderBillingError(ProviderError):
    """The account cannot pay for the call (402).

    A ``402`` reports a *status*, not a balance: the provider says "not now",
    it does not say how many credits remain. Any numeric balance has to come
    from the provider's own dashboard.
    """

    remedy = "the account is out of credits"


class ProviderRateLimitError(ProviderError):
    """The provider is throttling us (429). Worth retrying, after a wait."""

    retryable = True
    remedy = "the rate limit was exceeded; wait and retry"


class ProviderTransientError(ProviderError):
    """The provider failed in a way that may resolve on its own (5xx)."""

    retryable = True
    remedy = "the provider reported a temporary server-side failure"


#: HTTP status codes that mean "stop; a different provider or credential is needed".
_STATUS_MAP: dict[int, type[ProviderError]] = {
    401: ProviderAuthError,
    402: ProviderBillingError,
    403: ProviderAuthError,
    429: ProviderRateLimitError,
}

# httpx renders failures as: Client error '402 Payment Required' for url '...'
_QUOTED_STATUS = re.compile(r"'(\d{3})\s+[A-Za-z]")
# Fallback for SDKs that write the code out in prose.
_PROSE_STATUS = re.compile(r"(?:HTTP|status(?:\s+code)?)\D{0,3}(\d{3})", re.IGNORECASE)


def classify_status(
    provider: str, status_code: int, detail: str
) -> Optional[ProviderError]:
    """Map an HTTP status code onto a typed provider error.

    Args:
        provider: Name of the provider that returned the status.
        status_code: HTTP status code from the failed call.
        detail: Provider-supplied description of the failure.

    Returns:
        The matching error, or None when the status is not one we classify.

    >>> type(classify_status("consensus", 401, "bad key")).__name__
    'ProviderAuthError'
    >>> type(classify_status("falcon", 503, "upstream down")).__name__
    'ProviderTransientError'
    >>> classify_status("falcon", 404, "no such task") is None
    True
    """
    error_class = _STATUS_MAP.get(status_code)
    if error_class is None:
        if 500 <= status_code < 600:
            error_class = ProviderTransientError
        else:
            return None
    return error_class(provider, detail, status_code)


def extract_status_code(exc: BaseException) -> Optional[int]:
    """Dig an HTTP status code out of an exception raised by a provider SDK.

    SDKs bury the status in different places, and a retry wrapper buries it
    another layer down. This walks the response object, the exception chain,
    and finally the rendered message.

    Args:
        exc: The exception raised by the provider SDK.

    Returns:
        The status code, or None when none can be found.

    >>> extract_status_code(RuntimeError("Client error '402 Payment Required' for url 'x'"))
    402
    >>> extract_status_code(RuntimeError("HTTP 503 from upstream"))
    503
    >>> extract_status_code(RuntimeError("connection reset")) is None
    True
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        response = getattr(current, "response", None)
        code = getattr(response, "status_code", None) or getattr(current, "status_code", None)
        if isinstance(code, int):
            return code

        # tenacity.RetryError hides the real failure behind the final attempt.
        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None and getattr(last_attempt, "failed", False):
            nested = last_attempt.exception()
            if nested is not None and id(nested) not in seen:
                return extract_status_code(nested)

        for pattern in (_QUOTED_STATUS, _PROSE_STATUS):
            match = pattern.search(str(current))
            if match:
                return int(match.group(1))

        current = current.__cause__ or current.__context__

    return None


def classify_exception(provider: str, exc: BaseException) -> Optional[ProviderError]:
    """Classify an arbitrary provider SDK exception, if we recognise it.

    Args:
        provider: Name of the provider that raised.
        exc: The exception raised by the provider SDK.

    Returns:
        A typed error to raise in its place, or None to let the original
        exception propagate untouched.

    >>> exc = RuntimeError("Client error '402 Payment Required' for url 'x'")
    >>> type(classify_exception("falcon", exc)).__name__
    'ProviderBillingError'
    >>> classify_exception("falcon", ValueError("malformed response")) is None
    True
    """
    if isinstance(exc, ProviderError):
        return exc
    status_code = extract_status_code(exc)
    if status_code is None:
        return None
    return classify_status(provider, status_code, str(exc))
