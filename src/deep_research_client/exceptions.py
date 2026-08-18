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

#: Cap on how much provider-supplied text is allowed into an exception message.
#: A 5xx HTML page or a Node stack trace is a hint, not something to reprint.
MAX_DETAIL_CHARS = 200

#: However long the remedy is, keep at least this much of what the provider
#: said -- a message that is all advice and no evidence helps nobody. Every
#: remedy in this module is short enough that this floor is unreachable; it
#: exists for a subclass whose remedy is longer than expected, and entering it
#: means the composed diagnosis can exceed MAX_DETAIL_CHARS.
MIN_DETAIL_CHARS = 60

#: Cap on a provider-reported reset time. The value lands inside a remedy, so an
#: unbounded one would inflate the framing until the budget could not protect
#: the reset time itself. The floor stays unreachable up to about 85 characters
#: of reset text; 60 leaves headroom while still fitting an ISO-8601 timestamp
#: with a zone name.
MAX_RESET_CHARS = 60

__all__ = [
    "MAX_DETAIL_CHARS",
    "MIN_DETAIL_CHARS",
    "MAX_RESET_CHARS",
    "ProviderError",
    "ProviderAuthError",
    "ProviderBillingError",
    "ProviderQuotaError",
    "ProviderNotConfiguredError",
    "ProviderNotInstalledError",
    "ProviderRateLimitError",
    "ProviderTransientError",
    "truncate_detail",
    "classify_status",
    "classify_exception",
    "extract_status_code",
]


def truncate_detail(text: str, limit: int) -> str:
    """Shorten text to a limit, marking the cut so a reader can see it happened.

    Args:
        text: The text to shorten.
        limit: Maximum length of the result, including the marker.

    Returns:
        The text, with a trailing ellipsis if anything was removed.

    >>> truncate_detail("short", 20)
    'short'
    >>> truncate_detail("a much longer sentence than the limit allows", 20)
    'a much longer sente…'
    """
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "\u2026"


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

    #: What the failure means and what would fix it. Usually a class-level
    #: constant, but an instance may sharpen it with provider-supplied detail.
    remedy: str = "check the provider configuration"

    def __init__(self, provider: str, detail: str, status_code: Optional[int] = None):
        """Build a provider error carrying its own remediation advice."""
        self.provider = provider
        self.status_code = status_code
        # Capped here so every construction site inherits it -- an SDK's
        # exception text or an HTML error page is a hint, not a payload.
        #
        # The budget subtracts the framing `diagnosis` will add, so the whole
        # composed line stays within the cap and the remedy -- the part that
        # says what to do about it -- is never the piece that gets cut.
        framing = len(self._render_diagnosis(""))
        budget = max(MIN_DETAIL_CHARS, MAX_DETAIL_CHARS - framing)
        self.detail = truncate_detail(detail, budget)
        super().__init__(self.actionable_message())

    def _render_diagnosis(self, detail: str) -> str:
        """Compose the diagnosis line around a given detail.

        Kept separate so ``__init__`` can measure the framing this adds by
        rendering it empty, rather than by re-deriving the format by hand.

        Args:
            detail: The provider-supplied text to frame.

        Returns:
            Status, detail, and what it means
        """
        status = f"{self.status_code} " if self.status_code else ""
        return f"{status}{detail} -- {self.remedy}"

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
        return self._render_diagnosis(self.detail)

    def __reduce__(self) -> tuple:
        """Rebuild through ``__init__`` so the error survives pickling.

        The base ``ValueError.__reduce__`` would replay only the rendered
        message, which this class does not accept as its sole argument.

        Returns:
            Callable and arguments that reconstruct an equal error
        """
        return (type(self), (self.provider, self.detail, self.status_code))

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


class ProviderQuotaError(ProviderError):
    """A plan's usage allowance is spent.

    Distinct from :class:`ProviderBillingError`, where the remedy is to pay,
    and from :class:`ProviderRateLimitError`, which clears in seconds. A spent
    allowance clears when the plan's window rolls over, which is usually hours
    away -- so retrying the same call is pointless, but the wait is bounded and
    the provider often says when it ends.

    Args:
        provider: Name of the provider that failed.
        detail: Provider-supplied description of the failure.
        status_code: HTTP status code, when the failure came from an HTTP call.
        resets_at: Provider-reported time the allowance renews, if it says.

    >>> print(ProviderQuotaError("claude_code", "usage limit reached", resets_at="3pm"))
    claude_code: usage limit reached -- the plan's usage limit is spent, and renews at 3pm. Try: `deep-research-client providers --check`, or re-run with --provider <other>
    """

    remedy = "the plan's usage limit is spent"

    def __init__(
        self,
        provider: str,
        detail: str,
        status_code: Optional[int] = None,
        resets_at: Optional[str] = None,
    ):
        """Build a quota error, noting when the allowance renews if known."""
        # Bounded here rather than by the caller: this class is public, and a
        # long reset string would inflate the remedy until the message budget
        # could no longer keep the reset time on the line.
        resets_at = truncate_detail(resets_at, MAX_RESET_CHARS) if resets_at else resets_at
        self.resets_at = resets_at
        if resets_at:
            self.remedy = f"{type(self).remedy}, and renews at {resets_at}"
        super().__init__(provider, detail, status_code)

    def __reduce__(self) -> tuple:
        """Rebuild through ``__init__``, keeping the reset time.

        Returns:
            Callable and arguments that reconstruct an equal error
        """
        return (type(self), (self.provider, self.detail, self.status_code, self.resets_at))


class ProviderNotConfiguredError(ProviderError):
    """The provider cannot be used until someone sets it up.

    Distinct from an auth failure: nothing was rejected, because nothing was
    sent. Callers that want to skip a provider rather than diagnose it can
    catch this one class and cover both a missing key and a missing CLI.
    """

    remedy = "the provider is not configured"


class ProviderNotInstalledError(ProviderNotConfiguredError):
    """A provider backed by a local command-line tool has no tool to run.

    No credential fixes this one, which is why it is not an auth failure.
    """

    remedy = "the required command-line tool is not installed or not on PATH"


class ProviderRateLimitError(ProviderError):
    """The provider is throttling us (429). Worth retrying, after a wait."""

    retryable = True
    remedy = "the rate limit was exceeded; wait and retry"


class ProviderTransientError(ProviderError):
    """The provider failed in a way that may resolve on its own (5xx)."""

    retryable = True
    remedy = "the provider reported a temporary server-side failure"


#: Statuses we have a specific type for. Not all of them are terminal -- 429 is
#: retryable -- but each one implies a different remedy.
_STATUS_MAP: dict[int, type[ProviderError]] = {
    401: ProviderAuthError,
    402: ProviderBillingError,
    403: ProviderAuthError,
    429: ProviderRateLimitError,
}

# httpx renders failures as: Client error '402 Payment Required' for url '...'
_QUOTED_STATUS = re.compile(r"'([1-5]\d{2})\s+[A-Za-z]")
# Fallback for SDKs that write the code out in prose. The separator excludes
# slashes so that a URL ("http://127.0.0.1/...") cannot pass its first octet off
# as a status, and the value must look like one.
_PROSE_STATUS = re.compile(
    r"(?:HTTP|status(?:\s+code)?)[\s:=-]{0,3}([1-5]\d{2})\b(?!\.\d)", re.IGNORECASE
)


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
    >>> extract_status_code(RuntimeError("connecting to http://127.0.0.1/v1")) is None
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
                nested_code = extract_status_code(nested)
                if nested_code is not None:
                    return nested_code

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
