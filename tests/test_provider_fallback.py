"""Tests for opt-in provider fallback and the provenance it records.

See https://github.com/monarch-initiative/deep-research-client/issues/69

Nothing here is mocked. The failures are produced by the shipped
:class:`MockProvider`, which raises the same typed errors a real provider
raises, so the client is exercised through its ordinary code path.
"""

import json
import logging

import pytest
from typer.testing import CliRunner

from deep_research_client import cli as cli_module
from deep_research_client.client import DeepResearchClient
from deep_research_client.exceptions import (
    FALLBACK_WORTHY_ERRORS,
    ProviderAuthError,
    ProviderBillingError,
    ProviderNotConfiguredError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderTransientError,
    is_fallback_worthy,
)
from deep_research_client.formatter import ResultFormatter as LegacyResultFormatter
from deep_research_client.processing import ResultFormatter
from deep_research_client.models import (
    CacheConfig,
    ProviderAttempt,
    ProviderConfig,
    ResearchResult,
)
from deep_research_client.processing import ResearchProcessor
from deep_research_client.provider_params import MockParams
from deep_research_client.providers.mock import _SIMULATED_ERRORS, MockProvider

PRIMARY = "mock"
BACKUP = "mock_backup"
SPARE = "mock_spare"


class StandInProvider(MockProvider):
    """A mock standing in for a provider that does real research.

    MockProvider itself is kept out of the *automatic* fallback ordering,
    because a run that ran out of credits should fail rather than quietly hand
    a curation record an invented report. Testing that ordering therefore needs
    a provider that is not excluded from it -- so this subclass says so, rather
    than the tests reaching around the rule they are meant to exercise.
    """

    produces_real_reports = True


class UnclassifiableFailure(StandInProvider):
    """A provider whose SDK error nothing recognises.

    `openai.py` re-raises bare what `_classify_openai_error` cannot place, so
    an exception with no `provider_attempts` field really does reach the loop.
    """

    async def research(self, query):
        raise RuntimeError("connection reset by peer")


# The CLI's _setup_logging calls setLevel on the *package* logger, which is
# process-global and was never restored -- so a CLI test anywhere in the suite
# left every later test running under whatever level it chose. That is why the
# INFO assertions here passed run alone and failed in the full suite.
_PACKAGE_LOGGER = "deep_research_client"

# Asserting on records still names the child explicitly, and the fixture below
# does not make that unnecessary -- dropping it fails four tests. The fixture
# stops *this* file leaking onward; it cannot undo a level another file already
# pinned before these tests ran. caplog.at_level with no logger raises the
# *root* level, which the package logger's level is consulted before, so an
# INFO record is dropped on the way. Setting a non-NOTSET level on the child
# stops the effective-level walk there instead, whatever the parent holds.
_CLIENT_LOGGER = "deep_research_client.client"


@pytest.fixture(autouse=True)
def _restore_package_log_level():
    """Stop this file's CLI tests leaving a level behind for whoever runs next.

    Containing the leak at its source beats working around it downstream,
    though it only covers tests that run after these -- see _CLIENT_LOGGER for
    the half that protects against pollution arriving from elsewhere.

    The level only. _setup_logging also calls basicConfig(force=True), which
    removes and closes every root handler, and that is deliberately out of
    scope here: restoring closed handlers is not something a fixture can do
    honestly. So this narrows the leak rather than closing it.
    """
    package_logger = logging.getLogger(_PACKAGE_LOGGER)
    original = package_logger.level
    yield
    package_logger.setLevel(original)


def _params(**kwargs) -> MockParams:
    """Build mock parameters with no artificial delay."""
    kwargs.setdefault("response_delay", 0.0)
    return MockParams(**kwargs)


def _client(*named_params, cache_dir=None) -> DeepResearchClient:
    """Build a client whose registry holds exactly the given mock providers.

    Args:
        named_params: ``(name, MockParams)`` pairs, in registration order.
        cache_dir: Enable the on-disk cache in this directory when given.

    Returns:
        A client with no provider except the ones named.
    """
    cache_config = CacheConfig(
        enabled=cache_dir is not None,
        directory=str(cache_dir) if cache_dir else None,
    )
    # Passing provider_configs skips environment detection entirely, so the
    # registry cannot pick up whatever happens to be configured on this machine.
    client = DeepResearchClient(
        cache_config=cache_config,
        provider_configs={PRIMARY: ProviderConfig(name=PRIMARY)},
    )
    for name, params in named_params:
        client.registry.register(StandInProvider(ProviderConfig(name=name), params))
    return client


# --------------------------------------------------------------------------
# Which failures justify switching provider
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error, expected",
    [
        (ProviderAuthError("p", "bad key", 401), True),
        (ProviderBillingError("p", "no credits", 402), True),
        (ProviderQuotaError("p", "limit spent"), True),
        (ProviderNotConfiguredError("p", "no key"), True),
        (ProviderRateLimitError("p", "slow down", 429), False),
        (ProviderTransientError("p", "bad gateway", 502), False),
        (ValueError("malformed response"), False),
        (RuntimeError("boom"), False),
    ],
)
def test_only_provider_specific_failures_justify_a_switch(error, expected):
    """A switch needs evidence that *this* provider cannot do the work."""
    assert is_fallback_worthy(error) is expected


def test_retryable_errors_are_never_fallback_worthy():
    """The two axes must not be confused: wait-and-retry is not switch-provider."""
    for error_class in FALLBACK_WORTHY_ERRORS:
        assert error_class.retryable is False


# --------------------------------------------------------------------------
# The mock provider's typed failures
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error_type, expected_class",
    [
        ("auth", ProviderAuthError),
        ("billing", ProviderBillingError),
        ("quota", ProviderQuotaError),
        ("rate_limit", ProviderRateLimitError),
        ("transient", ProviderTransientError),
        ("not_configured", ProviderNotConfiguredError),
    ],
)
def test_mock_provider_raises_the_named_failure(error_type, expected_class):
    """Each error_type produces the real exception class it names."""
    client = _client((PRIMARY, _params(error_type=error_type)))
    with pytest.raises(expected_class):
        client.research("q")


def test_the_mocks_quota_failure_has_the_shape_a_real_one_has():
    """The mock exists to show real behaviour, so it must not invent a shape.

    A real spent allowance is recognised from the CLI's own wording and carries
    no status code. Giving the mock a 429 made its report entry disagree with
    every real one -- and with the documented rule that a 429 means wait for
    the same provider rather than switch, which is a different error class.

    Compared against the real classifier rather than a hardcoded shape, so the
    two cannot drift apart.
    """
    from deep_research_client.providers import claude_code

    real = claude_code._classify_cli_failure(
        "claude_code", "Usage limit reached. Your limit will reset at 3pm."
    )
    assert isinstance(real, ProviderQuotaError)

    error_class, status_code, extra = _SIMULATED_ERRORS["quota"]
    simulated = error_class("mock", "simulated quota failure", status_code, **extra)

    real_entry = ProviderAttempt.from_exception("x", real).frontmatter_entry()
    mock_entry = ProviderAttempt.from_exception("x", simulated).frontmatter_entry()
    assert sorted(real_entry) == sorted(mock_entry)
    assert "status_code" not in mock_entry


def test_typed_error_wins_over_the_generic_one():
    """Answering with the vaguer error would throw information away."""
    client = _client((PRIMARY, _params(error_type="billing", include_error=True)))
    with pytest.raises(ProviderBillingError):
        client.research("q")


# --------------------------------------------------------------------------
# Fallback is opt-in
# --------------------------------------------------------------------------


def test_no_fallback_by_default():
    """Without opting in, a failure propagates even though a backup exists."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    with pytest.raises(ProviderBillingError):
        client.research("q", provider=PRIMARY)


def test_default_run_records_who_produced_it_without_claiming_a_fallback():
    """A single successful provider is recorded, but is not a fallback."""
    client = _client((PRIMARY, _params()))
    result = client.research("q", provider=PRIMARY)
    assert result.provider == PRIMARY
    assert result.fell_back is False
    assert [(a.provider, a.succeeded) for a in result.provider_attempts] == [(PRIMARY, True)]


def test_fallback_uses_the_next_available_provider():
    """Opting in lets a second provider produce the report."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    result = client.research("q", provider=PRIMARY, fallback=True)
    assert result.provider == BACKUP
    assert result.fell_back is True


def test_fallback_records_what_was_asked_for_and_what_was_tried():
    """The trail must name the requested provider and why it was not used."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    result = client.research("q", provider=PRIMARY, fallback=True)

    assert result.requested_provider == PRIMARY
    assert [(a.provider, a.succeeded) for a in result.provider_attempts] == [
        (PRIMARY, False),
        (BACKUP, True),
    ]
    failure = result.provider_attempts[0]
    assert failure.error_type == "ProviderBillingError"
    assert failure.retryable is False
    assert "out of credits" in failure.reason


@pytest.mark.parametrize("error_type", ["rate_limit", "transient"])
def test_a_wait_and_retry_failure_does_not_switch_provider(error_type):
    """A throttle or a 5xx says wait, not switch -- even with fallback on."""
    client = _client(
        (PRIMARY, _params(error_type=error_type)),
        (BACKUP, _params()),
    )
    with pytest.raises((ProviderRateLimitError, ProviderTransientError)):
        client.research("q", provider=PRIMARY, fallback=True)


def test_an_unclassified_failure_does_not_switch_provider():
    """An unexplained failure is not evidence that another provider would do."""
    client = _client(
        (PRIMARY, _params(include_error=True)),
        (BACKUP, _params()),
    )
    with pytest.raises(ValueError, match="simulated API error"):
        client.research("q", provider=PRIMARY, fallback=True)


# --------------------------------------------------------------------------
# Ordering and explicit lists
# --------------------------------------------------------------------------


def test_an_explicit_list_sets_the_order():
    """A named list is an instruction, not a hint."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params(error_type="auth")),
        (SPARE, _params()),
    )
    result = client.research("q", provider=PRIMARY, fallback=[BACKUP, SPARE])
    assert result.provider == SPARE
    assert [a.provider for a in result.provider_attempts] == [PRIMARY, BACKUP, SPARE]


def test_an_explicit_list_can_skip_an_available_provider():
    """Naming providers replaces the automatic ordering rather than extending it."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
        (SPARE, _params()),
    )
    result = client.research("q", provider=PRIMARY, fallback=[SPARE])
    assert result.provider == SPARE
    assert BACKUP not in [a.provider for a in result.provider_attempts]


def test_a_named_but_unconfigured_provider_is_recorded_not_silently_dropped():
    """Every name in an explicit list gets an answer in the trail."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    result = client.research("q", provider=PRIMARY, fallback=["openai", BACKUP])

    assert result.provider == BACKUP
    skipped = result.provider_attempts[1]
    assert skipped.provider == "openai"
    assert skipped.error_type == "ProviderNotConfiguredError"
    assert skipped.reason


def test_fallback_without_a_named_provider_starts_from_the_first_available():
    """Letting the client choose still falls back from whatever it chose."""
    client = _client(
        (PRIMARY, _params(error_type="quota")),
        (BACKUP, _params()),
    )
    result = client.research("q", fallback=True)
    assert result.provider == BACKUP
    assert result.requested_provider is None
    assert result.fell_back is True


# --------------------------------------------------------------------------
# Exhaustion and fail-closed
# --------------------------------------------------------------------------


def test_the_last_failure_propagates_when_everything_fails():
    """Exhausting every candidate raises rather than returning a partial result."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params(error_type="auth")),
    )
    with pytest.raises(ProviderAuthError):
        client.research("q", provider=PRIMARY, fallback=True)


def test_exhausting_every_candidate_carries_the_whole_trail_out():
    """With no result there is no `provider_attempts` -- so it goes on the error.

    That is the case a caller most wants it: three providers were tried and
    the exception is all they get. The trail includes the candidate whose
    failure is being raised, so it describes the run rather than everything
    before its end.
    """
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params(error_type="auth")),
        (SPARE, _params(error_type="quota")),
    )
    with pytest.raises(ProviderQuotaError) as caught:
        client.research("q", provider=PRIMARY, fallback=[BACKUP, SPARE])

    assert [a.provider for a in caught.value.provider_attempts] == [
        PRIMARY, BACKUP, SPARE
    ]
    assert [a.error_type for a in caught.value.provider_attempts] == [
        "ProviderBillingError", "ProviderAuthError", "ProviderQuotaError"
    ]


def test_the_trail_does_not_take_the_cause_a_provider_set_itself():
    """Three shipped providers do `raise classified from e`; that must survive.

    No other test can see this: every provider in this file raises bare, so
    `__cause__` is always free to take. Chaining the trail onto it displaced
    the upstream SDK error to `__context__`, where `__suppress_context__`
    hides it from the printed traceback -- losing the one frame an operator
    needs to see what the API actually said.
    """

    class ChainsItsOwnCause(StandInProvider):
        """A provider that classifies an SDK failure, as the real ones do."""

        async def research(self, query):
            try:
                raise RuntimeError("upstream SDK said 402")
            except RuntimeError as sdk_error:
                raise ProviderBillingError(self.name, "no credits", 402) from sdk_error

    client = _client((PRIMARY, _params(error_type="auth")))
    client.registry.register(ChainsItsOwnCause(ProviderConfig(name=BACKUP), _params()))

    with pytest.raises(ProviderBillingError) as caught:
        client.research("q", provider=PRIMARY, fallback=[BACKUP])

    # The provider's own `raise ... from` also sets __suppress_context__, so
    # __cause__ is the assertion that matters: it is still the SDK error and
    # not the previous candidate.
    assert isinstance(caught.value.__cause__, RuntimeError)
    # And the trail is still carried, on its own attribute.
    assert [a.provider for a in caught.value.provider_attempts] == [PRIMARY, BACKUP]


def test_an_unclassified_terminal_failure_logs_the_trail_instead(caplog):
    """A failure we did not classify has nowhere to carry the trail.

    `openai.py` re-raises bare what `_classify_openai_error` cannot recognise,
    so a plain SDK error really does reach the loop -- and a plain exception
    has no field for this. Reading `err.provider_attempts` there is an
    AttributeError, not an empty trail, so the run says it in the log instead
    of losing it.
    """

    client = _client((PRIMARY, _params(error_type="billing")))
    client.registry.register(UnclassifiableFailure(ProviderConfig(name=BACKUP), _params()))

    with caplog.at_level("WARNING", logger=_CLIENT_LOGGER):
        with pytest.raises(RuntimeError) as caught:
            client.research("q", provider=PRIMARY, fallback=[BACKUP])

    # Nothing to attach it to, so the caller must not be told there is.
    assert not hasattr(caught.value, "provider_attempts")
    messages = [r.getMessage() for r in caplog.records]
    trail = [m for m in messages if "Providers tried" in m]
    assert len(trail) == 1
    assert PRIMARY in trail[0] and BACKUP in trail[0]


def test_an_unclassified_failure_before_the_last_candidate_names_itself(caplog):
    """An unclassified failure ends the run wherever it happens.

    It is not fallback-worthy, so the loop stops even with candidates behind
    it -- and the log must not tell an operator that every provider failed
    when the last one was never reached.
    """

    client = _client((PRIMARY, _params(error_type="billing")), (SPARE, _params()))
    client.registry.register(UnclassifiableFailure(ProviderConfig(name=BACKUP), _params()))

    with caplog.at_level("WARNING", logger=_CLIENT_LOGGER):
        with pytest.raises(RuntimeError):
            client.research("q", provider=PRIMARY, fallback=[BACKUP, SPARE])

    trail = [r.getMessage() for r in caplog.records if "Providers tried" in r.getMessage()]
    assert len(trail) == 1
    # The candidate that ended the run, not a claim about the whole list.
    assert f"Provider {BACKUP} failed with RuntimeError" in trail[0]
    # SPARE was never reached, so it must not appear as an attempt.
    assert SPARE not in trail[0]


def test_a_requested_fallback_with_nobody_to_switch_to_says_so(caplog):
    """A flag that changes nothing must not do it silently.

    The automatic ordering drops providers that invent their reports. When
    that leaves one candidate, the run fails exactly as it would with no flag
    at all -- and the trail is empty at position 0 either way, so nothing
    distinguishes it from "the fallback ran and also failed". The reason is
    the invisible part, so the run names it.
    """
    client = _client((PRIMARY, _params(error_type="billing")))
    # MockProvider, not StandInProvider: it is excluded from the ordering
    # precisely because it does not produce real reports.
    client.registry.register(MockProvider(ProviderConfig(name=BACKUP), _params()))

    with caplog.at_level("INFO", logger=_CLIENT_LOGGER):
        with pytest.raises(ProviderBillingError):
            client.research("q", provider=PRIMARY, fallback=True)

    said = [r.getMessage() for r in caplog.records if "only candidate" in r.getMessage()]
    assert len(said) == 1
    assert "do not produce real reports" in said[0]
    assert BACKUP in said[0]


def test_a_named_fallback_that_repeats_the_primary_says_so(caplog):
    """The other route to one candidate: the name deduplicates away."""
    client = _client((PRIMARY, _params(error_type="billing")), (BACKUP, _params()))

    with caplog.at_level("INFO", logger=_CLIENT_LOGGER):
        with pytest.raises(ProviderBillingError):
            client.research("q", provider=PRIMARY, fallback=[PRIMARY])

    said = [r.getMessage() for r in caplog.records if "only candidate" in r.getMessage()]
    assert len(said) == 1
    # The exact clause, to the sentence boundary: "no other provider was
    # named" is a prefix of "...named or available", so a substring check
    # would pass against the wrong branch. BACKUP is registered and available
    # here -- it simply was not named -- so that branch would be false.
    assert "candidate: no other provider was named." in said[0]


def test_the_automatic_route_says_available(caplog):
    """The automatic route is the one case where "available" is the truth.

    `fallback=True` takes whatever else is registered, so one candidate really
    does mean there is nobody else -- unlike the list route, where the usual
    cause is that nobody else was listed. Two situations, two sentences.
    """
    # Nobody else at all: _client passes provider_configs, so the registry
    # holds only what is named here.
    client = _client((PRIMARY, _params(error_type="billing")))

    with caplog.at_level("INFO", logger=_CLIENT_LOGGER):
        with pytest.raises(ProviderBillingError):
            client.research("q", provider=PRIMARY, fallback=True)

    said = [r.getMessage() for r in caplog.records if "only candidate" in r.getMessage()]
    assert len(said) == 1
    assert "candidate: no other provider is available." in said[0]


def test_the_reason_given_is_the_reason_that_applied(caplog):
    """Naming a cause that is not the cause is the bug this PR is about.

    The `produces_real_reports` filter runs only on the automatic route. An
    explicitly named provider that invents its reports is honoured -- so when
    a list route dedups to one candidate while a mock happens to sit in the
    registry, that mock was not excluded for fabricating. It simply was not
    named. Blaming the filter would tell an operator that their own
    `--fallback-provider mock_backup` would be dropped, which is the reverse
    of what happens.
    """
    client = _client((PRIMARY, _params(error_type="billing")))
    # Available, and not a real-report provider -- the bait for the wrong reason.
    client.registry.register(MockProvider(ProviderConfig(name=BACKUP), _params()))

    with caplog.at_level("INFO", logger=_CLIENT_LOGGER):
        with pytest.raises(ProviderBillingError):
            client.research("q", provider=PRIMARY, fallback=[PRIMARY])

    said = [r.getMessage() for r in caplog.records if "only candidate" in r.getMessage()]
    assert len(said) == 1
    assert "candidate: no other provider was named." in said[0]
    assert "do not produce real reports" not in said[0]
    assert BACKUP not in said[0]


def test_a_real_fallback_does_not_claim_it_had_nobody(caplog):
    """The diagnostic must not fire on the runs it is not about."""
    client = _client((PRIMARY, _params(error_type="billing")), (BACKUP, _params()))

    with caplog.at_level("INFO", logger=_CLIENT_LOGGER):
        result = client.research("q", provider=PRIMARY, fallback=[BACKUP])

    assert result.provider == BACKUP
    assert not [r for r in caplog.records if "only candidate" in r.getMessage()]


def test_the_log_says_the_run_ends_rather_than_leaving_it_open(caplog):
    """Not misleading is not the same as informing.

    Naming the candidate removed the false claim that every provider failed.
    It did not tell an operator what became of the candidates behind it, and
    `Providers tried:` reads as the whole list rather than a prefix of it --
    on this path the log is the entire record, since an unclassified failure
    carries no `provider_attempts`.
    """
    client = _client((PRIMARY, _params(error_type="billing")), (SPARE, _params()))
    client.registry.register(UnclassifiableFailure(ProviderConfig(name=BACKUP), _params()))

    with caplog.at_level("WARNING", logger=_CLIENT_LOGGER):
        with pytest.raises(RuntimeError):
            client.research("q", provider=PRIMARY, fallback=[BACKUP, SPARE])

    trail = [r.getMessage() for r in caplog.records if "Providers tried" in r.getMessage()]
    assert len(trail) == 1
    assert "The run ends here; any candidate after it was not tried." in trail[0]


def test_a_lone_failure_carries_no_trail():
    """One provider is not a trail, and must not read as one."""
    client = _client((PRIMARY, _params(error_type="billing")))
    with pytest.raises(ProviderBillingError) as caught:
        client.research("q", provider=PRIMARY)

    assert caught.value.provider_attempts == ()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_nothing_is_cached_when_every_provider_fails(tmp_path):
    """Fail-closed: no report exists, so no cache entry may either."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params(error_type="auth")),
        cache_dir=tmp_path,
    )
    with pytest.raises(ProviderAuthError):
        client.research("q", provider=PRIMARY, fallback=True)
    assert list(tmp_path.glob("*.json")) == []


def test_no_providers_at_all_is_still_a_plain_error():
    """An empty registry reports itself the same way it always did."""
    client = DeepResearchClient(
        cache_config=CacheConfig(enabled=False),
        provider_configs={PRIMARY: ProviderConfig(name=PRIMARY)},
    )
    client.registry._providers.clear()
    with pytest.raises(ValueError, match="No research providers available"):
        client.research("q", fallback=True)


def test_an_unknown_provider_name_is_still_a_plain_error():
    """A typo must not be reported as a configuration problem."""
    client = _client((PRIMARY, _params()))
    with pytest.raises(ValueError, match="not found"):
        client.research("q", provider="flacon", fallback=True)


# --------------------------------------------------------------------------
# Bad input in the fallback list
# --------------------------------------------------------------------------


def test_a_typo_in_the_fallback_list_fails_before_anything_is_called(caplog):
    """A name that is not a provider is caught up front, not mid-run.

    Asserting the message alone would prove nothing: the same ValueError
    surfaces either way, just later. What matters is that no provider was
    called first -- discovering the typo mid-run would waste that call, never
    reach the good candidate behind it, and replace the failure that started
    the fallback with a bare "not found".
    """
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    with caplog.at_level("WARNING", logger=_CLIENT_LOGGER):
        with pytest.raises(ValueError, match="'mok_backup' not found"):
            client.research("q", provider=PRIMARY, fallback=["mok_backup", BACKUP])

    # The primary logs a warning the moment it is called and fails, so silence
    # here is the evidence that nothing ran.
    assert [r.getMessage() for r in caplog.records] == []


def test_a_typo_is_caught_even_when_the_primary_would_have_succeeded():
    """Fail fast means fast: the primary is never called either."""
    client = _client((PRIMARY, _params()))
    with pytest.raises(ValueError, match="not found"):
        client.research("q", provider=PRIMARY, fallback=["nonesuch"])


def test_every_unknown_name_is_named_at_once():
    """Fixing one typo only to be told about the next is a poor trade."""
    client = _client((PRIMARY, _params()))
    with pytest.raises(ValueError, match="alpha, beta"):
        client.research("q", provider=PRIMARY, fallback=["alpha", "beta"])


def test_a_known_but_unconfigured_name_is_not_treated_as_a_typo():
    """The distinction the up-front check must preserve.

    ``openai`` is a real provider that simply has no credential here, so it
    belongs in the trail with its reason -- unlike a name that is not a
    provider at all.
    """
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    result = client.research("q", provider=PRIMARY, fallback=["openai", BACKUP])
    assert result.provider == BACKUP
    assert [a.provider for a in result.provider_attempts] == [PRIMARY, "openai", BACKUP]


def test_a_single_provider_name_is_not_read_as_a_list_of_letters():
    """``Sequence[str]`` accepts ``str``; ``list("openai")`` is six providers."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    result = client.research("q", provider=PRIMARY, fallback=BACKUP)
    assert result.provider == BACKUP
    assert [a.provider for a in result.provider_attempts] == [PRIMARY, BACKUP]


# --------------------------------------------------------------------------
# A provider that invents its reports is not a stand-in for one that does not
# --------------------------------------------------------------------------


def test_a_fabricating_provider_is_left_out_of_the_automatic_ordering():
    """The premise of the feature: a failed run beats an invented report."""
    client = _client((PRIMARY, _params(error_type="billing")))
    client.registry.register(MockProvider(ProviderConfig(name=BACKUP), _params()))

    assert BACKUP in [p.name for p in client.registry.get_available_providers()]
    with pytest.raises(ProviderBillingError):
        client.research("q", provider=PRIMARY, fallback=True)


def test_naming_a_fabricating_provider_still_honours_it():
    """Asking for it explicitly is a decision, not a stand-in nobody wanted."""
    client = _client((PRIMARY, _params(error_type="billing")))
    client.registry.register(MockProvider(ProviderConfig(name=BACKUP), _params()))

    result = client.research("q", provider=PRIMARY, fallback=[BACKUP])
    assert result.provider == BACKUP
    assert result.fell_back is True


def test_real_providers_declare_themselves_real_by_default():
    """The exclusion must be opt-in, or it would quietly empty the ordering."""
    from deep_research_client.providers import ResearchProvider

    assert ResearchProvider.produces_real_reports is True
    assert MockProvider.produces_real_reports is False


def test_a_fabricating_provider_can_still_be_chosen_directly():
    """Excluding it from fallback must not stop it doing its actual job."""
    client = DeepResearchClient(
        cache_config=CacheConfig(enabled=False),
        provider_configs={PRIMARY: ProviderConfig(name=PRIMARY)},
    )
    result = client.research("q", provider=PRIMARY)
    assert result.provider == PRIMARY


# --------------------------------------------------------------------------
# Caching must not inherit another run's provenance
# --------------------------------------------------------------------------


def test_the_cache_does_not_store_which_providers_were_tried(tmp_path):
    """Attempts describe one run; replaying them would credit a later run."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
        cache_dir=tmp_path,
    )
    result = client.research("q", provider=PRIMARY, fallback=True)
    assert result.fell_back is True

    cached_files = list(tmp_path.glob("*.json"))
    assert len(cached_files) == 1
    payload = json.loads(cached_files[0].read_text())
    assert payload["provider_attempts"] == []
    assert payload["requested_provider"] is None


def test_a_cache_hit_is_stamped_with_this_run_not_the_stored_one(tmp_path):
    """Reading from cache still records who was asked and who answered."""
    warm = _client((BACKUP, _params()), cache_dir=tmp_path)
    warm.research("q", provider=BACKUP)

    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
        cache_dir=tmp_path,
    )
    result = client.research("q", provider=PRIMARY, fallback=True)

    assert result.cached is True
    assert result.provider == BACKUP
    assert result.requested_provider == PRIMARY
    assert [(a.provider, a.succeeded) for a in result.provider_attempts] == [
        (PRIMARY, False),
        (BACKUP, True),
    ]


def test_a_cached_report_is_served_even_from_a_provider_we_cannot_reach(tmp_path, caplog):
    """Reading a report off disk needs no credential.

    On a fallback-worthy preparation failure, and only while another candidate
    remains, that candidate's cache is consulted before the next provider is
    called -- so a provider whose key has gone still answers from a report it
    produced earlier. Reachable because the CLI now stands its availability
    check down whenever a fallback was asked for.
    """
    # A falcon report is cached while falcon is configured.
    warm = _client(("falcon", _params()), cache_dir=tmp_path)
    warm.research("q", provider="falcon")

    # Later the credential is gone, so falcon cannot be prepared at all.
    later = _client((BACKUP, _params()), cache_dir=tmp_path)
    assert "falcon" not in [p.name for p in later.registry.get_available_providers()]

    with caplog.at_level("WARNING", logger=_CLIENT_LOGGER):
        result = later.research("q", provider="falcon", fallback=[BACKUP])

    assert result.provider == "falcon"
    assert result.cached is True
    # The run succeeds, but the revoked credential is still the operator's
    # news -- the next uncached query will fail. Nothing else on this path
    # records it: no attempt is appended, because the cached report really
    # was produced by falcon.
    messages = [r.getMessage() for r in caplog.records]
    assert any("falcon" in m and "not configured" in m for m in messages)
    # No fallback happened, so the report claims nothing it should not.
    assert result.fell_back is False
    assert [(a.provider, a.succeeded) for a in result.provider_attempts] == [
        ("falcon", True)
    ]


def test_without_a_fallback_an_unreachable_provider_still_raises(tmp_path):
    """The default path must not quietly gain cache-before-credential reads.

    The cache is consulted for a provider that cannot be prepared only while
    another candidate remains. With no fallback there is none, so the run fails
    exactly as it did before any of this -- which keeps the CLI and the library
    agreeing about what an unconfigured provider does.

    `ENABLE_MOCK_PROVIDER` still gates a cached mock report on *this* path.
    It does not on the fallback path; see the test below, which pins both.
    """
    warm = _client(("falcon", _params()), cache_dir=tmp_path)
    warm.research("q", provider="falcon")

    later = _client((BACKUP, _params()), cache_dir=tmp_path)
    with pytest.raises(ProviderNotConfiguredError):
        later.research("q", provider="falcon")


def test_naming_the_mock_reaches_its_cache_even_with_the_gate_off(tmp_path, monkeypatch):
    """What ENABLE_MOCK_PROVIDER does and does not gate, pinned on both paths.

    The gate keeps `mock` from being *chosen*, and `produces_real_reports`
    keeps it out of the automatic ordering. Neither is a rule about reading a
    report you already have from a provider you named yourself -- so with a
    fallback requested, the cached mock report is served, and says it came
    from mock.

    Recorded as a decision rather than left to be discovered: an earlier commit
    message of mine claimed the narrowing restored this gate outright, which is
    true only of the no-fallback path above.
    """
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    monkeypatch.setenv("DISABLE_CLAUDE_CODE_PROVIDER", "true")
    cache = CacheConfig(enabled=True, directory=str(tmp_path))
    DeepResearchClient(cache_config=cache).research("q", provider="mock")

    monkeypatch.delenv("ENABLE_MOCK_PROVIDER")
    client = DeepResearchClient(cache_config=cache)
    assert client.registry.get_provider("mock") is None

    with pytest.raises(ProviderNotConfiguredError):
        client.research("q", provider="mock")

    result = client.research("q", provider="mock", fallback=["openai"])
    assert result.provider == "mock"
    assert result.cached is True
    assert result.fell_back is False


def test_a_later_unreachable_candidate_also_answers_from_its_own_cache(tmp_path):
    """The cached-serve path runs at any non-last position, not only the first.

    At position > 0 the earlier failures are recorded and `fell_back` is true,
    but the serving candidate's own unreachability is not -- the same as at
    position 0. Pinned because nothing else exercises the interaction between
    that read and an already-populated trail.
    """
    # `falcon`, not a mock name: the middle candidate has to be known-but-
    # unregistered so it raises ProviderNotConfiguredError rather than being
    # rejected up front as a typo.
    warm = _client(("falcon", _params()), cache_dir=tmp_path)
    warm.research("q", provider="falcon")

    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (SPARE, _params()),
        cache_dir=tmp_path,
    )
    result = client.research("q", provider=PRIMARY, fallback=["falcon", SPARE])

    assert result.provider == "falcon"
    assert result.cached is True
    assert result.fell_back is True
    # The earlier failure is recorded; the server's own unreachability is not.
    assert [(a.provider, a.succeeded) for a in result.provider_attempts] == [
        (PRIMARY, False),
        ("falcon", True),
    ]


@pytest.mark.parametrize("no_fallback", [False, None], ids=["false", "none"])
def test_no_fallback_is_spelled_two_ways(no_fallback):
    """None arrives the same way the bare string did, and means the same as False.

    A wrapper passing `config.get("fallback")` for an absent key would
    otherwise reach `list(None)` and get a TypeError from inside the client
    naming nothing.
    """
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    with pytest.raises(ProviderBillingError):
        client.research("q", provider=PRIMARY, fallback=no_fallback)


def test_an_unreachable_provider_with_no_cache_still_falls_back(tmp_path):
    """The cache-first lookup must not swallow the fallback it sits in front of."""
    client = _client((BACKUP, _params()), cache_dir=tmp_path)
    result = client.research("uncached query", provider="falcon", fallback=[BACKUP])

    assert result.provider == BACKUP
    assert result.fell_back is True
    assert result.provider_attempts[0].error_type == "ProviderNotConfiguredError"


@pytest.mark.parametrize("warm_the_cache", [False, True], ids=["cold", "warm"])
def test_a_fallback_is_announced_whether_or_not_the_answer_was_cached(
    tmp_path, caplog, warm_the_cache
):
    """A cached answer is still someone else's answer.

    The two return paths are easy to let drift: the CLI reads `fell_back` so
    it warned on both, which is why nothing noticed that the library warned
    only on the cold one.
    """
    if warm_the_cache:
        _client((BACKUP, _params()), cache_dir=tmp_path).research("q", provider=BACKUP)

    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
        cache_dir=tmp_path,
    )
    with caplog.at_level("WARNING", logger=_CLIENT_LOGGER):
        result = client.research("q", provider=PRIMARY, fallback=True)

    assert result.cached is warm_the_cache
    assert result.fell_back is True
    announcements = [
        m for m in (r.getMessage() for r in caplog.records)
        if "not the provider first tried" in m
    ]
    assert len(announcements) == 1
    assert BACKUP in announcements[0] and PRIMARY in announcements[0]


def test_no_fallback_is_announced_when_none_happened(caplog):
    """The warning must not fire on an ordinary run."""
    client = _client((PRIMARY, _params()))
    with caplog.at_level("WARNING", logger=_CLIENT_LOGGER):
        client.research("q", provider=PRIMARY)

    assert not [
        r for r in caplog.records if "not the provider first tried" in r.getMessage()
    ]


# --------------------------------------------------------------------------
# Overrides chosen for one provider are not handed to another
# --------------------------------------------------------------------------


def test_a_fallback_provider_runs_on_its_own_defaults():
    """A model and parameters chosen for one provider mean nothing to another.

    Without this the run would die on the fallback provider's schema, since
    provider parameter models reject unknown fields.
    """
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    # Passing provider_params rebuilds the provider from them, so the failure
    # has to be requested here too -- the registered instance is replaced.
    result = client.research(
        "q",
        provider=PRIMARY,
        provider_params={
            "response_length": "short",
            "error_type": "billing",
            "response_delay": 0.0,
        },
        model="mock-model-v1",
        fallback=True,
    )
    assert result.provider == BACKUP
    # The backup's own default length, not the "short" asked of the primary.
    assert "medium-length mock response" in result.markdown


def test_dropping_overrides_is_said_out_loud(caplog):
    """Changing what was asked for is never silent."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    with caplog.at_level("WARNING", logger=_CLIENT_LOGGER):
        client.research(
            "q",
            provider=PRIMARY,
            provider_params={"error_type": "billing", "response_delay": 0.0},
            fallback=True,
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("own defaults" in message for message in messages)
    assert any("--param" in message for message in messages)


def test_the_dropped_override_warning_names_the_provider_that_got_them(caplog):
    """Past the first hop, the provider that just failed had no overrides either.

    Naming it would assert that parameters were applied to a provider which in
    fact ran on its own defaults -- in a message whose whole job is provenance.
    """
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params(error_type="auth")),
        (SPARE, _params()),
    )
    with caplog.at_level("WARNING", logger=_CLIENT_LOGGER):
        client.research(
            "q",
            provider=PRIMARY,
            provider_params={"error_type": "billing", "response_delay": 0.0},
            fallback=[BACKUP, SPARE],
        )

    dropped = [m for m in (r.getMessage() for r in caplog.records) if "own defaults" in m]
    assert len(dropped) == 2
    # Both hops must credit the primary, the only candidate that got them.
    assert all(f"applied to {PRIMARY}" in message for message in dropped)
    assert not any(f"applied to {BACKUP}" in message for message in dropped)


# --------------------------------------------------------------------------
# What the report says
# --------------------------------------------------------------------------


# Both formatters are live surfaces: the CLI writes reports through the
# processing one, while the top-level one remains importable. A fallback that
# only one of them admitted to would be worse than one neither mentioned.
FORMATTERS = [ResultFormatter, LegacyResultFormatter]


@pytest.mark.parametrize("formatter_class", FORMATTERS)
def test_frontmatter_reports_a_fallback(formatter_class):
    """A curator reading the report must see it came from someone else."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    result = client.research("q", provider=PRIMARY, fallback=True)
    frontmatter = formatter_class().format_full_markdown(result).split("---")[1]

    assert "fell_back: true" in frontmatter
    assert f"requested_provider: {PRIMARY}" in frontmatter
    assert "ProviderBillingError" in frontmatter


@pytest.mark.parametrize("formatter_class", FORMATTERS)
def test_frontmatter_is_unchanged_when_no_fallback_happened(formatter_class):
    """An ordinary report gains nothing: silence is the absence of a finding."""
    client = _client((PRIMARY, _params()))
    result = client.research("q", provider=PRIMARY)
    frontmatter = formatter_class().format_full_markdown(result).split("---")[1]

    assert "fell_back" not in frontmatter
    assert "provider_attempts" not in frontmatter
    assert "requested_provider" not in frontmatter


@pytest.mark.parametrize("formatter_class", FORMATTERS)
def test_the_report_never_carries_the_providers_own_error_text(formatter_class):
    """A 401 body is the one most likely to quote the credential it rejected.

    These reports get committed, so the frontmatter carries our reading of the
    failure, never the provider's prose. The full text stays on the object and
    in the logs, where it was before any of this reached disk.
    """
    secret = "sk-live-NOTAREALKEY123456"
    leaked = ProviderAuthError("falcon", f"Invalid API key: {secret}", 401)
    result = ResearchResult(
        markdown="body", provider=BACKUP, query="q", requested_provider="falcon",
        provider_attempts=[
            ProviderAttempt.from_exception("falcon", leaked),
            ProviderAttempt(provider=BACKUP, succeeded=True),
        ],
    )
    rendered = formatter_class().format_full_markdown(result)

    assert secret not in rendered
    assert "Invalid API key" not in rendered
    # What justifies the switch survives, in words we chose.
    assert "ProviderAuthError" in rendered
    assert "status_code: 401" in rendered
    assert "lacks access to this endpoint" in rendered
    # And the object still has the whole thing for a caller who wants it.
    assert secret in result.provider_attempts[0].reason


@pytest.mark.parametrize("formatter_class", FORMATTERS)
def test_a_quota_resets_at_does_not_reach_the_report(formatter_class):
    """The one error type that sharpens its own remedy with provider text.

    `_LIMIT_RESET` stops at a full stop or semicolon but deliberately keeps
    commas, so "10am, Tuesday 19 August" survives as one time -- and so does
    anything else comma-separated after it, up to the character cap.
    """
    spent = ProviderQuotaError(
        "claude_code", "usage limit reached", resets_at="10am, acct_9f3b21c7ee"
    )
    result = ResearchResult(
        markdown="body", provider=BACKUP, query="q", requested_provider="claude_code",
        provider_attempts=[
            ProviderAttempt.from_exception("claude_code", spent),
            ProviderAttempt(provider=BACKUP, succeeded=True),
        ],
    )
    rendered = formatter_class().format_full_markdown(result)

    assert "acct_9f3b21c7ee" not in rendered
    assert "renews at" not in rendered
    assert "the plan's usage limit is spent" in rendered
    # Still on the object, for whoever is actually debugging the outage.
    assert "acct_9f3b21c7ee" in result.provider_attempts[0].reason


def test_the_report_the_cli_writes_admits_the_fallback():
    """The processor is the path a saved report actually travels."""
    client = _client(
        (PRIMARY, _params(error_type="billing")),
        (BACKUP, _params()),
    )
    result = client.research("q", provider=PRIMARY, fallback=True)
    content = ResearchProcessor().format_research_result(result)

    assert "fell_back: true" in content
    assert f"requested_provider: {PRIMARY}" in content


# --------------------------------------------------------------------------
# End to end through the CLI
# --------------------------------------------------------------------------


def test_cli_reaches_the_named_fallback_provider(tmp_path, monkeypatch):
    """The flags are wired end to end, from command line to the second provider.

    ``deeper_med`` is the honest target here: it is a registered stub with no
    upstream service, so it is always unavailable and always explains itself.
    That makes it a real second candidate rather than a contrived one.
    """
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    monkeypatch.setenv("DISABLE_CLAUDE_CODE_PROVIDER", "true")
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "research", "q",
            "--provider", "mock",
            "--param", "error_type=billing",
            "--param", "response_delay=0.0",
            "--fallback-provider", "deeper_med",
            "--output", str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 1
    # The trail is visible to whoever is watching the run, not just in a file.
    assert "Falling back to deeper_med" in result.output
    assert "out of credits" in result.output
    # And the stub's own explanation is what ends the run, not the mock's.
    assert "no public API or code release yet" in result.output
    # Fail-closed: every candidate failed, so no report was written.
    assert not output.exists()


def _cli_client_with(*named_params):
    """Build a substitute for the client the research command constructs.

    The command builds its own client from the environment, and no environment
    a test can arrange gives it two *available* providers: `mock` is the only
    one needing no credential, and the rest would make real network calls.

    Nothing is mocked in the `unittest.mock` sense -- the CLI, the client, the
    providers, the formatter and the file write all run for real; only the
    wiring between two of them is redirected.

    `provider_configs` is accepted and deliberately not forwarded: passing our
    own skips environment detection, so whatever this machine happens to have
    configured cannot join the ordering. Without that, an early version of
    these tests found the `claude` CLI on PATH and made a real subprocess call.

    Args:
        named_params: ``(name, MockParams)`` pairs to register, in order.

    Returns:
        A callable matching the constructor call the research command makes.
    """
    real_client_class = cli_module.DeepResearchClient

    def _factory(cache_config=None, provider_configs=None):
        client = real_client_class(
            cache_config=cache_config,
            provider_configs={PRIMARY: ProviderConfig(name=PRIMARY)},
        )
        for name, params in named_params:
            client.registry.register(StandInProvider(ProviderConfig(name=name), params))
        return client

    return _factory


def test_cli_reports_the_trail_on_a_successful_fallback(tmp_path, monkeypatch):
    """The CLI's own fallback output, which no other test reaches.

    `test_cli_reaches_the_named_fallback_provider` ends in exit code 1, so the
    `if result.fell_back:` branch never runs there -- deleting it would fail
    nothing. See `_cli_client_with` for why the client is substituted.
    """
    monkeypatch.setattr(
        cli_module,
        "DeepResearchClient",
        _cli_client_with((PRIMARY, _params(error_type="billing")), (BACKUP, _params())),
    )
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "research", "q",
            "--provider", PRIMARY,
            "--fallback",
            "--output", str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0
    # The client states the switch; the CLI adds the trail. Neither repeats
    # the other, which is what the de-duplication was for.
    assert "not the provider first tried" in result.output
    assert "Providers tried:" in result.output
    assert f"{BACKUP}: produced the report" in result.output
    # The console keeps the provider's own words; the report does not.
    assert "simulated billing failure" in result.output
    content = output.read_text()
    assert "fell_back: true" in content
    assert "simulated billing failure" not in content


def test_cli_falls_back_when_the_named_provider_is_not_configured(tmp_path, monkeypatch):
    """The row the docs list, on the surface most people use.

    The command checks the named provider against the registry before calling
    the client at all. That check has to stand down when a fallback was asked
    for, or the CLI refuses what the library allows -- and "not configured" is
    one of the failures a fallback exists to handle.
    """
    monkeypatch.setattr(
        cli_module, "DeepResearchClient", _cli_client_with((BACKUP, _params()))
    )
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "research", "q",
            "--provider", "falcon",
            "--fallback-provider", BACKUP,
            "--output", str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0
    content = output.read_text()
    assert f"provider: {BACKUP}" in content
    assert "requested_provider: falcon" in content
    # The provider that could not run is in the trail, not dropped from it.
    assert "ProviderNotConfiguredError" in content


def test_cli_v_surfaces_the_inert_fallback_diagnostic(tmp_path, monkeypatch):
    """The docs promise `-v` shows it, so pin that the wiring delivers.

    The message is INFO and the CLI defaults to WARNING, so by default an
    operator sees nothing -- deliberately, since it also fires on runs that
    succeed. That makes `-v` the whole of its reachability from the command
    line, and `-v` is a *global* option: it goes before the subcommand.
    """
    monkeypatch.setattr(
        cli_module,
        "DeepResearchClient",
        _cli_client_with((PRIMARY, _params(error_type="billing"))),
    )
    output = tmp_path / "report.md"

    # Asserted on the captured output, not caplog: the CLI's setup_logging
    # calls basicConfig(force=True), which closes caplog's handler, so records
    # emitted inside the invocation never reach it. This is the handler half
    # of the leak that _restore_package_log_level deliberately does not cover.
    result = CliRunner().invoke(
        cli_module.app,
        [
            "-v",
            "research", "q",
            "--provider", PRIMARY,
            "--fallback",
            "--output", str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 1
    assert "candidate: no other provider is available." in result.output
    assert not output.exists()


def test_cli_prints_no_trail_header_when_there_is_no_trail(tmp_path, monkeypatch):
    """An ordinary single-provider failure must not grow a bare header.

    `ProviderError` subclasses `ValueError`, so a run with no fallback reaches
    the same handler that prints the trail -- carrying `provider_attempts ==
    ()`. Without the guard the handler prints "Providers tried:" with nothing
    under it. The typo test cannot see this: that path exits from the
    pre-check and never reaches the handler at all.
    """
    monkeypatch.setattr(
        cli_module,
        "DeepResearchClient",
        _cli_client_with((PRIMARY, _params(error_type="billing"))),
    )
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        cli_module.app,
        ["research", "q", "--provider", PRIMARY, "--output", str(output), "--no-cache"],
    )

    assert result.exit_code == 1
    # The failure itself is reported...
    assert "out of credits" in result.output
    # ...but a list of one provider we already named is not a trail.
    assert "Providers tried:" not in result.output
    assert not output.exists()


def test_cli_still_names_the_available_providers_for_a_typo(tmp_path, monkeypatch):
    """Standing the pre-check down must not swallow the remedy for a typo.

    A name that is not a provider at all is not "unconfigured", and the fix for
    it is the list of names that exist -- which lives only in this check. So
    the stand-down asks the client whether the name is a provider, rather than
    treating every unavailable name alike.
    """
    monkeypatch.setattr(
        cli_module, "DeepResearchClient", _cli_client_with((BACKUP, _params()))
    )
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "research", "q",
            "--provider", "falcn",
            "--fallback-provider", BACKUP,
            "--output", str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 1
    assert "not available" in result.output
    assert BACKUP in result.output
    # Never the sentence for a provider that exists but has no credential.
    assert "not configured" not in result.output
    assert not output.exists()


def test_cli_still_refuses_an_unconfigured_provider_without_a_fallback(tmp_path, monkeypatch):
    """Standing the check down is conditional on having somewhere else to go."""
    monkeypatch.setattr(
        cli_module, "DeepResearchClient", _cli_client_with((BACKUP, _params()))
    )
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        cli_module.app,
        ["research", "q", "--provider", "falcon", "--output", str(output), "--no-cache"],
    )

    assert result.exit_code == 1
    assert "not available" in result.output
    assert not output.exists()


def test_the_mock_can_demonstrate_the_quota_redaction_end_to_end(tmp_path, monkeypatch):
    """The one error type that folds provider text into its own remedy.

    Constructing ProviderQuotaError directly tests the redaction, but leaves
    the guard with no route a reader can run. This is that route.
    """
    monkeypatch.setattr(
        cli_module, "DeepResearchClient", _cli_client_with((BACKUP, _params()))
    )
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "research", "q",
            "--provider", PRIMARY,
            "--param", "error_type=quota",
            "--param", "response_delay=0.0",
            "--fallback-provider", BACKUP,
            "--output", str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0
    # The console keeps the provider's reset text, identifier and all.
    assert "quota_pool_7f21" in result.output
    # The committed file keeps only our reading of it.
    content = output.read_text()
    assert "quota_pool_7f21" not in content
    assert "renews at" not in content
    assert "the plan's usage limit is spent" in content


def test_cli_shows_the_trail_when_every_candidate_fails(tmp_path, monkeypatch):
    """The worst case should be as legible as the best.

    The CLI prints `Providers tried:` when a fallback *succeeds*; without this
    it printed only the last provider's message when the whole run failed --
    saying more about who was tried when it worked than when it did not.
    """
    monkeypatch.setattr(
        cli_module,
        "DeepResearchClient",
        _cli_client_with(
            (PRIMARY, _params(error_type="billing")),
            (BACKUP, _params(error_type="auth")),
        ),
    )
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "research", "q",
            "--provider", PRIMARY,
            "--fallback-provider", BACKUP,
            "--output", str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 1
    assert "Providers tried:" in result.output
    assert PRIMARY in result.output and BACKUP in result.output
    assert not output.exists()


def test_cli_rejects_nothing_when_fallback_is_absent(tmp_path, monkeypatch):
    """The flag is opt-in: an ordinary run is untouched by any of this."""
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")
    monkeypatch.setenv("DISABLE_CLAUDE_CODE_PROVIDER", "true")
    output = tmp_path / "report.md"

    result = CliRunner().invoke(
        cli_module.app,
        ["research", "q", "--provider", "mock", "--output", str(output), "--no-cache"],
    )

    assert result.exit_code == 0
    content = output.read_text()
    assert "fell_back" not in content
