"""Tests for opt-in provider fallback and the provenance it records.

See https://github.com/monarch-initiative/deep-research-client/issues/69

Nothing here is mocked. The failures are produced by the shipped
:class:`MockProvider`, which raises the same typed errors a real provider
raises, so the client is exercised through its ordinary code path.
"""

import json

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
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.processing import ResearchProcessor
from deep_research_client.provider_params import MockParams
from deep_research_client.providers.mock import MockProvider

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
    with caplog.at_level("WARNING"):
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
    with caplog.at_level("WARNING"):
        result = client.research("q", provider=PRIMARY, fallback=True)

    assert result.cached is warm_the_cache
    assert result.fell_back is True
    announcements = [
        m for m in (r.getMessage() for r in caplog.records)
        if "not the provider first tried" in m
    ]
    assert len(announcements) == 1
    assert BACKUP in announcements[0] and PRIMARY in announcements[0]


def test_no_fallback_is_announced_when_none_happened():
    """The warning must not fire on an ordinary run."""
    client = _client((PRIMARY, _params()))
    import logging as _logging

    records = []

    class _Capture(_logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    logger = _logging.getLogger("deep_research_client.client")
    logger.addHandler(handler)
    try:
        client.research("q", provider=PRIMARY)
    finally:
        logger.removeHandler(handler)

    assert not any("not the provider first tried" in m for m in records)


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
    with caplog.at_level("WARNING"):
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
    with caplog.at_level("WARNING"):
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
