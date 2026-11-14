"""Tests for the ToxIndex provider (issue #98).

ToxIndex has no route that submits a question to Sage, so the provider is a
stub for research; what it can do is verify a key and turn a finished Sage
conversation into a report. The parsing tests run on
``tests/input/toxindex_conversation.json``, a real ``/messages`` response
trimmed to seven messages, with its final answer shortened and given one tag
that no tool result backs.
"""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import _settable_credential_hints, app
from deep_research_client.client import DeepResearchClient
from deep_research_client.exceptions import ProviderAuthError
from deep_research_client.model_cards import get_provider_model_cards
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.provider_params import ToxIndexParams, get_provider_params_class
from deep_research_client.providers.toxindex import (
    TOXINDEX_UNAVAILABLE_MESSAGE,
    SageMessages,
    ToxIndexProvider,
    sage_result,
)

CONVERSATION_FILE = Path(__file__).parent / "input" / "toxindex_conversation.json"
CONVERSATION_ID = "6c906ccf-a1e9-4dc7-820f-5d4626155e9f"


@pytest.fixture
def conversation() -> SageMessages:
    """The recorded conversation, parsed as the provider parses the API's answer."""
    return SageMessages.model_validate(json.loads(CONVERSATION_FILE.read_text(encoding="utf-8")))


def test_the_question_and_answer_come_from_the_first_user_and_last_assistant_messages(conversation):
    result = sage_result(conversation, CONVERSATION_ID)

    assert result.query.startswith("What data is available on human exposure to carbamate")
    assert result.markdown.startswith("# Carbamate exposure (trimmed)")
    assert result.provider == "toxindex"
    assert result.model == "sage"


def test_each_cited_tag_resolves_to_its_tool_call_once_in_order_of_use(conversation):
    """tc-45957cb5 is cited twice and listed once; tc-deadbeef has no tool result."""
    result = sage_result(conversation, CONVERSATION_ID)

    assert result.citations == [
        '[tc-45957cb5] tox_chemical_identity {"identifier": "carbamate"} '
        "(tool call toolu_vrtx_01ERSWx7NP76TWGxsHeiLmkY)"
    ]
    assert result.run_metadata["unresolved_citations"] == ["tc-deadbeef"]


def test_run_metadata_counts_tool_calls_and_failures(conversation):
    result = sage_result(conversation, CONVERSATION_ID)

    assert result.run_metadata["tool_calls"] == 3
    assert result.run_metadata["tools_used"] == {
        "tox_chemical_identity": 1, "ask_user": 1, "kiln_query": 1,
    }
    assert result.run_metadata["failed_tool_calls"] == 1
    assert result.provider_config == {
        "conversation_id": CONVERSATION_ID,
        "retrieval_mode": "toxindex_conversation",
    }


def test_duration_spans_the_question_to_the_answer(conversation):
    result = sage_result(conversation, CONVERSATION_ID)

    assert result.duration_seconds == pytest.approx(402.231)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.update(messages=[]), "has no messages"),
        (lambda data: data.update(running=True), "is still running"),
        (lambda data: data["messages"].pop(), "does not end on an answer"),
        (lambda data: data["messages"].pop(0), "has no question in it"),
    ],
    ids=["empty", "running", "ends-on-tool-result", "no-question"],
)
def test_a_conversation_without_a_finished_answer_is_refused(change, message):
    data = json.loads(CONVERSATION_FILE.read_text(encoding="utf-8"))
    change(data)

    with pytest.raises(ValueError, match=message):
        sage_result(SageMessages.model_validate(data), CONVERSATION_ID)


@pytest.mark.parametrize("api_key", [None, "a-real-looking-key"])
def test_the_stub_is_never_available(api_key):
    """A key serves retrieval and the health probe, never research."""
    provider = ToxIndexProvider(ProviderConfig(name="toxindex", api_key=api_key))

    assert provider.is_available() is False
    assert provider.unavailable_reason() == TOXINDEX_UNAVAILABLE_MESSAGE


@pytest.mark.asyncio
async def test_research_raises_and_says_what_to_do_instead():
    provider = ToxIndexProvider(ProviderConfig(name="toxindex", api_key="k"))

    with pytest.raises(NotImplementedError, match="toxindex-conversation"):
        await provider.research("Is bisphenol A genotoxic?")


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (None, "https://platform.toxindex.com/v1"),
        ("http://localhost:8080/v1/", "http://localhost:8080/v1"),
    ],
)
def test_base_url_defaults_and_drops_a_trailing_slash(base_url, expected):
    provider = ToxIndexProvider(ProviderConfig(name="toxindex", base_url=base_url))

    assert provider.base_url == expected


@pytest.mark.parametrize("alias", ["sage", "toxindex", "toxindex-sage"])
def test_model_aliases_resolve_to_sage(alias):
    provider = ToxIndexProvider(ProviderConfig(name="toxindex"), ToxIndexParams(model=alias))

    assert provider.model == "sage"


def test_params_and_model_cards_are_registered():
    assert get_provider_params_class("toxindex") is ToxIndexParams
    cards = get_provider_model_cards("toxindex")
    assert cards is not None
    assert cards.default_model == "sage"


@pytest.mark.asyncio
async def test_health_without_a_key_names_the_variable_and_makes_no_call():
    health = await ToxIndexProvider(ProviderConfig(name="toxindex")).check_health()

    assert health.configured is False
    assert health.reachable is False
    assert "TOXINDEX_API_KEY" in health.detail


def test_the_client_registers_it_unconditionally_but_never_offers_it(monkeypatch):
    monkeypatch.delenv("TOXINDEX_API_KEY", raising=False)
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))

    assert client.registry.get_provider("toxindex") is not None
    assert "toxindex" not in client.get_available_providers()
    assert client.unregistered_reason("toxindex") == TOXINDEX_UNAVAILABLE_MESSAGE


def test_the_client_hands_the_registered_stub_its_key(monkeypatch):
    monkeypatch.setenv("TOXINDEX_API_KEY", "from-the-environment")
    monkeypatch.setenv("TOXINDEX_BASE_URL", "http://localhost:8080/v1")
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))

    provider = client.registry.get_provider("toxindex")
    assert provider.config.api_key == "from-the-environment"
    assert provider.base_url == "http://localhost:8080/v1"


def test_a_stub_is_never_offered_as_a_key_to_set():
    """Setting TOXINDEX_API_KEY would not make research work, so it is no hint.

    Which is why the class declares ``stub_reason`` and not ``credential_env_var``.
    """
    assert ToxIndexProvider.credential_env_var is None
    assert "toxindex" not in _settable_credential_hints()


def test_providers_lists_it_as_a_stub(bare_machine):
    result = CliRunner().invoke(app, ["providers"])

    assert result.exit_code == 0
    stubs = result.output.split("Stub providers (not yet callable):")[1]
    assert "toxindex: ToxIndex - Sage has no API route" in stubs
    assert "TOXINDEX_API_KEY" not in result.output


def test_retrieval_without_a_key_fails_before_any_call(monkeypatch):
    monkeypatch.delenv("TOXINDEX_API_KEY", raising=False)

    result = CliRunner().invoke(app, ["toxindex-conversation", CONVERSATION_ID])

    assert result.exit_code == 1
    assert "TOXINDEX_API_KEY is required" in result.output


needs_key = pytest.mark.skipif(
    not os.getenv("TOXINDEX_API_KEY"), reason="TOXINDEX_API_KEY is not set"
)


def _live_provider(api_key: str | None = None) -> ToxIndexProvider:
    """A provider pointed at the real API, with the environment's key by default."""
    return ToxIndexProvider(ProviderConfig(
        name="toxindex",
        api_key=api_key or os.environ["TOXINDEX_API_KEY"],
        base_url=os.getenv("TOXINDEX_BASE_URL"),
    ))


@pytest.mark.integration
@needs_key
@pytest.mark.asyncio
async def test_live_health_accepts_the_key_and_still_reports_no_research():
    health = await _live_provider().check_health()

    assert health.configured is True
    assert health.reachable is False
    assert health.detail.startswith("key accepted (")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_bad_key_is_an_auth_error():
    provider = ToxIndexProvider(ProviderConfig(name="toxindex", api_key="not-a-key"))

    with pytest.raises(ProviderAuthError, match="invalid api key"):
        await provider.fetch_conversation(CONVERSATION_ID)


@pytest.mark.integration
@needs_key
@pytest.mark.asyncio
async def test_live_unknown_conversation_is_refused():
    """The API answers an unknown id with an empty 200, so the parser must catch it."""
    with pytest.raises(ValueError, match="has no messages"):
        await _live_provider().retrieve_conversation("00000000-0000-0000-0000-000000000000")
