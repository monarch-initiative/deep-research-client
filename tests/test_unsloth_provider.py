"""Tests for the Unsloth Studio provider.

The parsing tests run on ``tests/input/unsloth_research_run.json``: a real
``GET /api/chat/research-runs/{id}`` response from Studio 2026.11.11 (Gemma 4
E4B), with its IDs replaced, its report cut to the opening sections, and each
step's result reduced to what was done. The run collected twelve pages and its
kept text links to three of them.

The live tests are marked ``integration`` and need a running Studio with a
chat model loaded, and its key in ``UNSLOTH_API_KEY``.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import time

import pytest

from deep_research_client.client import DeepResearchClient
from deep_research_client.exceptions import ProviderError, ProviderNotConfiguredError
from deep_research_client.model_cards import get_provider_model_cards
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.provider_params import UnslothParams, get_provider_params_class
from deep_research_client.providers.unsloth import (
    StudioRun,
    UnslothProvider,
    run_result,
)

RUN_FILE = Path(__file__).parent / "input" / "unsloth_research_run.json"
LLM = "unsloth/gemma-4-E4B-it-qat-GGUF"
QUESTION = "What is the function of PbgA (YejM) in outer membrane lipid homeostasis?"


@pytest.fixture
def recorded() -> dict:
    """The recorded run, as Studio returned it."""
    return json.loads(RUN_FILE.read_text(encoding="utf-8"))


def result_of(run: dict, run_kept: bool = False):
    """Turn a run payload into a result the way the provider does."""
    now = datetime(2026, 11, 24, 17, 0, 0)
    return run_result(
        StudioRun.model_validate(run), QUESTION,
        provider="unsloth", model="unsloth-studio-research", llm=LLM,
        instance_url="http://127.0.0.1:8888", start_time=now, end_time=now, run_kept=run_kept,
    )


def test_the_report_and_its_provenance_come_from_the_run(recorded):
    result = result_of(recorded)

    assert result.markdown.startswith("# Regulatory Mechanisms of Outer Membrane Lipid Homeostasis")
    assert result.title == recorded["plan"]["title"]
    assert result.provider == "unsloth"
    assert result.model == "unsloth-studio-research"
    assert result.provider_instance_url == "http://127.0.0.1:8888"
    assert result.run_metadata["llm"] == LLM


@pytest.mark.parametrize("run_kept", [True, False])
def test_the_run_id_is_recorded_only_while_the_run_still_exists_in_studio(recorded, run_kept):
    """Deleting the thread deletes the run, so its ID would be a handle on nothing."""
    result = result_of(recorded, run_kept=run_kept)

    assert result.provider_job_id == (recorded["id"] if run_kept else None)


def test_citations_are_the_pages_the_report_links_to_not_every_page_collected(recorded):
    """Twelve pages were collected; the kept text cites three, one of them three times."""
    result = result_of(recorded)

    assert [citation.rsplit(" - ", 1)[1] for citation in result.citations] == [
        "https://www.nature.com/articles/s41467-022-32277-1",
        "https://journals.asm.org/doi/10.1128/mbio.00939-20",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC13125744/",
    ]
    assert result.run_metadata["sources_collected"] == 12
    assert result.run_metadata["sources_cited"] == 3


def test_run_metadata_lists_each_search_and_fetch_in_order(recorded):
    steps = result_of(recorded).run_metadata["steps"]

    assert [step["action"] for step in steps] == ["search", "fetch"] * 3
    assert steps[1]["input"].startswith("https://")


@pytest.mark.parametrize(
    "change, message",
    [
        ({"status": "failed", "error": "The model stopped responding"}, "failed: The model stopped responding"),
        ({"status": "cancelled", "report": None}, "cancelled"),
        ({"report": ""}, "completed without a report"),
        ({"report": None}, "completed without a report"),
    ],
)
def test_a_run_that_did_not_produce_a_report_is_refused(recorded, change, message):
    with pytest.raises(ValueError, match=message):
        result_of({**recorded, **change})


def test_a_failed_run_is_refused_even_when_it_saved_a_partial_report(recorded):
    """Studio keeps a partial report on failure; it must not pass as research."""
    with pytest.raises(ValueError, match="failed: out of context"):
        result_of({**recorded, "status": "failed", "error": "out of context"})


@pytest.mark.parametrize(
    "params",
    [
        {"max_steps": 0},
        {"max_steps": 31},
        {"max_sources": 101},
        {"temperature": 2.5},
        {"poll_interval": 0},
        {"llm": ""},
        {"system_prompt": "x" * 32001},
        {"unknown": 1},
    ],
)
def test_parameters_outside_what_studio_accepts_are_rejected(params):
    with pytest.raises(ValueError):
        UnslothParams(**params)


def test_the_provider_is_registered_everywhere_a_provider_is_looked_up():
    assert get_provider_params_class("unsloth") is UnslothParams
    cards = get_provider_model_cards("unsloth")
    assert cards.default_model == "unsloth-studio-research"
    assert cards.resolve_model_name("unsloth") == "unsloth-studio-research"


@pytest.mark.parametrize("key, registered", [("sk-unsloth-test", True), (None, False)])
def test_auto_detection_registers_the_provider_only_with_a_key(monkeypatch, tmp_path, key, registered):
    monkeypatch.delenv("UNSLOTH_API_KEY", raising=False)
    monkeypatch.setenv("UNSLOTH_BASE_URL", "http://studio.example:9000/v1")
    if key:
        monkeypatch.setenv("UNSLOTH_API_KEY", key)

    client = DeepResearchClient(cache_config=CacheConfig(enabled=False, directory=str(tmp_path)))
    provider = client.registry.get_provider("unsloth")

    assert (provider is not None) is registered
    if registered:
        assert provider.base_url == "http://studio.example:9000"
        assert provider.is_available()


@pytest.mark.parametrize("model", [None, "unsloth", "studio", "unsloth-studio-research"])
def test_the_model_field_accepts_only_the_research_model_and_its_aliases(model):
    provider = UnslothProvider(ProviderConfig(name="unsloth", api_key="k"), UnslothParams(model=model))

    assert provider.model == "unsloth-studio-research"


def test_a_studio_model_id_given_as_the_model_is_refused_and_pointed_at_llm():
    """Accepted, it would be stamped on a report that another model wrote."""
    with pytest.raises(ValueError, match="use the llm parameter"):
        UnslothProvider(ProviderConfig(name="unsloth", api_key="k"), UnslothParams(model="unsloth/phi-4"))


def test_a_local_model_is_never_an_automatic_fallback():
    """A hosted provider that failed must not be quietly replaced by a small local model."""
    assert UnslothProvider.automatic_fallback is False
    assert UnslothProvider.produces_real_reports is True


async def test_research_without_a_key_says_which_variable_to_set():
    provider = UnslothProvider(ProviderConfig(name="unsloth"))

    with pytest.raises(ProviderNotConfiguredError, match="UNSLOTH_API_KEY"):
        await provider.research("anything")


async def test_an_empty_question_is_refused_before_studio_is_contacted():
    provider = UnslothProvider(ProviderConfig(name="unsloth", api_key="k", base_url="http://127.0.0.1:9"))

    with pytest.raises(ValueError, match="must not be empty"):
        await provider.research("  ")


async def test_health_reports_a_studio_that_is_not_running():
    """Nothing listens on port 9 (discard) on a normal machine. The connection is
    refused, or on WSL with mirrored networking times out; either is unreachable."""
    provider = UnslothProvider(ProviderConfig(name="unsloth", api_key="k", base_url="http://127.0.0.1:9"))

    health = await provider.check_health()

    assert health.configured is True
    assert health.reachable is False
    assert "cannot reach Unsloth Studio at http://127.0.0.1:9" in health.detail


async def test_research_on_a_studio_that_is_not_running_fails_soon_and_says_so():
    """Not after the five minutes a busy Studio is given, and not with a raw transport error."""
    provider = UnslothProvider(ProviderConfig(name="unsloth", api_key="k", base_url="http://127.0.0.1:9"))
    started = time.monotonic()

    with pytest.raises(ProviderError, match="is it running"):
        await provider.research("anything")

    assert time.monotonic() - started < 60


async def test_health_without_a_key_is_unconfigured():
    health = await UnslothProvider(ProviderConfig(name="unsloth")).check_health()

    assert health.configured is False
    assert "UNSLOTH_API_KEY" in health.detail


def live_provider(**params) -> UnslothProvider:
    """A provider on the running Studio, or a skip when no key is set."""
    key = os.getenv("UNSLOTH_API_KEY")
    if not key:
        pytest.skip("UNSLOTH_API_KEY is not set")
    config = ProviderConfig(name="unsloth", api_key=key, base_url=os.getenv("UNSLOTH_BASE_URL"), timeout=900)
    return UnslothProvider(config, UnslothParams(**params))


@pytest.mark.integration
async def test_live_health_names_the_loaded_model():
    health = await live_provider().check_health()

    assert health.reachable is True
    assert health.detail.startswith("model loaded: ")


@pytest.mark.integration
async def test_live_research_returns_a_report_whose_citations_are_pages_it_links_to():
    """How many pages a model cites varies run to run, and can be none; what must
    hold is that every citation listed is a collected page the report links to."""
    provider = live_provider(max_steps=6)

    result = await provider.research("What is the role of LRRK2 in Parkinson disease? Give a short cited report.")

    assert len(result.markdown.split()) > 100
    assert len(result.citations) == result.run_metadata["sources_cited"]
    assert all(citation.rsplit(" - ", 1)[-1] in result.markdown for citation in result.citations)
    assert result.run_metadata["sources_cited"] <= result.run_metadata["sources_collected"]
    assert result.provider_job_id is None


@pytest.mark.integration
async def test_live_run_asking_for_a_model_that_is_not_loaded_is_refused_before_it_starts():
    provider = live_provider(llm="unsloth/not-a-loaded-model")

    with pytest.raises(ProviderNotConfiguredError, match="is not loaded in Unsloth Studio"):
        await provider.research("anything")
