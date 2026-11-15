"""Tests for the RSS Agent (rss_agent_claude) provider."""

import os
from pathlib import Path
import sys
import time

import pytest

from deep_research_client.client import DeepResearchClient
from deep_research_client.exceptions import (
    ProviderError,
    ProviderNotInstalledError,
    ProviderTimeoutError,
)
from deep_research_client.model_cards import get_provider_model_cards
from deep_research_client.models import CacheConfig, ProviderConfig
from deep_research_client.provider_params import RssAgentParams, create_provider_params
from deep_research_client.providers.rss_agent import (
    DIGEST_CSV,
    DIGEST_MARKDOWN,
    LIBRARY_INDEX,
    PIPELINE_SCRIPT,
    RssAgentProvider,
    prepare_workspace,
    read_digest_links,
    render_ranking_prompt,
)


def make_checkout(root: Path) -> Path:
    """Lay out the files the provider checks for in an rss_agent_claude checkout."""
    (root / PIPELINE_SCRIPT).parent.mkdir(parents=True)
    (root / PIPELINE_SCRIPT).write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (root / "scripts" / "common_paths.py").write_text("PROJECT_ROOT = None\n", encoding="utf-8")
    (root / LIBRARY_INDEX).parent.mkdir(parents=True)
    (root / LIBRARY_INDEX).write_bytes(b"")
    return root


def checkout_with_pipeline(root: Path, script_body: str) -> Path:
    """Make a checkout whose ``run_pipeline.sh`` runs ``script_body``.

    The body runs with the per-run workspace as its working directory, exactly
    as the real pipeline does, so it can write the digest files or exit to order.
    """
    checkout = make_checkout(root)
    (checkout / PIPELINE_SCRIPT).write_text(f"#!/usr/bin/env bash\n{script_body}", encoding="utf-8")
    return checkout


def make_provider(**params) -> RssAgentProvider:
    """Build an RSS Agent provider with the given parameters."""
    return RssAgentProvider(ProviderConfig(name="rss_agent"), RssAgentParams(**params))


def test_available_with_a_complete_checkout(tmp_path):
    provider = make_provider(repo_dir=str(make_checkout(tmp_path)), python_bin=sys.executable)
    assert provider.is_available()


@pytest.mark.parametrize(
    "remove, expected",
    [
        (None, "no rss_agent_claude checkout configured"),
        (PIPELINE_SCRIPT, "is not an rss_agent_claude checkout"),
        (LIBRARY_INDEX, "build_library_index.py"),
    ],
)
def test_unavailable_reason_names_what_is_missing(tmp_path, monkeypatch, remove, expected):
    monkeypatch.delenv("RSS_AGENT_DIR", raising=False)
    checkout = make_checkout(tmp_path)
    if remove is None:
        provider = make_provider(python_bin=sys.executable)
    else:
        (checkout / remove).unlink()
        provider = make_provider(repo_dir=str(checkout), python_bin=sys.executable)
    assert not provider.is_available()
    assert expected in provider.unavailable_reason()


def test_unavailable_reason_names_a_missing_interpreter(tmp_path):
    provider = make_provider(repo_dir=str(make_checkout(tmp_path)), python_bin="no-such-python-xyz")
    assert not provider.is_available()
    assert "RSS_AGENT_PYTHON" in provider.unavailable_reason()


@pytest.mark.asyncio
async def test_research_without_a_checkout_raises_not_installed(monkeypatch):
    monkeypatch.delenv("RSS_AGENT_DIR", raising=False)
    with pytest.raises(ProviderNotInstalledError, match="RSS_AGENT_DIR"):
        await make_provider().research("Marine nitrogen fixation")


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   \n# only a comment\n"])
async def test_research_rejects_a_query_with_no_topics(query):
    with pytest.raises(ValueError, match="must not be empty"):
        await make_provider().research(query)


def test_prepare_workspace(tmp_path):
    checkout = make_checkout(tmp_path / "checkout")
    (checkout / "config" / "examples").mkdir(parents=True)
    (checkout / "config" / "feeds.txt").write_text("https://mine.example/rss\n", encoding="utf-8")
    (checkout / "config" / "examples" / "feeds.txt").write_text("https://example.example/rss\n", encoding="utf-8")
    (checkout / "config" / "examples" / "biorxiv_subjects.txt").write_text("microbiology\n", encoding="utf-8")
    (checkout / "config" / "interest_profile.txt").write_text("the owner's own interests\n", encoding="utf-8")
    (checkout / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    workdir = tmp_path / "work"
    workdir.mkdir()

    prepare_workspace(checkout, workdir, ["Topic A", "Topic B"], "PROMPT")

    config = workdir / "config"
    assert (config / "feeds.txt").read_text(encoding="utf-8") == "https://mine.example/rss\n"
    assert (config / "biorxiv_subjects.txt").read_text(encoding="utf-8") == "microbiology\n"
    assert not (config / "crossref_issn_fallbacks.txt").exists()
    for name in ("interest_profile.txt", "crossref_terms.txt", "biorxiv_terms.txt"):
        assert (config / name).read_text(encoding="utf-8") == "Topic A\nTopic B\n"
    assert (config / "ranking_prompt.txt").read_text(encoding="utf-8") == "PROMPT"
    assert (workdir / PIPELINE_SCRIPT).is_file()
    assert (workdir / "library").is_symlink()
    assert (workdir / LIBRARY_INDEX).is_file()
    assert (workdir / ".env").read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=x\n"


def test_ranking_prompt_lists_every_topic():
    prompt = render_ranking_prompt(["Topic A", "Topic B"])
    assert "- Topic A\n- Topic B" in prompt


def test_pipeline_env_passes_only_the_knobs_that_were_set():
    env = make_provider(python_bin="/opt/py", max_age_days=7, skip_biorxiv=True).pipeline_env()
    assert env["PYTHON_BIN"] == "/opt/py"
    assert env["MAX_AGE_DAYS"] == "7"
    assert env["SKIP_BIORXIV"] == "1"
    assert env["SKIP_CROSSREF"] == "0"
    assert "MIN_SCORE" not in env


def test_pipeline_env_drops_an_unset_knob_inherited_from_the_environment(monkeypatch):
    # A knob the caller did not set must not leak in from the ambient
    # environment (e.g. the owner's daily-digest settings); it has to fall
    # through to the pipeline's own default.
    monkeypatch.setenv("MIN_SCORE", "0.99")
    env = make_provider(python_bin="/opt/py").pipeline_env()
    assert "MIN_SCORE" not in env


def test_default_python_prefers_the_checkout_venv(tmp_path):
    checkout = make_checkout(tmp_path)
    venv_python = checkout / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    provider = RssAgentProvider(ProviderConfig(name="rss_agent"), RssAgentParams(repo_dir=str(checkout)))
    assert provider.python_bin == str(venv_python)


def test_default_python_falls_back_to_plain_python(tmp_path, monkeypatch):
    monkeypatch.delenv("RSS_AGENT_PYTHON", raising=False)
    provider = RssAgentProvider(
        ProviderConfig(name="rss_agent"), RssAgentParams(repo_dir=str(make_checkout(tmp_path)))
    )
    assert provider.python_bin == "python"


def test_read_digest_links(tmp_path):
    csv_path = tmp_path / "curated_today.csv"
    csv_path.write_text(
        "title,link,score\n"
        "First,https://doi.org/10.1/a,0.9\n"
        "No link,,0.8\n"
        "Second,https://www.nature.com/articles/s1,0.7\n",
        encoding="utf-8",
    )
    assert read_digest_links(csv_path) == [
        "https://doi.org/10.1/a",
        "https://www.nature.com/articles/s1",
    ]


@pytest.mark.parametrize("alias", ["rss-agent", "rss", "digest"])
def test_model_aliases_resolve(alias):
    assert make_provider(model=alias).model == "rss-agent-digest"


def test_params_and_model_cards_are_registered():
    assert isinstance(create_provider_params("rss_agent"), RssAgentParams)
    assert get_provider_model_cards("rss_agent").default_model == "rss-agent-digest"


@pytest.mark.parametrize("env_value, registered", [(None, False), ("/somewhere", True)])
def test_client_registers_only_when_a_checkout_is_named(monkeypatch, env_value, registered):
    if env_value is None:
        monkeypatch.delenv("RSS_AGENT_DIR", raising=False)
    else:
        monkeypatch.setenv("RSS_AGENT_DIR", env_value)
    client = DeepResearchClient(cache_config=CacheConfig(enabled=False))
    assert (client.registry.get_provider("rss_agent") is not None) is registered


@pytest.mark.asyncio
async def test_research_returns_the_digest_and_its_links(tmp_path):
    body = (
        'mkdir -p data/outputs/markdown data/outputs/csv\n'
        'printf "# Digest\\n\\n- A paper\\n" > "{md}"\n'
        'printf "title,link,score\\nFirst,https://doi.org/10.1/a,0.9\\n" > "{csv}"\n'
    ).format(md=DIGEST_MARKDOWN, csv=DIGEST_CSV)
    checkout = checkout_with_pipeline(tmp_path, body)
    provider = make_provider(repo_dir=str(checkout), python_bin=sys.executable)

    result = await provider.research("Marine nitrogen fixation")

    assert result.answer_successful
    assert result.markdown.strip().startswith("# Digest")
    assert result.citations == ["https://doi.org/10.1/a"]


@pytest.mark.asyncio
async def test_research_reports_no_digest_when_nothing_is_written(tmp_path):
    checkout = checkout_with_pipeline(tmp_path, 'echo "no articles cleared MIN_SCORE today"\n')
    provider = make_provider(repo_dir=str(checkout), python_bin=sys.executable)

    result = await provider.research("Marine nitrogen fixation")

    assert not result.answer_successful
    assert result.citations == []
    assert "no articles cleared MIN_SCORE today" in result.markdown


@pytest.mark.asyncio
async def test_research_raises_with_the_log_on_a_non_zero_exit(tmp_path):
    body = 'echo "fetching feeds"\necho "ranking failed: boom"\nexit 3\n'
    checkout = checkout_with_pipeline(tmp_path, body)
    provider = make_provider(repo_dir=str(checkout), python_bin=sys.executable)

    with pytest.raises(ProviderError) as excinfo:
        await provider.research("Marine nitrogen fixation")

    message = str(excinfo.value)
    assert "code 3" in message
    assert "ranking failed: boom" in message  # the pipeline's own output is surfaced


@pytest.mark.asyncio
async def test_run_pipeline_timeout_kills_the_whole_process_group(tmp_path):
    # The stub launches a long-lived grandchild and records its pid. A timeout
    # must take down that child too, not just the bash that started it.
    workdir = tmp_path / "work"
    (workdir / PIPELINE_SCRIPT).parent.mkdir(parents=True)
    (workdir / PIPELINE_SCRIPT).write_text(
        '#!/usr/bin/env bash\nsleep 30 &\necho $! > child.pid\nwait\n', encoding="utf-8"
    )
    provider = make_provider(python_bin=sys.executable)

    with pytest.raises(ProviderTimeoutError):
        await provider._run_pipeline(workdir, timeout=1)

    child_pid = int((workdir / "child.pid").read_text(encoding="utf-8").strip())
    for _ in range(100):  # SIGKILL and reaping are not instantaneous
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"pipeline child {child_pid} survived the timeout")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rss_agent_end_to_end():
    """Run the real pipeline. Requires RSS_AGENT_DIR, its environment and ANTHROPIC_API_KEY."""
    provider = make_provider(skip_biorxiv=True, skip_nature=True, skip_asm=True, max_llm=50)
    if not provider.is_available():
        pytest.skip(provider.unavailable_reason())
    result = await provider.research("Marine microbial ecology\nOcean biogeochemistry")
    assert result.provider == "rss_agent"
    assert result.markdown.strip()
    if result.answer_successful:
        assert result.citations
