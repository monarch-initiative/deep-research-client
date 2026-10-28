"""RSS Agent literature-digest provider.

`rss_agent_claude <https://github.com/jcthrash/rss_agent_claude>`_ is a
literature curation pipeline: it fetches recent papers from RSS/Atom feeds,
Crossref, bioRxiv, Nature-family and ASM journals, keeps those semantically
close to an interest profile (sentence-transformers embeddings, plus FAISS
novelty against a personal paper library), and has Claude Haiku score the
survivors against a ranking prompt. Its output is a digest of the most
relevant *new* papers, not a synthesized answer.

The project is a set of scripts rather than an installable package, and it
brings torch, transformers and faiss with it, so this provider does not
import it. It runs ``scripts/run_pipeline.sh`` from a local checkout (named by
``RSS_AGENT_DIR``) under that checkout's own Python environment, inside a
throwaway workspace. The query becomes the run's interest profile and search
terms; the feed lists and the paper library come from the checkout.
"""

import asyncio
import csv
from datetime import datetime
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

from . import ResearchProvider
from ..exceptions import ProviderError, ProviderNotInstalledError
from ..models import ProviderConfig, ResearchResult
from ..provider_params import RssAgentParams
from ..model_cards import ProviderModelCards, create_rss_agent_model_cards

logger = logging.getLogger(__name__)

#: Environment variable naming the local rss_agent_claude checkout.
RSS_AGENT_DIR_ENV = "RSS_AGENT_DIR"

#: Environment variable naming the Python interpreter that has the checkout's
#: requirements installed (torch, faiss-cpu, anthropic, ...).
RSS_AGENT_PYTHON_ENV = "RSS_AGENT_PYTHON"

# A full fetch walks every feed and several Crossref queries before embedding
# and LLM scoring, so allow an hour unless the caller says otherwise.
RSS_AGENT_DEFAULT_TIMEOUT = 3600

PIPELINE_SCRIPT = Path("scripts") / "run_pipeline.sh"
LIBRARY_INDEX = Path("library") / "faiss" / "library_faiss.index"
DIGEST_MARKDOWN = Path("data") / "outputs" / "markdown" / "curated_today.md"
DIGEST_CSV = Path("data") / "outputs" / "csv" / "curated_today.csv"

#: Source lists taken from the checkout's ``config/`` (falling back to
#: ``config/examples/``). Everything topical is written from the query instead.
SOURCE_CONFIG_FILES = (
    "feeds.txt",
    "crossref_issn_fallbacks.txt",
    "biorxiv_subjects.txt",
)

#: Config files whose content is the query's topics, one per line. The
#: pipeline embeds the interest profile for its semantic filter, searches
#: Crossref with each term, and substring-matches bioRxiv against them.
TOPIC_CONFIG_FILES = (
    "interest_profile.txt",
    "crossref_terms.txt",
    "biorxiv_terms.txt",
)

RANKING_PROMPT_TEMPLATE = """\
You are a domain expert screening newly published scientific articles.

Task: Given ONLY the Title + Abstract/Summary, score how relevant the article
is to the following research interests on a 0-1 scale.

Research interests:
{interests}

Scoring rubric (be strict and consistent):
- 0.90-1.00: Directly addresses the core of these interests; a paper to read in full.
- 0.70-0.89: Clearly relevant to at least one interest; worth reading the abstract.
- 0.40-0.69: Adjacent or enabling; useful background or methods.
- 0.15-0.39: Broad connection only; likely skippable.
- 0.00-0.14: Unrelated.

Important rules:
- Short summaries are common in RSS feeds; do NOT assume irrelevance because of brevity.
- Do not give high scores for generic biology or methods without clear relevance.
- Do not explain your reasoning.

Return ONLY a JSON object exactly like:
{{"score": 0.0}}
"""


def query_to_topics(query: str) -> list[str]:
    """Split a query into the interest topics the pipeline expects.

    Each non-blank line is one topic, matching ``interest_profile.txt``; lines
    starting with ``#`` are comments there, so they are dropped here too.

    >>> query_to_topics("Soil microbial ecology\\n# a comment\\n\\n  Rhizosphere microbiome  ")
    ['Soil microbial ecology', 'Rhizosphere microbiome']
    >>> query_to_topics("SAR11 bacterioplankton physiology")
    ['SAR11 bacterioplankton physiology']
    """
    return [
        line.strip()
        for line in query.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def render_ranking_prompt(topics: list[str]) -> str:
    """Build the default Claude scoring prompt for a set of topics.

    The pipeline sends this as the system prompt for each scoring batch, so it
    must end by asking for the JSON score shape the pipeline parses.

    >>> prompt = render_ranking_prompt(["Marine nitrogen fixation"])
    >>> "- Marine nitrogen fixation" in prompt
    True
    >>> prompt.rstrip().endswith('{"score": 0.0}')
    True
    """
    interests = "\n".join(f"- {topic}" for topic in topics)
    return RANKING_PROMPT_TEMPLATE.format(interests=interests)


def prepare_workspace(
    repo_dir: Path,
    workdir: Path,
    topics: list[str],
    ranking_prompt: str,
) -> None:
    """Lay out a private copy of the pipeline for one run.

    The pipeline keeps all its state (seen ids, digested ids, score caches)
    under the project root it runs from. Running it in place would mix a
    research query into the owner's daily digest and hide every paper the
    digest had already shown, so each run gets a fresh project root: copied
    scripts, a config built from the query, and the checkout's paper library
    linked read-only.

    Args:
        repo_dir: The rss_agent_claude checkout.
        workdir: Empty directory to become the run's project root.
        topics: Interest topics, one per config line.
        ranking_prompt: System prompt for Claude scoring.
    """
    shutil.copytree(repo_dir / "scripts", workdir / "scripts")

    config_dir = workdir / "config"
    config_dir.mkdir()
    for name in SOURCE_CONFIG_FILES:
        for candidate in (repo_dir / "config" / name, repo_dir / "config" / "examples" / name):
            if candidate.is_file():
                shutil.copyfile(candidate, config_dir / name)
                break
    topic_text = "\n".join(topics) + "\n"
    for name in TOPIC_CONFIG_FILES:
        (config_dir / name).write_text(topic_text, encoding="utf-8")
    (config_dir / "ranking_prompt.txt").write_text(ranking_prompt, encoding="utf-8")

    (workdir / "library").symlink_to((repo_dir / "library").resolve(), target_is_directory=True)

    env_file = repo_dir / ".env"
    if env_file.is_file():
        shutil.copyfile(env_file, workdir / ".env")


def read_digest_links(csv_path: Path) -> list[str]:
    """Read the article links from the pipeline's CSV digest, in digest order.

    Args:
        csv_path: ``curated_today.csv`` written by ``generate_markdown.py``.

    Returns:
        Each article's link, skipping rows without one.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return [row["link"] for row in csv.DictReader(handle) if row.get("link")]


class RssAgentProvider(ResearchProvider):
    """Provider that runs a local rss_agent_claude pipeline as a digest search.

    No credential of its own: Claude scoring uses ``ANTHROPIC_API_KEY`` from the
    environment or the checkout's ``.env``.
    """

    #: Auto-detection registers it only when a checkout is named.
    registration_requirement = (
        f"requires {RSS_AGENT_DIR_ENV} to point at a local rss_agent_claude "
        "checkout with its Python requirements installed"
    )

    #: Nothing to run is an install problem, not a missing key.
    unavailable_error = ProviderNotInstalledError

    def __init__(self, config: ProviderConfig, params: Optional[RssAgentParams] = None):
        """Initialize the RSS Agent provider.

        Args:
            config: Provider configuration.
            params: RSS Agent-specific parameters.
        """
        self.params = params or RssAgentParams()
        super().__init__(config, self.params.model)
        repo_dir = self.params.repo_dir or os.getenv(RSS_AGENT_DIR_ENV)
        self.repo_dir = Path(repo_dir).expanduser() if repo_dir else None
        self.python_bin = self.params.python_bin or os.getenv(RSS_AGENT_PYTHON_ENV) or "python"

    def get_default_model(self) -> str:
        """Return the default model card identifier."""
        return "rss-agent-digest"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for the RSS Agent provider."""
        return create_rss_agent_model_cards()

    def _missing(self) -> Optional[str]:
        """Say what the checkout lacks, or None if it can run."""
        if self.repo_dir is None:
            return (
                f"no rss_agent_claude checkout configured (set {RSS_AGENT_DIR_ENV} "
                "or the repo_dir parameter)"
            )
        if not (self.repo_dir / PIPELINE_SCRIPT).is_file():
            return f"{self.repo_dir} is not an rss_agent_claude checkout (no {PIPELINE_SCRIPT})"
        if not (self.repo_dir / LIBRARY_INDEX).is_file():
            # filter_and_rank.py refuses to start without it, so fail before
            # spending a full fetch on a run that cannot finish.
            return (
                f"{self.repo_dir} has no paper library index ({LIBRARY_INDEX}); "
                "build it with scripts/build_library_index.py"
            )
        if shutil.which(self.python_bin) is None:
            return (
                f"Python interpreter '{self.python_bin}' not found; set "
                f"{RSS_AGENT_PYTHON_ENV} to the interpreter with rss_agent_claude's "
                "requirements installed"
            )
        return None

    def unavailable_reason(self) -> str:
        """Name what is missing from the local checkout.

        Returns:
            Human-readable explanation suitable for an error message
        """
        if not self.config.enabled:
            return super().unavailable_reason()
        return self._missing() or super().unavailable_reason()

    def is_available(self) -> bool:
        """Check that a runnable checkout is configured."""
        return self.config.enabled and self._missing() is None

    def pipeline_env(self) -> dict[str, str]:
        """Build the environment for ``run_pipeline.sh``.

        Only knobs the caller set are passed, so everything else keeps the
        pipeline's own defaults.

        Returns:
            The current environment plus the pipeline's tuning variables.
        """
        env = dict(os.environ)
        env["PYTHON_BIN"] = self.python_bin
        knobs = {
            "MIN_SCORE": self.params.min_score,
            "TOP_PICKS_SCORE": self.params.top_picks_score,
            "MAX_AGE_DAYS": self.params.max_age_days,
            "MAX_LLM": self.params.max_llm,
            "SEMANTIC_THRESHOLD": self.params.semantic_threshold,
            "CROSSREF_DAYS_BACK": self.params.crossref_days_back,
        }
        env.update({key: str(value) for key, value in knobs.items() if value is not None})
        switches = {
            "SKIP_CROSSREF": self.params.skip_crossref,
            "SKIP_BIORXIV": self.params.skip_biorxiv,
            "SKIP_NATURE": self.params.skip_nature,
            "SKIP_ASM": self.params.skip_asm,
        }
        env.update({key: "1" if value else "0" for key, value in switches.items()})
        return env

    async def research(self, query: str) -> ResearchResult:
        """Run the pipeline with the query as its interest profile.

        Args:
            query: Research interests, one topic per line.

        Returns:
            ResearchResult whose markdown is the pipeline's digest and whose
            citations are the digested articles' links.
        """
        topics = query_to_topics(query)
        if not topics:
            raise ValueError("Research query must not be empty.")
        self.require_available()
        assert self.repo_dir is not None  # guaranteed by require_available

        ranking_prompt = (
            Path(self.params.ranking_prompt_file).read_text(encoding="utf-8")
            if self.params.ranking_prompt_file
            else render_ranking_prompt(topics)
        )
        timeout = self.config.timeout or RSS_AGENT_DEFAULT_TIMEOUT
        start_time = datetime.now()

        with tempfile.TemporaryDirectory(prefix="rss_agent_", dir=self.params.workdir_base) as tmp:
            workdir = Path(tmp)
            prepare_workspace(self.repo_dir, workdir, topics, ranking_prompt)
            logger.info("Running rss_agent_claude pipeline from %s in %s", self.repo_dir, workdir)
            log = await self._run_pipeline(workdir, timeout)

            digest_path = workdir / DIGEST_MARKDOWN
            if digest_path.is_file():
                markdown = digest_path.read_text(encoding="utf-8")
                citations = read_digest_links(workdir / DIGEST_CSV)
                successful = True
            else:
                # generate_markdown.py writes nothing when no article clears
                # MIN_SCORE. That is an answer about today's feeds, not a
                # report, so it is marked unsuccessful and not cached.
                last_line = log.strip().splitlines()[-1] if log.strip() else ""
                logger.warning("rss_agent_claude produced no digest: %s", last_line)
                markdown = (
                    "No recent articles matched these interests closely enough "
                    f"for a digest.\n\nPipeline said: {last_line}\n"
                )
                citations = []
                successful = False

        end_time = datetime.now()
        return ResearchResult(
            markdown=markdown,
            citations=citations,
            provider=self.name,
            query=query,
            model=self.model,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            answer_successful=successful,
        )

    async def _run_pipeline(self, workdir: Path, timeout: int) -> str:
        """Run ``run_pipeline.sh`` in the workspace and return its output.

        Args:
            workdir: Workspace from :func:`prepare_workspace`.
            timeout: Seconds before the run is killed.

        Returns:
            Combined stdout and stderr.

        Raises:
            ProviderError: If the pipeline exits non-zero.
            TimeoutError: If it runs past ``timeout``.
        """
        process = await asyncio.create_subprocess_exec(
            "bash", str(workdir / PIPELINE_SCRIPT),
            cwd=workdir,
            env=self.pipeline_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.wait()
            raise
        log = stdout.decode("utf-8", errors="replace")
        logger.debug("rss_agent_claude output:\n%s", log)
        if process.returncode != 0:
            tail = "\n".join(log.strip().splitlines()[-20:])
            raise ProviderError(
                self.name,
                f"run_pipeline.sh exited with code {process.returncode}:\n{tail}",
            )
        return log
