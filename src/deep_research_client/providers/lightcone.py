"""Lightcone / ASTRA research provider.

Unlike the literature providers, Lightcone is not a "query in, report out"
service. `Lightcone <https://lightconeresearch.org>`_ is a local command-line
tool (the ``lc`` binary) that materializes an **ASTRA** (*Agentic Schema for
Transparent Research Analysis*) specification -- an ``astra.yaml`` file -- into
a tree of outputs, recording the methodological decisions and their provenance
along the way.

This provider is therefore a *spec runner*: the research "query" is a path to
an ASTRA project directory (or an ``astra.yaml`` file), not a free-text
question. The spec already declares the inputs, decisions, and expected
outputs; the provider drives ``lc`` to execute them in the project directory,
then harvests the materialized output tree into the standard
:class:`~deep_research_client.models.ResearchResult` (report ``markdown`` plus
non-text ``artifacts``).

No provider API key is required: Lightcone drives a local agent (Claude Code),
so authentication and billing are handled by that local installation.

Provisional behavior
--------------------
Lightcone's published command reference documents ``lc init``, ``lc verify``,
``lc export``, and a set of Claude Code slash-commands, but the exact
subcommand that materializes a spec -- and the directory that outputs land in
-- are not yet pinned against a verified CLI release. Those two unknowns are
isolated behind :class:`~deep_research_client.provider_params.LightconeParams`
(``materialize_args`` and ``output_subdir``) and an ``extra_args`` escape
hatch, so the provider can be corrected without code changes once the real
interface is confirmed. Everything else -- availability checks, the
``ResearchProvider`` contract, artifact harvesting, and citation extraction --
is standard.

Security
--------
Materializing an ASTRA spec runs an agent that executes code. Only run
Lightcone against trusted projects in a sandboxed environment.
"""

import asyncio
import base64
import logging
import mimetypes
import re
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import List, Optional

from . import ResearchProvider
from ..models import ProviderConfig, ResearchArtifact, ResearchResult, sanitize_artifact_filename
from ..model_cards import ProviderModelCards, create_lightcone_model_cards
from ..provider_params import LightconeParams

logger = logging.getLogger(__name__)

# Canonical spec filename an ASTRA project is expected to contain.
ASTRA_SPEC_FILENAME = "astra.yaml"

# Report filenames preferred when locating the materialized report markdown.
_REPORT_MARKDOWN_BASENAMES = ("final_report.md", "report.md")

# Sidecar output files worth preserving as artifacts.
_ALLOWED_ARTIFACT_EXTENSIONS = {
    ".csv",
    ".gif",
    ".html",
    ".ipynb",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".parquet",
    ".pdf",
    ".png",
    ".svg",
    ".tsv",
    ".txt",
    ".webp",
    ".yaml",
    ".yml",
}
# Runtime scaffolding / caches that should never be harvested.
_NOISY_ARTIFACT_PREFIXES = (
    ".cache/",
    ".claude/",
    ".git/",
    ".ipynb_checkpoints/",
    ".venv/",
    "__pycache__/",
    "node_modules/",
)
_NOISY_ARTIFACT_SUFFIXES = (".log", ".tmp", ".pyc")

# Citation patterns: PMIDs and bare URLs, mirroring the other providers.
_PMID_PATTERN = re.compile(r"PMID:\s*(\d{7,8})")
_URL_PATTERN = re.compile(r"https?://[^\s\)\]>]+")


class LightconeProvider(ResearchProvider):
    """Provider that runs the local Lightcone (``lc``) CLI to materialize a spec."""

    def __init__(self, config: ProviderConfig, params: Optional[LightconeParams] = None):
        """Initialize the Lightcone provider.

        Args:
            config: Provider configuration (no API key required).
            params: Lightcone-specific parameters.
        """
        self.params = params or LightconeParams()
        super().__init__(config, self.params.model)

        self.lc_executable = self.params.lc_executable
        # ProviderConfig.timeout wins when set; otherwise use the params default.
        self.timeout = config.timeout or self.params.timeout

    def get_default_model(self) -> str:
        """Get the default model identifier for this provider."""
        return "lightcone-astra"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for the Lightcone provider."""
        return create_lightcone_model_cards()

    def is_available(self) -> bool:
        """Check whether the Lightcone CLI is available.

        Returns:
            True if the provider is enabled and the ``lc`` executable is
            resolvable on the PATH (or as an absolute path).
        """
        import shutil

        if not self.config.enabled:
            return False

        if shutil.which(self.lc_executable) is None:
            logger.warning(
                "Lightcone executable %r not found in PATH", self.lc_executable
            )
            return False

        return True

    def _resolve_project(self, query: str) -> Path:
        """Resolve the research query to an ASTRA project directory.

        The query may be a project directory (containing ``astra.yaml``) or a
        path to an ``astra.yaml`` file. An explicit ``working_dir`` param
        overrides the query. Fails fast if no spec is found.

        Args:
            query: The research "query": a project dir or astra.yaml path.

        Returns:
            The resolved project directory containing ``astra.yaml``.

        Raises:
            ValueError: If the path does not resolve to an ASTRA project.
        """
        raw = (self.params.working_dir or query or "").strip()
        if not raw:
            raise ValueError(
                "Lightcone requires an ASTRA project path as the query "
                "(a directory containing astra.yaml, or the astra.yaml file)."
            )

        path = Path(raw).expanduser()
        if path.is_file() and path.name == ASTRA_SPEC_FILENAME:
            project_dir = path.parent
        elif path.is_dir():
            project_dir = path
        else:
            raise ValueError(
                f"Lightcone project path not found: {raw!r}. Expected a directory "
                f"containing {ASTRA_SPEC_FILENAME}, or the {ASTRA_SPEC_FILENAME} file."
            )

        if not (project_dir / ASTRA_SPEC_FILENAME).is_file():
            raise ValueError(
                f"No {ASTRA_SPEC_FILENAME} found in Lightcone project {project_dir}."
            )

        return project_dir

    def _build_command(self) -> List[str]:
        """Build the ``lc`` command-line invocation.

        The spec is discovered from the working directory rather than passed as
        an argument. This method is pure and side-effect free to keep it easy to
        unit test.

        Returns:
            The argument list to pass to the subprocess.

        Examples:
            >>> from deep_research_client.models import ProviderConfig
            >>> from deep_research_client.provider_params import LightconeParams
            >>> p = LightconeProvider(
            ...     ProviderConfig(name="lightcone", api_key=None, enabled=True),
            ...     LightconeParams(universe="baseline"),
            ... )
            >>> p._build_command()
            ['lc', 'run', '--universe', 'baseline']
        """
        command: List[str] = [self.lc_executable, *self.params.materialize_args]
        if self.params.universe:
            command.extend(["--universe", self.params.universe])
        command.extend(self.params.extra_args)
        return command

    async def research(self, query: str) -> ResearchResult:
        """Materialize an ASTRA spec by running the Lightcone CLI.

        Args:
            query: Path to an ASTRA project directory or ``astra.yaml`` file.

        Returns:
            ResearchResult with the materialized report and harvested artifacts.

        Raises:
            ValueError: If the provider is unavailable, the project cannot be
                resolved, or the ``lc`` run fails.
        """
        start_time = datetime.now()

        if not self.is_available():
            raise ValueError(
                "Lightcone provider not available. Ensure the 'lc' CLI is "
                "installed and on your PATH."
            )

        project_dir = self._resolve_project(query)
        command = self._build_command()
        logger.info(
            "Running Lightcone in %s (timeout=%ss)", project_dir, self.timeout
        )
        logger.debug("Lightcone command: %s", " ".join(command))

        stdout, stderr, returncode = await self._run_process(command, project_dir)
        if returncode != 0:
            raise ValueError(
                f"Lightcone exited with code {returncode}: "
                f"{stderr.strip() or '<no stderr>'}"
            )

        output_dir = project_dir / self.params.output_subdir
        report_name, markdown = self._select_report(output_dir)
        artifacts = (
            self._harvest_artifacts(output_dir, skip_names={report_name} if report_name else set())
            if self.params.save_artifacts
            else []
        )

        if markdown is None:
            markdown = self._synthesize_report(project_dir, artifacts, stdout)

        citations = self._extract_citations(markdown)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            "Lightcone run completed in %.1fs (%d chars, %d citations, %d artifacts)",
            duration,
            len(markdown),
            len(citations),
            len(artifacts),
        )

        return ResearchResult(
            markdown=markdown,
            citations=citations,
            artifacts=artifacts,
            provider=self.name,
            query=query,
            model=self.model,
            run_metadata={"project_dir": str(project_dir), "universe": self.params.universe},
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
        )

    async def _run_process(
        self, command: List[str], cwd: Path
    ) -> tuple[str, str, int]:
        """Run the ``lc`` subprocess in the project directory.

        Args:
            command: The command argument list.
            cwd: The project directory to run in.

        Returns:
            Tuple of (stdout, stderr, return_code).

        Raises:
            ValueError: If the process does not finish within the timeout.
        """
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise ValueError(f"Lightcone run timed out after {self.timeout}s.")

        returncode = -1 if process.returncode is None else process.returncode
        return (
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            returncode,
        )

    def _select_report(self, output_dir: Path) -> tuple[Optional[str], Optional[str]]:
        """Locate and read the best markdown report in the output tree.

        Prefers canonical report basenames, then any ``report*.md``, then any
        ``.md`` file. Returns ``(relative_name, markdown)`` or ``(None, None)``
        when no report is present.
        """
        if not output_dir.is_dir():
            return None, None

        md_files = [
            p
            for p in sorted(output_dir.rglob("*.md"))
            if p.is_file() and not self._is_noisy_artifact_path(p.relative_to(output_dir))
        ]
        if not md_files:
            return None, None

        def rank(path: Path) -> tuple[int, str]:
            name = path.name.lower()
            if name in _REPORT_MARKDOWN_BASENAMES:
                return (0, str(path))
            if "report" in name:
                return (1, str(path))
            return (2, str(path))

        chosen = min(md_files, key=rank)
        markdown = chosen.read_text(encoding="utf-8", errors="replace")
        return str(chosen.relative_to(output_dir).as_posix()), markdown

    def _harvest_artifacts(
        self, output_dir: Path, skip_names: set[str]
    ) -> List[ResearchArtifact]:
        """Harvest useful materialized outputs from the output tree.

        Walks ``output_dir``, preserving non-noisy files with an allowed
        extension under the size limit as :class:`ResearchArtifact` objects.
        """
        if not output_dir.is_dir():
            return []

        artifacts: List[ResearchArtifact] = []
        used_filenames: set[str] = set()

        for path in sorted(output_dir.rglob("*")):
            rel = path.relative_to(output_dir)
            rel_posix = rel.as_posix()
            if rel_posix in skip_names:
                continue
            if not self._should_preserve_artifact(path, rel):
                continue

            content = path.read_bytes()
            filename = self._artifact_filename(rel_posix, used_filenames)
            artifacts.append(
                ResearchArtifact(
                    filename=filename,
                    content_base64=base64.b64encode(content).decode("ascii"),
                    media_type=mimetypes.guess_type(path.name)[0],
                    path=rel_posix,
                    source="lightcone_outputs",
                    description=self._artifact_description(rel_posix),
                )
            )

        return artifacts

    def _should_preserve_artifact(self, path: Path, rel: Path) -> bool:
        """Return whether a materialized file should become a ResearchArtifact."""
        if path.is_symlink() or not path.is_file():
            return False
        if self._is_noisy_artifact_path(rel):
            return False
        if path.suffix.lower() not in _ALLOWED_ARTIFACT_EXTENSIONS:
            return False
        size = path.stat().st_size
        if size > self.params.artifact_max_bytes:
            logger.info(
                "Skipping Lightcone artifact %s: %s bytes exceeds %s byte limit",
                rel.as_posix(),
                size,
                self.params.artifact_max_bytes,
            )
            return False
        return True

    @staticmethod
    def _is_noisy_artifact_path(rel: Path) -> bool:
        """Return whether a relative path is runtime scaffolding or logs.

        Examples:
            >>> from pathlib import Path
            >>> LightconeProvider._is_noisy_artifact_path(Path(".git/config"))
            True
            >>> LightconeProvider._is_noisy_artifact_path(Path("run/output.log"))
            True
            >>> LightconeProvider._is_noisy_artifact_path(Path("figures/fig1.png"))
            False
        """
        normalized = rel.as_posix().lstrip("/").lower()
        if normalized.startswith(_NOISY_ARTIFACT_PREFIXES):
            return True
        parts = PurePosixPath(normalized).parts
        # Skip anything under a hidden (dot-prefixed) parent directory.
        if any(part.startswith(".") for part in parts[:-1]):
            return True
        return PurePosixPath(normalized).name.endswith(_NOISY_ARTIFACT_SUFFIXES)

    @staticmethod
    def _artifact_filename(rel_posix: str, used_filenames: set[str]) -> str:
        """Return a stable, unique filename for a harvested artifact."""
        filename = sanitize_artifact_filename(rel_posix, fallback="artifact")
        candidate = filename
        suffix = PurePosixPath(filename).suffix
        stem = filename.removesuffix(suffix) if suffix else filename
        stem = stem or "artifact"
        counter = 2
        while candidate in used_filenames:
            candidate = f"{stem}-{counter}{suffix}"
            counter += 1
        used_filenames.add(candidate)
        return candidate

    @staticmethod
    def _artifact_description(rel_posix: str) -> str:
        """Build a short human-readable artifact description."""
        stem = PurePosixPath(rel_posix).stem.replace("_", " ").replace("-", " ").strip()
        return f"Lightcone output {stem}" if stem else f"Lightcone output {rel_posix}"

    def _synthesize_report(
        self, project_dir: Path, artifacts: List[ResearchArtifact], stdout: str
    ) -> str:
        """Build a fallback markdown report when no report file was materialized."""
        name = self._read_spec_name(project_dir) or project_dir.name
        lines = [f"# Lightcone analysis: {name}", ""]
        if self.params.universe:
            lines.append(f"Universe: `{self.params.universe}`")
            lines.append("")
        if artifacts:
            lines.append("## Materialized outputs")
            lines.append("")
            for art in artifacts:
                lines.append(f"- `{art.path or art.filename}`")
            lines.append("")
        else:
            lines.append("_No output artifacts were harvested._")
            lines.append("")
        tail = stdout.strip()
        if tail:
            lines.append("## Run log (tail)")
            lines.append("")
            lines.append("```")
            lines.append("\n".join(tail.splitlines()[-40:]))
            lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def _read_spec_name(project_dir: Path) -> Optional[str]:
        """Read the top-level ``name:`` from astra.yaml without a YAML dependency."""
        spec = project_dir / ASTRA_SPEC_FILENAME
        if not spec.is_file():
            return None
        for line in spec.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"""name:\s*["']?(.+?)["']?\s*$""", line)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_citations(markdown: str) -> List[str]:
        """Extract PMID and URL citations from the report, preserving order.

        Examples:
            >>> LightconeProvider._extract_citations(
            ...     "See PMID: 12345678 and https://example.org/x for details."
            ... )
            ['PMID:12345678', 'https://example.org/x']
        """
        citations: List[str] = []
        seen: set[str] = set()
        for pmid in _PMID_PATTERN.findall(markdown):
            ref = f"PMID:{pmid}"
            if ref not in seen:
                seen.add(ref)
                citations.append(ref)
        for url in _URL_PATTERN.findall(markdown):
            if url not in seen:
                seen.add(url)
                citations.append(url)
        return citations
