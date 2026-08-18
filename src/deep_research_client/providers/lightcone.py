"""Lightcone / ASTRA research provider.

Unlike the literature providers, Lightcone is not a "query in, report out"
service. `Lightcone <https://lightconeresearch.org>`_ is a local command-line
tool (the ``lc`` binary) that materializes an **ASTRA** (*Agentic Schema for
Transparent Research Analysis*) specification -- an ``astra.yaml`` file -- into
a tree of outputs, recording the methodological decisions and their provenance
along the way. ``lc run`` generates a Snakefile and dispatches it through
Snakemake + Dask; each materialized output is written with a sidecar
``.lightcone-manifest.json`` recording its provenance.

This provider is therefore a *spec runner*: the research "query" is a path to
an ASTRA project directory (or an ``astra.yaml`` file), not a free-text
question. The spec already declares the inputs, decisions, and expected
outputs; the provider drives ``lc`` to execute them in the project directory,
then discovers the materialized outputs via their manifest sidecars and returns
them in the standard :class:`~deep_research_client.models.ResearchResult`
(report ``markdown``, non-text ``artifacts``, and a per-output provenance trail
in ``run_metadata``).

No provider API key is required: Lightcone drives a local agent (Claude Code),
so authentication and billing are handled by that local installation.

Security
--------
Materializing an ASTRA spec runs an agent that executes code (potentially in
containers via ``lc build``). Only run Lightcone against trusted projects in a
sandboxed environment.
"""

import asyncio
import base64
import json
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

# Sidecar manifest `lc` writes next to each materialized output.
MANIFEST_FILENAME = ".lightcone-manifest.json"

# Provenance fields lifted from a manifest into run_metadata.
_MANIFEST_PROVENANCE_FIELDS = (
    "output_id",
    "universe_id",
    "code_version",
    "data_version",
    "container_image",
    "recipe",
    "decisions",
    "input_versions",
    "lc_version",
    "git_sha",
    "finished_at",
)

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
    ".snakemake/",
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
        an argument. Mirrors ``lc run [OPTIONS] [OUTPUTS]...``. This method is
        pure and side-effect free to keep it easy to unit test.

        Returns:
            The argument list to pass to the subprocess.

        Examples:
            >>> from deep_research_client.models import ProviderConfig
            >>> from deep_research_client.provider_params import LightconeParams
            >>> p = LightconeProvider(
            ...     ProviderConfig(name="lightcone", api_key=None, enabled=True),
            ...     LightconeParams(universe="baseline", jobs=4, outputs=["accuracy"]),
            ... )
            >>> p._build_command()
            ['lc', 'run', '--universe', 'baseline', '--jobs', '4', 'accuracy']
        """
        command: List[str] = [self.lc_executable, *self.params.materialize_args]
        if self.params.universe:
            command.extend(["--universe", self.params.universe])
        if self.params.jobs is not None:
            command.extend(["--jobs", str(self.params.jobs)])
        if self.params.force:
            command.append("--force")
        command.extend(self.params.extra_args)
        command.extend(self.params.outputs)
        return command

    async def research(self, query: str) -> ResearchResult:
        """Materialize an ASTRA spec by running the Lightcone CLI.

        Args:
            query: Path to an ASTRA project directory or ``astra.yaml`` file.

        Returns:
            ResearchResult with the materialized report, harvested artifacts,
            and a per-output provenance trail in ``run_metadata``.

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

        scan_root = project_dir / self.params.scan_subdir if self.params.scan_subdir else project_dir
        manifest_dirs = self._find_manifest_dirs(scan_root)
        owned_files = self._owned_files(scan_root, set(manifest_dirs))

        report_name, markdown, report_path = self._select_report(owned_files, project_dir)
        artifacts = (
            self._harvest_artifacts(owned_files, project_dir, manifest_dirs, skip=report_path)
            if self.params.save_artifacts
            else []
        )
        provenance = [self._summarize_manifest(m) for m in manifest_dirs.values() if m]

        if markdown is None:
            markdown = self._synthesize_report(project_dir, artifacts, provenance, stdout)

        citations = self._extract_citations(markdown)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(
            "Lightcone run completed in %.1fs (%d chars, %d citations, %d artifacts, %d outputs)",
            duration,
            len(markdown),
            len(citations),
            len(artifacts),
            len(provenance),
        )

        return ResearchResult(
            markdown=markdown,
            citations=citations,
            artifacts=artifacts,
            provider=self.name,
            query=query,
            model=self.model,
            run_metadata={
                "project_dir": str(project_dir),
                "universe": self.params.universe,
                "outputs": provenance,
            },
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

    # --- Output discovery via manifest sidecars ----------------------------

    def _find_manifest_dirs(self, scan_root: Path) -> dict[Path, Optional[dict]]:
        """Map each output directory (containing a manifest) to its parsed manifest.

        An output directory is any directory holding a ``.lightcone-manifest.json``
        sidecar. A manifest that fails to parse maps to ``None`` (the output is
        still harvested; only its provenance is dropped).
        """
        manifest_dirs: dict[Path, Optional[dict]] = {}
        if not scan_root.is_dir():
            return manifest_dirs

        for manifest_path in sorted(scan_root.rglob(MANIFEST_FILENAME)):
            if not manifest_path.is_file():
                continue
            manifest_dirs[manifest_path.parent] = self._read_manifest(manifest_path)
        return manifest_dirs

    @staticmethod
    def _read_manifest(manifest_path: Path) -> Optional[dict]:
        """Parse a manifest sidecar, returning ``None`` on malformed JSON."""
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            logger.warning("Ignoring malformed Lightcone manifest: %s", manifest_path)
            return None
        return data if isinstance(data, dict) else None

    def _owned_files(self, scan_root: Path, manifest_dirs: set[Path]) -> List[Path]:
        """List materialized files, each owned by its nearest ancestor manifest dir.

        Files with no manifest ancestor (project inputs, the spec itself) are
        excluded, so only genuinely materialized outputs are considered. The
        manifest sidecars themselves are excluded too.
        """
        if not scan_root.is_dir() or not manifest_dirs:
            return []

        owned: List[Path] = []
        for path in sorted(scan_root.rglob("*")):
            if path.name == MANIFEST_FILENAME:
                continue
            if path.is_symlink() or not path.is_file():
                continue
            if self._nearest_manifest_dir(path, manifest_dirs) is not None:
                owned.append(path)
        return owned

    @staticmethod
    def _nearest_manifest_dir(path: Path, manifest_dirs: set[Path]) -> Optional[Path]:
        """Return the deepest manifest directory that is an ancestor of ``path``."""
        for parent in path.parents:
            if parent in manifest_dirs:
                return parent
        return None

    def _select_report(
        self, owned_files: List[Path], project_dir: Path
    ) -> tuple[Optional[str], Optional[str], Optional[Path]]:
        """Pick the best markdown report among materialized outputs.

        Prefers canonical report basenames, then any ``report*.md``, then any
        ``.md`` file. Returns ``(relative_name, markdown, abs_path)`` or
        ``(None, None, None)`` when no report was materialized.
        """
        md_files = [
            p
            for p in owned_files
            if p.suffix.lower() == ".md"
            and not self._is_noisy_artifact_path(p.relative_to(project_dir))
        ]
        if not md_files:
            return None, None, None

        def rank(path: Path) -> tuple[int, str]:
            name = path.name.lower()
            if name in _REPORT_MARKDOWN_BASENAMES:
                return (0, str(path))
            if "report" in name:
                return (1, str(path))
            return (2, str(path))

        chosen = min(md_files, key=rank)
        markdown = chosen.read_text(encoding="utf-8", errors="replace")
        return str(chosen.relative_to(project_dir).as_posix()), markdown, chosen

    def _harvest_artifacts(
        self,
        owned_files: List[Path],
        project_dir: Path,
        manifest_dirs: dict[Path, Optional[dict]],
        skip: Optional[Path],
    ) -> List[ResearchArtifact]:
        """Harvest materialized outputs (except the chosen report) as artifacts.

        Each artifact is tagged with its owning output's id/universe (from the
        manifest) for provenance.
        """
        artifacts: List[ResearchArtifact] = []
        used_filenames: set[str] = set()

        for path in owned_files:
            if skip is not None and path == skip:
                continue
            rel = path.relative_to(project_dir)
            if not self._should_preserve_artifact(path, rel):
                continue

            owner = self._nearest_manifest_dir(path, set(manifest_dirs))
            manifest = manifest_dirs.get(owner) if owner is not None else None
            rel_posix = rel.as_posix()
            artifacts.append(
                ResearchArtifact(
                    filename=self._artifact_filename(rel_posix, used_filenames),
                    content_base64=base64.b64encode(path.read_bytes()).decode("ascii"),
                    media_type=mimetypes.guess_type(path.name)[0],
                    path=rel_posix,
                    source="lightcone_outputs",
                    description=self._artifact_description(rel_posix, manifest),
                )
            )

        return artifacts

    def _should_preserve_artifact(self, path: Path, rel: Path) -> bool:
        """Return whether a materialized file should become a ResearchArtifact."""
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
    def _artifact_description(rel_posix: str, manifest: Optional[dict]) -> str:
        """Build a short human-readable artifact description with provenance."""
        stem = PurePosixPath(rel_posix).stem.replace("_", " ").replace("-", " ").strip()
        base = f"Lightcone output {stem}" if stem else f"Lightcone output {rel_posix}"
        if manifest:
            output_id = manifest.get("output_id")
            universe = manifest.get("universe_id")
            tags = [t for t in (output_id, universe) if t]
            if tags:
                base += f" ({' / '.join(str(t) for t in tags)})"
        return base

    @staticmethod
    def _summarize_manifest(manifest: dict) -> dict:
        """Extract the provenance-relevant fields from a manifest for run_metadata."""
        return {k: manifest[k] for k in _MANIFEST_PROVENANCE_FIELDS if k in manifest}

    def _synthesize_report(
        self,
        project_dir: Path,
        artifacts: List[ResearchArtifact],
        provenance: List[dict],
        stdout: str,
    ) -> str:
        """Build a fallback markdown report when no report file was materialized."""
        name = self._read_spec_name(project_dir) or project_dir.name
        lines = [f"# Lightcone analysis: {name}", ""]
        if self.params.universe:
            lines.append(f"Universe: `{self.params.universe}`")
            lines.append("")
        if provenance:
            lines.append("## Materialized outputs")
            lines.append("")
            for entry in provenance:
                oid = entry.get("output_id", "?")
                uid = entry.get("universe_id")
                recipe = entry.get("recipe")
                suffix = f" — universe `{uid}`" if uid else ""
                recipe_note = f" (`{recipe}`)" if recipe else ""
                lines.append(f"- **{oid}**{suffix}{recipe_note}")
            lines.append("")
        if artifacts:
            lines.append("## Output files")
            lines.append("")
            for art in artifacts:
                lines.append(f"- `{art.path or art.filename}`")
            lines.append("")
        if not provenance and not artifacts:
            lines.append("_No materialized outputs were found._")
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
