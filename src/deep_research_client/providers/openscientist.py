"""OpenScientist research provider.

OpenScientist is an autonomous AI research agent from Berkeley Lab that runs
iterative hypothesis-driven research using PubMed search and code execution.
It produces markdown reports with PMID citations.

This provider submits jobs to a hosted OpenScientist instance (e.g.,
openscientist.io), polls for completion, and downloads the final report.
"""

import asyncio
import base64
from dataclasses import dataclass
import io
import logging
import mimetypes
import os
from pathlib import PurePosixPath
import re
from typing import List, Optional
import zipfile

import httpx

from . import ResearchProvider
from ..exceptions import ProviderNotConfiguredError
from ..models import (
    ResearchArtifact,
    ResearchResult,
    ProviderConfig,
    sanitize_artifact_filename,
)
from ..provider_params import OpenScientistParams
from ..model_cards import ProviderModelCards, create_openscientist_model_cards

logger = logging.getLogger(__name__)

# Job statuses from OpenScientist API
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
IN_PROGRESS_STATUSES = {"pending", "queued", "running", "generating_report", "awaiting_feedback"}

# Default base URL if not overridden
DEFAULT_BASE_URL = "https://www.openscientist.io"


_ALLOWED_ARTIFACT_EXTENSIONS = {
    ".csv",
    ".gif",
    ".htm",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".pdf",
    ".png",
    ".svg",
    ".tsv",
    ".webp",
}
_ARCHIVE_ARTIFACT_EXTENSIONS = {
    ".7z",
    ".bz2",
    ".gz",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
}
_NOISY_ARTIFACT_PREFIXES = (
    ".cache/",
    ".claude/",
    ".git/",
    ".ipynb_checkpoints/",
    ".venv/",
    "__macosx/",
    "cache/",
    "node_modules/",
)
_NOISY_ARTIFACT_SUFFIXES = (".log", ".tmp")
_NOISY_ARTIFACT_NAME_FRAGMENTS = (
    "stderr",
    "stdout",
    "transcript",
)
_REPORT_MARKDOWN_BASENAMES = {"final_report.md", "report.md"}
_ARTIFACT_SOURCE = "openscientist_artifacts_zip"


@dataclass(frozen=True)
class _DownloadedOpenScientistReport:
    """OpenScientist markdown plus selected sidecar artifacts."""

    markdown: str
    artifacts: list[ResearchArtifact]


class OpenScientistProvider(ResearchProvider):
    """Provider for OpenScientist hosted research API.

    Submits research questions as asynchronous jobs, polls for completion,
    and downloads the final markdown report with PMID citations.

    Requires:
        - OPENSCIENTIST_API_KEY env var (format: "name:secret")
        - Optionally OPENSCIENTIST_URL env var (defaults to https://www.openscientist.io)
    """

    credential_label = "OpenScientist"
    credential_env_var = "OPENSCIENTIST_API_KEY"

    def __init__(self, config: ProviderConfig, params: Optional[OpenScientistParams] = None):
        effective_params = params or OpenScientistParams()
        if config.timeout is not None and "timeout" not in effective_params.model_fields_set:
            effective_params = effective_params.model_copy(update={"timeout": config.timeout})

        self.params = effective_params
        super().__init__(config, self.params.model)

        base: str = config.base_url or os.getenv("OPENSCIENTIST_URL") or DEFAULT_BASE_URL
        self.base_url = base.rstrip("/")
        self.api_key = config.api_key

    def get_default_model(self) -> str:
        return "openscientist-autonomous"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        return create_openscientist_model_cards()

    def is_available(self) -> bool:
        if not self.config.enabled:
            return False
        return self.api_key is not None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def research(self, query: str) -> ResearchResult:
        """Submit a research job, poll until completion, and return the report.

        Args:
            query: The research question sent to the OpenScientist agent.

        Returns:
            ResearchResult with markdown report and extracted PMID citations.
        """
        if not self.is_available():
            raise ProviderNotConfiguredError(self.name, self.unavailable_reason())

        if not query or not query.strip():
            raise ValueError("Research query must not be empty.")

        timeout = self.params.timeout
        poll_interval = self.params.poll_interval
        max_iterations = self.params.max_iterations

        # Use a longer default timeout for report/artifact downloads
        default_timeout = httpx.Timeout(30, read=120)
        async with httpx.AsyncClient(timeout=default_timeout, follow_redirects=True) as client:
            # 1. Health check
            await self._health_check(client)

            # 2. Submit job
            job_id = await self._submit_job(client, query, max_iterations)
            logger.info(f"OpenScientist job submitted: {job_id}")

            # 3. Poll for completion
            try:
                final_status = await self._poll_until_done(
                    client, job_id, timeout, poll_interval
                )
            except asyncio.TimeoutError:
                # Cancel the job on timeout
                logger.warning(f"Job {job_id} timed out after {timeout}s, cancelling...")
                await self._cancel_job(client, job_id)
                raise ValueError(
                    f"OpenScientist job timed out after {timeout}s. "
                    f"Job {job_id} has been cancelled."
                )

            if final_status in ("failed", "cancelled"):
                # Get detailed error from job info
                job_info = await self._get_job_detail(client, job_id)
                error_msg = job_info.get("error_message", "Unknown error")
                raise ValueError(
                    f"OpenScientist job {final_status}: {error_msg}"
                )

            # 4. Download report and selected artifacts
            download = await self._download_report_with_artifacts(client, job_id)
            markdown = download.markdown

            # 5. Extract citations
            citations = self._extract_citations(markdown)

            logger.info(
                f"OpenScientist research complete: "
                f"{len(markdown)} chars, {len(citations)} citations, "
                f"{len(download.artifacts)} artifacts"
            )

            return ResearchResult(
                markdown=markdown,
                citations=citations,
                artifacts=download.artifacts,
                provider=self.name,
                query=query,
                model=self.model,
            )

    async def _health_check(self, client: httpx.AsyncClient) -> None:
        """Verify the OpenScientist API is reachable."""
        url = f"{self.base_url}/api/v1/health"
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            logger.debug(f"Health check passed: {resp.json()}")
        except httpx.HTTPError as e:
            raise ValueError(
                f"OpenScientist API not reachable at {self.base_url}: {e}"
            )

    async def _submit_job(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_iterations: int,
    ) -> str:
        """Submit a research job and return the job ID."""
        url = f"{self.base_url}/api/v1/jobs"

        payload = {
            "research_question": query,
            "max_iterations": max_iterations,
            "use_hypotheses": self.params.use_hypotheses,
            "investigation_mode": self.params.investigation_mode,
        }

        resp = await client.post(url, json=payload, headers=self._headers())
        resp.raise_for_status()
        data = resp.json()

        job_id = data.get("job_id") or data.get("id")
        if not job_id:
            raise ValueError(f"No job_id in response: {data}")

        return str(job_id)

    async def _poll_until_done(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        timeout: int,
        poll_interval: int,
    ) -> str:
        """Poll job status until terminal state or timeout.

        Returns the final status string.
        """
        url = f"{self.base_url}/api/v1/jobs/{job_id}/status"
        elapsed = 0

        while elapsed < timeout:
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status", "unknown")
            current_iter = data.get("current_iteration", 0)
            max_iter = data.get("max_iterations", "?")

            if status in TERMINAL_STATUSES:
                logger.info(f"Job {job_id} reached terminal state: {status}")
                return status

            logger.info(
                f"Job {job_id}: {status} "
                f"(iteration {current_iter}/{max_iter}, "
                f"{elapsed}s/{timeout}s elapsed)"
            )

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise asyncio.TimeoutError(
            f"Job {job_id} did not complete within {timeout}s"
        )

    async def _cancel_job(self, client: httpx.AsyncClient, job_id: str) -> None:
        """Cancel a running job.

        Logs a warning on HTTP 4xx errors (e.g. job already finished) but
        re-raises network/transport errors so callers know the request failed.
        """
        url = f"{self.base_url}/api/v1/jobs/{job_id}/cancel"
        try:
            resp = await client.post(url, headers=self._headers())
            resp.raise_for_status()
            logger.info(f"Job {job_id} cancelled successfully")
        except httpx.HTTPStatusError as e:
            # Server responded but rejected the cancel (e.g. already completed)
            logger.warning(f"Failed to cancel job {job_id}: {e}")
        except httpx.HTTPError:
            # Network/transport error — re-raise so callers are aware
            raise

    async def _get_job_detail(self, client: httpx.AsyncClient, job_id: str) -> dict:
        """Get detailed job info."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}"
        resp = await client.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    async def _download_report(self, client: httpx.AsyncClient, job_id: str) -> str:
        """Download the job report as markdown.

        The API returns PDF by default; we request markdown via Accept header,
        and fall back to the artifacts ZIP if needed.
        """
        url = f"{self.base_url}/api/v1/jobs/{job_id}/report"

        # Try requesting with Accept: text/markdown
        headers = {**self._headers(), "Accept": "text/markdown"}
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")

        if "text/" in content_type or "markdown" in content_type:
            return resp.text

        # If we got a PDF or binary, try the artifacts endpoint
        logger.info("Report returned as PDF, trying artifacts endpoint...")
        return await self._extract_markdown_from_artifacts(client, job_id)

    async def _download_report_with_artifacts(
        self,
        client: httpx.AsyncClient,
        job_id: str,
    ) -> _DownloadedOpenScientistReport:
        """Download the job report and selected provider artifacts."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}/report"
        headers = {**self._headers(), "Accept": "text/markdown"}
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/" in content_type or "markdown" in content_type:
            artifacts = await self._download_artifacts(client, job_id)
            return _DownloadedOpenScientistReport(markdown=resp.text, artifacts=artifacts)

        logger.info("Report returned as PDF, trying artifacts endpoint...")
        bundle = await self._download_artifact_bundle(client, job_id)
        report_name, markdown = self._extract_markdown_from_artifact_zip(bundle, job_id)
        artifacts = (
            self._extract_artifacts_from_artifact_zip(bundle, report_names={report_name})
            if self.params.save_artifacts
            else []
        )
        return _DownloadedOpenScientistReport(markdown=markdown, artifacts=artifacts)

    async def _download_artifacts(
        self,
        client: httpx.AsyncClient,
        job_id: str,
    ) -> list[ResearchArtifact]:
        """Download and extract useful OpenScientist sidecar artifacts."""
        if not self.params.save_artifacts:
            return []

        try:
            bundle = await self._download_artifact_bundle(client, job_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {404, 405}:
                logger.info(f"No OpenScientist artifact bundle available for job {job_id}")
                return []
            raise

        return self._extract_artifacts_from_artifact_zip(bundle)

    async def _download_artifact_bundle(
        self,
        client: httpx.AsyncClient,
        job_id: str,
    ) -> bytes:
        """Download the OpenScientist artifacts ZIP for a completed job."""
        url = f"{self.base_url}/api/v1/jobs/{job_id}/artifacts"
        resp = await client.get(url, headers=self._headers())
        resp.raise_for_status()
        return resp.content

    async def _extract_markdown_from_artifacts(
        self, client: httpx.AsyncClient, job_id: str
    ) -> str:
        """Download artifacts ZIP and extract the markdown report."""
        bundle = await self._download_artifact_bundle(client, job_id)
        return self._extract_markdown_from_artifact_zip(bundle, job_id)[1]

    def _extract_markdown_from_artifact_zip(
        self,
        bundle: bytes,
        job_id: str,
    ) -> tuple[str, str]:
        """Extract the best markdown report from an OpenScientist artifact ZIP."""
        with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
            zip_names = zf.namelist()
            # Prefer canonical report filenames over similarly named runtime skills.
            md_files = [
                n for n in zip_names
                if self._is_report_markdown_name(n) and not self._is_noisy_artifact_path(n)
            ]
            if not md_files:
                md_files = [
                    n for n in zip_names
                    if n.lower().endswith(".md")
                    and "report" in n.lower()
                    and not self._is_noisy_artifact_path(n)
                ]
            if not md_files:
                # Fall back to any non-runtime .md file.
                md_files = [
                    n for n in zip_names
                    if n.lower().endswith(".md")
                    and not self._is_noisy_artifact_path(n)
                ]

            if not md_files:
                raise ValueError(
                    f"No markdown files found in artifacts ZIP for job {job_id}"
                )

            # Read the first matching file
            report_name = md_files[0]
            with zf.open(report_name) as f:
                raw = f.read()
                try:
                    return report_name, raw.decode("utf-8")
                except UnicodeDecodeError:
                    return report_name, raw.decode("latin-1")

    def _extract_artifacts_from_artifact_zip(
        self,
        bundle: bytes,
        report_names: set[str] | None = None,
    ) -> list[ResearchArtifact]:
        """Extract useful sidecar artifacts from an OpenScientist ZIP archive."""
        report_names = report_names or set()
        artifacts: list[ResearchArtifact] = []
        used_filenames: set[str] = set()

        with zipfile.ZipFile(io.BytesIO(bundle)) as zf:
            for info in sorted(zf.infolist(), key=lambda item: item.filename):
                if not self._should_preserve_artifact(info, report_names):
                    continue

                with zf.open(info) as file:
                    content = file.read()

                filename = self._artifact_filename(info.filename, used_filenames)
                media_type = mimetypes.guess_type(info.filename)[0]
                artifacts.append(
                    ResearchArtifact(
                        filename=filename,
                        content_base64=base64.b64encode(content).decode("ascii"),
                        media_type=media_type,
                        source=_ARTIFACT_SOURCE,
                        description=self._artifact_description(info.filename),
                    )
                )

        return artifacts

    def _should_preserve_artifact(
        self,
        info: zipfile.ZipInfo,
        report_names: set[str],
    ) -> bool:
        """Return whether a ZIP member should become a ResearchArtifact."""
        if info.is_dir():
            return False

        name = info.filename
        if name in report_names:
            return False

        if self._is_report_markdown_name(name) or self._is_noisy_artifact_path(name):
            return False

        if info.file_size > self.params.artifact_max_bytes:
            logger.info(
                "Skipping OpenScientist artifact %s: %s bytes exceeds %s byte limit",
                name,
                info.file_size,
                self.params.artifact_max_bytes,
            )
            return False

        path = PurePosixPath(name.lower())
        suffix = path.suffix
        if suffix in _ARCHIVE_ARTIFACT_EXTENSIONS:
            return False

        media_type = mimetypes.guess_type(name)[0]
        return suffix in _ALLOWED_ARTIFACT_EXTENSIONS or (
            media_type is not None and media_type.startswith("image/")
        )

    def _is_report_markdown_name(self, name: str) -> bool:
        """Return whether a ZIP member is the markdown report already captured."""
        path = PurePosixPath(name.lower())
        return path.name in _REPORT_MARKDOWN_BASENAMES

    def _is_noisy_artifact_path(self, name: str) -> bool:
        """Return whether a ZIP member is runtime scaffolding or verbose logs."""
        normalized = PurePosixPath(name).as_posix().lstrip("/").lower()
        basename = PurePosixPath(normalized).name
        if normalized.startswith(_NOISY_ARTIFACT_PREFIXES):
            return True
        if basename.endswith(_NOISY_ARTIFACT_SUFFIXES):
            return True
        return any(fragment in basename for fragment in _NOISY_ARTIFACT_NAME_FRAGMENTS)

    def _artifact_filename(self, raw_name: str, used_filenames: set[str]) -> str:
        """Return a stable, unique filename for a ZIP artifact member."""
        filename = sanitize_artifact_filename(raw_name, fallback="artifact")
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

    def _artifact_description(self, name: str) -> str:
        """Build a short human-readable artifact description."""
        path = PurePosixPath(name)
        stem = path.stem.replace("_", " ").replace("-", " ").strip()
        if path.name.lower() == "final_report.pdf":
            return "OpenScientist final report"
        if stem:
            return f"OpenScientist {stem}"
        return f"OpenScientist artifact {path.name}"

    @staticmethod
    def _extract_citations(markdown: str) -> List[str]:
        """Extract PMID citations from the markdown report.

        Looks for patterns like:
            [PMID: 12345678](...)
            PMID:12345678
            PMID: 12345678
            (PMID: 12345678)
        """
        # Match PMID patterns
        pmid_pattern = r"PMID:\s*(\d{7,8})"
        pmids = re.findall(pmid_pattern, markdown)

        # Deduplicate while preserving order
        seen = set()
        unique_citations = []
        for pmid in pmids:
            ref = f"PMID:{pmid}"
            if ref not in seen:
                seen.add(ref)
                unique_citations.append(ref)

        return unique_citations
