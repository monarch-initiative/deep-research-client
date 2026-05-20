"""Tests for the OpenScientist provider."""

import asyncio
import io
import os
import zipfile
from pathlib import Path

import httpx
import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import OpenScientistParams
from deep_research_client.providers.openscientist import OpenScientistProvider


def _load_openscientist_api_key() -> str:
    """Load the OpenScientist API key from env or the repo-local helper file."""
    api_key = os.getenv("OPENSCIENTIST_API_KEY", "").strip()
    if api_key:
        return api_key

    key_path = Path(__file__).resolve().parents[1] / "opensci"
    if key_path.exists():
        file_key = key_path.read_text(encoding="utf-8").strip()
        if file_key:
            return file_key

    pytest.skip("OPENSCIENTIST_API_KEY not set and opensci file not present")


def _load_openscientist_base_url() -> str:
    """Load the OpenScientist base URL for integration tests."""
    return os.getenv("OPENSCIENTIST_URL", "https://www.openscientist.io").strip()


def create_zip_bytes(files: dict[str, str]) -> bytes:
    """Create an in-memory ZIP archive for artifact download tests."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class DummyAsyncClient:
    """Minimal async client context manager for orchestration tests."""

    def __init__(self, *args, **kwargs):
        """Accept the same constructor shape as httpx.AsyncClient."""

    async def __aenter__(self):
        """Enter the async context."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit the async context."""
        return False


class TestOpenScientistProvider:
    """Test cases for OpenScientistProvider."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ProviderConfig(
            name="openscientist",
            api_key="test-api-key",
            base_url="https://example.test/",
            enabled=True,
        )
        self.provider = OpenScientistProvider(self.config)

    def test_provider_initialization(self):
        """Test provider initialization."""
        assert self.provider.name == "openscientist"
        assert self.provider.api_key == "test-api-key"
        assert self.provider.base_url == "https://example.test"
        assert self.provider.model == "openscientist-autonomous"
        assert self.provider.is_available() is True

    def test_provider_initialization_with_custom_params(self):
        """Test provider initialization with custom parameters."""
        params = OpenScientistParams(
            model="os",
            max_iterations=7,
            use_hypotheses=True,
            investigation_mode="coinvestigate",
            poll_interval=15,
            timeout=900,
        )

        provider = OpenScientistProvider(self.config, params)

        assert provider.model == "openscientist-autonomous"
        assert provider.params.max_iterations == 7
        assert provider.params.use_hypotheses is True
        assert provider.params.investigation_mode == "coinvestigate"
        assert provider.params.poll_interval == 15
        assert provider.params.timeout == 900

    def test_provider_uses_config_timeout_as_default(self):
        """ProviderConfig.timeout should become the effective timeout by default."""
        config = ProviderConfig(
            name="openscientist",
            api_key="test-api-key",
            base_url="https://example.test/",
            enabled=True,
            timeout=123,
        )

        provider = OpenScientistProvider(config)

        assert provider.config.timeout == 123
        assert provider.params.timeout == 123

    def test_provider_params_timeout_overrides_config_timeout(self):
        """Explicit provider params should take precedence over ProviderConfig.timeout."""
        config = ProviderConfig(
            name="openscientist",
            api_key="test-api-key",
            base_url="https://example.test/",
            enabled=True,
            timeout=123,
        )
        params = OpenScientistParams(timeout=900)

        provider = OpenScientistProvider(config, params)

        assert provider.config.timeout == 123
        assert provider.params.timeout == 900

    def test_get_default_model(self):
        """Test default model."""
        assert self.provider.get_default_model() == "openscientist-autonomous"

    def test_provider_without_api_key(self):
        """Test provider initialization without API key."""
        config = ProviderConfig(name="openscientist", api_key=None, enabled=True)
        provider = OpenScientistProvider(config)
        assert provider.is_available() is False

    def test_provider_disabled(self):
        """Test provider availability when disabled."""
        config = ProviderConfig(
            name="openscientist",
            api_key="test-api-key",
            enabled=False,
        )
        provider = OpenScientistProvider(config)
        assert provider.is_available() is False

    def test_headers(self):
        """Test request headers."""
        assert self.provider._headers() == {
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json",
        }

    def test_extract_citations(self):
        """Test PMID citation extraction and deduplication."""
        markdown = """
        First citation [PMID: 12345678](https://pubmed.ncbi.nlm.nih.gov/12345678/).
        Duplicate mention PMID:12345678 should be ignored.
        Another citation (PMID: 23456789) appears here.
        PMID: 34567890 appears in plain text.
        """

        citations = self.provider._extract_citations(markdown)

        assert citations == [
            "PMID:12345678",
            "PMID:23456789",
            "PMID:34567890",
        ]

    async def test_research_provider_not_available(self):
        """Test research when provider is not available."""
        provider = OpenScientistProvider(
            ProviderConfig(name="openscientist", api_key=None, enabled=True)
        )

        with pytest.raises(ValueError, match="OpenScientist provider not available"):
            await provider.research("test query")

    async def test_research_empty_query(self):
        """Test research rejects empty queries."""
        with pytest.raises(ValueError, match="Research query must not be empty"):
            await self.provider.research("   ")

    async def test_submit_job_uses_expected_payload(self):
        """Test job submission payload and job ID extraction."""
        provider = OpenScientistProvider(
            self.config,
            OpenScientistParams(
                max_iterations=7,
                use_hypotheses=True,
                investigation_mode="coinvestigate",
            ),
        )
        seen = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["auth"] = request.headers["Authorization"]
            seen["content_type"] = request.headers["Content-Type"]
            seen["body"] = request.read().decode("utf-8")
            return httpx.Response(200, json={"job_id": "job-123"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            job_id = await provider._submit_job(
                client,
                "What regulates autophagy in cancer cells?",
                provider.params.max_iterations,
            )

        assert job_id == "job-123"
        assert seen["url"] == "https://example.test/api/v1/jobs"
        assert seen["auth"] == "Bearer test-api-key"
        assert seen["content_type"] == "application/json"
        assert '"research_question":"What regulates autophagy in cancer cells?"' in seen["body"]
        assert '"max_iterations":7' in seen["body"]
        assert '"use_hypotheses":true' in seen["body"]
        assert '"investigation_mode":"coinvestigate"' in seen["body"]

    async def test_submit_job_accepts_id_fallback(self):
        """Test job submission accepts an id field when job_id is absent."""
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": "job-456"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            job_id = await self.provider._submit_job(client, "test query", 5)

        assert job_id == "job-456"

    async def test_submit_job_requires_job_identifier(self):
        """Test job submission fails when no job identifier is returned."""
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "accepted"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError, match="No job_id in response"):
                await self.provider._submit_job(client, "test query", 5)

    async def test_poll_until_done_returns_terminal_status(self, monkeypatch):
        """Test polling until a job reaches completion."""
        responses = iter(
            [
                {"status": "running", "current_iteration": 1, "max_iterations": 5},
                {"status": "completed", "current_iteration": 5, "max_iterations": 5},
            ]
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=next(responses))

        async def fake_sleep(_seconds: int) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            status = await self.provider._poll_until_done(
                client,
                "job-123",
                timeout=60,
                poll_interval=5,
            )

        assert status == "completed"

    async def test_download_report_returns_markdown(self):
        """Test report download when the API directly returns markdown."""
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Accept"] == "text/markdown"
            return httpx.Response(
                200,
                text="# Report\n\nPMID: 12345678",
                headers={"content-type": "text/markdown; charset=utf-8"},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            markdown = await self.provider._download_report(client, "job-123")

        assert markdown == "# Report\n\nPMID: 12345678"

    async def test_download_report_falls_back_to_artifacts_zip(self):
        """Test report download falls back to ZIP artifacts for binary responses."""
        zip_bytes = create_zip_bytes(
            {
                "notes.txt": "ignored",
                "reports/final_report.md": "# Final Report\n\nPMID: 12345678",
            }
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/report"):
                return httpx.Response(
                    200,
                    content=b"%PDF-1.4 fake pdf",
                    headers={"content-type": "application/pdf"},
                )

            if request.url.path.endswith("/artifacts"):
                return httpx.Response(
                    200,
                    content=zip_bytes,
                    headers={"content-type": "application/zip"},
                )

            raise AssertionError(f"Unexpected request path: {request.url.path}")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            markdown = await self.provider._download_report(client, "job-123")

        assert markdown == "# Final Report\n\nPMID: 12345678"

    async def test_research_success(self, monkeypatch):
        """Test the main research orchestration path."""
        async def fake_health_check(self, client) -> None:
            return None

        async def fake_submit_job(self, client, query: str, max_iterations: int) -> str:
            assert query == "What is autophagy?"
            assert max_iterations == 5
            return "job-123"

        async def fake_poll_until_done(self, client, job_id: str, timeout: int, poll_interval: int) -> str:
            assert job_id == "job-123"
            assert timeout == 3600
            assert poll_interval == 30
            return "completed"

        async def fake_download_report(self, client, job_id: str) -> str:
            assert job_id == "job-123"
            return "# Report\n\nPMID: 12345678\n\nPMID: 23456789"

        async def fake_download_artifacts(self, client, job_id: str):
            return [], 0

        monkeypatch.setattr(
            "deep_research_client.providers.openscientist.httpx.AsyncClient",
            DummyAsyncClient,
        )
        monkeypatch.setattr(OpenScientistProvider, "_health_check", fake_health_check)
        monkeypatch.setattr(OpenScientistProvider, "_submit_job", fake_submit_job)
        monkeypatch.setattr(OpenScientistProvider, "_poll_until_done", fake_poll_until_done)
        monkeypatch.setattr(OpenScientistProvider, "_download_report", fake_download_report)
        monkeypatch.setattr(OpenScientistProvider, "_download_artifacts", fake_download_artifacts)

        result = await self.provider.research("What is autophagy?")

        assert result.provider == "openscientist"
        assert result.query == "What is autophagy?"
        assert result.model == "openscientist-autonomous"
        assert result.markdown == "# Report\n\nPMID: 12345678\n\nPMID: 23456789"
        assert result.citations == ["PMID:12345678", "PMID:23456789"]
        assert result.artifacts == []

    async def test_research_timeout_cancels_job(self, monkeypatch):
        """Test timeout handling cancels the submitted job."""
        seen = {"cancelled": None}

        async def fake_health_check(self, client) -> None:
            return None

        async def fake_submit_job(self, client, query: str, max_iterations: int) -> str:
            return "job-timeout"

        async def fake_poll_until_done(self, client, job_id: str, timeout: int, poll_interval: int) -> str:
            raise asyncio.TimeoutError("timed out")

        async def fake_cancel_job(self, client, job_id: str) -> None:
            seen["cancelled"] = job_id

        monkeypatch.setattr(
            "deep_research_client.providers.openscientist.httpx.AsyncClient",
            DummyAsyncClient,
        )
        monkeypatch.setattr(OpenScientistProvider, "_health_check", fake_health_check)
        monkeypatch.setattr(OpenScientistProvider, "_submit_job", fake_submit_job)
        monkeypatch.setattr(OpenScientistProvider, "_poll_until_done", fake_poll_until_done)
        monkeypatch.setattr(OpenScientistProvider, "_cancel_job", fake_cancel_job)

        with pytest.raises(ValueError, match="timed out after 3600s"):
            await self.provider.research("What is autophagy?")

        assert seen["cancelled"] == "job-timeout"

    async def test_research_failed_job_surfaces_error(self, monkeypatch):
        """Test failed jobs include the provider error message."""
        async def fake_health_check(self, client) -> None:
            return None

        async def fake_submit_job(self, client, query: str, max_iterations: int) -> str:
            return "job-failed"

        async def fake_poll_until_done(self, client, job_id: str, timeout: int, poll_interval: int) -> str:
            return "failed"

        async def fake_get_job_detail(self, client, job_id: str) -> dict:
            return {"error_message": "Agent execution failed"}

        monkeypatch.setattr(
            "deep_research_client.providers.openscientist.httpx.AsyncClient",
            DummyAsyncClient,
        )
        monkeypatch.setattr(OpenScientistProvider, "_health_check", fake_health_check)
        monkeypatch.setattr(OpenScientistProvider, "_submit_job", fake_submit_job)
        monkeypatch.setattr(OpenScientistProvider, "_poll_until_done", fake_poll_until_done)
        monkeypatch.setattr(OpenScientistProvider, "_get_job_detail", fake_get_job_detail)

        with pytest.raises(ValueError, match="OpenScientist job failed: Agent execution failed"):
            await self.provider.research("What is autophagy?")


class TestOpenScientistArtifacts:
    """Tests for OpenScientist artifact extraction from the job artifacts ZIP."""

    def setup_method(self):
        self.config = ProviderConfig(
            name="openscientist",
            api_key="test-api-key",
            base_url="https://example.test/",
            enabled=True,
        )
        self.provider = OpenScientistProvider(self.config)

    # ------------------------------------------------------------------
    # _should_exclude_artifact_path
    # ------------------------------------------------------------------

    def test_excludes_dot_claude_paths(self):
        assert OpenScientistProvider._should_exclude_artifact_path(".claude/settings.json") is True
        assert OpenScientistProvider._should_exclude_artifact_path("work/.claude/skills/foo.py") is True

    def test_excludes_skills_paths(self):
        assert OpenScientistProvider._should_exclude_artifact_path("skills/helper.py") is True

    def test_excludes_logs_paths(self):
        assert OpenScientistProvider._should_exclude_artifact_path("output/logs/run.log") is True

    def test_excludes_transcripts_paths(self):
        assert OpenScientistProvider._should_exclude_artifact_path("transcripts/session.txt") is True
        assert OpenScientistProvider._should_exclude_artifact_path("transcript_20260101.txt") is True

    def test_allows_normal_artifact_paths(self):
        assert OpenScientistProvider._should_exclude_artifact_path("evidence_matrix.png") is False
        assert OpenScientistProvider._should_exclude_artifact_path("results/knowledge_gaps.json") is False
        assert OpenScientistProvider._should_exclude_artifact_path("final_summary.md") is False

    # ------------------------------------------------------------------
    # _unique_artifact_filename
    # ------------------------------------------------------------------

    def test_unique_filename_no_collision(self):
        used: set[str] = set()
        name = OpenScientistProvider._unique_artifact_filename("figure.png", used)
        assert name == "figure.png"
        assert "figure.png" in used

    def test_unique_filename_collision_adds_counter(self):
        used = {"figure.png"}
        name = OpenScientistProvider._unique_artifact_filename("figure.png", used)
        assert name == "figure-2.png"

    def test_unique_filename_multiple_collisions(self):
        used = {"figure.png", "figure-2.png"}
        name = OpenScientistProvider._unique_artifact_filename("figure.png", used)
        assert name == "figure-3.png"

    def test_unique_filename_sanitizes_unsafe_chars(self):
        used: set[str] = set()
        name = OpenScientistProvider._unique_artifact_filename("fig/ure:1.png", used)
        assert "/" not in name
        assert ":" not in name

    # ------------------------------------------------------------------
    # _download_artifacts
    # ------------------------------------------------------------------

    async def test_download_artifacts_extracts_images_and_tables(self):
        """Useful artifact types are extracted from the ZIP."""
        zip_bytes = create_zip_bytes({
            "evidence_matrix.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
            "knowledge_gaps.json": b'{"gaps": ["gap1"]}',
            "final_summary.md": b"# Summary\n\nKey findings.",
            "results/table.csv": b"col1,col2\nval1,val2",
        })

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/artifacts")
            return httpx.Response(200, content=zip_bytes,
                                  headers={"content-type": "application/zip"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, skipped = await self.provider._download_artifacts(client, "job-123")

        assert len(artifacts) == 4
        filenames = {a.filename for a in artifacts}
        assert "evidence_matrix.png" in filenames
        assert "knowledge_gaps.json" in filenames
        assert "final_summary.md" in filenames
        assert "table.csv" in filenames
        assert all(a.source == "openscientist_artifacts" for a in artifacts)
        assert skipped == 0

    async def test_download_artifacts_skips_excluded_paths(self):
        """Files in .claude/, skills/, logs/, transcripts/ are excluded."""
        zip_bytes = create_zip_bytes({
            ".claude/settings.json": b"{}",
            "skills/helper.py": b"def foo(): pass",
            "output/logs/run.log": b"log line",
            "transcripts/session.txt": b"session data",
            "evidence_matrix.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
        })

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=zip_bytes,
                                  headers={"content-type": "application/zip"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, skipped = await self.provider._download_artifacts(client, "job-123")

        assert len(artifacts) == 1
        assert artifacts[0].filename == "evidence_matrix.png"
        assert skipped == 4

    async def test_download_artifacts_skips_non_allowed_extensions(self):
        """Files with non-whitelisted extensions are excluded."""
        zip_bytes = create_zip_bytes({
            "run.py": b"import os",
            "binary.bin": b"\x00\x01\x02",
            "useful.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
        })

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=zip_bytes,
                                  headers={"content-type": "application/zip"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, skipped = await self.provider._download_artifacts(client, "job-123")

        assert len(artifacts) == 1
        assert artifacts[0].filename == "useful.png"
        assert skipped == 2

    async def test_download_artifacts_skips_oversized_files(self):
        """Files exceeding artifact_max_size are excluded."""
        large_content = b"x" * (6 * 1024 * 1024)  # 6 MB
        zip_bytes = create_zip_bytes({
            "large_data.json": large_content.decode("latin-1"),
            "small.json": b'{"ok": true}',
        })

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=zip_bytes,
                                  headers={"content-type": "application/zip"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, skipped = await self.provider._download_artifacts(client, "job-123")

        assert len(artifacts) == 1
        assert artifacts[0].filename == "small.json"
        assert skipped == 1

    async def test_download_artifacts_skips_main_report_md(self):
        """The main report.md is not duplicated as an artifact."""
        zip_bytes = create_zip_bytes({
            "final_report.md": "# Report\n\nMain output.",
            "knowledge_gaps.md": "# Knowledge Gaps\n\nGap 1.",
        })

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=zip_bytes,
                                  headers={"content-type": "application/zip"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, skipped = await self.provider._download_artifacts(client, "job-123")

        # final_report.md should be skipped; knowledge_gaps.md should be kept
        filenames = {a.filename for a in artifacts}
        assert "final_report.md" not in filenames
        assert "knowledge_gaps.md" in filenames
        assert skipped == 1

    async def test_download_artifacts_handles_bad_zip(self):
        """A non-ZIP response is handled gracefully."""
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not a zip",
                                  headers={"content-type": "application/zip"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, skipped = await self.provider._download_artifacts(client, "job-123")

        assert artifacts == []
        assert skipped == 0

    async def test_download_artifacts_handles_http_error(self):
        """An HTTP error on the artifacts endpoint is handled gracefully."""
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"detail": "not found"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, skipped = await self.provider._download_artifacts(client, "job-123")

        assert artifacts == []
        assert skipped == 0

    async def test_download_artifacts_deduplicates_filenames(self):
        """Collision-prone filenames from different ZIP paths get unique names."""
        zip_bytes = create_zip_bytes({
            "dir_a/figure.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
            "dir_b/figure.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
        })

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=zip_bytes,
                                  headers={"content-type": "application/zip"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, _ = await self.provider._download_artifacts(client, "job-123")

        assert len(artifacts) == 2
        assert artifacts[0].filename != artifacts[1].filename

    async def test_download_artifacts_respects_custom_include_extensions(self):
        """artifact_include_extensions overrides the default allowed extensions."""
        params = OpenScientistParams(artifact_include_extensions=[".svg"])
        provider = OpenScientistProvider(self.config, params)

        zip_bytes = create_zip_bytes({
            "chart.svg": b"<svg/>",
            "data.json": b'{"x": 1}',
        })

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=zip_bytes,
                                  headers={"content-type": "application/zip"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, skipped = await provider._download_artifacts(client, "job-123")

        assert len(artifacts) == 1
        assert artifacts[0].filename == "chart.svg"
        assert skipped == 1

    async def test_download_artifacts_respects_custom_max_size(self):
        """artifact_max_size is respected when set explicitly."""
        params = OpenScientistParams(artifact_max_size=1024)
        provider = OpenScientistProvider(self.config, params)

        zip_bytes = create_zip_bytes({
            "tiny.json": b'{"x":1}',                      # 7 bytes — under limit
            "large.json": b'{"x": ' + b"1" * 2000 + b"}",  # > 1024 bytes — over limit
        })

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=zip_bytes,
                                  headers={"content-type": "application/zip"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            artifacts, skipped = await provider._download_artifacts(client, "job-123")

        assert len(artifacts) == 1
        assert artifacts[0].filename == "tiny.json"
        assert skipped == 1

    # ------------------------------------------------------------------
    # research() integration with artifacts
    # ------------------------------------------------------------------

    async def test_research_success_includes_artifacts(self, monkeypatch):
        """research() returns ResearchResult.artifacts populated from the ZIP."""
        zip_bytes = create_zip_bytes({
            "evidence_matrix.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 16,
            "knowledge_gaps.json": b'{"gaps": []}',
        })

        async def fake_health_check(self, client) -> None:
            return None

        async def fake_submit_job(self, client, query, max_iterations) -> str:
            return "job-with-artifacts"

        async def fake_poll_until_done(self, client, job_id, timeout, poll_interval) -> str:
            return "completed"

        async def fake_download_report(self, client, job_id) -> str:
            return "# Report\n\nPMID: 12345678"

        async def fake_download_artifacts(self, client, job_id):
            from deep_research_client.models import ResearchArtifact
            import base64
            artifact = ResearchArtifact(
                filename="evidence_matrix.png",
                content_base64=base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode(),
                media_type="image/png",
                source="openscientist_artifacts",
            )
            return [artifact], 0

        monkeypatch.setattr(
            "deep_research_client.providers.openscientist.httpx.AsyncClient",
            DummyAsyncClient,
        )
        monkeypatch.setattr(OpenScientistProvider, "_health_check", fake_health_check)
        monkeypatch.setattr(OpenScientistProvider, "_submit_job", fake_submit_job)
        monkeypatch.setattr(OpenScientistProvider, "_poll_until_done", fake_poll_until_done)
        monkeypatch.setattr(OpenScientistProvider, "_download_report", fake_download_report)
        monkeypatch.setattr(OpenScientistProvider, "_download_artifacts", fake_download_artifacts)

        result = await self.provider.research("What is autophagy?")

        assert len(result.artifacts) == 1
        assert result.artifacts[0].filename == "evidence_matrix.png"
        assert result.artifacts[0].source == "openscientist_artifacts"
        assert result.artifacts[0].is_image is True

    async def test_research_save_artifacts_false_skips_download(self, monkeypatch):
        """save_artifacts=False prevents artifact download entirely."""
        params = OpenScientistParams(save_artifacts=False)
        provider = OpenScientistProvider(self.config, params)
        download_called = {"called": False}

        async def fake_health_check(self, client) -> None:
            return None

        async def fake_submit_job(self, client, query, max_iterations) -> str:
            return "job-no-artifacts"

        async def fake_poll_until_done(self, client, job_id, timeout, poll_interval) -> str:
            return "completed"

        async def fake_download_report(self, client, job_id) -> str:
            return "# Report\n\nPMID: 12345678"

        async def fake_download_artifacts(self, client, job_id):
            download_called["called"] = True
            return [], 0

        monkeypatch.setattr(
            "deep_research_client.providers.openscientist.httpx.AsyncClient",
            DummyAsyncClient,
        )
        monkeypatch.setattr(OpenScientistProvider, "_health_check", fake_health_check)
        monkeypatch.setattr(OpenScientistProvider, "_submit_job", fake_submit_job)
        monkeypatch.setattr(OpenScientistProvider, "_poll_until_done", fake_poll_until_done)
        monkeypatch.setattr(OpenScientistProvider, "_download_report", fake_download_report)
        monkeypatch.setattr(OpenScientistProvider, "_download_artifacts", fake_download_artifacts)

        result = await provider.research("What is autophagy?")

        assert result.artifacts == []
        assert download_called["called"] is False

    async def test_research_records_skipped_artifact_count(self, monkeypatch):
        """Skipped artifacts are recorded in provider_config."""
        async def fake_health_check(self, client) -> None:
            return None

        async def fake_submit_job(self, client, query, max_iterations) -> str:
            return "job-skipped"

        async def fake_poll_until_done(self, client, job_id, timeout, poll_interval) -> str:
            return "completed"

        async def fake_download_report(self, client, job_id) -> str:
            return "# Report"

        async def fake_download_artifacts(self, client, job_id):
            return [], 7

        monkeypatch.setattr(
            "deep_research_client.providers.openscientist.httpx.AsyncClient",
            DummyAsyncClient,
        )
        monkeypatch.setattr(OpenScientistProvider, "_health_check", fake_health_check)
        monkeypatch.setattr(OpenScientistProvider, "_submit_job", fake_submit_job)
        monkeypatch.setattr(OpenScientistProvider, "_poll_until_done", fake_poll_until_done)
        monkeypatch.setattr(OpenScientistProvider, "_download_report", fake_download_report)
        monkeypatch.setattr(OpenScientistProvider, "_download_artifacts", fake_download_artifacts)

        result = await self.provider.research("What is autophagy?")

        assert result.provider_config is not None
        assert result.provider_config["skipped_artifact_count"] == 7

    async def test_research_no_provider_config_when_no_skipped(self, monkeypatch):
        """provider_config is None when no artifacts are skipped."""
        async def fake_health_check(self, client) -> None:
            return None

        async def fake_submit_job(self, client, query, max_iterations) -> str:
            return "job-clean"

        async def fake_poll_until_done(self, client, job_id, timeout, poll_interval) -> str:
            return "completed"

        async def fake_download_report(self, client, job_id) -> str:
            return "# Report"

        async def fake_download_artifacts(self, client, job_id):
            return [], 0

        monkeypatch.setattr(
            "deep_research_client.providers.openscientist.httpx.AsyncClient",
            DummyAsyncClient,
        )
        monkeypatch.setattr(OpenScientistProvider, "_health_check", fake_health_check)
        monkeypatch.setattr(OpenScientistProvider, "_submit_job", fake_submit_job)
        monkeypatch.setattr(OpenScientistProvider, "_poll_until_done", fake_poll_until_done)
        monkeypatch.setattr(OpenScientistProvider, "_download_report", fake_download_report)
        monkeypatch.setattr(OpenScientistProvider, "_download_artifacts", fake_download_artifacts)

        result = await self.provider.research("What is autophagy?")

        assert result.provider_config is None


@pytest.mark.integration
async def test_openscientist_health_check_integration():
    """OpenScientist health endpoint should be reachable with a live deployment."""
    provider = OpenScientistProvider(
        ProviderConfig(
            name="openscientist",
            api_key=_load_openscientist_api_key(),
            base_url=_load_openscientist_base_url(),
            enabled=True,
        )
    )

    timeout = httpx.Timeout(30, read=120)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        await provider._health_check(client)


@pytest.mark.integration
async def test_openscientist_research_integration():
    """OpenScientist should complete a live end-to-end research run."""
    query = "What is autophagy?"
    provider = OpenScientistProvider(
        ProviderConfig(
            name="openscientist",
            api_key=_load_openscientist_api_key(),
            base_url=_load_openscientist_base_url(),
            enabled=True,
            timeout=1800,
        ),
        OpenScientistParams(
            max_iterations=1,
            poll_interval=10,
            timeout=1800,
            use_hypotheses=False,
            investigation_mode="autonomous",
        ),
    )

    try:
        result = await provider.research(query)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in {401, 403}:
            pytest.skip(
                "OpenScientist job submission is not authorized for this account. "
                "The deployment is reachable, but the API key may still need account approval."
            )
        raise

    assert result.provider == "openscientist"
    assert result.query == query
    assert result.model == "openscientist-autonomous"
    assert result.markdown.strip()
    assert len(result.markdown) > 100
    assert len(result.citations) > 0
    assert all(citation.startswith("PMID:") for citation in result.citations)
