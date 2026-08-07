"""Tests for the OpenScientist provider."""

import asyncio
import base64
import io
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from deep_research_client.cli import _write_result_artifacts
from deep_research_client.models import ProviderConfig, ResearchArtifact, ResearchResult
from deep_research_client.processing.result_formatter import ResultFormatter
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


def create_zip_bytes(files: dict[str, str | bytes]) -> bytes:
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

    def test_extract_markdown_from_artifacts_prefers_final_report(self):
        """Runtime skill files mentioning reports should not shadow the real report."""
        zip_bytes = create_zip_bytes(
            {
                ".claude/skills/clinical-reports--clinical-reports.md": "wrong",
                "final_report.md": "# Final Report\n\nPMID: 12345678",
            }
        )

        report_name, markdown = self.provider._extract_markdown_from_artifact_zip(
            zip_bytes,
            "job-123",
        )

        assert report_name == "final_report.md"
        assert markdown == "# Final Report\n\nPMID: 12345678"

    def test_extract_artifacts_from_zip_filters_useful_files(self):
        """OpenScientist artifact ZIP extraction should keep useful review artifacts."""
        png_content = b"\x89PNG\r\n\x1a\nimage"
        zip_bytes = create_zip_bytes(
            {
                ".claude/skills/runtime.md": "runtime scaffolding",
                "agent-container.log": "log output",
                "final_report.md": "# Report",
                "final_report.pdf": b"%PDF-1.4 fake",
                "provenance/evidence_matrix.json": '{"rows":[]}',
                "provenance/evidence_matrix.png": png_content,
                "provenance/iter1_transcript.json": '{"messages":[]}',
                "raw/archive.zip": b"PK",
                "results/table.csv": "gene,score\nATP7B,0.9\n",
            }
        )

        artifacts = self.provider._extract_artifacts_from_artifact_zip(
            zip_bytes,
            report_names={"final_report.md"},
        )

        assert [artifact.filename for artifact in artifacts] == [
            "final_report.pdf",
            "provenance_evidence_matrix.json",
            "provenance_evidence_matrix.png",
            "results_table.csv",
        ]
        assert [artifact.source for artifact in artifacts] == [
            "openscientist_artifacts_zip",
            "openscientist_artifacts_zip",
            "openscientist_artifacts_zip",
            "openscientist_artifacts_zip",
        ]
        assert artifacts[2].media_type == "image/png"
        assert base64.b64decode(artifacts[2].content_base64) == png_content

    def test_extract_artifacts_from_zip_enforces_size_limit(self):
        """Artifacts larger than the configured limit should be skipped before readout."""
        provider = OpenScientistProvider(
            self.config,
            OpenScientistParams(artifact_max_bytes=3),
        )
        zip_bytes = create_zip_bytes(
            {
                "provenance/large.png": b"1234",
                "provenance/small.json": "{}",
            }
        )

        artifacts = provider._extract_artifacts_from_artifact_zip(zip_bytes)

        assert [artifact.filename for artifact in artifacts] == [
            "provenance_small.json"
        ]

    def test_openscientist_artifacts_write_and_render(self, tmp_path):
        """OpenScientist artifacts should become sidecars, frontmatter, and links."""
        zip_bytes = create_zip_bytes(
            {
                "provenance/evidence_matrix.json": '{"rows":[]}',
                "provenance/evidence_matrix.png": b"\x89PNG\r\n\x1a\nimage",
            }
        )
        artifacts = self.provider._extract_artifacts_from_artifact_zip(zip_bytes)
        result = ResearchResult(
            markdown="# Report",
            provider="openscientist",
            query="query",
            artifacts=artifacts,
        )

        _write_result_artifacts(result, tmp_path / "report.md")
        rendered = ResultFormatter().format_full_markdown(result)
        frontmatter = yaml.safe_load(rendered.split("---", 2)[1])

        assert (tmp_path / "report_artifacts" / "provenance_evidence_matrix.png").exists()
        assert frontmatter["artifact_count"] == 2
        assert frontmatter["artifact_sources"] == {"openscientist_artifacts_zip": 2}
        assert frontmatter["artifacts"][0]["path"].startswith("report_artifacts/")
        assert "- [OpenScientist evidence matrix](report_artifacts/provenance_evidence_matrix.json)" in rendered
        assert "![OpenScientist evidence matrix](report_artifacts/provenance_evidence_matrix.png)" in rendered

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

        async def fake_download_report_with_artifacts(self, client, job_id: str):
            assert job_id == "job-123"
            return SimpleNamespace(
                markdown="# Report\n\nPMID: 12345678\n\nPMID: 23456789",
                artifacts=[
                    ResearchArtifact(
                        filename="figure.png",
                        content_base64="aW1hZ2U=",
                        media_type="image/png",
                        source="openscientist_artifacts_zip",
                    )
                ],
            )

        monkeypatch.setattr(
            "deep_research_client.providers.openscientist.httpx.AsyncClient",
            DummyAsyncClient,
        )
        monkeypatch.setattr(OpenScientistProvider, "_health_check", fake_health_check)
        monkeypatch.setattr(OpenScientistProvider, "_submit_job", fake_submit_job)
        monkeypatch.setattr(OpenScientistProvider, "_poll_until_done", fake_poll_until_done)
        monkeypatch.setattr(
            OpenScientistProvider,
            "_download_report_with_artifacts",
            fake_download_report_with_artifacts,
        )

        result = await self.provider.research("What is autophagy?")

        assert result.provider == "openscientist"
        assert result.query == "What is autophagy?"
        assert result.model == "openscientist-autonomous"
        assert result.markdown == "# Report\n\nPMID: 12345678\n\nPMID: 23456789"
        assert result.citations == ["PMID:12345678", "PMID:23456789"]
        assert [artifact.filename for artifact in result.artifacts] == ["figure.png"]

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
