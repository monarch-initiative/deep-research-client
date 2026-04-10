"""Tests for the OpenScientist provider."""

import asyncio
import io
import zipfile

import httpx
import pytest

from deep_research_client.models import ProviderConfig
from deep_research_client.provider_params import OpenScientistParams
from deep_research_client.providers.openscientist import OpenScientistProvider


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

        monkeypatch.setattr(
            "deep_research_client.providers.openscientist.httpx.AsyncClient",
            DummyAsyncClient,
        )
        monkeypatch.setattr(OpenScientistProvider, "_health_check", fake_health_check)
        monkeypatch.setattr(OpenScientistProvider, "_submit_job", fake_submit_job)
        monkeypatch.setattr(OpenScientistProvider, "_poll_until_done", fake_poll_until_done)
        monkeypatch.setattr(OpenScientistProvider, "_download_report", fake_download_report)

        result = await self.provider.research("What is autophagy?")

        assert result.provider == "openscientist"
        assert result.query == "What is autophagy?"
        assert result.model == "openscientist-autonomous"
        assert result.markdown == "# Report\n\nPMID: 12345678\n\nPMID: 23456789"
        assert result.citations == ["PMID:12345678", "PMID:23456789"]

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
