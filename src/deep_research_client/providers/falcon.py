"""Edison Scientific provider (formerly FutureHouse Falcon)."""

import base64
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional
from uuid import UUID

from edison_client import EdisonClient, JobNames
from edison_client.models.app import PQATaskResponse, TaskResponseVerbose
from edison_client.models.data_storage_methods import RawFetchResponse

from . import ResearchProvider
from ..models import ResearchArtifact, ResearchResult, ProviderConfig
from ..provider_params import FalconParams
from ..model_cards import ProviderModelCards, create_falcon_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class FalconProvider(ResearchProvider):
    """Provider for Edison Scientific API (formerly FutureHouse Falcon)."""

    def __init__(self, config: ProviderConfig, params: Optional[FalconParams] = None):
        """Initialize Falcon provider."""
        self.params = params or FalconParams()
        super().__init__(config, self.params.model)

        logger.debug(f"Initializing Falcon provider with model: {self.model}")
        if config.api_key:
            key_preview = config.api_key[:8] + "..." if len(config.api_key) > 8 else "***"
            logger.debug(f"API key configured (starts with: {key_preview})")

    def get_default_model(self) -> str:
        """Get default model."""
        return "Edison Scientific Literature"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for Falcon provider."""
        return create_falcon_model_cards()

    async def research(self, query: str) -> ResearchResult:
        """Perform research using Edison Scientific API."""
        logger.info(f"Starting Edison research query (model: {self.model})")
        logger.debug(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")

        if not self.is_available():
            raise ValueError(f"Edison provider not available (API key: {bool(self.config.api_key)})")

        client = EdisonClient(api_key=self.config.api_key)

        # Use custom system prompt or default
        system_prompt = self.params.system_prompt or DEFAULT_RESEARCH_SYSTEM_PROMPT

        # Edison combines system prompt and user query
        full_query = f"{system_prompt}\n\n{query}"

        # Use Edison LITERATURE API for deep research
        task_data = {
            "name": JobNames.LITERATURE,
            "query": full_query
        }

        try:
            logger.debug("Making API request to Edison")
            response = client.run_tasks_until_done(task_data, verbose=True)
            logger.info("Edison API request completed successfully")

            # Extract the report text and citations
            markdown_content = self._extract_text_content(response)
            logger.debug(f"Extracted markdown content: {len(markdown_content)} characters")

            citations = self._extract_citations(response, markdown_content)
            logger.info(f"Extracted {len(citations)} citations from response")

            artifacts = self._extract_artifacts(client, response)
            logger.info(f"Extracted {len(artifacts)} artifacts from response")

            return ResearchResult(
                markdown=markdown_content,
                citations=citations,
                artifacts=artifacts,
                provider=self.name,
                query=query
            )
        except Exception as e:
            logger.error(f"Edison API request failed: {e}")
            logger.debug("Error details:", exc_info=True)
            raise

    def _extract_text_content(self, response) -> str:
        """Extract text content from Edison response.

        Edison returns PQATaskResponse objects for standard calls. With
        verbose=True, it returns TaskResponseVerbose objects and the answer must
        be read from the final environment frame.
        """
        if not isinstance(response, list) or len(response) == 0:
            raise ValueError(f"Unexpected Edison response structure: {type(response)}")

        task_response = response[0]

        if isinstance(task_response, TaskResponseVerbose):
            answer = self._get_verbose_answer(task_response)
            if formatted_answer := answer.get("formatted_answer"):
                return str(formatted_answer)
            if plain_answer := answer.get("answer"):
                return str(plain_answer)
            raise ValueError(
                f"Verbose Edison response has no answer. Status: {task_response.status}"
            )

        if not isinstance(task_response, PQATaskResponse):
            raise ValueError(
                f"Expected PQATaskResponse or TaskResponseVerbose, got {type(task_response)}. "
                f"This indicates an API change in edison-client."
            )

        # Prefer formatted_answer as it includes references
        if task_response.formatted_answer:
            return task_response.formatted_answer
        elif task_response.answer:
            return task_response.answer
        else:
            raise ValueError(
                f"PQATaskResponse has no answer. Status: {task_response.status}, "
                f"has_successful_answer: {task_response.has_successful_answer}"
            )

    def _get_verbose_answer(self, response: TaskResponseVerbose) -> dict[str, Any]:
        """Extract the answer object from a verbose Edison response."""
        environment_frame = response.environment_frame or {}
        state = environment_frame.get("state", {})
        nested_state = state.get("state", {})

        response_answer = (
            nested_state
            .get("response", {})
            .get("answer", {})
        )
        if isinstance(response_answer, dict):
            return response_answer

        direct_answer = nested_state.get("answer")
        if isinstance(direct_answer, str):
            return {"answer": direct_answer}

        return {}

    def _extract_citations(self, response, report_text: str) -> List[str]:
        """Extract citations from Edison response.

        Citations are embedded in the formatted_answer text using various patterns.
        """
        # Extract inline citations from the formatted answer text
        # Look for PaperQA-style citations like (Author2020Title pages 6-8)
        paperqa_citations = re.findall(r'\(([a-z]+\d{4}[a-z\s]+pages?\s+[\d\-]+)\)', report_text, re.IGNORECASE)

        # Look for standard reference patterns like [PMID:12345678], [DOI:10.xxx], [1]
        standard_refs = re.findall(r'\[([^\]]+)\]', report_text)

        # Look for URL citations
        url_citations = re.findall(r'https?://[^\s\)]+', report_text)

        # Combine all citation sources
        all_citations = paperqa_citations + standard_refs + url_citations

        # Remove duplicates while preserving order
        if all_citations:
            seen = set()
            unique_citations = []
            for citation in all_citations:
                citation_str = str(citation).strip()
                if citation_str and citation_str not in seen:
                    seen.add(citation_str)
                    unique_citations.append(citation_str)
            return unique_citations

        return []

    def _extract_artifacts(self, client: EdisonClient, response: list[Any]) -> list[ResearchArtifact]:
        """Fetch artifacts listed in the final Edison environment frame."""
        if not response:
            return []

        task_response = response[0]
        if not isinstance(task_response, TaskResponseVerbose):
            return []

        output_data = self._extract_output_data(task_response.environment_frame or {})
        artifacts: list[ResearchArtifact] = []
        used_storage_ids: set[str] = set()
        used_filenames: set[str] = set()

        for item in output_data:
            storage_id = self._get_data_storage_id(item)
            if storage_id is None:
                logger.debug(f"Skipping Edison artifact without entry_id: {item}")
                continue
            if str(storage_id) in used_storage_ids:
                continue
            used_storage_ids.add(str(storage_id))

            fetched = client.fetch_data_from_storage(data_storage_id=storage_id)
            for artifact in self._artifacts_from_fetch_response(
                fetched,
                source_item=item,
                storage_id=storage_id,
                used_filenames=used_filenames,
            ):
                artifacts.append(artifact)

        return artifacts

    def _extract_output_data(self, environment_frame: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract Edison output_data entries from an environment frame."""
        direct_output_data = (
            environment_frame
            .get("state", {})
            .get("info", {})
            .get("output_data")
        )
        if isinstance(direct_output_data, list):
            return [item for item in direct_output_data if isinstance(item, dict)]

        output_data: list[dict[str, Any]] = []
        for value in self._walk_values(environment_frame):
            if isinstance(value, dict) and isinstance(value.get("output_data"), list):
                output_data.extend(
                    item for item in value["output_data"] if isinstance(item, dict)
                )

        return output_data

    def _walk_values(self, value: Any) -> Iterable[Any]:
        """Yield nested dict and list values."""
        yield value
        if isinstance(value, dict):
            for nested_value in value.values():
                yield from self._walk_values(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                yield from self._walk_values(nested_value)

    def _get_data_storage_id(self, item: dict[str, Any]) -> UUID | None:
        """Extract a data-storage UUID from an Edison output_data item."""
        raw_id = item.get("entry_id") or item.get("data_storage_id") or item.get("id")
        if raw_id is None:
            return None

        id_text = str(raw_id)
        if id_text.startswith("data_entry:"):
            id_text = id_text.removeprefix("data_entry:")

        return UUID(id_text)

    def _artifacts_from_fetch_response(
        self,
        fetched: RawFetchResponse | Path | list[Path] | None,
        source_item: dict[str, Any],
        storage_id: UUID,
        used_filenames: set[str],
    ) -> list[ResearchArtifact]:
        """Convert an Edison data-storage fetch result into research artifacts."""
        if fetched is None:
            return []

        if isinstance(fetched, RawFetchResponse):
            filename = self._artifact_filename(
                fetched.filename or source_item.get("file_path") or fetched.entry_name,
                used_filenames,
            )
            content = fetched.content.encode("utf-8")
            return [
                self._make_artifact(
                    filename=filename,
                    content=content,
                    source_item=source_item,
                    storage_id=storage_id,
                )
            ]

        if isinstance(fetched, list):
            artifacts: list[ResearchArtifact] = []
            for path in fetched:
                artifacts.extend(
                    self._artifacts_from_path(path, source_item, storage_id, used_filenames)
                )
            return artifacts

        return self._artifacts_from_path(fetched, source_item, storage_id, used_filenames)

    def _artifacts_from_path(
        self,
        path: Path,
        source_item: dict[str, Any],
        storage_id: UUID,
        used_filenames: set[str],
    ) -> list[ResearchArtifact]:
        """Convert a downloaded file or directory into research artifacts."""
        if path.is_dir():
            artifacts: list[ResearchArtifact] = []
            for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
                relative_name = file_path.relative_to(path).as_posix()
                filename = self._artifact_filename(relative_name, used_filenames)
                artifacts.append(
                    self._make_artifact(
                        filename=filename,
                        content=file_path.read_bytes(),
                        source_item=source_item,
                        storage_id=storage_id,
                    )
                )
            return artifacts

        filename = self._artifact_filename(path.name, used_filenames)
        return [
            self._make_artifact(
                filename=filename,
                content=path.read_bytes(),
                source_item=source_item,
                storage_id=storage_id,
            )
        ]

    def _artifact_filename(self, raw_name: Any, used_filenames: set[str]) -> str:
        """Return a stable, unique filename for a fetched artifact."""
        filename = Path(str(raw_name or "artifact")).name
        if not filename or filename in {".", ".."}:
            filename = "artifact"

        candidate = filename
        suffix = Path(filename).suffix
        stem = Path(filename).stem or "artifact"
        counter = 2
        while candidate in used_filenames:
            candidate = f"{stem}-{counter}{suffix}"
            counter += 1

        used_filenames.add(candidate)
        return candidate

    def _make_artifact(
        self,
        filename: str,
        content: bytes,
        source_item: dict[str, Any],
        storage_id: UUID,
    ) -> ResearchArtifact:
        """Build a ResearchArtifact from fetched bytes."""
        media_type = mimetypes.guess_type(filename)[0]
        return ResearchArtifact(
            filename=filename,
            content_base64=base64.b64encode(content).decode("ascii"),
            media_type=media_type,
            source="edison_output_data",
            data_storage_id=str(storage_id),
            description=source_item.get("name") or source_item.get("description"),
        )
