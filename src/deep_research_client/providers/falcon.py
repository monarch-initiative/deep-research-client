"""Edison Scientific provider (formerly FutureHouse Falcon)."""

import base64
import json
import logging
import mimetypes
import re
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence
from uuid import UUID

from edison_client import EdisonClient, JobNames
from edison_client.models.app import PQATaskResponse, TaskResponseVerbose
from edison_client.models.data_storage_methods import RawFetchResponse

from . import ResearchProvider
from ..models import (
    ProviderConfig,
    ResearchArtifact,
    ResearchResult,
    sanitize_artifact_filename,
)
from ..provider_params import FalconParams
from ..model_cards import ProviderModelCards, create_falcon_model_cards
from ..system_prompts import DEFAULT_RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


_DATA_URL_PREFIX = "data:"
type EdisonTaskResponse = PQATaskResponse | TaskResponseVerbose
type EdisonResponse = Sequence[EdisonTaskResponse]


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
            return self._result_from_response(client, response, query)
        except Exception as e:
            logger.error(f"Edison API request failed: {e}")
            logger.debug("Error details:", exc_info=True)
            raise

    def retrieve_trajectory(self, trajectory_id: str) -> ResearchResult:
        """Retrieve an existing Edison trajectory and preserve its artifacts."""
        if not self.is_available():
            raise ValueError(f"Edison provider not available (API key: {bool(self.config.api_key)})")

        client = EdisonClient(api_key=self.config.api_key)
        logger.info(f"Retrieving Edison trajectory {trajectory_id}")
        response = [client.get_task(task_id=trajectory_id, verbose=True)]
        query = response[0].query or f"Edison trajectory {trajectory_id}"
        result = self._result_from_response(client, response, query)
        result.provider_config = {
            **(result.provider_config or {}),
            "trajectory_id": trajectory_id,
            "retrieval_mode": "edison_trajectory",
        }
        return result

    def _result_from_response(
        self,
        client: EdisonClient,
        response: EdisonResponse,
        query: str,
    ) -> ResearchResult:
        """Build a research result from an Edison verbose response."""
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
            query=query,
        )

    def _extract_text_content(self, response: EdisonResponse) -> str:
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

    def _extract_citations(self, response: EdisonResponse, report_text: str) -> List[str]:
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

    def _extract_artifacts(self, client: Any, response: EdisonResponse) -> list[ResearchArtifact]:
        """Fetch artifacts listed in the final Edison environment frame."""
        if not response:
            return []

        task_response = response[0]
        if not isinstance(task_response, TaskResponseVerbose):
            return []

        output_data = self._extract_output_data(task_response.environment_frame or {})
        artifacts = self._extract_answer_artifacts(task_response)
        artifacts.extend(
            self._extract_message_image_artifacts(
                task_response,
                used_filenames={artifact.filename for artifact in artifacts},
            )
        )
        used_storage_ids: set[str] = set()
        used_filenames: set[str] = {artifact.filename for artifact in artifacts}

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

    def _extract_message_image_artifacts(
        self,
        response: TaskResponseVerbose,
        used_filenames: set[str],
    ) -> list[ResearchArtifact]:
        """Extract one representative image from each verbose Edison image message."""
        if self.params.max_embedded_images == 0:
            return []

        # Only the agent-state message history is needed for embedded images.
        payload = {"agent_state": response.agent_state}
        artifacts: list[ResearchArtifact] = []
        seen_urls: set[str] = set()

        for image_group in self._walk_image_message_groups(payload):
            image_url = image_group["url"]
            if image_url in seen_urls:
                continue
            seen_urls.add(image_url)

            artifact = self._artifact_from_data_url(
                image_url,
                used_filenames=used_filenames,
                index=len(artifacts) + 1,
                description=image_group["description"],
            )
            if artifact is not None:
                artifacts.append(artifact)
                if len(artifacts) >= self.params.max_embedded_images:
                    logger.info(
                        "Reached Falcon embedded image limit (%s); skipping remaining embedded images",
                        self.params.max_embedded_images,
                    )
                    break

        return artifacts

    def _walk_image_message_groups(self, value: Any) -> Iterable[dict[str, str | None]]:
        """Yield one representative image and description per nested message content block."""
        if isinstance(value, dict):
            content = value.get("content")
            if isinstance(content, list):
                image_urls = [
                    item.get("image_url", {}).get("url")
                    for item in content
                    if isinstance(item, dict)
                    and isinstance(item.get("image_url"), dict)
                    and isinstance(item["image_url"].get("url"), str)
                    and item["image_url"]["url"].startswith(f"{_DATA_URL_PREFIX}image/")
                ]
                if image_urls:
                    yield {
                        "url": image_urls[0],
                        "description": self._image_description_from_content(content),
                    }

            for nested_value in value.values():
                yield from self._walk_image_message_groups(nested_value)
            return

        if isinstance(value, list):
            for nested_value in value:
                yield from self._walk_image_message_groups(nested_value)

    def _image_description_from_content(self, content: list[Any]) -> str | None:
        """Build a concise artifact description from Edison message text content."""
        text_items = [
            item.get("text", "").strip()
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        for text in text_items:
            if not text or text.startswith("Retrieved "):
                continue
            condensed = " ".join(text.split())
            return condensed[:160]
        return None

    def _artifact_from_data_url(
        self,
        data_url: str,
        used_filenames: set[str],
        index: int,
        description: str | None = None,
    ) -> ResearchArtifact | None:
        """Convert a base64 data URL into a research artifact."""
        if not data_url.startswith(_DATA_URL_PREFIX) or ";base64," not in data_url:
            return None

        header, payload = data_url[len(_DATA_URL_PREFIX):].split(",", 1)
        if not header.endswith(";base64"):
            return None

        media_type = header.removesuffix(";base64") or None
        if not media_type or not media_type.startswith("image/"):
            return None

        filename = self._artifact_filename(
            f"image-{index}{self._artifact_extension_for_media_type(media_type)}",
            used_filenames,
        )
        return ResearchArtifact(
            filename=filename,
            content_base64="".join(payload.split()),
            media_type=media_type,
            source="edison_message_content",
            description=description or f"Edison image artifact {index}",
        )

    def _artifact_extension_for_media_type(self, media_type: str) -> str:
        """Return a filename extension suitable for a MIME type."""
        extension = mimetypes.guess_extension(media_type)
        if extension is not None:
            return extension
        if media_type == "image/svg+xml":
            return ".svg"
        return ".bin"

    def _extract_answer_artifacts(
        self,
        response: TaskResponseVerbose,
    ) -> list[ResearchArtifact]:
        """Extract artifacts embedded directly in a verbose Edison answer."""
        answer = self._get_verbose_answer(response)
        raw_artifacts = answer.get("artifacts")
        if not isinstance(raw_artifacts, dict):
            return []

        artifacts: list[ResearchArtifact] = []
        used_filenames: set[str] = set()
        for raw_name, raw_content in raw_artifacts.items():
            artifact = self._artifact_from_answer_value(
                raw_name=str(raw_name),
                raw_content=raw_content,
                used_filenames=used_filenames,
            )
            if artifact is not None:
                artifacts.append(artifact)

        return artifacts

    def _artifact_from_answer_value(
        self,
        raw_name: str,
        raw_content: Any,
        used_filenames: set[str],
    ) -> ResearchArtifact | None:
        """Convert an Edison answer artifact value into a research artifact."""
        if isinstance(raw_content, dict) and isinstance(raw_content.get("content_base64"), str):
            filename = self._artifact_filename(
                raw_content.get("filename") or raw_name,
                used_filenames,
            )
            return ResearchArtifact(
                filename=filename,
                content_base64=raw_content["content_base64"],
                media_type=raw_content.get("media_type") or mimetypes.guess_type(filename)[0],
                source="edison_answer_artifacts",
                description=raw_content.get("description") or f"Edison artifact {raw_name}",
            )

        if isinstance(raw_content, str):
            filename = self._filename_with_default_suffix(raw_name, ".md")
            filename = self._artifact_filename(filename, used_filenames)
            media_type = "text/markdown"
            content = raw_content.encode("utf-8")
        elif raw_content is None:
            return None
        else:
            filename = self._filename_with_default_suffix(raw_name, ".json")
            filename = self._artifact_filename(filename, used_filenames)
            media_type = "application/json"
            content = json.dumps(raw_content, indent=2, sort_keys=True).encode("utf-8")

        return ResearchArtifact(
            filename=filename,
            content_base64=base64.b64encode(content).decode("ascii"),
            media_type=media_type,
            source="edison_answer_artifacts",
            description=f"Edison artifact {raw_name}",
        )

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
        filename = sanitize_artifact_filename(str(raw_name or "artifact"))
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

    def _filename_with_default_suffix(self, raw_name: str, suffix: str) -> str:
        """Add a suffix to an artifact name if Edison did not provide one."""
        path = Path(raw_name)
        if path.suffix:
            return raw_name
        return f"{raw_name}{suffix}"

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
