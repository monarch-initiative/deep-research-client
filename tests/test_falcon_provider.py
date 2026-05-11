"""Tests for Falcon provider response handling."""

import pytest
from datetime import datetime
from pathlib import Path
from uuid import UUID

# Skip all tests in this module if edison_client is not installed
pytest.importorskip("edison_client")

from deep_research_client.providers.falcon import FalconProvider
from deep_research_client.models import ProviderConfig


def create_mock_pqa_response(
    answer: str = "Test answer",
    formatted_answer: str = "Test formatted answer with references",
    has_successful_answer: bool = True
):
    """Create a mock PQATaskResponse for testing."""
    from edison_client.models.app import PQATaskResponse

    # Use model_construct to bypass validation for testing
    return PQATaskResponse.model_construct(
        status="completed",
        query="test query",
        user=None,
        created_at=datetime.now(),
        job_name="job-futurehouse-paperqa2-deep",
        public=False,
        shared_with=None,
        build_owner=None,
        environment_name=None,
        agent_name=None,
        task_id=None,
        answer=answer,
        formatted_answer=formatted_answer,
        answer_reasoning=None,
        has_successful_answer=has_successful_answer,
        total_cost=None,
        total_queries=None
    )


def create_verbose_response(environment_frame: dict):
    """Create a verbose Edison response for testing."""
    from edison_client.models.app import TaskResponseVerbose

    return TaskResponseVerbose.model_construct(
        status="completed",
        query="test query",
        user=None,
        created_at=datetime.now(),
        job_name="literature-20260216",
        share_status="private",
        permitted_accessors=None,
        build_owner=None,
        environment_name=None,
        agent_name=None,
        task_id=None,
        project_id=None,
        agent_state=None,
        environment_frame=environment_frame,
        metadata=None,
        deployment_config=None,
    )


def test_extract_text_from_pqa_response():
    """Test extracting text content from PQATaskResponse."""
    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)

    # Test with formatted_answer (preferred)
    response = [create_mock_pqa_response(
        answer="Plain answer",
        formatted_answer="Formatted answer with (smith2020study pages 1-5) citations"
    )]

    text = provider._extract_text_content(response)
    assert text == "Formatted answer with (smith2020study pages 1-5) citations"


def test_extract_text_from_verbose_response():
    """Verbose Edison responses should read formatted answers from the frame."""
    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)

    response = [
        create_verbose_response(
            {
                "state": {
                    "state": {
                        "response": {
                            "answer": {
                                "answer": "Plain answer",
                                "formatted_answer": "Formatted answer",
                            }
                        }
                    }
                }
            }
        )
    ]

    text = provider._extract_text_content(response)

    assert text == "Formatted answer"


def test_extract_text_fallback_to_answer():
    """Test fallback to answer field when formatted_answer is empty."""
    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)

    response = [create_mock_pqa_response(
        answer="Plain answer only",
        formatted_answer=""  # Empty formatted answer
    )]

    text = provider._extract_text_content(response)
    assert text == "Plain answer only"


def test_extract_text_raises_on_no_answer():
    """Test that extraction raises error when no answer is available."""
    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)

    response = [create_mock_pqa_response(
        answer="",
        formatted_answer="",
        has_successful_answer=False
    )]

    with pytest.raises(ValueError, match="PQATaskResponse has no answer"):
        provider._extract_text_content(response)


def test_extract_citations_from_paperqa_format():
    """Test extracting citations from PaperQA-style formatted answer."""
    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)

    response = [create_mock_pqa_response()]
    report_text = (
        "CRISPR is a gene editing tool (smith2020crispr pages 1-5). "
        "It was discovered in bacteria (jones2019discovery pages 10-15). "
        "Recent advances include (wang2021advances pages 3-7)."
    )

    citations = provider._extract_citations(response, report_text)

    assert len(citations) == 3
    assert "smith2020crispr pages 1-5" in citations
    assert "jones2019discovery pages 10-15" in citations
    assert "wang2021advances pages 3-7" in citations


def test_extract_citations_from_mixed_formats():
    """Test extracting citations from mixed citation formats."""
    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)

    response = [create_mock_pqa_response()]
    report_text = (
        "Gene editing (smith2020study pages 1-5) is important. "
        "See also [PMID:12345678] and [DOI:10.1234/test]. "
        "More info at https://example.com/paper."
    )

    citations = provider._extract_citations(response, report_text)

    # Should extract all citation types
    assert len(citations) >= 3
    assert any("smith2020study" in c for c in citations)
    assert any("PMID:12345678" in c for c in citations)
    assert any("https://example.com/paper" in c for c in citations)


def test_extract_citations_removes_duplicates():
    """Test that duplicate citations are removed."""
    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)

    response = [create_mock_pqa_response()]
    report_text = (
        "First mention (smith2020study pages 1-5). "
        "Second mention (smith2020study pages 1-5). "
        "Third mention (smith2020study pages 1-5)."
    )

    citations = provider._extract_citations(response, report_text)

    # Should only have one unique citation
    assert len(citations) == 1
    assert "smith2020study pages 1-5" in citations


def test_extract_no_citations():
    """Test handling text with no citations."""
    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)

    response = [create_mock_pqa_response()]
    report_text = "This is plain text with no citations at all."

    citations = provider._extract_citations(response, report_text)

    assert citations == []


def test_extract_output_data_from_environment_frame():
    """Output data entries should be found in the final Edison frame."""
    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)

    environment_frame = {
        "state": {
            "info": {
                "output_data": [
                    {"entry_id": "data_entry:11111111-1111-1111-1111-111111111111"}
                ]
            }
        }
    }

    output_data = provider._extract_output_data(environment_frame)

    assert output_data == [
        {"entry_id": "data_entry:11111111-1111-1111-1111-111111111111"}
    ]


def test_artifacts_from_raw_fetch_response():
    """Raw Edison storage content should become a durable research artifact."""
    from edison_client.models.data_storage_methods import RawFetchResponse

    config = ProviderConfig(name="falcon", api_key="test-key")
    provider = FalconProvider(config)
    storage_id = UUID("11111111-1111-1111-1111-111111111111")

    artifacts = provider._artifacts_from_fetch_response(
        RawFetchResponse(
            filename=Path("figure.svg"),
            content="<svg></svg>",
            entry_id=storage_id,
            entry_name="Figure 1",
        ),
        source_item={"name": "Figure 1"},
        storage_id=storage_id,
        used_filenames=set(),
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.filename == "figure.svg"
    assert artifact.media_type == "image/svg+xml"
    assert artifact.is_image is True
    assert artifact.description == "Figure 1"
    assert artifact.data_storage_id == str(storage_id)
