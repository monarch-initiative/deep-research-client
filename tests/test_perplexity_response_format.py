"""Tests for Perplexity response_format and sonar-reasoning-pro model support."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from deep_research_client.providers.perplexity import PerplexityProvider
from deep_research_client.provider_params import PerplexityParams
from deep_research_client.models import ProviderConfig


class TestPerplexityResponseFormat:
    """Test suite for Perplexity response_format support."""

    def test_sonar_reasoning_pro_model_in_model_cards(self):
        """Test that sonar-reasoning-pro model is available in model cards."""
        provider_cards = PerplexityProvider.model_cards()

        assert "sonar-reasoning-pro" in provider_cards.models

        # Test model details
        reasoning_pro_card = provider_cards.models["sonar-reasoning-pro"]
        assert reasoning_pro_card.name == "sonar-reasoning-pro"
        assert "structured output" in reasoning_pro_card.description.lower()
        assert "reasoning" in reasoning_pro_card.aliases

        # Test capabilities
        from deep_research_client.model_cards import ModelCapability
        assert ModelCapability.STRUCTURED_OUTPUT in reasoning_pro_card.capabilities
        assert ModelCapability.WEB_SEARCH in reasoning_pro_card.capabilities

    def test_response_format_parameter_validation(self):
        """Test response_format parameter validation."""
        # Valid JSON object format
        params = PerplexityParams(
            model="sonar-reasoning-pro",
            reasoning_effort="high",
            response_format={"type": "json_object"}
        )
        assert params.response_format == {"type": "json_object"}

        # Valid JSON schema format
        schema_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "gene_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "gene_name": {"type": "string"},
                        "function": {"type": "string"}
                    }
                }
            }
        }
        params = PerplexityParams(
            model="sonar-reasoning-pro",
            reasoning_effort="high",
            response_format=schema_format
        )
        assert params.response_format == schema_format

    def test_response_format_requires_high_reasoning_effort(self):
        """Test that response_format requires reasoning_effort='high'."""
        with pytest.raises(ValueError, match="response_format requires reasoning_effort='high'"):
            PerplexityParams(
                response_format={"type": "json_object"},
                reasoning_effort="medium"  # Should fail
            )

    def test_invalid_response_format_types(self):
        """Test validation of invalid response_format values."""
        # Missing type field
        with pytest.raises(ValueError, match="must include a 'type' field"):
            PerplexityParams(
                reasoning_effort="high",
                response_format={"format": "json"}  # Wrong field name
            )

        # Invalid type value
        with pytest.raises(ValueError, match="must be 'json_object' or 'json_schema'"):
            PerplexityParams(
                reasoning_effort="high",
                response_format={"type": "xml"}  # Invalid type
            )

        # JSON schema without schema field
        with pytest.raises(ValueError, match="must include 'json_schema' field"):
            PerplexityParams(
                reasoning_effort="high",
                response_format={"type": "json_schema"}  # Missing json_schema field
            )

    def test_auto_reasoning_effort_for_reasoning_model(self):
        """Test automatic reasoning_effort='high' for sonar-reasoning-pro."""
        config = ProviderConfig(name="perplexity", api_key="test-key", enabled=True)

        # Test auto-enabling reasoning effort
        params = PerplexityParams(model="sonar-reasoning-pro", reasoning_effort="medium")
        provider = PerplexityProvider(config, params)

        # Should be auto-upgraded to high
        assert provider.params.reasoning_effort == "high"

    def test_response_format_only_with_reasoning_model(self):
        """Test that response_format is only allowed with sonar-reasoning-pro."""
        config = ProviderConfig(name="perplexity", api_key="test-key", enabled=True)

        # Should fail with non-reasoning model
        params = PerplexityParams(
            model="sonar-pro",  # Not reasoning model
            reasoning_effort="high",
            response_format={"type": "json_object"}
        )

        with pytest.raises(ValueError, match="response_format is only supported with sonar-reasoning-pro"):
            PerplexityProvider(config, params)

    @patch('deep_research_client.providers.perplexity.httpx.AsyncClient')
    async def test_response_format_sent_to_api(self, mock_client_class):
        """Test that response_format is included in API payload."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": '{"gene_name": "TP53", "function": "tumor suppressor"}'}
            }]
        }
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Create provider with response_format
        config = ProviderConfig(name="perplexity", api_key="test-key", enabled=True)
        params = PerplexityParams(
            model="sonar-reasoning-pro",
            reasoning_effort="high",
            response_format={"type": "json_object"}
        )
        provider = PerplexityProvider(config, params)

        # Execute research
        await provider.research("Analyze gene TP53")

        # Verify API call
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        payload = call_args[1]['json']

        # Check that response_format was included
        assert "response_format" in payload
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["reasoning_effort"] == "high"
        assert payload["model"] == "sonar-reasoning-pro"

    @patch('deep_research_client.providers.perplexity.httpx.AsyncClient')
    async def test_json_schema_response_format(self, mock_client_class):
        """Test JSON schema response format functionality."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": '{"gene_name": "TP53", "primary_function": "tumor suppressor"}'}
            }]
        }
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__aenter__.return_value = mock_client

        # Create provider with JSON schema
        config = ProviderConfig(name="perplexity", api_key="test-key", enabled=True)
        schema_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "gene_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "gene_name": {"type": "string"},
                        "primary_function": {"type": "string"}
                    },
                    "required": ["gene_name", "primary_function"]
                }
            }
        }
        params = PerplexityParams(
            model="sonar-reasoning-pro",
            reasoning_effort="high",
            response_format=schema_format
        )
        provider = PerplexityProvider(config, params)

        # Execute research
        result = await provider.research("Analyze gene TP53 structure")

        # Verify API call and response
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        payload = call_args[1]['json']

        assert payload["response_format"] == schema_format
        assert result.markdown == '{"gene_name": "TP53", "primary_function": "tumor suppressor"}'

    def test_model_alias_resolution(self):
        """Test that model aliases work for sonar-reasoning-pro."""
        provider_cards = PerplexityProvider.model_cards()

        # Test that aliases resolve correctly
        assert provider_cards.resolve_model_name("reasoning-pro") == "sonar-reasoning-pro"
        assert provider_cards.resolve_model_name("srp") == "sonar-reasoning-pro"
        assert provider_cards.resolve_model_name("reasoning") == "sonar-reasoning-pro"

    def test_backward_compatibility(self):
        """Test that existing functionality still works without response_format."""
        config = ProviderConfig(name="perplexity", api_key="test-key", enabled=True)

        # Regular parameters without response_format should work
        params = PerplexityParams(model="sonar-pro", reasoning_effort="medium")
        provider = PerplexityProvider(config, params)

        assert provider.params.response_format is None
        assert provider.params.reasoning_effort == "medium"
        assert provider.model == "sonar-pro"

    def test_comprehensive_example(self):
        """Test a comprehensive example matching the reported issue scenario."""
        # This tests the exact use case from the issue description
        config = ProviderConfig(name="perplexity", api_key="test-key", enabled=True)

        # Create params for JSON-only gene analysis
        params = PerplexityParams(
            model="sonar-reasoning-pro",
            reasoning_effort="high",  # Required for response_format
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "gene_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "gene_name": {"type": "string"},
                            "primary_function": {"type": "string"},
                            "disease_associations": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "expression_tissues": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["gene_name", "primary_function", "disease_associations", "expression_tissues"]
                    }
                }
            }
        )

        provider = PerplexityProvider(config, params)

        # Verify configuration
        assert provider.model == "sonar-reasoning-pro"
        assert provider.params.reasoning_effort == "high"
        assert provider.params.response_format is not None
        assert provider.params.response_format["type"] == "json_schema"

    def test_model_capabilities_query(self):
        """Test querying models by STRUCTURED_OUTPUT capability."""
        from deep_research_client.model_cards import ModelCapability

        provider_cards = PerplexityProvider.model_cards()
        structured_models = provider_cards.get_models_with_capability(ModelCapability.STRUCTURED_OUTPUT)

        assert len(structured_models) == 1
        assert structured_models[0].name == "sonar-reasoning-pro"