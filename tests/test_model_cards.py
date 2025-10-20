"""Test model cards functionality."""


from deep_research_client.model_cards import (
    CostLevel,
    TimeEstimate,
    ModelCapability,
    ModelCard,
    get_provider_model_cards,
    list_all_models,
    find_models_by_cost,
    find_models_by_capability,
    resolve_model_alias,
    list_all_aliases
)
from deep_research_client.providers.openai import OpenAIProvider
from deep_research_client.providers.perplexity import PerplexityProvider
from deep_research_client.providers.falcon import FalconProvider
from deep_research_client.providers.consensus import ConsensusProvider


class TestModelCards:
    """Test model cards functionality."""

    def test_model_card_creation(self):
        """Test creating a basic model card."""
        card = ModelCard(
            name="test-model",
            display_name="Test Model",
            description="A test model for verification",
            cost_level=CostLevel.MEDIUM,
            time_estimate=TimeEstimate.FAST,
            capabilities=[ModelCapability.WEB_SEARCH]
        )

        assert card.name == "test-model"
        assert card.display_name == "Test Model"
        assert card.cost_level == CostLevel.MEDIUM
        assert card.time_estimate == TimeEstimate.FAST
        assert ModelCapability.WEB_SEARCH in card.capabilities

    def test_openai_model_cards(self):
        """Test OpenAI provider model cards."""
        cards = OpenAIProvider.model_cards()

        assert cards.provider_name == "openai"
        assert cards.default_model == "o3-deep-research-2025-06-26"
        assert "o3-deep-research-2025-06-26" in cards.models
        assert "o4-mini-deep-research-2025-06-26" in cards.models

        # Test o3 model card
        o3_card = cards.get_model_card("o3-deep-research-2025-06-26")
        assert o3_card is not None
        assert o3_card.cost_level == CostLevel.VERY_HIGH
        assert o3_card.time_estimate == TimeEstimate.VERY_SLOW
        assert ModelCapability.WEB_SEARCH in o3_card.capabilities
        assert ModelCapability.CODE_INTERPRETATION in o3_card.capabilities

        # Test o4-mini model card
        o4_card = cards.get_model_card("o4-mini-deep-research-2025-06-26")
        assert o4_card is not None
        assert o4_card.cost_level == CostLevel.MEDIUM
        assert o4_card.time_estimate == TimeEstimate.MEDIUM
        assert "$2/$8 per million" in o4_card.pricing_notes

    def test_perplexity_model_cards(self):
        """Test Perplexity provider model cards."""
        cards = PerplexityProvider.model_cards()

        assert cards.provider_name == "perplexity"
        assert cards.default_model == "sonar-deep-research"
        assert "sonar-deep-research" in cards.models
        assert "sonar-pro" in cards.models
        assert "sonar" in cards.models

        # Test cost progression
        sonar = cards.get_model_card("sonar")
        sonar_pro = cards.get_model_card("sonar-pro")
        sonar_deep = cards.get_model_card("sonar-deep-research")

        assert sonar.cost_level == CostLevel.LOW
        assert sonar_pro.cost_level == CostLevel.MEDIUM
        assert sonar_deep.cost_level == CostLevel.HIGH

    def test_falcon_model_cards(self):
        """Test Falcon provider model cards."""
        cards = FalconProvider.model_cards()

        assert cards.provider_name == "falcon"
        assert cards.default_model == "FutureHouse Falcon API"

        falcon_card = cards.get_model_card("FutureHouse Falcon API")
        assert falcon_card is not None
        assert ModelCapability.SCIENTIFIC_LITERATURE in falcon_card.capabilities
        assert ModelCapability.ACADEMIC_SEARCH in falcon_card.capabilities
        assert "Scientific literature reviews" in falcon_card.use_cases

    def test_consensus_model_cards(self):
        """Test Consensus provider model cards."""
        cards = ConsensusProvider.model_cards()

        assert cards.provider_name == "consensus"
        assert cards.default_model == "Consensus Academic Search"

        consensus_card = cards.get_model_card("Consensus Academic Search")
        assert consensus_card is not None
        assert consensus_card.cost_level == CostLevel.LOW
        assert consensus_card.time_estimate == TimeEstimate.FAST
        assert ModelCapability.ACADEMIC_SEARCH in consensus_card.capabilities
        assert "$6.99/month" in consensus_card.pricing_notes

    def test_provider_model_cards_filtering(self):
        """Test filtering capabilities of ProviderModelCards."""
        cards = PerplexityProvider.model_cards()

        # Test filtering by cost
        low_cost_models = cards.get_models_by_cost(CostLevel.LOW)
        assert len(low_cost_models) == 1
        assert low_cost_models[0].name == "sonar"

        medium_cost_models = cards.get_models_by_cost(CostLevel.MEDIUM)
        assert len(medium_cost_models) == 1
        assert medium_cost_models[0].name == "sonar-pro"

        # Test filtering by time
        fast_models = cards.get_models_by_time(TimeEstimate.FAST)
        assert len(fast_models) == 1
        assert fast_models[0].name == "sonar"

        # Test filtering by capability
        web_search_models = cards.get_models_with_capability(ModelCapability.WEB_SEARCH)
        assert len(web_search_models) == 3  # All Perplexity models have web search

    def test_global_model_registry(self):
        """Test global model registry functions."""
        # Test getting provider model cards
        openai_cards = get_provider_model_cards("openai")
        assert openai_cards is not None
        assert openai_cards.provider_name == "openai"

        perplexity_cards = get_provider_model_cards("perplexity")
        assert perplexity_cards is not None
        assert perplexity_cards.provider_name == "perplexity"

        # Test non-existent provider
        nonexistent_cards = get_provider_model_cards("nonexistent")
        assert nonexistent_cards is None

    def test_list_all_models(self):
        """Test listing all models across providers."""
        all_models = list_all_models()

        assert "openai" in all_models
        assert "perplexity" in all_models
        assert "falcon" in all_models
        assert "consensus" in all_models

        # Check specific models
        assert "o3-deep-research-2025-06-26" in all_models["openai"]
        assert "o4-mini-deep-research-2025-06-26" in all_models["openai"]
        assert "sonar-deep-research" in all_models["perplexity"]
        assert "sonar-pro" in all_models["perplexity"]
        assert "sonar" in all_models["perplexity"]

    def test_find_models_by_cost(self):
        """Test finding models across providers by cost level."""
        # Test low cost models
        low_cost = find_models_by_cost(CostLevel.LOW)
        assert "perplexity" in low_cost  # sonar
        assert "consensus" in low_cost   # consensus search

        # Test very high cost models
        very_high_cost = find_models_by_cost(CostLevel.VERY_HIGH)
        assert "openai" in very_high_cost  # o3-deep-research

        # Check specific models
        perplexity_low = low_cost["perplexity"]
        assert len(perplexity_low) == 1
        assert perplexity_low[0].name == "sonar"

    def test_find_models_by_capability(self):
        """Test finding models across providers by capability."""
        # Test web search capability
        web_search_models = find_models_by_capability(ModelCapability.WEB_SEARCH)
        assert "openai" in web_search_models
        assert "perplexity" in web_search_models
        # falcon and consensus don't have general web search

        # Test academic search capability
        academic_search_models = find_models_by_capability(ModelCapability.ACADEMIC_SEARCH)
        assert "falcon" in academic_search_models
        assert "consensus" in academic_search_models
        # perplexity doesn't have dedicated academic search

        # Test scientific literature capability
        scientific_models = find_models_by_capability(ModelCapability.SCIENTIFIC_LITERATURE)
        assert "falcon" in scientific_models

    def test_model_card_use_cases_and_limitations(self):
        """Test that model cards include use cases and limitations."""
        openai_cards = OpenAIProvider.model_cards()
        o3_card = openai_cards.get_model_card("o3-deep-research-2025-06-26")

        # Check use cases
        assert len(o3_card.use_cases) > 0
        assert "Comprehensive research reports" in o3_card.use_cases

        # Check limitations
        assert len(o3_card.limitations) > 0
        assert "Very high cost per query" in o3_card.limitations

    def test_model_card_pricing_information(self):
        """Test that model cards include pricing information."""
        openai_cards = OpenAIProvider.model_cards()

        o3_card = openai_cards.get_model_card("o3-deep-research-2025-06-26")
        assert o3_card.pricing_notes is not None
        assert "$10/$40 per million" in o3_card.pricing_notes

        o4_card = openai_cards.get_model_card("o4-mini-deep-research-2025-06-26")
        assert o4_card.pricing_notes is not None
        assert "$2/$8 per million" in o4_card.pricing_notes

    def test_context_window_information(self):
        """Test that model cards include context window information where available."""
        openai_cards = OpenAIProvider.model_cards()
        o3_card = openai_cards.get_model_card("o3-deep-research-2025-06-26")

        assert o3_card.context_window == 128000

        perplexity_cards = PerplexityProvider.model_cards()
        sonar_deep_card = perplexity_cards.get_model_card("sonar-deep-research")

        assert sonar_deep_card.context_window == 200000

    def test_model_aliases(self):
        """Test that model cards include aliases."""
        openai_cards = OpenAIProvider.model_cards()

        # Test o3 aliases
        o3_card = openai_cards.get_model_card("o3-deep-research-2025-06-26")
        assert "o3" in o3_card.aliases
        assert "o3-deep" in o3_card.aliases
        assert "o3dr" in o3_card.aliases

        # Test o4-mini aliases
        o4_card = openai_cards.get_model_card("o4-mini-deep-research-2025-06-26")
        assert "o4m" in o4_card.aliases
        assert "mini" in o4_card.aliases

        # Test Perplexity aliases
        perplexity_cards = PerplexityProvider.model_cards()
        sonar_card = perplexity_cards.get_model_card("sonar")
        assert "fast" in sonar_card.aliases
        assert "basic" in sonar_card.aliases

    def test_alias_resolution(self):
        """Test alias resolution functionality."""
        openai_cards = OpenAIProvider.model_cards()

        # Test direct model name resolution
        assert openai_cards.resolve_model_name("o3-deep-research-2025-06-26") == "o3-deep-research-2025-06-26"

        # Test alias resolution
        assert openai_cards.resolve_model_name("o3") == "o3-deep-research-2025-06-26"
        assert openai_cards.resolve_model_name("o4m") == "o4-mini-deep-research-2025-06-26"
        assert openai_cards.resolve_model_name("mini") == "o4-mini-deep-research-2025-06-26"

        # Test non-existent alias
        assert openai_cards.resolve_model_name("nonexistent") is None

        # Test Perplexity aliases
        perplexity_cards = PerplexityProvider.model_cards()
        assert perplexity_cards.resolve_model_name("fast") == "sonar"
        assert perplexity_cards.resolve_model_name("pro") == "sonar-pro"
        assert perplexity_cards.resolve_model_name("deep") == "sonar-deep-research"

    def test_get_model_card_by_alias(self):
        """Test getting model cards by alias."""
        openai_cards = OpenAIProvider.model_cards()

        # Get by alias
        card_by_alias = openai_cards.get_model_card_by_alias("o4m")
        card_by_name = openai_cards.get_model_card("o4-mini-deep-research-2025-06-26")

        assert card_by_alias is not None
        assert card_by_alias == card_by_name
        assert card_by_alias.name == "o4-mini-deep-research-2025-06-26"

        # Non-existent alias
        assert openai_cards.get_model_card_by_alias("nonexistent") is None

    def test_list_aliases(self):
        """Test listing all aliases for a provider."""
        openai_cards = OpenAIProvider.model_cards()
        aliases = openai_cards.list_aliases()

        # Check some expected aliases
        assert "o3" in aliases
        assert aliases["o3"] == "o3-deep-research-2025-06-26"
        assert "o4m" in aliases
        assert aliases["o4m"] == "o4-mini-deep-research-2025-06-26"

        # Test Perplexity aliases
        perplexity_cards = PerplexityProvider.model_cards()
        perp_aliases = perplexity_cards.list_aliases()

        assert "fast" in perp_aliases
        assert perp_aliases["fast"] == "sonar"
        assert "pro" in perp_aliases
        assert perp_aliases["pro"] == "sonar-pro"

    def test_global_alias_functions(self):
        """Test global alias resolution functions."""
        # Test resolve_model_alias
        assert resolve_model_alias("openai", "o4m") == "o4-mini-deep-research-2025-06-26"
        assert resolve_model_alias("perplexity", "fast") == "sonar"
        assert resolve_model_alias("nonexistent", "alias") is None

        # Test list_all_aliases
        all_aliases = list_all_aliases()
        assert "openai" in all_aliases
        assert "perplexity" in all_aliases
        assert "o3" in all_aliases["openai"]
        assert "fast" in all_aliases["perplexity"]

    def test_provider_alias_resolution_in_constructor(self):
        """Test that providers resolve aliases in their constructors."""
        from deep_research_client.models import ProviderConfig
        from deep_research_client.provider_params import OpenAIParams

        # Test OpenAI provider with alias
        config = ProviderConfig(name="openai", api_key="test")
        params = OpenAIParams(model="o4m")
        provider = OpenAIProvider(config, params)
        assert provider.model == "o4-mini-deep-research-2025-06-26"

        # Test with full model name (should remain unchanged)
        params2 = OpenAIParams(model="o3-deep-research-2025-06-26")
        provider2 = OpenAIProvider(config, params2)
        assert provider2.model == "o3-deep-research-2025-06-26"

        # Test with unknown alias (should remain unchanged)
        params3 = OpenAIParams(model="unknown-model")
        provider3 = OpenAIProvider(config, params3)
        assert provider3.model == "unknown-model"

    def test_all_providers_have_aliases(self):
        """Test that all providers have defined aliases for their models."""
        # OpenAI
        openai_cards = OpenAIProvider.model_cards()
        for card in openai_cards.models.values():
            assert len(card.aliases) > 0, f"Model {card.name} should have aliases"

        # Perplexity
        perplexity_cards = PerplexityProvider.model_cards()
        for card in perplexity_cards.models.values():
            assert len(card.aliases) > 0, f"Model {card.name} should have aliases"

        # Falcon
        falcon_cards = FalconProvider.model_cards()
        for card in falcon_cards.models.values():
            assert len(card.aliases) > 0, f"Model {card.name} should have aliases"

        # Consensus
        consensus_cards = ConsensusProvider.model_cards()
        for card in consensus_cards.models.values():
            assert len(card.aliases) > 0, f"Model {card.name} should have aliases"