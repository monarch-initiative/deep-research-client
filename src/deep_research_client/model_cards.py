"""Model cards for research providers - descriptions, costs, and capabilities.

The capability, resource, and archetype controlled vocabularies
(:class:`ResearchCapability`, :class:`ResearchResource`,
:class:`ProviderArchetype`) are defined in the LinkML schema at
``src/deep_research_client/schema/deep_research_client.yaml`` and imported here
from the generated datamodel. See ``docs/reference/capabilities.md`` for the
conceptual model.
"""

from enum import Enum
from typing import Dict, Optional, List
from pydantic import BaseModel, Field, ConfigDict

from .datamodel import (
    ProviderArchetype,
    ResearchCapability,
    ResearchResource,
)

# Backwards-compatible alias. ``ResearchCapability`` is the canonical name; older
# code and imports referring to ``ModelCapability`` continue to work.
ModelCapability = ResearchCapability

# The generated enum uses lower_case member names (e.g. ``web_search``), whereas
# the historical ``ModelCapability`` enum used UPPER_CASE names (``WEB_SEARCH``).
# Register UPPER_CASE attribute aliases pointing at the same members so existing
# attribute-style access (``ModelCapability.WEB_SEARCH``) keeps working. The
# string values were already identical, so value-based access was never affected.
for _capability in ResearchCapability:
    setattr(ResearchCapability, _capability.name.upper(), _capability)
del _capability


class CostLevel(str, Enum):
    """Cost levels for research models."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TimeEstimate(str, Enum):
    """Estimated response time categories."""
    FAST = "fast"          # < 30 seconds
    MEDIUM = "medium"      # 30 seconds - 2 minutes
    SLOW = "slow"          # 2-10 minutes
    VERY_SLOW = "very_slow"  # > 10 minutes


class ModelCard(BaseModel):
    """Information card for a research model."""

    name: str = Field(description="Model identifier")
    display_name: str = Field(description="Human-readable model name")
    description: str = Field(description="Detailed description of model capabilities")
    cost_level: CostLevel = Field(description="Relative cost level")
    time_estimate: TimeEstimate = Field(description="Expected response time")
    capabilities: List[ResearchCapability] = Field(
        default_factory=list,
        description="Functional capabilities the model exposes (what it can do)"
    )
    resources: List[ResearchResource] = Field(
        default_factory=list,
        description="Data sources / knowledge bases the model wraps (what it can reach)"
    )
    archetype: Optional[ProviderArchetype] = Field(
        default=None,
        description=(
            "Where the provider sits on the retrieval -> co-scientist spectrum. "
            "A conventional deep-research tool is a 'synthesizer'; a 'co_scientist' "
            "is a superset that also forms hypotheses and runs code."
        )
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Short aliases for convenient CLI usage"
    )

    # Optional detailed information
    context_window: Optional[int] = Field(default=None, description="Context window size in tokens")
    max_output: Optional[int] = Field(default=None, description="Maximum output tokens")
    pricing_notes: Optional[str] = Field(default=None, description="Additional pricing information")
    use_cases: List[str] = Field(
        default_factory=list,
        description="Recommended use cases"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Known limitations"
    )

    model_config = ConfigDict(use_enum_values=True)


class ProviderModelCards(BaseModel):
    """Collection of model cards for a provider."""

    provider_name: str = Field(description="Provider identifier")
    default_model: str = Field(description="Default model name")
    models: Dict[str, ModelCard] = Field(description="Map of model names to model cards")

    def get_model_card(self, model_name: str) -> Optional[ModelCard]:
        """Get model card by name."""
        return self.models.get(model_name)

    def resolve_model_name(self, model_or_alias: str) -> Optional[str]:
        """Resolve a model name or alias to the full model name.

        Args:
            model_or_alias: Either a full model name or a short alias

        Returns:
            Full model name if found, None otherwise
        """
        # First check if it's a direct model name
        if model_or_alias in self.models:
            return model_or_alias

        # Then check aliases
        for model_name, card in self.models.items():
            if model_or_alias in card.aliases:
                return model_name

        return None

    def get_model_card_by_alias(self, model_or_alias: str) -> Optional[ModelCard]:
        """Get model card by name or alias."""
        resolved_name = self.resolve_model_name(model_or_alias)
        if resolved_name:
            return self.models[resolved_name]
        return None

    def list_models(self) -> List[str]:
        """List available model names."""
        return list(self.models.keys())

    def list_aliases(self) -> Dict[str, str]:
        """List all aliases mapped to their full model names."""
        alias_map = {}
        for model_name, card in self.models.items():
            for alias in card.aliases:
                alias_map[alias] = model_name
        return alias_map

    def _unique_cards(self, cards: List[ModelCard]) -> List[ModelCard]:
        """Deduplicate cards by name, preserving order.

        Some providers alias one ``ModelCard`` object under several keys (e.g.
        cyberian maps both its model name and workflow name to the same card),
        which would otherwise surface the same card twice from these finders.
        """
        seen: set[str] = set()
        unique: List[ModelCard] = []
        for card in cards:
            if card.name not in seen:
                seen.add(card.name)
                unique.append(card)
        return unique

    def get_models_by_cost(self, cost_level: CostLevel) -> List[ModelCard]:
        """Get models filtered by cost level."""
        return self._unique_cards(
            [card for card in self.models.values() if card.cost_level == cost_level]
        )

    def get_models_by_time(self, time_estimate: TimeEstimate) -> List[ModelCard]:
        """Get models filtered by time estimate."""
        return self._unique_cards(
            [card for card in self.models.values() if card.time_estimate == time_estimate]
        )

    def get_models_with_capability(self, capability: ResearchCapability) -> List[ModelCard]:
        """Get models that have a specific capability."""
        return self._unique_cards(
            [card for card in self.models.values() if capability in card.capabilities]
        )

    def get_models_with_resource(self, resource: ResearchResource) -> List[ModelCard]:
        """Get models that wrap a specific data resource."""
        return self._unique_cards(
            [card for card in self.models.values() if resource in card.resources]
        )

    def get_models_by_archetype(self, archetype: ProviderArchetype) -> List[ModelCard]:
        """Get models matching a given provider archetype."""
        return self._unique_cards(
            [card for card in self.models.values() if card.archetype == archetype]
        )


def create_openai_model_cards() -> ProviderModelCards:
    """Create model cards for OpenAI provider."""

    o3_deep_research = ModelCard(
        name="o3-deep-research-2025-06-26",
        display_name="OpenAI o3 Deep Research",
        description=(
            "Comprehensive deep research model optimized for in-depth analysis and "
            "synthesis. Performs exhaustive web searches with multiple iterations and "
            "produces analyst-grade reports with detailed citations."
        ),
        cost_level=CostLevel.VERY_HIGH,
        time_estimate=TimeEstimate.VERY_SLOW,
        capabilities=[
            ResearchCapability.web_search,
            ResearchCapability.real_time_data,
            ResearchCapability.code_interpretation,
            ResearchCapability.citation_tracking,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.synthesizer,
        resources=[ResearchResource.general_web],
        aliases=["o3", "o3-deep", "o3dr"],
        context_window=128000,
        pricing_notes="$10/$40 per million input/output tokens + $10/1K web searches + $0.03/code interpreter session",
        use_cases=[
            "Comprehensive research reports",
            "Academic literature reviews",
            "Market research analysis",
            "Technical deep dives",
            "Multi-source fact checking"
        ],
        limitations=[
            "Very high cost per query",
            "Long response times (5-15 minutes)",
            "Limited monthly usage quotas",
            "Requires patience for complex queries"
        ]
    )

    o4_mini_deep_research = ModelCard(
        name="o4-mini-deep-research-2025-06-26",
        display_name="OpenAI o4-mini Deep Research",
        description=(
            "Lightweight and cost-effective deep research model designed for faster "
            "responses while maintaining research quality. Ideal for latency-sensitive "
            "use cases and frequent queries."
        ),
        cost_level=CostLevel.MEDIUM,
        time_estimate=TimeEstimate.MEDIUM,
        capabilities=[
            ResearchCapability.web_search,
            ResearchCapability.real_time_data,
            ResearchCapability.code_interpretation,
            ResearchCapability.citation_tracking,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.synthesizer,
        resources=[ResearchResource.general_web],
        aliases=["o4m", "o4-mini", "o4mini", "mini"],
        context_window=128000,
        pricing_notes="$2/$8 per million input/output tokens + tool usage fees",
        use_cases=[
            "Quick research queries",
            "Fact checking",
            "News summaries",
            "Educational content",
            "Rapid prototyping of research"
        ],
        limitations=[
            "Less comprehensive than o3",
            "May require follow-up queries for complex topics",
            "Still subject to tool usage costs"
        ]
    )

    return ProviderModelCards(
        provider_name="openai",
        default_model="o3-deep-research-2025-06-26",
        models={
            "o3-deep-research-2025-06-26": o3_deep_research,
            "o4-mini-deep-research-2025-06-26": o4_mini_deep_research
        }
    )


def create_perplexity_model_cards() -> ProviderModelCards:
    """Create model cards for Perplexity provider."""

    sonar_deep_research = ModelCard(
        name="sonar-deep-research",
        display_name="Perplexity Sonar Deep Research",
        description=(
            "Comprehensive research model with real-time web search and extensive "
            "source analysis. Optimized for thorough investigation with multiple "
            "search iterations and source validation."
        ),
        cost_level=CostLevel.HIGH,
        time_estimate=TimeEstimate.SLOW,
        capabilities=[
            ResearchCapability.web_search,
            ResearchCapability.real_time_data,
            ResearchCapability.citation_tracking,
            ResearchCapability.multi_language,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.synthesizer,
        resources=[ResearchResource.general_web],
        aliases=["deep", "deep-research", "sdr"],
        context_window=200000,
        pricing_notes="Higher cost per query, includes comprehensive web search",
        use_cases=[
            "In-depth research projects",
            "Academic literature reviews",
            "Current events analysis",
            "Multi-source verification",
            "Comprehensive fact-checking"
        ],
        limitations=[
            "Higher cost than basic models",
            "Longer response times",
            "May over-research simple queries"
        ]
    )

    sonar_pro = ModelCard(
        name="sonar-pro",
        display_name="Perplexity Sonar Pro",
        description=(
            "Fast and efficient search model with enhanced reasoning capabilities. "
            "Balanced approach between speed and research depth, suitable for "
            "professional use cases."
        ),
        cost_level=CostLevel.MEDIUM,
        time_estimate=TimeEstimate.MEDIUM,
        capabilities=[
            ResearchCapability.web_search,
            ResearchCapability.real_time_data,
            ResearchCapability.citation_tracking,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.synthesizer,
        resources=[ResearchResource.general_web],
        aliases=["pro", "sp"],
        context_window=200000,
        pricing_notes="Mid-tier pricing with good performance",
        use_cases=[
            "Business research",
            "News analysis",
            "Quick fact-checking",
            "Professional reports",
            "Market intelligence"
        ],
        limitations=[
            "Less comprehensive than deep research",
            "May require follow-up for complex topics"
        ]
    )

    sonar = ModelCard(
        name="sonar",
        display_name="Perplexity Sonar",
        description=(
            "Standard search model providing quick answers with real-time web search. "
            "Cost-effective option for basic research needs and frequent queries."
        ),
        cost_level=CostLevel.LOW,
        time_estimate=TimeEstimate.FAST,
        capabilities=[
            ResearchCapability.web_search,
            ResearchCapability.real_time_data,
            ResearchCapability.citation_tracking,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.synthesizer,
        resources=[ResearchResource.general_web],
        aliases=["basic", "fast", "s"],
        context_window=100000,
        pricing_notes="Most cost-effective option",
        use_cases=[
            "Quick questions",
            "Basic fact-checking",
            "News updates",
            "Simple research queries",
            "High-frequency usage"
        ],
        limitations=[
            "Limited depth for complex topics",
            "Fewer search iterations",
            "Less comprehensive analysis"
        ]
    )

    return ProviderModelCards(
        provider_name="perplexity",
        default_model="sonar-deep-research",
        models={
            "sonar-deep-research": sonar_deep_research,
            "sonar-pro": sonar_pro,
            "sonar": sonar
        }
    )


def create_falcon_model_cards() -> ProviderModelCards:
    """Create model cards for FutureHouse Falcon provider."""

    falcon_api = ModelCard(
        name="FutureHouse Falcon API",
        display_name="FutureHouse Falcon",
        description=(
            "Specialized scientific literature search and synthesis model with access "
            "to curated academic databases and scientific literature. Optimized for "
            "research-quality academic analysis."
        ),
        cost_level=CostLevel.HIGH,
        time_estimate=TimeEstimate.SLOW,
        capabilities=[
            ResearchCapability.academic_search,
            ResearchCapability.scientific_literature,
            ResearchCapability.citation_tracking,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.synthesizer,
        resources=[
            ResearchResource.pubmed,
            ResearchResource.semantic_scholar,
            ResearchResource.preprint_servers,
        ],
        # "Edison Scientific Literature" is what FalconProvider.get_default_model()
        # returns after the FutureHouse -> Edison rename; without it here, the
        # provider's own default resolves to no card at all.
        aliases=["falcon", "fh", "science", "Edison Scientific Literature"],
        pricing_notes="Academic research pricing, varies by usage",
        use_cases=[
            "Scientific literature reviews",
            "Academic research synthesis",
            "Medical research analysis",
            "Technical paper discovery",
            "Grant writing support"
        ],
        limitations=[
            "Limited to academic/scientific sources",
            "No general web search",
            "Filtering reliability issues noted",
            "Specialized domain focus"
        ]
    )

    return ProviderModelCards(
        provider_name="falcon",
        default_model="FutureHouse Falcon API",
        models={
            "FutureHouse Falcon API": falcon_api
        }
    )


def create_asta_model_cards() -> ProviderModelCards:
    """Create model cards for Asta retrieval."""

    asta_retrieval = ModelCard(
        name="Asta Scientific Corpus Retrieval",
        display_name="Asta Scientific Corpus Retrieval",
        description=(
            "Semantic Scholar-backed literature retrieval using Asta's paper search "
            "and snippet search tools, returned directly without synthesis."
        ),
        cost_level=CostLevel.LOW,
        time_estimate=TimeEstimate.FAST,
        capabilities=[
            ResearchCapability.academic_search,
            ResearchCapability.scientific_literature,
            ResearchCapability.citation_tracking,
            # The one term that exists to mark this archetype, on the one
            # provider that is it: Asta returns evidence, never a synthesis.
            ResearchCapability.retrieval_only,
        ],
        archetype=ProviderArchetype.retriever,
        resources=[ResearchResource.semantic_scholar],
        aliases=["asta", "retrieval", "snippets"],
        pricing_notes="Free retrieval-only provider using the Asta MCP service",
        use_cases=[
            "Literature discovery",
            "Passage-level evidence lookup",
            "Paper triage before downstream analysis",
            "Scientific corpus exploration"
        ],
        limitations=[
            "No synthesis step",
            "Scientific literature only",
            "Report content mirrors retrieved evidence"
        ]
    )

    return ProviderModelCards(
        provider_name="asta",
        default_model="Asta Scientific Corpus Retrieval",
        models={
            "Asta Scientific Corpus Retrieval": asta_retrieval,
        }
    )


def create_consensus_model_cards() -> ProviderModelCards:
    """Create model cards for Consensus provider."""

    consensus_search = ModelCard(
        name="Consensus Academic Search",
        display_name="Consensus AI Academic Search",
        description=(
            "Peer-reviewed academic paper search and analysis focused on evidence-based "
            "research. Provides structured analysis of academic literature with "
            "study quality assessment and meta-analysis capabilities."
        ),
        cost_level=CostLevel.LOW,
        time_estimate=TimeEstimate.FAST,
        capabilities=[
            ResearchCapability.academic_search,
            ResearchCapability.citation_tracking,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.synthesizer,
        resources=[ResearchResource.semantic_scholar],
        aliases=["consensus", "academic", "papers", "c"],
        pricing_notes="$6.99/month for premium, free tier available",
        use_cases=[
            "Academic literature search",
            "Evidence-based research",
            "Meta-analysis preparation",
            "Systematic reviews",
            "Research validation"
        ],
        limitations=[
            "Academic papers only",
            "No web search capability",
            "Limited to peer-reviewed content",
            "Requires API approval"
        ]
    )

    return ProviderModelCards(
        provider_name="consensus",
        default_model="Consensus Academic Search",
        models={
            "Consensus Academic Search": consensus_search
        }
    )


def create_openscientist_model_cards() -> ProviderModelCards:
    """Create model cards for OpenScientist research provider."""

    autonomous = ModelCard(
        name="openscientist-autonomous",
        display_name="OpenScientist Autonomous Research",
        description=(
            "Autonomous iterative research agent from Berkeley Lab. Runs "
            "multi-iteration hypothesis-driven research using PubMed search "
            "and code execution. Produces comprehensive markdown reports "
            "with structured PMID citations. Best for deep scientific "
            "literature reviews where thoroughness matters more than speed."
        ),
        cost_level=CostLevel.HIGH,
        time_estimate=TimeEstimate.VERY_SLOW,
        capabilities=[
            ResearchCapability.academic_search,
            ResearchCapability.scientific_literature,
            ResearchCapability.citation_tracking,
            ResearchCapability.code_interpretation,
            ResearchCapability.hypothesis_generation,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.co_scientist,
        resources=[ResearchResource.pubmed],
        aliases=["openscientist", "os", "berkeley"],
        pricing_notes=(
            "Runs Claude under the hood; costs depend on iteration count. "
            "5 iterations ~ moderate cost, 10+ iterations ~ high cost."
        ),
        use_cases=[
            "Disease pathophysiology research",
            "Scientific literature reviews",
            "Hypothesis-driven investigation",
            "PubMed-backed evidence synthesis",
            "Biomedical mechanism discovery",
        ],
        limitations=[
            "Very slow (10-60+ minutes per job)",
            "Requires account approval on openscientist.io",
            "PubMed-focused (limited general web search)",
            "High cost for many iterations",
        ],
    )

    return ProviderModelCards(
        provider_name="openscientist",
        default_model="openscientist-autonomous",
        models={
            "openscientist-autonomous": autonomous,
        },
    )


def create_claude_code_model_cards() -> ProviderModelCards:
    """Create model cards for the Claude Code research provider."""

    default = ModelCard(
        name="claude-code-default",
        display_name="Claude Code (account default model)",
        description=(
            "Runs the local Claude Code CLI in non-interactive mode, letting its "
            "agentic tools (web search, web fetch, file reading) drive a multi-step "
            "research process that yields a cited markdown report. Uses whichever "
            "model the local Claude Code installation defaults to."
        ),
        cost_level=CostLevel.MEDIUM,
        time_estimate=TimeEstimate.SLOW,
        capabilities=[
            ResearchCapability.web_search,
            ResearchCapability.citation_tracking,
            ResearchCapability.code_interpretation,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.agentic_researcher,
        resources=[ResearchResource.general_web],
        aliases=["claude", "claude-code", "cc", "default"],
        pricing_notes=(
            "Billing is handled by the local Claude Code installation (subscription "
            "or Anthropic API key); no separate provider API key is required."
        ),
        use_cases=[
            "Deep research using an already-configured Claude Code environment",
            "Agentic web research with citations",
            "Research in environments where curation already runs Claude Code",
        ],
        limitations=[
            "Requires the `claude` CLI to be installed and authenticated locally",
            "Restricted to a read-only research toolset by default (web search/fetch)",
            "Non-deterministic results",
        ],
    )

    return ProviderModelCards(
        provider_name="claude_code",
        default_model="claude-code-default",
        models={
            "claude-code-default": default,
        },
    )


# Single source of truth for the DeepER-Med citation. Defined here rather than in
# providers/deeper_med.py because that module imports from this one; the provider
# re-exports it so callers can reference either.
DEEPER_MED_ARXIV_ID = "2604.15456"
DEEPER_MED_ARXIV_URL = f"https://arxiv.org/abs/{DEEPER_MED_ARXIV_ID}"


def create_deeper_med_model_cards() -> ProviderModelCards:
    """Create model cards for the DeepER-Med stub provider.

    DeepER-Med (arXiv:2604.15456) is an evidence-based agentic deep research
    framework for medicine. No public API exists yet; this card documents the
    system so it appears in `providers` listings and so the wrapper can flip
    on once an endpoint is available.

    Every attribute below is transcribed from the paper, not measured against a
    running system -- see the ``limitations`` entries.
    """

    deeper_med = ModelCard(
        name="deeper-med-agentic",
        display_name="DeepER-Med Agentic Medical Research (stub)",
        description=(
            "Evidence-based agentic deep research framework for medicine "
            f"(Wang et al., arXiv:{DEEPER_MED_ARXIV_ID}). As described in the "
            "paper: decomposes queries into hierarchical sub-questions, "
            "retrieves from PubMed, ClinicalTrials.gov, and the PrimeKG "
            "knowledge graph, and synthesizes with traceable references drawn "
            "directly from source databases. STUB: no public API has been "
            "released; calls raise NotImplementedError."
        ),
        # Inferred from the paper's description of a multi-agent retrieval and
        # synthesis pipeline. Nothing here has been observed -- there is no
        # endpoint to observe.
        cost_level=CostLevel.HIGH,
        time_estimate=TimeEstimate.SLOW,
        capabilities=[
            ResearchCapability.academic_search,
            ResearchCapability.scientific_literature,
            ResearchCapability.citation_tracking,
            ResearchCapability.evidence_synthesis,
        ],
        # Also from the paper: it plans and retrieves over named sources but
        # does not run experiments or code, so it stops short of co_scientist.
        archetype=ProviderArchetype.agentic_researcher,
        resources=[
            ResearchResource.pubmed,
            ResearchResource.clinical_trials,
            # PrimeKG, the paper's knowledge graph.
            ResearchResource.biomedical_databases,
        ],
        aliases=["deeper-med", "deepermed"],
        pricing_notes="Unknown - upstream API not yet released",
        use_cases=[
            "Evidence-based medical research synthesis",
            "Hypothesis verification grounded in PubMed",
            "Precision oncology tumor board preparation",
            "Clinical trial evidence aggregation",
        ],
        limitations=[
            "Stub: no upstream endpoint, so this model cannot be invoked",
            "Cost, speed, and capabilities are claims from the paper, not measured",
            f"Reference: {DEEPER_MED_ARXIV_URL}",
        ],
    )

    return ProviderModelCards(
        provider_name="deeper_med",
        default_model="deeper-med-agentic",
        models={"deeper-med-agentic": deeper_med},
    )


# Registry of all provider model cards
# NOTE: PROVIDER_MODEL_CARDS is defined at the end of this module, after every
# card factory (including cyberian and biomni) has been declared, so all
# providers are registered in a single place.


def get_provider_model_cards(provider_name: str) -> Optional[ProviderModelCards]:
    """Get model cards for a specific provider."""
    return PROVIDER_MODEL_CARDS.get(provider_name)


def list_all_models() -> Dict[str, List[str]]:
    """List all available models by provider."""
    return {
        provider: cards.list_models()
        for provider, cards in PROVIDER_MODEL_CARDS.items()
    }


def find_models_by_cost(cost_level: CostLevel) -> Dict[str, List[ModelCard]]:
    """Find models across all providers by cost level."""
    result = {}
    for provider, cards in PROVIDER_MODEL_CARDS.items():
        models = cards.get_models_by_cost(cost_level)
        if models:
            result[provider] = models
    return result


def find_models_by_capability(capability: ResearchCapability) -> Dict[str, List[ModelCard]]:
    """Find models across all providers by capability."""
    result = {}
    for provider, cards in PROVIDER_MODEL_CARDS.items():
        models = cards.get_models_with_capability(capability)
        if models:
            result[provider] = models
    return result


def find_models_by_resource(resource: ResearchResource) -> Dict[str, List[ModelCard]]:
    """Find models across all providers that wrap a given data resource."""
    result = {}
    for provider, cards in PROVIDER_MODEL_CARDS.items():
        models = cards.get_models_with_resource(resource)
        if models:
            result[provider] = models
    return result


def find_models_by_archetype(archetype: ProviderArchetype) -> Dict[str, List[ModelCard]]:
    """Find models across all providers matching a given archetype."""
    result = {}
    for provider, cards in PROVIDER_MODEL_CARDS.items():
        models = cards.get_models_by_archetype(archetype)
        if models:
            result[provider] = models
    return result


def resolve_model_alias(provider_name: str, model_or_alias: str) -> Optional[str]:
    """Resolve a model alias to the full model name for a specific provider.

    Args:
        provider_name: Name of the provider
        model_or_alias: Model name or alias to resolve

    Returns:
        Full model name if found, None if provider or model not found
    """
    cards = get_provider_model_cards(provider_name)
    if cards:
        return cards.resolve_model_name(model_or_alias)
    return None


def list_all_aliases() -> Dict[str, Dict[str, str]]:
    """List all aliases across all providers.

    Returns:
        Dict mapping provider names to their alias->model_name mappings
    """
    result = {}
    for provider, cards in PROVIDER_MODEL_CARDS.items():
        aliases = cards.list_aliases()
        if aliases:
            result[provider] = aliases
    return result


def create_cyberian_model_cards() -> ProviderModelCards:
    """Create model cards for Cyberian agent-based research provider."""

    deep_research = ModelCard(
        name="Cyberian Deep Research",
        display_name="Cyberian Agent-Based Deep Research",
        description=(
            "Agent-based iterative research using AI agents (Claude, Aider, etc.) "
            "to perform multi-step research workflows with citation management, "
            "report synthesis, and systematic literature review capabilities."
        ),
        cost_level=CostLevel.HIGH,  # Agent-based, potentially many LLM calls
        time_estimate=TimeEstimate.VERY_SLOW,  # Iterative multi-step process
        capabilities=[
            ResearchCapability.web_search,
            ResearchCapability.academic_search,
            ResearchCapability.citation_tracking,
            ResearchCapability.evidence_synthesis,
        ],
        archetype=ProviderArchetype.agentic_researcher,
        resources=[
            ResearchResource.general_web,
            ResearchResource.semantic_scholar,
        ],
        aliases=["cyberian", "agent-research", "cy", "deep-research"],
        pricing_notes=(
            "Costs depend on underlying agent (Claude, etc.) and research depth. "
            "May involve multiple LLM API calls during iterative research."
        ),
        use_cases=[
            "Deep comprehensive research",
            "Systematic literature reviews",
            "Iterative hypothesis exploration",
            "Citation graph generation",
            "Multi-source synthesis"
        ],
        limitations=[
            "Slow (multiple agent iterations)",
            "High cost (multiple LLM calls)",
            "Requires agentapi server",
            "Non-deterministic results",
            "Needs local compute resources"
        ]
    )

    return ProviderModelCards(
        provider_name="cyberian",
        default_model="Cyberian Deep Research",
        models={
            # "deep-research" is exposed as an alias (see aliases=) rather than a
            # second key so the card is not listed twice by list_models().
            "Cyberian Deep Research": deep_research,
        }
    )


def create_biomni_model_cards() -> ProviderModelCards:
    """Create model cards for the Biomni biomedical co-scientist provider."""

    a1 = ModelCard(
        name="biomni-a1",
        display_name="Biomni A1 Biomedical Agent",
        description=(
            "General-purpose biomedical AI agent from Stanford SNAP. Wraps a large "
            "toolbox of biomedical software and curated databases and executes "
            "generated code to plan and carry out research tasks (e.g. designing a "
            "CRISPR screen, annotating variants, analysing omics data). Runs "
            "locally via the `biomni` package against an auto-downloaded data lake. "
            "A hypothesis-driven co-scientist rather than a pure literature "
            "reviewer."
        ),
        cost_level=CostLevel.HIGH,  # Drives an LLM plus heavy local computation
        time_estimate=TimeEstimate.VERY_SLOW,  # Multi-step agentic execution
        capabilities=[
            ResearchCapability.scientific_literature,
            ResearchCapability.code_interpretation,
            ResearchCapability.data_analysis,
            ResearchCapability.hypothesis_generation,
            ResearchCapability.experiment_design,
            ResearchCapability.evidence_synthesis,
            ResearchCapability.citation_tracking,
        ],
        archetype=ProviderArchetype.co_scientist,
        resources=[
            ResearchResource.pubmed,
            ResearchResource.general_web,
            ResearchResource.biomedical_databases,
            ResearchResource.genomic_databases,
            ResearchResource.chemical_databases,
            ResearchResource.protein_structure_databases,
        ],
        aliases=["biomni", "a1", "coscientist"],
        pricing_notes=(
            "Drives an underlying LLM (Claude by default) and downloads an ~11GB "
            "data lake on first run. Cost depends on the LLM provider and task "
            "complexity; local compute and disk are also required."
        ),
        use_cases=[
            "Designing experiments (e.g. CRISPR screens)",
            "Variant annotation and interpretation",
            "Omics and sequence data analysis",
            "Hypothesis-driven biomedical investigation",
            "Wet-lab / dry-lab protocol drafting",
        ],
        limitations=[
            "Requires the optional `biomni` package (pip install deep-research-client[biomni])",
            "Downloads a large (~11GB) data lake on first run",
            "Executes generated code locally; run in a trusted/sandboxed environment",
            "Needs an LLM API key (e.g. ANTHROPIC_API_KEY) for the underlying model",
            "Very slow and non-deterministic",
        ],
    )

    return ProviderModelCards(
        provider_name="biomni",
        default_model="biomni-a1",
        models={
            "biomni-a1": a1,
        },
    )


# Registry of all provider model cards. Defined here, after every factory, so
# each provider is registered in exactly one place.
PROVIDER_MODEL_CARDS: Dict[str, ProviderModelCards] = {
    "openai": create_openai_model_cards(),
    "perplexity": create_perplexity_model_cards(),
    "falcon": create_falcon_model_cards(),
    "asta": create_asta_model_cards(),
    "consensus": create_consensus_model_cards(),
    "openscientist": create_openscientist_model_cards(),
    "claude_code": create_claude_code_model_cards(),
    "deeper_med": create_deeper_med_model_cards(),
    "cyberian": create_cyberian_model_cards(),
    "biomni": create_biomni_model_cards(),
}
