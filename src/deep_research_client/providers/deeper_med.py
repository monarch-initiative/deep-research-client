"""DeepER-Med provider stub.

DeepER-Med is an evidence-based deep medical research framework introduced in
Wang et al., "DeepER-Med: Advancing Deep Evidence-Based Research in Medicine
Through Agentic AI" (arXiv:2604.15456, submitted 16 April 2026). The paper
describes an open-source paradigm with a public website and agent API, but at
the time of writing no code, API endpoint, or dataset has been published. This
provider reserves the slot in the registry so the wrapper can be flipped on
once the upstream release lands.

Reference: https://arxiv.org/abs/2604.15456
"""

from typing import Optional

from . import ResearchProvider
from ..model_cards import (
    DEEPER_MED_ARXIV_URL,
    ProviderModelCards,
    create_deeper_med_model_cards,
)
from ..models import ProviderConfig, ResearchResult
from ..provider_params import DeeperMedParams

__all__ = [
    "DEEPER_MED_ARXIV_URL",
    "DEEPER_MED_UNAVAILABLE_MESSAGE",
    "DeeperMedProvider",
]

DEEPER_MED_UNAVAILABLE_MESSAGE = (
    "DeepER-Med has no public API or code release yet, so this provider cannot "
    f"run research. See {DEEPER_MED_ARXIV_URL} for the paper describing it."
)


class DeeperMedProvider(ResearchProvider):
    """Placeholder provider for the DeepER-Med agentic medical research system.

    The provider is registered so users can discover it and so downstream
    integrations (CLI, model cards, params) can refer to it by name. Calling
    :meth:`research` raises :class:`NotImplementedError` until an upstream
    endpoint exists.

    Because :meth:`is_available` is always ``False``, callers going through
    :class:`~deep_research_client.client.DeepResearchClient` are rejected before
    :meth:`research` runs; :meth:`unavailable_reason` is what carries the
    explanation (and the arXiv pointer) out to them.

    >>> provider = DeeperMedProvider(ProviderConfig(name="deeper_med"))
    >>> provider.is_available()
    False
    >>> provider.get_default_model()
    'deeper-med-agentic'
    >>> "arxiv.org" in provider.unavailable_reason()
    True
    """

    def __init__(self, config: ProviderConfig, params: Optional[DeeperMedParams] = None):
        """Initialize the stub provider.

        Args:
            config: Provider configuration (no API key is used)
            params: Optional DeepER-Med parameters
        """
        self.params = params or DeeperMedParams()
        super().__init__(config, self.params.model)

    def get_default_model(self) -> str:
        """Return the only model card name this stub publishes."""
        return "deeper-med-agentic"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Return the model cards describing the (not yet callable) system."""
        return create_deeper_med_model_cards()

    def is_available(self) -> bool:
        """DeepER-Med is never available until an upstream endpoint ships."""
        return False

    def unavailable_reason(self) -> str:
        """Explain the stub status and point at the paper.

        Overrides the generic base-class message so the arXiv reference reaches
        users who request this provider through the client or CLI.
        """
        return DEEPER_MED_UNAVAILABLE_MESSAGE

    async def research(self, query: str) -> ResearchResult:
        """Always raise: there is no upstream endpoint to call.

        Args:
            query: The research question (ignored)

        Raises:
            NotImplementedError: Always, with a pointer to the arXiv paper
        """
        raise NotImplementedError(DEEPER_MED_UNAVAILABLE_MESSAGE)
