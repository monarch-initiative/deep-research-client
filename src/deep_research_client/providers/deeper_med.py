"""DeepER-Med provider stub.

DeepER-Med is an evidence-based deep medical research framework introduced in
Wang et al., "DeepER-Med: Advancing Deep Evidence-Based Research in Medicine
Through Agentic AI" (arXiv:2604.15456, NIH/NLM, April 2026). The paper
describes an open-source paradigm with a public website and agent API, but at
the time of writing no code, API endpoint, or dataset has been published. This
provider reserves the slot in the registry so the wrapper can be flipped on
once the upstream release lands.

Reference: https://arxiv.org/abs/2604.15456
Likely upstream home: https://github.com/ncbi-nlp (Zhiyong Lu's lab)
"""

from typing import Optional

from . import ResearchProvider
from ..model_cards import ProviderModelCards, create_deeper_med_model_cards
from ..models import ProviderConfig, ResearchResult
from ..provider_params import DeeperMedParams

DEEPER_MED_ARXIV_URL = "https://arxiv.org/abs/2604.15456"
DEEPER_MED_UNAVAILABLE_MESSAGE = (
    "DeepER-Med has no public API or code release yet. "
    f"See {DEEPER_MED_ARXIV_URL} and watch https://github.com/ncbi-nlp."
)


class DeeperMedProvider(ResearchProvider):
    """Placeholder provider for the DeepER-Med agentic medical research system.

    The provider is registered so users can discover it and so downstream
    integrations (CLI, model cards, params) can refer to it by name. Calling
    :meth:`research` raises :class:`NotImplementedError` until an upstream
    endpoint exists.

    >>> provider = DeeperMedProvider(ProviderConfig(name="deeper_med"))
    >>> provider.is_available()
    False
    >>> provider.get_default_model()
    'deeper-med-agentic'
    """

    def __init__(self, config: ProviderConfig, params: Optional[DeeperMedParams] = None):
        self.params = params or DeeperMedParams()
        super().__init__(config, self.params.model)

    def get_default_model(self) -> str:
        return "deeper-med-agentic"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        return create_deeper_med_model_cards()

    def is_available(self) -> bool:
        """DeepER-Med is never available until an upstream endpoint ships."""
        return False

    async def research(self, query: str) -> ResearchResult:
        raise NotImplementedError(DEEPER_MED_UNAVAILABLE_MESSAGE)
