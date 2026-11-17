"""Align one source's claims against another's: two ClaimSets in, one alignment out.

The approach has two stages. First, candidate generation proposes the pairs worth
looking at, cheaply and without a model (``candidates``). Then a model classifies
each candidate's semantic relation (``align_llm``). The evidential relation and the
shared-citation flag are not asked of the model; they are derived from the claims
(``alignment_models``). The result is a ``ClaimAlignment``, with a pair per
classified candidate (including those judged unrelated) and enough provenance to
reproduce it.

``compare_claims`` is the one-pair interface: classify a single A claim against a
single B claim, bypassing candidate generation.
"""

import asyncio
from typing import Any, Optional

from .align_llm import ALIGN_PROMPT_VERSION, DEFAULT_ALIGN_MAX_TOKENS, classify_pairs
from .alignment_datamodel import AlignedSourceType
from .alignment_models import (
    AlignedClaimPair,
    AlignedSource,
    AlignerInfo,
    ClaimAlignment,
    evidential_relation,
    shares_citation,
)
from .candidates import (
    DEFAULT_EMBED_THRESHOLD,
    DEFAULT_MIN_OVERLAP,
    CandidatePair,
    Embedder,
    candidate_pairs,
    candidate_pairs_within,
)
from .models import Claim, ClaimSet, SourceDocument

__all__ = [
    "aalign_claim_sets",
    "aalign_within",
    "acompare_claims",
    "align_claim_sets",
    "align_within",
    "compare_claims",
]


def _aligned_source(source: SourceDocument) -> AlignedSource:
    """A ClaimSet's source as the lightweight record an alignment carries."""
    return AlignedSource(
        id=source.id,
        source_type=AlignedSourceType(source.source_type),
        title=source.title,
        content_sha256=source.content_sha256,
    )


def _pair(index: int, candidate: CandidatePair, relation: Any) -> AlignedClaimPair:
    """Assemble one classified pair: the model's relation plus the derived axes."""
    return AlignedClaimPair(
        id=f"p{index}",
        claim_a_id=candidate.claim_a.id,
        claim_b_id=candidate.claim_b.id,
        claim_a_text=candidate.claim_a.claim_text,
        claim_b_text=candidate.claim_b.claim_text,
        semantic_relation=relation.semantic_relation,
        evidential_relation=evidential_relation(candidate.claim_a, candidate.claim_b),
        shares_citation=shares_citation(candidate.claim_a, candidate.claim_b),
        candidate_signals=candidate.signal_list or None,
        confidence=relation.confidence,
        rationale=relation.rationale,
    )


async def aalign_claim_sets(
    set_a: ClaimSet,
    set_b: ClaimSet,
    *,
    llm_client: Any,
    model: str,
    embedder: Optional[Embedder] = None,
    concurrency: int = 4,
    max_tokens: int = DEFAULT_ALIGN_MAX_TOKENS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
) -> ClaimAlignment:
    """Align the claims of set A against set B.

    Args:
        set_a: The first source's claims; they fill the A side of every pair.
        set_b: The second source's claims; the B side.
        llm_client: An ``openai.AsyncOpenAI``-compatible client.
        model: Model that classifies the semantic relation.
        embedder: Optional batch embedder; without it, candidate generation uses
            entity, citation and lexical signals only (no embedding dependency).
        concurrency: Classification requests in flight at once.
        max_tokens: Reply budget per pair; a reply cut off at it raises.
        min_overlap: Lexical Jaccard at or above which a pair is a candidate.
        embed_threshold: Cosine at or above which the embedder proposes a pair.

    Returns:
        The alignment: one pair per classified candidate, with the semantic
        relation judged and the evidential relation and shared-citation flag
        derived, plus the two sources and the aligner recorded.
    """
    candidates = candidate_pairs(
        set_a.claim_list, set_b.claim_list,
        min_overlap=min_overlap, embedder=embedder, embed_threshold=embed_threshold,
    )
    classification = await classify_pairs(
        candidates, llm_client, model, concurrency=concurrency, max_tokens=max_tokens,
    )
    pairs = [
        _pair(number, candidate, verdict)
        for number, (candidate, verdict) in enumerate(zip(candidates, classification.verdicts), 1)
    ]
    return ClaimAlignment(
        source_a=_aligned_source(set_a.source),
        source_b=_aligned_source(set_b.source),
        aligner=AlignerInfo(
            name="llm-pairwise",
            model=classification.model_label(model),
            prompt_version=ALIGN_PROMPT_VERSION,
            min_overlap=min_overlap,
            embedder_used=embedder is not None,
        ),
        pairs=pairs,
    )


async def aalign_within(
    claim_set: ClaimSet,
    *,
    llm_client: Any,
    model: str,
    embedder: Optional[Embedder] = None,
    concurrency: int = 4,
    max_tokens: int = DEFAULT_ALIGN_MAX_TOKENS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
) -> ClaimAlignment:
    """Align one source's claims against each other: compare a report with itself.

    This finds where a source repeats, refines or contradicts its own claims.
    Each unordered pair of distinct claims is classified once; the result is a
    ClaimAlignment whose A and B sources are the one source. In a single-topic
    document lexical overlap is less discriminative, so a higher ``min_overlap``
    than the cross-source default is often wanted.

    Args:
        claim_set: The source's claims; both sides of every pair come from it.
        llm_client: An ``openai.AsyncOpenAI``-compatible client.
        model: Model that classifies the semantic relation.
        embedder: Optional batch embedder; the texts are embedded once.
        concurrency: Classification requests in flight at once.
        max_tokens: Reply budget per pair; a reply cut off at it raises.
        min_overlap: Lexical Jaccard at or above which a pair is a candidate.
        embed_threshold: Cosine at or above which the embedder proposes a pair.

    Returns:
        The within-source alignment: a pair per classified candidate, with both
        axes, and the source recorded as both source_a and source_b.
    """
    candidates = candidate_pairs_within(
        claim_set.claim_list,
        min_overlap=min_overlap, embedder=embedder, embed_threshold=embed_threshold,
    )
    classification = await classify_pairs(
        candidates, llm_client, model, concurrency=concurrency, max_tokens=max_tokens,
    )
    pairs = [
        _pair(number, candidate, verdict)
        for number, (candidate, verdict) in enumerate(zip(candidates, classification.verdicts), 1)
    ]
    source = _aligned_source(claim_set.source)
    return ClaimAlignment(
        source_a=source,
        source_b=source,
        aligner=AlignerInfo(
            name="llm-pairwise-within",
            model=classification.model_label(model),
            prompt_version=ALIGN_PROMPT_VERSION,
            min_overlap=min_overlap,
            embedder_used=embedder is not None,
        ),
        pairs=pairs,
    )


def align_within(
    claim_set: ClaimSet,
    *,
    llm_client: Any,
    model: str,
    embedder: Optional[Embedder] = None,
    concurrency: int = 4,
    max_tokens: int = DEFAULT_ALIGN_MAX_TOKENS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
) -> ClaimAlignment:
    """Synchronous :func:`aalign_within`, for callers without an event loop."""
    return asyncio.run(aalign_within(
        claim_set, llm_client=llm_client, model=model, embedder=embedder,
        concurrency=concurrency, max_tokens=max_tokens,
        min_overlap=min_overlap, embed_threshold=embed_threshold,
    ))


def align_claim_sets(
    set_a: ClaimSet,
    set_b: ClaimSet,
    *,
    llm_client: Any,
    model: str,
    embedder: Optional[Embedder] = None,
    concurrency: int = 4,
    max_tokens: int = DEFAULT_ALIGN_MAX_TOKENS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
) -> ClaimAlignment:
    """Synchronous :func:`aalign_claim_sets`, for callers without an event loop.

    It runs its own loop with :func:`asyncio.run`, which raises ``RuntimeError``
    inside one that is already running, as in Jupyter. There,
    ``await aalign_claim_sets(...)`` instead.
    """
    return asyncio.run(aalign_claim_sets(
        set_a, set_b, llm_client=llm_client, model=model, embedder=embedder,
        concurrency=concurrency, max_tokens=max_tokens,
        min_overlap=min_overlap, embed_threshold=embed_threshold,
    ))


async def acompare_claims(
    claim_a: Claim,
    claim_b: Claim,
    *,
    llm_client: Any,
    model: str,
    max_tokens: int = DEFAULT_ALIGN_MAX_TOKENS,
) -> AlignedClaimPair:
    """Classify a single A claim against a single B claim.

    The one-pair interface, for a caller who already has the two claims and does
    not need candidate generation. The pair carries no candidate signals, since
    nothing proposed it.

    Args:
        claim_a: The A-side claim.
        claim_b: The B-side claim.
        llm_client: An ``openai.AsyncOpenAI``-compatible client.
        model: Model that classifies the semantic relation.
        max_tokens: Reply budget; a reply cut off at it raises.

    Returns:
        The classified pair, with the semantic relation judged and the evidential
        relation and shared-citation flag derived.
    """
    candidate = CandidatePair(claim_a, claim_b, frozenset())
    classification = await classify_pairs([candidate], llm_client, model, max_tokens=max_tokens)
    return _pair(1, candidate, classification.verdicts[0])


def compare_claims(
    claim_a: Claim,
    claim_b: Claim,
    *,
    llm_client: Any,
    model: str,
    max_tokens: int = DEFAULT_ALIGN_MAX_TOKENS,
) -> AlignedClaimPair:
    """Synchronous :func:`acompare_claims`, for callers without an event loop."""
    return asyncio.run(acompare_claims(
        claim_a, claim_b, llm_client=llm_client, model=model, max_tokens=max_tokens,
    ))
