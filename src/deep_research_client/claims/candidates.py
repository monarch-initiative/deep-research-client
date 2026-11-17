"""Propose claim pairs worth classifying, so the model is not asked about every pair.

Comparing every A claim with every B claim with a model does not scale: two
reports of a few hundred claims each would be a hundred thousand calls. Instead a
cheap pass proposes the pairs that share something, and only those are sent to the
model (see ``align``).

Four signals propose a pair, and a pair may carry several:

- ``ENTITY``: the two claims share a grounded entity (the same CURIE).
- ``CITATION``: the two claims cite a work with the same resolved reference_id.
- ``LEXICAL``: the two claims' content words overlap at or above a threshold
  (Jaccard over :func:`deep_research_client.validation.relevance.content_terms`,
  which strips markup, drops stopwords and folds plurals).
- ``EMBEDDING``: a caller-supplied embedder scores the two claim texts as similar.
  Absent unless an embedder is given; this is the optional dependency hook.

Everything here is a pure function of the two claim lists (and, for embeddings, a
caller-supplied function), so candidate generation is tested without a model.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..validation.relevance import content_terms
from .alignment_datamodel import CandidateSignal
from .models import Claim, resolved_reference_ids

__all__ = [
    "DEFAULT_EMBED_THRESHOLD",
    "DEFAULT_MIN_OVERLAP",
    "CandidatePair",
    "Embedder",
    "candidate_pairs",
    "candidate_pairs_within",
    "grounded_ids",
    "jaccard",
]

#: An embedder maps a batch of texts to a vector each, in the same order. A
#: caller supplies one (an OpenAI embeddings call, a local model); none is built
#: in, so the default path needs no embedding dependency.
Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]

#: Lexical overlap at or above which a pair is proposed. A third of the pooled
#: content words shared is enough to be worth a model's time without flooding it.
DEFAULT_MIN_OVERLAP = 0.34

#: Cosine similarity at or above which an embedder proposes a pair.
DEFAULT_EMBED_THRESHOLD = 0.80


@dataclass(frozen=True)
class CandidatePair:
    """One A claim paired with one B claim, with why it was proposed.

    Attributes:
        claim_a: The A-side claim.
        claim_b: The B-side claim.
        signals: The signals that proposed the pair, at least one.
    """

    claim_a: Claim
    claim_b: Claim
    signals: frozenset[CandidateSignal]

    @property
    def signal_list(self) -> list[CandidateSignal]:
        """The signals in a stable order, for recording on a pair.

        Returns:
            The signals sorted by name.
        """
        return sorted(self.signals, key=lambda s: s.value)


def grounded_ids(claim: Claim) -> set[str]:
    """The CURIEs a claim is grounded to: its entities, subject and object.

    Args:
        claim: The claim to read.

    Returns:
        Every non-empty id on the claim's entities, subject and object.

    >>> from deep_research_client.claims import EntityMention
    >>> claim = Claim(id="c", claim_text="X.", anchor_status="EXACT", citation_status="UNCITED",
    ...               subject=EntityMention(label="Marfan", id="MONDO:0007947"),
    ...               entities=[EntityMention(label="FBN1", id="HGNC:3603"),
    ...                         EntityMention(label="aorta")])
    >>> sorted(grounded_ids(claim))
    ['HGNC:3603', 'MONDO:0007947']
    """
    mentions = list(claim.entities or [])
    mentions += [m for m in (claim.subject, claim.object) if m is not None]
    return {m.id for m in mentions if m.id}


def jaccard(terms_a: set[str], terms_b: set[str]) -> float:
    """The Jaccard overlap of two term sets, 0 when both are empty.

    Args:
        terms_a: One set of terms.
        terms_b: The other.

    Returns:
        ``|a ∩ b| / |a ∪ b|``, or 0.0 when the union is empty.

    >>> jaccard({"fbn1", "aorta"}, {"fbn1", "lens"})
    0.3333333333333333
    >>> jaccard(set(), set())
    0.0
    """
    union = terms_a | terms_b
    if not union:
        return 0.0
    return len(terms_a & terms_b) / len(union)


def _unit(vec: Sequence[float]) -> list[float]:
    """The vector scaled to length 1, or the zero vector unchanged.

    Normalising once per vector turns each later cosine into a single dot
    product, instead of recomputing both norms for every pair.
    """
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else list(vec)


def candidate_pairs(
    claims_a: Sequence[Claim],
    claims_b: Sequence[Claim],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    embedder: Optional[Embedder] = None,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
) -> list[CandidatePair]:
    """Propose the claim pairs worth classifying, with the signal(s) behind each.

    Entity and citation signals are found through inverted indexes; lexical and
    embedding signals compare the pooled pairs directly. A pair is proposed if any
    signal fires, and carries all that did.

    Args:
        claims_a: Source A's claims.
        claims_b: Source B's claims.
        min_overlap: Lexical Jaccard at or above which a pair is proposed.
        embedder: Optional batch embedder; without it, no embedding signal.
        embed_threshold: Cosine at or above which the embedder proposes a pair.

    Returns:
        Candidate pairs, ordered by A-claim then B-claim position.

    A shared grounded entity proposes a pair; so does a lexical overlap:

    >>> from deep_research_client.claims import EntityMention
    >>> def claim(i, text, ent=None):
    ...     return Claim(id=i, claim_text=text, anchor_status="EXACT", citation_status="UNCITED",
    ...                  entities=[EntityMention(label="x", id=ent)] if ent else None)
    >>> a = [claim("a1", "FBN1 variants cause Marfan syndrome.", "HGNC:3603")]
    >>> b = [claim("b1", "Marfan syndrome is caused by FBN1 variants.", "HGNC:3603"),
    ...      claim("b2", "The sky is blue.")]
    >>> pairs = candidate_pairs(a, b)
    >>> [(p.claim_a.id, p.claim_b.id, [s.value for s in p.signal_list]) for p in pairs]
    [('a1', 'b1', ['ENTITY', 'LEXICAL'])]
    """
    return _candidate_pairs(
        claims_a, claims_b, accept=_all_pairs,
        min_overlap=min_overlap, embedder=embedder, embed_threshold=embed_threshold,
    )


def candidate_pairs_within(
    claims: Sequence[Claim],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    embedder: Optional[Embedder] = None,
    embed_threshold: float = DEFAULT_EMBED_THRESHOLD,
) -> list[CandidatePair]:
    """Propose the pairs worth classifying *within* one claim set.

    For comparing a source against itself: finding where a report repeats,
    refines, or contradicts its own claims. Each unordered pair of distinct
    claims is proposed at most once (``i`` before ``j``); a claim is never paired
    with itself. In a single-topic document lexical overlap is less
    discriminative than across sources, so a higher ``min_overlap`` is often
    wanted.

    Args:
        claims: The claims to compare against each other.
        min_overlap: Lexical Jaccard at or above which a pair is proposed.
        embedder: Optional batch embedder; without it, no embedding signal.
        embed_threshold: Cosine at or above which the embedder proposes a pair.

    Returns:
        Candidate pairs with ``claim_a`` earlier than ``claim_b`` in ``claims``.

    >>> from deep_research_client.claims import EntityMention
    >>> def claim(i, text, ent=None):
    ...     return Claim(id=i, claim_text=text, anchor_status="EXACT", citation_status="UNCITED",
    ...                  entities=[EntityMention(label="x", id=ent)] if ent else None)
    >>> claims = [claim("c1", "FBN1 variants cause Marfan syndrome.", "HGNC:3603"),
    ...           claim("c2", "Marfan syndrome is caused by FBN1 variants.", "HGNC:3603"),
    ...           claim("c3", "The sky is blue.")]
    >>> [(p.claim_a.id, p.claim_b.id) for p in candidate_pairs_within(claims)]
    [('c1', 'c2')]
    """
    return _candidate_pairs(
        claims, claims, accept=_lower_triangle,
        min_overlap=min_overlap, embedder=embedder, embed_threshold=embed_threshold,
    )


def _all_pairs(i: int, j: int) -> bool:
    """Accept every (i, j): the cross-set case."""
    return True


def _lower_triangle(i: int, j: int) -> bool:
    """Accept only i < j: unordered distinct pairs, the within-set case."""
    return i < j


def _candidate_pairs(
    claims_a: Sequence[Claim],
    claims_b: Sequence[Claim],
    *,
    accept: Callable[[int, int], bool],
    min_overlap: float,
    embedder: Optional[Embedder],
    embed_threshold: float,
) -> list[CandidatePair]:
    """Collect candidate pairs over claims_a × claims_b, keeping those ``accept``s.

    Shared by :func:`candidate_pairs` (every pair) and
    :func:`candidate_pairs_within` (``i < j`` of one list). When the two lists are
    the same object, the embedder is called once.
    """
    signals: dict[tuple[int, int], set[CandidateSignal]] = {}

    def add(i: int, j: int, signal: CandidateSignal) -> None:
        if accept(i, j):
            signals.setdefault((i, j), set()).add(signal)

    # Entity and citation: inverted index from id to the B claims that carry it.
    b_by_entity: dict[str, list[int]] = {}
    b_by_reference: dict[str, list[int]] = {}
    for j, claim_b in enumerate(claims_b):
        for curie in grounded_ids(claim_b):
            b_by_entity.setdefault(curie, []).append(j)
        for ref in resolved_reference_ids(claim_b):
            b_by_reference.setdefault(ref, []).append(j)
    for i, claim_a in enumerate(claims_a):
        for curie in grounded_ids(claim_a):
            for j in b_by_entity.get(curie, ()):
                add(i, j, CandidateSignal.ENTITY)
        for ref in resolved_reference_ids(claim_a):
            for j in b_by_reference.get(ref, ()):
                add(i, j, CandidateSignal.CITATION)

    # Lexical: Jaccard over content words, computed once per claim.
    terms_a = [set(content_terms(c.claim_text)) for c in claims_a]
    terms_b = terms_a if claims_b is claims_a else [set(content_terms(c.claim_text)) for c in claims_b]
    for i, ta in enumerate(terms_a):
        for j, tb in enumerate(terms_b):
            if jaccard(ta, tb) >= min_overlap:
                add(i, j, CandidateSignal.LEXICAL)

    # Embedding: optional, cosine over caller-supplied vectors. Vectors are
    # normalised once, so each pair's cosine is one dot product; within one set
    # the texts are embedded once.
    if embedder is not None and claims_a and claims_b:
        vecs_a = list(embedder([c.claim_text for c in claims_a]))
        vecs_b = vecs_a if claims_b is claims_a else list(embedder([c.claim_text for c in claims_b]))
        if len(vecs_a) != len(claims_a) or len(vecs_b) != len(claims_b):
            raise ValueError(
                f"embedder returned {len(vecs_a)} and {len(vecs_b)} vectors for "
                f"{len(claims_a)} and {len(claims_b)} claims; it must return one per text"
            )
        units_a = [_unit(v) for v in vecs_a]
        units_b = units_a if vecs_b is vecs_a else [_unit(v) for v in vecs_b]
        if len({len(v) for v in units_a} | {len(v) for v in units_b}) > 1:
            raise ValueError("embedder returned vectors of differing dimensions")
        for i, ua in enumerate(units_a):
            for j, ub in enumerate(units_b):
                if sum(x * y for x, y in zip(ua, ub)) >= embed_threshold:
                    add(i, j, CandidateSignal.EMBEDDING)

    return [
        CandidatePair(claims_a[i], claims_b[j], frozenset(fired))
        for (i, j), fired in sorted(signals.items())
    ]
