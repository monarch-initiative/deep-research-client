"""Ground a claim set's entities to ontology CURIEs (issue #43, item 3).

Extraction records an entity's label as written; grounding fills its
``id`` with an ontology CURIE, so claims become comparable on shared terms and
alignment's entity signal (``candidates``) can fire. Grounding is a separate,
opt-in step: nothing here runs unless a caller asks for it.

The work splits in two. Deciding a CURIE for one label is a backend's job (the
:class:`Grounder` protocol; the OAK-plus-model backend is in ``grounding_oak``).
Everything else -- finding the labels, ranking them so the most frequent are
grounded first (ontology search is the slow step), grounding each distinct label
once, and propagating the result to every mention -- is a pure function of the
claim set and the grounder, and is tested with an injected grounder rather than a
live ontology service.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Protocol

from ..ontologies import OntologyConfig
from .models import Claim, ClaimSet, EntityMention, GrounderInfo

__all__ = [
    "GroundedTerm",
    "GroundingCandidate",
    "Grounder",
    "aground_claims",
    "biomedical_ontology_config",
    "ground_claims",
    "normalized_label",
    "rank_labels",
]

#: The default built-in ontology set: common biomedical concept types that OLS
#: serves. Genes and proteins (HGNC, UniProt, PR) are deliberately not here;
#: supply them with an OntologyConfig when a corpus needs them.
_BIOMEDICAL_PREFIXES = ("MONDO", "HP", "GO", "CHEBI", "CL", "UBERON")


def biomedical_ontology_config() -> OntologyConfig:
    """A ready-made config over common biomedical ontologies, via the default adapter.

    Returns:
        An OntologyConfig naming MONDO, HP, GO, CHEBI, CL and UBERON, each with no
        adapter so the default (``ols:``) is used.

    >>> biomedical_ontology_config().prefixes
    ['MONDO', 'HP', 'GO', 'CHEBI', 'CL', 'UBERON']
    """
    return OntologyConfig(ontology_adapters={prefix: None for prefix in _BIOMEDICAL_PREFIXES})


@dataclass(frozen=True)
class GroundingCandidate:
    """One ontology term proposed for a label, for a backend to choose among.

    Attributes:
        id: The term's CURIE, for example ``MONDO:0007947``.
        label: The term's own label in the ontology.
        definition: The term's definition, when the ontology has one.
    """

    id: str
    label: str
    definition: Optional[str] = None


@dataclass(frozen=True)
class GroundedTerm:
    """The CURIE a label was grounded to.

    Attributes:
        id: The chosen CURIE.
        label: The ontology's own label for it (not the label as written).
    """

    id: str
    label: str


class Grounder(Protocol):
    """Grounds one entity label to an ontology term, or declines.

    A backend implements this; the orchestration here is independent of how a
    CURIE is decided, so a fake grounder drives the tests.
    """

    async def aground_term(self, label: str, context: str) -> Optional[GroundedTerm]:
        """Ground one label, given a sentence it appears in, or return None."""
        ...

    def provenance(self, considered: int, grounded: int) -> GrounderInfo:
        """A record of how this grounder ran, for the claim set's ``grounder`` slot."""
        ...


def normalized_label(label: str) -> str:
    """A label reduced to a key for deduplicating mentions.

    Case and surrounding or repeated whitespace do not distinguish one mention of
    a term from another.

    >>> normalized_label("  Marfan   syndrome ")
    'marfan syndrome'
    >>> normalized_label("FBN1") == normalized_label("fbn1")
    True
    """
    return " ".join(label.split()).casefold()


def _mentions(claim: Claim) -> list[EntityMention]:
    """Every entity mention on a claim: its entities, subject and object."""
    mentions = list(claim.entities or [])
    mentions += [m for m in (claim.subject, claim.object) if m is not None]
    return mentions


@dataclass(frozen=True)
class _RankedLabel:
    """A distinct ungrounded label, with a context sentence and how often it occurs."""

    label: str
    context: str
    count: int


def rank_labels(claim_set: ClaimSet) -> list[_RankedLabel]:
    """The distinct ungrounded entity labels, most frequent first.

    A mention that already carries an id (the source grounded it) is left alone.
    Labels are deduplicated by :func:`normalized_label`; the first written form and
    the first claim it appears in are kept, and ties break by label for a stable
    order.

    Args:
        claim_set: The claims to read.

    Returns:
        One entry per distinct ungrounded label, ordered by descending count then
        label.

    >>> from deep_research_client.claims import (Claim, ClaimSet, EntityMention,
    ...     ExtractorInfo, SourceDocument)
    >>> def claim(i, *labels):
    ...     return Claim(id=i, claim_text=f"Sentence {i}.", anchor_status="EXACT",
    ...                  citation_status="UNCITED",
    ...                  entities=[EntityMention(label=l) for l in labels])
    >>> cs = ClaimSet(source=SourceDocument(id="r.md", source_type="MARKDOWN_REPORT"),
    ...               extractor=ExtractorInfo(name="x"),
    ...               claims=[claim("c1", "acne", "TLR2"), claim("c2", "Acne")])
    >>> [(r.label, r.count) for r in rank_labels(cs)]
    [('acne', 2), ('TLR2', 1)]
    """
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    context: dict[str, str] = {}
    for claim in claim_set.claim_list:
        for mention in _mentions(claim):
            if mention.id:
                continue
            key = normalized_label(mention.label)
            counts[key] = counts.get(key, 0) + 1
            display.setdefault(key, mention.label)
            context.setdefault(key, claim.claim_text)
    ranked = [_RankedLabel(display[k], context[k], counts[k]) for k in counts]
    ranked.sort(key=lambda r: (-r.count, r.label))
    return ranked


def _apply_grounding(claim: Claim, grounded: dict[str, GroundedTerm]) -> Claim:
    """A copy of the claim with ids filled on mentions whose label was grounded."""

    def ground(mention: Optional[EntityMention]) -> Optional[EntityMention]:
        if mention is None or mention.id:
            return mention
        term = grounded.get(normalized_label(mention.label))
        return mention.model_copy(update={"id": term.id}) if term else mention

    update: dict[str, object] = {}
    if claim.entities:
        update["entities"] = [ground(m) for m in claim.entities]
    if claim.subject is not None:
        update["subject"] = ground(claim.subject)
    if claim.object is not None:
        update["object"] = ground(claim.object)
    return claim.model_copy(update=update) if update else claim


async def aground_claims(
    claim_set: ClaimSet,
    grounder: Grounder,
    *,
    max_terms: Optional[int] = None,
    concurrency: int = 4,
) -> ClaimSet:
    """Ground a claim set's entities to ontology CURIEs.

    Distinct ungrounded labels are grounded most-frequent-first (ontology search
    is the slow step), each once, and the result is propagated to every mention of
    that label. ``max_terms`` grounds only the top labels, bounding the work.

    Args:
        claim_set: The claims to ground.
        grounder: The backend that decides a CURIE for one label.
        max_terms: Ground at most this many distinct labels (the most frequent),
            or all of them when None.
        concurrency: Labels grounded in flight at once.

    Returns:
        A new claim set with entity ids filled where grounding succeeded and a
        ``grounder`` provenance record set.
    """
    ranked = rank_labels(claim_set)
    if max_terms is not None:
        ranked = ranked[:max_terms]

    limit = asyncio.Semaphore(concurrency)

    async def one(entry: _RankedLabel) -> tuple[str, Optional[GroundedTerm]]:
        async with limit:
            return entry.label, await grounder.aground_term(entry.label, entry.context)

    tasks = [asyncio.ensure_future(one(entry)) for entry in ranked]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    grounded = {normalized_label(label): term for label, term in results if term is not None}
    new_claims = [_apply_grounding(claim, grounded) for claim in claim_set.claim_list]
    return claim_set.model_copy(update={
        "claims": new_claims,
        "grounder": grounder.provenance(considered=len(ranked), grounded=len(grounded)),
    })


def ground_claims(
    claim_set: ClaimSet,
    grounder: Grounder,
    *,
    max_terms: Optional[int] = None,
    concurrency: int = 4,
) -> ClaimSet:
    """Synchronous :func:`aground_claims`, for callers without an event loop.

    It runs its own loop with :func:`asyncio.run`, which raises ``RuntimeError``
    inside one already running, as in Jupyter. There, ``await aground_claims(...)``.
    """
    return asyncio.run(aground_claims(
        claim_set, grounder, max_terms=max_terms, concurrency=concurrency,
    ))
