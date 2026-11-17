"""Flag the claims and pairs that need review, with no model and no network.

Triage is the rule-based half of verification (issue #43, item 5). It reads a
claim set -- and, when given one, an alignment -- and raises a
:class:`~deep_research_client.claims.ReviewFlag` wherever something should be
looked at: a domain claim supported by nothing, a citation that resolves to no
identifier, a contradiction, a point framed as settled consensus. It does not
*adjudicate* any of these; every flag carries a
:class:`~deep_research_client.claims.Decidability` saying who can -- a machine
rule, an agent with tools, or only a person. Harry's point stands in the data:
most of what matters here is a human's call.

Everything in this module is a pure function of a claim set and an alignment, so
it is tested directly, without a model. The verifier hooks that *do* call out to
the world (citation resolution, agents) live in ``verification.py`` and
``verification_citation.py`` and compose with these flags.
"""

import re
from typing import Optional, Sequence

from .alignment_models import ClaimAlignment
from .models import CitationStatus, Claim, ClaimBasis, ClaimSet, ClaimTopic, resolved_reference_ids
from .verification_models import Decidability, FlagKind, ReviewFlag

__all__ = ["CONSENSUS_CUES", "triage_claims"]

#: Phrases that frame a point as settled or universal. A claim that uses one while
#: citing nothing, or while taking part in a contradiction, is dressing a disputed
#: point as consensus -- the flag a reviewer most wants to see. Deliberately a
#: small, conservative lexical list: it is a heuristic, so its flags are always a
#: human's call (Decidability.HUMAN), and a caller may pass its own list.
CONSENSUS_CUES: tuple[str, ...] = (
    "well established",
    "well-established",
    "firmly established",
    "it is well known",
    "it is known that",
    "universally",
    "universally accepted",
    "undisputed",
    "indisputable",
    "beyond doubt",
    "no doubt that",
    "consensus",
    "all studies show",
    "all studies agree",
    "every study",
    "it is settled",
    "settled science",
    "unequivocally",
)

#: How much claim text to keep as a flag's evidence snippet.
_SNIPPET_LEN = 200

#: Words that negate a consensus cue when they precede it ("no consensus", "lack of
#: consensus", "little agreement"). A negated cue is the opposite of overstatement,
#: so it must not be flagged. Checked against the few words before the cue.
_NEGATORS = frozenset({
    "no", "not", "never", "little", "lack", "lacks", "lacking", "without",
    "absent", "absence", "nor", "neither",
})

#: How many whole words before a cue to scan for a negator.
_NEGATION_LOOKBACK = 3


def _snippet(text: str) -> str:
    """A short, single-line piece of claim text for a flag's evidence."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= _SNIPPET_LEN else collapsed[: _SNIPPET_LEN - 1] + "…"


def _is_domain(claim: Claim) -> bool:
    return claim.about == ClaimTopic.DOMAIN


def _matched_cue(text: str, cues: Sequence[str]) -> Optional[str]:
    """The first non-negated consensus cue in the text, or None.

    A plain substring match would flag "there is *no consensus*" as framing a
    point as settled -- the opposite of what the cue means. Each occurrence is
    kept only when no negator sits in the few words before it.
    """
    folded = text.casefold()
    for cue in cues:
        needle = cue.casefold()
        start = 0
        while True:
            at = folded.find(needle, start)
            if at == -1:
                break
            # Whole words before the cue (the full preceding text, so a window can't
            # cut a word mid-way and invent a spurious token).
            prior_words = re.findall(r"[a-z]+", folded[:at])[-_NEGATION_LOOKBACK:]
            if not _NEGATORS.intersection(prior_words):
                return cue
            start = at + 1
    return None


def _contradicted_claim_ids(alignment: ClaimAlignment, claim_set: ClaimSet) -> set[str]:
    """Ids *in this claim set* that take part in a contradictory pair.

    A pair's ``claim_a_id`` and ``claim_b_id`` are ids in two different claim sets,
    and both sets number claims ``c1``, ``c2``, ... With ``within_source`` the two
    sides are the same set, so both ids belong to it. Otherwise this set is source
    A, and only ``claim_a_id`` is one of its ids -- taking ``claim_b_id`` too would
    flag the wrong claim.
    """
    ids: set[str] = set()
    for pair in alignment.contradictions:
        ids.add(pair.claim_a_id)
        if alignment.within_source:
            ids.add(pair.claim_b_id)
    return ids


def triage_claims(
    claim_set: ClaimSet,
    alignment: Optional[ClaimAlignment] = None,
    *,
    cues: Sequence[str] = CONSENSUS_CUES,
) -> list[ReviewFlag]:
    """Raise a review flag wherever a claim set (and alignment) needs a look.

    The rules, each pure and each tagging who can settle what it raises:

    - **UNSUPPORTED_CLAIM** (AGENT): a domain claim given as a finding or an
      attributed result, citing nothing. Background-knowledge claims are expected
      to be uncited and are not flagged.
    - **UNRESOLVED_CITATION** (AGENT): a cited claim whose citations resolve to no
      identifier -- the marker points nowhere checkable.
    - **CONSENSUS_OVERSTATED** (HUMAN): a claim using a consensus/certainty cue
      while either citing nothing or taking part in a contradiction.
    - **CONTRADICTION** / **SELF_CONTRADICTION** (HUMAN): a contradictory pair,
      the latter when the alignment is within one source. Only when an alignment
      is given.

    Flags come in claim order (claim-level rules first, per claim), then the
    contradictions in pair order, so the output is deterministic.

    Args:
        claim_set: The claims to triage. It is the A side of the alignment, if one
            is given.
        alignment: An alignment of this set, for contradiction flags and to know
            which claims take part in one. Its ``source_a`` must be this claim
            set's source (its ``source_b`` too, for a within-source alignment),
            so the A-side ids line up with the claims here.
        cues: Consensus/certainty phrases for CONSENSUS_OVERSTATED. Defaults to
            :data:`CONSENSUS_CUES`.

    Returns:
        The review flags, most-local first (claim-level), then contradictions.

    Raises:
        ValueError: If the alignment's A side is not this claim set's source.

    Examples:
        >>> from deep_research_client.claims import (AnchorStatus, CitationStatus,
        ...     Claim, ClaimBasis, ClaimSet, ClaimTopic, ExtractorInfo,
        ...     SourceDocument, SourceType)
        >>> def claim(i, text, about=ClaimTopic.DOMAIN, basis=ClaimBasis.OBSERVATION,
        ...           cited=CitationStatus.UNCITED):
        ...     return Claim(id=i, claim_text=text, anchor_status=AnchorStatus.EXACT,
        ...                  about=about, basis=basis, citation_status=cited)
        >>> cs = ClaimSet(
        ...     source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        ...     extractor=ExtractorInfo(name="x"),
        ...     claims=[claim("c1", "TLR2 drives inflammation in acne."),
        ...             claim("c2", "It is well known that acne is multifactorial.")])
        >>> flags = triage_claims(cs)
        >>> [(f.kind, f.decidability, f.claim_ids) for f in flags]
        ... # doctest: +NORMALIZE_WHITESPACE
        [('UNSUPPORTED_CLAIM', 'AGENT', ['c1']),
         ('UNSUPPORTED_CLAIM', 'AGENT', ['c2']),
         ('CONSENSUS_OVERSTATED', 'HUMAN', ['c2'])]
    """
    if alignment is not None:
        if alignment.source_a.id != claim_set.source.id:
            raise ValueError(
                "the alignment does not match this claim set: it aligns "
                f"{alignment.source_a.id!r} (A), but the claim set is "
                f"{claim_set.source.id!r}. Pass the alignment whose A side is this set."
            )
        if alignment.within_source and alignment.source_b.id != claim_set.source.id:
            raise ValueError(
                "the alignment is marked within_source but its B side "
                f"{alignment.source_b.id!r} is not this claim set {claim_set.source.id!r}."
            )
    contradicted = (
        _contradicted_claim_ids(alignment, claim_set) if alignment is not None else set()
    )
    flags: list[ReviewFlag] = []

    for claim in claim_set.claim_list:
        text = claim.claim_text or ""

        if (
            _is_domain(claim)
            and claim.citation_status == CitationStatus.UNCITED
            and claim.basis in (ClaimBasis.OBSERVATION, ClaimBasis.SECONDARY_SOURCE)
        ):
            flags.append(ReviewFlag(
                kind=FlagKind.UNSUPPORTED_CLAIM,
                decidability=Decidability.AGENT,
                claim_ids=[claim.id],
                message=f"{claim.id} asserts a domain finding but cites nothing to support it.",
                evidence=_snippet(text),
            ))

        if claim.citation_status == CitationStatus.CITED and not resolved_reference_ids(claim):
            flags.append(ReviewFlag(
                kind=FlagKind.UNRESOLVED_CITATION,
                decidability=Decidability.AGENT,
                claim_ids=[claim.id],
                message=(
                    f"{claim.id} is cited, but none of its citations resolve to an "
                    "identifier that can be checked."
                ),
                evidence=_snippet(text),
            ))

        cue = _matched_cue(text, cues)
        if cue is not None and (
            claim.citation_status != CitationStatus.CITED or claim.id in contradicted
        ):
            why = (
                "takes part in a contradiction"
                if claim.id in contradicted
                else "cites nothing"
            )
            flags.append(ReviewFlag(
                kind=FlagKind.CONSENSUS_OVERSTATED,
                decidability=Decidability.HUMAN,
                claim_ids=[claim.id],
                message=(
                    f'{claim.id} frames a point as settled ("{cue}") yet {why}; '
                    "check whether the consensus is real."
                ),
                evidence=_snippet(text),
            ))

    if alignment is not None:
        within = alignment.within_source
        kind = FlagKind.SELF_CONTRADICTION if within else FlagKind.CONTRADICTION
        other = alignment.source_b.id
        for pair in alignment.contradictions:
            # Within one source both ids are in this set. Across sources only the
            # A-side id is; name the B-side claim by its source so flags_for cannot
            # collide it with an A-side claim of the same id (both sets use c1, c2...).
            if within:
                claim_ids = [pair.claim_a_id, pair.claim_b_id]
                message = (
                    f"{pair.claim_a_id} and {pair.claim_b_id} cannot both be true "
                    "(the source contradicts itself); a person must judge which, if "
                    "either, holds."
                )
            else:
                claim_ids = [pair.claim_a_id]
                message = (
                    f"{pair.claim_a_id} cannot both be true with {pair.claim_b_id} in "
                    f"{other} (two sources disagree); a person must judge which, if "
                    "either, holds."
                )
            flags.append(ReviewFlag(
                kind=kind,
                decidability=Decidability.HUMAN,
                claim_ids=claim_ids,
                pair_id=pair.id,
                message=message,
                evidence=_snippet(f"{pair.claim_a_text}  ≠  {pair.claim_b_text}"),
            ))

    return flags
