"""Verify a claim set and report what needs review (issue #43, item 5).

Verification has two halves that compose. Triage (``triage.py``) is pure: it reads
the claims and an optional alignment and raises review flags, each tagged with who
can settle it. Verifiers reach out to the world -- resolving a cited work,
searching for support, asking an agent -- and return a verdict per claim. This
module runs both and assembles a :class:`VerificationReport`.

A :class:`Verifier` is the hook (issue #43 item 5 asks for exactly these places to
attach): citation verification is the backend shipped here
(``verification_citation.py``); retrieval-based checking, identifier resolution and
human adjudication are others a downstream tool can add behind the same protocol.
The orchestration does not know which backend it drives, so a fake verifier drives
the tests rather than a live service.

With no verifiers, verification is triage-only: pure, no network, no model. That is
the default, and it is what makes the human-review surface usable on its own.
"""

import asyncio
from typing import Optional, Protocol, Sequence

from .alignment_models import ClaimAlignment
from .models import CitationStatus, Claim, ClaimSet, resolved_reference_ids
from .triage import triage_claims
from .verification_models import (
    ClaimVerification,
    Decidability,
    FlagKind,
    ReviewFlag,
    VerificationReport,
    VerificationStatus,
    VerifiedSourceType,
    VerifierInfo,
)

__all__ = [
    "Verifier",
    "averify_claims",
    "triage_claims",
    "verify_claims",
]

def _flag_for(verdict: ClaimVerification) -> Optional[tuple[FlagKind, Decidability]]:
    """The review flag a verdict warrants, or None when it needs no review.

    The citation-specific kinds are keyed on a ``citation:`` method, not on status
    alone: a future retrieval verifier returning UNSUPPORTED for an *uncited* claim
    means "found no support", not "the cited identifier is fabricated", so it must
    not be labelled CONFABULATED_CITATION. A verifier of a new kind needs its own
    case here rather than silently borrowing a citation flag.
    """
    status = VerificationStatus(verdict.status)
    is_citation = (verdict.method or "").startswith("citation:")
    if is_citation and status == VerificationStatus.UNSUPPORTED:
        return FlagKind.CONFABULATED_CITATION, Decidability.MACHINE
    if is_citation and status == VerificationStatus.NEEDS_REVIEW:
        return FlagKind.WEAK_CITATION, Decidability.HUMAN
    if status == VerificationStatus.CONTRADICTED:
        return FlagKind.CONTRADICTION, Decidability.HUMAN
    return None


def _flags_from_verdicts(verifications: list[ClaimVerification]) -> list[ReviewFlag]:
    """Turn notable verifier verdicts into review flags.

    A verdict and a flag are different products: the verdict records what a
    verifier found, the flag routes it to whoever can act on it. Mapped in one
    pure place (:func:`_flag_for`) so a new verifier of a known kind surfaces
    without new flag code.
    """
    flags: list[ReviewFlag] = []
    for v in verifications:
        mapped = _flag_for(v)
        if mapped is None:
            continue
        kind, decidability = mapped
        flags.append(ReviewFlag(
            kind=kind,
            decidability=decidability,
            claim_ids=[v.claim_id],
            message=v.note or f"{v.claim_id}: {v.status} ({v.method}).",
            evidence=v.reference_id,
        ))
    return flags


class Verifier(Protocol):
    """Checks one claim against the world and returns verdicts, or none.

    A backend implements this; the orchestration here is independent of how a
    claim is checked, so a fake verifier drives the tests. A verifier returns an
    empty list for a claim it has nothing to say about (for example a citation
    verifier given an uncited claim).
    """

    name: str

    async def averify(self, claim: Claim) -> list[ClaimVerification]:
        """Verdicts for one claim, or an empty list when the verifier does not apply."""
        ...


def _source_type(claim_set: ClaimSet) -> Optional[VerifiedSourceType]:
    """The claim set's source type as a VerifiedSourceType, or None if absent.

    VerifiedSourceType mirrors claims.SourceType by value (a drift test checks it),
    so the stored string maps straight across.
    """
    raw = claim_set.source.source_type
    return VerifiedSourceType(raw) if raw else None


def _unchecked_flags(
    claim_set: ClaimSet,
    verifications: list[ClaimVerification],
    verdict_flags: list[ReviewFlag],
) -> list[ReviewFlag]:
    """Flag cited, resolvable claims a verifier neither supported nor faulted.

    Only meaningful once at least one verifier has run. A claim is *addressed* when
    a verifier returned SUPPORTED for it, or raised a flag about it (fabricated,
    weak, contradicted). A cited, resolvable claim that is not addressed stands
    unchecked -- whether because no verifier reached it, or because every verdict
    came back UNVERIFIABLE (no resolver, nothing learned). An agent could still
    settle it, so it is flagged for one. A claim citing several works is addressed
    as soon as one is SUPPORTED: one supporting work is enough that the claim is not
    unsupported, so a second work that came back UNVERIFIABLE raises no flag.
    """
    addressed = {
        v.claim_id for v in verifications
        if VerificationStatus(v.status) == VerificationStatus.SUPPORTED
    }
    for flag in verdict_flags:
        addressed.update(flag.claim_ids or [])
    flags: list[ReviewFlag] = []
    for claim in claim_set.claim_list:
        if (
            claim.citation_status == CitationStatus.CITED
            and resolved_reference_ids(claim)
            and claim.id not in addressed
        ):
            flags.append(ReviewFlag(
                kind=FlagKind.UNCHECKED_CITATION,
                decidability=Decidability.AGENT,
                claim_ids=[claim.id],
                message=(
                    f"{claim.id} cites a resolvable work, but no verifier confirmed it "
                    "supports the claim."
                ),
            ))
    return flags


async def averify_claims(
    claim_set: ClaimSet,
    verifiers: Sequence[Verifier] = (),
    *,
    alignment: Optional[ClaimAlignment] = None,
    cues: Optional[Sequence[str]] = None,
    max_claims: Optional[int] = None,
    concurrency: int = 4,
) -> VerificationReport:
    """Triage a claim set and run any verifiers over it, into one report.

    Triage always runs (pure, no network). Each verifier is then run over the
    claims -- all of them, so a future retrieval verifier can reach uncited claims
    too; a verifier returns nothing for a claim it does not apply to. A cited,
    resolvable claim that verifiers ran over but left unsettled is flagged
    UNCHECKED_CITATION.

    Args:
        claim_set: The claims to verify.
        verifiers: Verifier backends to run. Empty means triage-only.
        alignment: An alignment of the set, for contradiction flags.
        cues: Consensus/certainty phrases for triage; its default when None.
        max_claims: Run verifiers over at most this many claims (the first in
            order), bounding verifier work. Triage still sees every claim.
        concurrency: Verifier calls in flight at once.

    Returns:
        A :class:`VerificationReport` with the triage flags, any verifier verdicts,
        and a VerifierInfo provenance stamp.
    """
    flags = triage_claims(claim_set, alignment, **({"cues": cues} if cues is not None else {}))

    verifications: list[ClaimVerification] = []
    if verifiers:
        claims = claim_set.claim_list
        if max_claims is not None:
            claims = claims[:max_claims]
        limit = asyncio.Semaphore(concurrency)

        async def one(verifier: Verifier, claim: Claim) -> list[ClaimVerification]:
            async with limit:
                return await verifier.averify(claim)

        tasks = [
            asyncio.ensure_future(one(verifier, claim))
            for claim in claims
            for verifier in verifiers
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for verdicts in results:
            verifications.extend(verdicts)
        verdict_flags = _flags_from_verdicts(verifications)
        flags.extend(verdict_flags)
        flags.extend(_unchecked_flags(claim_set, verifications, verdict_flags))

    names = [v.name for v in verifiers]
    verified_claims = {v.claim_id for v in verifications}
    verifier_info = VerifierInfo(
        name="+".join(["triage", *names]),
        methods=["triage", *names],
        considered=len(claim_set.claim_list),
        flagged=len(flags),
        verified=len(verified_claims),
    )
    return VerificationReport(
        source_id=claim_set.source.id,
        source_type=_source_type(claim_set),
        verifier=verifier_info,
        flags=flags or None,
        verifications=verifications or None,
    )


def verify_claims(
    claim_set: ClaimSet,
    verifiers: Sequence[Verifier] = (),
    *,
    alignment: Optional[ClaimAlignment] = None,
    cues: Optional[Sequence[str]] = None,
    max_claims: Optional[int] = None,
    concurrency: int = 4,
) -> VerificationReport:
    """Synchronous :func:`averify_claims`, for callers without an event loop.

    With no verifiers this is triage-only and does no I/O. It runs its own loop
    with :func:`asyncio.run`, which raises ``RuntimeError`` inside one already
    running, as in Jupyter; there, ``await averify_claims(...)``.

    >>> from deep_research_client.claims import (AnchorStatus, CitationStatus, Claim,
    ...     ClaimBasis, ClaimSet, ClaimTopic, ExtractorInfo, SourceDocument, SourceType)
    >>> cs = ClaimSet(
    ...     source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
    ...     extractor=ExtractorInfo(name="x"),
    ...     claims=[Claim(id="c1", claim_text="X drives Y.", anchor_status=AnchorStatus.EXACT,
    ...                   about=ClaimTopic.DOMAIN, basis=ClaimBasis.OBSERVATION,
    ...                   citation_status=CitationStatus.UNCITED)])
    >>> report = verify_claims(cs)  # triage-only, no network
    >>> report.verifier.name
    'triage'
    >>> [f.kind for f in report.flags_for("c1")]
    ['UNSUPPORTED_CLAIM']
    """
    return asyncio.run(averify_claims(
        claim_set, verifiers, alignment=alignment, cues=cues,
        max_claims=max_claims, concurrency=concurrency,
    ))
