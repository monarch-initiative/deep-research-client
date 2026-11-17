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

#: Statuses that count as a citation having been checked and settled. A resolved
#: citation that reached none of these after verifiers ran is UNCHECKED.
_SETTLED = (VerificationStatus.SUPPORTED, VerificationStatus.CONTRADICTED)


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
    claim_set: ClaimSet, verifications: list[ClaimVerification]
) -> list[ReviewFlag]:
    """Flag cited, resolvable claims that verifiers ran over but never settled.

    Only meaningful once at least one verifier has run: it means a citation that
    could have been checked was not confirmed either way, so an agent should.
    """
    settled = {v.claim_id for v in verifications if v.status in _SETTLED}
    seen = {v.claim_id for v in verifications}
    flags: list[ReviewFlag] = []
    for claim in claim_set.claim_list:
        if (
            claim.citation_status == CitationStatus.CITED
            and resolved_reference_ids(claim)
            and claim.id not in settled
            and claim.id not in seen
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
        flags.extend(_unchecked_flags(claim_set, verifications))

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
