"""The citation-checking verifier: does a claim's cited work actually exist?

For one claim, this resolves each identifier the claim cites against the engine
in ``validation/validator.py`` (the ``validation`` extra, already in the repo)
and returns a verdict per identifier. It implements the
:class:`~deep_research_client.claims.Verifier` protocol, so the orchestration in
``verification.py`` runs it without knowing any of this.

The verdict is honest about its reach. A resolver settles whether an identifier
points to a real record, which catches the plausible-looking PMID or DOI that
providers confabulate -- a machine-decidable finding. Topical relevance (a
keyword-overlap heuristic) weighs whether the resolved work is even about the
claim. Neither is a semantic judgement that the work *supports* the specific
claim; that needs a model reading both, and is the agent hook left for a
following PR. So SUPPORTED here means "the cited work resolves, and is on topic",
not "the work entails the claim".

``linkml-reference-validator`` is the optional ``validation`` extra; it is
imported lazily, so importing this module costs nothing until a verifier is
built and run. Resolution is blocking and networked, so each call runs in a
worker thread, and -- because the validator shares a cache directory and is not
assumed thread-safe -- one call runs at a time.
"""

import asyncio
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

from ..validation.datamodel import ReferenceCheck, ReferenceStatus, TopicalRelevance
from ..validation.extraction import FoundReference
from .models import Claim, resolved_reference_ids
from .verification_models import ClaimVerification, VerificationStatus

__all__ = ["CitationVerifier", "verdict_from_check"]

#: Confidence that a VERIFIED, on-topic citation supports the claim -- high for
#: the resolution-plus-relevance check this makes, short of the 1.0 that only a
#: semantic reading of the work could earn.
_ON_TOPIC_CONFIDENCE = 0.8
#: Confidence for a VERIFIED citation whose topicality is uncertain or unassessed:
#: the work resolves, but nothing weighs on whether it is the right one.
_RESOLVED_ONLY_CONFIDENCE = 0.6
#: Confidence that a NOT_FOUND identifier is genuinely fabricated. Not 1.0: a
#: resolver can miss a real but obscure record.
_NOT_FOUND_CONFIDENCE = 0.9


def verdict_from_check(check: ReferenceCheck, claim_id: str) -> ClaimVerification:
    """Map one resolved-reference result to a claim verdict (pure).

    Args:
        check: A reference check from ``ReferenceValidator.validate_references``.
        claim_id: The claim the citation belongs to.

    Returns:
        A :class:`ClaimVerification`:

        - NOT_FOUND -> UNSUPPORTED (the identifier resolves to no record; the
          orchestration turns this into a machine-decidable CONFABULATED_CITATION
          flag).
        - UNVERIFIABLE -> UNVERIFIABLE (skipped, or no resolver for its prefix).
        - VERIFIED and off topic -> NEEDS_REVIEW (real work, but it looks like it
          is about something else).
        - VERIFIED otherwise -> SUPPORTED (resolves, and is on topic or topicality
          was not weighed).

    Examples:
        >>> verdict_from_check(ReferenceCheck(reference_id="PMID:999999999",
        ...     status=ReferenceStatus.NOT_FOUND), "c1").status
        'UNSUPPORTED'
        >>> verdict_from_check(ReferenceCheck(reference_id="PMID:1",
        ...     status=ReferenceStatus.VERIFIED,
        ...     relevance=TopicalRelevance.ON_TOPIC), "c1").status
        'SUPPORTED'
    """
    status = ReferenceStatus(check.status)
    if status == ReferenceStatus.NOT_FOUND:
        return ClaimVerification(
            claim_id=claim_id, status=VerificationStatus.UNSUPPORTED,
            method="citation:resolution", reference_id=check.reference_id,
            confidence=_NOT_FOUND_CONFIDENCE,
            note=f"{check.reference_id} resolves to no record; the citation looks fabricated.",
            verifier="citation",
        )
    if status == ReferenceStatus.UNVERIFIABLE:
        return ClaimVerification(
            claim_id=claim_id, status=VerificationStatus.UNVERIFIABLE,
            method="citation:resolution", reference_id=check.reference_id,
            note=check.message or f"{check.reference_id} could not be resolved either way.",
            verifier="citation",
        )

    # VERIFIED: the work exists. Relevance decides whether it is the right one.
    if TopicalRelevance(check.relevance) == TopicalRelevance.OFF_TOPIC:
        return ClaimVerification(
            claim_id=claim_id, status=VerificationStatus.NEEDS_REVIEW,
            method="citation:relevance", reference_id=check.reference_id,
            confidence=check.relevance_score,
            note=f"{check.reference_id} resolves but looks off topic for this claim.",
            verifier="citation",
        )
    on_topic = TopicalRelevance(check.relevance) == TopicalRelevance.ON_TOPIC
    return ClaimVerification(
        claim_id=claim_id, status=VerificationStatus.SUPPORTED,
        method="citation:resolution", reference_id=check.reference_id,
        confidence=_ON_TOPIC_CONFIDENCE if on_topic else _RESOLVED_ONLY_CONFIDENCE,
        note=(
            f"{check.reference_id} resolves"
            + (" and is on topic." if on_topic else "; topicality not weighed.")
        ),
        verifier="citation",
    )


class CitationVerifier:
    """Verify a claim's citations by resolving them with ``ReferenceValidator``.

    Args:
        topic_text: Vocabulary to weigh each resolved work's relevance against,
            usually the whole report's text. Without it, relevance is not assessed
            and a resolved citation is SUPPORTED on existence alone.
        email: Contact address for the NCBI Entrez API, which asks for one to
            raise the rate limit. None by default; a caller may set it. Never set
            from the user's own address automatically.
        cache_dir: Where resolved references are cached between runs.
        max_references: Cap on identifiers resolved per claim, for safety.
        skip_prefixes: Identifier prefixes to report UNVERIFIABLE instead of
            resolving.
        check_relevance: Whether to weigh topical relevance (no extra requests).
    """

    name = "citation"

    def __init__(
        self,
        *,
        topic_text: Optional[str] = None,
        email: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        max_references: Optional[int] = None,
        skip_prefixes: Sequence[str] = (),
        check_relevance: bool = True,
    ) -> None:
        self._topic_text = topic_text
        self._email = email
        self._cache_dir = cache_dir
        self._max_references = max_references
        self._skip_prefixes = list(skip_prefixes)
        self._check_relevance = check_relevance
        self._validator: Any = None
        self._lock = threading.Lock()

    def _build_validator(self) -> Any:
        """The reference validator, built once. Raises with a hint if the extra is absent."""
        from ..validation.validator import (  # lazy: the optional `validation` extra
            INSTALL_HINT,
            ReferenceValidator,
            validator_is_available,
        )

        if not validator_is_available():
            raise ImportError(INSTALL_HINT)
        return ReferenceValidator(
            email=self._email,
            cache_dir=self._cache_dir,
            max_references=self._max_references,
            skip_prefixes=self._skip_prefixes,
            check_relevance=self._check_relevance,
        )

    def _resolve(self, found: list[FoundReference]) -> list[ReferenceCheck]:
        """Resolve a claim's references; callers run this in a worker thread.

        Taken under a lock because the validator shares a cache directory and is
        not assumed thread-safe, and the lazy build happens once. The lock is held
        inside the worker thread only, never across an ``await``.
        """
        with self._lock:
            if self._validator is None:
                self._validator = self._build_validator()
            report = self._validator.validate_references(found, topic_text=self._topic_text)
            return list(report.references)

    async def averify(self, claim: Claim) -> list[ClaimVerification]:
        """Resolve each identifier the claim cites and return a verdict per identifier."""
        refs = sorted(resolved_reference_ids(claim))
        if not refs:
            return []
        found = [FoundReference(normalized_id=r, raw=r) for r in refs]
        checks = await asyncio.to_thread(self._resolve, found)
        return [verdict_from_check(check, claim.id) for check in checks]
