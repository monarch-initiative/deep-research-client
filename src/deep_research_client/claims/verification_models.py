"""Verification models: the generated datamodel plus derived views.

The fields come from ``verification.yaml`` via ``verification_datamodel.py``.
``VerificationReport`` is subclassed here to add properties computed from those
fields; it adds no fields of its own, so its JSON schema stays exactly what
LinkML would emit.

The generated base sets ``use_enum_values=True``, so an enum slot reads back as
its string value; compare with ``==`` against the enum member, which is a ``str``
subclass.
"""

from typing import Optional

from .verification_datamodel import (
    ClaimVerification,
    Decidability,
    FlagKind,
    ReviewFlag,
    VerificationStatus,
    VerifiedSourceType,
    VerifierInfo,
)
from .verification_datamodel import VerificationReport as GeneratedVerificationReport

__all__ = [
    "ClaimVerification",
    "Decidability",
    "FlagKind",
    "ReviewFlag",
    "VerificationReport",
    "VerificationStatus",
    "VerifiedSourceType",
    "VerifierInfo",
]


class VerificationReport(GeneratedVerificationReport):
    """A verification report, with its flags and verdicts read through views.

    >>> report = VerificationReport(
    ...     source_id="r.md", source_type=VerifiedSourceType.MARKDOWN_REPORT,
    ...     verifier=VerifierInfo(name="triage", flagged=2),
    ...     flags=[
    ...         ReviewFlag(kind=FlagKind.UNSUPPORTED_CLAIM, decidability=Decidability.AGENT,
    ...                    claim_ids=["c1"], message="c1 cites nothing"),
    ...         ReviewFlag(kind=FlagKind.CONTRADICTION, decidability=Decidability.HUMAN,
    ...                    claim_ids=["c2", "c3"], pair_id="p1", message="c2 vs c3"),
    ...     ])
    >>> [f.kind for f in report.needs_human]
    ['CONTRADICTION']
    >>> [f.kind for f in report.flags_for("c1")]
    ['UNSUPPORTED_CLAIM']
    >>> report.flag_count
    2
    """

    @property
    def flag_list(self) -> list[ReviewFlag]:
        """Every flag, as a list even when the slot is unset."""
        return self.flags or []

    @property
    def verification_list(self) -> list[ClaimVerification]:
        """Every verdict, as a list even when the slot is unset."""
        return self.verifications or []

    @property
    def flag_count(self) -> int:
        """How many review flags the report holds."""
        return len(self.flag_list)

    @property
    def needs_human(self) -> list[ReviewFlag]:
        """Flags only a person can settle (decidability HUMAN)."""
        return [f for f in self.flag_list if f.decidability == Decidability.HUMAN]

    @property
    def needs_agent(self) -> list[ReviewFlag]:
        """Flags an automated agent with tools could settle (decidability AGENT)."""
        return [f for f in self.flag_list if f.decidability == Decidability.AGENT]

    def flags_for(self, claim_id: str) -> list[ReviewFlag]:
        """The flags that name a given claim.

        Args:
            claim_id: A claim id within the verified source's ClaimSet.

        Returns:
            Every flag whose ``claim_ids`` includes it, in report order.
        """
        return [f for f in self.flag_list if claim_id in (f.claim_ids or [])]

    def flags_of_kind(self, kind: FlagKind) -> list[ReviewFlag]:
        """The flags of a given kind, in report order.

        Args:
            kind: The :class:`FlagKind` to select.

        Returns:
            Every flag of that kind.
        """
        return [f for f in self.flag_list if f.kind == kind]

    def verification_for(self, claim_id: str) -> Optional[ClaimVerification]:
        """The first verdict recorded for a claim, if any.

        A claim citing several works has one verdict per work; this returns the
        first. Use :meth:`verifications_for` to see them all — the first being
        SUPPORTED does not mean a later one is not UNSUPPORTED.

        Args:
            claim_id: A claim id within the verified source's ClaimSet.

        Returns:
            Its first :class:`ClaimVerification`, or None when none was recorded.
        """
        for v in self.verification_list:
            if v.claim_id == claim_id:
                return v
        return None

    def verifications_for(self, claim_id: str) -> list[ClaimVerification]:
        """Every verdict recorded for a claim, in report order.

        Args:
            claim_id: A claim id within the verified source's ClaimSet.

        Returns:
            Its :class:`ClaimVerification` records — one per cited work checked.
        """
        return [v for v in self.verification_list if v.claim_id == claim_id]
