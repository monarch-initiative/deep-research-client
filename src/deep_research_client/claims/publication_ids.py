"""Normalise a raw publication citation to a canonical identifier (issue #41).

Providers emit the same paper a dozen ways: ``PMID:7913883``, a bare PubMed URL,
``doi:10.1038/x``, a numbered footnote, a loose "Smith et al. 2020". Downstream
evaluation gets noisy when formatting differences look like evidence differences,
so this is the shared, deterministic normalisation layer the issue asks for.

:func:`normalize_publication_id` takes one raw citation string and returns a
:class:`NormalizedPublicationId`: a canonical id when one can be read, its kind,
a confidence, and the source URL when a link gave it. An unresolved string is
**kept, not dropped**, with ``canonical_id=None`` and ``confidence=0.0``, so a
caller can preserve and later resolve it.

This is the deterministic core of #41. It wraps
:func:`deep_research_client.validation.extraction.find_reference_ids` (standard
library, no network) for PMID/DOI/PMCID/GEO, and adds clinical-trial (NCT) and
bare-URL handling. Online resolution of a paper's title, authors and year from an
identifier -- the rest of #41 -- is left as a hook for a following PR. The
``canonical_id`` here is exactly the form
:attr:`deep_research_client.claims.CitationHandle.reference_id` already uses, so
the two are interchangeable.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..validation.extraction import find_reference_ids

__all__ = [
    "NormalizedPublicationId",
    "PublicationIdType",
    "normalize_publication_id",
]

#: A ClinicalTrials.gov registration: NCT followed by exactly eight digits, for
#: example NCT00000419. find_reference_ids does not cover these, so this module
#: adds them.
_NCT_PATTERN = re.compile(r"NCT\d{8}(?!\d)", re.IGNORECASE)

#: A bare http(s) URL, used only as a fallback when no identifier can be read from
#: the string -- the citation points somewhere, even if not to a canonical id.
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

#: Confidence for an identifier recovered from a URL rather than written directly:
#: still a real id, but one indirection from the source's own words.
_FROM_URL_CONFIDENCE = 0.7

#: Confidence for a bare URL with no extractable identifier: it resolves to a page,
#: not to a canonical publication id.
_BARE_URL_CONFIDENCE = 0.5


class PublicationIdType(str, Enum):
    """The kind of identifier a citation resolved to."""

    PMID = "PMID"
    PMCID = "PMCID"
    DOI = "DOI"
    NCT = "NCT"
    GEO = "GEO"
    URL = "URL"
    UNKNOWN = "UNKNOWN"


#: The id-type for each prefix find_reference_ids emits (normalized_id is
#: ``PREFIX:value``). PMC is PubMed Central, reported here as PMCID.
_TYPE_BY_PREFIX: dict[str, PublicationIdType] = {
    "PMID": PublicationIdType.PMID,
    "PMC": PublicationIdType.PMCID,
    "DOI": PublicationIdType.DOI,
    "GEO": PublicationIdType.GEO,
}


@dataclass(frozen=True)
class NormalizedPublicationId:
    """The normalisation of one raw citation string.

    Attributes:
        raw: The citation exactly as given, always kept.
        canonical_id: The normalised identifier (``PMID:7913883``,
            ``DOI:10.1038/x``), or None when nothing could be resolved.
        id_type: Which kind of identifier it is; UNKNOWN when unresolved.
        confidence: How sure the normalisation is, 0 to 1. 1.0 for a cleanly
            written identifier, lower when recovered from a URL, 0.0 when
            unresolved.
        source_url: The URL the identifier came from, when a link gave it.

    Examples:
        >>> NormalizedPublicationId(raw="PMID:1", canonical_id="PMID:1",
        ...     id_type=PublicationIdType.PMID, confidence=1.0).resolved
        True
        >>> NormalizedPublicationId(raw="Smith 2020", canonical_id=None,
        ...     id_type=PublicationIdType.UNKNOWN, confidence=0.0).resolved
        False
    """

    raw: str
    canonical_id: Optional[str]
    id_type: PublicationIdType
    confidence: float
    source_url: Optional[str] = None

    @property
    def resolved(self) -> bool:
        """Whether a canonical identifier could be read from the string."""
        return self.canonical_id is not None


def normalize_publication_id(raw: str) -> NormalizedPublicationId:
    """Normalise one raw citation string to a canonical identifier.

    Equivalent forms resolve to the same id: ``PMID:7913883`` and the matching
    PubMed URL both give ``PMID:7913883``. A string with no readable identifier is
    returned unresolved (``canonical_id=None``, ``confidence=0.0``) with its raw
    text kept, never dropped.

    Args:
        raw: One citation marker or reference string.

    Returns:
        Its :class:`NormalizedPublicationId`.

    Examples:
        >>> normalize_publication_id("PMID: 7913883").canonical_id
        'PMID:7913883'
        >>> normalize_publication_id("https://pubmed.ncbi.nlm.nih.gov/7913883").canonical_id
        'PMID:7913883'
        >>> r = normalize_publication_id("https://pubmed.ncbi.nlm.nih.gov/7913883")
        >>> (r.id_type.value, r.confidence)
        ('PMID', 0.7)
        >>> normalize_publication_id("doi:10.1038/ng1234").canonical_id
        'DOI:10.1038/ng1234'
        >>> normalize_publication_id("Registered as NCT00000419.").canonical_id
        'NCT:NCT00000419'
        >>> u = normalize_publication_id("see https://example.org/paper")
        >>> (u.id_type.value, u.canonical_id, u.confidence)
        ('URL', 'https://example.org/paper', 0.5)
        >>> bad = normalize_publication_id("Smith et al. 2020, Nature")
        >>> (bad.resolved, bad.id_type.value, bad.confidence)
        (False, 'UNKNOWN', 0.0)
    """
    found = find_reference_ids(raw)
    if found:
        # Take the first identifier in appearance order. Its prefix names the kind;
        # a recovered-from-URL hit (url set) is one indirection, so less confident.
        ref = found[0]
        prefix = ref.normalized_id.split(":", 1)[0]
        return NormalizedPublicationId(
            raw=raw,
            canonical_id=ref.normalized_id,
            id_type=_TYPE_BY_PREFIX.get(prefix, PublicationIdType.UNKNOWN),
            confidence=_FROM_URL_CONFIDENCE if ref.url else 1.0,
            source_url=ref.url,
        )

    nct = _NCT_PATTERN.search(raw)
    if nct:
        accession = nct.group(0).upper()
        return NormalizedPublicationId(
            raw=raw,
            canonical_id=f"NCT:{accession}",
            id_type=PublicationIdType.NCT,
            confidence=1.0,
        )

    url = _URL_PATTERN.search(raw)
    if url:
        # A bare URL with no extractable id: it points to a page, not to a
        # canonical publication id, so it resolves only weakly.
        return NormalizedPublicationId(
            raw=raw,
            canonical_id=url.group(0),
            id_type=PublicationIdType.URL,
            confidence=_BARE_URL_CONFIDENCE,
            source_url=url.group(0),
        )

    # Nothing readable. Keep the raw string rather than drop it: a later, online
    # resolver may still turn "Smith et al. 2020" into an id.
    return NormalizedPublicationId(
        raw=raw,
        canonical_id=None,
        id_type=PublicationIdType.UNKNOWN,
        confidence=0.0,
    )
