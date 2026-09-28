"""Check claims against the works they cite. Not implemented yet.

Extraction records what a source says and which work it cites for each claim;
it does not judge whether the cited work supports it (issue #43 keeps the two
apart). This module is where that judgement will go, in a following PR.

The plan, reusing linkml-reference-validator, which this package already
depends on and which ``validation/validator.py`` already drives:

- For each claim with a ``reference_id`` (a PMID or DOI, which most cited
  claims now have, section citations included), fetch the cited work once with
  ``ReferenceFetcher.fetch``, which caches abstracts and PMC full text on disk.
- Decide whether the work supports the claim. The validator's
  ``SupportingTextValidator.validate(supporting_text, reference_id)`` answers
  a narrower question, whether a quote appears in the work, so it fits only a
  claim whose source quotes the work. A report paraphrases, so most claims
  will need a judge that compares ``claim_text`` with the work's text. That
  judge, and a result model to record its verdict with provenance, are the
  design work still to do.
"""

from typing import NoReturn

from .models import ClaimSet

__all__ = ["verify_claims"]


def verify_claims(claims: ClaimSet) -> NoReturn:
    """Check each claim against the work it cites. Not implemented yet.

    Args:
        claims: A claim set from extraction.

    Raises:
        NotImplementedError: Always, until verification is built. See this
            module's docstring for the plan.

    >>> from deep_research_client.claims import ExtractorInfo, SourceDocument, SourceType
    >>> verify_claims(ClaimSet(
    ...     source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
    ...     extractor=ExtractorInfo(name="x"),
    ... ))
    Traceback (most recent call last):
    ...
    NotImplementedError: Checking claims against the works they cite is not implemented yet; see deep_research_client.claims.verification for the plan
    """
    raise NotImplementedError(
        "Checking claims against the works they cite is not implemented yet; "
        "see deep_research_client.claims.verification for the plan"
    )
