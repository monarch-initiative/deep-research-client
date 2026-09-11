"""Scoring functions for the evaluation framework.

Three scoring approaches:

1. **FACT** — Citation verification: extract claims and citations from DR output,
   fetch the cited paper (PubMed abstract), and verify via LLM whether the source
   supports the claim. Produces citation_accuracy and effective_citations.

2. **Claim Recall** — Compare DR output claims against ground truth claims using
   LLM-based semantic matching. Produces claim_recall and optionally claim_precision.

3. **RACE** — Report quality assessment via LLM judge on four dimensions:
   comprehensiveness, accuracy, organization, and terminology correctness.
"""

import json
import logging
import re
from typing import Any

import httpx

from .models import (
    CitationAlignmentResult,
    CitationAlignmentScore,
    CitationExistence,
    CitationVerifiabilityScore,
    CitationVerification,
    ClaimMatch,
    ClaimRecallScore,
    DROutput,
    ExtractedCitation,
    ExtractedClaim,
    FACTScore,
    FactualSpotCheck,
    FactualSpotCheckScore,
    IntrinsicScore,
    RACEDimension,
    RACEScore,
    TopicCoverage,
    TopicCoverageScore,
)
from .datamodel import EvalTask, ReferenceClaim, Rubric, SpotCheck
from .datamodel_helpers import is_prefix_match
from ..validation.extraction import find_reference_ids

logger = logging.getLogger(__name__)

# Truncation limits for LLM context windows
#: Characters of a report sent to a judge. Truncation is recorded on the score
#: as `judged_chars` against `report_chars`, because a report longer than this
#: is judged on its opening only -- coverage understated with nothing saying so.
MAX_REPORT_CHARS = 12000
MAX_ABSTRACT_CHARS = 3000
MAX_DESCRIPTION_CHARS = 500

# ---------------------------------------------------------------------------
# Citation extraction helpers
# ---------------------------------------------------------------------------


def extract_citations_from_markdown(markdown: str) -> list[ExtractedCitation]:
    """Extract all PMID and DOI citations from markdown text.

    Thin wrapper over :func:`deep_research_client.validation.find_reference_ids`,
    which owns the identifier patterns shared with reference validation.

    >>> cits = extract_citations_from_markdown("This was shown (PMID:7913883) and confirmed (DOI:10.1038/ng1234).")
    >>> [c.normalized_id for c in cits]
    ['PMID:7913883', 'DOI:10.1038/ng1234']
    >>> cits2 = extract_citations_from_markdown("See https://pubmed.ncbi.nlm.nih.gov/12345678")
    >>> [c.normalized_id for c in cits2]
    ['PMID:12345678']
    """
    return [
        ExtractedCitation(
            raw_reference=found.raw,
            normalized_id=found.normalized_id,
            url=found.url,
        )
        for found in find_reference_ids(markdown)
    ]


def extract_claims_with_citations(markdown: str) -> list[ExtractedClaim]:
    """Extract claims (sentences) paired with their inline citations.

    Splits the markdown into sentences and associates each sentence with any
    PMIDs or DOIs it contains. Only returns sentences that have at least one
    citation.

    >>> claims = extract_claims_with_citations("FGFR3 causes achondroplasia (PMID:7913883). No citation here.")
    >>> len(claims)
    1
    >>> claims[0].text
    'FGFR3 causes achondroplasia (PMID:7913883).'
    """
    # Split into sentences (rough but good enough for markdown)
    sentences = re.split(r"(?<=[.!?])\s+", markdown)
    claims: list[ExtractedClaim] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        cits = extract_citations_from_markdown(sentence)
        if cits:
            claims.append(ExtractedClaim(text=sentence, citations=cits))

    return claims


# ---------------------------------------------------------------------------
# PubMed abstract fetching
# ---------------------------------------------------------------------------


async def fetch_pubmed_abstract(pmid: str, client: httpx.AsyncClient | None = None) -> str | None:
    """Fetch the abstract text for a PubMed article.

    Args:
        pmid: A PMID string like "PMID:7913883" or just "7913883".
        client: Optional httpx async client for connection reuse.

    Returns:
        Abstract text, or None if not found.
    """
    numeric_id = pmid.replace("PMID:", "").strip()
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=pubmed&id={numeric_id}&rettype=abstract&retmode=text"
    )
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=30) as c:
                resp = await c.get(url)
        else:
            resp = await client.get(url)
        resp.raise_for_status()
        return resp.text.strip() if resp.text.strip() else None
    except Exception:
        logger.warning("Failed to fetch PubMed abstract for %s", pmid, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# LLM judge helpers
# ---------------------------------------------------------------------------


def _nested_with_key(obj: object, key: str) -> dict | None:
    """First dict at or below ``obj`` that has ``key`` at its own top level.

    Searched only when no top-level object in a reply carries the key, so that
    a wrapper -- ``{"response": {"supported": true}}`` -- is still read, without
    letting a key buried in a preamble outrank a later top-level verdict.

    Document-ordered, with no notion of which nested object is the verdict: in
    ``{"analysis": {"supported": false}, "verdict": {"supported": true}}`` the
    analysis answers. The ordering problem the top-level rule solves has no
    equivalent solution one level in, so the descent is a last resort rather
    than a parser.

    >>> _nested_with_key({"response": {"supported": True}}, "supported")
    {'supported': True}
    >>> _nested_with_key({"items": [{"x": 1}, {"score": 4}]}, "score")
    {'score': 4}
    >>> _nested_with_key({"a": 1}, "score") is None
    True

    The ordering the paragraph above describes, pinned rather than asserted --
    first key wins over a later one, first element over a later one, and a
    child over its parent's next sibling:

    >>> _nested_with_key(
    ...     {"analysis": {"supported": False}, "verdict": {"supported": True}},
    ...     "supported")
    {'supported': False}
    >>> _nested_with_key({"items": [{"score": 1}, {"score": 2}]}, "score")
    {'score': 1}
    >>> _nested_with_key({"a": {"b": {"score": 9}}, "c": {"score": 1}}, "score")
    {'score': 9}

    Iterative, over an explicit stack. The recursive version descended one
    frame per level of the decoded value, so a judge reply nested deeper than
    the stack allows raised RecursionError *after* the decoder had read it --
    the fourth place a deep reply broke a recursive consumer, and the one no
    amount of care in the decoder can reach. Depth is a property of the reply,
    not of the parser that read it.

    >>> deep = {"score": 1}
    >>> for _ in range(5000):
    ...     deep = {"wrapper": deep}
    >>> _nested_with_key(deep, "score")
    {'score': 1}
    """
    # Pre-order, document-ordered: a node is answered before its children, and
    # `reversed` puts the first child on top of the stack so siblings are
    # visited left to right -- the order the recursive version had.
    stack: list[object] = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if key in current:
                return current
            stack.extend(reversed(list(current.values())))
        elif isinstance(current, list):
            stack.extend(reversed(current))
    return None


def _balanced_span(text: str, start: int) -> int | None:
    """Index just past the region the bracket at ``text[start]`` encloses.

    None when the bracket never closes at all, and for a ``start`` that is not
    a bracket -- the two cases no caller can confuse, since no caller passes a
    non-bracket. A bracket closed by the *wrong* one is not among them: it ends
    its region at that closer, because the text is damaged either way and the
    caller's question is where the damage stops, not why it is damaged.

    Answering None for both was a two-valued answer to a three-valued question.
    The caller read None as "never closed" and mined the whole remainder as
    salvage, so an ordinary confusion -- an array closed with a brace --
    demoted the genuine verdict after it and let a per-criterion breakdown
    answer instead. That is precisely the outcome the salvage tiering was built
    to avoid, reached by the one shape where the two Nones meant different
    things.

    String-aware, because a brace inside a string value is not a brace: counting
    them naively is the defect `raw_decode` was brought in to remove, and this
    is the one place that still has to find an end without parsing -- the text
    it is asked about is text the parser has already refused.

    >>> _balanced_span('{"a": {"b": 1}} tail', 0)
    15
    >>> _balanced_span('{"note": "a } brace"} tail', 0)
    21
    >>> _balanced_span('{"a": 1', 0) is None
    True

    A mismatched closer bounds the region rather than erasing it:

    >>> _balanced_span('{"a": [1} tail', 0)
    9

    ``start`` must actually be a bracket. The sole caller only ever passes one,
    but the guard is not dead code: without it a non-bracket start walks to the
    first stray closer and returns a region the bracket there does not own --
    harmless while the return value meant "closed cleanly", and wrong now that
    it means "the damage stops here".

    >>> _balanced_span('abc} tail', 0) is None
    True
    """
    closers = {"{": "}", "[": "]"}
    if text[start] not in closers:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in closers:
            stack.append(closers[char])
        elif char in "}]":
            # `stack` is never empty here: the scan starts at an opener, which
            # pushes, and returns the moment a pop empties it.
            if stack[-1] != char:
                # Closed with the wrong bracket. The region ends here: past
                # this point the text is no longer inside the damaged
                # container, so whatever follows is top-level again.
                return i + 1
            stack.pop()
            if not stack:
                return i + 1
    return None


def _without_trailing_commas(text: str) -> str:
    """Drop commas that directly precede a closing bracket.

    A trailing comma is the commonest way an LLM's JSON fails to parse, and the
    cost of not repairing it is not a missing verdict but the wrong one: the
    object holding the answer is unreadable, so the only thing left to recover
    is whatever it happens to contain -- a per-criterion breakdown, say, whose
    verdict is the opposite of the one the judge reached.

    String-aware, so a comma inside a string value survives.

    The comma becomes a space rather than being removed, so every offset in the
    text is unchanged: a span measured on the original still addresses the same
    characters in the repaired copy.

    >>> _without_trailing_commas('{"a": 1,}')
    '{"a": 1 }'
    >>> _without_trailing_commas('{"a": [1, 2, ], }')
    '{"a": [1, 2  ]  }'
    >>> _without_trailing_commas('{"note": "a, }", "b": 2}')
    '{"note": "a, }", "b": 2}'
    """
    out = list(text)
    in_string = False
    escaped = False
    comma_at: int | None = None
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            comma_at = None
        elif char == ",":
            comma_at = i
        elif char in "}]":
            if comma_at is not None:
                out[comma_at] = " "
            comma_at = None
        elif not char.isspace():
            comma_at = None
    return "".join(out)


def _decode_candidates(text: str) -> tuple[list[Any], list[Any]]:
    """Split a reply's JSON values into top-level ones and salvaged ones.

    A value is top-level when the scan reached it without being inside a
    container it could not read. One found inside such a container is salvage:
    recoverable, but not evidence of what the reply *says*, because whatever
    enclosed it is unreadable.

    The distinction exists because the tiering was previously inferred from
    where the parser happened to succeed. On a decode failure the scan advanced
    a single character, so an object nested inside a malformed outer became a
    *top-level* candidate and outranked a genuine verdict after it -- a judge
    answering `true` with a trailing comma and a nested breakdown was recorded
    as `false`. A malformed container is now measured with `_balanced_span`,
    repaired if a trailing comma is all that was wrong with it, and otherwise
    mined for salvage and stepped over.

    Returns:
        ``(top_level, salvage)``. `top_level` is in document order; `salvage`
        is not, since a mined container contributes its own top-level values
        before its own salvage. Nothing reads that order -- the tier that
        consults salvage says so in as many words -- but the two statements
        should not disagree.
    """
    decoder = json.JSONDecoder()
    top_level: list[Any] = []
    salvage: list[Any] = []
    # Set once the scan enters a container that never closes. Nothing after
    # that point is top-level, because the container has no end to be after.
    inside_unclosed = False

    def record(value: Any) -> None:
        """Put a decoded value in the tier the scan is currently in."""
        (salvage if inside_unclosed else top_level).append(value)

    # A container needs a closer to be complete, so an opener with none left
    # after it cannot begin one. A reply that simply runs out -- the shape
    # truncation produces -- has no closer at all, and this turns the whole
    # run into one comparison instead of one descent per opener. Sound rather
    # than heuristic: it rules out only openers that provably cannot close.
    last_closer = max(text.rfind("}"), text.rfind("]"))

    i = 0
    while i < len(text):
        if i > last_closer:
            break
        if text[i] not in "{[":
            i += 1
            continue
        try:
            parsed, end = decoder.raw_decode(text, i)
        except RecursionError:
            # `raw_decode` descends one frame per nesting level, so a reply
            # nested deeper than the stack allows raises this instead of
            # returning -- and `RecursionError` is not a `JSONDecodeError`, so
            # it escaped this function entirely and the callers' bare
            # `except Exception` recorded it as a scoring error. That is the
            # outcome removing the recursion from this loop existed to
            # prevent: an unjudged measurement reported as an error, when
            # returning None lets the regex fallback have its turn.
            #
            # Unreadable, which is what `inside_unclosed` means to the tiers
            # below -- with a caveat worth stating, since this branch is the
            # one place the flag is set for something that may be perfectly
            # well-formed: it means "could not read it *this deep*", not
            # "damaged". Everything after such a preamble is demoted to
            # salvage, so a genuine later verdict can lose to a value mined
            # out of a valid-but-deep preamble -- the outranking the tiering
            # exists to prevent, reached by the one route it cannot see. The
            # alternative was escaping as a scoring error, so this is recorded
            # rather than fixed.
            #
            # It needs ~10,000 levels of nesting, which is where `raw_decode`
            # gives out. That is NOT the budget the mine below runs against:
            # the mine is Python recursion, which gives out at ~999 frames.
            # Both bisected on both interpreters CI runs -- 9,998 and 999 on
            # 3.12.3, 9,999 and 999 on 3.13.12 -- since a budget quoted from
            # one version is a claim about the other. Two branches of one
            # function, two limits an order of magnitude apart, and a case
            # sized against the wrong one lands between them.
            inside_unclosed = True
            i += 1
            continue
        except json.JSONDecodeError:
            if inside_unclosed:
                # Already inside a container with no end. Whatever is here is
                # salvage whichever way it is bounded, so there is nothing to
                # decide and `raw_decode` alone finds the well-formed values.
                #
                # Skipping the bound keeps `_balanced_span` off this path:
                # it scans to the end of the text when nothing closes, and
                # calling it at every subsequent opener made a reply that
                # degenerates into a run of braces -- an ordinary LLM failure
                # -- take six seconds at a 12,000-character reply, measured,
                # where it used to fail fast with a RecursionError. (Measured
                # at that size, not bounded by it: MAX_REPORT_CHARS truncates
                # the report sent to the judge, not the reply that comes back.)
                # The cost of the skip is a trailing comma left unrepaired
                # inside text already declared unreadable.
                #
                # It does not on its own make the loop linear, and saying so
                # hid the sibling opener for a round: `raw_decode` is the
                # other per-opener cost, a run of `[` is valid JSON
                # continuation rather than an immediate decode error, and the
                # skip does not touch it. The no-closer bound above is what
                # covers that, and what is left after both -- unclosed
                # openers with a stray closer somewhere after them -- is
                # quadratic in the run, bounded by the reply's length.
                i += 1
                continue
            span = _balanced_span(text, i)
            if span is None:
                # Never closed -- a reply cut off mid-object, say. The scan
                # carries on in place so a value inside it stays reachable, and
                # everything from here on is salvage. Recursing on the
                # remainder instead made the depth the number of consecutive
                # unclosed openers, so a reply with a thousand of them raised
                # RecursionError -- turning an unjudged measurement into an
                # error, which is the worse report of the same thing.
                inside_unclosed = True
                i += 1
                continue
            try:
                repaired = json.loads(_without_trailing_commas(text[i:span]))
            except (json.JSONDecodeError, RecursionError):
                # Damaged beyond a trailing comma, or nested deeper than the
                # parser descends. From here those are the same thing -- a
                # bounded container this parser cannot read -- so both mine it
                # and step over it.
                #
                # The depth case used to `pass` instead, on the argument that
                # mining recurses the same way and reaches the same answer. It
                # does not: `pass` discards the container whole, so a verdict
                # sitting in a *readable* branch of a container that is too
                # deep somewhere else went with it. `{"a": [1,], "verdict":
                # {"score": 4}, "b": <10,000 deep>}` answered None where the
                # mine finds the verdict.
                #
                # The mine recurses, so it can reach the limit again -- and
                # malformed containers nest as deeply as a reply is long, so
                # that is caught rather than asserted away. Recovering from a
                # RecursionError and immediately recursing is safe here: the
                # interpreter's old fatal-error-on-overflow path is gone in
                # 3.12, and `requires-python` is >=3.12.
                try:
                    inner_top, inner_salvage = _decode_candidates(
                        text[i + 1 : span - 1]
                    )
                except RecursionError:
                    # Not individually observable, and said out loud rather
                    # than implied otherwise: the mine bottoms out at the
                    # recursion limit inside a nested frame's `json.loads`,
                    # which that frame's own handler above already catches, so
                    # removing this one changes no measured result. Kept
                    # because it guards the call at its own site -- the
                    # recursion here is bounded only by how deeply malformed
                    # containers nest, which is the reply's length over a
                    # constant.
                    inner_top, inner_salvage = [], []
                salvage.extend(inner_top)
                salvage.extend(inner_salvage)
            else:
                record(repaired)
            i = span
            continue
        record(parsed)
        i = end  # never rescan inside a value already read
    return top_level, salvage


def _extract_json_object(text: str, key: str | None = None) -> dict | None:
    """Extract a JSON object from text, handling nested braces.

    Every ``{...}`` object at the top level of the reply is a candidate; an
    array is decoded and stepped over rather than mined, so its members are
    reachable only through the descent. With ``key``, the first candidate that
    carries it wins; if none does, the search descends before falling back to
    the first object found.

    Three things have been wrong here, and they compound:

    1. Only the first ``{...}`` run was tried. A judge that narrates before
       answering -- ``{"thinking": "..."} {"supported": true}`` -- took the
       no-verdict path with its verdict sitting in the text.
    2. Then every run was tried, but the first that *parsed* won, so the
       narration object still won. Each caller now passes the key it asked
       for; a reply that never answers is still unjudged, which is a lost
       measurement rather than an invented one.
    3. Then the scan resumed one character after a parsed object, which is a
       brace *inside* it -- so a key nested in a preamble outranked a later
       top-level verdict. ``{"evidence": {"supported": false}}`` followed by
       ``{"supported": true}`` returned **false**: not a missing verdict but
       the opposite one, counted against the provider. The scan now advances
       past each decoded object, and nesting is consulted only when no
       top-level object answers.

    Cost, which three rewrites of this paragraph have got wrong in a different
    way each time, so stated per shape:

    - Linear while values parse -- the scan advances past each decoded span.
    - Constant for a reply that simply runs out, which is what truncation
      produces: no closer anywhere means no opener can begin a container, and
      one comparison rules out the whole run.
    - Quadratic in the run only for unclosed openers with a stray closer
      somewhere after them, since neither the bound above nor the skip inside
      `_decode_candidates` applies. Bounded by the length of the reply.

    The length is in CHARACTERS -- the scan steps one at a time -- and what
    bounds it is `max_tokens`, set where the judge is called and currently
    2048, NOT `MAX_REPORT_CHARS`, which two earlier versions of this paragraph
    cited. That constant truncates the report sent *to* the judge and says
    nothing about what comes back.

    A third version of this sentence then said the mistake made the bound look
    "about six times worse", which is 12,000 divided by 2048: characters over
    tokens, in the paragraph arguing that the two measure different things.
    2048 tokens is roughly 6,000-9,000 characters of JSON-ish text, so the
    overstatement was nearer 1.5x.

    And `max_tokens` bounds a reply THIS CLIENT asked for. Every scorer takes
    an `llm_client` from its caller and `eval score` accepts `--llm-base-url`,
    so an endpoint that ignores it, or a library caller driving `score_race`
    directly, is not bounded by it at all.

    ``raw_decode`` does the scanning rather than a brace counter, so a closing
    brace inside a string value no longer ends the object early -- an
    explanation mentioning "a } brace" lost the verdict entirely.

    >>> _extract_json_object('blah {"a": 1, "b": {"c": 2}} done')
    {'a': 1, 'b': {'c': 2}}
    >>> _extract_json_object('no json here') is None
    True
    >>> _extract_json_object('{"supported": true, "explanation": "yes"}')
    {'supported': True, 'explanation': 'yes'}

    The narrating judge, with and without the key:

    >>> _extract_json_object('{"thinking": "hmm"} then {"supported": true}')
    {'thinking': 'hmm'}
    >>> _extract_json_object('{"thinking": "hmm"} then {"supported": true}',
    ...                      key="supported")
    {'supported': True}

    A key inside a preamble does not outrank a top-level verdict:

    >>> _extract_json_object('{"evidence": {"supported": false}} '
    ...                      '{"supported": true}', key="supported")
    {'supported': True}

    But a wrapper whose only content is the verdict is still read:

    >>> _extract_json_object('{"response": {"supported": true}}', key="supported")
    {'supported': True}

    An array is not a candidate and neither are its members, so a breakdown
    emitted as a list does not outrank the verdict after it:

    >>> _extract_json_object('[{"criterion": "depth", "score": 2}] {"score": 4}',
    ...                      key="score")
    {'score': 4}

    But a reply that is only an array still has its verdict found, through the
    descent rather than as a top-level candidate:

    >>> _extract_json_object('[{"score": 3}]', key="score")
    {'score': 3}

    A run that does not parse is skipped, and one nested inside it is still
    reachable:

    >>> _extract_json_object('not json {oops} but {"supported": false}')
    {'supported': False}
    >>> _extract_json_object('{oops {"supported": true}}', key="supported")
    {'supported': True}

    A brace inside a string value no longer truncates the object:

    >>> _extract_json_object('{"explanation": "a } brace", "supported": true}',
    ...                      key="supported")["supported"]
    True

    When nothing carries the key, the first object found is returned, so the
    caller can quote the judge's actual reply rather than nothing:

    >>> _extract_json_object('{"verdict": "yes"}', key="supported")
    {'verdict': 'yes'}
    """
    top_level, salvage = _decode_candidates(text)
    top_dicts = [v for v in top_level if isinstance(v, dict)]
    salvage_dicts = [v for v in salvage if isinstance(v, dict)]

    def _fallback() -> dict | None:
        # The judge's actual reply, so the caller can quote it in the
        # explanation rather than reporting nothing. A list is never returned:
        # the caller calls `.get` on whatever comes back.
        if top_dicts:
            return top_dicts[0]
        return salvage_dicts[0] if salvage_dicts else None

    if key is None:
        return _fallback()

    # Three tiers, in order of how much the reply commits to each answer: a
    # top-level object that carries the key; one nested inside a readable
    # top-level value; anything at all inside a container that could not be
    # read. Salvage is one tier, not two -- `_nested_with_key` returns a dict
    # that carries the key itself, so a separate "top-level salvage" pass could
    # never answer a case the descent does not, and the ordering it would
    # impose within salvage has nothing behind it: the container was
    # unreadable, so nothing there is more the reply's answer than anything
    # else.
    for candidate in top_dicts:
        if key in candidate:
            return candidate
    for value in top_level:
        nested = _nested_with_key(value, key)
        if nested is not None:
            return nested
    for value in salvage:
        nested = _nested_with_key(value, key)
        if nested is not None:
            return nested
    return _fallback()


async def _llm_judge(prompt: str, llm_client: Any, model: str = "gpt-4o-mini") -> str:
    """Call an LLM to judge/evaluate. Expects an OpenAI-compatible client.

    Args:
        prompt: The evaluation prompt.
        llm_client: An openai.AsyncOpenAI-compatible client.
        model: Model name to use for the judge.

    Returns:
        The LLM response text.
    """
    response = await llm_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# FACT scorer
# ---------------------------------------------------------------------------


async def score_fact(
    dr_output: DROutput,
    llm_client: Any,
    pubmed_client: httpx.AsyncClient | None = None,
    model: str = "gpt-4o-mini",
) -> FACTScore:
    """Compute FACT score for a DR output.

    For each claim-citation pair:
    1. Fetch the cited paper's abstract from PubMed
    2. Ask an LLM whether the abstract supports the claim
    3. Tally results into citation_accuracy and effective_citations

    Args:
        dr_output: Parsed DR output with extracted claims and citations.
        llm_client: OpenAI-compatible async client for verification.
        pubmed_client: Optional httpx client for PubMed API calls.

    Returns:
        FACTScore with verification details.
    """
    claims = dr_output.extracted_claims
    if not claims:
        claims = extract_claims_with_citations(dr_output.raw_markdown)

    verifications: list[CitationVerification] = []

    for claim in claims:
        for citation in claim.citations:
            if not citation.normalized_id or not citation.normalized_id.startswith("PMID:"):
                # Skip non-PMID citations for now (DOI verification is harder)
                verifications.append(
                    CitationVerification(
                        citation=citation,
                        claim_text=claim.text,
                        error="Non-PMID citation, skipped",
                    )
                )
                continue

            abstract = await fetch_pubmed_abstract(citation.normalized_id, client=pubmed_client)
            if not abstract:
                verifications.append(
                    CitationVerification(
                        citation=citation,
                        claim_text=claim.text,
                        error="Could not retrieve abstract",
                    )
                )
                continue

            prompt = (
                "You are a scientific citation verifier. Given a CLAIM from a research report "
                "and the ABSTRACT of the cited paper, determine whether the abstract provides "
                "evidence that supports the claim.\n\n"
                f"CLAIM: {claim.text}\n\n"
                f"CITED PAPER ABSTRACT:\n{abstract[:MAX_ABSTRACT_CHARS]}\n\n"
                "Does this abstract support the claim? Respond with a JSON object:\n"
                '{"supported": true/false, "explanation": "brief explanation"}'
            )
            try:
                result_text = await _llm_judge(prompt, llm_client, model=model)
                result = _extract_json_object(result_text, key="supported")
                if result and result.get("supported") is not None:
                    supported = result.get("supported")
                    explanation = result.get("explanation", "")
                else:
                    # Reached for a reply that is not JSON *and* for well-formed
                    # JSON under a different key -- {"verdict": "yes"}, an
                    # {"error": ...} envelope from a proxy. `.get(key, False)`
                    # read those as a negative verdict, which lands in the
                    # checkable set and counts against the provider: the same
                    # "no verdict is not a verdict of no" argued just below, in
                    # the branch that did not have it.
                    #
                    # No verdict. `"true" in result_text[:50]` scored a judge
                    # that replied in prose on whether those four letters
                    # happened to appear: "It is not true that this abstract
                    # supports the claim" was read as support. That is the
                    # invent-an-answer defect the multiple-choice extractor
                    # documents, and unlike an unfetchable abstract it landed
                    # in `checkable`, counting as a *verified* citation.
                    supported = None
                    explanation = (
                        "Judge returned no parseable verdict: " + result_text[:200]
                    )

                verifications.append(
                    CitationVerification(
                        citation=citation,
                        claim_text=claim.text,
                        supported=supported,
                        explanation=explanation,
                        source_text=abstract[:500],
                    )
                )
            except Exception as e:
                verifications.append(
                    CitationVerification(
                        citation=citation,
                        claim_text=claim.text,
                        error=str(e),
                    )
                )

    # Compute aggregate scores
    # A verdict of None means the pair was never judged -- a DOI, whose
    # abstract this scorer cannot fetch, a PMID with no abstract, or a reply
    # with no verdict in it. Those leave the rate, as they always have; what is
    # new is that the count is reported, so that "nothing was judged" and
    # "nothing was supported" stop rendering as the same 0.00.
    judged = [v for v in verifications if v.supported is not None]
    verified = sum(1 for v in judged if v.supported)

    return FACTScore(
        # Every pair found, so the name means what it says. The rate is over
        # the judged ones, and the difference is reported rather than folded
        # into the total -- which is how a report citing only DOIs used to
        # print `0/0` in the format of a measured zero.
        total_citations=len(verifications),
        verified_citations=verified,
        citation_accuracy=verified / len(judged) if judged else 0.0,
        effective_citations=verified,
        unjudged_citations=len(verifications) - len(judged),
        verifications=verifications,
    )


# ---------------------------------------------------------------------------
# Claim recall scorer
# ---------------------------------------------------------------------------


async def score_claim_recall(
    dr_output: DROutput,
    ground_truth_claims: list[ReferenceClaim],
    llm_client: Any,
    model: str = "gpt-4o-mini",
) -> ClaimRecallScore:
    """Compute claim recall: what fraction of ground truth claims appear in the DR output.

    Uses an LLM to semantically match each ground truth claim against the
    full DR output text.

    Args:
        dr_output: Parsed DR output.
        ground_truth_claims: List of ground truth claims to check for.
        llm_client: OpenAI-compatible async client.

    Returns:
        ClaimRecallScore with per-claim match details.
    """
    if not ground_truth_claims:
        # Provenance recorded here too, so the two fields are not
        # sometimes-absent for two different reasons. `judged_chars` is 0
        # rather than the truncation length: no judge was asked anything, so
        # saying a report's first 12,000 characters were read would be a claim
        # about work that never happened -- and would print a truncation note
        # for a scorer that did not truncate.
        return ClaimRecallScore(
            total_ground_truth_claims=0, matched_claims=0, claim_recall=0.0,
            judged_chars=0,
            report_chars=len(dr_output.raw_markdown),
        )

    # Truncate output for LLM context
    report_text = dr_output.raw_markdown[:MAX_REPORT_CHARS]

    matches: list[ClaimMatch] = []
    for gt_claim in ground_truth_claims:
        prompt = (
            "You are evaluating whether a research report covers a specific claim.\n\n"
            f"GROUND TRUTH CLAIM:\n"
            f"Category: {gt_claim.category}\n"
            f"Name: {gt_claim.name}\n"
            f"Description: {gt_claim.description[:MAX_DESCRIPTION_CHARS]}\n\n"
            f"RESEARCH REPORT (excerpt):\n{report_text}\n\n"
            "Does the research report contain information that substantially covers "
            "this ground truth claim? The report does not need to use identical wording, "
            "but should convey the same key facts.\n\n"
            "Respond with a JSON object:\n"
            '{"matched": true/false, "best_matching_text": "quote from report or null", '
            '"explanation": "brief explanation"}'
        )
        try:
            result_text = await _llm_judge(prompt, llm_client, model=model)
            result = _extract_json_object(result_text, key="matched")
            if result and result.get("matched") is not None:
                matched = result.get("matched")
                best_text = result.get("best_matching_text")
                explanation = result.get("explanation", "")
            else:
                # As above: no verdict is not a verdict of "no". Left unmatched
                # *and* unscored, so it shrinks the denominator rather than
                # being counted as a claim the report failed to cover.
                matched = None
                best_text = None
                explanation = (
                    "Judge returned no parseable verdict: " + result_text[:200]
                )

            matches.append(
                ClaimMatch(
                    ground_truth_claim_name=gt_claim.name,
                    ground_truth_claim_description=gt_claim.description[:200],
                    matched=matched,
                    best_matching_text=best_text,
                    explanation=explanation,
                )
            )
        except Exception as e:
            logger.warning("Failed to match claim %s: %s", gt_claim.name, e)
            matches.append(
                ClaimMatch(
                    ground_truth_claim_name=gt_claim.name,
                    ground_truth_claim_description=gt_claim.description[:200],
                    matched=None,
                    explanation=f"Error: {e}",
                )
            )

    # Recall is over the claims the judge actually ruled on. Counting an
    # unanswered claim as unmatched made recall fall with the judge's uptime --
    # a provider scored for an outage. `score_fact` already divides by the
    # citations it could check; this is the same argument.
    judged = [m for m in matches if m.matched is not None]
    matched_count = sum(1 for m in judged if m.matched)
    total = len(ground_truth_claims)

    return ClaimRecallScore(
        judged_chars=len(report_text),
        report_chars=len(dr_output.raw_markdown),
        total_ground_truth_claims=total,
        matched_claims=matched_count,
        unjudged_claims=total - len(judged),
        claim_recall=matched_count / len(judged) if judged else 0.0,
        matches=matches,
    )


# ---------------------------------------------------------------------------
# RACE scorer
# ---------------------------------------------------------------------------

_RACE_DIMENSIONS = [
    (
        "comprehensiveness",
        "How thoroughly does the report cover the topic? Does it address all major aspects "
        "including molecular mechanisms, key genes/proteins, relevant pathways, and clinical implications?",
    ),
    (
        "accuracy",
        "Are the biological and medical facts in the report correct? Are the described mechanisms, "
        "gene functions, protein interactions, and clinical details accurate based on current scientific understanding?",
    ),
    (
        "organization",
        "Is the report well-structured with clear sections, logical flow, and appropriate use of "
        "headings? Does it present information in a way that builds understanding progressively?",
    ),
    (
        "terminology",
        "Does the report use correct scientific terminology? Does it appropriately reference "
        "ontology terms (Gene Ontology, Human Phenotype Ontology, disease ontologies) where relevant?",
    ),
]


def _in_scale(raw: object) -> bool:
    """Whether a judge's `score` is a number on the 1-5 scale it was asked for.

    `bool` is rejected before the conversion, because it is a subclass of
    `int`: `float(True)` is 1.0, which sits *inside* the scale, so a judge
    replying `{"score": true}` was recorded as a genuine bottom-of-scale
    score and counted in the average. `true` is no more a number than
    "excellent" is, and unlike a 9 nothing downstream could tell it from a
    measurement.

    >>> [_in_scale(v) for v in (1, 3.5, 5, "4")]
    [True, True, True, True]
    >>> [_in_scale(v) for v in (0, 9, -1, "excellent", None, [4], True, False)]
    [False, False, False, False, False, False, False, False]
    """
    if isinstance(raw, bool):
        return False
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return 1.0 <= value <= 5.0


async def score_race(
    dr_output: DROutput,
    task: EvalTask,
    llm_client: Any,
    model: str = "gpt-4o-mini",
) -> RACEScore:
    """Compute RACE score for a DR output: report quality via LLM judge.

    Evaluates four dimensions: comprehensiveness, accuracy, organization, terminology.
    Each scored 1-5 by an LLM judge.

    Args:
        dr_output: Parsed DR output.
        task: The evaluation task (provides context for scoring).
        llm_client: OpenAI-compatible async client.

    Returns:
        RACEScore with per-dimension details.
    """
    report_text = dr_output.raw_markdown[:MAX_REPORT_CHARS]

    # Build ground truth summary for the judge
    gt_summary_parts = []
    for claim in _reference_claims(task)[:20]:
        terms_str = ", ".join(f"{t.id} ({t.label})" for t in (claim.ontology_terms or [])[:3])
        gt_summary_parts.append(
            f"- [{claim.category}] {claim.name}: {claim.description[:150]}"
            + (f" (Terms: {terms_str})" if terms_str else "")
        )
    gt_summary = "\n".join(gt_summary_parts) if gt_summary_parts else "No specific ground truth provided."

    dimensions: list[RACEDimension] = []
    for dim_name, dim_description in _RACE_DIMENSIONS:
        prompt = (
            "You are an expert scientific evaluator assessing a deep research report.\n\n"
            f"TASK QUERY: {task.prompt}\n\n"
            f"KNOWN GROUND TRUTH (key claims that should be covered):\n{gt_summary}\n\n"
            f"RESEARCH REPORT:\n{report_text}\n\n"
            f"EVALUATION DIMENSION: {dim_name}\n"
            f"CRITERIA: {dim_description}\n\n"
            "Score the report on this dimension from 1 (very poor) to 5 (excellent).\n"
            "Respond with a JSON object:\n"
            '{"score": <1-5>, "explanation": "brief justification"}'
        )
        try:
            result_text = await _llm_judge(prompt, llm_client, model=model)
            result = _extract_json_object(result_text, key="score")
            raw = result.get("score") if result else None
            if result is not None and raw is not None and _in_scale(raw):
                score = float(raw)
                explanation = result.get("explanation", "")
            elif raw is not None:
                # A judge answering 9 on a 1-5 scale is not a confident 5, and
                # one answering "excellent" is not a number at all. Clamping
                # turned both into the top of the scale -- a value the judge
                # never gave, in the one dimension the clamp guaranteed would
                # look best. Recorded as unscored, like any other reply this
                # cannot read.
                score = None
                # Truncated and coerced for the same reason `result_text` is
                # below: a judge that puts an object here would otherwise
                # render its repr into the explanation at full length.
                said = str(result.get("explanation", ""))[:200] if result else ""
                explanation = f"Judge returned a score outside 1-5: {raw!r}"
                if said:
                    # Kept, not replaced: the dimension is unscored either way,
                    # and the judge's reasoning is the only thing left to
                    # diagnose a mis-prompted judge with.
                    explanation += f" -- it said: {said}"
            else:
                # No parseable verdict is not a middling verdict. Recorded as
                # unscored so it leaves the average rather than dragging it to
                # the middle -- a judge outage used to report 3.0 out of 5 for
                # every dimension of every report.
                score = None
                explanation = "Judge returned no parseable score: " + result_text[:200]

            dimensions.append(
                RACEDimension(
                    dimension=dim_name,
                    score=score,
                    max_score=5.0,
                    explanation=explanation,
                )
            )
        except Exception as e:
            logger.warning("Failed to score dimension %s: %s", dim_name, e)
            dimensions.append(
                RACEDimension(
                    dimension=dim_name, score=None, max_score=5.0,
                    explanation=f"Error: {e}",
                )
            )

    return RACEScore(
        dimensions=dimensions,
        judged_chars=len(report_text),
        report_chars=len(dr_output.raw_markdown),
    )


# ---------------------------------------------------------------------------
# Intrinsic (LLM-free) scorers
# ---------------------------------------------------------------------------

# Biomedical stop words to exclude from term overlap calculations
_BIO_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "is", "are",
    "was", "were", "by", "with", "from", "on", "at", "as", "its", "this",
    "that", "which", "but", "not", "has", "have", "had", "been", "be",
    "can", "may", "will", "also", "than", "into", "both", "through",
    "between", "via", "role", "study", "studies", "analysis", "using",
    "effect", "effects", "novel", "new", "results", "data", "evidence",
    "activity", "function", "functions", "involved", "associated",
    "specific", "revealed", "showed", "found", "identified", "demonstrated",
    "important", "required", "dependent", "independent", "human",
})


def _extract_key_terms(text: str) -> set[str]:
    """Extract meaningful biomedical terms from text.

    Returns lowercase terms of 3+ characters, excluding common stop words.

    >>> sorted(_extract_key_terms("BRCA1 DNA repair in homologous recombination"))
    ['brca1', 'dna', 'homologous', 'recombination', 'repair']
    """
    words = re.findall(r"[A-Za-z0-9]{3,}", text.lower())
    return {w for w in words if w not in _BIO_STOP_WORDS}


async def fetch_pubmed_metadata(
    pmid: str, client: httpx.AsyncClient | None = None
) -> dict[str, str | int | None]:
    """Fetch title and year for a PubMed article.

    Args:
        pmid: A PMID string like "PMID:7913883" or just "7913883".
        client: Optional httpx async client for connection reuse.

    Returns:
        Dict with 'title', 'year', and 'exists' keys.

    Example::

        >>> import asyncio
        >>> result = asyncio.run(fetch_pubmed_metadata("PMID:7913883"))  # doctest: +SKIP
        >>> result["exists"]  # doctest: +SKIP
        True
    """
    numeric_id = pmid.replace("PMID:", "").strip()
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&id={numeric_id}&retmode=json"
    )
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=30) as c:
                resp = await c.get(url)
        else:
            resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

        result_data = data.get("result", {}).get(numeric_id)
        if result_data is None:
            # A 200 whose body says nothing about this uid. That is what NCBI
            # returns for a *top-level* failure -- `{"esummaryresult": ["Invalid
            # db name", ...]}`, a rate-limit envelope, an error page that still
            # parses as JSON -- and `raise_for_status` cannot see it, because
            # the status is 200. Nothing was learned about the citation, so it
            # is a failed lookup.
            #
            # Defaulting it to "not a failed lookup" was the round-twenty defect
            # by the one route that fix did not enumerate, and by the route a
            # burst of citation lookups is likeliest to take: rate-limited by
            # NCBI, every citation in the report reported as fabricated.
            return {
                "exists": False, "title": None, "year": None,
                "error": f"no result entry for {numeric_id} in the response",
                "lookup_failed": True,
            }

        if "error" in result_data:
            # NCBI reports an unknown uid as a per-uid `error`. That is the
            # authoritative negative -- the single thing this scorer exists to
            # detect -- not a failed lookup, so `lookup_failed` stays False and
            # the citation counts against verifiability.
            return {
                "exists": False, "title": None, "year": None,
                "error": result_data["error"], "lookup_failed": False,
            }

        title = result_data.get("title") or None
        pubdate = result_data.get("pubdate", "")
        year = None
        if pubdate:
            year_match = re.search(r"(\d{4})", pubdate)
            if year_match:
                year = int(year_match.group(1))

        # A record came back, so the uid is real: `exists` is about the record,
        # not about the title. It was `bool(title)`, which reported a real paper
        # whose summary happens to carry no title as a fabricated citation --
        # the same conflation, one field along. A missing title costs alignment
        # its comparison, and alignment says so itself.
        return {
            "exists": True, "title": title, "year": year,
            "lookup_failed": False,
        }
    except Exception as e:
        # Transport: a timeout, a 5xx, a connection error. Nothing was learned
        # about the citation, so it is excluded from the rate rather than
        # counted as fabricated.
        logger.warning("Failed to fetch PubMed metadata for %s: %s", pmid, e)
        return {
            "exists": False, "title": None, "year": None,
            "error": str(e), "lookup_failed": True,
        }


async def resolve_doi(
    doi: str, client: httpx.AsyncClient | None = None
) -> dict[str, str | int | None]:
    """Resolve a DOI to get paper title and year via CrossRef API.

    Args:
        doi: A DOI string like "DOI:10.1038/ng1234" or "10.1038/ng1234".
        client: Optional httpx async client for connection reuse.

    Returns:
        Dict with 'title', 'year', and 'exists' keys.
    """
    doi_id = doi.replace("DOI:", "").strip()
    url = f"https://api.crossref.org/works/{doi_id}"
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=30) as c:
                resp = await c.get(url, headers={"Accept": "application/json"})
        else:
            resp = await client.get(url, headers={"Accept": "application/json"})

        if resp.status_code == 404:
            # CrossRef does not know this DOI. That is the authoritative
            # negative -- the finding, not a failed lookup -- so `lookup_failed`
            # is False, and stated rather than left to the default: every caller
            # that draws this distinction reads the key, and a branch that
            # carries neither key is one rename away from being read as either.
            # `error` too, so a fabricated DOI reaches `--output` with a reason
            # beside it, the way a fabricated PMID carries NCBI's.
            return {
                "exists": False, "title": None, "year": None,
                "error": "CrossRef does not know this DOI",
                "lookup_failed": False,
            }
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message")
        if not message:
            # 200 with no work record: the same "answered, but not about this
            # identifier" shape as NCBI's top-level error above.
            return {
                "exists": False, "title": None, "year": None,
                "error": "no work record in the CrossRef response",
                "lookup_failed": True,
            }

        title_list = message.get("title", [])
        title = title_list[0] if title_list else None

        year = None
        published = message.get("published", {})
        date_parts = published.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        # CrossRef returned the work, so the DOI resolves, whether or not the
        # record carries a title.
        return {
            "exists": True, "title": title, "year": year,
            "lookup_failed": False,
        }
    except Exception as e:
        logger.warning("Failed to resolve DOI %s: %s", doi, e)
        return {
            "exists": False, "title": None, "year": None,
            "error": str(e), "lookup_failed": True,
        }


async def score_citation_verifiability(
    dr_output: DROutput,
    pubmed_client: httpx.AsyncClient | None = None,
) -> CitationVerifiabilityScore:
    """Check whether each citation in the DR output resolves to a real paper.

    No LLM needed — just checks PubMed/CrossRef APIs.

    Args:
        dr_output: Parsed DR output with extracted citations.
        pubmed_client: Optional httpx client for connection reuse.

    Returns:
        CitationVerifiabilityScore with per-citation existence checks.
    """
    citations = dr_output.extracted_citations
    if not citations:
        citations = extract_citations_from_markdown(dr_output.raw_markdown)

    results: list[CitationExistence] = []
    years: list[int] = []

    for cit in citations:
        cid = cit.normalized_id
        if not cid:
            # A property of the report's own reference list, not of anyone's
            # network: it counts against verifiability.
            results.append(CitationExistence(
                citation_id=cit.raw_reference, exists=False,
                error="Could not normalize", lookup_failed=False,
            ))
            continue

        if cid.startswith("PMID:"):
            meta = await fetch_pubmed_metadata(cid, pubmed_client)
        elif cid.startswith("DOI:"):
            meta = await resolve_doi(cid, pubmed_client)
        else:
            # An identifier this scorer has no resolver for -- the extractor
            # also emits PMC accessions and GEO accessions, which are real,
            # resolvable identifiers that simply are not looked up here.
            # Nothing was learned about the citation, so it leaves the rate.
            #
            # It used to count against verifiability, so a report citing a real
            # PMC article and a real GEO series printed `0/2 (0.00)` -- both
            # invented -- one line above the alignment line saying neither
            # could be checked. Its sibling was taught this distinction one
            # commit ago; the two scorers read the same reference list and
            # should not disagree about whether citing it is a defect.
            #
            # Distinct from an identifier that would not normalise, which is a
            # property of the report's own text and still counts against it.
            results.append(CitationExistence(
                citation_id=cid, exists=None,
                error=(
                    "No resolver for this identifier kind: "
                    f"{cid.split(':', 1)[0] if ':' in cid else cid}"
                ),
                lookup_failed=True,
            ))
            continue

        lookup_failed = bool(meta.get("lookup_failed", False))
        exists = meta.get("exists", False)
        title = meta.get("title")
        year = meta.get("year")
        error = meta.get("error")

        results.append(CitationExistence(
            citation_id=cid,
            # None rather than False when the lookup established nothing: a
            # timeout is not evidence that a paper does not exist.
            exists=None if lookup_failed else bool(exists),
            title=str(title) if title else None,
            year=int(year) if year else None,
            error=str(error) if error else None,
            lookup_failed=lookup_failed,
        ))
        if year:
            years.append(int(year))

    # The distinction is "nothing was learned" versus "the registry answered
    # no", not the presence of an `error` string. A CrossRef outage tells us
    # nothing and is excluded, and so is an identifier kind with no resolver,
    # which is never attempted; a PMID that NCBI reports as unknown is
    # precisely what this scorer exists to catch, and NCBI reports it *as* a
    # per-uid error -- so filtering on `error` dropped fabricated citations out
    # of the denominator and scored a report that invented one at 1.00.
    #
    # On the excluded side: a lookup that raised, a 200 whose body answers
    # about no identifier at all -- NCBI's `esummaryresult` envelope, a
    # rate-limit page that still parses as JSON, CrossRef's no-work-record
    # body -- and an identifier kind with no resolver, never attempted.
    #
    # Listed rather than counted. This sentence has twice been a tally that was
    # a case short within a commit, most recently asserting "three causes" and
    # naming two of them, while the alignment field's description has been
    # right for three rounds by enumerating instead: a wrong count reads as
    # authoritative, a short list reads as incomplete.
    total = len(results)
    checkable = [r for r in results if not r.lookup_failed]
    # `is True` rather than truthiness. This changes no number and no test can
    # tell the two spellings apart: the constructor above derives `exists=None`
    # from `lookup_failed` and `checkable` filters on `lookup_failed`, so
    # nothing here is ever None. Written for the reader -- `exists` is
    # three-valued, and truthiness over a three-valued field is the exact shape
    # that put `exists: false` on a real PMC article one field along.
    verified = sum(1 for r in checkable if r.exists is True)

    # Year distribution
    year_dist: dict[int, int] = {}
    for y in years:
        year_dist[y] = year_dist.get(y, 0) + 1
    # Upper middle element for an even-length list rather than the mean of the
    # two middles, so the value is always a year that actually appears. The
    # field is still called `median_year`; this is the one place that says it
    # is the upper middle rather than a true median.
    median_year = sorted(years)[len(years) // 2] if years else None

    return CitationVerifiabilityScore(
        total_citations=total,
        verified_exist=verified,
        unresolvable=total - len(checkable),
        verifiability=verified / len(checkable) if checkable else 0.0,
        year_distribution=year_dist,
        median_year=median_year,
        citations=results,
    )


async def score_citation_alignment(
    dr_output: DROutput,
    pubmed_client: httpx.AsyncClient | None = None,
) -> CitationAlignmentScore:
    """Check whether cited papers' titles share key terms with the claims they support.

    No LLM needed — uses keyword overlap between paper title and claim text.
    This catches hallucinated citations where a real PMID is paired with an
    unrelated claim (e.g., citing a cancer paper for a claim about diabetes).

    Args:
        dr_output: Parsed DR output.
        pubmed_client: Optional httpx client.

    Returns:
        CitationAlignmentScore with per-citation alignment checks.
    """
    claims = dr_output.extracted_claims
    if not claims:
        claims = extract_claims_with_citations(dr_output.raw_markdown)

    results: list[CitationAlignmentResult] = []
    unresolvable = 0

    for claim in claims:
        claim_terms = _extract_key_terms(claim.text)
        for cit in claim.citations:
            cid = cit.normalized_id

            # Fetch paper metadata
            if cid and cid.startswith("PMID:"):
                meta = await fetch_pubmed_metadata(cid, pubmed_client)
            elif cid and cid.startswith("DOI:"):
                meta = await resolve_doi(cid, pubmed_client)
            else:
                # A citation this scorer has no way to look up: an identifier
                # that would not normalise, or one of a kind it cannot resolve
                # -- the extractor also emits PMC and GEO accessions. Counted,
                # not dropped. Both were previously missing from `results` and
                # from `unresolvable` alike, so a report whose whole reference
                # list is PMC ids printed `0/0 (0.00)` with nothing
                # unresolvable -- byte-identical to a report that cited
                # nothing, which is the reading this counter exists to
                # prevent. Ways in: an identifier that would not normalise,
                # one of a kind neither scorer resolves, and a citation with
                # no identifier at all.
                #
                # The sibling agrees about the middle one -- it is
                # `lookup_failed` there too, so neither scorer treats a PMC
                # accession as a defect. It disagrees about the other two by
                # design: an identifier the report's own text mangled is a
                # finding for verifiability, because it is a property of the
                # report rather than of anyone's registry, and there is no
                # claim-title pair for alignment to have an opinion about. One
                # documented exception to "the two scorers should not disagree
                # about the same reference list", recorded here because that
                # argument is load-bearing in two commits now.
                unresolvable += 1
                continue

            title = meta.get("title")
            if meta.get("lookup_failed") or (meta.get("exists") and not title):
                # Ways to learn nothing: a failed lookup -- a timeout, a 5xx,
                # a body that answers about no uid at all -- and a record that
                # resolves but carries no title, which is a real paper with
                # nothing to align a claim against. Both leave the rate rather
                # than counting against it.
                #
                # Counted rather than dropped, because without this counter a
                # PubMed outage and a report whose citations support nothing
                # render identically. (That used to be `0/0 (0.00)` on both;
                # the CLI now says `not measured, N with nothing to align
                # against` for the outage, which it can only say because this
                # count exists.)
                unresolvable += 1
                continue

            if not title:
                # The lookup succeeded and said there is no such paper: NCBI
                # reporting an unknown uid, or CrossRef answering 404.
                # That is the authoritative negative, and a citation to a paper
                # that does not exist supports nothing -- so it counts against
                # alignment rather than leaving the rate.
                #
                # Reading this off "no title" instead of off `lookup_failed`
                # was the same inversion the verifiability scorer had one
                # commit earlier: nine real citations and one invented one
                # scored 1.00, the metric that exists to catch hallucinated
                # references calling a report that hallucinated one perfectly
                # aligned.
                results.append(CitationAlignmentResult(
                    citation_id=cid,
                    claim_text=claim.text[:200],
                    paper_title="",
                    aligned=False,
                    shared_terms=[],
                    term_overlap_score=0.0,
                ))
                continue

            title_terms = _extract_key_terms(str(title))
            shared = claim_terms & title_terms
            union = claim_terms | title_terms
            overlap = len(shared) / len(union) if union else 0.0

            # Consider aligned if at least 2 meaningful terms overlap
            # or Jaccard > 0.1 (titles are short, so low threshold is fine)
            aligned = len(shared) >= 2 or overlap > 0.1

            results.append(CitationAlignmentResult(
                citation_id=cid,
                claim_text=claim.text[:200],
                paper_title=str(title),
                aligned=aligned,
                shared_terms=sorted(shared)[:10],
                term_overlap_score=overlap,
            ))

    aligned_count = sum(1 for r in results if r.aligned)
    total = len(results)

    return CitationAlignmentScore(
        total_checked=total,
        aligned_count=aligned_count,
        unresolvable=unresolvable,
        alignment_rate=aligned_count / total if total > 0 else 0.0,
        results=results,
    )


# ---------------------------------------------------------------------------
# Rubric-driven intrinsic scorers
# ---------------------------------------------------------------------------

# Spot-check patterns and topic keyword lists used to live here as Python
# constants - including a dict of facts about exactly two named genes. That made
# these scorers useless for any third subject and impossible to extend without
# editing this module. They are supplied by the eval set now, as
# ``EvalTask.rubric``; the bundled Monarch rubrics are in ``evaluation/rubrics``.


def _rubric_of(task: EvalTask) -> Rubric:
    """Return a task's rubric, or an empty one when it has none."""
    return task.rubric or Rubric()


def _reference_claims(task: EvalTask) -> list[ReferenceClaim]:
    """Return the reference claims a task expects a report to cover."""
    return _rubric_of(task).reference_claims or []


def _most_specific(
    verdicts: list[tuple[re.Match[str], bool, str]],
) -> tuple[re.Match[str], bool, str]:
    r"""The occurrence of a non-empty list that captured the longest value.

    Used only on the *correct* occurrences of one check, where length does mean
    specificity: under `prefix` every correct capture is a prefix of the same
    `expected`, so they are prefixes of one another and the longest is the most
    precise; under `exact` they are all equal to it and the choice is
    immaterial.

    The values compared are already normalised (lowercased, thousands
    separators stripped), so length is a character count on the compared form
    -- fine for the hierarchical facts `prefix` is documented for, where a
    longer value is a longer path, and meaningless for a numeric fact, where
    18630 is not a more precise 1863. Numeric facts are `exact`, where the
    choice among equal correct captures is immaterial.

    It must not be read as a general specificity ranking. A check's occurrences
    are not all about the same fact -- the bundled pattern is
    `chromosome\s+(17...)`, not scoped to the subject gene -- so comparing an
    incorrect capture's length against a correct one compares two different
    facts' precision, which is how a correct TP53 report came to be scored
    wrong on BRCA1's longer locus. The disagreement rule next door uses
    containment for that reason.

    Requires a non-empty list rather than returning None for an empty one: both
    callers have already established that theirs is non-empty, and an Optional
    return meant a third fallback that could not fire.

    Two correct captures of one check, which is what it is for. A mixed list
    would run, since the function has no opinion about the flag -- but it would
    be an executable example of the ranking the paragraph above refuses.

    >>> import re
    >>> m = re.match("a", "a")
    >>> _most_specific([(m, True, "17"), (m, True, "17q21.31")])[2]
    '17q21.31'
    """
    return max(verdicts, key=lambda t: len(t[2]))


def _compare_capture(
    spec: SpotCheck, match: re.Match[str]
) -> tuple[bool, bool, str]:
    r"""Compare one occurrence of a spot check's pattern against its `expected`.

    Returns ``(correct, compared, captured)``. ``compared`` is False when this
    occurrence settles nothing -- a presence-only check, or a pattern that
    matched without capturing a value -- so that the accuracy rate is over the
    occurrences that actually tested something. ``captured`` is the normalised
    value that was compared, empty when nothing was, and lets the caller rank
    occurrences by how specific a claim each one made.

    "Group 1 has a non-empty captured value" is the property, and three weaker
    predicates have been wrong here. ``groups()`` is a truthy tuple of Nones
    when an optional group did not participate -- the bundled
    ``RING (finger)? domain`` case, which crashed on the commonest phrasing.
    ``lastindex`` is the index of the *last* group that matched, so
    ``(?:(17q\d+)|chromosome (17))`` gives lastindex 2 with group(1) still None
    and crashes identically. And ``is not None`` alone admits the empty string,
    which any group that can match nothing produces: ``chromosome\s*(\d*)``
    against "chromosome seventeen" captured "" and then scored a factual error
    under `exact` and a correct answer under `prefix` -- two opposite verdicts
    on one report, neither of them a measurement. ``match.re.groups`` guards the
    groupless pattern, where ``group(1)`` raises IndexError.

    >>> import re
    >>> from .datamodel import SpotCheck
    >>> spec = SpotCheck(name="n", pattern=r"(\d+)\s*amino acid", expected="1863")
    >>> _compare_capture(spec, re.search(spec.pattern, "1863 amino acids"))
    (True, True, '1863')
    >>> _compare_capture(spec, re.search(spec.pattern, "999 amino acids"))
    (False, True, '999')

    Thousands separators are removed on both sides: a report writing
    "1,863 amino acids" is not disagreeing about the number.

    >>> wide = SpotCheck(name="n", pattern=r"([\d,]+)\s*amino acid", expected="1863")
    >>> _compare_capture(wide, re.search(wide.pattern, "1,863 amino acids"))
    (True, True, '1863')

    `prefix` is for hierarchical facts -- a locus, an ontology id, a version --
    where a shorter answer is less precise rather than incorrect. A report
    saying "chromosome 17" where the answer is 17q21.31 is the commonest
    phrasing in the literature; scoring it a factual error is the mirror of the
    defect that scored the precise report wrong. A different locus is still
    wrong, which a presence-only check could not tell apart from silence.

    >>> loc = SpotCheck(name="n", pattern=r"chromosome (17[pq\d.]*)",
    ...                 expected="17q21.31", match="prefix")
    >>> _compare_capture(loc, re.search(loc.pattern, "chromosome 17"))
    (True, True, '17')
    >>> _compare_capture(loc, re.search(loc.pattern, "chromosome 17p13.1"))
    (False, True, '17p13.1')
    """
    if spec.expected is None:
        return True, False, ""

    captured = ""
    if match.re.groups and match.group(1) is not None:
        captured = match.group(1).strip().lower().replace(",", "")
    if not captured:
        return True, False, ""

    wanted = spec.expected.strip().lower().replace(",", "")
    if is_prefix_match(spec):
        return wanted.startswith(captured), True, captured
    return captured == wanted, True, captured


def score_factual_spot_checks(
    dr_output: DROutput,
    task: EvalTask,
) -> FactualSpotCheckScore:
    r"""Verify a task's spot checks against the report text.

    Each check is a regular expression from the task's rubric. A check with an
    ``expected`` value and a capturing group is an accuracy check: the captured
    text must match. A check without one only asks whether the pattern appears
    at all, which measures coverage rather than correctness - so presence and
    accuracy are reported as separate rates.

    Args:
        dr_output: Parsed DR output.
        task: The evaluation task, supplying the checks via its rubric.

    Returns:
        FactualSpotCheckScore with per-check details.

    >>> from .datamodel import AnswerType, EvalTask, Rubric, SpotCheck
    >>> from .models import DROutput
    >>> task = EvalTask(id="t", prompt="p", answer_type=AnswerType.REPORT,
    ...                 rubric=Rubric(spot_checks=[
    ...                     SpotCheck(name="length", pattern=r"(\d+)\s*amino acid", expected="1863"),
    ...                     SpotCheck(name="mentions_ring", pattern=r"RING domain"),
    ...                 ]))
    >>> out = DROutput(task_id="t", provider="mock",
    ...                raw_markdown="A 1863 amino acid protein with a RING domain.")
    >>> score = score_factual_spot_checks(out, task)
    >>> score.present_count, score.correct_count
    (2, 2)
    >>> wrong = DROutput(task_id="t", provider="mock", raw_markdown="A 999 amino acid protein.")
    >>> s2 = score_factual_spot_checks(wrong, task)
    >>> s2.present_count, s2.correct_count
    (1, 0)
    """
    text = dr_output.raw_markdown
    checks: list[FactualSpotCheck] = []

    for spec in _rubric_of(task).spot_checks or []:
        # Every occurrence, not just the first. `re.search` stops at the first
        # one, so a BRCA1 report that mentions TP53's locus before stating
        # BRCA1's own -- an ordinary thing for a report on a tumour suppressor
        # to do -- captured 17p13.1 and was scored a factual error for a fact
        # it had got right two sentences later.
        matches = list(re.finditer(spec.pattern, text, re.IGNORECASE))

        if not matches:
            present, found, correct, compared = False, None, False, False
        else:
            present = True
            verdicts = [(m, *_compare_capture(spec, m)) for m in matches]
            comparable = [(m, ok, cap) for m, ok, did, cap in verdicts if did]

            if not comparable:
                # Nothing was compared anywhere: a presence-only check, or a
                # pattern that matched without capturing. Appearing is the
                # whole test, and `compared` stays False so the accuracy rate
                # does not count agreement that was never tested.
                found, correct, compared = matches[0].group(0), True, False
            else:
                # A comparison that succeeded anywhere settles it -- except
                # against a more specific one that failed.
                #
                # Under `prefix` a correct capture may be a shorter, vaguer
                # form of the answer, and taking any correct occurrence made
                # the bundled `chromosome` check unfalsifiable: "Genes on
                # chromosome 17 include BRCA1 and TP53" captures "17", which
                # is a valid prefix of 17q21.31, so a report going on to place
                # BRCA1 at 17p13.1 scored correct. A report on a chromosome-17
                # tumour suppressor writes the bare number routinely, and this
                # is one of only two bundled checks that compare anything at
                # all -- so `compared_count` stayed 2 with one of the two
                # unable to register a disagreement.
                right = [t for t in comparable if t[1]]
                wrong = [t for t in comparable if not t[1]]
                hit = _most_specific(right) if right else None

                # Under `prefix` a correct capture may be a vaguer form of the
                # answer, so a disagreement that *extends* it is the report's
                # real claim and overrides it. `17p13.1` extends a bare `17`,
                # so a report saying "Genes on chromosome 17 include BRCA1"
                # before placing BRCA1 at 17p13.1 is wrong; `17q21.31` does
                # not extend `17p13.1`, so a TP53 report that states TP53's
                # own locus and also mentions BRCA1's stays right.
                #
                # Containment, not length. Length was the first predicate
                # tried, and it condemned whichever gene had the shorter
                # expected string: TP53 expects 17p13.1 (7) and BRCA1's
                # 17q21.31 (8) is longer, so a correct TP53 report was scored
                # a factual error with the other gene's locus printed as the
                # evidence. That is the mirror of the defect the previous
                # predicate had, and the property `prefix` is defined by --
                # "a shorter answer is less precise, a different arm is still
                # wrong" -- is about containment all along.
                #
                # Not under `exact`, where a correct capture *is* the answer
                # and nothing can extend it meaningfully.
                overrode: tuple[re.Match[str], bool, str] | None = None
                if hit is not None and is_prefix_match(spec):
                    extending = [
                        t for t in wrong if t[2] != hit[2] and t[2].startswith(hit[2])
                    ]
                    # The most specific of them, not the first in the report:
                    # with several, "the disagreement that overrode it" names
                    # nothing in particular, and the report's own claim is the
                    # most precise one it made.
                    overrode = _most_specific(extending) if extending else None
                    if overrode is not None:
                        hit = None

                # The occurrence that settled it is the one worth reporting:
                # the most specific agreement, or the disagreement that
                # overrode it. Reporting the first comparable match instead
                # showed "chromosome 17" beside a verdict of *incorrect*,
                # leaving a reader to guess which mention was judged.
                # `right` and `wrong` partition a non-empty list, and the
                # override only fires when `wrong` is non-empty -- so reaching
                # the last arm means `wrong` has something in it. Written as a
                # chain of branches rather than an `or` with a third fallback,
                # because that fallback could not fire and a reader was left
                # guessing which case it guarded.
                if hit is not None:
                    decisive = hit
                elif overrode is not None:
                    decisive = overrode
                else:
                    decisive = _most_specific(wrong)
                found = decisive[0].group(0)
                correct, compared = hit is not None, True

        checks.append(FactualSpotCheck(
            fact_name=spec.name,
            expected=spec.expected,
            found_in_report=found,
            correct=correct,
            present=present,
            compared=compared,
        ))

    present_count = sum(1 for c in checks if c.present)
    correct_count = sum(1 for c in checks if c.correct)
    total = len(checks)

    # Accuracy is over the checks that actually compared something. A
    # presence-only check is `correct` whenever it matched, so including it
    # here reports agreement that was never tested -- the same "a number where
    # there was no question" the multiple-choice path refuses.
    compared_checks = [c for c in checks if c.compared]
    compared_correct = sum(1 for c in compared_checks if c.correct)

    return FactualSpotCheckScore(
        total_checks=total,
        present_count=present_count,
        correct_count=correct_count,
        compared_count=len(compared_checks),
        presence_rate=present_count / total if total > 0 else 0.0,
        accuracy_rate=(
            compared_correct / len(compared_checks) if compared_checks else 0.0
        ),
        checks=checks,
    )


def score_topic_coverage(
    dr_output: DROutput,
    task: EvalTask,
) -> TopicCoverageScore:
    """Check whether the report addresses the topics the task's rubric expects.

    Keyword matching, so no LLM judge and no API calls. Any one keyword counts,
    which means this measures whether the report went somewhere at all rather
    than how well it covered it.

    Args:
        dr_output: Parsed DR output.
        task: The evaluation task, supplying the topics via its rubric.

    Returns:
        TopicCoverageScore with per-topic details.

    >>> from .datamodel import AnswerType, EvalTask, ExpectedTopic, Rubric
    >>> from .models import DROutput
    >>> task = EvalTask(id="t", prompt="p", answer_type=AnswerType.REPORT,
    ...                 rubric=Rubric(expected_topics=[
    ...                     ExpectedTopic(name="repair", keywords=["homologous recombination"]),
    ...                     ExpectedTopic(name="epidemiology", keywords=["prevalence"]),
    ...                 ]))
    >>> out = DROutput(task_id="t", provider="mock",
    ...                raw_markdown="BRCA1 acts in homologous recombination repair.")
    >>> score = score_topic_coverage(out, task)
    >>> score.covered_count, score.total_topics, score.coverage_rate
    (1, 2, 0.5)
    """
    text_lower = dr_output.raw_markdown.lower()
    topics: list[TopicCoverage] = []

    for spec in _rubric_of(task).expected_topics or []:
        found = [kw for kw in spec.keywords if kw.lower() in text_lower]
        snippet = None
        if found:
            index = text_lower.find(found[0].lower())
            snippet = dr_output.raw_markdown[max(0, index - 60): index + 120].strip()

        topics.append(TopicCoverage(
            topic=spec.name,
            covered=bool(found),
            evidence_snippet=snippet,
            keywords_found=found,
        ))

    covered = sum(1 for t in topics if t.covered)
    total = len(topics)

    return TopicCoverageScore(
        total_topics=total,
        covered_count=covered,
        coverage_rate=covered / total if total > 0 else 0.0,
        topics=topics,
    )


async def score_intrinsic(
    dr_output: DROutput,
    task: EvalTask,
    pubmed_client: httpx.AsyncClient | None = None,
    run_verifiability: bool = True,
    run_alignment: bool = True,
    run_spot_checks: bool = True,
    run_topic_coverage: bool = True,
) -> IntrinsicScore:
    """Run all LLM-free intrinsic quality checks on a DR output.

    This is the main entry point for intrinsic scoring. Each sub-scorer
    can be toggled independently.

    Args:
        dr_output: Parsed DR output.
        task: The evaluation task.
        pubmed_client: Optional httpx client for PubMed/CrossRef.
        run_verifiability: Check if citations resolve to real papers.
        run_alignment: Check if paper titles align with claims.
        run_spot_checks: Verify known facts by regex.
        run_topic_coverage: Check expected topic coverage.

    Returns:
        IntrinsicScore with all enabled sub-scores.
    """
    result = IntrinsicScore()

    if run_verifiability:
        result.citation_verifiability = await score_citation_verifiability(
            dr_output, pubmed_client
        )

    if run_alignment:
        result.citation_alignment = await score_citation_alignment(
            dr_output, pubmed_client
        )

    if run_spot_checks:
        result.factual_spot_checks = score_factual_spot_checks(dr_output, task)

    if run_topic_coverage:
        result.topic_coverage = score_topic_coverage(dr_output, task)

    return result
