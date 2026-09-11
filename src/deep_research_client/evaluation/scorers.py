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
from .datamodel import EvalTask, ReferenceClaim, Rubric
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


def _extract_json_object(text: str) -> dict | None:
    """Extract the first JSON object from text, handling nested braces.

    >>> _extract_json_object('blah {"a": 1, "b": {"c": 2}} done')
    {'a': 1, 'b': {'c': 2}}
    >>> _extract_json_object('no json here') is None
    True
    >>> _extract_json_object('{"supported": true, "explanation": "yes"}')
    {'supported': True, 'explanation': 'yes'}
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


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
                result = _extract_json_object(result_text)
                if result:
                    supported = result.get("supported", False)
                    explanation = result.get("explanation", "")
                else:
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
    checkable = [v for v in verifications if v.supported is not None]
    total = len(checkable)
    verified = sum(1 for v in checkable if v.supported)

    return FACTScore(
        total_citations=total,
        verified_citations=verified,
        citation_accuracy=verified / total if total > 0 else 0.0,
        effective_citations=verified,
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
        return ClaimRecallScore(
            total_ground_truth_claims=0, matched_claims=0, claim_recall=0.0
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
            result = _extract_json_object(result_text)
            if result:
                matched = result.get("matched", False)
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
            result = _extract_json_object(result_text)
            if result and result.get("score") is not None:
                score = min(max(float(result["score"]), 1.0), 5.0)
                explanation = result.get("explanation", "")
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

        result_data = data.get("result", {}).get(numeric_id, {})
        if "error" in result_data:
            return {"exists": False, "title": None, "year": None, "error": result_data["error"]}

        title = result_data.get("title", "")
        pubdate = result_data.get("pubdate", "")
        year = None
        if pubdate:
            year_match = re.search(r"(\d{4})", pubdate)
            if year_match:
                year = int(year_match.group(1))

        return {"exists": bool(title), "title": title, "year": year}
    except Exception as e:
        logger.warning("Failed to fetch PubMed metadata for %s: %s", pmid, e)
        return {"exists": False, "title": None, "year": None, "error": str(e)}


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
            return {"exists": False, "title": None, "year": None}
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        title_list = message.get("title", [])
        title = title_list[0] if title_list else None

        year = None
        published = message.get("published", {})
        date_parts = published.get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        return {"exists": bool(title), "title": title, "year": year}
    except Exception as e:
        logger.warning("Failed to resolve DOI %s: %s", doi, e)
        return {"exists": False, "title": None, "year": None, "error": str(e)}


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
            results.append(CitationExistence(
                citation_id=cit.raw_reference, exists=False, error="Could not normalize"
            ))
            continue

        if cid.startswith("PMID:"):
            meta = await fetch_pubmed_metadata(cid, pubmed_client)
        elif cid.startswith("DOI:"):
            meta = await resolve_doi(cid, pubmed_client)
        else:
            results.append(CitationExistence(
                citation_id=cid, exists=False, error="Unknown citation type"
            ))
            continue

        exists = meta.get("exists", False)
        title = meta.get("title")
        year = meta.get("year")
        error = meta.get("error")

        results.append(CitationExistence(
            citation_id=cid,
            exists=bool(exists),
            title=str(title) if title else None,
            year=int(year) if year else None,
            error=str(error) if error else None,
        ))
        if year:
            years.append(int(year))

    # A lookup that errored is not a citation that does not exist: a CrossRef
    # outage used to report every DOI in a report as hallucinated. Excluded
    # from the rate, and counted so the number can say what it did not cover.
    total = len(results)
    checkable = [r for r in results if r.error is None]
    verified = sum(1 for r in checkable if r.exists)

    # Year distribution
    year_dist: dict[int, int] = {}
    for y in years:
        year_dist[y] = year_dist.get(y, 0) + 1
    # Upper middle element for an even-length list rather than the mean of the
    # two middles. Kept as-is -- a publication year should be a year that
    # exists -- but named so the field is not read as a true median.
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

    for claim in claims:
        claim_terms = _extract_key_terms(claim.text)
        for cit in claim.citations:
            cid = cit.normalized_id
            if not cid:
                continue

            # Fetch paper metadata
            if cid.startswith("PMID:"):
                meta = await fetch_pubmed_metadata(cid, pubmed_client)
            elif cid.startswith("DOI:"):
                meta = await resolve_doi(cid, pubmed_client)
            else:
                continue

            title = meta.get("title")
            if not title:
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
        match = re.search(spec.pattern, text, re.IGNORECASE)

        if match is None:
            present, found, correct = False, None, False
        else:
            present, found = True, match.group(0)
            compared = False
            if spec.expected is None:
                # Presence-only check: appearing is the whole test.
                correct = True
            elif match.lastindex:
                # `lastindex` is None when no group participated, where
                # `groups()` would be a truthy tuple of Nones -- and an optional
                # group that did not participate has no captured value to
                # compare. `\bRING\s*(finger)?\s*domain\b` against "a RING
                # domain" is the bundled case: groups() is (None,), which is
                # truthy, so this branch ran and `.strip()` raised on None.
                # That crash reached score_intrinsic, whose `except Exception`
                # discarded all four intrinsic scores for a correct report.
                captured = match.group(1)
                correct = captured.strip().lower() == spec.expected.strip().lower()
                compared = True
            else:
                # A pattern with no participating group cannot disagree with
                # `expected`; appearing is the whole test, as above.
                correct = True

        checks.append(FactualSpotCheck(
            fact_name=spec.name,
            expected=spec.expected or "",
            found_in_report=found,
            correct=correct,
            present=present,
            compared=compared if present else False,
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
