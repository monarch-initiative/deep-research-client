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
    CitationVerification,
    ClaimMatch,
    ClaimRecallScore,
    DROutput,
    EvalTask,
    ExtractedCitation,
    ExtractedClaim,
    FACTScore,
    GroundTruthClaim,
    RACEDimension,
    RACEScore,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Citation extraction helpers
# ---------------------------------------------------------------------------

# Patterns for PMID and DOI references in markdown text
_PMID_PATTERN = re.compile(r"PMID[:\s]*(\d{6,9})", re.IGNORECASE)
_DOI_PATTERN = re.compile(r"(?:doi[:\s]*|https?://doi\.org/)(10\.\d{4,}/\S+)", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


def extract_citations_from_markdown(markdown: str) -> list[ExtractedCitation]:
    """Extract all PMID and DOI citations from markdown text.

    >>> cits = extract_citations_from_markdown("This was shown (PMID:7913883) and confirmed (DOI:10.1038/ng1234).")
    >>> [c.normalized_id for c in cits]
    ['PMID:7913883', 'DOI:10.1038/ng1234']
    >>> cits2 = extract_citations_from_markdown("See https://pubmed.ncbi.nlm.nih.gov/12345678")
    >>> [c.normalized_id for c in cits2]
    ['PMID:12345678']
    """
    seen: set[str] = set()
    citations: list[ExtractedCitation] = []

    for match in _PMID_PATTERN.finditer(markdown):
        pmid = f"PMID:{match.group(1)}"
        if pmid not in seen:
            seen.add(pmid)
            citations.append(ExtractedCitation(raw_reference=match.group(0), normalized_id=pmid))

    for match in _URL_PATTERN.finditer(markdown):
        pmid = f"PMID:{match.group(1)}"
        if pmid not in seen:
            seen.add(pmid)
            citations.append(ExtractedCitation(raw_reference=match.group(0), normalized_id=pmid, url=match.group(0)))

    for match in _DOI_PATTERN.finditer(markdown):
        doi = f"DOI:{match.group(1).rstrip('.,;)')}"
        if doi not in seen:
            seen.add(doi)
            citations.append(ExtractedCitation(raw_reference=match.group(0), normalized_id=doi))

    return citations


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


async def _llm_judge(prompt: str, llm_client: Any) -> str:
    """Call an LLM to judge/evaluate. Expects an OpenAI-compatible client.

    Args:
        prompt: The evaluation prompt.
        llm_client: An openai.AsyncOpenAI-compatible client.

    Returns:
        The LLM response text.
    """
    response = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
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
                f"CITED PAPER ABSTRACT:\n{abstract[:3000]}\n\n"
                "Does this abstract support the claim? Respond with a JSON object:\n"
                '{"supported": true/false, "explanation": "brief explanation"}'
            )
            try:
                result_text = await _llm_judge(prompt, llm_client)
                # Parse JSON from response
                json_match = re.search(r"\{[^}]+\}", result_text)
                if json_match:
                    result = json.loads(json_match.group())
                    supported = result.get("supported", False)
                    explanation = result.get("explanation", "")
                else:
                    supported = "true" in result_text.lower()[:50]
                    explanation = result_text[:200]

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
    ground_truth_claims: list[GroundTruthClaim],
    llm_client: Any,
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
    report_text = dr_output.raw_markdown[:12000]

    matches: list[ClaimMatch] = []
    for gt_claim in ground_truth_claims:
        prompt = (
            "You are evaluating whether a research report covers a specific claim.\n\n"
            f"GROUND TRUTH CLAIM:\n"
            f"Category: {gt_claim.category}\n"
            f"Name: {gt_claim.name}\n"
            f"Description: {gt_claim.description[:500]}\n\n"
            f"RESEARCH REPORT (excerpt):\n{report_text}\n\n"
            "Does the research report contain information that substantially covers "
            "this ground truth claim? The report does not need to use identical wording, "
            "but should convey the same key facts.\n\n"
            "Respond with a JSON object:\n"
            '{"matched": true/false, "best_matching_text": "quote from report or null", '
            '"explanation": "brief explanation"}'
        )
        try:
            result_text = await _llm_judge(prompt, llm_client)
            json_match = re.search(r"\{[^}]+\}", result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                matched = result.get("matched", False)
                best_text = result.get("best_matching_text")
                explanation = result.get("explanation", "")
            else:
                matched = "true" in result_text.lower()[:50]
                best_text = None
                explanation = result_text[:200]

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
                    matched=False,
                    explanation=f"Error: {e}",
                )
            )

    matched_count = sum(1 for m in matches if m.matched)
    total = len(ground_truth_claims)

    return ClaimRecallScore(
        total_ground_truth_claims=total,
        matched_claims=matched_count,
        claim_recall=matched_count / total if total > 0 else 0.0,
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
    report_text = dr_output.raw_markdown[:12000]

    # Build ground truth summary for the judge
    gt_summary_parts = []
    for claim in task.ground_truth_claims[:20]:
        terms_str = ", ".join(f"{t.id} ({t.label})" for t in claim.ontology_terms[:3])
        gt_summary_parts.append(
            f"- [{claim.category}] {claim.name}: {claim.description[:150]}"
            + (f" (Terms: {terms_str})" if terms_str else "")
        )
    gt_summary = "\n".join(gt_summary_parts) if gt_summary_parts else "No specific ground truth provided."

    dimensions: list[RACEDimension] = []
    for dim_name, dim_description in _RACE_DIMENSIONS:
        prompt = (
            "You are an expert scientific evaluator assessing a deep research report.\n\n"
            f"TASK QUERY: {task.query}\n\n"
            f"KNOWN GROUND TRUTH (key claims that should be covered):\n{gt_summary}\n\n"
            f"RESEARCH REPORT:\n{report_text}\n\n"
            f"EVALUATION DIMENSION: {dim_name}\n"
            f"CRITERIA: {dim_description}\n\n"
            "Score the report on this dimension from 1 (very poor) to 5 (excellent).\n"
            "Respond with a JSON object:\n"
            '{"score": <1-5>, "explanation": "brief justification"}'
        )
        try:
            result_text = await _llm_judge(prompt, llm_client)
            json_match = re.search(r"\{[^}]+\}", result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get("score", 3))
                explanation = result.get("explanation", "")
            else:
                score = 3.0
                explanation = result_text[:200]

            dimensions.append(
                RACEDimension(
                    dimension=dim_name,
                    score=min(max(score, 1.0), 5.0),
                    max_score=5.0,
                    explanation=explanation,
                )
            )
        except Exception as e:
            logger.warning("Failed to score dimension %s: %s", dim_name, e)
            dimensions.append(
                RACEDimension(dimension=dim_name, score=3.0, max_score=5.0, explanation=f"Error: {e}")
            )

    return RACEScore(dimensions=dimensions)
