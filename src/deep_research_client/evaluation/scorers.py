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
    EvalTask,
    ExtractedCitation,
    ExtractedClaim,
    FACTScore,
    FactualSpotCheck,
    FactualSpotCheckScore,
    GroundTruthClaim,
    IntrinsicScore,
    RACEDimension,
    RACEScore,
    TopicCoverage,
    TopicCoverageScore,
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

    >>> import asyncio
    >>> result = asyncio.run(fetch_pubmed_metadata("PMID:7913883"))
    >>> result["exists"]
    True
    >>> "FGFR3" in result.get("title", "")
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

    verified = sum(1 for r in results if r.exists)
    total = len(results)

    # Year distribution
    year_dist: dict[int, int] = {}
    for y in years:
        year_dist[y] = year_dist.get(y, 0) + 1
    median_year = sorted(years)[len(years) // 2] if years else None

    return CitationVerifiabilityScore(
        total_citations=total,
        verified_exist=verified,
        verifiability=verified / total if total > 0 else 0.0,
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
# Factual spot-check definitions
# ---------------------------------------------------------------------------

# Each spot check is: (fact_name, pattern_to_find_in_text, expected_value_pattern)
# These are checked by regex against the DR output text.

_GENE_SPOT_CHECKS: dict[str, list[tuple[str, str, str]]] = {
    "BRCA1": [
        ("chromosome", r"chromosome\s+(17[qp]?\d*\.?\d*)", "17"),
        ("protein_length", r"(\d{3,4})\s*amino\s*acid", "1863"),
        ("gene_symbol", r"\bBRCA1\b", "BRCA1"),
        ("partner_protein", r"\bBARD1\b", "BARD1"),
        ("repair_pathway", r"\bhomologous\s+recombination\b", "homologous recombination"),
        ("ring_domain", r"\bRING\s*(finger)?\s*domain\b", "RING domain"),
        ("brct_domain", r"\bBRCT\b", "BRCT"),
        ("e3_ligase", r"\bE3\s*ubiquitin\s*ligase\b", "E3 ubiquitin ligase"),
        ("rad51_interaction", r"\bRAD51\b", "RAD51"),
        ("cancer_type", r"\b(breast|ovarian)\s+cancer\b", "breast/ovarian cancer"),
        ("parp_inhibitor", r"\bPARP\s*inhibitor", "PARP inhibitor"),
    ],
    "TP53": [
        ("chromosome", r"chromosome\s+(17[qp]?\d*\.?\d*)", "17"),
        ("gene_symbol", r"\bTP53\b", "TP53"),
        ("protein_name", r"\bp53\b", "p53"),
        ("tumor_suppressor", r"\btumor\s+suppressor\b", "tumor suppressor"),
        ("apoptosis", r"\bapoptosis\b", "apoptosis"),
        ("cell_cycle", r"\bcell\s+cycle\b", "cell cycle"),
    ],
}

# Generic checks that apply to any gene
_GENERIC_GENE_SPOT_CHECKS: list[tuple[str, str | None, str | None]] = [
    ("has_gene_symbol", None, None),  # Special: just checks gene name is mentioned
    ("has_protein_function", r"\b(catalytic|binding|signaling|kinase|ligase|transferase|receptor)\b", "any enzymatic/binding term"),
    ("has_cellular_location", r"\b(nucleus|cytoplasm|membrane|mitochondria|endoplasmic|golgi)\b", "any subcellular location"),
    ("has_pathway_context", r"\b(pathway|signaling|cascade|network)\b", "any pathway term"),
    ("has_disease_association", r"\b(disease|disorder|syndrome|cancer|mutation|pathogenic)\b", "any disease term"),
]

# Expected topics for different report types
_GENE_FUNCTION_TOPICS: list[tuple[str, list[str]]] = [
    ("molecular_function", ["catalytic", "binding", "enzyme", "kinase", "ligase", "transferase", "receptor", "transporter"]),
    ("biological_process", ["signaling", "repair", "replication", "transcription", "translation", "apoptosis", "cell cycle", "differentiation", "proliferation"]),
    ("protein_structure", ["domain", "motif", "fold", "terminus", "residue", "helix", "sheet", "conformation"]),
    ("protein_interactions", ["interact", "complex", "partner", "binding partner", "hetero", "homo", "recruit", "associate"]),
    ("subcellular_localization", ["nucleus", "cytoplasm", "membrane", "foci", "compartment", "localization", "nuclear", "cytoplasmic"]),
    ("regulation", ["phosphorylation", "ubiquitin", "acetylation", "methylation", "regulation", "post-translational", "modification"]),
    ("disease_relevance", ["cancer", "disease", "pathogenic", "mutation", "variant", "clinical", "therapeutic", "hereditary"]),
    ("model_organisms", ["mouse", "yeast", "drosophila", "zebrafish", "model organism", "knockout", "homolog"]),
]

_DISEASE_MECHANISM_TOPICS: list[tuple[str, list[str]]] = [
    ("genetic_basis", ["gene", "mutation", "variant", "allele", "inheritance", "autosomal", "dominant", "recessive"]),
    ("molecular_mechanism", ["protein", "pathway", "signaling", "receptor", "enzyme", "binding"]),
    ("cellular_effects", ["cell", "proliferation", "apoptosis", "differentiation", "migration"]),
    ("clinical_features", ["phenotype", "symptom", "presentation", "diagnosis", "severity"]),
    ("treatment", ["therapy", "treatment", "drug", "inhibitor", "intervention", "clinical trial"]),
    ("epidemiology", ["prevalence", "incidence", "population", "risk", "frequency"]),
]


def score_factual_spot_checks(
    dr_output: DROutput,
    gene_symbol: str | None = None,
) -> FactualSpotCheckScore:
    """Run factual spot-checks against the DR output.

    Verifies specific, objectively correct facts (chromosome location,
    protein domains, key interaction partners) by regex matching.
    No LLM or API calls needed.

    Args:
        dr_output: Parsed DR output.
        gene_symbol: Gene symbol to use gene-specific checks. Falls back to generic.

    Returns:
        FactualSpotCheckScore with per-check details.
    """
    text = dr_output.raw_markdown
    checks: list[FactualSpotCheck] = []

    # Use gene-specific checks if available
    specific_checks = _GENE_SPOT_CHECKS.get(gene_symbol or "", [])
    for fact_name, pattern, expected in specific_checks:
        match = re.search(pattern, text, re.IGNORECASE)
        present = match is not None
        found_value = match.group(0) if match else None

        # For some checks we just care about presence, not exact value
        correct = present  # conservative: if present, assume correct for keyword checks
        if fact_name == "protein_length" and match:
            # Specific numeric check
            correct = match.group(1) == expected

        checks.append(FactualSpotCheck(
            fact_name=fact_name,
            expected=expected,
            found_in_report=found_value,
            correct=correct,
            present=present,
        ))

    # Generic checks for any gene
    for gen_name, gen_pattern, gen_expected in _GENERIC_GENE_SPOT_CHECKS:
        if gen_name == "has_gene_symbol" and gene_symbol:
            present = gene_symbol.upper() in text.upper()
            checks.append(FactualSpotCheck(
                fact_name=gen_name,
                expected=gene_symbol,
                found_in_report=gene_symbol if present else None,
                correct=present,
                present=present,
            ))
        elif gen_pattern:
            match = re.search(gen_pattern, text, re.IGNORECASE)
            checks.append(FactualSpotCheck(
                fact_name=gen_name,
                expected=gen_expected or "",
                found_in_report=match.group(0) if match else None,
                correct=match is not None,
                present=match is not None,
            ))

    present_count = sum(1 for c in checks if c.present)
    correct_count = sum(1 for c in checks if c.correct)
    total = len(checks)

    return FactualSpotCheckScore(
        total_checks=total,
        present_count=present_count,
        correct_count=correct_count,
        presence_rate=present_count / total if total > 0 else 0.0,
        accuracy_rate=correct_count / present_count if present_count > 0 else 0.0,
        checks=checks,
    )


def score_topic_coverage(
    dr_output: DROutput,
    task: EvalTask,
) -> TopicCoverageScore:
    """Check whether the report covers expected topics for the task type.

    Uses keyword matching against predefined topic lists. No LLM needed.
    Different topic lists are used for gene function vs disease mechanism tasks.

    Args:
        dr_output: Parsed DR output.
        task: The evaluation task (determines which topic list to use).

    Returns:
        TopicCoverageScore with per-topic details.
    """
    from .models import TaskType

    text_lower = dr_output.raw_markdown.lower()

    # Select topic list based on task type
    if task.task_type in (TaskType.GENE_FUNCTION, TaskType.GENE_DISEASE_LINK):
        topic_defs = _GENE_FUNCTION_TOPICS
    else:
        topic_defs = _DISEASE_MECHANISM_TOPICS

    topics: list[TopicCoverage] = []
    for topic_name, keywords in topic_defs:
        found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
        covered = len(found_keywords) >= 1

        # Find a snippet as evidence
        snippet = None
        if found_keywords:
            # Find the first occurrence in text
            kw = found_keywords[0]
            idx = text_lower.find(kw.lower())
            if idx >= 0:
                start = max(0, idx - 40)
                end = min(len(dr_output.raw_markdown), idx + len(kw) + 60)
                snippet = dr_output.raw_markdown[start:end].strip()

        topics.append(TopicCoverage(
            topic=topic_name,
            covered=covered,
            evidence_snippet=snippet,
            keywords_found=found_keywords,
        ))

    covered_count = sum(1 for t in topics if t.covered)
    total = len(topics)

    return TopicCoverageScore(
        total_topics=total,
        covered_count=covered_count,
        coverage_rate=covered_count / total if total > 0 else 0.0,
        topics=topics,
    )


async def score_intrinsic(
    dr_output: DROutput,
    task: EvalTask,
    gene_symbol: str | None = None,
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
        gene_symbol: Gene symbol for gene-specific spot checks.
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
        result.factual_spot_checks = score_factual_spot_checks(
            dr_output, gene_symbol
        )

    if run_topic_coverage:
        result.topic_coverage = score_topic_coverage(dr_output, task)

    return result
