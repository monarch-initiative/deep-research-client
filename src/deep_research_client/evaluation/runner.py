"""Evaluation runner: orchestrates task generation, DR execution, and scoring.

The main entry point is ``run_evaluation``, which:
1. Loads ground truth from dismech and/or ai-gene-review repos
2. Generates evaluation tasks
3. Runs each task through specified DR providers
4. Scores the outputs with FACT, claim recall, and RACE
5. Returns structured results
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .loaders import load_dismech_entity, load_dismech_repo, load_gene_review_entity, load_gene_review_repo
from .models import DROutput, EvalResult, EvalTask, GroundTruthEntity
from .scorers import (
    extract_claims_with_citations,
    extract_citations_from_markdown,
    score_claim_recall,
    score_fact,
    score_intrinsic,
    score_race,
)
from .tasks import generate_tasks

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    dismech_dir: Path | None = None
    gene_review_dir: Path | None = None
    entity_names: list[str] = field(default_factory=list)
    entity_files: list[Path] = field(default_factory=list)
    max_entities: int | None = None
    run_fact: bool = True
    run_claim_recall: bool = True
    run_race: bool = True
    run_intrinsic: bool = True
    gene_symbol: str | None = None
    llm_model: str = "gpt-4o-mini"


def load_entities(config: EvalConfig) -> list[GroundTruthEntity]:
    """Load ground truth entities based on config.

    Args:
        config: Evaluation configuration specifying data sources.

    Returns:
        List of GroundTruthEntity objects to evaluate against.
    """
    entities: list[GroundTruthEntity] = []

    # Load from specific files
    for fpath in config.entity_files:
        if not fpath.exists():
            logger.warning("File not found: %s", fpath)
            continue
        if "disorders" in str(fpath) or (fpath.name.endswith(".yaml") and "ai-review" not in fpath.name):
            entities.append(load_dismech_entity(fpath))
        else:
            entities.append(load_gene_review_entity(fpath))

    # Load from repos
    if config.dismech_dir:
        entities.extend(load_dismech_repo(config.dismech_dir))
    if config.gene_review_dir:
        entities.extend(load_gene_review_repo(config.gene_review_dir))

    # Filter by name if specified
    if config.entity_names:
        name_set = {n.lower() for n in config.entity_names}
        entities = [e for e in entities if e.name.lower() in name_set]

    # Limit
    if config.max_entities is not None:
        entities = entities[: config.max_entities]

    logger.info("Loaded %d entities for evaluation", len(entities))
    return entities


def generate_all_tasks(entities: list[GroundTruthEntity]) -> list[EvalTask]:
    """Generate evaluation tasks for all entities.

    Args:
        entities: Ground truth entities.

    Returns:
        List of EvalTask objects.
    """
    tasks = []
    for entity in entities:
        tasks.extend(generate_tasks(entity))
    logger.info("Generated %d evaluation tasks from %d entities", len(tasks), len(entities))
    return tasks


def parse_dr_output(task: EvalTask, markdown: str, provider: str, model: str | None = None) -> DROutput:
    """Parse raw markdown DR output into a structured DROutput.

    Args:
        task: The evaluation task.
        markdown: Raw markdown output from the DR tool.
        provider: Name of the DR provider.
        model: Model name if known.

    Returns:
        DROutput with extracted claims and citations.
    """
    return DROutput(
        task_id=task.task_id,
        provider=provider,
        model=model,
        raw_markdown=markdown,
        extracted_claims=extract_claims_with_citations(markdown),
        extracted_citations=extract_citations_from_markdown(markdown),
    )


async def score_output(
    dr_output: DROutput,
    task: EvalTask,
    llm_client: Any,
    config: EvalConfig,
    pubmed_client: httpx.AsyncClient | None = None,
) -> EvalResult:
    """Score a single DR output against its task's ground truth.

    Args:
        dr_output: Parsed DR output.
        task: The evaluation task with ground truth claims.
        llm_client: OpenAI-compatible async client for LLM judging.
        config: Evaluation config (controls which scorers to run).
        pubmed_client: Optional httpx client for PubMed API.

    Returns:
        EvalResult with FACT, claim recall, and RACE scores.
    """
    result = EvalResult(
        task_id=dr_output.task_id,
        provider=dr_output.provider,
        model=dr_output.model,
        duration_seconds=dr_output.duration_seconds,
    )

    try:
        if config.run_fact:
            result.fact_score = await score_fact(dr_output, llm_client, pubmed_client, model=config.llm_model)
            logger.info(
                "FACT score for %s/%s: accuracy=%.2f, effective=%d",
                task.task_id,
                dr_output.provider,
                result.fact_score.citation_accuracy,
                result.fact_score.effective_citations,
            )
    except Exception as e:
        logger.error("FACT scoring failed for %s: %s", task.task_id, e)
        result.error = f"FACT error: {e}"

    try:
        if config.run_claim_recall:
            result.claim_recall_score = await score_claim_recall(
                dr_output, task.ground_truth_claims, llm_client, model=config.llm_model
            )
            logger.info(
                "Claim recall for %s/%s: %.2f (%d/%d)",
                task.task_id,
                dr_output.provider,
                result.claim_recall_score.claim_recall,
                result.claim_recall_score.matched_claims,
                result.claim_recall_score.total_ground_truth_claims,
            )
    except Exception as e:
        logger.error("Claim recall scoring failed for %s: %s", task.task_id, e)
        if result.error:
            result.error += f"; Claim recall error: {e}"
        else:
            result.error = f"Claim recall error: {e}"

    try:
        if config.run_race:
            result.race_score = await score_race(dr_output, task, llm_client, model=config.llm_model)
            logger.info(
                "RACE score for %s/%s: %.2f",
                task.task_id,
                dr_output.provider,
                result.race_score.overall_score,
            )
    except Exception as e:
        logger.error("RACE scoring failed for %s: %s", task.task_id, e)
        if result.error:
            result.error += f"; RACE error: {e}"
        else:
            result.error = f"RACE error: {e}"

    try:
        if config.run_intrinsic:
            result.intrinsic_score = await score_intrinsic(
                dr_output, task,
                gene_symbol=config.gene_symbol,
                pubmed_client=pubmed_client,
            )
            if result.intrinsic_score.citation_verifiability:
                cv = result.intrinsic_score.citation_verifiability
                logger.info(
                    "Citation verifiability for %s/%s: %.2f (%d/%d exist)",
                    task.task_id, dr_output.provider,
                    cv.verifiability, cv.verified_exist, cv.total_citations,
                )
            if result.intrinsic_score.topic_coverage:
                tc = result.intrinsic_score.topic_coverage
                logger.info(
                    "Topic coverage for %s/%s: %.2f (%d/%d topics)",
                    task.task_id, dr_output.provider,
                    tc.coverage_rate, tc.covered_count, tc.total_topics,
                )
    except Exception as e:
        logger.error("Intrinsic scoring failed for %s: %s", task.task_id, e)
        if result.error:
            result.error += f"; Intrinsic error: {e}"
        else:
            result.error = f"Intrinsic error: {e}"

    return result
