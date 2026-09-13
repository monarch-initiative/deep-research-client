"""Evaluation runner: loads eval sets and scores outputs against them.

The runner is deliberately thin and knows nothing about any particular
benchmark. It asks an adapter for an :class:`EvalSet`, and it asks the scorers
appropriate to each task's ``answer_type`` to grade an output. Adding a
benchmark or a scoring dimension therefore does not touch this module.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .adapters import get_adapter
from .datamodel import AnswerType, EvalSet, EvalTask
from .models import DROutput, EvalResult
from .scorers import (
    extract_claims_with_citations,
    extract_citations_from_markdown,
    score_claim_recall,
    score_fact,
    score_intrinsic,
    score_race,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Which scorers to run, and how.

    Deliberately free of any benchmark-specific fields. An earlier version
    carried a ``gene_symbol``, which is the kind of thing that belongs on a task
    rather than on a run.
    """

    run_fact: bool = True
    run_claim_recall: bool = True
    run_race: bool = True
    run_intrinsic: bool = True
    llm_model: str = "gpt-4o-mini"


def load_eval_set(adapter_name: str, source: str | Path, **options: Any) -> EvalSet:
    """Load an eval set through a named adapter.

    Args:
        adapter_name: Registered adapter name, e.g. ``lab-bench`` or ``yaml``.
        source: Whatever that adapter takes as its source.
        **options: Adapter-specific options.

    Returns:
        The loaded EvalSet.
    """
    eval_set = get_adapter(adapter_name).load(source, **options)
    logger.info(
        "Loaded eval set %r: %d tasks from %s",
        eval_set.name, len(eval_set.tasks or []), eval_set.source,
    )
    return eval_set


def parse_dr_output(
    task: EvalTask, markdown: str, provider: str, model: str | None = None
) -> DROutput:
    """Parse raw markdown output into a structured DROutput.

    Args:
        task: The evaluation task.
        markdown: Raw markdown output from the research tool.
        provider: Name of the provider.
        model: Model name if known.

    Returns:
        DROutput with extracted claims and citations.
    """
    return DROutput(
        task_id=task.id,
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
    """Score one report-shaped output against its task.

    Each scorer is run independently and its failure recorded rather than
    raised, so that one unreachable API does not discard the dimensions that did
    compute.

    Args:
        dr_output: Parsed output.
        task: The evaluation task, carrying the rubric.
        llm_client: OpenAI-compatible async client for LLM judging.
        config: Which scorers to run.
        pubmed_client: Optional httpx client for PubMed.

    Returns:
        EvalResult with whichever scores succeeded.
    """
    if task.answer_type != AnswerType.REPORT:
        raise ValueError(
            f"Task {task.id!r} has answer_type={task.answer_type}; "
            f"report scorers apply only to REPORT tasks. Use "
            f"deep_research_client.evaluation.mcq for multiple-choice tasks."
        )

    result = EvalResult(
        task_id=dr_output.task_id,
        provider=dr_output.provider,
        model=dr_output.model,
        duration_seconds=dr_output.duration_seconds,
    )
    errors: list[str] = []

    async def _run(label: str, enabled: bool, coro_factory: Any) -> Any:
        if not enabled:
            return None
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 - one scorer must not sink the rest
            logger.error("%s scoring failed for %s: %s", label, task.id, exc)
            errors.append(f"{label} error: {exc}")
            return None

    result.fact_score = await _run(
        "FACT", config.run_fact,
        lambda: score_fact(dr_output, llm_client, pubmed_client, model=config.llm_model),
    )
    result.claim_recall_score = await _run(
        "Claim recall", config.run_claim_recall,
        lambda: score_claim_recall(
            dr_output,
            (task.rubric.reference_claims if task.rubric else []) or [],
            llm_client,
            model=config.llm_model,
        ),
    )
    result.race_score = await _run(
        "RACE", config.run_race,
        lambda: score_race(dr_output, task, llm_client, model=config.llm_model),
    )
    result.intrinsic_score = await _run(
        "Intrinsic", config.run_intrinsic,
        lambda: score_intrinsic(dr_output, task, pubmed_client=pubmed_client),
    )

    if errors:
        result.error = "; ".join(errors)
    return result
