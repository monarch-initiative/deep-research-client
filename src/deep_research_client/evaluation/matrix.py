"""The matrix runner: every task in an eval set, under every arm.

One run is a grid of cells. Each cell is one task sent to one arm, and each
cell records its own outcome, so an arm that runs out of quota halfway through
costs you that arm's remaining cells and nothing else. Cells are written to disk
as they complete rather than at the end, which means an interrupted run is
resumable and a long one can be inspected while it is still going.

What a run produces is the record: for every cell, exactly what the provider was
sent and exactly what it returned, on disk, addressable. Scoring is deliberately
not part of that. A run is worth keeping whether or not anyone has yet decided
how to grade it, and a grading method that changes should never require paying
for the outputs again.

Provisional grading of multiple-choice cells is available behind ``grade``, off
by default. It reads the chosen option out of the response with regular
expressions, which is a stopgap and not the intended design - see
``evaluation/mcq.py``.

The output layout is fixed and predictable:

    <run_dir>/
      manifest.json          the run: arms, dataset revision, every cell
      results.tsv            one row per cell, for loading into anything
      scores.tsv             per-arm aggregates, for multiple-choice runs
      <task_id>/
        <arm_id>/
          prompt.md          exactly what the provider was sent
          output.md          exactly what it returned
          cell.json          the cell record, including any graded answer
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from .. import __version__
from ..client import DeepResearchClient
from ..models import CacheConfig
from . import mcq
from ._fs import atomic_write
from .datamodel import (
    AnswerType,
    ArmSpec,
    CellResult,
    CellStatus,
    EvalSet,
    EvalTask,
    MetadataItem,
    RunManifest,
    ScoreDisposition,
)
from .models import MCQAnswer, MCQScore

logger = logging.getLogger(__name__)

#: Characters allowed in a path segment derived from an id.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Columns of results.tsv, in order.
_RESULT_COLUMNS = (
    "task_id", "arm_id", "provider", "model", "status", "disposition",
    "correct", "chosen_letter", "duration_seconds", "citation_count",
    "resumed", "cached", "output_path", "error",
)


def safe_segment(value: str) -> str:
    """Make a string safe to use as one path segment.

    Task and arm ids come from benchmark data, so they can be anything at all.
    Leading dots are stripped as well as separators: replacing only the unsafe
    characters leaves ``..`` intact, and a task id of ``..`` would then write a
    run's output into the run directory's parent.

    An id that needs no rewriting is used as-is; one that does gets a digest of
    the original appended, so a rewrite can never collide with another id.

    >>> safe_segment("LitQA2__e3b5/a4af").startswith("LitQA2__e3b5_a4af-")
    True
    >>> safe_segment("plain_id")
    'plain_id'
    >>> safe_segment("MONDO:0007037") != safe_segment("MONDO_0007037")
    True
    >>> safe_segment("MONDO:0007037").startswith("MONDO_0007037-")
    True
    >>> safe_segment("..").startswith("unnamed-")
    True
    >>> safe_segment("../escape").startswith("escape-")
    True
    """
    cleaned = _UNSAFE.sub("_", value).strip("_")
    while cleaned.startswith("."):
        cleaned = cleaned.lstrip(".").strip("_")
    if not cleaned:
        cleaned = "unnamed"
    if cleaned == value:
        return cleaned
    # Sanitising is lossy, so two distinct ids can arrive at one directory -
    # "MONDO:0007037" and "MONDO_0007037" both become "MONDO_0007037", and the
    # second cell would overwrite the first's output and, on resume, report the
    # first's result as its own. A digest of the original keeps rewritten ids
    # apart from each other and from any id that needed no rewriting.
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned}-{digest}"


def load_arms(path: str | Path) -> list[ArmSpec]:
    """Read arm definitions from a YAML file.

    The file is a mapping with an ``arms`` list, or a bare list::

        arms:
          - id: edison
            provider: falcon
          - id: baseline-agent
            provider: claude_code
            description: Plain agent with web search, as a control
          - id: baseline-noweb
            provider: claude_code
            description: Closed-book control, to probe contamination
            params:
              allowed_tools: []

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed arms.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arms file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data if isinstance(data, list) else data.get("arms") or []
    if not rows:
        raise ValueError(f"{path}: no arms found")

    return [_arm_from_row(row, i) for i, row in enumerate(rows)]


def _arm_from_row(row: dict[str, Any], index: int) -> ArmSpec:
    """Build one ArmSpec from a YAML mapping."""
    provider = row.get("provider")
    if not provider:
        raise ValueError(f"arm {index + 1} has no 'provider'")

    params = row.get("params") or {}
    return ArmSpec(
        id=str(row.get("id") or _default_arm_id(provider, row.get("model"))),
        provider=str(provider),
        model=row.get("model"),
        description=row.get("description"),
        params=[MetadataItem(key=k, value=json.dumps(v)) for k, v in sorted(params.items())],
    )


def _default_arm_id(provider: str, model: str | None) -> str:
    """Derive an arm id from its provider and model.

    >>> _default_arm_id("falcon", None)
    'falcon'
    >>> _default_arm_id("openai", "o3-deep-research")
    'openai__o3-deep-research'
    """
    return f"{provider}__{model}" if model else provider


def parse_arm_flag(value: str) -> ArmSpec:
    """Parse a ``--arm`` command-line value into an ArmSpec.

    Accepts ``provider``, ``provider:model``, or ``id=provider:model``.

    >>> parse_arm_flag("falcon").id
    'falcon'
    >>> a = parse_arm_flag("openai:o3-deep-research")
    >>> a.provider, a.model
    ('openai', 'o3-deep-research')
    >>> parse_arm_flag("baseline=claude_code").id
    'baseline'
    """
    arm_id, separator, rest = value.partition("=")
    if not separator:
        arm_id, rest = "", value

    provider, _, model = rest.partition(":")
    provider = provider.strip()
    if not provider:
        raise ValueError(
            f"Could not read a provider from --arm {value!r}; expected "
            f"'provider', 'provider:model', or 'id=provider:model'"
        )

    resolved_model = model.strip() or None
    return ArmSpec(
        id=arm_id.strip() or _default_arm_id(provider, resolved_model),
        provider=provider,
        model=resolved_model,
    )


def _arm_params(arm: ArmSpec) -> dict[str, Any]:
    """Decode an arm's parameters back into a provider kwargs dict.

    >>> arm = ArmSpec(id="a", provider="claude_code",
    ...               params=[MetadataItem(key="allowed_tools", value="[]")])
    >>> _arm_params(arm)
    {'allowed_tools': []}
    """
    return {item.key: json.loads(item.value) for item in (arm.params or [])}


@dataclass
class RunLayout:
    """Where a run's files go."""

    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def results_path(self) -> Path:
        return self.root / "results.tsv"

    @property
    def scores_path(self) -> Path:
        return self.root / "scores.tsv"

    def cell_dir(self, task_id: str, arm_id: str) -> Path:
        """Directory holding one cell's files."""
        return self.root / safe_segment(task_id) / safe_segment(arm_id)

    def relative(self, path: Path) -> str:
        """Render a path relative to the run root, for the manifest."""
        return str(path.relative_to(self.root))


@dataclass
class MatrixConfig:
    """How to run a matrix."""

    output_dir: Path
    concurrency: int = 4
    resume: bool = True
    #: Grade multiple-choice cells during the run. Off by default: a run's job
    #: is to materialise results, and the current grader is a provisional
    #: regex-based one whose numbers should not be produced by accident.
    grade: bool = False
    #: Serve cells from the client's response cache when it has them. Left on
    #: by default so a resumed or repeated run is cheap, but a benchmarking run
    #: that must measure the provider now should turn it off: a replay is a
    #: measurement that did not happen in the run reporting it.
    #:
    #: Applies only when `run_matrix` builds the client, which is the CLI path.
    #: A caller passing its own `client` has already chosen its cache settings,
    #: and this field is not applied on top of them.
    use_cache: bool = True
    #: Where that cache lives. None means the client's default. Applies only
    #: when `run_matrix` builds the client, as `use_cache` does.
    cache_dir: str | None = None
    #: Called with each completed cell, for progress reporting.
    on_cell: Callable[[CellResult], None] | None = field(default=None, repr=False)


def _prompt_for(task: EvalTask) -> tuple[str, list[mcq.Choice]]:
    """Render the prompt for a task, plus its options when it has any.

    Multiple-choice tasks are rendered with lettered options in a deterministic
    order; everything else is sent as written.
    """
    if task.answer_type == AnswerType.MULTIPLE_CHOICE:
        choices = mcq.present_choices(task)
        return mcq.format_prompt(task, choices), choices
    return task.prompt, []


def _completed_cell(
    layout: RunLayout,
    task: EvalTask,
    arm: ArmSpec,
    prompt: str,
    choices: list[mcq.Choice],
    grade: bool,
) -> CellResult | None:
    """Return a reusable previously completed cell, if this run directory has one.

    Resumption is by cell, not by run: a matrix that died three arms in should
    cost only the cells that never finished.

    A stored cell is reused only when the question it was asked still matches
    the one this run would ask. Editing the eval set changes the prompt - and
    adding a distractor reshuffles the options - so without that check a resumed
    run would mix answers to the old question into a manifest that records the
    new eval set.

    When grading is on and a cell was materialised without it, the cell is
    graded from the saved response rather than re-run. That is the whole point
    of separating the two: deciding to score a run later must not cost the
    provider calls again.
    """
    cell_dir = layout.cell_dir(task.id, arm.id)
    path = cell_dir / "cell.json"
    if not path.exists():
        return None

    cell = CellResult(**json.loads(path.read_text(encoding="utf-8")))
    if cell.status != CellStatus.COMPLETED:
        return None

    stored_prompt = cell_dir / "prompt.md"
    if not stored_prompt.exists() or stored_prompt.read_text(encoding="utf-8") != prompt:
        logger.info(
            "Cell %s/%s was asked a different question; re-running", task.id, arm.id
        )
        return None

    needs_grade = (
        grade
        and task.answer_type == AnswerType.MULTIPLE_CHOICE
        and cell.disposition is None
    )
    if needs_grade:
        output = cell_dir / "output.md"
        if not output.exists():
            # Nothing to grade from, and returning the cell ungraded would drop
            # it from the aggregate without saying so, quietly shrinking the
            # denominator. Re-run it instead.
            logger.warning(
                "Cell %s/%s has no saved response to grade; re-running", task.id, arm.id
            )
            return None
        answer = mcq.grade(task.id, arm.id, output.read_text(encoding="utf-8"), choices)
        cell.disposition = answer.disposition
        cell.chosen_letter = answer.chosen_letter
        cell.correct = answer.correct
        atomic_write(cell_dir / "answer.json", answer.model_dump_json(indent=2))
        atomic_write(path, cell.model_dump_json(indent=2, exclude_none=True))
        logger.info("Graded stored response for %s/%s", task.id, arm.id)

    return cell


async def _run_cell(
    client: DeepResearchClient,
    task: EvalTask,
    arm: ArmSpec,
    layout: RunLayout,
    grade: bool = False,
) -> CellResult:
    """Run one task under one arm and write its files.

    A provider failure becomes a FAILED cell rather than an exception: the point
    of a matrix is that the cells are independent.
    """
    cell_dir = layout.cell_dir(task.id, arm.id)
    cell_dir.mkdir(parents=True, exist_ok=True)

    prompt, choices = _prompt_for(task)
    atomic_write(cell_dir / "prompt.md", prompt)

    started = datetime.now(timezone.utc)
    try:
        result = await client.aresearch(
            prompt,
            provider=arm.provider,
            model=arm.model,
            provider_params=_arm_params(arm) or None,
        )
    except Exception as exc:  # noqa: BLE001 - one failed cell must not end the run
        logger.warning("Cell %s/%s failed: %s", task.id, arm.id, exc)
        cell = CellResult(
            task_id=task.id,
            arm_id=arm.id,
            status=CellStatus.FAILED,
            duration_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
            error=f"{type(exc).__name__}: {exc}",
        )
        if grade and task.answer_type == AnswerType.MULTIPLE_CHOICE:
            cell.disposition = ScoreDisposition.PROVIDER_ERROR
        atomic_write(cell_dir / "cell.json", cell.model_dump_json(indent=2, exclude_none=True))
        return cell

    output_path = cell_dir / "output.md"
    atomic_write(output_path, result.markdown or "")

    cell = CellResult(
        task_id=task.id,
        arm_id=arm.id,
        status=CellStatus.COMPLETED,
        output_path=layout.relative(output_path),
        duration_seconds=result.duration_seconds,
        citation_count=len(result.citations or []),
        provider_used=result.provider,
        model_used=result.model,
        # Copied like the rest: a replayed response is otherwise indis-
        # tinguishable from a live one, because the cache re-stamps the times.
        cached=bool(result.cached),
    )

    # Grading is opt-in: materialising the result is the run's job, and the
    # provisional grader should never produce numbers nobody asked for.
    if grade and task.answer_type == AnswerType.MULTIPLE_CHOICE:
        answer = mcq.grade(task.id, arm.id, result.markdown or "", choices)
        cell.disposition = answer.disposition
        cell.chosen_letter = answer.chosen_letter
        cell.correct = answer.correct
        atomic_write(cell_dir / "answer.json", answer.model_dump_json(indent=2))

    atomic_write(cell_dir / "cell.json", cell.model_dump_json(indent=2, exclude_none=True))
    return cell


def write_results_tsv(layout: RunLayout, cells: Sequence[CellResult]) -> None:
    """Write one row per cell, for loading into a dataframe or a spreadsheet."""
    lines = ["\t".join(_RESULT_COLUMNS)]
    for cell in cells:
        row = {
            "task_id": cell.task_id,
            "arm_id": cell.arm_id,
            "provider": cell.provider_used or "",
            "model": cell.model_used or "",
            "status": cell.status,
            "disposition": cell.disposition or "",
            "correct": "" if cell.correct is None else str(cell.correct).lower(),
            "chosen_letter": cell.chosen_letter or "",
            "duration_seconds": "" if cell.duration_seconds is None else f"{cell.duration_seconds:.1f}",
            "citation_count": "" if cell.citation_count is None else str(cell.citation_count),
            "resumed": "" if cell.resumed is None else str(cell.resumed).lower(),
            "cached": "" if cell.cached is None else str(cell.cached).lower(),
            "output_path": cell.output_path or "",
            # Tabs and newlines in an error message would corrupt the row.
            "error": " ".join((cell.error or "").split()),
        }
        lines.append("\t".join(row[c] for c in _RESULT_COLUMNS))
    atomic_write(layout.results_path, "\n".join(lines) + "\n")


def score_by_arm(eval_set: EvalSet, cells: Sequence[CellResult]) -> dict[str, MCQScore]:
    """Aggregate multiple-choice cells into a score per arm.

    Args:
        eval_set: The eval set that was run.
        cells: Every cell of the run.

    Returns:
        Arm id -> MCQScore, for arms that ran multiple-choice tasks.
    """
    mcq_task_ids = {
        t.id for t in (eval_set.tasks or []) if t.answer_type == AnswerType.MULTIPLE_CHOICE
    }
    if not mcq_task_ids:
        return {}

    by_arm: dict[str, list[MCQAnswer]] = {}
    for cell in cells:
        if cell.task_id not in mcq_task_ids or cell.disposition is None:
            continue
        by_arm.setdefault(cell.arm_id, []).append(MCQAnswer(
            task_id=cell.task_id,
            provider=cell.arm_id,
            chosen_letter=cell.chosen_letter,
            disposition=cell.disposition,
            # NOT `bool(cell.correct)`. `CellResult.correct` is Optional and
            # documented as meaningful only when SCORED, so collapsing it here
            # turned "never established" into "wrong" -- three states into two,
            # in the bridge between two models rather than in either of them.
            correct=cell.correct,
            error=cell.error,
        ))

    return {arm_id: mcq.score_mcq(answers) for arm_id, answers in by_arm.items()}


def write_scores_tsv(layout: RunLayout, scores: dict[str, MCQScore]) -> None:
    """Write per-arm aggregate scores.

    Accuracy is never written without coverage beside it: a low accuracy from
    wrong answers and a low accuracy from declining to answer are different
    results, and only coverage tells them apart.
    """
    columns = (
        "arm_id", "total", "attempted", "correct", "accuracy", "coverage",
        "precision", "abstained", "extraction_failures", "provider_errors",
    )
    lines = ["\t".join(columns)]
    for arm_id, score in sorted(scores.items()):
        lines.append("\t".join([
            arm_id,
            str(score.total), str(score.attempted), str(score.correct),
            f"{score.accuracy:.4f}", f"{score.coverage:.4f}",
            # Empty rather than 0.0000, for the reason the CLI prints a dash:
            # a spreadsheet averaging this column must not average in an arm
            # that made no attempt.
            "" if score.precision is None else f"{score.precision:.4f}",
            str(score.abstained), str(score.extraction_failures), str(score.provider_errors),
        ]))
    atomic_write(layout.scores_path, "\n".join(lines) + "\n")


def _manifest(
    eval_set: EvalSet,
    arms: Sequence[ArmSpec],
    layout: RunLayout,
    config: "MatrixConfig",
    cells: Sequence[CellResult],
    cache_enabled: bool,
    cache_dir: str | None,
) -> RunManifest:
    """Build the run manifest from the cells finished so far.

    `cache_enabled` is read off the client rather than off the config, because
    `config.use_cache` applies only when `run_matrix` builds the client. A
    caller that passes its own client -- every library caller, and every test
    here -- would otherwise have the manifest assert the default, which for a
    client with caching off is the opposite of what happened. This file's job
    is provenance; it must not be the one field that can state the reverse.
    """
    return RunManifest(
        run_id=layout.root.name,
        created_at=datetime.now(timezone.utc).isoformat(),
        eval_set_name=eval_set.name,
        eval_set_source=eval_set.source,
        eval_set_revision=eval_set.source_revision,
        is_partial=bool(eval_set.is_partial),
        partial_reason=eval_set.partial_reason,
        client_version=__version__,
        concurrency=config.concurrency,
        resume_enabled=config.resume,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
        arms=list(arms),
        cells=list(cells),
    )


async def run_matrix(
    eval_set: EvalSet,
    arms: Sequence[ArmSpec],
    config: MatrixConfig,
    client: DeepResearchClient | None = None,
) -> RunManifest:
    """Run every task in an eval set under every arm.

    Args:
        eval_set: The tasks to run.
        arms: The configurations to run them under.
        config: Output directory, concurrency and resume behaviour.
        client: Client to use; one is constructed if not given.

    Returns:
        The run manifest, also written to the output directory.
    """
    tasks = eval_set.tasks or []
    if not tasks:
        raise ValueError(f"Eval set {eval_set.name!r} has no tasks")
    if not arms:
        raise ValueError("No arms to run")

    # Built here when the caller passes none, so the cache settings on the
    # config are the ones that apply on the CLI path -- which never passes a
    # client, and so silently took the default: caching on, against a directory
    # shared with every other use of this client on the machine.
    if client is None:
        client = DeepResearchClient(cache_config=CacheConfig(
            enabled=config.use_cache,
            directory=config.cache_dir,
        ))

    # Whatever the client ended up with, which for a caller-supplied client is
    # not config.use_cache. Read directly rather than through getattr defaults:
    # if `cache_config` is ever renamed or wrapped, a default would record
    # `cache_enabled: false` for every run that used the cache -- the manifest
    # stating the reverse of what happened, quietly, which is the exact failure
    # this field was added to remove. Fail fast instead.
    cache_enabled = bool(client.cache_config.enabled)
    # Only when the cache was actually consulted: a directory recorded beside
    # `cache_enabled: false` names somewhere the run never read, which is the
    # same family as the field this one was added to close.
    cache_dir = client.cache_config.directory if cache_enabled else None
    layout = RunLayout(Path(config.output_dir))
    layout.root.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max(1, config.concurrency))
    cells: list[CellResult] = []

    async def one(task: EvalTask, arm: ArmSpec) -> CellResult:
        if config.resume:
            prompt, choices = _prompt_for(task)
            done = _completed_cell(layout, task, arm, prompt, choices, config.grade)
            if done is not None:
                logger.info("Cell %s/%s already complete; skipping", task.id, arm.id)
                # Marked after `_completed_cell` returns, which is what keeps
                # it out of the stored copy: the resume-and-grade path does
                # rewrite cell.json, but it does so before this line. So the
                # file keeps the flags of the run that produced it and this
                # manifest describes this run.
                done.resumed = True
                return done
        async with semaphore:
            cell = await _run_cell(client, task, arm, layout, grade=config.grade)
            cell.resumed = False
            return cell

    # Stable order in the written outputs, independent of completion order.
    order = {(task.id, arm.id): i
             for i, (task, arm) in enumerate((t, a) for t in tasks for a in arms)}

    def _ordered(done: list[CellResult]) -> list[CellResult]:
        return sorted(done, key=lambda c: order.get((c.task_id, c.arm_id), 0))

    pending = [one(task, arm) for task in tasks for arm in arms]
    logger.info(
        "Running %d tasks x %d arms = %d cells, %d at a time",
        len(tasks), len(arms), len(pending), config.concurrency,
    )

    last_manifest = 0.0

    def snapshot() -> None:
        """Rewrite the summary files from the cells finished so far.

        Written as the run goes rather than only at the end, so an interrupted
        run still leaves a readable results.tsv and manifest, and a long one can
        genuinely be inspected while it is running. Both are small beside the
        provider outputs already written per cell.
        """
        nonlocal last_manifest
        ordered = _ordered(cells)
        write_results_tsv(layout, ordered)
        # The manifest grows with the run, so re-serialising it after every cell
        # is quadratic in bytes written. Once a second is frequent enough for a
        # file whose purpose is to survive an interruption, and the final write
        # after the loop is unconditional.
        now = time.monotonic()
        if now - last_manifest >= 1.0:
            last_manifest = now
            atomic_write(
                layout.manifest_path,
                _manifest(eval_set, arms, layout, config, ordered, cache_enabled, cache_dir)
                .model_dump_json(indent=2, exclude_none=True),
            )

    for coro in asyncio.as_completed(pending):
        cell = await coro
        cells.append(cell)
        snapshot()
        if config.on_cell:
            config.on_cell(cell)

    ordered = _ordered(cells)
    manifest = _manifest(eval_set, arms, layout, config, ordered, cache_enabled, cache_dir)

    atomic_write(layout.manifest_path, manifest.model_dump_json(indent=2, exclude_none=True))
    write_results_tsv(layout, ordered)

    if config.grade:
        scores = score_by_arm(eval_set, ordered)
        if scores:
            write_scores_tsv(layout, scores)

    return manifest
