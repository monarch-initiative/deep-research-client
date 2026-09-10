"""LAB-Bench adapter.

LAB-Bench (Laurent et al. 2024, arXiv:2407.10362) is a multiple-choice
benchmark of biology research tasks from FutureHouse. It is the first
foreign benchmark this framework reads, and it earns that place for a reason
beyond coverage: LitQA2 is the subset PaperQA2 was evaluated on, and this
client already wraps Edison, FutureHouse's descendant of that system. So there
is a published number to check the harness against. A scorer with no
calibration fixture is a scorer nobody has reason to trust.

The data is downloaded rather than vendored. Three reasons, in descending
order of weight:

1. The dataset ships a canary string so that contamination can be detected if
   its questions turn up in a training corpus. Committing the questions to a
   public repository feeds them to scrapers and degrades the benchmark for
   everyone - including FutureHouse, whose client is a dependency here.
2. LAB-Bench is CC-BY-SA-4.0 and this repository is BSD-3-Clause. Vendoring
   would put share-alike data inside a permissive tree.
3. It is ~334 MB, and a copy in the tree goes stale silently.

Pinning beats vendoring for reproducibility anyway: a download records the
dataset revision it came from, so a score can name the exact data that produced
it. The canary itself is never put into a prompt or written into an eval set -
it is a contamination marker, not question content.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from ..datamodel import AnswerSpec, AnswerType, EvalSet, EvalTask, MetadataItem
from .._fs import atomic_write
from .base import EvalSetAdapter, validate_tasks

logger = logging.getLogger(__name__)

#: HuggingFace dataset identifier.
DATASET = "futurehouse/lab-bench"

#: Homepage used for citation.
HOMEPAGE = "https://arxiv.org/abs/2407.10362"

#: License of the upstream data. Not this repository's license.
LICENSE = "CC-BY-SA-4.0"

#: The paper offers models "a specific option to decline to answer for lack of
#: information", and reports precision over attempted questions on that basis.
#: The paper text does not fix the exact wording, so this is the framework's
#: default rather than a quotation, and it is recorded per task in the eval set
#: so a run always says which convention produced its numbers. Override it with
#: the ``abstention_option`` load option to match another harness exactly.
DEFAULT_ABSTENTION_OPTION = "Insufficient information to answer this question."

#: Rows per datasets-server request. The API caps a page at 100.
_PAGE_SIZE = 100

#: Subset -> (expected public row count, is_text_only).
#:
#: Counts are from the dataset card and are checked after download: a mismatch
#: means upstream changed under a pin, which should be loud rather than quietly
#: shifting a score.
#:
#: FigQA and TableQA are marked not-text-only because their questions are about
#: figures and tables supplied as images. A text-in/text-out research client
#: cannot present those, so this adapter refuses them rather than silently
#: scoring a model on a question it was never shown.
SUBSETS: dict[str, tuple[int, bool]] = {
    "LitQA2": (199, True),
    "SuppQA": (82, True),
    "DbQA": (520, True),
    "ProtocolQA": (108, True),
    "SeqQA": (600, True),
    "CloningScenarios": (33, True),
    "FigQA": (181, False),
    "TableQA": (244, False),
}

#: Why every LAB-Bench eval set is partial. Scores computed here are not
#: comparable with the published leaderboard and anything reporting them must
#: say so; the schema carries this rather than a README so it cannot be lost.
PARTIAL_REASON = (
    "The publisher withholds roughly 20% of LAB-Bench privately for contamination "
    "monitoring, so only the public portion is scored here. These numbers are not "
    "comparable with published leaderboard figures."
)


def text_only_subsets() -> list[str]:
    """Return the subsets this client can actually present to a provider.

    >>> "LitQA2" in text_only_subsets()
    True
    >>> "FigQA" in text_only_subsets()
    False
    """
    return [name for name, (_, text_only) in SUBSETS.items() if text_only]


def _cache_root(cache_dir: str | Path | None) -> Path:
    """Return the directory downloaded LAB-Bench pages are cached under."""
    base = Path(cache_dir) if cache_dir else Path.home() / ".deep_research_cache"
    return base / "eval_datasets" / "lab-bench"


def resolve_revision(client: httpx.Client | None = None) -> str:
    """Return the current commit sha of the LAB-Bench dataset repository.

    Resolved rather than hardcoded so that the pin records what was actually
    downloaded instead of a hash copied into source and left to rot.

    Args:
        client: Optional httpx client to reuse.

    Returns:
        The dataset revision sha.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        response = client.get(f"https://huggingface.co/api/datasets/{DATASET}")
        response.raise_for_status()
        sha = response.json().get("sha")
        if not sha:
            raise ValueError(f"HuggingFace API returned no sha for {DATASET}")
        return str(sha)
    finally:
        if owns_client:
            client.close()


def _fetch_rows(subset: str, client: httpx.Client) -> list[dict[str, Any]]:
    """Download every public row of one LAB-Bench subset.

    Uses the datasets-server JSON API rather than the parquet files, so that
    reading LAB-Bench costs no dependency beyond httpx, which this package
    already requires.

    Args:
        subset: Subset (config) name.
        client: httpx client to use.

    Returns:
        The raw row dicts, in dataset order.
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = client.get(
            "https://datasets-server.huggingface.co/rows",
            params={
                "dataset": DATASET,
                "config": subset,
                "split": "train",
                "offset": offset,
                "length": _PAGE_SIZE,
            },
        )
        response.raise_for_status()
        payload = response.json()
        page = [entry["row"] for entry in payload.get("rows", [])]
        rows.extend(page)
        total = payload.get("num_rows_total")
        offset += len(page)
        if not page or total is None or offset >= total:
            break
    return rows


def newest_cached_revision(
    subsets: str | Sequence[str], cache_dir: str | Path | None = None
) -> str | None:
    """Return the newest cached revision holding *every* named subset.

    Used when the revision cannot be resolved - an outage should not turn a
    complete local copy of a benchmark into a failed run.

    All the requested subsets have to be present under the same revision. A
    cache assembled across two revisions would otherwise satisfy the first
    subset and then send the rest to a network that is not there, reporting a
    download failure rather than the real problem.

    Args:
        subsets: Subset name, or the names that must all be present.
        cache_dir: Cache root.

    Returns:
        The revision, or None when no single cached revision covers them all.
    """
    wanted = [subsets] if isinstance(subsets, str) else list(subsets)
    root = _cache_root(cache_dir)
    if not root.is_dir() or not wanted:
        return None

    candidates = [
        d for d in root.iterdir()
        if all((d / f"{s}.json").exists() for s in wanted)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda d: max((d / f"{s}.json").stat().st_mtime for s in wanted),
    ).name


def _resolve_or_fall_back(
    client: httpx.Client,
    subsets: str | Sequence[str],
    cache_dir: str | Path | None,
) -> str:
    """Resolve the current revision, falling back to a cached one when offline.

    An outage should not turn a complete local copy of a benchmark into a failed
    run. The fallback is loud, because the provenance then describes what is on
    disk rather than what is current.
    """
    try:
        return resolve_revision(client)
    except httpx.HTTPError as exc:
        cached = newest_cached_revision(subsets, cache_dir)
        if cached is None:
            raise
        logger.warning(
            "Could not reach HuggingFace to resolve the LAB-Bench revision (%s); "
            "using the cached revision %s. Provenance describes the cached copy, "
            "not necessarily the current dataset.",
            exc, cached[:8],
        )
        return cached


def fetch_subset(
    subset: str,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    refresh: bool = False,
    resolved_revision: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Download one LAB-Bench subset, caching it on disk by revision.

    Args:
        subset: Subset name, e.g. ``LitQA2``.
        cache_dir: Cache root; defaults to the client's cache directory.
        revision: Revision the caller expects. The current revision is resolved
            and used; this is checked against it, and a mismatch is an error
            rather than a silent relabelling.
        resolved_revision: An already-resolved revision, to avoid re-resolving
            once per subset when several are loaded together.
        refresh: Re-download even when a cached copy for this revision exists.

    Returns:
        A ``(rows, revision)`` pair.

    Raises:
        ValueError: If the subset is unknown, or the row count does not match
            the count recorded for it.
    """
    if subset not in SUBSETS:
        raise ValueError(
            f"Unknown LAB-Bench subset: {subset!r}. "
            f"Available: {', '.join(SUBSETS)}"
        )

    with httpx.Client(timeout=120.0) as client:
        # Resolve unless the caller already did. The datasets-server rows
        # endpoint serves whatever is current and takes no revision parameter,
        # so honouring a requested revision by simply filing the download under
        # that name would stamp current data with an old sha - exactly the false
        # provenance the pin exists to prevent. Hence: resolve, then assert.
        resolved = resolved_revision or _resolve_or_fall_back(client, subset, cache_dir)
        if revision and revision != resolved:
            raise ValueError(
                f"LAB-Bench is at revision {resolved}, but {revision} was requested. "
                f"The dataset API only serves the current revision, so the requested "
                f"one cannot be downloaded; a previously cached copy of it may still "
                f"be on disk under that revision."
            )

        path = _cache_root(cache_dir) / resolved / f"{subset}.json"
        from_cache = path.exists() and not refresh

        if from_cache:
            logger.info("Using cached LAB-Bench %s at revision %s", subset, resolved[:8])
            rows = json.loads(path.read_text())
        else:
            logger.info("Downloading LAB-Bench %s at revision %s", subset, resolved[:8])
            rows = _fetch_rows(subset, client)

    # Checked on both paths: a cached file can be short too, from a download that
    # was interrupted mid-write, and a silently short benchmark changes every
    # score computed from it.
    expected, _ = SUBSETS[subset]
    if len(rows) != expected:
        source = "cached copy of" if from_cache else "download of"
        raise ValueError(
            f"LAB-Bench {subset}: {source} revision {resolved} has {len(rows)} rows "
            f"but {expected} were expected. Either upstream changed - update SUBSETS "
            f"after confirming the new counts - or the cache is truncated, in which "
            f"case re-fetch with refresh=True."
        )

    if not from_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json.dumps(rows, indent=2))
    return rows, resolved


def _task_from_row(row: dict[str, Any], subset: str, abstention: str | None) -> EvalTask:
    """Convert one LAB-Bench row into an EvalTask.

    The ``canary`` field is deliberately dropped: it is a contamination marker
    for the dataset, not part of the question, and must reach neither a prompt
    nor a file this project writes.

    >>> row = {"id": "abc", "question": "Which?", "ideal": "A",
    ...        "distractors": ["B", "C"], "subtask": "litqa2-v1-public",
    ...        "canary": "lab-bench:DO-NOT-PROPAGATE"}
    >>> task = _task_from_row(row, "LitQA2", "Insufficient information.")
    >>> task.id, task.answer_type
    ('LitQA2__abc', 'MULTIPLE_CHOICE')
    >>> task.answer_spec.abstention_option
    'Insufficient information.'
    >>> [m.key for m in task.metadata]
    ['lab_bench_subset']
    """
    source_id = str(row.get("id", ""))
    distractors = [str(d) for d in (row.get("distractors") or [])]

    return EvalTask(
        # Namespaced by subset: ids are unique within a subset but the framework
        # allows several subsets in one eval set, and ids become directory names.
        id=f"{subset}__{source_id}",
        prompt=str(row.get("question", "")).strip(),
        answer_type=AnswerType.MULTIPLE_CHOICE,
        answer_spec=AnswerSpec(
            ideal=str(row.get("ideal", "")).strip(),
            distractors=distractors,
            abstention_option=abstention,
        ),
        task_type=str(row.get("subtask") or subset),
        tags=["lab-bench", subset],
        source_id=source_id,
        metadata=[MetadataItem(key="lab_bench_subset", value=subset)],
    )


class LabBenchAdapter(EvalSetAdapter):
    """Builds an :class:`EvalSet` from one or more LAB-Bench subsets.

    The ``source`` is a subset name, a comma-separated list of them, or ``all``
    for every text-only subset::

        get_adapter("lab-bench").load("LitQA2")
        get_adapter("lab-bench").load("LitQA2,SuppQA")
        get_adapter("lab-bench").load("all")
    """

    name = "lab-bench"
    description = "LAB-Bench biology multiple-choice benchmark (FutureHouse)"
    requires_network = True

    def load(self, source: str | Path, **options: Any) -> EvalSet:
        """Load LAB-Bench subsets as an eval set.

        Args:
            source: Subset name, comma-separated subset names, or ``all``.
            **options: ``cache_dir``, ``revision``, ``refresh``, and
                ``abstention_option`` (pass ``None`` to offer no abstention).

        Returns:
            The assembled EvalSet, marked partial.

        Raises:
            ValueError: If a requested subset needs figure or table input,
                which this client cannot present.
        """
        requested = str(source).strip()
        subsets = (
            text_only_subsets()
            if requested.lower() == "all"
            else [s.strip() for s in requested.split(",") if s.strip()]
        )

        multimodal = [s for s in subsets if s in SUBSETS and not SUBSETS[s][1]]
        if multimodal:
            raise ValueError(
                f"LAB-Bench subset(s) {', '.join(multimodal)} ask about figures or "
                f"tables supplied as images, which a text-only research client cannot "
                f"present. Scoring them here would measure the harness, not the "
                f"provider. Text-only subsets: {', '.join(text_only_subsets())}"
            )

        abstention = options.get("abstention_option", DEFAULT_ABSTENTION_OPTION)

        # Resolved once rather than per subset: "all" would otherwise make six
        # identical API calls, and a revision that changed between them would
        # silently mix two datasets into one eval set.
        requested_revision: str | None = options.get("revision")
        with httpx.Client(timeout=30.0) as client:
            pinned: str = _resolve_or_fall_back(client, subsets, options.get("cache_dir"))
        if requested_revision and requested_revision != pinned:
            raise ValueError(
                f"LAB-Bench is at revision {pinned}, but {requested_revision} "
                f"was requested."
            )

        tasks: list[EvalTask] = []
        revisions: set[str] = set()
        for subset in subsets:
            rows, revision = fetch_subset(
                subset,
                cache_dir=options.get("cache_dir"),
                resolved_revision=pinned,
                refresh=bool(options.get("refresh", False)),
            )
            revisions.add(revision)
            tasks.extend(_task_from_row(row, subset, abstention) for row in rows)

        validate_tasks(tasks, f"LAB-Bench {', '.join(subsets)}")
        return EvalSet(
            name=f"lab-bench-{'-'.join(s.lower() for s in subsets)}",
            description=(
                f"LAB-Bench {', '.join(subsets)} "
                f"(Laurent et al. 2024, arXiv:2407.10362)"
            ),
            source=DATASET,
            source_revision=", ".join(sorted(revisions)),
            license=LICENSE,
            homepage=HOMEPAGE,
            is_partial=True,
            partial_reason=PARTIAL_REASON,
            tasks=tasks,
        )
