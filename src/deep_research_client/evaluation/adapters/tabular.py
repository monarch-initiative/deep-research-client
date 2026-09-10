"""Adapters for eval sets you write yourself, as YAML or TSV.

These are the plainest way to evaluate a set of questions: put the questions in
a file and point the runner at it. Nothing about the subject matter is assumed,
so a TSV of two columns is a valid benchmark.

YAML is the fuller format - it can carry distractors, rubrics and provenance.
TSV is for the case where the questions came out of a spreadsheet and there is
nothing more to say about them.
"""

import csv
from pathlib import Path
from typing import Any

import yaml

from ..datamodel import AnswerSpec, AnswerType, EvalSet, EvalTask, MetadataItem
from .base import EvalSetAdapter, check_unique_ids

#: Column/key names accepted for the question text, in precedence order. Real
#: eval sets arrive spelled all three ways and rejecting two of them would be
#: pedantry rather than fail-fast.
_PROMPT_KEYS = ("prompt", "question", "query")

#: Keys consumed by named slots. Anything else in a row becomes metadata rather
#: than being silently dropped.
_KNOWN_KEYS = {
    "id", "answer_type", "ideal", "distractors", "abstention_option",
    "task_type", "subject_id", "tags", "source_id", *_PROMPT_KEYS,
}


def _prompt_of(row: dict[str, Any], where: str) -> str:
    """Return the question text from a row, whichever key it is under.

    >>> _prompt_of({"question": "Why?"}, "row 1")
    'Why?'
    >>> _prompt_of({"nope": "x"}, "row 1")
    Traceback (most recent call last):
        ...
    ValueError: row 1 has no question text: expected one of prompt, question, query
    """
    for key in _PROMPT_KEYS:
        value = row.get(key)
        if value:
            return str(value).strip()
    raise ValueError(
        f"{where} has no question text: expected one of {', '.join(_PROMPT_KEYS)}"
    )


def _split_list(value: Any, separator: str) -> list[str]:
    """Split a delimited scalar into a list, passing real lists through.

    >>> _split_list("a|b|c", "|")
    ['a', 'b', 'c']
    >>> _split_list(["a", "b"], "|")
    ['a', 'b']
    >>> _split_list("", "|")
    []
    """
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(separator) if part.strip()]


def _answer_type_of(row: dict[str, Any], distractors: list[str]) -> AnswerType:
    """Determine a row's answer type, inferring it when unstated.

    A row carrying distractors is multiple choice; a row carrying only an ideal
    answer is short answer; a row carrying neither is a report task. Stating
    ``answer_type`` explicitly always wins.

    >>> _answer_type_of({}, [])
    <AnswerType.REPORT: 'REPORT'>
    >>> _answer_type_of({}, ["b"])
    <AnswerType.MULTIPLE_CHOICE: 'MULTIPLE_CHOICE'>
    >>> _answer_type_of({"ideal": "42"}, [])
    <AnswerType.SHORT_ANSWER: 'SHORT_ANSWER'>
    >>> _answer_type_of({"answer_type": "report"}, ["b"])
    <AnswerType.REPORT: 'REPORT'>
    """
    stated = row.get("answer_type")
    if stated:
        return AnswerType(str(stated).strip().upper())
    if distractors:
        return AnswerType.MULTIPLE_CHOICE
    if row.get("ideal"):
        return AnswerType.SHORT_ANSWER
    return AnswerType.REPORT


def _task_from_row(
    row: dict[str, Any], index: int, list_separator: str
) -> EvalTask:
    """Build one EvalTask from a YAML mapping or TSV row.

    Args:
        row: The raw row.
        index: Zero-based position, used to generate an id when none is given.
        list_separator: Separator for delimited multi-value fields.

    Returns:
        The parsed task.

    >>> t = _task_from_row({"question": "Why?", "ideal": "A", "distractors": "B|C"}, 0, "|")
    >>> t.id, t.answer_type, t.answer_spec.distractors
    ('task_1', 'MULTIPLE_CHOICE', ['B', 'C'])
    """
    where = f"row {index + 1}"
    prompt = _prompt_of(row, where)
    distractors = _split_list(row.get("distractors"), list_separator)
    answer_type = _answer_type_of(row, distractors)

    answer_spec = None
    ideal = row.get("ideal")
    if ideal:
        answer_spec = AnswerSpec(
            ideal=str(ideal).strip(),
            distractors=distractors,
            abstention_option=row.get("abstention_option") or None,
        )
    elif answer_type != AnswerType.REPORT:
        raise ValueError(
            f"{where} declares answer_type={answer_type.value} but has no 'ideal' answer"
        )

    metadata = [
        MetadataItem(key=k, value=str(v))
        for k, v in sorted(row.items())
        if k not in _KNOWN_KEYS and v not in (None, "")
    ]

    return EvalTask(
        id=str(row.get("id") or f"task_{index + 1}").strip(),
        prompt=prompt,
        answer_type=answer_type,
        answer_spec=answer_spec,
        task_type=row.get("task_type") or None,
        subject_id=row.get("subject_id") or None,
        tags=_split_list(row.get("tags"), ","),
        source_id=row.get("source_id") or None,
        metadata=metadata,
    )


class YamlAdapter(EvalSetAdapter):
    """Reads an eval set from a YAML file.

    The file is either a mapping with a ``tasks`` list and optional provenance,
    or a bare list of tasks::

        name: coscientist-v1
        description: Mechanism questions for the co-scientist comparison
        tasks:
          - id: fgfr3_mech
            prompt: What are the pathophysiological mechanisms of achondroplasia?
            tags: [biomedical, mechanism]
          - id: pick_one
            prompt: Which residue is mutated?
            ideal: G380R
            distractors: [G375C, R248C]
    """

    name = "yaml"
    description = "Eval set written as YAML, with optional distractors and provenance"

    def load(self, source: str | Path, **options: Any) -> EvalSet:
        """Load an eval set from a YAML file.

        Args:
            source: Path to the YAML file.
            **options: ``list_separator`` for delimited fields (default ``|``).

        Returns:
            The parsed EvalSet.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Eval set file not found: {path}")

        separator = str(options.get("list_separator", "|"))
        data = yaml.safe_load(path.read_text()) or {}

        if isinstance(data, list):
            data = {"tasks": data}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping or a list of tasks")

        rows = data.get("tasks") or []
        if not rows:
            raise ValueError(f"{path}: no tasks found")

        tasks = [_task_from_row(row, i, separator) for i, row in enumerate(rows)]
        check_unique_ids(tasks, str(path))

        return EvalSet(
            name=data.get("name") or path.stem,
            version=data.get("version"),
            description=data.get("description"),
            source=data.get("source") or str(path),
            source_revision=data.get("source_revision"),
            license=data.get("license"),
            homepage=data.get("homepage"),
            is_partial=bool(data.get("is_partial", False)),
            partial_reason=data.get("partial_reason"),
            tasks=tasks,
        )


class TsvAdapter(EvalSetAdapter):
    """Reads an eval set from a TSV (or CSV) file with a header row.

    Recognised columns are ``id``, one of ``prompt``/``question``/``query``,
    ``ideal``, ``distractors``, ``answer_type``, ``task_type``, ``subject_id``,
    ``tags`` and ``source_id``. Any other column is kept as task metadata, so a
    spreadsheet's extra bookkeeping columns survive into the results.
    """

    name = "tsv"
    description = "Eval set written as TSV/CSV with a header row"

    def load(self, source: str | Path, **options: Any) -> EvalSet:
        """Load an eval set from a delimited text file.

        Args:
            source: Path to the TSV/CSV file.
            **options: ``delimiter`` (default tab, or comma for a ``.csv``
                suffix) and ``list_separator`` (default ``|``).

        Returns:
            The parsed EvalSet.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Eval set file not found: {path}")

        default_delimiter = "," if path.suffix.lower() == ".csv" else "\t"
        delimiter = str(options.get("delimiter", default_delimiter))
        separator = str(options.get("list_separator", "|"))

        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=delimiter))
        if not rows:
            raise ValueError(f"{path}: no rows found")

        tasks = [_task_from_row(row, i, separator) for i, row in enumerate(rows)]
        check_unique_ids(tasks, str(path))

        return EvalSet(name=path.stem, source=str(path), tasks=tasks)
