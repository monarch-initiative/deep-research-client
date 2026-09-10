"""Adapters for the Monarch curated knowledge bases.

Two benchmarks live here because they share a shape: dismech records disease
mechanisms and ai-gene-review records gene function annotations, both as curated
YAML with claims and per-claim published evidence. Neither has a single correct
answer, so both produce REPORT-shaped tasks scored against a rubric.

These were the framework's only benchmarks, and the framework was built around
them - loaders addressed by name, a closed enum of biomedical task types,
scorers reaching for gene symbols. They are now one adapter among several, which
is the point: what makes a benchmark work here should be true of any benchmark.
"""

import logging
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any, Optional

from pydantic import BaseModel, Field

import yaml

from ..datamodel import (
    AnswerType,
    EvalSet,
    EvalTask,
    EvidenceItem,
    ExpectedTopic,
    OntologyTerm,
    ReferenceClaim,
    Rubric,
    SpotCheck,
)
from .base import EvalSetAdapter, validate_tasks

logger = logging.getLogger(__name__)


class GroundTruthEntity(BaseModel):
    """Curated knowledge about one disease or gene, as read from a Monarch repo.

    An intermediate representation, not part of the generic model: entity types
    and source repositories are facts about these two knowledge bases, and
    pushing them into the shared schema would be the hardwiring this framework
    was restructured to remove. Tasks generated from an entity carry only what
    a scorer needs.

    >>> entity = GroundTruthEntity(
    ...     entity_id="MONDO:0007037", entity_type="disease",
    ...     name="Achondroplasia", source_repo="dismech", claims=[],
    ... )
    >>> entity.name
    'Achondroplasia'
    """

    entity_id: str = Field(..., description="Primary identifier (MONDO, HGNC, UniProt)")
    entity_type: str = Field(..., description="'disease' or 'gene'")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(default=None, description="Summary description")
    source_repo: str = Field(..., description="Source repository: 'dismech' or 'ai-gene-review'")
    source_file: Optional[str] = Field(default=None, description="Path to source YAML file")
    claims: list[ReferenceClaim] = Field(default_factory=list, description="All curated claims")

    @property
    def all_references(self) -> set[str]:
        """Collect all unique reference IDs across all claims.

        >>> entity = GroundTruthEntity(
        ...     entity_id="X", entity_type="disease", name="X", source_repo="dismech",
        ...     claims=[ReferenceClaim(
        ...         category="pathophysiology", name="m1", description="desc",
        ...         evidence=[EvidenceItem(reference="PMID:123"), EvidenceItem(reference="PMID:456")],
        ...     )],
        ... )
        >>> sorted(entity.all_references)
        ['PMID:123', 'PMID:456']
        """
        refs: set[str] = set()
        for claim in self.claims:
            for ev in claim.evidence or []:
                refs.add(ev.reference)
            for sub in claim.subclaims or []:
                for ev in sub.evidence or []:
                    refs.add(ev.reference)
        return refs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ontology_term(data: dict[str, Any]) -> OntologyTerm | None:
    """Extract an OntologyTerm from a nested dict with a 'term' key.

    >>> _parse_ontology_term({"term": {"id": "GO:0008543", "label": "FGFR signaling"}})
    OntologyTerm(id='GO:0008543', label='FGFR signaling')
    >>> _parse_ontology_term({}) is None
    True
    """
    term_data = data.get("term")
    if not term_data or not isinstance(term_data, dict):
        return None
    tid = term_data.get("id", "")
    label = term_data.get("label", "")
    if not tid:
        return None
    return OntologyTerm(id=tid, label=label)


def _parse_evidence(evidence_list: list[dict[str, Any]] | None) -> list[EvidenceItem]:
    """Parse a list of evidence dicts into EvidenceItem objects.

    >>> _parse_evidence([{"reference": "PMID:123", "snippet": "text", "supports": "SUPPORT"}])
    [EvidenceItem(reference='PMID:123', snippet='text', supports='SUPPORT', explanation=None)]
    >>> _parse_evidence(None)
    []
    """
    if not evidence_list:
        return []
    items = []
    for ev in evidence_list:
        if not isinstance(ev, dict):
            continue
        ref = ev.get("reference", "")
        if not ref:
            continue
        items.append(
            EvidenceItem(
                reference=str(ref),
                snippet=ev.get("snippet"),
                supports=ev.get("supports"),
                explanation=ev.get("explanation"),
            )
        )
    return items


# ---------------------------------------------------------------------------
# dismech loader
# ---------------------------------------------------------------------------


def _extract_dismech_pathophysiology(data: dict[str, Any]) -> list[ReferenceClaim]:
    """Extract pathophysiology claims from a dismech disorder dict."""
    claims = []
    for entry in data.get("pathophysiology", []):
        if not isinstance(entry, dict):
            continue
        terms: list[OntologyTerm] = []
        # Collect biological_processes terms
        for bp in entry.get("biological_processes", []):
            t = _parse_ontology_term(bp)
            if t:
                terms.append(t)
        # Collect cell_type terms
        for ct in entry.get("cell_types", []):
            t = _parse_ontology_term(ct)
            if t:
                terms.append(t)
        # Gene term
        gene_data = entry.get("gene")
        if isinstance(gene_data, dict):
            t = _parse_ontology_term(gene_data)
            if t:
                terms.append(t)

        claims.append(
            ReferenceClaim(
                category="pathophysiology",
                name=entry.get("name", "unnamed"),
                description=entry.get("description", ""),
                ontology_terms=terms,
                evidence=_parse_evidence(entry.get("evidence")),
            )
        )
    return claims


def _extract_dismech_phenotypes(data: dict[str, Any]) -> list[ReferenceClaim]:
    """Extract phenotype claims from a dismech disorder dict."""
    claims = []
    for entry in data.get("phenotypes", []):
        if not isinstance(entry, dict):
            continue
        terms: list[OntologyTerm] = []
        t = _parse_ontology_term(entry.get("phenotype_term", {}))
        if t:
            terms.append(t)
        claims.append(
            ReferenceClaim(
                category="phenotype",
                name=entry.get("name", "unnamed"),
                description=entry.get("description", ""),
                ontology_terms=terms,
                evidence=_parse_evidence(entry.get("evidence")),
            )
        )
    return claims


def _extract_dismech_treatments(data: dict[str, Any]) -> list[ReferenceClaim]:
    """Extract treatment claims from a dismech disorder dict."""
    claims = []
    for entry in data.get("treatments", []):
        if not isinstance(entry, dict):
            continue
        terms: list[OntologyTerm] = []
        t = _parse_ontology_term(entry.get("treatment_term", {}))
        if t:
            terms.append(t)
        claims.append(
            ReferenceClaim(
                category="treatment",
                name=entry.get("name", "unnamed"),
                description=entry.get("description", ""),
                ontology_terms=terms,
                evidence=_parse_evidence(entry.get("evidence")),
            )
        )
    return claims


def _extract_dismech_inheritance(data: dict[str, Any]) -> list[ReferenceClaim]:
    """Extract inheritance/genetic claims from a dismech disorder dict."""
    claims = []
    for entry in data.get("inheritance", []):
        if not isinstance(entry, dict):
            continue
        terms: list[OntologyTerm] = []
        t = _parse_ontology_term(entry.get("inheritance_term", {}))
        if t:
            terms.append(t)
        claims.append(
            ReferenceClaim(
                category="genetic_factor",
                name=entry.get("name", "unnamed"),
                description=entry.get("description", ""),
                ontology_terms=terms,
                evidence=_parse_evidence(entry.get("evidence")),
            )
        )
    return claims


def load_dismech_entity(yaml_path: Path) -> GroundTruthEntity:
    """Load a single dismech disorder YAML file as a GroundTruthEntity.

    Args:
        yaml_path: Path to a dismech disorder YAML file.

    Returns:
        GroundTruthEntity with claims extracted from pathophysiology,
        phenotypes, treatments, and inheritance sections.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Extract disease identifier
    disease_term = data.get("disease_term", {})
    term_data = disease_term.get("term", {}) if isinstance(disease_term, dict) else {}
    entity_id = term_data.get("id", data.get("name", yaml_path.stem))

    claims = (
        _extract_dismech_pathophysiology(data)
        + _extract_dismech_phenotypes(data)
        + _extract_dismech_treatments(data)
        + _extract_dismech_inheritance(data)
    )

    return GroundTruthEntity(
        entity_id=entity_id,
        entity_type="disease",
        name=data.get("name", yaml_path.stem),
        description=data.get("description"),
        source_repo="dismech",
        source_file=str(yaml_path),
        claims=claims,
    )


def load_dismech_repo(kb_dir: Path) -> list[GroundTruthEntity]:
    """Load all disorder entities from a dismech kb/disorders/ directory.

    Args:
        kb_dir: Path to the ``kb/disorders/`` directory.

    Returns:
        List of GroundTruthEntity objects, one per disorder YAML file.
        History files (``*.history.yaml``) are skipped.
    """
    entities = []
    for yaml_file in sorted(kb_dir.glob("*.yaml")):
        if ".history." in yaml_file.name:
            continue
        try:
            entities.append(load_dismech_entity(yaml_file))
        except Exception:
            logger.warning("Failed to load %s", yaml_file, exc_info=True)
    logger.info("Loaded %d dismech entities from %s", len(entities), kb_dir)
    return entities


# ---------------------------------------------------------------------------
# ai-gene-review loader
# ---------------------------------------------------------------------------


def _extract_gene_annotation_claims(data: dict[str, Any]) -> list[ReferenceClaim]:
    """Extract claims from existing_annotations in ai-gene-review format."""
    claims = []
    for ann in data.get("existing_annotations", []):
        if not isinstance(ann, dict):
            continue
        review = ann.get("review", {})
        if not isinstance(review, dict):
            continue

        # Skip removed annotations
        action = review.get("action", "")
        if action == "REMOVE":
            continue

        term_data = ann.get("term", {})
        terms: list[OntologyTerm] = []
        if isinstance(term_data, dict) and term_data.get("id"):
            terms.append(OntologyTerm(id=term_data["id"], label=term_data.get("label", "")))

        # Build evidence from supported_by
        evidence = []
        for ref in review.get("supported_by", []):
            if not isinstance(ref, dict):
                continue
            ref_id = ref.get("reference_id", "")
            if not ref_id or ref_id.startswith("file:"):
                continue
            evidence.append(
                EvidenceItem(
                    reference=str(ref_id),
                    snippet=ref.get("supporting_text"),
                    supports="SUPPORT",
                )
            )

        summary = review.get("summary", "")
        reason = review.get("reason", "")
        description = f"{summary} {reason}".strip() if summary or reason else ""

        term_label = term_data.get("label", "unknown") if isinstance(term_data, dict) else "unknown"
        claims.append(
            ReferenceClaim(
                category="gene_function",
                name=f"{term_label} ({action})",
                description=description,
                ontology_terms=terms,
                evidence=evidence,
            )
        )
    return claims


def _extract_gene_core_functions(data: dict[str, Any]) -> list[ReferenceClaim]:
    """Extract claims from core_functions in ai-gene-review format."""
    claims = []
    for cf in data.get("core_functions", []):
        if not isinstance(cf, dict):
            continue
        mf = cf.get("molecular_function", "unnamed")
        terms: list[OntologyTerm] = []
        if isinstance(mf, dict):
            name = mf.get("label", "unnamed")
            if mf.get("id"):
                terms.append(OntologyTerm(id=mf["id"], label=name))
        else:
            name = str(mf)
        claims.append(
            ReferenceClaim(
                category="gene_function",
                name=name,
                description=cf.get("description", ""),
                ontology_terms=terms,
            )
        )
    return claims


def load_gene_review_entity(yaml_path: Path) -> GroundTruthEntity:
    """Load a single ai-gene-review YAML file as a GroundTruthEntity.

    Args:
        yaml_path: Path to a ``*-ai-review.yaml`` file.

    Returns:
        GroundTruthEntity with claims from annotations and core functions.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    gene_symbol = data.get("gene_symbol", yaml_path.parent.name)
    uniprot_id = data.get("id", "")

    claims = _extract_gene_annotation_claims(data) + _extract_gene_core_functions(data)

    return GroundTruthEntity(
        entity_id=uniprot_id or gene_symbol,
        entity_type="gene",
        name=gene_symbol,
        description=data.get("description"),
        source_repo="ai-gene-review",
        source_file=str(yaml_path),
        claims=claims,
    )


def load_gene_review_repo(genes_dir: Path) -> list[GroundTruthEntity]:
    """Load all gene entities from an ai-gene-review genes/human/ directory.

    Args:
        genes_dir: Path to the ``genes/human/`` directory.

    Returns:
        List of GroundTruthEntity objects, one per gene with an ai-review YAML.
    """
    entities = []
    for yaml_file in sorted(genes_dir.glob("*/*-ai-review.yaml")):
        try:
            entities.append(load_gene_review_entity(yaml_file))
        except Exception:
            logger.warning("Failed to load %s", yaml_file, exc_info=True)
    logger.info("Loaded %d gene review entities from %s", len(entities), genes_dir)
    return entities


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------

#: Directory holding the rubric data these adapters attach to their tasks.
RUBRIC_DIR = Path(__file__).parent.parent / "rubrics"


def load_rubric_data(stem: str) -> dict[str, Any]:
    """Read one rubric YAML file from the bundled rubric directory.

    Args:
        stem: File stem, e.g. ``gene_function``.

    Returns:
        The parsed mapping.

    >>> sorted(load_rubric_data("disease_mechanism"))
    ['expected_topics']
    """
    path = RUBRIC_DIR / f"{stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Rubric not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def build_rubric(
    stem: str, claims: list[ReferenceClaim], subject: str | None = None
) -> Rubric:
    """Assemble a Rubric from bundled rubric data plus this task's claims.

    Subject-specific spot checks are looked up by ``subject`` and merged with the
    generic ones, so a report about a gene with curated checks is graded against
    both, and a gene without them is still graded against the generic set.

    Args:
        stem: Rubric file stem, ``gene_function`` or ``disease_mechanism``.
        claims: Reference claims this task expects a report to cover.
        subject: Subject key for spot-check lookup, e.g. a gene symbol.

    Returns:
        The assembled rubric.

    >>> r = build_rubric("gene_function", [], subject="BRCA1")
    >>> len(r.expected_topics) > 0 and len(r.spot_checks) > 0
    True
    >>> "protein_length" in {c.name for c in r.spot_checks}
    True
    >>> "protein_length" in {c.name for c in build_rubric("gene_function", []).spot_checks}
    False
    """
    data = load_rubric_data(stem)
    topics = [ExpectedTopic(**t) for t in data.get("expected_topics", [])]
    spot_checks = [SpotCheck(**c) for c in data.get("spot_checks", [])]

    if subject:
        by_subject = load_rubric_data("gene_spot_checks")
        spot_checks.extend(SpotCheck(**c) for c in by_subject.get(subject, []))

    return Rubric(reference_claims=claims, spot_checks=spot_checks, expected_topics=topics)


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def _make_task_id(entity_id: str, task_type: str) -> str:
    """Generate a deterministic, filesystem-safe task ID.

    >>> _make_task_id("MONDO:0007037", "disease_mechanism")
    'MONDO_0007037__disease_mechanism'
    """
    safe_id = entity_id.replace(":", "_").replace("/", "_")
    return f"{safe_id}__{task_type}"


def _filter_claims(claims: list[ReferenceClaim], categories: Sequence[str]) -> list[ReferenceClaim]:
    """Filter claims to those matching given categories."""
    return [c for c in claims if c.category in categories]


def _report_task(
    entity: GroundTruthEntity,
    task_type: str,
    prompt: str,
    claims: list[ReferenceClaim],
    rubric_stem: str,
    subject: str | None = None,
) -> EvalTask:
    """Build one REPORT-shaped task for an entity."""
    return EvalTask(
        id=_make_task_id(entity.entity_id, task_type),
        prompt=prompt,
        answer_type=AnswerType.REPORT,
        rubric=build_rubric(rubric_stem, claims, subject=subject),
        task_type=task_type,
        subject_id=entity.entity_id,
        tags=[entity.source_repo, entity.entity_type],
    )


def generate_disease_tasks(entity: GroundTruthEntity) -> list[EvalTask]:
    """Generate evaluation tasks for a disease entity from dismech.

    Up to three tasks per disease, depending on which claim categories the
    curation actually carries: mechanism, treatment rationale, and phenotype
    explanation.

    Args:
        entity: A GroundTruthEntity with entity_type='disease'.

    Returns:
        List of EvalTask objects.

    >>> entity = GroundTruthEntity(
    ...     entity_id="MONDO:0007037", entity_type="disease", name="Achondroplasia",
    ...     source_repo="dismech",
    ...     claims=[
    ...         ReferenceClaim(category="pathophysiology", name="FGFR3", description="gain-of-function"),
    ...         ReferenceClaim(category="phenotype", name="Short stature", description="disproportionate"),
    ...     ],
    ... )
    >>> tasks = generate_disease_tasks(entity)
    >>> [t.task_type for t in tasks]
    ['disease_mechanism', 'phenotype_explanation']
    >>> tasks[0].answer_type
    'REPORT'
    """
    name = entity.name
    tasks: list[EvalTask] = []

    mech_claims = _filter_claims(entity.claims, ["pathophysiology"])
    if mech_claims:
        tasks.append(_report_task(
            entity, "disease_mechanism",
            f"What are the pathophysiological mechanisms of {name}? "
            f"Describe the molecular and cellular pathways involved, "
            f"the genes and proteins implicated, and how they lead to disease. "
            f"Cite primary research papers with PMIDs.",
            mech_claims, "disease_mechanism",
        ))

    tx_claims = _filter_claims(entity.claims, ["treatment"])
    if tx_claims:
        tasks.append(_report_task(
            entity, "treatment_rationale",
            f"What are the current treatments for {name} and what is the "
            f"mechanistic rationale for each? Include both approved therapies "
            f"and promising experimental approaches. "
            f"Cite primary research papers with PMIDs.",
            tx_claims, "disease_mechanism",
        ))

    if _filter_claims(entity.claims, ["phenotype"]):
        tasks.append(_report_task(
            entity, "phenotype_explanation",
            f"What are the clinical features (phenotypes) of {name}? "
            f"For each phenotype, explain the underlying molecular mechanism "
            f"that produces it. "
            f"Cite primary research papers with PMIDs.",
            _filter_claims(entity.claims, ["phenotype", "pathophysiology"]),
            "disease_mechanism",
        ))

    return tasks


def generate_gene_tasks(entity: GroundTruthEntity) -> list[EvalTask]:
    """Generate evaluation tasks for a gene entity from ai-gene-review.

    Args:
        entity: A GroundTruthEntity with entity_type='gene'.

    Returns:
        List of EvalTask objects.

    >>> entity = GroundTruthEntity(
    ...     entity_id="P38398", entity_type="gene", name="BRCA1",
    ...     source_repo="ai-gene-review",
    ...     claims=[ReferenceClaim(category="gene_function", name="E3 ligase", description="ubiquitin ligase")],
    ... )
    >>> tasks = generate_gene_tasks(entity)
    >>> tasks[0].task_type, tasks[0].subject_id
    ('gene_function', 'P38398')
    """
    func_claims = _filter_claims(entity.claims, ["gene_function"])
    if not func_claims:
        return []

    gene = entity.name
    return [_report_task(
        entity, "gene_function",
        f"What are the molecular functions and biological roles of the "
        f"human gene {gene}? Describe its protein products, enzymatic "
        f"activities, protein interactions, and involvement in biological "
        f"processes and pathways. Include the relevant Gene Ontology (GO) terms. "
        f"Cite primary research papers with PMIDs.",
        func_claims, "gene_function", subject=gene,
    )]


def generate_tasks(entity: GroundTruthEntity) -> list[EvalTask]:
    """Generate all applicable evaluation tasks for an entity.

    Args:
        entity: A GroundTruthEntity.

    Returns:
        List of EvalTask objects.
    """
    if entity.entity_type == "disease":
        return generate_disease_tasks(entity)
    if entity.entity_type == "gene":
        return generate_gene_tasks(entity)
    raise ValueError(f"Unknown entity type: {entity.entity_type}")


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class _MonarchAdapter(EvalSetAdapter):
    """Shared loading logic for the two Monarch knowledge bases."""

    #: Loader for a single entity file.
    _load_entity: Callable[[Path], GroundTruthEntity]
    #: Loader for a whole repository directory.
    _load_repo: Callable[[Path], list[GroundTruthEntity]]

    def load(self, source: str | Path, **options: Any) -> EvalSet:
        """Load curated entities and turn them into report tasks.

        Args:
            source: A single YAML file, or a directory of them.
            **options: ``entity_names`` to filter by name, ``max_entities`` to cap.

        Returns:
            The assembled EvalSet.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")

        entities = (
            type(self)._load_repo(path) if path.is_dir() else [type(self)._load_entity(path)]
        )

        names = options.get("entity_names")
        if names:
            wanted = {n.lower() for n in names}
            entities = [e for e in entities if e.name.lower() in wanted]

        max_entities = options.get("max_entities")
        if max_entities is not None:
            entities = entities[:max_entities]

        tasks: list[EvalTask] = []
        for entity in entities:
            tasks.extend(generate_tasks(entity))

        # Entity ids fall back to a name and then a filename, so two curated
        # files can land on the same task id.
        validate_tasks(tasks, str(path))
        logger.info(
            "Loaded %d entities from %s, generating %d tasks", len(entities), path, len(tasks)
        )
        return EvalSet(name=self.name, source=str(path), tasks=tasks)


class DismechAdapter(_MonarchAdapter):
    """Disease mechanism knowledge from the dismech repository."""

    name = "dismech"
    description = "Curated disease mechanisms (Monarch dismech)"
    _load_entity = staticmethod(load_dismech_entity)
    _load_repo = staticmethod(load_dismech_repo)


class GeneReviewAdapter(_MonarchAdapter):
    """Gene function annotations from the ai-gene-review repository."""

    name = "ai-gene-review"
    description = "Curated gene function annotations (Monarch ai-gene-review)"
    _load_entity = staticmethod(load_gene_review_entity)
    _load_repo = staticmethod(load_gene_review_repo)
