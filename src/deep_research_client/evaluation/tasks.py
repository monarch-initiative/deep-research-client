"""Generate evaluation tasks from ground truth entities.

Given a GroundTruthEntity (disease or gene), generates EvalTask objects
with natural-language queries suitable for deep research tools.

Task types follow the taxonomy from our evaluation framework:
- disease_mechanism: pathophysiology and molecular mechanisms
- gene_function: molecular functions and biological processes
- gene_disease_link: how a gene contributes to disease
- treatment_rationale: treatments and their mechanisms
- phenotype_explanation: phenotypes and their molecular basis
"""

import hashlib
from typing import Sequence

from .models import EvalTask, GroundTruthClaim, GroundTruthEntity, TaskType


def _make_task_id(entity_id: str, task_type: str) -> str:
    """Generate a deterministic task ID.

    >>> _make_task_id("MONDO:0007037", "disease_mechanism")[:20]
    'MONDO_0007037__disea'
    """
    safe_id = entity_id.replace(":", "_").replace("/", "_")
    return f"{safe_id}__{task_type}"


def _filter_claims(claims: list[GroundTruthClaim], categories: Sequence[str]) -> list[GroundTruthClaim]:
    """Filter claims to those matching given categories."""
    return [c for c in claims if c.category in categories]


def generate_disease_tasks(entity: GroundTruthEntity) -> list[EvalTask]:
    """Generate evaluation tasks for a disease entity from dismech.

    Creates up to 3 task types per disease depending on available claims:
    - disease_mechanism (from pathophysiology claims)
    - treatment_rationale (from treatment claims)
    - phenotype_explanation (from phenotype + pathophysiology claims)

    Args:
        entity: A GroundTruthEntity with entity_type='disease'.

    Returns:
        List of EvalTask objects.

    >>> from .models import GroundTruthEntity, GroundTruthClaim
    >>> entity = GroundTruthEntity(
    ...     entity_id="MONDO:0007037", entity_type="disease", name="Achondroplasia",
    ...     source_repo="dismech",
    ...     claims=[
    ...         GroundTruthClaim(category="pathophysiology", name="FGFR3", description="gain-of-function"),
    ...         GroundTruthClaim(category="phenotype", name="Short stature", description="disproportionate"),
    ...     ],
    ... )
    >>> tasks = generate_disease_tasks(entity)
    >>> len(tasks)
    2
    >>> tasks[0].task_type
    <TaskType.DISEASE_MECHANISM: 'disease_mechanism'>
    """
    tasks = []
    name = entity.name

    # Disease mechanism task
    mech_claims = _filter_claims(entity.claims, ["pathophysiology"])
    if mech_claims:
        tasks.append(
            EvalTask(
                task_id=_make_task_id(entity.entity_id, "disease_mechanism"),
                task_type=TaskType.DISEASE_MECHANISM,
                query=(
                    f"What are the pathophysiological mechanisms of {name}? "
                    f"Describe the molecular and cellular pathways involved, "
                    f"the genes and proteins implicated, and how they lead to disease. "
                    f"Cite primary research papers with PMIDs."
                ),
                ground_truth_entity_id=entity.entity_id,
                ground_truth_claims=mech_claims,
            )
        )

    # Treatment rationale task
    tx_claims = _filter_claims(entity.claims, ["treatment"])
    if tx_claims:
        tasks.append(
            EvalTask(
                task_id=_make_task_id(entity.entity_id, "treatment_rationale"),
                task_type=TaskType.TREATMENT_RATIONALE,
                query=(
                    f"What are the current treatments for {name} and what is the "
                    f"mechanistic rationale for each? Include both approved therapies "
                    f"and promising experimental approaches. "
                    f"Cite primary research papers with PMIDs."
                ),
                ground_truth_entity_id=entity.entity_id,
                ground_truth_claims=tx_claims,
            )
        )

    # Phenotype explanation task
    pheno_claims = _filter_claims(entity.claims, ["phenotype", "pathophysiology"])
    if _filter_claims(entity.claims, ["phenotype"]):
        tasks.append(
            EvalTask(
                task_id=_make_task_id(entity.entity_id, "phenotype_explanation"),
                task_type=TaskType.PHENOTYPE_EXPLANATION,
                query=(
                    f"What are the clinical features (phenotypes) of {name}? "
                    f"For each phenotype, explain the underlying molecular mechanism "
                    f"that produces it. "
                    f"Cite primary research papers with PMIDs."
                ),
                ground_truth_entity_id=entity.entity_id,
                ground_truth_claims=pheno_claims,
            )
        )

    return tasks


def generate_gene_tasks(entity: GroundTruthEntity) -> list[EvalTask]:
    """Generate evaluation tasks for a gene entity from ai-gene-review.

    Creates a gene_function task per gene.

    Args:
        entity: A GroundTruthEntity with entity_type='gene'.

    Returns:
        List of EvalTask objects.

    >>> from .models import GroundTruthEntity, GroundTruthClaim
    >>> entity = GroundTruthEntity(
    ...     entity_id="P38398", entity_type="gene", name="BRCA1",
    ...     source_repo="ai-gene-review",
    ...     claims=[
    ...         GroundTruthClaim(category="gene_function", name="E3 ligase", description="ubiquitin ligase"),
    ...     ],
    ... )
    >>> tasks = generate_gene_tasks(entity)
    >>> len(tasks)
    1
    >>> tasks[0].task_type
    <TaskType.GENE_FUNCTION: 'gene_function'>
    """
    tasks = []
    gene = entity.name

    func_claims = _filter_claims(entity.claims, ["gene_function"])
    if func_claims:
        tasks.append(
            EvalTask(
                task_id=_make_task_id(entity.entity_id, "gene_function"),
                task_type=TaskType.GENE_FUNCTION,
                query=(
                    f"What are the molecular functions and biological roles of the "
                    f"human gene {gene}? Describe its protein products, enzymatic "
                    f"activities, protein interactions, and involvement in biological "
                    f"processes and pathways. Include the relevant Gene Ontology (GO) terms. "
                    f"Cite primary research papers with PMIDs."
                ),
                ground_truth_entity_id=entity.entity_id,
                ground_truth_claims=func_claims,
            )
        )

    return tasks


def generate_tasks(entity: GroundTruthEntity) -> list[EvalTask]:
    """Generate all applicable evaluation tasks for an entity.

    Dispatches to disease or gene task generators based on entity_type.

    Args:
        entity: A GroundTruthEntity.

    Returns:
        List of EvalTask objects.
    """
    if entity.entity_type == "disease":
        return generate_disease_tasks(entity)
    elif entity.entity_type == "gene":
        return generate_gene_tasks(entity)
    else:
        raise ValueError(f"Unknown entity type: {entity.entity_type}")
