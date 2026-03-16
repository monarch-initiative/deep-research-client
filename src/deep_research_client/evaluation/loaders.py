"""Load ground truth from dismech and ai-gene-review repositories.

Each loader reads structured YAML files and produces ``GroundTruthEntity``
objects with typed claims and evidence.

Example usage::

    from pathlib import Path
    from deep_research_client.evaluation.loaders import load_dismech_entity

    entity = load_dismech_entity(Path("kb/disorders/Achondroplasia.yaml"))
    print(entity.name, len(entity.claims))
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from .models import EvidenceItem, GroundTruthClaim, GroundTruthEntity, OntologyTerm

logger = logging.getLogger(__name__)


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


def _extract_dismech_pathophysiology(data: dict[str, Any]) -> list[GroundTruthClaim]:
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
            GroundTruthClaim(
                category="pathophysiology",
                name=entry.get("name", "unnamed"),
                description=entry.get("description", ""),
                ontology_terms=terms,
                evidence=_parse_evidence(entry.get("evidence")),
            )
        )
    return claims


def _extract_dismech_phenotypes(data: dict[str, Any]) -> list[GroundTruthClaim]:
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
            GroundTruthClaim(
                category="phenotype",
                name=entry.get("name", "unnamed"),
                description=entry.get("description", ""),
                ontology_terms=terms,
                evidence=_parse_evidence(entry.get("evidence")),
            )
        )
    return claims


def _extract_dismech_treatments(data: dict[str, Any]) -> list[GroundTruthClaim]:
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
            GroundTruthClaim(
                category="treatment",
                name=entry.get("name", "unnamed"),
                description=entry.get("description", ""),
                ontology_terms=terms,
                evidence=_parse_evidence(entry.get("evidence")),
            )
        )
    return claims


def _extract_dismech_inheritance(data: dict[str, Any]) -> list[GroundTruthClaim]:
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
            GroundTruthClaim(
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


def _extract_gene_annotation_claims(data: dict[str, Any]) -> list[GroundTruthClaim]:
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
            GroundTruthClaim(
                category="gene_function",
                name=f"{term_label} ({action})",
                description=description,
                ontology_terms=terms,
                evidence=evidence,
            )
        )
    return claims


def _extract_gene_core_functions(data: dict[str, Any]) -> list[GroundTruthClaim]:
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
            GroundTruthClaim(
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
