"""Tests for the evaluation framework.

Tests cover:
- Models (Pydantic validation, doctests)
- Loaders (dismech and ai-gene-review YAML parsing)
- Task generation
- Citation/claim extraction from markdown
- Scorer helpers (no LLM calls — those are integration tests)
"""

from pathlib import Path
import pytest

from deep_research_client.evaluation.models import (
    EvalTask,
    EvidenceItem,
    FACTScore,
    GroundTruthClaim,
    GroundTruthEntity,
    OntologyTerm,
    RACEDimension,
    RACEScore,
    TaskType,
)
from deep_research_client.evaluation.loaders import (
    _parse_evidence,
    _parse_ontology_term,
    load_dismech_entity,
    load_gene_review_entity,
)
from deep_research_client.evaluation.tasks import (
    generate_disease_tasks,
    generate_gene_tasks,
    generate_tasks,
)
from deep_research_client.evaluation.scorers import (
    extract_citations_from_markdown,
    extract_claims_with_citations,
)
from deep_research_client.evaluation.runner import (
    EvalConfig,
    generate_all_tasks,
    load_entities,
    parse_dr_output,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_disease_entity():
    return GroundTruthEntity(
        entity_id="MONDO:0007037",
        entity_type="disease",
        name="Achondroplasia",
        source_repo="dismech",
        claims=[
            GroundTruthClaim(
                category="pathophysiology",
                name="FGFR3 gain-of-function",
                description="FGFR3 G380R causes constitutive receptor activation",
                ontology_terms=[OntologyTerm(id="GO:0008543", label="FGFR signaling")],
                evidence=[EvidenceItem(reference="PMID:7913883", snippet="point mutations in FGFR3")],
            ),
            GroundTruthClaim(
                category="phenotype",
                name="Short stature",
                description="Disproportionate short stature with rhizomelic limb shortening",
                ontology_terms=[OntologyTerm(id="HP:0008873", label="Disproportionate short-limb short stature")],
            ),
            GroundTruthClaim(
                category="treatment",
                name="Vosoritide",
                description="C-type natriuretic peptide analog antagonizing FGFR3",
                evidence=[EvidenceItem(reference="PMID:31269546")],
            ),
        ],
    )


@pytest.fixture
def sample_gene_entity():
    return GroundTruthEntity(
        entity_id="P38398",
        entity_type="gene",
        name="BRCA1",
        source_repo="ai-gene-review",
        claims=[
            GroundTruthClaim(
                category="gene_function",
                name="E3 ubiquitin ligase (ACCEPT)",
                description="BRCA1-BARD1 heterodimer functions as RING-type E3 ubiquitin ligase",
                ontology_terms=[OntologyTerm(id="GO:0004842", label="ubiquitin-protein transferase activity")],
                evidence=[EvidenceItem(reference="PMID:12890688", snippet="K6-linked polyubiquitin")],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_ground_truth_entity_all_references(self, sample_disease_entity):
        refs = sample_disease_entity.all_references
        assert "PMID:7913883" in refs
        assert "PMID:31269546" in refs
        assert len(refs) == 2

    def test_race_dimension_normalized_score(self):
        d = RACEDimension(dimension="accuracy", score=4.0, max_score=5.0)
        assert d.normalized_score == pytest.approx(0.8)

    def test_race_score_overall(self):
        s = RACEScore(
            dimensions=[
                RACEDimension(dimension="a", score=5.0),
                RACEDimension(dimension="b", score=3.0),
            ]
        )
        assert s.overall_score == pytest.approx(0.8)

    def test_fact_score(self):
        s = FACTScore(total_citations=10, verified_citations=8, citation_accuracy=0.8, effective_citations=8)
        assert s.citation_accuracy == pytest.approx(0.8)

    def test_eval_task_creation(self, sample_disease_entity):
        task = EvalTask(
            task_id="test",
            task_type=TaskType.DISEASE_MECHANISM,
            query="What causes achondroplasia?",
            ground_truth_entity_id=sample_disease_entity.entity_id,
        )
        assert task.task_type == TaskType.DISEASE_MECHANISM


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


class TestLoaderHelpers:
    def test_parse_ontology_term(self):
        t = _parse_ontology_term({"term": {"id": "GO:0008543", "label": "FGFR signaling"}})
        assert t is not None
        assert t.id == "GO:0008543"

    def test_parse_ontology_term_empty(self):
        assert _parse_ontology_term({}) is None

    def test_parse_evidence(self):
        items = _parse_evidence([
            {"reference": "PMID:123", "snippet": "text", "supports": "SUPPORT"},
            {"reference": "PMID:456"},
        ])
        assert len(items) == 2
        assert items[0].snippet == "text"
        assert items[1].snippet is None

    def test_parse_evidence_none(self):
        assert _parse_evidence(None) == []


@pytest.mark.integration
@pytest.mark.parametrize("yaml_path", [
    Path("/tmp/dismech/kb/disorders/Achondroplasia.yaml"),
])
def test_load_dismech_entity(yaml_path):
    if not yaml_path.exists():
        pytest.skip(f"{yaml_path} not available")
    entity = load_dismech_entity(yaml_path)
    assert entity.entity_type == "disease"
    assert entity.name == "Achondroplasia"
    assert entity.entity_id == "MONDO:0007037"
    assert len(entity.claims) > 0
    # Should have pathophysiology, phenotype, and treatment claims
    categories = {c.category for c in entity.claims}
    assert "pathophysiology" in categories
    assert "phenotype" in categories
    assert "treatment" in categories


@pytest.mark.integration
@pytest.mark.parametrize("yaml_path", [
    Path("/tmp/ai-gene-review/genes/human/BRCA1/BRCA1-ai-review.yaml"),
])
def test_load_gene_review_entity(yaml_path):
    if not yaml_path.exists():
        pytest.skip(f"{yaml_path} not available")
    entity = load_gene_review_entity(yaml_path)
    assert entity.entity_type == "gene"
    assert entity.name == "BRCA1"
    assert entity.entity_id == "P38398"
    assert len(entity.claims) > 0


# ---------------------------------------------------------------------------
# Task generation tests
# ---------------------------------------------------------------------------


class TestTaskGeneration:
    def test_disease_tasks(self, sample_disease_entity):
        tasks = generate_disease_tasks(sample_disease_entity)
        assert len(tasks) == 3  # mechanism, treatment, phenotype
        types = {t.task_type for t in tasks}
        assert TaskType.DISEASE_MECHANISM in types
        assert TaskType.TREATMENT_RATIONALE in types
        assert TaskType.PHENOTYPE_EXPLANATION in types

    def test_gene_tasks(self, sample_gene_entity):
        tasks = generate_gene_tasks(sample_gene_entity)
        assert len(tasks) == 1
        assert tasks[0].task_type == TaskType.GENE_FUNCTION
        assert "BRCA1" in tasks[0].query

    def test_generate_tasks_dispatches(self, sample_disease_entity, sample_gene_entity):
        disease_tasks = generate_tasks(sample_disease_entity)
        gene_tasks = generate_tasks(sample_gene_entity)
        assert len(disease_tasks) == 3
        assert len(gene_tasks) == 1

    def test_disease_tasks_no_treatment(self):
        entity = GroundTruthEntity(
            entity_id="X", entity_type="disease", name="X", source_repo="dismech",
            claims=[
                GroundTruthClaim(category="pathophysiology", name="m1", description="desc"),
            ],
        )
        tasks = generate_disease_tasks(entity)
        assert len(tasks) == 1
        assert tasks[0].task_type == TaskType.DISEASE_MECHANISM


# ---------------------------------------------------------------------------
# Citation extraction tests
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    def test_extract_pmid(self):
        cits = extract_citations_from_markdown("This was shown (PMID:7913883).")
        assert len(cits) == 1
        assert cits[0].normalized_id == "PMID:7913883"

    def test_extract_doi(self):
        cits = extract_citations_from_markdown("See DOI:10.1038/ng1234.")
        assert len(cits) == 1
        assert cits[0].normalized_id == "DOI:10.1038/ng1234"

    def test_extract_pubmed_url(self):
        cits = extract_citations_from_markdown("https://pubmed.ncbi.nlm.nih.gov/12345678")
        assert len(cits) == 1
        assert cits[0].normalized_id == "PMID:12345678"

    def test_extract_multiple(self):
        text = "A (PMID:111111). B (PMID:222222). C (DOI:10.1000/xyz)."
        cits = extract_citations_from_markdown(text)
        assert len(cits) == 3

    def test_dedup(self):
        text = "A (PMID:7913883). B (PMID:7913883)."
        cits = extract_citations_from_markdown(text)
        assert len(cits) == 1

    def test_extract_claims_with_citations(self):
        text = "FGFR3 causes disease (PMID:7913883). No ref here. Another fact (DOI:10.1038/x)."
        claims = extract_claims_with_citations(text)
        assert len(claims) == 2
        assert "FGFR3" in claims[0].text
        assert len(claims[0].citations) == 1

    def test_extract_claims_no_citations(self):
        claims = extract_claims_with_citations("No citations in this text at all.")
        assert len(claims) == 0


# ---------------------------------------------------------------------------
# Runner helper tests
# ---------------------------------------------------------------------------


class TestRunner:
    def test_parse_dr_output(self, sample_disease_entity):
        task = EvalTask(
            task_id="test",
            task_type=TaskType.DISEASE_MECHANISM,
            query="test query",
            ground_truth_entity_id="X",
        )
        markdown = "FGFR3 G380R mutation (PMID:7913883) causes achondroplasia."
        dr = parse_dr_output(task, markdown, "falcon")
        assert dr.provider == "falcon"
        assert len(dr.extracted_claims) == 1
        assert len(dr.extracted_citations) == 1
        assert dr.extracted_citations[0].normalized_id == "PMID:7913883"

    @pytest.mark.integration
    def test_load_entities_from_file(self):
        path = Path("/tmp/dismech/kb/disorders/Achondroplasia.yaml")
        if not path.exists():
            pytest.skip("dismech data not available")
        config = EvalConfig(entity_files=[path])
        entities = load_entities(config)
        assert len(entities) == 1
        assert entities[0].name == "Achondroplasia"

    def test_generate_all_tasks(self, sample_disease_entity, sample_gene_entity):
        tasks = generate_all_tasks([sample_disease_entity, sample_gene_entity])
        assert len(tasks) == 4  # 3 disease + 1 gene
