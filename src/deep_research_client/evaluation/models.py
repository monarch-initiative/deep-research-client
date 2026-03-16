"""Pydantic models for the evaluation framework.

Covers ground truth representation, DR output parsing, and scoring results.
Ground truth is loaded from dismech (disease mechanisms) and ai-gene-review
(gene function annotations) repositories.

Scoring follows two complementary frameworks:
- FACT: citation verification (does the cited paper support the claim?)
- RACE: report quality (comprehensiveness, accuracy, organization, terminology)
- Claim recall: does the DR output cover the known ground truth claims?
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ground truth models
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """A single piece of evidence backing a ground truth claim.

    >>> e = EvidenceItem(reference="PMID:8078586", snippet="autosomal dominant trait")
    >>> e.reference
    'PMID:8078586'
    """

    reference: str = Field(..., description="PMID, DOI, or other reference identifier")
    snippet: Optional[str] = Field(default=None, description="Verbatim quote from the source")
    supports: Optional[str] = Field(default=None, description="SUPPORT, REFUTE, or PARTIAL")
    explanation: Optional[str] = Field(default=None, description="Why this evidence is relevant")


class OntologyTerm(BaseModel):
    """An ontology term (GO, HP, MONDO, CL, UBERON, etc.).

    >>> t = OntologyTerm(id="GO:0008543", label="fibroblast growth factor receptor signaling pathway")
    >>> t.id
    'GO:0008543'
    """

    id: str = Field(..., description="Ontology CURIE, e.g. GO:0008543")
    label: str = Field(..., description="Human-readable label")


class GroundTruthClaim(BaseModel):
    """A single verifiable claim from the ground truth knowledge base.

    Claims form a flat list per entity (gene or disease).  Each claim has
    a category (e.g. pathophysiology, phenotype, treatment), a text description,
    optional ontology terms, and supporting evidence with references.

    >>> c = GroundTruthClaim(
    ...     category="pathophysiology",
    ...     name="FGFR3 gain-of-function",
    ...     description="FGFR3 G380R causes constitutive receptor activation",
    ...     evidence=[EvidenceItem(reference="PMID:7913883", snippet="point mutations in FGFR3")],
    ... )
    >>> c.category
    'pathophysiology'
    """

    category: str = Field(..., description="Claim category: pathophysiology, phenotype, treatment, genetic_factor, gene_function")
    name: str = Field(..., description="Short name for the claim")
    description: str = Field(..., description="Full text of the claim")
    ontology_terms: list[OntologyTerm] = Field(default_factory=list, description="Associated ontology terms")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Supporting evidence with references")
    subclaims: list["GroundTruthClaim"] = Field(default_factory=list, description="Nested sub-claims")


class GroundTruthEntity(BaseModel):
    """Ground truth for a single entity (disease or gene).

    >>> entity = GroundTruthEntity(
    ...     entity_id="MONDO:0007037",
    ...     entity_type="disease",
    ...     name="Achondroplasia",
    ...     source_repo="dismech",
    ...     claims=[],
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
    claims: list[GroundTruthClaim] = Field(default_factory=list, description="All ground truth claims")

    @property
    def all_references(self) -> set[str]:
        """Collect all unique reference IDs across all claims.

        >>> entity = GroundTruthEntity(
        ...     entity_id="X", entity_type="disease", name="X", source_repo="dismech",
        ...     claims=[GroundTruthClaim(
        ...         category="pathophysiology", name="m1", description="desc",
        ...         evidence=[EvidenceItem(reference="PMID:123"), EvidenceItem(reference="PMID:456")],
        ...     )],
        ... )
        >>> sorted(entity.all_references)
        ['PMID:123', 'PMID:456']
        """
        refs: set[str] = set()
        for claim in self.claims:
            for ev in claim.evidence:
                refs.add(ev.reference)
            for sub in claim.subclaims:
                for ev in sub.evidence:
                    refs.add(ev.reference)
        return refs


# ---------------------------------------------------------------------------
# Evaluation task models
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Types of deep research evaluation tasks."""

    DISEASE_MECHANISM = "disease_mechanism"
    GENE_FUNCTION = "gene_function"
    GENE_DISEASE_LINK = "gene_disease_link"
    TREATMENT_RATIONALE = "treatment_rationale"
    PHENOTYPE_EXPLANATION = "phenotype_explanation"


class EvalTask(BaseModel):
    """A single evaluation task to send to a deep research tool.

    >>> task = EvalTask(
    ...     task_id="achondroplasia_mechanism",
    ...     task_type=TaskType.DISEASE_MECHANISM,
    ...     query="What are the pathophysiological mechanisms of Achondroplasia?",
    ...     ground_truth_entity_id="MONDO:0007037",
    ... )
    >>> task.task_type
    <TaskType.DISEASE_MECHANISM: 'disease_mechanism'>
    """

    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Category of the evaluation task")
    query: str = Field(..., description="The research question to send to the DR tool")
    ground_truth_entity_id: str = Field(..., description="ID of the ground truth entity")
    ground_truth_claims: list[GroundTruthClaim] = Field(
        default_factory=list, description="Subset of ground truth claims relevant to this task"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


# ---------------------------------------------------------------------------
# DR output parsing models
# ---------------------------------------------------------------------------


class ExtractedCitation(BaseModel):
    """A citation extracted from a DR output.

    >>> c = ExtractedCitation(raw_reference="PMID:7913883", normalized_id="PMID:7913883")
    >>> c.normalized_id
    'PMID:7913883'
    """

    raw_reference: str = Field(..., description="Citation as it appears in the text")
    normalized_id: Optional[str] = Field(default=None, description="Normalized identifier (PMID, DOI)")
    url: Optional[str] = Field(default=None, description="URL if available")


class ExtractedClaim(BaseModel):
    """A claim extracted from a DR output, with its supporting citations.

    >>> c = ExtractedClaim(
    ...     text="FGFR3 mutations cause achondroplasia",
    ...     citations=[ExtractedCitation(raw_reference="PMID:7913883")],
    ... )
    >>> len(c.citations)
    1
    """

    text: str = Field(..., description="The claim text")
    section: Optional[str] = Field(default=None, description="Section of the report where claim appears")
    citations: list[ExtractedCitation] = Field(default_factory=list, description="Citations supporting this claim")


class DROutput(BaseModel):
    """Parsed output from a deep research tool, ready for scoring.

    >>> out = DROutput(
    ...     task_id="test", provider="falcon", raw_markdown="# Report\\n...",
    ...     extracted_claims=[], extracted_citations=[],
    ... )
    >>> out.provider
    'falcon'
    """

    task_id: str = Field(..., description="ID of the evaluation task")
    provider: str = Field(..., description="DR provider that generated this output")
    model: Optional[str] = Field(default=None, description="Model used")
    raw_markdown: str = Field(..., description="Full markdown output from the DR tool")
    extracted_claims: list[ExtractedClaim] = Field(default_factory=list, description="Claims extracted from the output")
    extracted_citations: list[ExtractedCitation] = Field(default_factory=list, description="All citations found")
    duration_seconds: Optional[float] = Field(default=None, description="Time taken")


# ---------------------------------------------------------------------------
# Scoring result models
# ---------------------------------------------------------------------------


class CitationVerification(BaseModel):
    """Result of verifying a single citation against its claimed support.

    >>> v = CitationVerification(
    ...     citation=ExtractedCitation(raw_reference="PMID:123"),
    ...     claim_text="X causes Y",
    ...     supported=True,
    ...     explanation="The paper directly states X causes Y",
    ... )
    >>> v.supported
    True
    """

    citation: ExtractedCitation
    claim_text: str
    supported: Optional[bool] = Field(default=None, description="True if citation supports the claim")
    explanation: Optional[str] = Field(default=None, description="LLM explanation of verification")
    source_text: Optional[str] = Field(default=None, description="Retrieved abstract/text from the source")
    error: Optional[str] = Field(default=None, description="Error if verification failed")


class FACTScore(BaseModel):
    """FACT scoring result: citation accuracy and effective citations.

    >>> s = FACTScore(
    ...     total_citations=10, verified_citations=8,
    ...     citation_accuracy=0.8, effective_citations=8,
    ... )
    >>> s.citation_accuracy
    0.8
    """

    total_citations: int = Field(..., description="Total citation-claim pairs checked")
    verified_citations: int = Field(..., description="Citations that support their claims")
    citation_accuracy: float = Field(..., description="verified / total")
    effective_citations: int = Field(..., description="Count of verifiably supported citations")
    verifications: list[CitationVerification] = Field(default_factory=list, description="Individual verification results")


class ClaimMatch(BaseModel):
    """Result of matching a ground truth claim to DR output.

    >>> m = ClaimMatch(
    ...     ground_truth_claim_name="FGFR3 gain-of-function",
    ...     matched=True,
    ...     best_matching_text="FGFR3 G380R mutation causes constitutive activation",
    ... )
    >>> m.matched
    True
    """

    ground_truth_claim_name: str
    ground_truth_claim_description: str = ""
    matched: bool = Field(..., description="Whether the DR output covers this claim")
    best_matching_text: Optional[str] = Field(default=None, description="Best matching text from DR output")
    similarity_score: Optional[float] = Field(default=None, description="Semantic similarity score")
    explanation: Optional[str] = Field(default=None, description="LLM explanation of match")


class ClaimRecallScore(BaseModel):
    """Claim-level recall and precision against ground truth.

    >>> s = ClaimRecallScore(
    ...     total_ground_truth_claims=10, matched_claims=7,
    ...     claim_recall=0.7, matches=[],
    ... )
    >>> s.claim_recall
    0.7
    """

    total_ground_truth_claims: int
    matched_claims: int
    claim_recall: float = Field(..., description="matched / total ground truth claims")
    total_extracted_claims: Optional[int] = Field(default=None)
    claim_precision: Optional[float] = Field(default=None, description="matched / total extracted claims (if computed)")
    matches: list[ClaimMatch] = Field(default_factory=list, description="Per-claim match details")


class RACEDimension(BaseModel):
    """Score for a single RACE evaluation dimension.

    >>> d = RACEDimension(dimension="comprehensiveness", score=4.0, max_score=5.0, explanation="Covers most mechanisms")
    >>> d.normalized_score
    0.8
    """

    dimension: str = Field(..., description="E.g. comprehensiveness, accuracy, organization, terminology")
    score: float
    max_score: float = 5.0
    explanation: Optional[str] = None

    @property
    def normalized_score(self) -> float:
        """Return score normalized to 0-1 range."""
        return self.score / self.max_score if self.max_score > 0 else 0.0


class RACEScore(BaseModel):
    """RACE scoring result: report quality assessment.

    >>> s = RACEScore(dimensions=[
    ...     RACEDimension(dimension="comprehensiveness", score=4.0, max_score=5.0),
    ...     RACEDimension(dimension="accuracy", score=3.0, max_score=5.0),
    ... ])
    >>> s.overall_score
    0.7
    """

    dimensions: list[RACEDimension] = Field(default_factory=list)
    overall_explanation: Optional[str] = None

    @property
    def overall_score(self) -> float:
        """Weighted average of normalized dimension scores."""
        if not self.dimensions:
            return 0.0
        return sum(d.normalized_score for d in self.dimensions) / len(self.dimensions)


class EvalResult(BaseModel):
    """Complete evaluation result for a single task + provider combination.

    >>> r = EvalResult(task_id="test", provider="falcon")
    >>> r.task_id
    'test'
    """

    task_id: str
    provider: str
    model: Optional[str] = None
    fact_score: Optional[FACTScore] = None
    claim_recall_score: Optional[ClaimRecallScore] = None
    race_score: Optional[RACEScore] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
