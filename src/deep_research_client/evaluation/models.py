"""Pydantic models for scoring results.

The eval sets themselves - tasks, reference answers, rubrics - are defined by
the LinkML schema in ``evaluation.yaml`` and generated into ``datamodel.py``.
What lives here is the other half: results, which are computed rather than
authored, and so are hand-written Pydantic rather than schema-generated.

Scoring follows several complementary frameworks:
- MCQ: accuracy, coverage and precision for multiple-choice benchmarks
- FACT: citation verification (does the cited paper support the claim?)
- RACE: report quality (comprehensiveness, accuracy, organization, terminology)
- Claim recall: does the report cover the reference claims?
"""

from typing import Optional

from pydantic import BaseModel, Field

from .datamodel import ScoreDisposition

# Note: the schema-generated classes in datamodel.py are configured with
# ``use_enum_values``, so an enum slot read back off one of those is a plain
# string, while the same enum on a model here stays an enum member. Compare
# enum-valued fields with ``==`` rather than ``is`` so both forms behave.


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
    matched: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the DR output covers this claim. None when the judge could "
            "not be asked or returned no parseable verdict -- distinct from "
            "False, which is a ruling that the report does not cover it."
        ),
    )
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
    unjudged_claims: int = Field(
        default=0,
        description=(
            "Claims the judge could not rule on. Excluded from `claim_recall`: "
            "counting them as uncovered made recall fall with the judge's "
            "uptime, scoring a provider for someone else's outage."
        ),
    )
    claim_recall: float = Field(
        ..., description="matched / claims the judge ruled on (see unjudged_claims)"
    )
    judged_chars: Optional[int] = Field(
        default=None, description="Characters of the report sent to the judge."
    )
    report_chars: Optional[int] = Field(
        default=None,
        description=(
            "Characters the report actually had. Greater than `judged_chars` "
            "means a claim may be covered in a tail the judge never saw."
        ),
    )
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
    score: Optional[float] = Field(
        default=None,
        description=(
            "The judge's score, or None when it could not be asked or returned "
            "nothing parseable. None rather than a mid-scale default: a 3.0 "
            "written on failure is indistinguishable from a genuine 3."
        ),
    )
    max_score: float = 5.0
    explanation: Optional[str] = None

    @property
    def normalized_score(self) -> float:
        """Return score normalized to 0-1 range, or 0.0 if unscored."""
        if self.score is None or self.max_score <= 0:
            return 0.0
        return self.score / self.max_score


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
    judged_chars: Optional[int] = Field(
        default=None,
        description="Characters of the report sent to the judge.",
    )
    report_chars: Optional[int] = Field(
        default=None,
        description=(
            "Characters the report actually had. Greater than `judged_chars` "
            "means the tail was not read, so coverage is understated -- the "
            "same thing `is_partial` says about an eval set, one layer down."
        ),
    )

    @property
    def scored_dimensions(self) -> list[RACEDimension]:
        """The dimensions the judge actually returned a score for.

        A dimension whose judge call failed used to be recorded as 3.0 out of 5,
        indistinguishable from a genuine middling verdict -- so a run where the
        endpoint was down reported mid-scale quality for every report.

        >>> s = RACEScore(dimensions=[
        ...     RACEDimension(dimension="comprehensiveness", score=4.0, max_score=5.0),
        ...     RACEDimension(dimension="accuracy", score=None, max_score=5.0),
        ... ])
        >>> len(s.scored_dimensions), s.unscored_count
        (1, 1)
        """
        return [d for d in self.dimensions if d.score is not None]

    @property
    def unscored_count(self) -> int:
        """Dimensions the judge could not be asked about, or did not answer."""
        return len(self.dimensions) - len(self.scored_dimensions)

    @property
    def overall_score(self) -> float:
        """Weighted average over the dimensions that were actually scored.

        0.0 when none were, which `unscored_count` distinguishes from a report
        that genuinely scored zero.
        """
        scored = self.scored_dimensions
        if not scored:
            return 0.0
        return sum(d.normalized_score for d in scored) / len(scored)


# ---------------------------------------------------------------------------
# Intrinsic (LLM-free) scoring models
# ---------------------------------------------------------------------------


class CitationExistence(BaseModel):
    """Result of checking whether a single citation resolves to a real paper.

    >>> c = CitationExistence(citation_id="PMID:7913883", exists=True, title="Mutations in FGFR3...")
    >>> c.exists
    True
    """

    citation_id: str = Field(..., description="Normalized citation ID (PMID:xxx or DOI:xxx)")
    exists: bool = Field(..., description="Whether the citation resolves to a real paper")
    title: Optional[str] = Field(default=None, description="Paper title if retrieved")
    year: Optional[int] = Field(default=None, description="Publication year if retrieved")
    error: Optional[str] = Field(default=None, description="Error if lookup failed")


class CitationVerifiabilityScore(BaseModel):
    """Aggregate score for citation existence checking.

    >>> s = CitationVerifiabilityScore(total_citations=5, verified_exist=4, verifiability=0.8)
    >>> s.verifiability
    0.8
    """

    total_citations: int = Field(..., description="Total unique citations checked")
    verified_exist: int = Field(..., description="Citations that resolve to real papers")
    unresolvable: int = Field(
        default=0,
        description=(
            "Citations whose lookup errored. Excluded from `verifiability`: a "
            "CrossRef or PubMed outage otherwise reports every DOI in the "
            "report as hallucinated."
        ),
    )
    verifiability: float = Field(
        ...,
        description=(
            "Fraction of the citations that could be looked up which resolve to "
            "real papers. Read with `unresolvable`, which distinguishes "
            "'all fabricated' from 'none checkable'."
        ),
    )
    year_distribution: dict[int, int] = Field(default_factory=dict, description="Publication year -> count")
    median_year: Optional[int] = Field(default=None, description="Median publication year")
    citations: list[CitationExistence] = Field(default_factory=list, description="Per-citation results")


class CitationAlignmentResult(BaseModel):
    """Result of checking whether a cited paper's title aligns with the claim.

    >>> r = CitationAlignmentResult(
    ...     citation_id="PMID:7913883",
    ...     claim_text="FGFR3 mutations cause achondroplasia",
    ...     paper_title="Mutations in FGFR3 cause achondroplasia",
    ...     aligned=True,
    ...     shared_terms=["FGFR3", "mutations", "achondroplasia"],
    ... )
    >>> r.aligned
    True
    """

    citation_id: str
    claim_text: str
    paper_title: str
    aligned: bool = Field(..., description="Whether paper title shares key biomedical terms with claim")
    shared_terms: list[str] = Field(default_factory=list, description="Key terms shared between claim and title")
    term_overlap_score: float = Field(default=0.0, description="Jaccard-like overlap of key terms")


class CitationAlignmentScore(BaseModel):
    """Aggregate citation-claim alignment score.

    >>> s = CitationAlignmentScore(total_checked=10, aligned_count=7, alignment_rate=0.7)
    >>> s.alignment_rate
    0.7
    """

    total_checked: int
    aligned_count: int
    alignment_rate: float = Field(..., description="Fraction of checked citations where title aligns with claim")
    results: list[CitationAlignmentResult] = Field(default_factory=list)


class FactualSpotCheck(BaseModel):
    """Result of verifying a single extractable fact against a known value.

    >>> f = FactualSpotCheck(
    ...     fact_name="chromosome_location",
    ...     expected="17q21.31",
    ...     found_in_report="17q21",
    ...     correct=True,
    ... )
    >>> f.correct
    True
    """

    fact_name: str = Field(..., description="Name of the fact being checked")
    expected: str = Field(..., description="Known correct value")
    found_in_report: Optional[str] = Field(default=None, description="Value found in DR output, if any")
    correct: Optional[bool] = Field(default=None, description="Whether report value matches expected")
    present: bool = Field(default=False, description="Whether the fact is mentioned at all")
    compared: bool = Field(
        default=False,
        description=(
            "Whether this check actually compared a captured value against "
            "`expected`. False for a presence-only check, which is `correct` "
            "whenever it matched and so cannot be evidence of accuracy."
        ),
    )


class FactualSpotCheckScore(BaseModel):
    """Aggregate score for factual spot checks.

    >>> s = FactualSpotCheckScore(total_checks=5, present_count=4, correct_count=3, presence_rate=0.8, accuracy_rate=0.75)
    >>> s.accuracy_rate
    0.75
    """

    total_checks: int
    present_count: int = Field(..., description="Facts mentioned in the report")
    correct_count: int = Field(..., description="Facts correctly stated")
    compared_count: int = Field(
        default=0,
        description=(
            "Checks that actually compared a captured value against an expected "
            "one. A check whose pattern captures nothing cannot disagree with "
            "its expected value, so counting it as accuracy inflates the rate: "
            "nine presence-only checks and one wrong accuracy check reported "
            "0.9 accurate."
        ),
    )
    presence_rate: float = Field(..., description="present / total")
    accuracy_rate: float = Field(
        ...,
        description=(
            "correct / compared, over the checks that made a comparison. 0.0 "
            "when nothing was comparable -- read it with compared_count, which "
            "distinguishes 'wrong about everything' from 'measured nothing'."
        ),
    )
    checks: list[FactualSpotCheck] = Field(default_factory=list)


class TopicCoverage(BaseModel):
    """Whether a specific expected topic/section is covered in the report.

    >>> t = TopicCoverage(topic="DNA repair", covered=True, evidence_snippet="BRCA1 plays a role in DNA repair")
    >>> t.covered
    True
    """

    topic: str
    covered: bool
    evidence_snippet: Optional[str] = Field(default=None, description="Brief text from report demonstrating coverage")
    keywords_found: list[str] = Field(default_factory=list)


class TopicCoverageScore(BaseModel):
    """Aggregate topic coverage score.

    >>> s = TopicCoverageScore(total_topics=5, covered_count=4, coverage_rate=0.8)
    >>> s.coverage_rate
    0.8
    """

    total_topics: int
    covered_count: int
    coverage_rate: float
    topics: list[TopicCoverage] = Field(default_factory=list)


class IntrinsicScore(BaseModel):
    """Collection of all LLM-free intrinsic quality scores.

    These scores verify properties of the DR output without needing
    an LLM judge or ground truth reference overlap.

    >>> s = IntrinsicScore()
    >>> s.citation_verifiability is None
    True
    """

    citation_verifiability: Optional[CitationVerifiabilityScore] = None
    citation_alignment: Optional[CitationAlignmentScore] = None
    factual_spot_checks: Optional[FactualSpotCheckScore] = None
    topic_coverage: Optional[TopicCoverageScore] = None


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
    intrinsic_score: Optional[IntrinsicScore] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Multiple-choice scoring models
# ---------------------------------------------------------------------------


class MCQAnswer(BaseModel):
    """One provider's graded answer to one multiple-choice task.

    ``correct`` is only meaningful when ``disposition`` is SCORED. An abstention
    is not a wrong answer, and neither a provider failure nor an extraction
    failure says anything about whether the provider knew the answer, so all
    three leave ``correct`` false while being counted separately.

    >>> a = MCQAnswer(task_id="t1", provider="falcon", chosen_letter="B",
    ...               disposition="SCORED", correct=True)
    >>> a.correct
    True
    """

    task_id: str = Field(..., description="ID of the evaluation task")
    provider: str = Field(..., description="Provider or arm that produced the answer")
    chosen_letter: Optional[str] = Field(default=None, description="Option letter the provider chose")
    chosen_text: Optional[str] = Field(default=None, description="Text of the chosen option")
    disposition: ScoreDisposition = Field(..., description="What became of this task-provider pair")
    correct: bool = Field(default=False, description="Whether the chosen option was the ideal answer")
    error: Optional[str] = Field(default=None, description="Provider error, when the call failed")


class MCQScore(BaseModel):
    """Aggregate multiple-choice score, following LAB-Bench's metric definitions.

    Accuracy is over all questions, coverage is the fraction attempted, and
    precision is over attempted questions only. Reporting accuracy without
    coverage hides whether a low score means wrong answers or declined ones.

    >>> s = MCQScore(total=10, attempted=8, correct=6, accuracy=0.6,
    ...              coverage=0.8, precision=0.75)
    >>> s.precision
    0.75
    """

    total: int = Field(..., description="Questions in the eval set")
    attempted: int = Field(..., description="Questions the provider chose an option for")
    correct: int = Field(..., description="Questions answered correctly")
    abstained: int = Field(default=0, description="Questions the provider declined")
    extraction_failures: int = Field(
        default=0,
        description="Responses no option could be recovered from; a harness defect, not a provider one",
    )
    provider_errors: int = Field(default=0, description="Calls that failed outright")
    accuracy: float = Field(..., description="correct / total")
    coverage: float = Field(..., description="attempted / total")
    precision: float = Field(..., description="correct / attempted")
    answers: list[MCQAnswer] = Field(default_factory=list, description="Per-task graded answers")
