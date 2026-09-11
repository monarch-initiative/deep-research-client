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

Several field descriptions here carry a `History:` paragraph below a blank
line. That is deliberate and not cruft: each records a reading the field
previously invited and a number it produced, and each was written after that
reading cost a review round. The statement a consumer needs is always the
first sentence, so a reader can stop there.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

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

    total_citations: int = Field(
        ...,
        description=(
            "Every citation-claim pair found in the report. Held "
            "`len(checkable)` until `unjudged_citations` was added beside it "
            "and made the mismatch visible: the one name in this model that "
            "did not mean what it says. The accuracy denominator is "
            "`total_citations - unjudged_citations`."
        ),
    )
    verified_citations: int = Field(..., description="Citations that support their claims")
    citation_accuracy: float = Field(
        ...,
        description=(
            "verified / judged, where judged is "
            "`total_citations - unjudged_citations`. Not over "
            "`total_citations`: a pair the judge never ruled on is not a pair "
            "whose citation failed to support its claim."
        ),
    )
    effective_citations: int = Field(..., description="Count of verifiably supported citations")
    unjudged_citations: int = Field(
        default=0,
        description=(
            "Citation-claim pairs the judge returned no verdict on, and which "
            "are therefore in none of the counts above: a DOI, whose abstract "
            "this scorer cannot fetch; a PMID with no abstract; a reply with "
            "no verdict in it. Recorded for the same reason as its three "
            "siblings -- a report whose citations are all DOIs otherwise "
            "reports accuracy 0.00 over 0 of 0, which reads as a report whose "
            "citations support nothing."
        ),
    )
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
    max_score: float = Field(
        default=5.0,
        description=(
            "The scale this dimension was scored on. Deliberately NOT "
            "constrained to be positive: `gt=0` would make a non-positive "
            "scale unconstructible, and with it `normalized_score`'s guard "
            "dead -- and that guard is the only thing separating it from "
            "`score is None`, which is the distinction three accessors have "
            "now disagreed about. A degenerate scale is a describable state "
            "with a defined rendering (`unscored`), not a validation error"
        ),
    )
    explanation: Optional[str] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def normalized_score(self) -> Optional[float]:
        """Score normalized to 0-1, or None when the dimension was not scored.

        None rather than 0.0: this property is public, and returning the worst
        possible score for a dimension nobody measured is the defect the
        Optional `score` was introduced to remove, one accessor down. A caller
        that averages these without filtering now gets a TypeError rather than
        a plausible number.

        >>> RACEDimension(dimension="d", score=None).normalized_score is None
        True

        A `computed_field`, so `--output` carries it. The CLI decides whether
        to print a number or "unscored" from this property; a JSON consumer
        holding `{"score": 3.0, "max_score": 0.0}` would otherwise have to
        re-derive the predicate to learn what the terminal said.
        `CitationAlignmentScore.total_pairs` is the precedent.

        >>> "normalized_score" in RACEDimension(dimension="d", score=4.0).model_dump()
        True
        """
        if self.score is None or self.max_score <= 0:
            return None
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

        The one derivation here that is NOT a `computed_field`, deliberately:
        it returns model objects, so serialising it would copy every scored
        `RACEDimension` into the JSON beside `dimensions`. A consumer needs
        the COUNT, not the objects, and can take it as
        `len(dimensions) - unscored_count` -- both of which the artifact
        carries. Recorded because three siblings do carry the decorator and
        the odd one out otherwise reads as an oversight.

        A dimension whose judge call failed used to be recorded as 3.0 out of 5,
        indistinguishable from a genuine middling verdict -- so a run where the
        endpoint was down reported mid-scale quality for every report.

        >>> s = RACEScore(dimensions=[
        ...     RACEDimension(dimension="comprehensiveness", score=4.0, max_score=5.0),
        ...     RACEDimension(dimension="accuracy", score=None, max_score=5.0),
        ... ])
        >>> len(s.scored_dimensions), s.unscored_count
        (1, 1)

        Filtered on `normalized_score`, not on `score`. Those answered
        differently for a dimension carrying a raw score against a
        non-positive scale: `normalized_score` refuses it, `score is not None`
        admitted it, and the CLI reads *this* property to decide whether
        anything was measured. So four dimensions with `max_score=0` took the
        measured branch, and `overall_score` -- a mean over an empty list
        after its own filter -- printed `overall=0.00`, which is the reading
        the branch exists to prevent. Two accessors, one question.
        (`overall_score` is `None` over an empty list now, not 0.0; the
        `0.00` above is what the defect printed.)

        >>> degenerate = RACEScore(dimensions=[
        ...     RACEDimension(dimension="comprehensiveness", score=3.0, max_score=0.0),
        ... ])
        >>> len(degenerate.scored_dimensions), degenerate.unscored_count
        (0, 1)
        """
        return [d for d in self.dimensions if d.normalized_score is not None]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def unscored_count(self) -> int:
        """Dimensions with no usable score.

        Three ways in, not two: the judge could not be asked, it did not
        answer, or it answered against a non-positive scale -- which
        `normalized_score` refuses and `scored_dimensions` therefore excludes.
        The third was added when those two accessors were made to agree, and
        this list is their complement, so it gained the case with them.
        """
        return len(self.dimensions) - len(self.scored_dimensions)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overall_score(self) -> Optional[float]:
        """Mean over the dimensions that were actually scored, or None.

        None when none were scored, matching `RACEDimension.normalized_score`
        and `MCQScore.precision`: a rate that reaches a reader over a zero
        denominator is absent rather than zero, because no adjacent field
        reliably travels with it. `unscored_count` says how many are missing,
        but it is a disambiguator beside a rate -- the trade the citation
        lines and the `--grade` precision column both rejected.

        This was a float, on the argument that the zero could not reach a
        reader because the CLI gates on `scored_dimensions` before formatting
        it. Making it a `computed_field` in the same commit retired that
        argument without retiring the sentence: `eval score --output` dumps
        the whole result and does NOT go through the gate, so a report whose
        judge was unreachable wrote `overall_score: 0.0` into the artifact --
        a measured-looking zero for a report nothing was measured on. Once a
        surface is added, every rule about what may reach a reader applies to
        it.

        A `computed_field`, with `unscored_count`, so the headline RACE number
        and the count that distinguishes "the judge was down" from "a terrible
        report" both reach `--output`. They were plain properties, so neither
        did.

        >>> dumped = RACEScore(dimensions=[
        ...     RACEDimension(dimension="d", score=4.0, max_score=5.0)]).model_dump()
        >>> dumped["overall_score"], dumped["unscored_count"]
        (0.8, 0)

        The artifact says "not measured" where the terminal does:

        >>> down = RACEScore(dimensions=[
        ...     RACEDimension(dimension="d", score=None, max_score=5.0),
        ...     RACEDimension(dimension="e", score=None, max_score=5.0)])
        >>> dumped = down.model_dump()
        >>> print(dumped["overall_score"], dumped["unscored_count"])
        None 2
        """
        # The `if` is TYPE NARROWING, not a runtime guard, and cannot be
        # deleted: `scored_dimensions` already filters on exactly this
        # predicate, so it drops nothing -- but without it the comprehension
        # is `list[Optional[float]]` and `sum` fails mypy. Four unreachable
        # branches have been removed from this branch on fail-fast grounds and
        # this one reads like a fifth.
        scored = [
            d.normalized_score for d in self.scored_dimensions
            if d.normalized_score is not None
        ]
        if not scored:
            return None
        return sum(scored) / len(scored)


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
    exists: Optional[bool] = Field(
        ...,
        description=(
            "Whether the citation resolved to a real paper. True; False when "
            "the registry says it does not exist; None when nothing was "
            "established -- a lookup that raised, a 200 whose body answers "
            "about no identifier, or an identifier kind this scorer has no "
            "resolver for and never attempts.\n\n"
            "History: a `bool` answering a three-valued question meant the "
            "per-citation record of a real PMC article read `exists: false` "
            "in `--output`, so a consumer reading the obvious field got "
            "exactly the answer the aggregate was fixed to stop giving. "
            "`lookup_failed` still discriminates; nothing obliges a reader to "
            "consult it."
        ),
    )
    title: Optional[str] = Field(default=None, description="Paper title if retrieved")
    year: Optional[int] = Field(default=None, description="Publication year if retrieved")
    error: Optional[str] = Field(
        default=None,
        description=(
            "Why this citation has the `exists` it has, when there is anything "
            "to say. `exists` and `lookup_failed` are the fields to branch on; "
            "this one is for a human reading `--output`.\n\n"
            "History: it said 'why the citation did not resolve', which is "
            "also the easiest way to misread it. The no-resolver branch sets "
            "it (`No resolver for this identifier kind: PMC`) on a citation "
            "that very likely does resolve and that `lookup_failed` describes "
            "as never attempted, and an authoritative negative sets it too, "
            "which is the opposite of a failed lookup."
        ),
    )
    lookup_failed: bool = Field(
        default=False,
        description=(
            "Whether nothing was learned about this citation: a lookup that "
            "raised (a timeout, a 5xx, a connection error), a 200 whose body "
            "answers about no identifier, or an identifier kind this scorer "
            "has no resolver for, which is never looked up at all. Distinct "
            "from `error`, which is also set "
            "for an authoritative negative: NCBI reports an unknown PMID as a "
            "per-uid error, and that is a fabricated citation rather than a "
            "failed lookup. Only `lookup_failed` leaves the verifiability rate."
        ),
    )


class CitationVerifiabilityScore(BaseModel):
    """Aggregate score for citation existence checking.

    >>> s = CitationVerifiabilityScore(total_citations=5, verified_exist=4, verifiability=0.8)
    >>> s.verifiability
    0.8
    """

    total_citations: int = Field(
        ...,
        description=(
            "Distinct citations found in the report. `find_reference_ids` "
            "de-duplicates on the normalised identifier and records a `count` "
            "of textual mentions, so a PMID cited ten times is one lookup and "
            "one count here -- repeated citations are not weighted. Not "
            "'checked', though: the checked count is "
            "`total_citations - unresolvable`, which is what `verifiability` "
            "is over and what the CLI prints. The sibling name in `FACTScore` "
            "carried the same mismatch.\n\n"
            "History: an earlier correction of this description asserted that "
            "nothing de-duplicates, which was the opposite of what the "
            "extractor does."
        ),
    )
    verified_exist: int = Field(..., description="Citations that resolve to real papers")
    unresolvable: int = Field(
        default=0,
        description=(
            "Citations nothing was learned about, so they are excluded from "
            "`verifiability`: a lookup that raised, a 200 whose body answers "
            "about no identifier -- NCBI's `esummaryresult` envelope or a "
            "rate-limit page that still parses as JSON, CrossRef's "
            "no-work-record body -- and an identifier kind this scorer has no "
            "resolver for, such as a PMC accession or a GEO series, which is "
            "never attempted. A citation the registry says does not exist is "
            "NOT here: that is fabricated, and stays in the rate.\n\n"
            "History, because this field has been wrong twice: it said 'only "
            "transport failures' for a commit after the no-resolver cause was "
            "added, and then 'two causes' while the code had three. Both were "
            "tallies. The causes are listed rather than counted now."
        ),
    )
    verifiability: float = Field(
        ...,
        description=(
            "Fraction of the citations this scorer actually looked up which "
            "resolve to real papers -- not of the citations that 'could be' "
            "looked up, which is a different set now that an identifier kind "
            "with no resolver is excluded without being attempted. Read with "
            "`unresolvable`, which distinguishes 'all fabricated' from "
            "'none checkable'."
        ),
    )
    year_distribution: dict[int, int] = Field(default_factory=dict, description="Publication year -> count")
    median_year: Optional[int] = Field(
        default=None,
        description=(
            "Upper middle publication year, so the value is always a year "
            "that appears in the citations rather than an average of two."
        ),
    )
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

    total_checked: int = Field(
        ...,
        description=(
            "Citation-claim pairs a comparison was actually made for, which "
            "is `alignment_rate`'s denominator. NOT every pair -- see "
            "`total_pairs`"
        ),
    )
    aligned_count: int = Field(
        ...,
        description="Checked pairs whose title supports the claim",
    )
    unresolvable: int = Field(
        default=0,
        description=(
            "Citation-claim pairs nothing could be compared for: a lookup "
            "that failed, a record that resolved and carries no title, or a "
            "citation this scorer never looks up at all -- one with no "
            "identifier, one that would not normalise, or one of a kind it has "
            "no resolver for, such as a PMC accession or a GEO series. A "
            "citation the registry says does not exist is NOT here -- that is "
            "a finding, and counts against `alignment_rate`. Recorded because "
            "without it a PubMed outage and a report whose citations support "
            "nothing read the same: the CLI can only say `not measured, N "
            "with nothing to align against` for the outage because this count "
            "exists."
        ),
    )
    alignment_rate: float = Field(..., description="Fraction of checked citations where title aligns with claim")
    results: list[CitationAlignmentResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_pairs(self) -> int:
        """Every citation-claim pair found, checked or not.

        The sibling verifiability score carries this as a stored field, so its
        CLI line asks `not cv.total_citations` where this one had to ask
        `not total_checked and not unresolvable`. Two spellings of one question,
        side by side in the same block, is how the two lines drifted apart
        before.

        A `computed_field`, so it reaches `--output` too. A plain property
        made the two CLI lines ask one question while a JSON consumer still
        had to add two numbers here and read one from the sibling score.

        >>> CitationAlignmentScore(total_checked=3, unresolvable=2,
        ...                        aligned_count=2, alignment_rate=0.67).total_pairs
        5
        >>> "total_pairs" in CitationAlignmentScore(
        ...     total_checked=3, unresolvable=2, aligned_count=2,
        ...     alignment_rate=0.67).model_dump()
        True
        """
        return self.total_checked + self.unresolvable


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
    expected: Optional[str] = Field(
        default=None,
        description=(
            "Known correct value, or None for a presence-only check, which has "
            "no expected value to hold the report to. Written as an empty "
            "string until it was pointed out that a reader of the per-check "
            "detail could then not tell 'no value was specified' from 'the "
            "value specified was empty'."
        ),
    )
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
    correct_count: int = Field(
        ...,
        description=(
            "Checks recorded correct, which includes every presence-only check "
            "that matched. Not the numerator of `accuracy_rate` -- that is over "
            "`compared_count`, the checks that compared a captured value. Read "
            "this one as coverage, not as agreement."
        ),
    )
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
    three leave ``correct`` unset while being counted separately.

    >>> a = MCQAnswer(task_id="t1", provider="falcon", chosen_letter="B",
    ...               disposition="SCORED", correct=True)
    >>> a.correct
    True

    Three-valued, and it was a `bool` defaulting False. `score_by_arm` bridges
    `CellResult.correct`, which is `Optional[bool]`, into this one -- and did
    it with `bool(cell.correct)`, so a resumed cell carrying SCORED with no
    recorded correctness became a WRONG answer: counted in `attempted`, absent
    from `correct`, lowering both accuracy and precision with nothing saying
    so. That is the worse half of the disagreement fixed one commit earlier.
    `ABSTAINED` with `correct=True` gave `precision 2.0`, which announces
    itself; a real answer scored wrong produces a number in range.

    >>> MCQAnswer(task_id="t1", provider="p", disposition="SCORED").correct is None
    True
    """

    task_id: str = Field(..., description="ID of the evaluation task")
    provider: str = Field(..., description="Provider or arm that produced the answer")
    chosen_letter: Optional[str] = Field(default=None, description="Option letter the provider chose")
    chosen_text: Optional[str] = Field(default=None, description="Text of the chosen option")
    disposition: ScoreDisposition = Field(..., description="What became of this task-provider pair")
    correct: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the chosen option was the ideal answer, and None when "
            "that was never established -- every disposition but SCORED, and "
            "a SCORED record whose correctness is missing, which only a "
            "hand-edited or older-format `cell.json` produces. `score_mcq` "
            "counts such a record in `attempted` -- an option WAS chosen, so "
            "coverage must say so -- and out of both `correct` and "
            "`precision`'s denominator, where it is not evidence either way. "
            "See `MCQScore.unusable`, which counts them"
        ),
    )
    error: Optional[str] = Field(default=None, description="Provider error, when the call failed")


class MCQScore(BaseModel):
    """Aggregate multiple-choice score, following LAB-Bench's metric definitions.

    Accuracy is over all questions, coverage is the fraction attempted, and
    precision is over `judged` -- the attempted questions whose correctness
    was actually established, `attempted - unusable`, which equals
    `attempted` unless a record came back with no correctness at all.
    Reporting accuracy without coverage hides whether a low score means wrong
    answers or declined ones.

    Every task has exactly one disposition, so `attempted` and the four
    failure counts partition `total`; a score whose counts do not add up is
    rejected rather than written to a file a spreadsheet subtracts from.

    Three guards live here and nowhere else in this module: rates derived
    INSTEAD of stored beside the counts they duplicate, the accounting
    validator below, and `ge=0` on every count. The first needs that
    qualifier: `computed_field` itself is used all over this module --
    `RACEDimension.normalized_score`, `RACEScore.unscored_count` and
    `overall_score`, `CitationAlignmentScore.total_pairs`, the last of which
    `normalized_score` names as the precedent for the technique. None of
    those duplicates a stored field, so nothing can disagree with them and
    deriving them was never this guard. Scoped to this class
    deliberately, not because the arguments are special to it. The six
    scores it sits beside -- `FACTScore`, `ClaimRecallScore`,
    `CitationVerifiabilityScore`, `CitationAlignmentScore`,
    `FactualSpotCheckScore` and `TopicCoverageScore` -- predate this branch
    on `main` and keep settable rates duplicating their own counts, so
    `CitationVerifiabilityScore(total_citations=1, verified_exist=99,
    verifiability=0.1)` is accepted today. The asymmetry runs the wrong way
    for blast radius: those six reach `eval score --output` through
    `EvalResult`, where this one reaches only `scores.tsv`. Hardening them is
    a change to code this branch does not otherwise touch, so it belongs in
    its own commit rather than widening this one.

    >>> s = MCQScore(total=10, attempted=8, correct=6, abstained=2)
    >>> s.accuracy, s.coverage, s.precision
    (0.6, 0.8, 0.75)
    """

    # The three rates became `computed_field`s, and pydantic DROPS unknown
    # keyword arguments by default -- so a call site still passing
    # `accuracy=0.6` would have had it silently ignored rather than flagged.
    # Five stale ones failed loudly instead, which is why this is here.
    #
    # The cost, and it is a real one: `model_dump()` INCLUDES computed
    # fields, so `MCQScore(**score.model_dump())` raises on all three. This
    # is the only score in the module that does not round-trip through its
    # own dump -- `RACEScore` and `CitationAlignmentScore` have computed
    # fields too and keep the default `extra="ignore"`, which drops them on
    # the way back in. Taken deliberately: `on_scores` hands these to library
    # callers, and a caller that persists and reloads should rebuild from the
    # COUNTS (`{k: v for k, v in dump.items() if k in MCQScore.model_fields}`)
    # rather than from a rate that was only ever a view of them. Silently
    # accepting a stale rate is the failure this pair exists to prevent.
    #
    # The recipe is EXECUTED rather than only written here, by the test named
    # `test_the_recipe_rebuilds_a_score_its_own_dump_cannot`, which asserts
    # both that the raw dump is refused and that the filtered one rebuilds an
    # equal score. That test compares the expression above against its own
    # source AND against this file, so the copy a caller pastes cannot drift
    # from the one that runs; a separate test fails if this comment ever
    # names a test that no longer exists.
    #
    # Naming a test from `src/` is deliberate and is the only place this
    # package does it. The alternative is to describe the recipe without
    # saying what runs it, which is what left this comment unexecuted for
    # four commits.
    model_config = ConfigDict(extra="forbid")

    total: int = Field(..., ge=0, description="Questions in the eval set")
    attempted: int = Field(
        ...,
        ge=0,
        description=(
            "Questions the provider chose an option for. Coverage's numerator, "
            "and NOT precision's denominator -- those differ by `unusable`"
        ),
    )
    correct: int = Field(
        ...,
        ge=0,
        description=(
            "Questions answered correctly AND scored. The second half is the "
            "invariant `score_mcq`'s numerator filter introduced: a record "
            "carrying `correct=True` on any other disposition is not counted. "
            "The bound is `correct <= judged <= attempted <= total`, and it "
            "is the model validator that holds it -- `correct <= attempted` "
            "is the weaker half and was what this said, credited to the "
            "producer's filter, which held it for `score_mcq`'s OUTPUT and "
            "not for the type: `MCQScore(total=1, attempted=1, correct=5)` "
            "was accepted and wrote `precision 5.0`. `judged` is the bound "
            "that matters, because it is what `precision` divides by"
        ),
    )
    abstained: int = Field(
        default=0, ge=0, description="Questions the provider declined"
    )
    extraction_failures: int = Field(
        default=0,
        ge=0,
        description="Responses no option could be recovered from; a harness defect, not a provider one",
    )
    provider_errors: int = Field(
        default=0, ge=0, description="Calls that failed outright"
    )
    skipped: int = Field(
        default=0,
        ge=0,
        description=(
            "Pairs that were not run -- typically because a filter "
            "excluded them, which is how `ScoreDisposition.SKIPPED` states "
            "it; the sibling `CellStatus.SKIPPED` also covers a pair a "
            "previous run had already completed. Nothing emits it today. "
            "Counted "
            "anyway so the four disposition counts ACCOUNT FOR `total` "
            "alongside `attempted`: without it, `scores.tsv` had no column "
            "for `SKIPPED` and no way to reach it, since its writer "
            "documented the other three as partitioning the remainder -- "
            "which told a reader the subtraction was always zero. The "
            "invariant is `attempted + abstained + extraction_failures + "
            "provider_errors + skipped == total`, and `unusable` is a subset "
            "of `attempted` rather than a sixth part of it"
        ),
    )
    unusable: int = Field(
        default=0,
        ge=0,
        description=(
            "Attempted questions whose correctness was never recorded, which "
            "only a hand-edited or older-format `cell.json` produces. They are "
            "in `attempted` -- an option was chosen -- and out of `precision`'s "
            "denominator, because nothing can be said about whether they were "
            "right. Counted here rather than deducted silently: excluding them "
            "from `attempted` instead fixed precision by making coverage "
            "report an attempt that was made as one that was not"
        ),
    )
    answers: list[MCQAnswer] = Field(default_factory=list, description="Per-task graded answers")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def accuracy(self) -> float:
        """correct / total, and 0.0 when `total` is 0.

        Stays a float where `precision` is Optional, and the line is where the
        number is RENDERED rather than where it is computed: an arm reaches
        the `--grade` table only with at least one disposed cell, so a total
        of 0 never reaches a reader, while an attempted of 0 does.

        Unlike `precision`, this rate keeps `unusable` records in its
        denominator and they cannot be in its numerator, so they cost accuracy
        exactly as a wrong answer would. Deliberate, and the same treatment
        `EXTRACTION_FAILED` gets: accuracy is over every question asked, which
        is LAB-Bench's definition, and a question the harness cannot report an
        answer for was not answered correctly.
        """
        return self.correct / self.total if self.total else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def coverage(self) -> float:
        """attempted / total, and 0.0 when `total` is 0, as `accuracy` is.

        This is the field that says whether an arm attempted anything, but it
        does NOT say why `precision` is absent: an arm can reach a `None`
        precision at any coverage, so 0.0 here means nothing was attempted and
        says nothing about the other way in. Over every question an option was
        chosen for, INCLUDING the ones `unusable` counts: an option was
        chosen on those too, so they belong here. They are left out of
        precision's denominator rather than counted against it, and they
        lower accuracy.

        History: this said those records "cost precision, not coverage",
        which is backwards -- leaving the denominator raises the rate. It
        survived the sweep that corrected the same claim on the how-to
        because 4654ff1 MOVED this text verbatim from a `Field(description=)`
        into a property docstring, and because the phrase wrapped across two
        lines, so a line-based `grep "cost precision"` matched nothing.
        """
        return self.attempted / self.total if self.total else 0.0

    @property
    def judged(self) -> int:
        """`attempted - unusable`: precision's denominator.

        A plain `@property`, not a `computed_field`, because it buys a
        consumer nothing: `judged` is two counts subtracted, and both keys
        are already in the dump, so anyone who wants it writes the same
        expression this line does. The `extra="forbid"` trade above is the
        standing reason not to add keys that buy nothing -- not a cost that
        grows with each one. That gap is binary and already open: a fourth
        computed field would put one more name in the error, and the
        documented way back filters on `model_fields`, so it excludes every
        computed field however many there are.

        History: the definition was computed in the rate, again in the
        validator, and restated in four descriptions -- two independent
        computations of one definition, the second-source-of-truth shape one
        notch weaker than a stored copy. The first version of this paragraph
        gave a different reason, that a `computed_field` would reach
        `--output`: no `--output` in this CLI carries an `MCQScore` (`eval
        score` dumps an `EvalResult`, which has no MCQ member; `eval run`'s
        flag is `--output-dir`), and `scores.tsv` is written from an explicit
        column tuple read off attributes, so it would not have gained a
        column either.
        """
        return self.attempted - self.unusable

    @computed_field  # type: ignore[prop-decorator]
    @property
    def precision(self) -> Optional[float]:
        """correct / `judged`, and None when `judged` is 0.

        None when NO attempted answer had its correctness established. That is
        the condition, stated instead of its causes because the causes
        compose. An arm can attempt nothing (every question declined, the
        endpoint down all run, every response unreadable by the extractor, the
        pair skipped, or any mixture of those), or attempt and have every
        attempt come back with no recorded correctness. So `coverage` beside
        the dash is not one of two values: 0.000 when nothing was attempted,
        1.000 when everything was and none of it was usable, and anything in
        between for a mixture -- five declined and five unusable out of ten
        gives `cov 0.500` beside the dash.

        `0.000` in a comparison column would read as 'answered and got them
        all wrong'. Rendered as an em dash in the `--grade` table and as an
        empty field in `scores.tsv`. The denominator is `judged`, narrower
        than `attempted`: a record whose correctness was never written down
        is not evidence either way.

        History: this and its two siblings were settable fields, so a caller
        could write rates that contradicted the counts in the same row -- and
        a fixture in this repo did, giving `cov 0.400` beside `attempted 4` of
        `total 15`. All three are pure functions of the counts, so a stored
        copy was a second source of truth with nothing keeping the two
        together. Before that, `precision` counted the ways in and said four,
        which was one short (`SKIPPED` reaches `attempted == 0` as well), and
        admitted no mixtures -- a reader who met a dash beside `cov 0.500` had
        been told it could not happen. Earlier still it was a float documented
        as readable-against-`coverage`: a disambiguator beside a rate, the
        trade the citation lines had just rejected one command over.
        """
        return self.correct / self.judged if self.judged else None

    @model_validator(mode="after")
    def _counts_account_for_total(self) -> "MCQScore":
        """Reject a score whose counts cannot describe a run.

        The accounting is stated on `skipped` and again in
        `write_scores_tsv`'s docstring, where a spreadsheet reader is told the
        subtraction is exact. Both were claims about `score_mcq`'s output
        asserted of the TYPE and of the FILE -- and `MCQScore`'s counts are
        independently settable, so anything could be handed to the writer. A
        fixture in this repo did exactly that: `total=3, attempted=2` with
        every failure count at its default, a row whose remainder says a pair
        was skipped when none was, and a shape `score_mcq` cannot produce.

        Enforced here rather than documented, because a claim that holds for
        one producer's output is not a claim about the type.

        The `answers` check is the one conditional clause: the field defaults
        to empty and a score built from counts alone carries none, which is
        every hand-built fixture in the suite, so it can only compare the two
        when there is something to compare. That rationale is here rather
        than in the message because a caller who trips it has a non-empty
        `answers` by construction -- and reading "a score built from counts
        alone legitimately carries none" while holding a row whose counts and
        answers disagree suggests dropping the answers, which is the one
        repair that loses data instead of fixing it.

        History: this closed "-- the way `correct`'s does", citing that
        description as the model of a claim holding by construction. Fair
        when written; `14c9662` then corrected `correct` on the grounds that
        its construction claim was FALSE, and rewrote it to credit this
        validator -- so the two pointed at each other, and the exemplar was
        the one text whose own `History:` records it did not hold. A claim
        stated as "the way X does" rots whenever X does, and no sweep over
        this claim's subject visits X.

        Note what was NOT the mechanism: distance. `correct`'s description
        sits about 190 lines up, in this same class, and same-file proximity
        saved nothing -- the commit that rewrote it had every reason to be
        reading this file. A later comment elsewhere blamed the rot on the
        citation being cross-file; it was not, and saying so would imply a
        nearby citation is safe.

        >>> MCQScore(total=2, attempted=1, correct=1, abstained=1).skipped
        0

        >>> MCQScore(total=3, attempted=2, correct=1)
        Traceback (most recent call last):
        ...
        pydantic_core._pydantic_core.ValidationError: ...
        """
        accounted = (
            self.attempted + self.abstained + self.extraction_failures
            + self.provider_errors + self.skipped
        )
        if accounted != self.total:
            raise ValueError(
                f"counts do not account for total: attempted={self.attempted} "
                f"+ abstained={self.abstained} "
                f"+ extraction_failures={self.extraction_failures} "
                f"+ provider_errors={self.provider_errors} "
                f"+ skipped={self.skipped} = {accounted}, not {self.total}. "
                f"Every task has exactly one disposition, so the five counts "
                f"partition `total`."
            )
        if self.answers and len(self.answers) != self.total:
            raise ValueError(
                f"total={self.total} does not match the {len(self.answers)} "
                f"answer(s) carried: `score_mcq` sets `total = len(answers)` "
                f"and grades every one of them, so `total` counts the answers "
                f"in this row and nothing else."
            )
        if self.unusable > self.attempted:
            raise ValueError(
                f"unusable={self.unusable} exceeds attempted={self.attempted}: "
                f"an unusable record is a SCORED one, so it is a subset of "
                f"`attempted`, not a sixth part of `total`"
            )
        judged = self.judged
        if self.correct > judged:
            raise ValueError(
                f"correct={self.correct} exceeds judged={judged} "
                f"(attempted={self.attempted} - unusable={self.unusable}): a "
                f"question can only be right if an option was chosen for it "
                f"AND its correctness was recorded. `score_mcq` counts "
                f"`correct` where that record says True and `unusable` where "
                f"it says nothing, so those two are disjoint and both sit "
                f"inside `attempted`. Checked against `judged` rather than "
                f"`attempted` because `precision` divides by `judged`: "
                f"`correct <= attempted` alone admits a precision above 1"
            )
        return self
