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

from deep_research_client.evaluation.datamodel import (
    AnswerType,
    EvalTask,
    EvidenceItem,
    OntologyTerm,
    ReferenceClaim,
    Rubric,
    SpotCheck,
)
from deep_research_client.evaluation.adapters.monarch import (
    GroundTruthEntity,
    _parse_evidence,
    _parse_ontology_term,
    generate_disease_tasks,
    generate_gene_tasks,
    generate_tasks,
    load_dismech_entity,
    load_gene_review_entity,
)
from deep_research_client.evaluation.models import (
    FACTScore,
    RACEDimension,
    RACEScore,
)
from deep_research_client.evaluation.scorers import (
    extract_citations_from_markdown,
    extract_claims_with_citations,
)
from deep_research_client.evaluation.runner import parse_dr_output


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
            ReferenceClaim(
                category="pathophysiology",
                name="FGFR3 gain-of-function",
                description="FGFR3 G380R causes constitutive receptor activation",
                ontology_terms=[OntologyTerm(id="GO:0008543", label="FGFR signaling")],
                evidence=[EvidenceItem(reference="PMID:7913883", snippet="point mutations in FGFR3")],
            ),
            ReferenceClaim(
                category="phenotype",
                name="Short stature",
                description="Disproportionate short stature with rhizomelic limb shortening",
                ontology_terms=[OntologyTerm(id="HP:0008873", label="Disproportionate short-limb short stature")],
            ),
            ReferenceClaim(
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
            ReferenceClaim(
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
            id="test",
            prompt="What causes achondroplasia?",
            answer_type=AnswerType.REPORT,
            task_type="disease_mechanism",
            subject_id=sample_disease_entity.entity_id,
        )
        assert task.task_type == "disease_mechanism"
        assert task.answer_type == AnswerType.REPORT


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
        assert types == {"disease_mechanism", "treatment_rationale", "phenotype_explanation"}

    def test_gene_tasks(self, sample_gene_entity):
        tasks = generate_gene_tasks(sample_gene_entity)
        assert len(tasks) == 1
        assert tasks[0].task_type == "gene_function"
        assert "BRCA1" in tasks[0].prompt

    def test_generate_tasks_dispatches(self, sample_disease_entity, sample_gene_entity):
        disease_tasks = generate_tasks(sample_disease_entity)
        gene_tasks = generate_tasks(sample_gene_entity)
        assert len(disease_tasks) == 3
        assert len(gene_tasks) == 1

    def test_disease_tasks_no_treatment(self):
        entity = GroundTruthEntity(
            entity_id="X", entity_type="disease", name="X", source_repo="dismech",
            claims=[
                ReferenceClaim(category="pathophysiology", name="m1", description="desc"),
            ],
        )
        tasks = generate_disease_tasks(entity)
        assert len(tasks) == 1
        assert tasks[0].task_type == "disease_mechanism"


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
            id="test",
            prompt="test query",
            answer_type=AnswerType.REPORT,
            task_type="disease_mechanism",
        )
        markdown = "FGFR3 G380R mutation (PMID:7913883) causes achondroplasia."
        dr = parse_dr_output(task, markdown, "falcon")
        assert dr.provider == "falcon"
        assert len(dr.extracted_claims) == 1
        assert len(dr.extracted_citations) == 1
        assert dr.extracted_citations[0].normalized_id == "PMID:7913883"

    def test_generate_all_tasks(self, sample_disease_entity, sample_gene_entity):
        tasks = generate_tasks(sample_disease_entity) + generate_tasks(sample_gene_entity)
        assert len(tasks) == 4  # 3 disease + 1 gene


# ---------------------------------------------------------------------------
# Bundled rubrics against the scorer that reads them
# ---------------------------------------------------------------------------

#: A report stating every BRCA1 fact in the bundled rubric correctly. Written
#: in the ordinary phrasing rather than to suit the patterns -- "a RING domain",
#: not "a RING finger domain" -- because the phrasing was the crash.
_CORRECT_BRCA1_REPORT = """
BRCA1 is a tumour suppressor encoded on chromosome 17q21.31. The protein is
1863 amino acids long and localises to the nucleus. It heterodimerises with
BARD1 to form an E3 ubiquitin ligase, and its N-terminal RING domain mediates
that interaction while its BRCT repeats bind phosphopeptides. BRCA1 acts in the
homologous recombination repair pathway together with RAD51. Germline
loss-of-function variants predispose to breast cancer and ovarian cancer, and
such tumours are sensitive to PARP inhibitor therapy.
"""

_CORRECT_TP53_REPORT = """
TP53 lies on chromosome 17p13.1 and encodes p53, a tumor suppressor and
sequence-specific transcription factor whose DNA binding activity is essential.
p53 accumulates in the nucleus and drives an apoptosis signaling pathway,
inducing cell cycle arrest in response to DNA damage. It is the most frequently
mutated gene in human cancer.
"""


@pytest.mark.parametrize("gene,report", [
    ("BRCA1", _CORRECT_BRCA1_REPORT),
    ("TP53", _CORRECT_TP53_REPORT),
])
def test_the_bundled_rubrics_pass_against_a_correct_report(gene, report):
    """The one artifact claiming to measure accuracy, run against the scorer.

    Nothing exercised this file: `score_factual_spot_checks`' doctests build
    inline checks written to work, and `build_rubric`'s doctest asserts only
    that names are present. Three of BRCA1's ten checks were broken --
    `ring_domain` crashed the scorer on the commonest phrasing, `cancer_type`
    compared a captured "breast" against "breast/ovarian cancer" and so was
    wrong for every report ever written, and `chromosome` captured 17q21.31 and
    compared it against a bare "17", marking a precise report wrong and a vague
    one right.
    """
    from deep_research_client.evaluation.adapters.monarch import build_rubric
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id=gene.lower(), prompt=f"What does {gene} do?",
        answer_type=AnswerType.REPORT,
        rubric=build_rubric("gene_function", [], subject=gene),
    )
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)

    missing = [c.fact_name for c in score.checks if not c.present]
    assert not missing, f"a correct report did not mention: {missing}"

    wrong = [
        (c.fact_name, c.expected, c.found_in_report)
        for c in score.checks if c.compared and not c.correct
    ]
    assert not wrong, f"a correct report was scored wrong on: {wrong}"
    assert score.presence_rate == 1.0


def test_accuracy_is_reported_over_the_checks_that_compared_something():
    """A presence-only check is `correct` whenever it matched.

    Counting those as accuracy meant a rubric of nine presence-only checks and
    one wrong accuracy check reported 0.9 accurate -- agreement that was never
    tested, which is the "number with no question behind it" the
    multiple-choice path refuses.
    """
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="mixed", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[
            SpotCheck(name="presence", pattern=r"\btumour suppressor\b",
                      expected="tumour suppressor"),
            SpotCheck(name="accuracy", pattern=r"chromosome\s+(\S+)",
                      expected="17q21.31"),
        ]),
    )
    report = "A tumour suppressor on chromosome 11p15.5."
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)

    assert score.present_count == 2
    assert score.compared_count == 1, "only one check compared a captured value"
    # Wrong about the one thing it could be wrong about.
    assert score.accuracy_rate == 0.0


@pytest.mark.parametrize("pattern,report,why", [
    # groups() is (None,) -- truthy -- so the comparison branch ran.
    (r"\bRING\s*(finger)?\s*domain\b", "BRCA1 has a RING domain.",
     "optional group did not participate"),
    # lastindex is 2 while group(1) is still None, so the branch ran again.
    (r"(?:(17q\d+)|chromosome (17))", "It is on chromosome 17.",
     "a later group participated but group 1 did not"),
])
def test_group_one_without_a_captured_value_is_not_a_comparison(pattern, report, why):
    """Two weaker predicates were wrong here; the property is what is asked now.

    `groups()` is truthy for a tuple of Nones, and `lastindex` is the index of
    the *last* group that matched -- so an alternation whose second branch
    matched left `group(1)` None and crashed identically. The crash discards
    all four intrinsic scores, so the exact predicate is worth having over a
    sufficient one.
    """
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="g", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[
            SpotCheck(name="g", pattern=pattern, expected="something"),
        ]),
    )
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)

    check = score.checks[0]
    assert check.present
    assert not check.compared, why
    assert check.correct, "nothing was captured, so nothing can disagree"


def test_an_optional_group_that_did_not_participate_is_not_a_captured_value():
    """The scorer half of the RING-domain crash, pinned independently.

    The rubric fix (a non-capturing group) and the scorer fix (`lastindex`
    rather than `groups()`) each prevent the crash on their own, so reverting
    either alone leaves the bundled-rubric test green. This exercises the
    scorer directly, which is also the library-caller path: anyone building a
    `SpotCheck` in code can write an optional capturing group, and
    `groups()` on a non-participating one is `(None,)` -- truthy, so the
    comparison branch ran and `.strip()` raised on None.
    """
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="optional", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[
            SpotCheck(name="optional_group",
                      pattern=r"\bRING\s*(finger)?\s*domain\b",
                      expected="RING domain"),
        ]),
    )
    out = parse_dr_output(task, "BRCA1 has a RING domain.", "test")

    score = score_factual_spot_checks(out, task)  # must not raise
    check = score.checks[0]
    assert check.present
    assert check.correct, "nothing was captured, so nothing can disagree"
    assert not check.compared, "a non-participating group is not a comparison"
    assert score.compared_count == 0


# ---------------------------------------------------------------------------
# A judge that fails must not produce a plausible number
# ---------------------------------------------------------------------------


def test_race_records_an_unscored_dimension_rather_than_a_middling_one():
    """A judge outage used to report 3.0 out of 5 for every dimension.

    Indistinguishable from a genuine 3, so a run against a dead endpoint
    reported mid-scale quality for every report in the matrix.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def dead_judge(*args, **kwargs):
        raise RuntimeError("judge endpoint unreachable")

    task = EvalTask(id="r", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A report.", "test")

    original = scorers._llm_judge
    scorers._llm_judge = dead_judge
    try:
        score = asyncio.run(scorers.score_race(out, task, llm_client=object()))
    finally:
        scorers._llm_judge = original

    assert score.dimensions, "the dimensions should still be listed"
    assert all(d.score is None for d in score.dimensions)
    assert score.scored_dimensions == []
    assert score.unscored_count == len(score.dimensions)
    # 0.0 rather than 0.6, and unscored_count says which it is.
    assert score.overall_score == 0.0


def test_claim_recall_does_not_count_an_unreachable_judge_as_a_missed_claim():
    """Recall used to fall with the judge's uptime.

    An exception recorded `matched=False` and divided by the full ground-truth
    count, so a provider was scored for someone else's outage.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def dead_judge(*args, **kwargs):
        raise RuntimeError("judge endpoint unreachable")

    claims = [
        ReferenceClaim(name="c1", category="molecular_function",
                       description="First claim."),
        ReferenceClaim(name="c2", category="molecular_function",
                       description="Second claim."),
    ]
    task = EvalTask(id="r", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A report.", "test")

    original = scorers._llm_judge
    scorers._llm_judge = dead_judge
    try:
        score = asyncio.run(
            scorers.score_claim_recall(out, claims, llm_client=object())
        )
    finally:
        scorers._llm_judge = original

    assert score.total_ground_truth_claims == 2
    assert score.unjudged_claims == 2
    assert all(m.matched is None for m in score.matches)
    assert score.claim_recall == 0.0  # nothing judged, not "covered nothing"


def test_a_judge_reply_without_a_verdict_is_not_read_as_a_verdict():
    """`"true" in result_text[:50]` scored prose on whether four letters appear.

    "It is not true that this abstract supports the claim" was read as support,
    and unlike an unfetchable abstract it landed in the checkable set, counting
    as a *verified* citation.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def prose_judge(*args, **kwargs):
        return "It is not true that this abstract supports the claim."

    claims = [ReferenceClaim(name="c1", category="molecular_function",
                             description="First claim.")]
    task = EvalTask(id="r", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A report.", "test")

    original = scorers._llm_judge
    scorers._llm_judge = prose_judge
    try:
        score = asyncio.run(
            scorers.score_claim_recall(out, claims, llm_client=object())
        )
    finally:
        scorers._llm_judge = original

    assert score.matches[0].matched is None, (
        "a reply containing the letters 'true' is not a verdict of true"
    )
    assert score.unjudged_claims == 1


@pytest.mark.parametrize("scenario,pubmed_body,expect_verifiability,expect_unresolvable", [
    # NCBI reports an unknown uid as a per-uid error. That is the authoritative
    # negative -- a fabricated citation -- and must count against the rate.
    ("fabricated pmid",
     {"result": {"99999999": {"error": "cannot get document summary"}}},
     0.0, 0),
    # A real one.
    ("real pmid",
     {"result": {"99999999": {"title": "A real paper", "pubdate": "2019 Jan"}}},
     1.0, 0),
])
def test_a_fabricated_citation_counts_against_verifiability(
    scenario, pubmed_body, expect_verifiability, expect_unresolvable, monkeypatch,
):
    """Filtering on `error` dropped fabricated PMIDs out of the denominator.

    Nine real citations and one invented one scored 1.00 with unresolvable=1 --
    the metric that exists to detect hallucinated references reporting a report
    that hallucinated one as perfectly verifiable. Only a transport failure is
    unresolvable; "the registry says it does not exist" is the finding.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def fake_pubmed(pmid, client=None):
        entry = pubmed_body["result"]["99999999"]
        if "error" in entry:
            return {"exists": False, "title": None, "year": None,
                    "error": entry["error"], "lookup_failed": False}
        return {"exists": True, "title": entry["title"], "year": 2019}

    monkeypatch.setattr(scorers, "fetch_pubmed_metadata", fake_pubmed)

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A claim [PMID:99999999].", "test")

    score = asyncio.run(scorers.score_citation_verifiability(out))
    assert score.total_citations == 1
    assert score.verifiability == expect_verifiability, scenario
    assert score.unresolvable == expect_unresolvable, scenario


def test_only_a_transport_failure_leaves_the_verifiability_rate(monkeypatch):
    """An outage is still excluded -- that half of the round-nineteen fix stands."""
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def unreachable(pmid, client=None):
        return {"exists": False, "title": None, "year": None,
                "error": "ConnectTimeout", "lookup_failed": True}

    monkeypatch.setattr(scorers, "fetch_pubmed_metadata", unreachable)

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A claim [PMID:12345678].", "test")

    score = asyncio.run(scorers.score_citation_verifiability(out))
    assert score.unresolvable == 1
    assert score.verifiability == 0.0  # nothing checkable, not "all fabricated"
    assert score.total_citations == 1


@pytest.mark.parametrize("phrasing,expect_correct", [
    # A shorter locus is less precise, not wrong -- and is the commonest
    # phrasing in the literature. Scoring it a factual error was the mirror of
    # the defect that scored the *precise* report wrong.
    ("BRCA1 is on chromosome 17.", True),
    ("BRCA1 is on chromosome 17q21.", True),
    ("BRCA1 is on chromosome 17q21.31.", True),
    # A different arm is still wrong, which a presence-only check could not
    # have told apart from silence.
    ("BRCA1 is on chromosome 17p13.1.", False),
])
def test_a_hierarchical_fact_accepts_a_less_precise_answer(phrasing, expect_correct):
    from deep_research_client.evaluation.adapters.monarch import build_rubric
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="brca1", prompt="?", answer_type=AnswerType.REPORT,
        rubric=build_rubric("gene_function", [], subject="BRCA1"),
    )
    score = score_factual_spot_checks(parse_dr_output(task, phrasing, "test"), task)
    check = next(c for c in score.checks if c.fact_name == "chromosome")
    assert check.compared, "this check should be comparing a captured value"
    assert check.correct is expect_correct


@pytest.mark.parametrize("phrasing,expect_correct", [
    ("BRCA1 is 1,863 amino acids long.", True),
    ("BRCA1 is 1863 amino acids long.", True),
    ("BRCA1 is 1234 amino acids long.", False),
])
def test_a_thousands_separator_is_not_a_disagreement(phrasing, expect_correct):
    """`(\\d{3,4})` against "1,863" captured "863" and scored the report wrong."""
    from deep_research_client.evaluation.adapters.monarch import build_rubric
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="brca1", prompt="?", answer_type=AnswerType.REPORT,
        rubric=build_rubric("gene_function", [], subject="BRCA1"),
    )
    score = score_factual_spot_checks(parse_dr_output(task, phrasing, "test"), task)
    check = next(c for c in score.checks if c.fact_name == "protein_length")
    assert check.correct is expect_correct
