"""Tests for the evaluation framework.

Tests cover:
- Models (Pydantic validation, doctests)
- Loaders (dismech and ai-gene-review YAML parsing)
- Task generation
- Citation/claim extraction from markdown
- Scorer helpers (no LLM calls — those are integration tests)
"""

import asyncio
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


def test_race_records_an_unscored_dimension_rather_than_a_middling_one(monkeypatch):
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

    monkeypatch.setattr(scorers, "_llm_judge", dead_judge)
    score = asyncio.run(scorers.score_race(out, task, llm_client=object()))

    assert score.dimensions, "the dimensions should still be listed"
    assert all(d.score is None for d in score.dimensions)
    assert score.scored_dimensions == []
    assert score.unscored_count == len(score.dimensions)
    # 0.0 rather than 0.6, and unscored_count says which it is.
    assert score.overall_score == 0.0


def test_claim_recall_does_not_count_an_unreachable_judge_as_a_missed_claim(monkeypatch):
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

    monkeypatch.setattr(scorers, "_llm_judge", dead_judge)
    score = asyncio.run(scorers.score_claim_recall(out, claims, llm_client=object()))

    assert score.total_ground_truth_claims == 2
    assert score.unjudged_claims == 2
    assert all(m.matched is None for m in score.matches)
    assert score.claim_recall == 0.0  # nothing judged, not "covered nothing"


def test_a_judge_reply_without_a_verdict_is_not_read_as_a_verdict(monkeypatch):
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

    monkeypatch.setattr(scorers, "_llm_judge", prose_judge)
    score = asyncio.run(scorers.score_claim_recall(out, claims, llm_client=object()))

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


@pytest.mark.parametrize("reply,why", [
    ('{"verdict": "yes"}', "a different key"),
    ('{"error": "rate limited"}', "an envelope from a proxy"),
    ('{}', "an empty object"),
])
def test_well_formed_json_without_the_verdict_key_is_not_a_negative_verdict(
    monkeypatch, reply, why,
):
    """`result.get("matched", False)` read these as "the report does not cover it".

    Worse than the prose case, which at least took the no-verdict path: these
    landed in the judged set and counted against the provider. `score_race` got
    this guard; its two neighbours did not.
    """
    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def odd_reply(*args, **kwargs):
        return reply

    monkeypatch.setattr(scorers, "_llm_judge", odd_reply)

    claims = [ReferenceClaim(name="c1", category="molecular_function",
                             description="First claim.")]
    task = EvalTask(id="r", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A report.", "test")

    score = asyncio.run(scorers.score_claim_recall(out, claims, llm_client=object()))
    assert score.matches[0].matched is None, why
    assert score.unjudged_claims == 1


@pytest.mark.parametrize("style", ["exact", "prefix"])
def test_an_empty_capture_is_not_a_comparison(style):
    """The same report was a factual error under one style and correct under
    the other, on the strength of a capture that was the empty string.

    A group that can match nothing -- `(\\d*)`, `(17.*)`, anything with `?` or
    `*` at the top level -- captures "" when the report phrases the fact in
    words. `"" == "17q21.31"` is False, so `exact` charged the report with
    getting the locus wrong; `"17q21.31".startswith("")` is True, so `prefix`
    credited it with getting the locus right. Neither was a measurement, and
    the pair disagreeing is the proof: nothing about the report changed.
    """
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="brca1", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[SpotCheck(
            name="chromosome", pattern=r"chromosome\s*(\d*)",
            expected="17q21.31", match=style,
        )]),
    )
    report = "BRCA1 sits on chromosome seventeen, long arm."
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)
    check = score.checks[0]

    assert check.present, "the pattern did match"
    assert not check.compared, "but it captured nothing, so nothing was compared"
    assert score.compared_count == 0
    assert score.accuracy_rate == 0.0


def test_a_capture_that_is_present_still_compares():
    """The guard above must not disarm the checks that do measure something."""
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="brca1", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[SpotCheck(
            name="chromosome", pattern=r"chromosome\s*(\d*)",
            expected="17q21.31", match="prefix",
        )]),
    )
    score = score_factual_spot_checks(
        parse_dr_output(task, "BRCA1 sits on chromosome 17.", "test"), task
    )
    assert score.checks[0].compared and score.checks[0].correct
    assert score.compared_count == 1


@pytest.mark.parametrize("scenario,meta,expect_rate,expect_checked,expect_unresolvable", [
    # NCBI reports an unknown uid as a per-uid error: the citation is to a paper
    # that does not exist, and a paper that does not exist supports nothing.
    ("fabricated pmid",
     {"exists": False, "title": None, "year": None,
      "error": "cannot get document summary", "lookup_failed": False},
     0.9, 10, 0),
    # A timeout tells us nothing about the citation, so it leaves the rate.
    ("pubmed outage",
     {"exists": False, "title": None, "year": None,
      "error": "ConnectTimeout", "lookup_failed": True},
     1.0, 9, 1),
])
def test_alignment_separates_a_fabricated_citation_from_a_failed_lookup(
    scenario, meta, expect_rate, expect_checked, expect_unresolvable, monkeypatch,
):
    """The verifiability scorer's inversion, repeated one function along.

    `unresolvable` incremented on any missing title, and a fabricated PMID has
    no title -- so nine real citations and one invented one scored alignment
    1.00, with the invented one quietly out of the denominator. The distinction
    drawn next door in the same commit was not carried across; the predicate
    has to be `lookup_failed`, because that is the only key every one of the
    four branches that make the distinction actually sets.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    real = {"exists": True, "title": "FGFR3 mutations cause achondroplasia",
            "year": 2019}

    async def fake_pubmed(pmid, client=None):
        return meta if pmid == "PMID:99999999" else real

    monkeypatch.setattr(scorers, "fetch_pubmed_metadata", fake_pubmed)

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    body = " ".join(
        f"FGFR3 mutations cause achondroplasia [PMID:1000000{i}]." for i in range(9)
    ) + " FGFR3 mutations cause achondroplasia [PMID:99999999]."

    score = asyncio.run(scorers.score_citation_alignment(parse_dr_output(task, body, "test")))

    assert score.alignment_rate == pytest.approx(expect_rate), scenario
    assert score.total_checked == expect_checked, scenario
    assert score.unresolvable == expect_unresolvable, scenario


def test_a_doi_crossref_does_not_know_is_a_negative_not_a_failed_lookup():
    """CrossRef's 404 branch carried neither key, so it was a negative only by
    the default in `meta.get("lookup_failed", False)`.

    Both consumers draw the transport-failure/authoritative-negative line by
    reading that key. A branch that sets neither is one reader's default away
    from flipping sides, which is how the alignment scorer got it wrong.
    """
    import asyncio

    import httpx

    from deep_research_client.evaluation import scorers

    transport = httpx.MockTransport(lambda request: httpx.Response(404))

    async def resolve():
        async with httpx.AsyncClient(transport=transport) as client:
            return await scorers.resolve_doi("DOI:10.1038/nonexistent", client)

    meta = asyncio.run(resolve())
    assert meta["exists"] is False
    assert meta["lookup_failed"] is False, "a 404 is the finding, not an outage"
    # And it says why, the way NCBI's unknown-uid negative does. Without this
    # a fabricated DOI reached `--output` as exists: false with no reason
    # beside it, while a fabricated PMID carried NCBI's own message.
    assert meta["error"], "an authoritative negative should carry its reason"


@pytest.mark.parametrize("report,expect_correct,expect_found", [
    # The defect: a report on BRCA1 that mentions TP53's locus first. `re.search`
    # stopped at the first occurrence, so a report that stated BRCA1's own locus
    # correctly two sentences later was scored a factual error.
    ("BRCA1 works with TP53, on chromosome 17p13.1. "
     "BRCA1 itself is on chromosome 17q21.31.", True, "chromosome 17q21.31"),
    # Order must not matter either way round.
    ("BRCA1 is on chromosome 17q21.31. TP53 is on chromosome 17p13.1.",
     True, "chromosome 17q21.31"),
    # A report that only ever states the wrong locus is still wrong, and the
    # occurrence it is wrong at is the one reported.
    ("BRCA1 is on chromosome 17p13.1.", False, "chromosome 17p13.1"),
])
def test_a_spot_check_looks_at_every_occurrence_not_only_the_first(
    report, expect_correct, expect_found,
):
    """Otherwise the score depends on the order a report happens to mention
    things in, which is not a property of whether it got the fact right."""
    from deep_research_client.evaluation.adapters.monarch import build_rubric
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="brca1", prompt="?", answer_type=AnswerType.REPORT,
        rubric=build_rubric("gene_function", [], subject="BRCA1"),
    )
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)
    check = next(c for c in score.checks if c.fact_name == "chromosome")

    assert check.compared
    assert check.correct is expect_correct
    assert check.found_in_report == expect_found


def test_a_non_capturing_occurrence_does_not_hide_a_disagreement():
    """"Any occurrence is correct" must not be satisfied by an occurrence that
    compared nothing.

    A pattern whose group is optional matches both with and without a captured
    value. Taking the first `correct` verdict would let the non-capturing
    occurrence -- correct only in the presence-only sense -- stand in for a
    comparison the report really did fail.
    """
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="brca1", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[SpotCheck(
            name="chromosome", pattern=r"chromosome(?:\s+(17[pq][\d.]+))?",
            expected="17q21.31",
        )]),
    )
    # The bare "chromosome" comes first and captures nothing; the real claim,
    # which is wrong, comes second.
    report = "The chromosome in question is human. BRCA1 sits at chromosome 17p13.1."
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)

    assert score.checks[0].compared, "a comparison was available and was made"
    assert score.checks[0].correct is False
    # The trailing period is inside the match: `[\d.]+` swallows it.
    assert score.checks[0].found_in_report == "chromosome 17p13.1."


def test_a_presence_only_check_records_no_expected_value():
    """`expected or ""` made "nothing was specified" look like "" was.

    The per-check detail is what a reader consults when a rate surprises them,
    and a presence-only check has no expected value to hold the report to.
    """
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="t", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[
            SpotCheck(name="mentions_ring", pattern=r"\bRING domain\b"),
            SpotCheck(name="length", pattern=r"(\d+)\s*amino acid", expected="1863"),
        ]),
    )
    score = score_factual_spot_checks(
        parse_dr_output(task, "A 1863 amino acid protein with a RING domain.", "test"),
        task,
    )
    by_name = {c.fact_name: c for c in score.checks}

    assert by_name["mentions_ring"].expected is None
    assert by_name["length"].expected == "1863"


# Response bodies NCBI actually returns, stubbed at the transport rather than by
# monkeypatching `fetch_pubmed_metadata`. The earlier fakes for these tests
# returned `lookup_failed` themselves, which re-implemented the very
# classification under test: they pinned the scorer's filter and left the
# response-shape mapping -- the half that was wrong -- uncovered in both
# directions.
_PUBMED_BODIES = {
    # A uid NCBI does not know. The authoritative negative, and the single
    # thing the verifiability scorer exists to detect.
    "unknown uid": (
        {"result": {"99999999": {"error": "cannot get document summary"}}},
        {"exists": False, "lookup_failed": False},
    ),
    # A top-level failure. 200 OK, valid JSON, nothing about the uid --
    # `raise_for_status` cannot see it, and this is the shape a burst of
    # citation lookups hits when NCBI rate-limits.
    "top-level esummary error": (
        {"esummaryresult": ["Invalid db name", "Empty id list"]},
        {"exists": False, "lookup_failed": True},
    ),
    "rate-limit envelope": (
        {"error": "API rate limit exceeded"},
        {"exists": False, "lookup_failed": True},
    ),
    # A result dict that answers about other uids but not this one, which is
    # what a batched or mis-ordered esummary looks like.
    "other uids only": (
        {"result": {"uids": ["12345678"],
                    "12345678": {"title": "Someone else's paper"}}},
        {"exists": False, "lookup_failed": True},
    ),
    # A record came back, so the uid is real, even though the summary carries
    # no title. `exists = bool(title)` reported this as a fabricated citation.
    "record with no title": (
        {"result": {"99999999": {"title": "", "pubdate": "2019"}}},
        {"exists": True, "lookup_failed": False},
    ),
    "real record": (
        {"result": {"99999999": {"title": "A real paper", "pubdate": "2019 Jan"}}},
        {"exists": True, "lookup_failed": False},
    ),
}


@pytest.mark.parametrize("scenario", sorted(_PUBMED_BODIES))
def test_every_pubmed_response_shape_is_classified(scenario):
    """Each shape says explicitly whether the lookup succeeded.

    The classification covered a per-uid error and an exception and defaulted
    everything else to "the lookup succeeded", which is the side that counts a
    citation as fabricated. An NCBI rate limit therefore reported every
    citation in a report as hallucinated.
    """
    import asyncio

    import httpx

    from deep_research_client.evaluation import scorers

    body, expected = _PUBMED_BODIES[scenario]
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))

    async def fetch():
        async with httpx.AsyncClient(transport=transport) as client:
            return await scorers.fetch_pubmed_metadata("PMID:99999999", client)

    meta = asyncio.run(fetch())
    assert meta["exists"] is expected["exists"], scenario
    assert meta["lookup_failed"] is expected["lookup_failed"], scenario


def test_a_transport_error_is_a_failed_lookup():
    """The other half: an exception is not an authoritative negative."""
    import asyncio

    import httpx

    from deep_research_client.evaluation import scorers

    def boom(request):
        raise httpx.ConnectTimeout("no route to host")

    async def fetch():
        async with httpx.AsyncClient(transport=httpx.MockTransport(boom)) as client:
            return await scorers.fetch_pubmed_metadata("PMID:7913883", client)

    meta = asyncio.run(fetch())
    assert meta["lookup_failed"] is True
    assert meta["exists"] is False


def test_a_rate_limited_batch_is_not_a_report_that_invented_its_references():
    """The consequence, on the number a user would publish.

    Nine real citations and one invented one, with NCBI rate-limiting: every
    lookup returns the envelope shape, so the whole report was reported as
    0.00 verifiable -- indistinguishable from a report whose references are
    entirely fabricated.
    """
    import asyncio

    import httpx

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"error": "API rate limit exceeded"})
    )

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    body = " ".join(f"A claim [PMID:1000000{i}]." for i in range(9))
    out = parse_dr_output(task, body, "test")

    async def score():
        async with httpx.AsyncClient(transport=transport) as client:
            return await scorers.score_citation_verifiability(out, client)

    result = asyncio.run(score())
    assert result.total_citations == 9
    # This is the discriminating assertion: `verifiability` is 0.00 under both
    # the old behaviour and the new one, because the empty-`checkable` guard
    # returns 0.0 either way. What changed is whether these nine are excluded
    # from the rate or reported as nine fabricated references.
    assert result.unresolvable == 9, "every lookup failed, none was answered"
    assert result.verified_exist == 0


def test_a_fabricated_citation_still_drags_a_rate_full_of_real_ones():
    """The discriminating case, which a single-citation test cannot show.

    With one citation the empty-`checkable` guard returns 0.00 under both the
    old filter and the new one. The defect's real signature is nine real and
    one invented scoring 1.00; this pins 0.90 instead.
    """
    import asyncio

    import httpx

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    def route(request):
        if "99999999" in str(request.url):
            return httpx.Response(200, json={
                "result": {"99999999": {"error": "cannot get document summary"}}})
        uid = str(request.url).split("id=")[1].split("&")[0]
        return httpx.Response(200, json={
            "result": {uid: {"title": "A real paper", "pubdate": "2019 Jan"}}})

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    body = " ".join(f"A claim [PMID:1000000{i}]." for i in range(9)) \
        + " An invented one [PMID:99999999]."
    out = parse_dr_output(task, body, "test")

    async def score():
        async with httpx.AsyncClient(transport=httpx.MockTransport(route)) as client:
            return await scorers.score_citation_verifiability(out, client)

    result = asyncio.run(score())
    assert result.total_citations == 10
    assert result.unresolvable == 0, "nothing failed; one answer was 'no such paper'"
    assert result.verifiability == pytest.approx(0.9)


@pytest.mark.parametrize("scenario,meta,expect_checked,expect_unresolvable", [
    # A real paper whose summary carries no title. The lookup succeeded and the
    # paper exists; there is simply nothing to align a claim against, so it
    # leaves the rate rather than being scored a misalignment.
    ("real record, no title",
     {"exists": True, "title": None, "year": 2019, "lookup_failed": False},
     0, 1),
    # A uid the registry does not know. A paper that does not exist supports
    # nothing, so it counts against alignment.
    ("fabricated pmid",
     {"exists": False, "title": None, "year": None,
      "error": "cannot get document summary", "lookup_failed": False},
     1, 0),
    # Nothing was learned at all.
    ("failed lookup",
     {"exists": False, "title": None, "year": None,
      "error": "ConnectTimeout", "lookup_failed": True},
     0, 1),
])
def test_alignment_tells_three_kinds_of_missing_title_apart(
    scenario, meta, expect_checked, expect_unresolvable, monkeypatch,
):
    """"No title" is true of three different things, and only one of them is
    the report's fault."""
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def fake_pubmed(pmid, client=None):
        return meta

    monkeypatch.setattr(scorers, "fetch_pubmed_metadata", fake_pubmed)

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "FGFR3 causes achondroplasia [PMID:7913883].", "test")

    score = asyncio.run(scorers.score_citation_alignment(out))
    assert score.total_checked == expect_checked, scenario
    assert score.unresolvable == expect_unresolvable, scenario


@pytest.mark.parametrize("preamble", [
    '{"thinking": "the abstract mentions FGFR3"}',   # a judge that narrates
    '{"model": "gpt-4o-mini", "object": "chat"}',    # a wrapper's envelope
])
@pytest.mark.parametrize("scorer", ["fact", "recall", "race"])
def test_a_verdict_after_a_parseable_preamble_is_still_a_verdict(
    monkeypatch, preamble, scorer,
):
    """Trying every balanced run fixed only half of this.

    The loop returned the first candidate that *parsed*, and a narration object
    parses -- so `{"thinking": ...} {"matched": true}`, the shape the change was
    written for, still took the no-verdict path. The measurement left the
    denominator and turned up as an `unjudged_claims` or `unscored_count`
    nobody could explain, on a judge that had answered correctly.

    All three judge-backed scorers, because each has its own verdict key and
    each had to be told about it separately -- the "fixed one instance, not its
    siblings" shape this branch has hit before.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    verdicts = {
        "fact": '{"supported": true, "explanation": "yes"}',
        "recall": '{"matched": true, "best_matching_text": "FGFR3"}',
        "race": '{"score": 4, "explanation": "good"}',
    }

    async def narrating_judge(*args, **kwargs):
        return f"{preamble} then {verdicts[scorer]}"

    monkeypatch.setattr(scorers, "_llm_judge", narrating_judge)

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    body = "FGFR3 causes achondroplasia (PMID:7913883)."
    out = parse_dr_output(task, body, "test")

    if scorer == "recall":
        claims = [ReferenceClaim(name="fgfr3", category="mechanism",
                                 description="FGFR3 mutations cause achondroplasia")]
        score = asyncio.run(scorers.score_claim_recall(out, claims, object()))
        assert score.unjudged_claims == 0
        assert score.claim_recall == 1.0
    elif scorer == "race":
        score = asyncio.run(scorers.score_race(out, task, object()))
        assert score.unscored_count == 0
        assert all(d.score == 4.0 for d in score.dimensions)
    else:
        async def abstract(pmid, client=None):
            return "FGFR3 mutations cause achondroplasia."

        monkeypatch.setattr(scorers, "fetch_pubmed_abstract", abstract)
        score = asyncio.run(scorers.score_fact(out, object()))
        assert score.total_citations == 1
        assert score.citation_accuracy == 1.0


def test_a_reply_with_no_verdict_anywhere_is_still_unjudged():
    """The key must not turn every parseable object into a verdict: a reply
    that never answers is still a lost measurement, not a negative one."""

    from deep_research_client.evaluation import scorers

    assert scorers._extract_json_object('{"verdict": "yes"}', key="matched") == {
        "verdict": "yes"
    }
    assert scorers._extract_json_object("no json at all", key="matched") is None


@pytest.mark.parametrize("raw,expect_score", [
    (4, 4.0),
    (4.5, 4.5),
    ("4", 4.0),      # a judge that quotes its number is still answering
    (1, 1.0),
    (5, 5.0),
    (9, None),       # off the scale it was given
    (0, None),
    (-3, None),
    ("excellent", None),
])
def test_a_score_off_the_scale_is_unscored_not_clamped(monkeypatch, raw, expect_score):
    """`min(max(float(score), 1.0), 5.0)` made a judge answering 9 a confident 5.

    Clamping invents a value the judge never gave, and it does so in the
    direction that flatters the report -- an out-of-range answer became the top
    of the scale. The rest of this scorer had already settled that a reply it
    cannot read leaves the average rather than landing somewhere in it.
    """
    import asyncio
    import json as json_mod

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def judge(*args, **kwargs):
        return json_mod.dumps({"score": raw, "explanation": "why"})

    monkeypatch.setattr(scorers, "_llm_judge", judge)

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A report.", "test")

    score = asyncio.run(scorers.score_race(out, task, object()))

    assert {d.score for d in score.dimensions} == {expect_score}
    if expect_score is None:
        assert score.unscored_count == len(score.dimensions)
        assert "outside 1-5" in score.dimensions[0].explanation


def test_claim_recall_with_no_reference_claims_records_nothing_judged():
    """The early return recorded `judged_chars` as the truncation length.

    Both fields are filled in so they are not sometimes-absent for two
    different reasons, but the *value* said a report's first 12,000 characters
    had been read when no judge was asked anything. On a long report that
    printed "judged on 12,000 of 29,000 characters" beside a 0/0 -- a claim
    about work that never happened.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    long_report = "FGFR3 drives achondroplasia. " * 1000
    assert len(long_report) > scorers.MAX_REPORT_CHARS
    out = parse_dr_output(task, long_report, "test")

    score = asyncio.run(scorers.score_claim_recall(out, [], object()))

    assert score.judged_chars == 0, "no judge was asked anything"
    assert score.report_chars == len(long_report)


@pytest.mark.parametrize("subject,report,expect_correct,expect_found", [
    # The regression "any occurrence is correct" introduced. A bare
    # "chromosome 17" captures "17", which is a valid prefix of 17q21.31, so a
    # correct-by-vagueness occurrence outranked the report's actual, wrong
    # claim. A report on a chromosome-17 tumour suppressor writes the bare
    # number routinely, which made this check unable to register a wrong locus
    # at all -- while still counting in `compared_count` as one of only two
    # bundled checks that compare anything.
    ("BRCA1",
     "Genes on chromosome 17 include BRCA1 and TP53. "
     "BRCA1 is located at chromosome 17p13.1.", False, "chromosome 17p13.1"),
    # The fix must not undo the case it was built for: here the more specific
    # claim is the correct one, so it still wins over the earlier wrong arm.
    ("BRCA1",
     "BRCA1 works with TP53, on chromosome 17p13.1. "
     "BRCA1 is on chromosome 17q21.31.", True, "chromosome 17q21.31"),
    # And `prefix` still accepts a less precise answer when that is all the
    # report says -- the whole reason the style exists.
    ("BRCA1", "BRCA1 is on chromosome 17.", True, "chromosome 17"),
    # A vague mention beside a correct specific one reports the specific one.
    ("BRCA1",
     "Genes on chromosome 17 include BRCA1. BRCA1 is at chromosome 17q21.31.",
     True, "chromosome 17q21.31"),
    # Two equally specific claims that contradict each other: the override is
    # for a disagreement that *extends* the agreement, so an equally precise
    # wrong mention does not beat a right one -- neither extends the other.
    # There is no principled winner between claims of the same precision, and
    # `prefix` is the forgiving style by construction; recorded here so the
    # choice is deliberate rather than an artifact of the predicate.
    ("BRCA1",
     "Some sources say chromosome 17p21.31. BRCA1 is at chromosome 17q21.31.",
     True, "chromosome 17q21.31"),
    # The mirror, which a length-based override got wrong. TP53 expects
    # 17p13.1 (7 characters) and BRCA1's 17q21.31 is longer (8), so "a wrong
    # capture that is strictly longer overrides" condemned a correct TP53
    # report for mentioning BRCA1's locus -- printing the other gene's locus
    # as the evidence. 17q21.31 does not *extend* 17p13.1, so containment
    # leaves this alone. The BRCA1 cases above cannot see it: BRCA1 happens to
    # be the gene with the longer expected string.
    ("TP53",
     "TP53 lies on chromosome 17p13.1 and encodes p53. It acts with BRCA1, "
     "which is encoded at chromosome 17q21.31.", True, "chromosome 17p13.1"),
    # And TP53's own vague-then-wrong case, so the rule is pinned from both
    # sides for both subjects rather than only for the longer one.
    ("TP53",
     "Genes on chromosome 17 include TP53. TP53 is at chromosome 17q21.31.",
     False, "chromosome 17q21.31"),
    # A report that states the right locus keeps it even with a vaguer mention
    # and another gene's locus in the same text -- the reason `hit` is the most
    # specific agreement rather than the first one.
    ("BRCA1",
     "Chromosome 17 carries both. BRCA1 is at chromosome 17q21.31, "
     "TP53 at chromosome 17p13.1.", True, "chromosome 17q21.31"),
])
def test_a_specific_disagreement_outranks_a_vague_agreement(
    subject, report, expect_correct, expect_found,
):
    """Under `prefix`, a correct capture can be a vaguer form of the answer, so
    "any occurrence is correct" let context override the report's own claim.

    The override is containment, not length: a disagreement counts only when it
    *extends* the agreement. Both proxies tried before it were right for the
    example in the bug report and wrong for its mirror.
    """
    from deep_research_client.evaluation.adapters.monarch import build_rubric
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id=subject.lower(), prompt="?", answer_type=AnswerType.REPORT,
        rubric=build_rubric("gene_function", [], subject=subject),
    )
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)
    check = next(c for c in score.checks if c.fact_name == "chromosome")

    assert check.compared
    assert check.correct is expect_correct
    # The occurrence that settled it, not whichever came first.
    assert check.found_in_report == expect_found


def test_the_specificity_rule_does_not_reach_an_exact_check():
    """Under `exact` a correct capture *is* the answer, so nothing can be more
    specific than it; a wrong capture elsewhere is a second, different claim,
    not a more precise version of the same one.

    The assertion below reads as unremarkable -- a correct report scores
    correct -- so what it is for is worth saying: dropping the
    `is_prefix_match(spec)` condition on the override makes it fail. It is the
    only thing keeping a rule built for hierarchical facts out of the checks
    that are not hierarchical.
    """
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="brca1", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[SpotCheck(
            name="protein_length", pattern=r"(\d+)\s*amino acid", expected="1863",
        )]),
    )
    report = "BRCA1 is 1863 amino acids. A splice form runs to 18630 amino acids."
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)

    assert score.checks[0].correct is True


@pytest.mark.parametrize("scenario,body,expect_checked,expect_unresolvable", [
    ("fabricated pmid",
     {"result": {"99999999": {"error": "cannot get document summary"}}}, 1, 0),
    ("top-level esummary error",
     {"esummaryresult": ["Invalid db name"]}, 0, 1),
    ("real record with a title",
     {"result": {"99999999": {"title": "FGFR3 mutations cause achondroplasia",
                              "pubdate": "2019"}}}, 1, 0),
    # A real record carrying no title: the paper exists, so it is not a
    # finding, but there is nothing to align a claim against. This chain was
    # created by the same change that keyed alignment on `lookup_failed`, and
    # is the one it is easiest to get backwards.
    ("real record with no title",
     {"result": {"99999999": {"title": "", "pubdate": "2019"}}}, 0, 1),
])
def test_alignment_reads_the_same_responses_the_lookup_does(
    scenario, body, expect_checked, expect_unresolvable,
):
    """Driven from the response rather than from a hand-written `lookup_failed`.

    The parametrized test above stubs `fetch_pubmed_metadata`, which pins the
    alignment scorer's filter but replaces the response-shape mapping it
    depends on -- so a regression in the mapping would leave it green. This one
    goes through the transport, so the two halves are tested joined.
    """
    import asyncio

    import httpx

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(
        task, "FGFR3 mutations cause achondroplasia [PMID:99999999].", "test"
    )

    async def score():
        async with httpx.AsyncClient(transport=transport) as client:
            return await scorers.score_citation_alignment(out, client)

    result = asyncio.run(score())
    assert result.total_checked == expect_checked, scenario
    assert result.unresolvable == expect_unresolvable, scenario


@pytest.mark.parametrize("reply,expected,why", [
    # A key nested inside a preamble must not outrank a later top-level
    # verdict. The scan resumed one character after a parsed object, which is a
    # brace *inside* it, so this returned False -- not a missing verdict but
    # the opposite one, counted against the provider.
    ('Analysis: {"evidence": {"supported": false, "note": "about mice"}} '
     'Verdict: {"supported": true, "explanation": "it does"}',
     True, "a nested key before a top-level verdict"),
    # But a wrapper whose only content is the verdict is still read, which is
    # why the answer is "prefer top level" rather than "never descend".
    ('{"response": {"supported": true}}', True, "a wrapper around the verdict"),
    # An object nested inside unparseable text is still reachable.
    ('{oops {"supported": true}}', True, "a malformed outer run"),
    # A closing brace inside a string value used to end the object early, and
    # the whole verdict was lost.
    ('{"explanation": "the set {a} and a } brace", "supported": true}',
     True, "a brace inside a string value"),
])
def test_the_top_level_verdict_wins_over_one_buried_in_a_preamble(
    reply, expected, why,
):
    """Failing into the wrong verdict is worse than failing out of one.

    The two earlier fixes here moved a narrating judge from "no verdict" to
    "the narration is the verdict" to "the key inside the narration is the
    verdict" -- the last of which answers the opposite question confidently.
    """
    from deep_research_client.evaluation import scorers

    result = scorers._extract_json_object(reply, key="supported")
    assert result is not None, why
    assert result["supported"] is expected, why


def test_a_boolean_score_is_not_a_score_of_one(monkeypatch):
    """`bool` is a subclass of `int`, so `float(True)` is 1.0 -- inside the
    scale, and so indistinguishable downstream from a measured bottom mark.

    The same class as the clamp removed beside it: a reply that is not a number
    became a number, and this one lands where nothing can tell.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def judge(*args, **kwargs):
        return '{"score": true, "explanation": "very good"}'

    monkeypatch.setattr(scorers, "_llm_judge", judge)

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A report.", "test")

    score = asyncio.run(scorers.score_race(out, task, object()))

    assert all(d.score is None for d in score.dimensions)
    assert score.unscored_count == len(score.dimensions)
    # And the judge's own words survive, since they are all there is to
    # diagnose a mis-prompted judge with.
    assert "very good" in score.dimensions[0].explanation


def test_fact_total_citations_counts_every_pair_not_just_the_judged_ones(monkeypatch):
    """It held `len(checkable)`, so the one name in the model that did not
    mean what it says.

    Harmless while nothing else reported the difference; `unjudged_citations`
    beside it made a reader's arithmetic wrong -- ten pairs in the report, a
    `total_citations` of eight and an `unjudged_citations` of two invites
    "then six were judged". The rate stays over the judged ones, which is what
    a rate about whether citations support their claims can be over.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def judge(*args, **kwargs):
        return '{"supported": true, "explanation": "yes"}'

    async def abstract(pmid, client=None):
        return "FGFR3 mutations cause achondroplasia."

    monkeypatch.setattr(scorers, "_llm_judge", judge)
    monkeypatch.setattr(scorers, "fetch_pubmed_abstract", abstract)

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    # Two PMIDs the judge rules on, two DOIs whose abstracts cannot be fetched.
    out = parse_dr_output(
        task,
        "A claim (PMID:7913883). Another (PMID:12345678). "
        "A third (DOI:10.1038/ng1234). A fourth (DOI:10.1038/ng5678).",
        "test",
    )

    score = asyncio.run(scorers.score_fact(out, object()))

    assert score.total_citations == 4, "every pair the report carries"
    assert score.unjudged_citations == 2, "the two DOIs"
    assert score.citation_accuracy == 1.0, "over the two that were judged"
    assert score.total_citations - score.unjudged_citations == 2


def test_the_occurrence_reported_is_the_one_that_overrode_the_agreement():
    """Not merely the longest disagreement in the report.

    The two coincide for the bundled locus checks, because every capture from
    `chromosome\\s+(17...)` extends every shorter one. With a pattern whose
    captures are not all nested, a longer unrelated disagreement is not the one
    that settled the verdict, and printing it would point a reader at the wrong
    sentence -- the defect this field was fixed for one round earlier.
    """
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="go", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[SpotCheck(
            name="process", pattern=r"term ([A-Za-z0-9.:]+)",
            expected="GO:0006281", match="prefix",
        )]),
    )
    # "GO:" agrees vaguely; "GO:0006915" extends it and disagrees, so it is the
    # verdict. "UNRELATEDLONGTERM" also disagrees and is longer, but extends
    # nothing -- it is a different fact, not a more precise version of this one.
    report = ("Annotated under term GO: broadly. Specifically term GO:0006915. "
              "See also term UNRELATEDLONGTERM.")
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)

    assert score.checks[0].correct is False
    # The trailing period is inside the match: `.` is in the character class.
    assert score.checks[0].found_in_report == "term GO:0006915."


def test_every_bundled_subject_is_scored_not_only_loaded():
    """The load-time parametrize has a guard against falling behind the rubric
    file; the scoring one did not, which is the weaker half to leave unguarded.

    Loading a subject proves its patterns compile. Only running them against a
    correct report proves they match what they are about -- which is how three
    of BRCA1's ten checks were found broken.
    """
    from deep_research_client.evaluation.adapters.monarch import load_rubric_data

    from .rubric_coverage import parametrized_subjects

    subjects = set(load_rubric_data("gene_spot_checks"))
    covered = parametrized_subjects(test_the_bundled_rubrics_pass_against_a_correct_report, 0)
    assert subjects <= covered, (
        f"subjects in gene_spot_checks.yaml never scored: {subjects - covered}"
    )


@pytest.mark.parametrize("reply,expected,why", [
    # A breakdown emitted as an array, then the verdict. The scan only ever
    # started at `{`, so it walked past the `[` one character at a time and
    # collected each member as a peer of the later verdict -- the defect the
    # scan was rewritten to fix, reached through a bracket instead of a brace.
    ('[{"criterion": "citations", "score": 2}, {"criterion": "depth", "score": 5}] '
     'Overall: {"score": 4, "explanation": "good"}',
     4, "an array breakdown before the verdict"),
    # An array is not a candidate, but a reply that is *only* an array still
    # has its verdict found -- through the descent, not as a top-level peer.
    ('[{"score": 3, "explanation": "ok"}]', 3, "a reply that is only an array"),
    # Nested arrays are no different.
    ('{"criteria": [{"score": 2}]} {"score": 5}', 5, "an array inside an object"),
])
def test_an_array_in_the_reply_does_not_outrank_the_verdict(reply, expected, why):
    """`score` is a generic key, so a per-criterion breakdown carries it too.

    Reading the dimension from the breakdown records a number that is inside
    the scale, where nothing downstream can tell it from a measurement -- the
    same class as a boolean score, reached from the other side.
    """
    from deep_research_client.evaluation import scorers

    result = scorers._extract_json_object(reply, key="score")
    assert result is not None, why
    assert result["score"] == expected, why


def test_a_race_dimension_is_scored_from_the_verdict_not_the_breakdown(monkeypatch):
    """The consequence, through the scorer rather than the helper."""
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    async def judge(*args, **kwargs):
        return ('Per criterion:\n'
                '[{"criterion": "citations", "score": 2},\n'
                ' {"criterion": "depth", "score": 1}]\n\n'
                'Overall: {"score": 5, "explanation": "excellent"}')

    monkeypatch.setattr(scorers, "_llm_judge", judge)

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A report.", "test")

    score = asyncio.run(scorers.score_race(out, task, object()))

    assert {d.score for d in score.dimensions} == {5.0}
    assert score.unscored_count == 0


@pytest.mark.parametrize("reply", [
    '[{"criterion": "depth", "note": "thorough"}]',   # array, no verdict key
    '[1, 2, 3]',                                      # array of scalars
])
def test_an_array_is_never_returned_as_the_verdict(reply):
    """The fallback returns the judge's reply so the caller can quote it, and
    the caller then does `result.get(...)`.

    An array collected as a top-level candidate would be handed back from that
    fallback, and a list has no `.get` -- the scorer's `except` would turn a
    reply with no verdict into an error rather than an unjudged measurement,
    which is a worse report of the same thing. The declared return type is
    `dict | None` and it has to hold.
    """
    from deep_research_client.evaluation import scorers

    result = scorers._extract_json_object(reply, key="score")
    assert result is None or isinstance(result, dict)
    assert scorers._extract_json_object(reply) is None or isinstance(
        scorers._extract_json_object(reply), dict
    )


@pytest.mark.parametrize("reply,expected,why", [
    # The commonest way an LLM's JSON fails to parse, with a breakdown inside
    # the object that holds the verdict. The scan advanced one character on a
    # decode failure, so the nested breakdown became a *top-level* candidate
    # and answered for the reply -- a judge that said `true` recorded as
    # `false` and counted against the provider.
    ('{"criteria": {"supported": false}, "supported": true,}',
     True, "a trailing comma in the object holding the verdict"),
    # A malformed preamble followed by a real verdict: the failed region has to
    # be bounded, or the genuine verdict is demoted alongside the breadcrumbs.
    ('{"criteria": {"supported": false},} {"supported": true}',
     True, "a malformed preamble before a real verdict"),
    # An outer that is not JSON at all and cannot be repaired: its contents are
    # still recoverable, because nothing else answers.
    ('{oops {"supported": true}}', True, "an unparseable outer"),
    # Several values inside the unreadable container, the verdict not first.
    # With one, the fallback that quotes the judge's reply happens to return
    # the right object, so a single-value case cannot tell the salvage tier
    # from its absence.
    ('{oops {"note": "thinking"} {"supported": true}}',
     True, "an unparseable outer holding more than one value"),
    # A reply cut off mid-object, which is what `max_tokens` produces.
    ('{"criteria": {"supported": true}, "sup',
     True, "a reply cut off before its object closed"),
    # A trailing comma inside a string value is not a trailing comma.
    ('{"note": "a, }", "supported": true,}',
     True, "a comma inside a string value"),
])
def test_a_malformed_container_does_not_answer_for_the_reply(reply, expected, why):
    """Recovering the wrong verdict is worse than recovering none.

    Every predicate that has been wrong in this function was a proxy: a brace
    counter for "where does this object end", position for "which object is the
    verdict". This one was "where the parser happened to succeed" standing in
    for "is this at the top level", and the two disagree exactly when the outer
    object is malformed.
    """
    from deep_research_client.evaluation import scorers

    result = scorers._extract_json_object(reply, key="supported")
    assert result is not None, why
    assert result["supported"] is expected, why


def test_alignment_counts_a_citation_it_cannot_look_up(monkeypatch):
    """Verifiability counts these; alignment dropped them from both its results
    and its unresolvable count.

    The extractor emits PMC accessions and GEO ids as well as PMIDs and DOIs,
    so this is reachable from an ordinary report. A reference list made of them
    printed `0/0 (0.00)` with nothing unresolvable -- byte-identical to a report
    that cited nothing, which is the reading the counter exists to prevent.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    body = ("FGFR3 drives achondroplasia (PMC11000121). "
            "Expression was deposited under GSE68086.")
    out = parse_dr_output(task, body, "test")
    assert [c.normalized_id for c in out.extracted_citations] == [
        "PMC:PMC11000121", "GEO:GSE68086",
    ], "the fixture must exercise identifiers neither scorer can resolve"

    verifiability = asyncio.run(scorers.score_citation_verifiability(out))
    alignment = asyncio.run(scorers.score_citation_alignment(out))

    # The two scorers read the same reference list and must agree that it has
    # citations in it.
    assert verifiability.total_citations == 2
    assert alignment.unresolvable == 2
    assert alignment.total_checked == 0


def test_the_evidence_named_is_the_most_specific_disagreement():
    """With several disagreements extending the agreement, "the one that
    overrode it" names nothing in particular; the report's claim is its most
    precise one."""
    from deep_research_client.evaluation.runner import parse_dr_output
    from deep_research_client.evaluation.scorers import score_factual_spot_checks

    task = EvalTask(
        id="brca1", prompt="?", answer_type=AnswerType.REPORT,
        rubric=Rubric(spot_checks=[SpotCheck(
            name="chromosome", pattern=r"chromosome\s+(17[pq\d.]*)",
            expected="17q21.31", match="prefix",
        )]),
    )
    report = ("Genes on chromosome 17 include BRCA1. It is on chromosome 17p13 "
              "-- more precisely chromosome 17p13.1.")
    score = score_factual_spot_checks(parse_dr_output(task, report, "test"), task)

    assert score.checks[0].correct is False
    # The trailing period is inside the match: `.` is in the character class.
    assert score.checks[0].found_in_report == "chromosome 17p13.1."


@pytest.mark.parametrize("reply,key,expected,why", [
    # An array closed with a brace, then the real verdict. `_balanced_span`
    # answered None for "closed with the wrong bracket" as well as for "never
    # closed", and the caller read None as the latter -- mining the whole
    # remainder as salvage, which demotes the genuine verdict and lets the
    # breakdown answer. The RACE dimension was recorded as 2 where the judge
    # said 5: a number inside the scale, indistinguishable from a measurement.
    ('{"criteria": [{"score": 2}, {"score": 1}} {"score": 5}', "score", 5,
     "an array closed with a brace"),
    ('{"criteria": {"supported": false}] {"supported": true}', "supported", True,
     "an object closed with a bracket"),
    # A no-regression case rather than a discriminating one: with the fix
    # reverted the whole remainder was mined as salvage and the salvage tier
    # returned this same object. Kept because it says the mine still works when
    # the entire reply is damaged, and labelled because the two cases above are
    # what actually pin the change.
    ('{"criteria": [{"score": 2}}', "score", 2, "only a mismatched container"),
])
def test_a_mismatched_bracket_bounds_the_damage_rather_than_erasing_it(
    reply, key, expected, why,
):
    """Where the damage stops is the caller's question, not why it is damaged.

    Two distinct answers were collapsed into one `None`, and the caller needed
    to tell them apart -- the shape this function has now been wrong in four
    times, each a proxy or a collapsed domain rather than the property itself.
    """
    from deep_research_client.evaluation import scorers

    result = scorers._extract_json_object(reply, key=key)
    assert result is not None, why
    assert result[key] == expected, why


def test_the_two_citation_scorers_agree_about_an_identifier_neither_resolves():
    """A real PMC article is not a fabricated citation.

    Verifiability recorded an identifier kind it has no resolver for as
    `exists=False, lookup_failed=False`, so it counted against the rate: a
    report citing a real PMC article and a real GEO series printed `0/2 (0.00)`
    -- both invented -- one line above the alignment line saying neither could
    be checked. Distinct from an identifier that would not normalise, which is
    a property of the report's own text and still counts against it.
    """
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    body = ("FGFR3 drives achondroplasia (PMC11000121). "
            "Expression was deposited under GSE68086.")
    out = parse_dr_output(task, body, "test")

    verifiability = asyncio.run(scorers.score_citation_verifiability(out))
    alignment = asyncio.run(scorers.score_citation_alignment(out))

    assert verifiability.total_citations == 2
    # Excluded from the rate, not counted against it: the discriminating
    # assertion, since `verifiability` is 0.00 either way -- once because
    # nothing was checkable, once because everything was called fabricated.
    assert verifiability.unresolvable == 2
    assert alignment.unresolvable == 2


def test_an_unnormalisable_citation_still_counts_against_the_report():
    """The other side of the same line: a reference the report's own text
    mangled is the report's problem, and must not be excluded with it."""
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.models import DROutput, ExtractedCitation

    out = DROutput(
        task_id="t", provider="test", raw_markdown="A claim.",
        extracted_citations=[ExtractedCitation(raw_reference="[17]", normalized_id="")],
    )
    score = asyncio.run(scorers.score_citation_verifiability(out))

    assert score.total_citations == 1
    assert score.unresolvable == 0, "not excluded -- the report's own reference list"
    assert score.verifiability == 0.0


def test_repeated_citations_are_not_weighted():
    """`total_citations`' description said nothing de-duplicates, which is the
    opposite of what the extractor does.

    `find_reference_ids` keys on the normalised identifier and records a
    `count` of mentions precisely because the repeats are discarded, so a PMID
    cited ten times is one lookup and one count. The sentence was written while
    *correcting* that field's description, which is the worst place to invert a
    claim: it is the line a reader consults to decide whether repeats are
    weighted.
    """
    from deep_research_client.evaluation.scorers import extract_citations_from_markdown

    cited_once = extract_citations_from_markdown("A claim (PMID:7913883).")
    cited_thrice = extract_citations_from_markdown(
        "A (PMID:7913883). B (PMID:7913883). C (PMID:7913883)."
    )
    assert [c.normalized_id for c in cited_once] == ["PMID:7913883"]
    assert [c.normalized_id for c in cited_thrice] == ["PMID:7913883"]


def test_a_reply_of_unclosed_openers_is_read_rather_than_raising():
    """The unclosed branch recursed on the whole remainder, so the depth was
    the number of consecutive unclosed openers.

    Every call site is inside an `except Exception`, so a RecursionError would
    have degraded into a recorded scoring error rather than a crash -- which is
    the worse report of the same thing, by the argument the array test makes.
    """
    from deep_research_client.evaluation import scorers

    reply = "{" * 2000 + '{"supported": true}'
    result = scorers._extract_json_object(reply, key="supported")

    assert result == {"supported": True}


def test_a_readable_wrapper_beats_something_scavenged_from_an_unclosed_container():
    """Where the unclosed-region tier actually changes the answer.

    A value found after a container that never closes is inside it, so it is
    salvage -- ranked below even a key nested in a readable top-level object.
    With only one candidate the tier makes no difference, which is why the
    obvious cut-off-reply cases cannot see it: this needs a readable wrapper
    whose verdict disagrees with the scavenged one.
    """
    from deep_research_client.evaluation import scorers

    reply = '{"wrapper": {"supported": true}} {"criteria": {"supported": false}'
    result = scorers._extract_json_object(reply, key="supported")

    assert result == {"supported": True}, (
        "the wrapper is readable; the other object is inside a container that "
        "never closes"
    )


def test_a_mismatch_inside_a_later_closing_container_costs_the_verdict():
    """The case where bounding the damage loses something, pinned deliberately.

    `{"breakdown": [1}, "supported": true}` has a mismatched closer at index 16,
    so the region ends there and the `"supported": true` after it is not inside
    any value the scan can read. The answer is no verdict -- a lost
    measurement, which is the side this branch errs on.

    Running the bound out to the *outer* close instead would recover the
    verdict here and reintroduce the defect the bound exists for, since the
    scan would be back to guessing where a damaged container ends. Pinned so
    that trade is made on purpose rather than by someone tidying the bound.
    """
    from deep_research_client.evaluation import scorers

    assert scorers._extract_json_object(
        '{"breakdown": [1}, "supported": true}', key="supported"
    ) is None


def test_a_run_of_unclosed_openers_does_not_stall_the_scorer():
    """Removing the recursion moved the cost rather than removing it.

    Every unclosed opener drove a `_balanced_span` scan to the end of the text,
    so a judge reply that degenerates into a run of braces -- an ordinary LLM
    failure mode -- took about six seconds at MAX_REPORT_CHARS where it used to
    come back fast as a recorded error. `score_race` makes four judge calls per
    report and `score_claim_recall` one per claim, so a single degenerate arm
    could stall a matrix cell with nothing to show for it.

    Timed rather than asserted on wall clock alone: the bound is that the work
    is linear in the reply, so quadrupling the input must not multiply the time
    by sixteen. A generous ceiling, because this runs on shared CI.
    """
    import time

    from deep_research_client.evaluation import scorers

    def elapsed(openers: int) -> float:
        text = "{" * openers
        start = time.perf_counter()
        scorers._extract_json_object(text, key="score")
        return time.perf_counter() - start

    small = max(elapsed(1000), 1e-4)
    large = elapsed(4000)
    assert large < small * 40, (
        f"4x the input took {large / small:.1f}x the time; the per-opener scan "
        f"is quadratic again"
    )


def test_a_repair_inside_an_unclosed_container_is_given_up_deliberately():
    """What the cost bound above costs.

    A trailing-comma object nested inside a container that never closes is no
    longer repaired: once the scan is inside something with no end, whatever it
    finds is salvage whichever way it is bounded, so the bound is skipped and
    `raw_decode` alone finds the well-formed values. The repaired object would
    have been salvage too, so this is a lost measurement inside text already
    declared unreadable -- the side this branch errs on -- and the alternative
    was a six-second stall per judge call.

    Pinned so the trade stays deliberate rather than being "fixed" back into
    the quadratic scan.
    """
    from deep_research_client.evaluation import scorers

    assert scorers._extract_json_object('{oops {"x": 1,}', key="x") is None
    # The same object outside an unclosed container is still repaired.
    assert scorers._extract_json_object('{"x": 1,}', key="x") == {"x": 1}


@pytest.mark.parametrize("scenario,body,expected_exists", [
    # The registry answered: this paper does not exist.
    ("fabricated pmid",
     {"result": {"99999999": {"error": "cannot get document summary"}}}, False),
    # The registry answered: it does.
    ("real record",
     {"result": {"99999999": {"title": "A real paper", "pubdate": "2019"}}}, True),
    # Nothing was established -- the body says nothing about this uid.
    ("rate-limit envelope", {"error": "API rate limit exceeded"}, None),
])
def test_exists_is_three_valued_because_the_question_is(
    scenario, body, expected_exists,
):
    """A `bool` gave the per-citation record the answer the aggregate was fixed
    to stop giving.

    `citations[]` goes into `--output` verbatim, so a consumer reading the
    obvious field saw `exists: false` for a real paper nobody had asked about.
    `lookup_failed` discriminated, and nothing obliged a reader to consult it.
    """
    import asyncio

    import httpx

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "A claim [PMID:99999999].", "test")

    async def score():
        async with httpx.AsyncClient(transport=transport) as client:
            return await scorers.score_citation_verifiability(out, client)

    result = asyncio.run(score())
    assert result.citations[0].exists is expected_exists, scenario


def test_an_identifier_with_no_resolver_establishes_nothing():
    """The case this round added: a real PMC article, never looked up."""
    import asyncio

    from deep_research_client.evaluation import scorers
    from deep_research_client.evaluation.runner import parse_dr_output

    task = EvalTask(id="c", prompt="?", answer_type=AnswerType.REPORT)
    out = parse_dr_output(task, "FGFR3 drives achondroplasia (PMC11000121).", "test")

    result = asyncio.run(scorers.score_citation_verifiability(out))

    assert result.citations[0].exists is None
    assert result.citations[0].lookup_failed is True
