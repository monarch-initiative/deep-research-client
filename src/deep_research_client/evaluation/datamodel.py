from __future__ import annotations

import re
import sys
from datetime import (
    date,
    datetime,
    time
)
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Literal,
    Optional,
    Union
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer
)


metamodel_version = "1.11.0"
version = "None"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias = True,
        validate_by_name = True,
        validate_assignment = True,
        validate_default = True,
        extra = "forbid",
        arbitrary_types_allowed = True,
        use_enum_values = True,
        strict = False,
    )





class LinkMLMeta(RootModel):
    root: dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key:str):
        return getattr(self.root, key)

    def __getitem__(self, key:str):
        return self.root[key]

    def __setitem__(self, key:str, value):
        self.root[key] = value

    def __contains__(self, key:str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta({'default_prefix': 'evaluation',
     'default_range': 'string',
     'description': 'Data model for evaluating deep research tools and '
                    'co-scientists: the questions put to them, the shape of answer '
                    'each question expects, and the reference material a scorer '
                    'weighs an answer against.\n'
                    'The model is deliberately domain-neutral. An earlier '
                    'iteration of this framework could only read two Monarch '
                    'knowledge bases (dismech and ai-gene-review) and could only '
                    "score long-form biomedical reports, because the benchmark's "
                    'shape was baked into the code. That made every new benchmark '
                    'a rewrite. Here the benchmark is data: an adapter converts '
                    'some upstream format into `EvalSet` instances, and scorers '
                    'dispatch on `EvalTask.answer_type` rather than on what the '
                    'question happens to be about.\n'
                    'Two answer shapes matter today and they score in entirely '
                    'different ways. A multiple-choice benchmark such as LAB-Bench '
                    'has one right option and a fixed set of distractors, so it is '
                    'scored by matching the chosen option. A long-form report has '
                    'no single right answer, so it is scored against a `Rubric` - '
                    'reference claims to recall, facts to spot-check, topics to '
                    'cover. Keeping both in one model is what lets a single runner '
                    'send the same providers at both.\n'
                    'This schema is the source of truth for '
                    'deep_research_client/evaluation/datamodel.py, which is '
                    'generated from it with `just gen-datamodel-eval`. Scoring '
                    'results, which are computed rather than authored, stay in '
                    'deep_research_client/evaluation/models.py.',
     'id': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation',
     'imports': ['linkml:types'],
     'license': 'BSD-3-Clause',
     'name': 'evaluation',
     'prefixes': {'evaluation': {'prefix_prefix': 'evaluation',
                                 'prefix_reference': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'}},
     'source_file': 'src/deep_research_client/evaluation/evaluation.yaml',
     'title': 'Deep Research Client Evaluation'} )

class MatchStyle(str, Enum):
    """
    How a spot check compares a captured value against its expected one.
    """
    exact = "exact"
    """
    Case- and whitespace-insensitive equality.
    """
    prefix = "prefix"
    """
    Equality, or the captured value being a leading part of the expected one. For hierarchical facts, where a shorter answer is less precise rather than incorrect.
    """


class AnswerType(str, Enum):
    """
    The shape of answer a task expects, and so which scorers can read it. This is the dispatch key of the whole framework: a scorer declares the answer types it understands, instead of inspecting the subject matter.
    """
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    """
    One correct option among a fixed set of distractors. Scored by matching the option the model chose, which means an extraction step is needed whenever the provider emits prose rather than a bare choice.
    """
    SHORT_ANSWER = "SHORT_ANSWER"
    """
    A free-text answer short enough to compare against a reference answer directly, without a rubric. Distinct from MULTIPLE_CHOICE because there are no distractors to match against, and from REPORT because it is not long enough for recall and coverage to mean anything.
    """
    REPORT = "REPORT"
    """
    A long-form, cited report. There is no single correct answer, so scoring is against a `Rubric` and against the citations the report carries.
    """


class CellStatus(str, Enum):
    """
    How one task-arm cell of a matrix run ended. Kept apart from ``ScoreDisposition``: this says whether the provider produced anything at all, which is a different question from whether what it produced was right.
    """
    COMPLETED = "COMPLETED"
    """
    The provider returned a result, which was saved.
    """
    FAILED = "FAILED"
    """
    The provider raised. The run continues - one arm being out of quota must not discard the cells that did succeed.
    """
    SKIPPED = "SKIPPED"
    """
    The cell was not run, because a filter excluded it or a previous run had already completed it.
    """


class ScoreDisposition(str, Enum):
    """
    What became of one task-arm pair when it was scored. Kept separate from the numeric scores because a benchmark run's most important number is often how much of it did not run.
    """
    SCORED = "SCORED"
    """
    The answer was produced and scored.
    """
    ABSTAINED = "ABSTAINED"
    """
    The model declined to answer, by choosing the task's `abstention_option`. Counted apart from a wrong answer: a benchmark that rewards calibration wants precision over answered questions, and an abstention is not evidence of error.
    """
    PROVIDER_ERROR = "PROVIDER_ERROR"
    """
    The provider failed to return an answer at all - a quota, an authentication failure, a timeout. Not evidence about the model's ability, and must never be silently folded in as a wrong answer.
    """
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    """
    The provider answered, but no option could be recovered from its response. This is a measurement failure of the harness rather than of the provider, and is reported separately so that it can be driven down instead of being mistaken for provider quality.
    """
    SKIPPED = "SKIPPED"
    """
    The pair was not run, typically because a filter excluded it.
    """



class MetadataItem(ConfiguredBaseModel):
    """
    One free-form key/value pair. Used where an upstream benchmark carries fields this model does not name, so that an adapter never has to discard information it cannot place.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    key: str = Field(default=..., description="""The field name, as the upstream source spells it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['MetadataItem']} })
    value: str = Field(default=..., description="""The value, rendered as a string.""", json_schema_extra = { "linkml_meta": {'domain_of': ['MetadataItem']} })


class OntologyTerm(ConfiguredBaseModel):
    """
    An ontology term referenced by a reference claim.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    id: str = Field(default=..., description="""Ontology CURIE, for example GO:0008543.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm', 'EvalTask', 'ArmSpec']} })
    label: str = Field(default=..., description="""Human-readable label.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm']} })


class EvidenceItem(ConfiguredBaseModel):
    """
    A single piece of published evidence backing a reference claim. The reference is what a citation-verification scorer compares a report's own citations against.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    reference: str = Field(default=..., description="""PMID, DOI, or other reference identifier.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvidenceItem']} })
    snippet: Optional[str] = Field(default=None, description="""Verbatim quote from the source supporting the claim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvidenceItem']} })
    supports: Optional[str] = Field(default=None, description="""Whether the evidence supports, refutes or partially supports the claim, in the source's own vocabulary. Left as free text because curated knowledge bases disagree about the values here, and coercing them into one enum would lose the distinction the curator drew.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvidenceItem']} })
    explanation: Optional[str] = Field(default=None, description="""Why this evidence is relevant to the claim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvidenceItem']} })


class ReferenceClaim(ConfiguredBaseModel):
    """
    One verifiable claim that a good report on this task would be expected to make. Recall against these claims is the main quality signal for REPORT-shaped tasks.
    Claims nest: a curated knowledge base often records a broad mechanism with finer sub-mechanisms beneath it, and flattening that loses the distinction between missing a whole mechanism and missing one detail of it.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    name: str = Field(default=..., description="""Short name for the claim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim', 'SpotCheck', 'ExpectedTopic', 'EvalSet']} })
    description: str = Field(default=..., description="""Full text of the claim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim', 'EvalSet', 'ArmSpec']} })
    category: str = Field(default=..., description="""The kind of claim this is, in the source benchmark's own vocabulary - for example pathophysiology, phenotype, treatment, gene_function. Deliberately an open string rather than an enum: the set of useful categories is a property of each benchmark, and a closed list here would mean editing this schema to add a benchmark.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim']} })
    ontology_terms: Optional[list[OntologyTerm]] = Field(default=None, description="""Ontology terms associated with the claim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim']} })
    evidence: Optional[list[EvidenceItem]] = Field(default=None, description="""Published evidence supporting the claim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim']} })
    subclaims: Optional[list[ReferenceClaim]] = Field(default=None, description="""Finer-grained claims beneath this one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim']} })


class SpotCheck(ConfiguredBaseModel):
    """
    One objectively checkable fact, expressed as a regular expression over the report text. Spot checks live in benchmark data rather than in code: an earlier version of this framework hardcoded checks for two named genes in a Python dict, which made the scorer useless for any third subject.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    name: str = Field(default=..., description="""Name of the fact being checked, for example chromosome_location.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim', 'SpotCheck', 'ExpectedTopic', 'EvalSet']} })
    pattern: str = Field(default=..., description="""Regular expression searched for in the report, case-insensitively. A capturing group, when present, is the value compared against `expected`.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpotCheck']} })
    expected: Optional[str] = Field(default=None, description="""The correct value. When absent, the check tests only that the pattern appears at all, which is a coverage signal rather than an accuracy one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpotCheck']} })
    match: Optional[MatchStyle] = Field(default=MatchStyle.exact, description="""How the captured value is compared against `expected`. `exact` is case- and whitespace-insensitive equality. `prefix` additionally accepts a captured value that is a leading part of the expected one, for facts whose correct answer is hierarchical: a cytogenetic locus, an ontology identifier, a version. A report saying \"chromosome 17\" where the answer is 17q21.31 is less precise, not wrong, and scoring it wrong marks the commonest phrasing in the literature as a factual error - while a report saying 17p13.1 is still wrong, which a presence-only check could not tell apart from silence.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SpotCheck'], 'ifabsent': 'string(exact)'} })


class ExpectedTopic(ConfiguredBaseModel):
    """
    A subject area a good report on this task would be expected to address, detected by keyword. Coverage measured this way is cheap and needs no LLM judge, at the cost of being fooled by a passing mention.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    name: str = Field(default=..., description="""Name of the topic.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim', 'SpotCheck', 'ExpectedTopic', 'EvalSet']} })
    keywords: list[str] = Field(default=..., description="""Keywords whose presence counts as covering the topic. Any one match is enough.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExpectedTopic']} })


class Rubric(ConfiguredBaseModel):
    """
    Reference material for scoring a REPORT-shaped task, where there is no single correct answer to match against.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    reference_claims: Optional[list[ReferenceClaim]] = Field(default=None, description="""Claims a good report would be expected to make.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Rubric']} })
    spot_checks: Optional[list[SpotCheck]] = Field(default=None, description="""Objectively checkable facts.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Rubric']} })
    expected_topics: Optional[list[ExpectedTopic]] = Field(default=None, description="""Subject areas a good report would be expected to address.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Rubric']} })


class AnswerSpec(ConfiguredBaseModel):
    """
    The reference answer for a task that has one, together with the wrong options it must be told apart from.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    ideal: str = Field(default=..., description="""The correct answer.""", json_schema_extra = { "linkml_meta": {'annotations': {'lab_bench_field': {'tag': 'lab_bench_field',
                                             'value': 'ideal'}},
         'domain_of': ['AnswerSpec']} })
    distractors: Optional[list[str]] = Field(default=None, description="""Incorrect options presented alongside the correct answer. Empty for SHORT_ANSWER tasks.""", json_schema_extra = { "linkml_meta": {'annotations': {'lab_bench_field': {'tag': 'lab_bench_field',
                                             'value': 'distractors'}},
         'domain_of': ['AnswerSpec']} })
    abstention_option: Optional[str] = Field(default=None, description="""An option offered to let the model decline rather than guess, for example \"Insufficient information to answer this question\". Present only when the benchmark defines one.
Whether this option is offered materially changes the headline number, because it separates accuracy over all questions from precision over answered ones. It is recorded per task so that a run can never be reported without saying which convention produced it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AnswerSpec']} })


class EvalTask(ConfiguredBaseModel):
    """
    One question put to a research tool, together with whatever a scorer needs to judge the answer.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    id: str = Field(default=..., description="""Task identifier, unique within its `EvalSet`. Used as a directory name in run output, so it should stay filesystem-safe.""", json_schema_extra = { "linkml_meta": {'annotations': {'lab_bench_field': {'tag': 'lab_bench_field', 'value': 'id'}},
         'domain_of': ['OntologyTerm', 'EvalTask', 'ArmSpec']} })
    prompt: str = Field(default=..., description="""The question text, exactly as it should reach the provider.""", json_schema_extra = { "linkml_meta": {'annotations': {'lab_bench_field': {'tag': 'lab_bench_field',
                                             'value': 'question'}},
         'domain_of': ['EvalTask']} })
    answer_type: AnswerType = Field(default=..., description="""Shape of the expected answer, and so which scorers apply.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalTask']} })
    answer_spec: Optional[AnswerSpec] = Field(default=None, description="""Reference answer and distractors. Required for MULTIPLE_CHOICE and SHORT_ANSWER tasks; absent for REPORT tasks, which are scored against `rubric` instead.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalTask']} })
    rubric: Optional[Rubric] = Field(default=None, description="""Reference material for scoring a REPORT-shaped task. Absent for tasks that have a single reference answer.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalTask']} })
    task_type: Optional[str] = Field(default=None, description="""The family of question this is, in the source benchmark's vocabulary - for example litqa2, disease_mechanism, gene_function. Open string for the same reason as `ReferenceClaim.category`: closing it would mean editing this schema to add a benchmark.""", json_schema_extra = { "linkml_meta": {'annotations': {'lab_bench_field': {'tag': 'lab_bench_field',
                                             'value': 'subtask'}},
         'domain_of': ['EvalTask']} })
    subject_id: Optional[str] = Field(default=None, description="""Identifier of the entity the question is about, when it has one - a MONDO or HGNC identifier, a UniProt accession. Lets results be grouped by subject across benchmarks that ask different questions about the same thing.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalTask']} })
    tags: Optional[list[str]] = Field(default=None, description="""Free-form labels for slicing results - domain, difficulty, whatever the analysis needs.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalTask']} })
    source_id: Optional[str] = Field(default=None, description="""The task's identifier in the upstream benchmark, when the adapter had to rewrite it to make `id` unique or filesystem-safe. Keeps a run traceable back to the published dataset.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalTask']} })
    metadata: Optional[list[MetadataItem]] = Field(default=None, description="""Upstream fields this model does not name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalTask']} })


class EvalSet(ConfiguredBaseModel):
    """
    A named, versioned collection of tasks, together with the provenance a published score has to carry to be meaningful.
    The provenance slots are not bookkeeping. A benchmark whose upstream data changes will silently change scores, and a benchmark with a private holdout cannot be compared against its own leaderboard; both facts belong with the tasks rather than in a README.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    name: str = Field(default=..., description="""Short name of the eval set, for example lab-bench-litqa2.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim', 'SpotCheck', 'ExpectedTopic', 'EvalSet']} })
    version: Optional[str] = Field(default=None, description="""Version of the eval set as assembled here.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    description: Optional[str] = Field(default=None, description="""What this eval set measures.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim', 'EvalSet', 'ArmSpec']} })
    source: Optional[str] = Field(default=None, description="""Where the tasks came from - a dataset identifier, repository URL, or local path.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    source_revision: Optional[str] = Field(default=None, description="""The exact upstream revision the tasks were built from, such as a dataset commit hash. Pinned rather than implied: this is what makes a score reproducible, and it is stronger than vendoring a copy of the data because it also records which version produced a given number.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    license: Optional[str] = Field(default=None, description="""License of the upstream task data. Recorded because it governs what may be redistributed, and it is frequently not the license of this repository.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    homepage: Optional[str] = Field(default=None, description="""Documentation or citation URL for the benchmark.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    is_partial: bool = Field(default=False, description="""Whether these tasks are only part of the published benchmark - because a subset was selected, or because the publisher withholds a portion. When true, scores are not comparable with published leaderboard figures, and anything that reports them should say so.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet', 'RunManifest'], 'ifabsent': 'False'} })
    partial_reason: Optional[str] = Field(default=None, description="""Why the set is partial, when it is.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet', 'RunManifest']} })
    tasks: Optional[list[EvalTask]] = Field(default=None, description="""The tasks in this set.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })


class ArmSpec(ConfiguredBaseModel):
    """
    One configuration under test: a provider, optionally a model, optionally provider parameters. An \"arm\" rather than a \"provider\" because the same provider appears more than once in most useful comparisons - a plain agent with web search and the same agent with none are different arms, and telling them apart is the point of running both.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    id: str = Field(default=..., description="""Short identifier for this arm, unique within a run. Used as a directory name in run output, so it should stay filesystem-safe.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm', 'EvalTask', 'ArmSpec']} })
    provider: str = Field(default=..., description="""Registered provider name, for example falcon or claude_code.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ArmSpec']} })
    model: Optional[str] = Field(default=None, description="""Model override, when the provider's default is not wanted.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ArmSpec']} })
    description: Optional[str] = Field(default=None, description="""What this arm is for - notably, what it is a baseline against.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim', 'EvalSet', 'ArmSpec']} })
    params: Optional[list[MetadataItem]] = Field(default=None, description="""Provider-specific parameters.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ArmSpec']} })


class CellResult(ConfiguredBaseModel):
    """
    What one task produced under one arm. The unit of a run: a run of N tasks across M arms has N*M cells, and each records its own outcome so that one provider failing does not invalidate the rest.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    task_id: str = Field(default=..., description="""The task this cell ran.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    arm_id: str = Field(default=..., description="""The arm this cell ran under.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    status: CellStatus = Field(default=..., description="""How the cell ended.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    output_path: Optional[str] = Field(default=None, description="""Path to the saved output, relative to the run directory. Absent when the cell produced nothing.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    duration_seconds: Optional[float] = Field(default=None, description="""Wall-clock time the provider took.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    citation_count: Optional[int] = Field(default=None, description="""Number of citations the provider returned, when it reports them.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    provider_used: Optional[str] = Field(default=None, description="""The provider that actually answered. Normally the arm's own provider, but recorded separately because a provider may fall back, and a score attributed to the wrong system is worse than a missing one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    model_used: Optional[str] = Field(default=None, description="""The model that actually answered.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    resumed: Optional[bool] = Field(default=None, description="""Whether this cell was restored from an earlier run's `cell.json` rather than executed in this one. Distinct from `cached`: a resumed cell was read off disk and carries the *earlier* run's `cached` value, so counting the two together describes neither. It is a property of the run that produced this manifest, not of the stored cell, and is recomputed on every run.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    cached: Optional[bool] = Field(default=None, description="""Whether the response was replayed from the client's cache rather than produced by calling the provider. Read together with `resumed`: on a resumed cell this value describes the earlier run that stored it. Recorded because a replayed cell is otherwise indistinguishable from a live one: the cache re-stamps start and end times for the current run, so duration does not give it away. A score computed over replayed cells may describe a model version that is months old, and two arms sharing a provider, model and parameters return one sample reported as two.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    disposition: Optional[ScoreDisposition] = Field(default=None, description="""For a multiple-choice task, what became of the answer. Absent for report tasks, which are scored in a later pass.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    chosen_letter: Optional[str] = Field(default=None, description="""The option letter recovered from the response, when one was.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    correct: Optional[bool] = Field(default=None, description="""Whether the chosen option was the ideal one. Meaningful only when `disposition` is SCORED.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })
    error: Optional[str] = Field(default=None, description="""Error message, when the cell failed.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CellResult']} })


class RunManifest(ConfiguredBaseModel):
    """
    The record of one matrix run: which eval set, which arms, and what every cell produced.
    Written before results are analysed and kept beside them, because a number without the arms and dataset revision that produced it cannot be checked or reproduced.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/evaluation'})

    run_id: str = Field(default=..., description="""Identifier for this run, also its output directory name.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    created_at: str = Field(default=..., description="""When the run started, as an ISO-8601 timestamp.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    eval_set_name: str = Field(default=..., description="""Name of the eval set that was run.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    eval_set_source: Optional[str] = Field(default=None, description="""Where the eval set came from.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    eval_set_revision: Optional[str] = Field(default=None, description="""Upstream revision of the eval set data, when it has one. Copied here so the run record is self-contained.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    is_partial: bool = Field(default=False, description="""Whether the eval set covered only part of its benchmark, making these scores non-comparable with published figures.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet', 'RunManifest'], 'ifabsent': 'False'} })
    partial_reason: Optional[str] = Field(default=None, description="""Why the eval set is partial, when it is.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet', 'RunManifest']} })
    client_version: Optional[str] = Field(default=None, description="""Version of deep-research-client that produced the run.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    concurrency: Optional[int] = Field(default=None, description="""How many cells were run at a time.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    resume_enabled: Optional[bool] = Field(default=None, description="""Whether cells completed by an earlier run in this directory were reused rather than re-run. Recorded because it is one of the two settings that decide whether a cell was measured in this run, and unlike the cache it leaves no trace on the cells themselves.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    cache_dir: Optional[str] = Field(default=None, description="""The response cache directory the run read from. A run whose replays came from a per-project cache has a different provenance from one that used the shared default, and `cache_enabled` alone does not say which. Absent when the cache was not consulted at all, and absent when the client's default location was used - read it together with `cache_enabled`, which distinguishes those two.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    cache_enabled: Optional[bool] = Field(default=None, description="""Whether the response cache was consulted. Recoverable per cell from `cached`, but recorded here too so a manifest with no replays can be told apart from one where the cache was off.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    arms: Optional[list[ArmSpec]] = Field(default=None, description="""The arms under test.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })
    cells: Optional[list[CellResult]] = Field(default=None, description="""Per-cell outcomes.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RunManifest']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
MetadataItem.model_rebuild()
OntologyTerm.model_rebuild()
EvidenceItem.model_rebuild()
ReferenceClaim.model_rebuild()
SpotCheck.model_rebuild()
ExpectedTopic.model_rebuild()
Rubric.model_rebuild()
AnswerSpec.model_rebuild()
EvalTask.model_rebuild()
EvalSet.model_rebuild()
ArmSpec.model_rebuild()
CellResult.model_rebuild()
RunManifest.model_rebuild()
