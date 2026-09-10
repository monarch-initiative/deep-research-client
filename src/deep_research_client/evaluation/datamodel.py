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

    id: str = Field(default=..., description="""Ontology CURIE, for example GO:0008543.""", json_schema_extra = { "linkml_meta": {'domain_of': ['OntologyTerm', 'EvalTask']} })
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
    description: str = Field(default=..., description="""Full text of the claim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim', 'EvalSet']} })
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
         'domain_of': ['OntologyTerm', 'EvalTask']} })
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
    description: Optional[str] = Field(default=None, description="""What this eval set measures.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceClaim', 'EvalSet']} })
    source: Optional[str] = Field(default=None, description="""Where the tasks came from - a dataset identifier, repository URL, or local path.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    source_revision: Optional[str] = Field(default=None, description="""The exact upstream revision the tasks were built from, such as a dataset commit hash. Pinned rather than implied: this is what makes a score reproducible, and it is stronger than vendoring a copy of the data because it also records which version produced a given number.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    license: Optional[str] = Field(default=None, description="""License of the upstream task data. Recorded because it governs what may be redistributed, and it is frequently not the license of this repository.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    homepage: Optional[str] = Field(default=None, description="""Documentation or citation URL for the benchmark.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    is_partial: bool = Field(default=False, description="""Whether these tasks are only part of the published benchmark - because a subset was selected, or because the publisher withholds a portion. When true, scores are not comparable with published leaderboard figures, and anything that reports them should say so.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet'], 'ifabsent': 'False'} })
    partial_reason: Optional[str] = Field(default=None, description="""Why the set is partial, when it is.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })
    tasks: Optional[list[EvalTask]] = Field(default=None, description="""The tasks in this set.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EvalSet']} })


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
