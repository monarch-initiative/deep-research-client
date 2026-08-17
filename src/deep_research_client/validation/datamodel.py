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


metamodel_version = "None"
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

    @model_serializer(mode='wrap', when_used='unless-none')
    def treat_empty_lists_as_none(
            self, handler: SerializerFunctionWrapHandler,
            info: SerializationInfo) -> dict[str, Any]:
        if info.exclude_none:
            _instance = self.model_copy()
            for field, field_info in type(_instance).model_fields.items():
                if getattr(_instance, field) == [] and not(
                        field_info.is_required()):
                    setattr(_instance, field, None)
        else:
            _instance = self
        return handler(_instance, info)



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


linkml_meta = LinkMLMeta({'default_prefix': 'reference_validation',
     'default_range': 'string',
     'description': 'Data model for the outcome of validating the references cited '
                    'by a deep research report: whether each cited identifier '
                    'resolves to a real record, whether quotes attributed to a '
                    'reference actually appear in it, and whether a resolved '
                    "reference looks like it is even about the report's subject.\n"
                    'This schema is the source of truth for '
                    'deep_research_client/validation/datamodel.py, which is '
                    'generated from it with `just gen-datamodel`. Derived '
                    'quantities (counts, rates) and markdown rendering are added '
                    'in deep_research_client/validation/models.py; they are '
                    'computed from these slots rather than stored.',
     'id': 'https://w3id.org/monarch-initiative/deep-research-client/reference-validation',
     'imports': ['linkml:types'],
     'license': 'BSD-3-Clause',
     'name': 'reference_validation',
     'prefixes': {'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'reference_validation': {'prefix_prefix': 'reference_validation',
                                           'prefix_reference': 'https://w3id.org/monarch-initiative/deep-research-client/reference-validation/'}},
     'source_file': 'src/deep_research_client/validation/reference_validation.yaml',
     'title': 'Deep Research Client Reference Validation'} )

class ReferenceStatus(str, Enum):
    """
    Outcome of resolving a single reference identifier.
    """
    VERIFIED = "VERIFIED"
    """
    The identifier resolves to a real record.
    """
    NOT_FOUND = "NOT_FOUND"
    """
    The identifier could not be resolved against any known source, so it is likely to have been confabulated.
    """
    UNVERIFIABLE = "UNVERIFIABLE"
    """
    The identifier was deliberately skipped, or no resolver exists for its prefix. Distinct from NOT_FOUND: nothing was learned either way.
    """


class TopicalRelevance(str, Enum):
    """
    How well a resolved reference's own metadata matches the vocabulary of the report citing it. This is a keyword overlap heuristic, not a judgement about whether the reference supports the claim made from it: a paper can be squarely on topic and still not say what the report says it says.
    """
    ON_TOPIC = "ON_TOPIC"
    """
    The reference's title and abstract share a substantial part of the report's distinctive vocabulary.
    """
    UNCERTAIN = "UNCERTAIN"
    """
    Some overlap, but not enough to call either way - or too little metadata to draw a conclusion from. The usual outcome for a record that resolved to a title and nothing else.
    """
    OFF_TOPIC = "OFF_TOPIC"
    """
    An abstract's worth of text that shares almost none of the report's vocabulary. Worth a look: the identifier is real, so this is not a fabricated citation, but it may be an unrelated paper cited by mistake.
    """
    NOT_ASSESSED = "NOT_ASSESSED"
    """
    Relevance was not assessed, because the check was disabled, the reference did not resolve, or the report yielded no keywords.
    """



class ReferenceCheck(ConfiguredBaseModel):
    """
    Result of resolving one reference cited by a report.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/reference-validation'})

    reference_id: str = Field(default=..., description="""Normalized identifier, for example PMID:7913883.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck', 'SupportingTextCheck']} })
    status: ReferenceStatus = Field(default=..., description="""Resolution outcome.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck']} })
    occurrences: int = Field(default=1, description="""Number of times the identifier is mentioned in the report.""", ge=1, json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck'], 'ifabsent': 'int(1)'} })
    title: Optional[str] = Field(default=None, description="""Title of the resolved record.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck']} })
    year: Optional[str] = Field(default=None, description="""Publication year of the resolved record.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck']} })
    journal: Optional[str] = Field(default=None, description="""Journal or venue of the resolved record.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck']} })
    doi: Optional[str] = Field(default=None, description="""DOI of the resolved record.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck']} })
    content_type: Optional[str] = Field(default=None, description="""Kind of content retrieved, for example abstract_only or full_text_xml.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck']} })
    message: Optional[str] = Field(default=None, description="""Explanation of the outcome, present when not verified.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck', 'SupportingTextCheck']} })
    relevance: TopicalRelevance = Field(default=TopicalRelevance.NOT_ASSESSED, description="""Whether the resolved record looks like it is about the same subject as the report citing it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck'], 'ifabsent': 'TopicalRelevance(NOT_ASSESSED)'} })
    relevance_score: float = Field(default=0.0, description="""Share of the report's keyword weight that appears in this reference's own metadata, from 0 to 1. Zero when relevance was not assessed.""", ge=0.0, le=1.0, json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck'], 'ifabsent': 'float(0.0)'} })
    matched_keywords: Optional[list[str]] = Field(default=[], description="""Report keywords found in this reference's metadata, most heavily weighted first. Recorded so a reader can see what the verdict rests on.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck']} })


class SupportingTextCheck(ConfiguredBaseModel):
    """
    Result of checking a quoted claim against the text of the reference it is attributed to.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/reference-validation'})

    reference_id: str = Field(default=..., description="""Normalized identifier the quote is attributed to.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck', 'SupportingTextCheck']} })
    quote: str = Field(default=..., description="""Quoted text as it appears in the report.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SupportingTextCheck']} })
    was_checkable: bool = Field(default=True, description="""Whether there was anything to check the quote against. False when the reference did not resolve, exposed no abstract or full text, was skipped by prefix, or fell outside a reference limit. A quote that was not checkable is not evidence of confabulation, so it is reported separately from one that was checked and not found.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SupportingTextCheck'], 'ifabsent': 'True'} })
    is_valid: bool = Field(default=..., description="""Whether the quote was found in the reference. Always false when was_checkable is false.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SupportingTextCheck']} })
    similarity_score: float = Field(default=0.0, description="""Similarity of the closest match found, from 0 to 1. Clamped to that range, since it originates outside this codebase.""", ge=0.0, le=1.0, json_schema_extra = { "linkml_meta": {'domain_of': ['SupportingTextCheck'], 'ifabsent': 'float(0.0)'} })
    matched_text: Optional[str] = Field(default=None, description="""The matching span in the reference, when the quote was found.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SupportingTextCheck']} })
    best_match: Optional[str] = Field(default=None, description="""The closest non-matching span in the reference, when the quote was not found.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SupportingTextCheck']} })
    suggested_fix: Optional[str] = Field(default=None, description="""Suggested correction reported by the underlying validator.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SupportingTextCheck']} })
    source_content_type: Optional[str] = Field(default=None, description="""Kind of text the quote was searched in, for example abstract_only or full_text_xml. A failure against an abstract alone is much weaker evidence than one against retrieved full text, because the quote may simply come from a part of the paper that was never fetched.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SupportingTextCheck']} })
    message: Optional[str] = Field(default=None, description="""Validator explanation of the outcome.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceCheck', 'SupportingTextCheck']} })


class ReferenceValidationReport(ConfiguredBaseModel):
    """
    Aggregate result of validating every reference cited by a report.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/reference-validation'})

    references: Optional[list[ReferenceCheck]] = Field(default=[], description="""Per-reference resolution results.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceValidationReport']} })
    supporting_text: Optional[list[SupportingTextCheck]] = Field(default=[], description="""Per-quote supporting text results.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceValidationReport']} })
    report_keywords: Optional[list[str]] = Field(default=[], description="""The report's own distinctive terms, most heavily weighted first, as used to assess topical relevance. Empty when relevance was not assessed.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceValidationReport']} })
    validator_version: Optional[str] = Field(default=None, description="""Version of linkml-reference-validator used to produce this report.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceValidationReport']} })
    truncated: bool = Field(default=False, description="""Whether validation stopped early because a reference limit was reached, meaning the counts cover only part of the bibliography.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReferenceValidationReport'], 'ifabsent': 'False'} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
ReferenceCheck.model_rebuild()
SupportingTextCheck.model_rebuild()
ReferenceValidationReport.model_rebuild()
