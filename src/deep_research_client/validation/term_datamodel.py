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


linkml_meta = LinkMLMeta({'default_prefix': 'term_validation',
     'default_range': 'string',
     'description': 'Data model for the outcome of validating the ontology terms '
                    'cited by a deep research report: whether each CURIE resolves '
                    'to a real term, whether the label the report wrote next to it '
                    "is that term's label, and whether the term has since been "
                    'obsoleted.\n'
                    'A resolvable identifier is not the same as a correct one. '
                    'NCIT:C16814 is a real NCIT term, so every existence check '
                    'passes it; it denotes Malaysia, which only a label check can '
                    'tell you. That gap is what this model records.\n'
                    'This schema is the source of truth for '
                    'deep_research_client/validation/term_datamodel.py, which is '
                    'generated from it with `just gen-term-datamodel`. Derived '
                    'quantities (counts, rates) and markdown rendering are added '
                    'in deep_research_client/validation/term_models.py; they are '
                    'computed from these slots rather than stored.',
     'id': 'https://w3id.org/monarch-initiative/deep-research-client/term-validation',
     'imports': ['linkml:types'],
     'license': 'BSD-3-Clause',
     'name': 'term_validation',
     'prefixes': {'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'term_validation': {'prefix_prefix': 'term_validation',
                                      'prefix_reference': 'https://w3id.org/monarch-initiative/deep-research-client/term-validation/'}},
     'source_file': 'src/deep_research_client/validation/term_validation.yaml',
     'title': 'Deep Research Client Ontology Term Validation'} )

class TermStatus(str, Enum):
    """
    Outcome of resolving a single ontology term identifier.
    """
    VERIFIED = "VERIFIED"
    """
    The CURIE resolves to a term that is current.
    """
    NOT_FOUND = "NOT_FOUND"
    """
    The CURIE did not resolve, in an ontology that demonstrably resolves other terms. The identifier is likely to have been confabulated.
    """
    OBSOLETE = "OBSOLETE"
    """
    The CURIE resolves, but the term has been deprecated. Not a fabrication: the report cites something real that should no longer be used, and often has a stated replacement.
    """
    UNVERIFIABLE = "UNVERIFIABLE"
    """
    No resolver could be reached for this prefix, or the prefix was skipped. Distinct from NOT_FOUND: nothing was learned either way, so this is never evidence of fabrication.
    """


class LabelAgreement(str, Enum):
    """
    How the label a report wrote next to a CURIE compares with that term's own label. This is a string comparison, not a judgement about whether the term is the right one to have cited: a report can name a term perfectly and still have chosen the wrong term for the sentence.
    """
    MATCH = "MATCH"
    """
    The report's label is the term's label, up to case, punctuation, word order and plurals.
    """
    VARIANT = "VARIANT"
    """
    The labels differ but are recognisably related - a spelling variant, a subtype suffix, an added qualifier. Worth reading, because "Long QT syndrome" and "Long QT syndrome 1" are different terms, but not by itself evidence of a mistake.
    """
    MISMATCH = "MISMATCH"
    """
    The labels have almost nothing in common. The report is calling this identifier something the ontology does not call it, which usually means the identifier is wrong.
    """
    NOT_ASSESSED = "NOT_ASSESSED"
    """
    Label agreement was not judged, because the report wrote no label beside the CURIE, the term did not resolve, or label checking was disabled.
    """



class TermCheck(ConfiguredBaseModel):
    """
    Result of resolving one ontology term cited by a report.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/term-validation'})

    term_id: str = Field(default=..., description="""Normalized CURIE, for example HP:0001250.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck']} })
    prefix: str = Field(default=..., description="""The CURIE's prefix, for example HP.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck']} })
    status: TermStatus = Field(default=..., description="""Resolution outcome.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck']} })
    occurrences: int = Field(default=1, description="""Number of times the CURIE is mentioned in the report.""", ge=1, json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck'], 'ifabsent': 'int(1)'} })
    ontology_label: Optional[str] = Field(default=None, description="""The term's own label, when it resolved.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck']} })
    reported_labels: Optional[list[str]] = Field(default=None, description="""The labels the report wrote beside this CURIE, in first-appearance order. More than one entry means the report named the same identifier inconsistently, which is worth seeing even when each name is close enough to pass.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck']} })
    agreement: LabelAgreement = Field(default=LabelAgreement.NOT_ASSESSED, description="""How the reported label compares with the term's own label.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck'], 'ifabsent': 'LabelAgreement(NOT_ASSESSED)'} })
    label_similarity: float = Field(default=0.0, description="""Similarity between the reported label and the term's own label, from 0 to 1, for the reported label that agreed least. Zero when agreement was not assessed.
This is the raw measurement behind `agreement`; read the verdict rather than re-thresholding this slot.""", ge=0.0, le=1.0, json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck'], 'ifabsent': 'float(0.0)'} })
    replaced_by: Optional[str] = Field(default=None, description="""Replacement term for an obsolete one, when the ontology states it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck']} })
    message: Optional[str] = Field(default=None, description="""Explanation of the outcome, present when not verified.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermCheck']} })


class TermValidationReport(ConfiguredBaseModel):
    """
    Aggregate result of validating every ontology term cited by a report.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/term-validation'})

    terms: Optional[list[TermCheck]] = Field(default=None, description="""Per-term resolution results.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermValidationReport']} })
    unresolvable_prefixes: Optional[list[str]] = Field(default=None, description="""Prefixes encountered that no configured resolver covers, in first-appearance order. Terms carrying one are reported UNVERIFIABLE rather than NOT_FOUND: an unrecognised prefix may name an ontology this run could not reach as easily as one that does not exist.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermValidationReport']} })
    adapter: Optional[str] = Field(default=None, description="""OAK adapter string terms were resolved through.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermValidationReport']} })
    validator_version: Optional[str] = Field(default=None, description="""Version of linkml-term-validator used to produce this report.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermValidationReport']} })
    truncated: bool = Field(default=False, description="""Whether validation stopped early because a term limit was reached, meaning the counts cover only part of the report's terms.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TermValidationReport'], 'ifabsent': 'False'} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
TermCheck.model_rebuild()
TermValidationReport.model_rebuild()
