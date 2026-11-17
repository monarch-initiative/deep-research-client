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


linkml_meta = LinkMLMeta({'default_prefix': 'alignment',
     'default_range': 'string',
     'description': 'Data model for aligning the claims of one source against '
                    'another (issue #43, item 2). Given two ClaimSets A and B (see '
                    'claims.yaml), alignment pairs claims that are about the same '
                    'thing and classifies each pair on two orthogonal axes:\n'
                    '- semantic_relation: how the two propositions relate '
                    '(identical, one subsumes\n'
                    '  the other, close, consistent, complementary, contradictory, '
                    'unrelated). This\n'
                    '  is judged by a model from the claim texts.\n'
                    "- evidential_relation: how the two claims' evidential "
                    'standing relates, derived\n'
                    "  from each claim's `about` and `basis` in claims.yaml. It "
                    'records, for example,\n'
                    '  that the same conclusion is a firsthand observation in one '
                    'source and\n'
                    '  background knowledge in the other. This is computed, not '
                    'judged.\n'
                    '\n'
                    'A pair also records whether the two claims share a citation '
                    '(independent corroboration versus the same source echoed), '
                    'and which signal proposed the pair. Claims with no aligned '
                    'partner are not a relation; they are surfaced as coverage on '
                    'the container (see alignment_models.ClaimAlignment).\n'
                    'Report-level rubric summaries (issue #43, item 4) aggregate '
                    'these pair relations into auditable, count-based metrics. '
                    'Nothing here forces one scoring schema: a RubricSummary is a '
                    'list of named RubricMetrics, each carrying the counts and the '
                    'pair or claim ids that produced it, so a downstream project '
                    'can read, recompute or extend it.\n'
                    'This schema is standalone: it imports only linkml:types and '
                    'redeclares a lightweight AlignedSource rather than importing '
                    'claims.yaml, so each datamodel generates independently (as '
                    'claims.yaml and evaluation.yaml already do) and the drift '
                    'test stays a single-file regenerate-and-diff.\n'
                    'This schema is the source of truth for '
                    'deep_research_client/claims/alignment_datamodel.py, generated '
                    'with `just gen-datamodel`. Derived views and the computed '
                    'axes are added in '
                    'deep_research_client/claims/alignment_models.py; they are not '
                    'stored here.',
     'id': 'https://w3id.org/monarch-initiative/deep-research-client/alignment',
     'imports': ['linkml:types'],
     'license': 'BSD-3-Clause',
     'name': 'alignment',
     'prefixes': {'alignment': {'prefix_prefix': 'alignment',
                                'prefix_reference': 'https://w3id.org/monarch-initiative/deep-research-client/alignment/'},
                  'dcterms': {'prefix_prefix': 'dcterms',
                              'prefix_reference': 'http://purl.org/dc/terms/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'}},
     'source_file': 'src/deep_research_client/claims/alignment.yaml',
     'title': 'Deep Research Client Claim Alignment'} )

class AlignedSourceType(str, Enum):
    """
    What kind of document a set of claims came from. Mirrors claims.SourceType; redeclared here to keep this schema standalone.
    """
    MARKDOWN_REPORT = "MARKDOWN_REPORT"
    """
    A deep research report or other markdown prose.
    """
    STRUCTURED_DOCUMENT = "STRUCTURED_DOCUMENT"
    """
    A YAML or JSON knowledge artifact.
    """


class SemanticRelation(str, Enum):
    """
    How claim A's proposition relates to claim B's, read from A to B. The relation is the model's judgement of the two claim texts; it is not a check of whether either claim is true. Only SUBSUMES and SUBSUMED_BY are direction-sensitive; the rest are symmetric.
    """
    IDENTICAL = "IDENTICAL"
    """
    The two claims assert the same thing. They may be worded differently, but neither adds or restricts anything relative to the other.
    """
    SUBSUMES = "SUBSUMES"
    """
    A is the more general claim and B is a special case of it: everything B asserts, A asserts too, and A says more or applies more broadly.
    """
    SUBSUMED_BY = "SUBSUMED_BY"
    """
    A is the more specific claim: A is a special case of B. The mirror of SUBSUMES.
    """
    CLOSE = "CLOSE"
    """
    The two claims are about the same underlying fact but differ in wording, precision or scope in a way that is neither identity nor a clean subsumption (for example a different reported percentage for the same relationship, or overlapping but not nested populations).
    """
    CONSISTENT = "CONSISTENT"
    """
    The two claims make different assertions that can both hold at once. They are logically compatible but do not state the same fact.
    """
    COMPLEMENTARY = "COMPLEMENTARY"
    """
    A kind of CONSISTENT: the two claims are about the same subject and each supplies something the other does not, so together they are more informative than either alone.
    """
    CONTRADICTORY = "CONTRADICTORY"
    """
    The two claims cannot both be true: one asserts what the other denies, or they give incompatible values for the same relationship.
    """
    UNRELATED = "UNRELATED"
    """
    The two claims are about different subject matter, with no meaningful relation to classify. A pair that reaches the model but is unrelated is recorded so the judgement is explicit.
    """


class EvidentialRelation(str, Enum):
    """
    How the two claims' evidential standing relates, read from A to B and derived from each claim's `about` and `basis` (see claims.yaml). This says nothing about whether the claims agree; a pair can be IDENTICAL on the semantic axis while one side is a firsthand observation and the other is stated as background knowledge.
    """
    BOTH_OBSERVATION = "BOTH_OBSERVATION"
    """
    Both claims are presented as their source's own finding.
    """
    BOTH_SECONDARY = "BOTH_SECONDARY"
    """
    Both claims attribute the assertion to another work.
    """
    BOTH_BACKGROUND = "BOTH_BACKGROUND"
    """
    Both claims state the assertion as background knowledge, uncited.
    """
    A_OBSERVATION_B_SECONDARY = "A_OBSERVATION_B_SECONDARY"
    """
    A is a firsthand observation; B attributes it to another work.
    """
    A_SECONDARY_B_OBSERVATION = "A_SECONDARY_B_OBSERVATION"
    """
    A attributes it to another work; B is a firsthand observation.
    """
    A_OBSERVATION_B_BACKGROUND = "A_OBSERVATION_B_BACKGROUND"
    """
    A is a firsthand observation; B states it as background knowledge.
    """
    A_BACKGROUND_B_OBSERVATION = "A_BACKGROUND_B_OBSERVATION"
    """
    A states it as background knowledge; B is a firsthand observation.
    """
    A_SECONDARY_B_BACKGROUND = "A_SECONDARY_B_BACKGROUND"
    """
    A attributes it to another work; B states it as background knowledge.
    """
    A_BACKGROUND_B_SECONDARY = "A_BACKGROUND_B_SECONDARY"
    """
    A states it as background knowledge; B attributes it to another work.
    """
    NOT_APPLICABLE = "NOT_APPLICABLE"
    """
    At least one claim is about a work rather than the domain (claims.about is WORK), so the observation/attribution/background distinction does not apply.
    """
    UNDETERMINED = "UNDETERMINED"
    """
    At least one domain claim carries no basis, so the relation cannot be computed.
    """


class CandidateSignal(str, Enum):
    """
    Why a pair was proposed for classification.
    """
    ENTITY = "ENTITY"
    """
    The two claims share a grounded entity (the same CURIE).
    """
    CITATION = "CITATION"
    """
    The two claims share a citation that resolves to the same reference_id.
    """
    LEXICAL = "LEXICAL"
    """
    The two claims' content words overlap at or above the configured threshold (Jaccard over singularised, stop-word-filtered terms).
    """
    EMBEDDING = "EMBEDDING"
    """
    A caller-supplied embedder scored the two claim texts as similar. Only present when an embedder was provided.
    """



class ClaimAlignment(ConfiguredBaseModel):
    """
    The alignment of one source's claims (A) against another's (B): the pairs found and classified, with enough provenance to reproduce or audit them.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/alignment',
         'tree_root': True})

    source_a: AlignedSource = Field(default=..., description="""The first source, whose claims fill the A side of every pair.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimAlignment']} })
    source_b: AlignedSource = Field(default=..., description="""The second source, whose claims fill the B side of every pair.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimAlignment']} })
    aligner: AlignerInfo = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimAlignment']} })
    pairs: Optional[list[AlignedClaimPair]] = Field(default=None, description="""One per classified candidate pair, including pairs judged UNRELATED. A claim may appear in several pairs, or in none.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimAlignment']} })


class AlignedSource(ConfiguredBaseModel):
    """
    A source whose claims were aligned, identified as its ClaimSet.source was. A lightweight copy of claims.SourceDocument, carried so an alignment file is auditable on its own.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/alignment'})

    id: str = Field(default=..., description="""The source's id, as recorded on its ClaimSet (a path or URI).""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedSource', 'AlignedClaimPair']} })
    source_type: AlignedSourceType = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedSource']} })
    title: Optional[str] = Field(default=None, description="""The source's title, when it has one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedSource'], 'slot_uri': 'dcterms:title'} })
    content_sha256: Optional[str] = Field(default=None, description="""SHA-256 of the exact source text, copied from the ClaimSet, so spans in the aligned claims can still be checked against the source file.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedSource']} })


class AlignerInfo(ConfiguredBaseModel):
    """
    What produced an alignment, so it can be reproduced or compared.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/alignment'})

    name: str = Field(default=..., description="""Aligner identifier, for example llm-pairwise.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignerInfo', 'RubricMetric']} })
    model: Optional[str] = Field(default=None, description="""Model used to classify the semantic relation: the id the model's replies report, as ExtractorInfo records it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignerInfo']} })
    prompt_version: Optional[str] = Field(default=None, description="""Version of the classification prompt.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignerInfo']} })
    min_overlap: Optional[float] = Field(default=None, description="""The lexical overlap threshold used for candidate generation (Jaccard, 0 to 1).""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignerInfo']} })
    embedder_used: Optional[bool] = Field(default=None, description="""Whether a caller-supplied embedder augmented candidate generation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignerInfo']} })


class AlignedClaimPair(ConfiguredBaseModel):
    """
    One claim from source A paired with one from source B, classified on the semantic and evidential axes.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/alignment'})

    id: str = Field(default=..., description="""Identifier, unique within the alignment.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedSource', 'AlignedClaimPair']} })
    claim_a_id: str = Field(default=..., description="""The id of the A-side claim, within source A's ClaimSet.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })
    claim_b_id: str = Field(default=..., description="""The id of the B-side claim, within source B's ClaimSet.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })
    claim_a_text: str = Field(default=..., description="""The A-side claim's text, copied here so the pair reads on its own. The claim itself, with its provenance, is in source A's ClaimSet.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })
    claim_b_text: str = Field(default=..., description="""The B-side claim's text, copied here for the same reason.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })
    semantic_relation: SemanticRelation = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })
    evidential_relation: EvidentialRelation = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })
    shares_citation: Optional[bool] = Field(default=None, description="""True when both claims carry a citation resolving to the same reference_id: the same work is cited on both sides, so agreement between them is not independent corroboration.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })
    candidate_signals: Optional[list[CandidateSignal]] = Field(default=None, description="""Which signals proposed this pair for classification.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })
    confidence: Optional[float] = Field(default=None, description="""The model's confidence in the semantic relation, 0 to 1, when it gave one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })
    rationale: Optional[str] = Field(default=None, description="""One sentence from the model justifying the semantic relation.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignedClaimPair']} })


class RubricSummary(ConfiguredBaseModel):
    """
    A report-level summary of an alignment (issue #43, item 4): named metrics aggregated from the pair relations. Not a fixed schema: a downstream project may read these metrics, recompute them, or add its own.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/alignment'})

    reference: str = Field(default=..., description="""Which source the summary treats as the reference the other is judged against: \"a\" or \"b\". Completeness, for instance, is the share of the reference's claims that the other source covers.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RubricSummary']} })
    metrics: Optional[list[RubricMetric]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['RubricSummary']} })


class RubricMetric(ConfiguredBaseModel):
    """
    One named, auditable judgement aggregated from an alignment's pairs. The standard names are relevance, informativeness, support, completeness and citation_quality, but the set is open.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/alignment'})

    name: str = Field(default=..., description="""What the metric measures, for example completeness.""", json_schema_extra = { "linkml_meta": {'domain_of': ['AlignerInfo', 'RubricMetric']} })
    numerator: int = Field(default=..., description="""How many of the counted items met the metric's condition.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RubricMetric']} })
    denominator: int = Field(default=..., description="""How many items were counted. Zero means the metric did not apply (for example completeness when the reference has no claims); value is then null.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RubricMetric']} })
    value: Optional[float] = Field(default=None, description="""numerator / denominator, or null when denominator is zero.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RubricMetric']} })
    supporting_pair_ids: Optional[list[str]] = Field(default=None, description="""The pair ids this metric points to for audit: either the pairs behind the numerator or the shortfall behind it (for example the contradictory pairs), as the note says.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RubricMetric']} })
    supporting_claim_ids: Optional[list[str]] = Field(default=None, description="""The claim ids this metric points to for audit, for metrics counted over claims rather than pairs: either those behind the numerator or the shortfall (for example a source's uncovered or uncited claims), as the note says.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RubricMetric']} })
    note: Optional[str] = Field(default=None, description="""A short human-readable gloss of what the metric counted.""", json_schema_extra = { "linkml_meta": {'domain_of': ['RubricMetric']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
ClaimAlignment.model_rebuild()
AlignedSource.model_rebuild()
AlignerInfo.model_rebuild()
AlignedClaimPair.model_rebuild()
RubricSummary.model_rebuild()
RubricMetric.model_rebuild()
