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


linkml_meta = LinkMLMeta({'default_prefix': 'verification',
     'default_range': 'string',
     'description': "Data model for verifying a source's claims and surfacing the "
                    'ones that need human review (issue #43, item 5). Extraction '
                    '(claims.yaml) records what a source says; alignment '
                    "(alignment.yaml) relates one source's claims to another's; "
                    'this schema records what is wrong, or unproven, and who can '
                    'settle it.\n'
                    'Two products sit in a VerificationReport:\n'
                    '- flags: a ReviewFlag per claim or pair that a person (or a '
                    'downstream tool)\n'
                    '  should look at -- a domain claim supported by nothing, a '
                    'citation that\n'
                    '  resolves to no identifier, a contradiction, a claim that '
                    'frames a disputed\n'
                    '  point as settled consensus. Each flag carries a '
                    '`decidability`: whether the\n'
                    '  question it raises can be settled by a machine rule, by an '
                    'agent with tools,\n'
                    '  or only by a human. This is the heart of the schema: the '
                    'flags do not\n'
                    '  adjudicate, they route.\n'
                    '- verifications: a ClaimVerification per claim a Verifier '
                    'checked, recording\n'
                    '  the verdict (supported, unsupported, contradicted, '
                    'unverifiable, needs\n'
                    '  review), the method, and enough provenance to reproduce it. '
                    'A Verifier is a\n'
                    '  pluggable hook (see verification.py); citation verification '
                    'against the cited\n'
                    '  work is the one backend shipped here, and it connects to '
                    'publication\n'
                    '  identifier normalisation (issue #41).\n'
                    '\n'
                    'This schema is standalone: it imports only linkml:types and '
                    'redeclares a lightweight SourceType rather than importing '
                    'claims.yaml, so each datamodel generates independently (as '
                    'claims.yaml and alignment.yaml already do) and the drift test '
                    'stays a single-file regenerate-and-diff.\n'
                    'This schema is the source of truth for '
                    'deep_research_client/claims/verification_datamodel.py, '
                    'generated with `just gen-datamodel`. Derived views are added '
                    'in deep_research_client/claims/verification_models.py; they '
                    'are not stored here.',
     'id': 'https://w3id.org/monarch-initiative/deep-research-client/verification',
     'imports': ['linkml:types'],
     'license': 'BSD-3-Clause',
     'name': 'verification',
     'prefixes': {'dcterms': {'prefix_prefix': 'dcterms',
                              'prefix_reference': 'http://purl.org/dc/terms/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'verification': {'prefix_prefix': 'verification',
                                   'prefix_reference': 'https://w3id.org/monarch-initiative/deep-research-client/verification/'}},
     'source_file': 'src/deep_research_client/claims/verification.yaml',
     'title': 'Deep Research Client Claim Verification'} )

class VerifiedSourceType(str, Enum):
    """
    What kind of document the verified claims came from. Mirrors claims.SourceType; redeclared here to keep this schema standalone.
    """
    MARKDOWN_REPORT = "MARKDOWN_REPORT"
    """
    A deep research report or other markdown prose.
    """
    STRUCTURED_DOCUMENT = "STRUCTURED_DOCUMENT"
    """
    A YAML or JSON knowledge artifact.
    """


class Decidability(str, Enum):
    """
    Who can settle the question a review flag raises. The flag is always detected by a machine rule; this says who can adjudicate it, which is not the same thing. A claim that cites nothing is detected mechanically, but whether it is actually unsupported is an agent's or a person's judgement.
    """
    MACHINE = "MACHINE"
    """
    A deterministic rule settles it with no further input, for example a citation marker that resolves to no identifier at all.
    """
    AGENT = "AGENT"
    """
    An automated agent with tools (search, retrieval, identifier resolution) could settle it, though it may still escalate to a human.
    """
    HUMAN = "HUMAN"
    """
    Only a person can settle it. A contradiction between two claims, or a disputed point framed as consensus, is a judgement no rule or tool makes reliably.
    """


class FlagKind(str, Enum):
    """
    The kind of problem a review flag marks.
    """
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    """
    A claim about the domain is presented as a finding or an attributed result, yet cites nothing. It may be true, but the source gives nothing to stand on. Background-knowledge claims are expected to be uncited and are not flagged.
    """
    UNRESOLVED_CITATION = "UNRESOLVED_CITATION"
    """
    A claim is cited, but none of its citations resolve to a normalised identifier (a PMID, DOI, and so on): the marker points nowhere that can be checked. Connects to publication identifier normalisation (#41).
    """
    CONFABULATED_CITATION = "CONFABULATED_CITATION"
    """
    A claim cites a normalised identifier that resolves to no real record: the work appears to have been fabricated. A resolver settled this, so it is one of the few findings a machine can decide outright.
    """
    UNCHECKED_CITATION = "UNCHECKED_CITATION"
    """
    A claim cites a resolvable identifier, but no verifier confirmed the cited work supports it. Raised when verification ran but produced no supporting or contradicting verdict for the claim.
    """
    WEAK_CITATION = "WEAK_CITATION"
    """
    A citation resolves and the work exists, but the work appears to be about something other than the claim (low topical relevance). The citation is real but may not support this claim.
    """
    CONTRADICTION = "CONTRADICTION"
    """
    Two claims from different sources cannot both be true. Which, if either, is right is a human judgement.
    """
    SELF_CONTRADICTION = "SELF_CONTRADICTION"
    """
    Two claims from the same source cannot both be true: the source contradicts itself. Flagged apart from CONTRADICTION because a source disagreeing with itself is a different and usually worse problem.
    """
    CONSENSUS_OVERSTATED = "CONSENSUS_OVERSTATED"
    """
    A claim frames a point as settled or universal ("well established", "consensus", "it is well known") while either citing nothing or taking part in a contradiction. A disputed point dressed as consensus. A lexical heuristic, so always a human's call.
    """


class VerificationStatus(str, Enum):
    """
    A Verifier's verdict on one claim.
    """
    SUPPORTED = "SUPPORTED"
    """
    The cited work or the evidence checked supports the claim.
    """
    UNSUPPORTED = "UNSUPPORTED"
    """
    The claim's support could not be found: a cited identifier does not resolve to a real work, or no evidence was located for it.
    """
    CONTRADICTED = "CONTRADICTED"
    """
    The evidence checked contradicts the claim.
    """
    UNVERIFIABLE = "UNVERIFIABLE"
    """
    No check could be run: no resolver for the identifier's kind, or the claim offered nothing checkable. Nothing was learned either way.
    """
    NEEDS_REVIEW = "NEEDS_REVIEW"
    """
    The check ran but its result is not conclusive and a person should look, for example a citation that resolves but is only weakly on topic.
    """



class VerificationReport(ConfiguredBaseModel):
    """
    The verification of one source's claims: the review flags raised and the per-claim verdicts, with enough provenance to reproduce or audit them.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/verification',
         'tree_root': True})

    source_id: str = Field(default=..., description="""The verified source's id, as recorded on its ClaimSet (a path or URI).""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerificationReport']} })
    source_type: Optional[VerifiedSourceType] = Field(default=None, description="""What kind of document the claims came from, when known.""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerificationReport']} })
    verifier: VerifierInfo = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['VerificationReport', 'ClaimVerification']} })
    flags: Optional[list[ReviewFlag]] = Field(default=None, description="""One per claim or pair that needs review. A claim may raise several flags, or none.""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerificationReport']} })
    verifications: Optional[list[ClaimVerification]] = Field(default=None, description="""One per claim a Verifier checked. Empty when the report is triage-only (no Verifier was run).""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerificationReport']} })


class VerifierInfo(ConfiguredBaseModel):
    """
    What produced a verification, so it can be reproduced or compared.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/verification'})

    name: str = Field(default=..., description="""Verifier identifier, for example \"triage\" (flags only) or \"triage+citation\".""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerifierInfo']} })
    methods: Optional[list[str]] = Field(default=None, description="""The checks that ran, for example \"triage\" and \"citation:resolution\".""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerifierInfo']} })
    model: Optional[str] = Field(default=None, description="""Model used, when a verifier called one: the id its replies report. None for the rule-based triage and citation checks shipped here.""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerifierInfo']} })
    prompt_version: Optional[str] = Field(default=None, description="""Version of the prompt, when a model was used.""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerifierInfo']} })
    considered: Optional[int] = Field(default=None, description="""How many claims were examined.""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerifierInfo']} })
    flagged: Optional[int] = Field(default=None, description="""How many claims or pairs raised at least one review flag.""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerifierInfo']} })
    verified: Optional[int] = Field(default=None, description="""How many claims a Verifier reached a verdict on.""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerifierInfo']} })


class ReviewFlag(ConfiguredBaseModel):
    """
    One claim or pair a person or downstream tool should review, with the kind of problem and who can settle it.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/verification'})

    kind: FlagKind = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ReviewFlag']} })
    decidability: Decidability = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ReviewFlag']} })
    claim_ids: Optional[list[str]] = Field(default=None, description="""The claim ids the flag concerns, within the verified source's ClaimSet. One for a claim-level flag; two for a contradiction between claims.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReviewFlag']} })
    pair_id: Optional[str] = Field(default=None, description="""The alignment pair id this flag came from, for a contradiction. Empty for a claim-level flag.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReviewFlag']} })
    message: str = Field(default=..., description="""A short human-readable statement of what to review and why.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReviewFlag']} })
    evidence: Optional[str] = Field(default=None, description="""The claim text, cue, or other short snippet the flag was raised from, so a reviewer sees what triggered it without opening the source.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ReviewFlag']} })


class ClaimVerification(ConfiguredBaseModel):
    """
    One Verifier's verdict on one claim, with its method and provenance.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/verification'})

    claim_id: str = Field(default=..., description="""The id of the checked claim, within the verified source's ClaimSet.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimVerification']} })
    status: VerificationStatus = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimVerification']} })
    method: str = Field(default=..., description="""How the verdict was reached, for example \"citation:resolution\" or \"citation:relevance\".""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimVerification']} })
    reference_id: Optional[str] = Field(default=None, description="""The normalised identifier that was checked, when the verdict is about a cited work (a PMID, DOI, and so on).""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimVerification'], 'slot_uri': 'dcterms:references'} })
    confidence: Optional[float] = Field(default=None, description="""The verifier's confidence in the verdict, 0 to 1, when it gave one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimVerification']} })
    note: Optional[str] = Field(default=None, description="""A short human-readable gloss of the verdict.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimVerification']} })
    verifier: Optional[str] = Field(default=None, description="""Which verifier produced this verdict, when a report mixes several. The same name the VerifierInfo would carry for that backend.""", json_schema_extra = { "linkml_meta": {'domain_of': ['VerificationReport', 'ClaimVerification']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
VerificationReport.model_rebuild()
VerifierInfo.model_rebuild()
ReviewFlag.model_rebuild()
ClaimVerification.model_rebuild()
