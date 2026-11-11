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


linkml_meta = LinkMLMeta({'default_prefix': 'claims',
     'default_range': 'string',
     'description': 'Data model for the claims a source makes: a deep research '
                    'report, a curated knowledge artifact, a review. Each claim is '
                    'one atomic assertion with enough provenance for a person to '
                    'find it in the source again. Nothing here judges whether a '
                    'claim is true; that is left to alignment and verification, '
                    'which consume these records (issue #43).\n'
                    'Where a slot means what a W3C Web Annotation or Dublin Core '
                    "term means, it declares that term as its slot_uri: a span's "
                    'text is oa:exact and its offsets oa:start and oa:end, a '
                    "citation's identifier is dcterms:references, a source's title "
                    'dcterms:title. These are the URIs linkml-reference-validator '
                    'finds excerpt, reference and title fields by. It pairs an '
                    'excerpt with a reference only within one class, and a span '
                    "(the source's own words) and a citation (the work cited for "
                    'them) are different classes, so it reads these fields without '
                    "validating a report's sentence as if it were a quote from the "
                    'cited paper.\n'
                    "Slot names for the assertion's structure mirror OntoGPT's "
                    'core Triple and its ScientificClaim (subject, predicate, '
                    'object, qualifier, negated), so the two can be converted '
                    'without a mapping table. Unlike OntoGPT, a claim carries its '
                    'own span: character offsets into the source, checked by '
                    'locating the quote the extractor gave, rather than a label '
                    'searched for afterwards.\n'
                    'This schema is the source of truth for '
                    'deep_research_client/claims/datamodel.py, which is generated '
                    'from it with `just gen-datamodel`. Derived views are added in '
                    'deep_research_client/claims/models.py; they are computed from '
                    'these slots rather than stored.',
     'id': 'https://w3id.org/monarch-initiative/deep-research-client/claims',
     'imports': ['linkml:types'],
     'license': 'BSD-3-Clause',
     'name': 'claims',
     'prefixes': {'claims': {'prefix_prefix': 'claims',
                             'prefix_reference': 'https://w3id.org/monarch-initiative/deep-research-client/claims/'},
                  'dcterms': {'prefix_prefix': 'dcterms',
                              'prefix_reference': 'http://purl.org/dc/terms/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'oa': {'prefix_prefix': 'oa',
                         'prefix_reference': 'http://www.w3.org/ns/oa#'}},
     'source_file': 'src/deep_research_client/claims/claims.yaml',
     'title': 'Deep Research Client Claims'} )

class SourceType(str, Enum):
    """
    What kind of document the claims were extracted from.
    """
    MARKDOWN_REPORT = "MARKDOWN_REPORT"
    """
    A deep research report or other markdown prose. Claims are located by character offsets into the file's text.
    """
    STRUCTURED_DOCUMENT = "STRUCTURED_DOCUMENT"
    """
    A YAML or JSON knowledge artifact. Claims are located by a path into the parsed structure; prose inside it is additionally located by offsets into that field's text.
    """


class AnchorStatus(str, Enum):
    """
    How a claim was tied back to its source. Offsets are only as good as this: an extractor's quote that cannot be found in the source is not evidence of anything.
    """
    EXACT = "EXACT"
    """
    The extractor's quote occurs verbatim in the source.
    """
    NORMALIZED = "NORMALIZED"
    """
    The quote occurs once whitespace, markdown emphasis and quotation mark styles are ignored. The span still points at the original text.
    """
    UNANCHORED = "UNANCHORED"
    """
    The quote could not be found. The claim is kept so the gap is visible, but it has no span and should not be trusted as coming from the source.
    """
    NOT_APPLICABLE = "NOT_APPLICABLE"
    """
    The claim is a structured record rather than prose, so it is located by source_path and there is no quote to anchor.
    """


class ClaimTopic(str, Enum):
    """
    What a claim is about. A claim about a work is checked against bibliographic records; a claim about the domain is checked against what is known of the world.
    """
    WORK = "WORK"
    """
    A publication or other work as an object: its authors, venue, date, identifiers, or what it covers ("this review discusses X"). Deep research reports that list papers make many of these.
    """
    DOMAIN = "DOMAIN"
    """
    The subject matter the works study: genes, diseases, drugs, outcomes, mechanisms.
    """


class ClaimBasis(str, Enum):
    """
    What a claim about the domain rests on, as the source presents it. This is the extractor's judgement of how the source states the claim, not a check of whether the basis holds up.
    """
    OBSERVATION = "OBSERVATION"
    """
    The source document presents it as its own finding: an experiment, analysis, dataset or case it made or ran. A finding the source reports from another work, even word for word (a quoted abstract, a section headed by a cited paper), is SECONDARY_SOURCE.
    """
    SECONDARY_SOURCE = "SECONDARY_SOURCE"
    """
    The source attributes it to another work, by a citation marker or by naming the work or its authors ("Smith et al. showed").
    """
    BACKGROUND_KNOWLEDGE = "BACKGROUND_KNOWLEDGE"
    """
    The source states it with no citation or attribution, as knowledge taken to be true within the field.
    """


class CitationStatus(str, Enum):
    """
    Whether the source cites a claim, as checked against the source. This makes the absence of a citation explicit: an empty citations list alone cannot tell "the source cites nothing here" from "this was not checked".
    """
    CITED = "CITED"
    """
    The source attaches at least one citation to the claim, and every one is in citations. For prose, a marker counts when it is in the claim's own sentence, and so does the cited work a section is headed by; each citation's scope says which.
    """
    UNCITED = "UNCITED"
    """
    The claim was located and no citation is attached to it. For prose, its sentence may still cite a source for a different claim. For a curated record, the record carries no evidence.
    """
    UNKNOWN = "UNKNOWN"
    """
    Whether the source cites the claim could not be checked, because the claim's passage was not found (anchor_status UNANCHORED). Markers the extractor reported for it are not kept.
    """


class CitationScope(str, Enum):
    """
    How a citation came to be attached to a claim.
    """
    SENTENCE = "SENTENCE"
    """
    A marker in the claim's own sentence (or table row, or list item).
    """
    SECTION = "SECTION"
    """
    The claim sits in a section headed by a cited work, such as "### [3] Title" in a report that lists papers, so the whole section is attributed to that work.
    """
    RECORD = "RECORD"
    """
    The evidence a curated record lists for itself.
    """



class ClaimSet(ConfiguredBaseModel):
    """
    The claims extracted from one source document.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/claims',
         'tree_root': True})

    source: SourceDocument = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimSet']} })
    extractor: ExtractorInfo = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimSet']} })
    claims: Optional[list[Claim]] = Field(default=None, json_schema_extra = { "linkml_meta": {'domain_of': ['ClaimSet']} })


class SourceDocument(ConfiguredBaseModel):
    """
    The document claims were extracted from.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/claims'})

    id: str = Field(default=..., description="""Where the source came from: a file path or URI, as given to the extractor.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SourceDocument', 'Claim', 'EntityMention']} })
    source_type: SourceType = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['SourceDocument']} })
    title: Optional[str] = Field(default=None, description="""The document's own title, when it has one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SourceDocument'], 'slot_uri': 'dcterms:title'} })
    content_sha256: Optional[str] = Field(default=None, description="""SHA-256 of the exact text that offsets refer to. With it, anyone can confirm that a span still points at what it claims to, using only the source file.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SourceDocument']} })


class ExtractorInfo(ConfiguredBaseModel):
    """
    What produced the claims, so a set can be reproduced or compared.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/claims'})

    name: str = Field(default=..., description="""Extractor identifier, for example llm-atomic or dismech.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExtractorInfo']} })
    model: Optional[str] = Field(default=None, description="""Model used, for extractors that call one: the id the model's replies report, so an alias such as sonnet is recorded as the model it resolved to. Several ids, comma separated, if replies differed; the name requested, if no reply named one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExtractorInfo']} })
    prompt_version: Optional[str] = Field(default=None, description="""Version of the extraction prompt, for extractors that use one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['ExtractorInfo']} })


class Claim(ConfiguredBaseModel):
    """
    One atomic assertion made by the source: a single statement that could in principle be found true or false on its own.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/claims'})

    id: str = Field(default=..., description="""Identifier, unique within its ClaimSet.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SourceDocument', 'Claim', 'EntityMention']} })
    claim_text: str = Field(default=..., description="""The assertion as a standalone sentence. May be decontextualised from the source (a pronoun replaced by what it refers to, a table row's header folded in) so that claims can be compared across sources; the words the source actually used are in source_span.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    source_span: Optional[TextSpan] = Field(default=None, description="""Where the claim is stated, for prose. Offsets are into the whole source text for a report, or into the field's text for prose inside a structured document.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    source_path: Optional[str] = Field(default=None, description="""Where the claim sits in a structured document, as a path such as pathophysiology[2].description. Absent for markdown reports.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    anchor_status: AnchorStatus = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    section: Optional[str] = Field(default=None, description="""Heading path of the section the claim appears in, for example \"Mechanism > Fibrillin\". Absent when the source has no headings.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    subject: Optional[EntityMention] = Field(default=None, description="""What the claim is about, as in OntoGPT's Triple.subject.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    predicate: Optional[EntityMention] = Field(default=None, description="""The relationship asserted, as in OntoGPT's Triple.predicate. May be grounded to a relation ontology such as RO or biolink.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    object: Optional[EntityMention] = Field(default=None, description="""What the subject is related to, as in OntoGPT's Triple.object.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    negated: Optional[bool] = Field(default=None, description="""True when the source asserts that the relationship does not hold, as in OntoGPT's ScientificClaim.negated.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    qualifier: Optional[str] = Field(default=None, description="""A qualifier on the whole statement, as in OntoGPT's Triple.qualifier: for example a hedge (\"may\"), a population, or a condition.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    subject_qualifier: Optional[str] = Field(default=None, description="""A modifier of the subject, for example \"high dose\".""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    object_qualifier: Optional[str] = Field(default=None, description="""A modifier of the object, for example \"severe\".""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    entities: Optional[list[EntityMention]] = Field(default=None, description="""The things the claim mentions, whether or not they fill subject or object. Anchors for aligning claims across sources.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    citations: Optional[list[CitationHandle]] = Field(default=None, description="""The references the source attaches to this claim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    nearest_passage: Optional[NearestPassage] = Field(default=None, description="""For an UNANCHORED claim only: the passage of its unit closest to the quote the extractor gave, to show why the quote was not found (a paraphrase, a quote joined from two places, a wrong section). It is a diagnostic, not a location: nothing about the claim should be taken as coming from it.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    citation_status: CitationStatus = Field(default=..., description="""Whether the source cites this claim: CITED exactly when citations is non-empty, UNKNOWN exactly when the claim is UNANCHORED.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    about: Optional[ClaimTopic] = Field(default=None, description="""Whether the claim is about a work or about the domain. Absent when the extractor did not say.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })
    basis: Optional[ClaimBasis] = Field(default=None, description="""What a claim about the domain rests on, as the source presents it. Set only when about is DOMAIN. The extractor's judgement, except in a section headed by a cited work, where it is SECONDARY_SOURCE by the section's structure.""", json_schema_extra = { "linkml_meta": {'domain_of': ['Claim']} })


class TextSpan(ConfiguredBaseModel):
    """
    A contiguous stretch of source text. Offsets count Unicode code points, as Python string indexing does, with end exclusive, so text == source[start:end]. Together these are a Web Annotation text position selector (start, end) and text quote selector (exact).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/claims'})

    start: int = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['TextSpan'], 'slot_uri': 'oa:start'} })
    end: int = Field(default=..., json_schema_extra = { "linkml_meta": {'domain_of': ['TextSpan'], 'slot_uri': 'oa:end'} })
    text: str = Field(default=..., description="""The source text between start and end, verbatim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['TextSpan', 'NearestPassage'], 'slot_uri': 'oa:exact'} })


class NearestPassage(ConfiguredBaseModel):
    """
    The passage closest to a quote that could not be anchored, found by linkml-reference-validator's fuzzy matcher: the unit's sentences are scored against the quote, and one with a high enough score and enough of the quote's content words is kept.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/claims'})

    text: str = Field(default=..., description="""The passage, as the matcher returned it (whitespace collapsed).""", json_schema_extra = { "linkml_meta": {'domain_of': ['TextSpan', 'NearestPassage']} })
    score: float = Field(default=..., description="""Similarity to the quote, 0 to 100.""", json_schema_extra = { "linkml_meta": {'domain_of': ['NearestPassage']} })
    span: Optional[TextSpan] = Field(default=None, description="""Where the passage is in the source, when it can be located there.""", json_schema_extra = { "linkml_meta": {'domain_of': ['NearestPassage']} })


class EntityMention(ConfiguredBaseModel):
    """
    Something a claim refers to. Mirrors OntoGPT's NamedEntity: a label as written, and an identifier when it has been grounded.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/claims'})

    label: str = Field(default=..., description="""The entity as written or as named by the extractor.""", json_schema_extra = { "linkml_meta": {'domain_of': ['EntityMention']} })
    id: Optional[str] = Field(default=None, description="""A CURIE, when the entity is grounded to an ontology or database, for example MONDO:0007947.""", json_schema_extra = { "linkml_meta": {'domain_of': ['SourceDocument', 'Claim', 'EntityMention']} })


class CitationHandle(ConfiguredBaseModel):
    """
    A reference attached to a claim, kept both as written and, where it can be resolved, as a normalised identifier. Normalisation is deliberately shallow here; it is the job of publication identifier normalisation (issue #41).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://w3id.org/monarch-initiative/deep-research-client/claims'})

    marker: str = Field(default=..., description="""The citation as it appears in the source, for example \"[3]\", \"PMID:7913883\" or a URL.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CitationHandle']} })
    reference_id: Optional[str] = Field(default=None, description="""Normalised identifier when one could be read from the marker or the source's bibliography, for example PMID:7913883 or DOI:10.1038/x.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CitationHandle'], 'slot_uri': 'dcterms:references'} })
    url: Optional[str] = Field(default=None, description="""URL of the reference, when the marker or bibliography gives one.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CitationHandle']} })
    scope: CitationScope = Field(default=..., description="""How the citation was attached to the claim.""", json_schema_extra = { "linkml_meta": {'domain_of': ['CitationHandle']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
ClaimSet.model_rebuild()
SourceDocument.model_rebuild()
ExtractorInfo.model_rebuild()
Claim.model_rebuild()
TextSpan.model_rebuild()
NearestPassage.model_rebuild()
EntityMention.model_rebuild()
CitationHandle.model_rebuild()
