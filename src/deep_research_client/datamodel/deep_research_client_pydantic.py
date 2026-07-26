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


linkml_meta = LinkMLMeta({'default_prefix': 'drc',
     'default_range': 'string',
     'description': 'Controlled vocabularies describing what a deep research '
                    'provider can *do* (capabilities), what data resources it '
                    '*wraps* (resources), and where it sits on the spectrum from '
                    'pure literature retrieval to autonomous "co-scientist" agents '
                    '(archetype).\n'
                    'This schema is the single source of truth for those '
                    'enumerations. The Pydantic classes used at runtime are '
                    'generated from it into '
                    '``deep_research_client/datamodel/deep_research_client_pydantic.py`` '
                    '(see the ``gen-datamodel`` justfile target); do not edit the '
                    'generated file by hand.',
     'id': 'https://w3id.org/monarch/deep-research-client',
     'imports': ['linkml:types'],
     'license': 'BSD-3-Clause',
     'name': 'deep-research-client',
     'prefixes': {'drc': {'prefix_prefix': 'drc',
                          'prefix_reference': 'https://w3id.org/monarch/deep-research-client/'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'}},
     'source_file': 'src/deep_research_client/schema/deep_research_client.yaml',
     'title': 'Deep Research Client Data Model'} )

class ProviderArchetype(str, Enum):
    """
    Where a provider sits on the spectrum from pure retrieval to autonomous hypothesis-driven experimentation. The archetypes are ordered: each one is broadly a superset of the capabilities of the previous. A conventional "deep research" tool is a ``synthesizer``; it is a simple subset of what a ``co_scientist`` does (which additionally forms hypotheses, executes code, and designs experiments), sometimes traded off against more sophisticated literature-scanning algorithms.
    """
    retriever = "retriever"
    """
    Returns retrieved evidence (papers, passages, snippets) without an authored synthesis step. Example: Asta.
    """
    synthesizer = "synthesizer"
    """
    Searches one or more corpora and writes a cited narrative report. This is the classic "deep research" archetype. Example: OpenAI Deep Research, Perplexity, Consensus, Edison/Falcon.
    """
    agentic_researcher = "agentic_researcher"
    """
    A multi-step agent that plans, browses, and may execute tools/code to assemble a report, but is still oriented around producing a written research answer. Example: Claude Code, Cyberian.
    """
    co_scientist = "co_scientist"
    """
    An autonomous scientific agent that forms and tests hypotheses, executes code against biomedical data, and can propose or design experiments. A superset of the ``synthesizer`` case. Example: OpenScientist, Biomni.
    """


class ResearchCapability(str, Enum):
    """
    A functional capability a provider exposes: what kind of work it can perform. Values with legacy string identifiers are retained for backward compatibility with the earlier ``ModelCapability`` enumeration.
    """
    web_search = "web_search"
    """
    Searches the open web for current information.
    """
    academic_search = "academic_search"
    """
    Searches curated academic / scholarly corpora.
    """
    scientific_literature = "scientific_literature"
    """
    Specialised search and analysis over scientific literature.
    """
    citation_tracking = "citation_tracking"
    """
    Produces and tracks structured citations/references.
    """
    real_time_data = "real_time_data"
    """
    Can access recent / real-time information.
    """
    code_interpretation = "code_interpretation"
    """
    Executes code (data analysis, computation, plotting) as part of research. Formerly "runs code".
    """
    visual_analysis = "visual_analysis"
    """
    Interprets or generates figures, charts, and other visuals.
    """
    multi_language = "multi_language"
    """
    Operates across multiple natural languages.
    """
    retrieval_only = "retrieval_only"
    """
    Returns retrieved evidence verbatim without an authored synthesis step (pairs with the ``retriever`` archetype).
    """
    evidence_synthesis = "evidence_synthesis"
    """
    Synthesises multiple sources into an authored narrative report.
    """
    hypothesis_generation = "hypothesis_generation"
    """
    Proposes testable scientific hypotheses.
    """
    experiment_design = "experiment_design"
    """
    Designs experiments or analysis plans to test hypotheses.
    """
    data_analysis = "data_analysis"
    """
    Analyses structured biomedical or experimental datasets (beyond plain code execution), e.g. omics, sequences, or tabular study data.
    """


class ResearchResource(str, Enum):
    """
    A data source or knowledge base that a provider wraps or has direct access to. Distinct from capabilities: a capability is a verb (what it does), a resource is a noun (what it can reach).
    """
    general_web = "general_web"
    """
    The open web.
    """
    pubmed = "pubmed"
    """
    PubMed / MEDLINE biomedical literature.
    """
    semantic_scholar = "semantic_scholar"
    """
    Semantic Scholar scholarly corpus.
    """
    arxiv = "arxiv"
    """
    arXiv preprints.
    """
    preprint_servers = "preprint_servers"
    """
    Preprint servers such as bioRxiv / medRxiv.
    """
    clinical_trials = "clinical_trials"
    """
    Clinical trial registries (e.g. ClinicalTrials.gov).
    """
    biomedical_databases = "biomedical_databases"
    """
    Curated biomedical knowledge bases and ontologies (e.g. disease, pathway, and drug databases).
    """
    genomic_databases = "genomic_databases"
    """
    Genome, gene, and variant databases (e.g. Ensembl, ClinVar).
    """
    chemical_databases = "chemical_databases"
    """
    Chemical and compound databases (e.g. PubChem, ChEMBL).
    """
    protein_structure_databases = "protein_structure_databases"
    """
    Protein and structure databases (e.g. UniProt, PDB).
    """



# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
