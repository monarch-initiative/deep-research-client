"""The claim extraction entry point: one source in, one ClaimSet out.

How claims are found depends on what the source is:

- **markdown** (a deep research report, or any prose): an LLM decomposes each
  section into atomic claims, each tied back to the passage that states it.
- **dismech** and **gene-review** curated YAML: every curated record already
  is one claim, so records map directly, located by their path in the file,
  with their curated ontology terms as grounded entities and their evidence
  as citations. No model is involved.
- **structured** (any other YAML or JSON): the document's prose fields are
  decomposed by the LLM like a report's sections, each located by path.
"""

import asyncio
import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from ..evaluation.adapters.monarch import dismech_claims, gene_review_claims
from ..evaluation.datamodel import ReferenceClaim
from ..validation.extraction import find_reference_ids
from .llm import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, PROMPT_VERSION, decompose_units
from .models import (
    AnchorStatus,
    CitationHandle,
    CitationScope,
    Claim,
    ClaimBasis,
    ClaimSet,
    ClaimTopic,
    EntityMention,
    ExtractorInfo,
    SourceDocument,
    SourceType,
    citation_status_for,
    content_sha256,
)
from .units import markdown_units, report_title, structured_units

__all__ = [
    "SourceFormat",
    "aextract_claims",
    "detect_format",
    "extract_claims",
    "needs_llm",
    "resolve_format",
]


class SourceFormat(str, Enum):
    """How a source is read. ``AUTO`` decides from its extension and content."""

    AUTO = "auto"
    MARKDOWN = "markdown"
    DISMECH = "dismech"
    GENE_REVIEW = "gene-review"
    STRUCTURED = "structured"


#: Formats read by decomposing prose with a model.
_NEEDS_LLM = frozenset({SourceFormat.MARKDOWN, SourceFormat.STRUCTURED})

_STRUCTURED_SUFFIXES = frozenset({".yaml", ".yml", ".json"})


def _parse_structured(text: str, path: Path) -> Any:
    """Parse a YAML or JSON document."""
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def detect_format(path: Path, data: Any = None) -> SourceFormat:
    """Decide how to read a source from its extension and, if parsed, its keys.

    Args:
        path: The source file.
        data: The parsed document, for YAML or JSON sources.

    Returns:
        The format to read it as.

    >>> detect_format(Path("report.md"))
    <SourceFormat.MARKDOWN: 'markdown'>
    >>> detect_format(Path("x.yaml"), {"disease_term": {}, "phenotypes": []})
    <SourceFormat.DISMECH: 'dismech'>
    >>> detect_format(Path("x.yaml"), {"gene_symbol": "FBN1", "core_functions": []})
    <SourceFormat.GENE_REVIEW: 'gene-review'>
    >>> detect_format(Path("x.json"), {"notes": "..."})
    <SourceFormat.STRUCTURED: 'structured'>
    """
    if path.suffix.lower() not in _STRUCTURED_SUFFIXES:
        return SourceFormat.MARKDOWN
    if isinstance(data, dict):
        if "disease_term" in data or "pathophysiology" in data:
            return SourceFormat.DISMECH
        if "existing_annotations" in data or "core_functions" in data:
            return SourceFormat.GENE_REVIEW
    return SourceFormat.STRUCTURED


def _read(path: Path, source_format: SourceFormat) -> tuple[str, Any]:
    """Read a source, parsing it too when it is YAML or JSON not forced to markdown."""
    text = path.read_text(encoding="utf-8")
    data = None
    if source_format != SourceFormat.MARKDOWN and path.suffix.lower() in _STRUCTURED_SUFFIXES:
        data = _parse_structured(text, path)
    return text, data


def resolve_format(source: Path | str, source_format: SourceFormat | str = SourceFormat.AUTO) -> SourceFormat:
    """The format a source will be read as, reading it if ``auto`` must decide.

    Args:
        source: The source file.
        source_format: The requested format.

    Returns:
        A concrete format, never ``AUTO``.

    >>> resolve_format("report.md")
    <SourceFormat.MARKDOWN: 'markdown'>
    """
    fmt = SourceFormat(source_format)
    if fmt != SourceFormat.AUTO:
        return fmt
    path = Path(source)
    if path.suffix.lower() not in _STRUCTURED_SUFFIXES:
        return SourceFormat.MARKDOWN
    return detect_format(path, _read(path, fmt)[1])


def needs_llm(source_format: SourceFormat) -> bool:
    """Whether reading a source in this format calls a model.

    >>> needs_llm(SourceFormat.DISMECH), needs_llm(SourceFormat.MARKDOWN)
    (False, True)
    """
    return source_format in _NEEDS_LLM


def _structured_title(data: Any) -> Optional[str]:
    """A generic document's title: its top-level ``name``, when that is text.

    >>> _structured_title({"name": " Marfan notes "}), _structured_title({"name": 7})
    ('Marfan notes', '7')
    >>> _structured_title({"name": {"en": "x"}}), _structured_title([1, 2])
    (None, None)
    """
    name = data.get("name") if isinstance(data, dict) else None
    if isinstance(name, bool) or not isinstance(name, (str, int, float)):
        return None
    return str(name).strip() or None


def _curated_subject(data: dict[str, Any], source_format: SourceFormat) -> Optional[EntityMention]:
    """The disease or gene a curated file is about, which every record concerns."""
    if source_format == SourceFormat.DISMECH:
        term = (data.get("disease_term") or {}).get("term") or {}
        label = term.get("label") or data.get("name")
        return EntityMention(label=str(label), id=term.get("id")) if label else None
    label = data.get("gene_symbol")
    return EntityMention(label=str(label), id=_uniprot_curie(data.get("id"))) if label else None


def _uniprot_curie(accession: Any) -> Optional[str]:
    """An ai-gene-review file's ``id`` as a CURIE. Its ids are UniProt accessions.

    >>> _uniprot_curie("P35555"), _uniprot_curie("UniProtKB:P35555"), _uniprot_curie(None)
    ('UniProtKB:P35555', 'UniProtKB:P35555', None)
    """
    if not isinstance(accession, str) or not accession.strip():
        return None
    accession = accession.strip()
    return accession if ":" in accession else f"UniProtKB:{accession}"


#: What a curated record in each section asserts about the file's subject.
#: The section is the assertion; a record that also has a description keeps
#: it as the claim text, but the structure is the same either way.
_CURATED_PREDICATES = {
    "phenotype": "has phenotype",
    "treatment": "is treated by",
    "genetic_factor": "has mode of inheritance",
}


def _from_reference_claim(
    path: str, record: ReferenceClaim, subject: Optional[EntityMention],
) -> Claim:
    """One curated record as a claim, located by its path in the file.

    >>> from deep_research_client.evaluation.datamodel import OntologyTerm
    >>> record = ReferenceClaim(
    ...     category="phenotype", name="Ectopia lentis", description="",
    ...     ontology_terms=[OntologyTerm(id="HP:0001083", label="Ectopia lentis")],
    ... )
    >>> claim = _from_reference_claim("phenotypes[2]", record, EntityMention(label="Marfan syndrome"))
    >>> claim.claim_text, claim.predicate.label, claim.object.id
    ('Marfan syndrome has phenotype Ectopia lentis.', 'has phenotype', 'HP:0001083')
    >>> claim.about, claim.citation_status, claim.basis
    ('DOMAIN', 'UNCITED', None)
    """
    citations = []
    for evidence in record.evidence or []:
        found = find_reference_ids(evidence.reference)
        citations.append(CitationHandle(
            marker=evidence.reference,
            reference_id=found[0].normalized_id if found else None,
            scope=CitationScope.RECORD,
        ))
    entities = [
        EntityMention(label=term.label or term.id, id=term.id)
        for term in record.ontology_terms or []
    ]
    predicate_label = _CURATED_PREDICATES.get(record.category)
    predicate = EntityMention(label=predicate_label) if predicate_label else None
    related = None
    if predicate is not None:
        related = entities[0] if entities else EntityMention(label=record.name)

    claim_text = (record.description or "").strip()
    if not claim_text:
        claim_text = (
            f"{subject.label} {predicate_label} {record.name}."
            if subject is not None and predicate_label else record.name
        )
    return Claim(
        id=path,
        claim_text=claim_text,
        source_path=path,
        anchor_status=AnchorStatus.NOT_APPLICABLE,
        citation_status=citation_status_for(citations, AnchorStatus.NOT_APPLICABLE),
        about=ClaimTopic.DOMAIN,
        # Evidence is a cited work. A record with none does not say what it
        # rests on, so no basis is claimed for it.
        basis=ClaimBasis.SECONDARY_SOURCE if citations else None,
        subject=subject,
        predicate=predicate,
        object=related,
        entities=entities or None,
        citations=citations or None,
    )


async def aextract_claims(
    source: Path | str,
    *,
    source_format: SourceFormat | str = SourceFormat.AUTO,
    llm_client: Any = None,
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ClaimSet:
    """Extract the claims a source makes.

    Args:
        source: A markdown report, or a YAML or JSON document.
        source_format: How to read it; ``auto`` decides from the file.
        llm_client: An ``openai.AsyncOpenAI``-compatible client. Required for
            markdown and generic structured sources, unused for curated ones.
        model: Model for LLM decomposition.
        concurrency: LLM requests in flight at once.
        max_tokens: Reply budget per unit; a reply cut off at it raises.

    Returns:
        The claims, with the source's SHA-256 and the extractor recorded.

    Raises:
        ValueError: If the format needs a model and no client was given, or
            the source is not a document of the requested format.
    """
    path = Path(source)
    requested = SourceFormat(source_format)
    text, data = _read(path, requested)
    fmt = detect_format(path, data) if requested == SourceFormat.AUTO else requested
    if fmt == SourceFormat.STRUCTURED and not isinstance(data, (dict, list)):
        raise ValueError(
            f"{path} is not a YAML or JSON document with fields to read; "
            f"read prose as {SourceFormat.MARKDOWN.value}"
        )
    if needs_llm(fmt) and llm_client is None:
        raise ValueError(
            f"Extracting claims from {fmt.value} sources needs an LLM client"
        )

    title: Optional[str]
    if fmt == SourceFormat.MARKDOWN:
        title = report_title(text)
        claims = await decompose_units(
            markdown_units(text), llm_client, model,
            concurrency=concurrency, max_tokens=max_tokens,
        )
        extractor = ExtractorInfo(name="llm-atomic", model=model, prompt_version=PROMPT_VERSION)
    elif fmt == SourceFormat.STRUCTURED:
        title = _structured_title(data)
        claims = await decompose_units(
            structured_units(data), llm_client, model,
            concurrency=concurrency, max_tokens=max_tokens,
        )
        extractor = ExtractorInfo(name="llm-atomic", model=model, prompt_version=PROMPT_VERSION)
    else:
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not a {fmt.value} document")
        records = dismech_claims(data) if fmt == SourceFormat.DISMECH else gene_review_claims(data)
        subject = _curated_subject(data, fmt)
        title = subject.label if subject else None
        claims = [_from_reference_claim(p, record, subject) for p, record in records]
        extractor = ExtractorInfo(name=fmt.value)

    return ClaimSet(
        source=SourceDocument(
            id=str(source),
            source_type=(
                SourceType.MARKDOWN_REPORT if fmt == SourceFormat.MARKDOWN
                else SourceType.STRUCTURED_DOCUMENT
            ),
            title=title,
            content_sha256=content_sha256(text),
        ),
        extractor=extractor,
        claims=claims,
    )


def extract_claims(
    source: Path | str,
    *,
    source_format: SourceFormat | str = SourceFormat.AUTO,
    llm_client: Any = None,
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> ClaimSet:
    """Synchronous :func:`aextract_claims`, for callers without an event loop.

    It runs its own loop with :func:`asyncio.run`, which raises
    ``RuntimeError`` inside one that is already running, as in Jupyter. There,
    ``await aextract_claims(...)`` instead.

    Args:
        source: A markdown report, or a YAML or JSON document.
        source_format: How to read it; ``auto`` decides from the file.
        llm_client: An ``openai.AsyncOpenAI``-compatible client, for prose.
        model: Model for LLM decomposition.
        concurrency: LLM requests in flight at once.
        max_tokens: Reply budget per unit; a reply cut off at it raises.

    Returns:
        The claims the source makes.
    """
    return asyncio.run(aextract_claims(
        source, source_format=source_format, llm_client=llm_client,
        model=model, concurrency=concurrency, max_tokens=max_tokens,
    ))
