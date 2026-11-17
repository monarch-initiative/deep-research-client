"""Grounding orchestration: ranking, capping and propagation, with no live service.

The backend that decides a CURIE is injected (a real Grounder implementation, the
way the embedder is injected into candidate generation), so what is tested here is
the pure orchestration: which labels get grounded, in what order, and how the
result reaches every mention.
"""

from typing import Optional

import pytest

from deep_research_client.claims import (
    AnchorStatus,
    CitationStatus,
    Claim,
    ClaimSet,
    EntityMention,
    ExtractorInfo,
    GroundedTerm,
    GrounderInfo,
    SourceDocument,
    SourceType,
    ground_claims,
)
from deep_research_client.claims.grounding import normalized_label, rank_labels


class FakeGrounder:
    """A Grounder that grounds from a fixed label→CURIE map and records its calls."""

    def __init__(self, mapping: dict[str, GroundedTerm]) -> None:
        self.mapping = {normalized_label(k): v for k, v in mapping.items()}
        self.calls: list[str] = []

    async def aground_term(self, label: str, context: str) -> Optional[GroundedTerm]:
        self.calls.append(label)
        return self.mapping.get(normalized_label(label))

    def provenance(self, considered: int, grounded: int) -> GrounderInfo:
        return GrounderInfo(name="fake", considered=considered, grounded=grounded)


def _claim(cid, *, entities=None, subject=None, obj=None) -> Claim:
    return Claim(
        id=cid, claim_text=f"Sentence {cid}.",
        anchor_status=AnchorStatus.EXACT, citation_status=CitationStatus.UNCITED,
        entities=[EntityMention(label=lbl) for lbl in (entities or [])] or None,
        subject=EntityMention(label=subject) if subject else None,
        object=EntityMention(label=obj) if obj else None,
    )


def _set(claims) -> ClaimSet:
    return ClaimSet(
        source=SourceDocument(id="r.md", source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="x"), claims=claims,
    )


def _entity_ids(claim: Claim) -> dict[str, Optional[str]]:
    return {e.label: e.id for e in (claim.entities or [])}


def test_labels_are_ranked_by_frequency_then_name() -> None:
    cs = _set([
        _claim("c1", entities=["acne", "TLR2"]),
        _claim("c2", entities=["Acne"]),  # same as "acne" after normalising
        _claim("c3", entities=["sebum"]),
    ])
    assert [(r.label, r.count) for r in rank_labels(cs)] == [("acne", 2), ("TLR2", 1), ("sebum", 1)]


def test_an_already_grounded_mention_is_not_reconsidered() -> None:
    claim = _claim("c1", entities=["acne"])
    assert claim.entities is not None
    claim.entities[0].id = "MONDO:already"  # pretend the source grounded it
    assert rank_labels(_set([claim])) == []


def test_grounding_fills_ids_across_entities_subject_and_object() -> None:
    cs = _set([_claim("c1", entities=["acne"], subject="FBN1", obj="acne")])
    grounder = FakeGrounder({
        "acne": GroundedTerm(id="MONDO:0011438", label="acne"),
        "FBN1": GroundedTerm(id="HGNC:3603", label="FBN1"),
    })
    grounded = ground_claims(cs, grounder)
    claim = grounded.claim_list[0]
    assert _entity_ids(claim) == {"acne": "MONDO:0011438"}
    assert claim.subject is not None and claim.subject.id == "HGNC:3603"
    assert claim.object is not None and claim.object.id == "MONDO:0011438"  # grounded once by label
    assert claim.entities is not None and claim.entities[0].label == "acne"  # label untouched
    assert grounded.grounder == GrounderInfo(name="fake", considered=2, grounded=2)
    # The grounded set, provenance and all, round-trips through JSON.
    reloaded = ClaimSet.model_validate(grounded.model_dump(mode="json"))
    assert reloaded.grounder == GrounderInfo(name="fake", considered=2, grounded=2)
    assert reloaded.claim_list[0].subject is not None
    assert reloaded.claim_list[0].subject.id == "HGNC:3603"


def test_max_terms_grounds_only_the_most_frequent() -> None:
    cs = _set([
        _claim("c1", entities=["acne"]),
        _claim("c2", entities=["acne"]),
        _claim("c3", entities=["sebum"]),
    ])
    grounder = FakeGrounder({
        "acne": GroundedTerm(id="MONDO:0011438", label="acne"),
        "sebum": GroundedTerm(id="CHEBI:0", label="sebum"),
    })
    grounded = ground_claims(cs, grounder, max_terms=1)
    assert grounder.calls == ["acne"]  # only the most frequent was tried
    assert _entity_ids(grounded.claim_list[0]) == {"acne": "MONDO:0011438"}
    assert _entity_ids(grounded.claim_list[2]) == {"sebum": None}  # left ungrounded
    assert grounded.grounder is not None
    assert (grounded.grounder.considered, grounded.grounder.grounded) == (1, 1)


def test_a_label_the_grounder_declines_is_left_alone() -> None:
    cs = _set([_claim("c1", entities=["acne", "mysterious thing"])])
    grounder = FakeGrounder({"acne": GroundedTerm(id="MONDO:0011438", label="acne")})
    grounded = ground_claims(cs, grounder)
    assert _entity_ids(grounded.claim_list[0]) == {"acne": "MONDO:0011438", "mysterious thing": None}
    assert grounded.grounder is not None
    assert (grounded.grounder.considered, grounded.grounder.grounded) == (2, 1)


@pytest.mark.parametrize(
    ("a", "b"),
    [("  Marfan   syndrome ", "marfan syndrome"), ("FBN1", "fbn1"), ("IL-1β", "il-1β")],
)
def test_normalized_label_folds_case_and_whitespace(a, b) -> None:
    assert normalized_label(a) == normalized_label(b.upper()) == b
