"""The evidential axis: derived from two claims' about and basis, never judged.

This is the distinction the issue's flat relation list could not say: the same
conclusion can be a firsthand observation in one source and background knowledge
in another. It is a pure function of the claims, so it is tested without a model.
"""

import itertools

import pytest

from deep_research_client.claims import (
    AnchorStatus,
    CitationHandle,
    CitationScope,
    CitationStatus,
    Claim,
    ClaimBasis,
    ClaimTopic,
    EntityMention,
    EvidentialRelation,
    evidential_relation,
    shares_citation,
)


def _domain(basis) -> Claim:
    return Claim(
        id="x", claim_text="X.", anchor_status=AnchorStatus.EXACT, citation_status=CitationStatus.UNCITED,
        about=ClaimTopic.DOMAIN, basis=basis,
    )


OBS = ClaimBasis.OBSERVATION
SEC = ClaimBasis.SECONDARY_SOURCE
BG = ClaimBasis.BACKGROUND_KNOWLEDGE


@pytest.mark.parametrize(
    ("basis_a", "basis_b", "expected"),
    [
        (OBS, OBS, EvidentialRelation.BOTH_OBSERVATION),
        (SEC, SEC, EvidentialRelation.BOTH_SECONDARY),
        (BG, BG, EvidentialRelation.BOTH_BACKGROUND),
        (OBS, SEC, EvidentialRelation.A_OBSERVATION_B_SECONDARY),
        (SEC, OBS, EvidentialRelation.A_SECONDARY_B_OBSERVATION),
        (OBS, BG, EvidentialRelation.A_OBSERVATION_B_BACKGROUND),
        (BG, OBS, EvidentialRelation.A_BACKGROUND_B_OBSERVATION),
        (SEC, BG, EvidentialRelation.A_SECONDARY_B_BACKGROUND),
        (BG, SEC, EvidentialRelation.A_BACKGROUND_B_SECONDARY),
    ],
)
def test_the_basis_pair_maps_to_the_evidential_relation(basis_a, basis_b, expected) -> None:
    """Every ordered pair of bases has its own relation, read from A to B."""
    assert evidential_relation(_domain(basis_a), _domain(basis_b)) == expected


def test_every_pair_of_bases_has_a_mapping() -> None:
    """Two domain claims always get a concrete relation; a new basis would need one."""
    for basis_a, basis_b in itertools.product(ClaimBasis, ClaimBasis):
        relation = evidential_relation(_domain(basis_a), _domain(basis_b))
        assert isinstance(relation, EvidentialRelation)
        assert relation not in (EvidentialRelation.NOT_APPLICABLE, EvidentialRelation.UNDETERMINED)


def test_the_axis_is_direction_sensitive() -> None:
    """Swapping A and B swaps an asymmetric relation, the point Harry raised."""
    observed, assumed = _domain(OBS), _domain(BG)
    assert evidential_relation(observed, assumed) == EvidentialRelation.A_OBSERVATION_B_BACKGROUND
    assert evidential_relation(assumed, observed) == EvidentialRelation.A_BACKGROUND_B_OBSERVATION


def test_a_work_claim_makes_the_distinction_inapplicable() -> None:
    """about WORK has no basis, so the observation/attribution contrast cannot apply."""
    work = Claim(id="w", claim_text="A paper reports X.", anchor_status=AnchorStatus.EXACT,
                 citation_status=CitationStatus.UNCITED, about=ClaimTopic.WORK)
    assert evidential_relation(_domain(OBS), work) == EvidentialRelation.NOT_APPLICABLE
    assert evidential_relation(work, _domain(SEC)) == EvidentialRelation.NOT_APPLICABLE


def test_a_missing_basis_leaves_the_relation_undetermined() -> None:
    """A domain claim whose basis the extractor did not give cannot be placed."""
    no_basis = Claim(id="n", claim_text="X.", anchor_status=AnchorStatus.EXACT,
                     citation_status=CitationStatus.UNCITED, about=ClaimTopic.DOMAIN)
    assert evidential_relation(_domain(OBS), no_basis) == EvidentialRelation.UNDETERMINED


def _cited(ref) -> Claim:
    return Claim(
        id="c", claim_text="X.", anchor_status=AnchorStatus.EXACT, citation_status=CitationStatus.CITED,
        citations=[CitationHandle(marker=ref, reference_id=ref, scope=CitationScope.SENTENCE)],
    )


def test_shares_citation_needs_a_resolved_id_on_both_sides() -> None:
    """Independent corroboration versus the same source echoed turns on a shared id."""
    assert shares_citation(_cited("PMID:1"), _cited("PMID:1")) is True
    assert shares_citation(_cited("PMID:1"), _cited("PMID:2")) is False


def test_an_unresolved_marker_is_not_a_shared_citation() -> None:
    """Two reports' [1] markers are not the same work, so an unresolved id never matches."""
    unresolved = Claim(
        id="u", claim_text="X.", anchor_status=AnchorStatus.EXACT, citation_status=CitationStatus.CITED,
        citations=[CitationHandle(marker="[1]", scope=CitationScope.SENTENCE)],
    )
    assert shares_citation(unresolved, unresolved) is False


def test_subject_and_object_count_as_grounding_not_only_entities() -> None:
    """A claim grounded only through subject still shares an entity for candidate generation."""
    from deep_research_client.claims import candidate_pairs

    marfan = EntityMention(label="Marfan syndrome", id="MONDO:0007947")
    a = Claim(id="a", claim_text="Marfan syndrome is a connective tissue disorder.",
              anchor_status=AnchorStatus.EXACT, citation_status=CitationStatus.UNCITED, subject=marfan)
    b = Claim(id="b", claim_text="Something entirely different about Marfan syndrome here.",
              anchor_status=AnchorStatus.EXACT, citation_status=CitationStatus.UNCITED, entities=[marfan])
    pairs = candidate_pairs([a], [b], min_overlap=0.99)  # force entity, not lexical
    assert [s.value for s in pairs[0].signal_list] == ["ENTITY"]
