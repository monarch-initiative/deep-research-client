"""Pure review-flag triage (issue #43, item 5)."""

from deep_research_client.claims import (
    AlignedClaimPair,
    AlignedSource,
    AlignedSourceType,
    AlignerInfo,
    AnchorStatus,
    CitationHandle,
    CitationScope,
    CitationStatus,
    Claim,
    ClaimAlignment,
    ClaimBasis,
    ClaimSet,
    ClaimTopic,
    Decidability,
    EvidentialRelation,
    ExtractorInfo,
    FlagKind,
    SemanticRelation,
    SourceDocument,
    SourceType,
    triage_claims,
)


def _claim(
    cid: str,
    text: str = "X drives Y.",
    *,
    about: ClaimTopic = ClaimTopic.DOMAIN,
    basis: ClaimBasis = ClaimBasis.OBSERVATION,
    citation_status: CitationStatus = CitationStatus.UNCITED,
    citations=None,
) -> Claim:
    return Claim(
        id=cid, claim_text=text, anchor_status=AnchorStatus.EXACT,
        about=about, basis=basis, citation_status=citation_status, citations=citations,
    )


def _set(*claims: Claim, source_id: str = "r.md") -> ClaimSet:
    return ClaimSet(
        source=SourceDocument(id=source_id, source_type=SourceType.MARKDOWN_REPORT),
        extractor=ExtractorInfo(name="x"), claims=list(claims),
    )


def _kinds(flags) -> list[str]:
    return [f.kind for f in flags]


def test_uncited_domain_finding_is_flagged_unsupported() -> None:
    flags = triage_claims(_set(_claim("c1", basis=ClaimBasis.OBSERVATION)))
    assert _kinds(flags) == [FlagKind.UNSUPPORTED_CLAIM.value]
    assert flags[0].decidability == Decidability.AGENT.value
    assert flags[0].claim_ids == ["c1"]
    assert flags[0].evidence  # the claim text is kept for the reviewer


def test_secondary_source_uncited_domain_claim_is_also_unsupported() -> None:
    """A claim attributed to a source but carrying no citation stands on nothing."""
    flags = triage_claims(_set(_claim("c1", basis=ClaimBasis.SECONDARY_SOURCE)))
    assert _kinds(flags) == [FlagKind.UNSUPPORTED_CLAIM.value]


def test_background_knowledge_is_not_flagged_unsupported() -> None:
    """Textbook background is expected to be uncited; it is not an unsupported finding."""
    flags = triage_claims(_set(_claim("c1", basis=ClaimBasis.BACKGROUND_KNOWLEDGE)))
    assert flags == []


def test_claim_about_a_work_is_not_flagged_unsupported() -> None:
    """A claim about a work (about=WORK), not the domain, is out of scope for this rule."""
    work = Claim(id="c1", claim_text="Smith 2020 reports X.", anchor_status=AnchorStatus.EXACT,
                 about=ClaimTopic.WORK, citation_status=CitationStatus.UNCITED)  # WORK carries no basis
    assert triage_claims(_set(work)) == []


def test_cited_claim_is_not_flagged_unsupported() -> None:
    cited = _claim(
        "c1", citation_status=CitationStatus.CITED,
        citations=[CitationHandle(marker="[1]", reference_id="PMID:1", scope=CitationScope.SENTENCE)],
    )
    assert triage_claims(_set(cited)) == []


def test_cited_but_unresolved_marker_is_flagged() -> None:
    cited = _claim(
        "c1", citation_status=CitationStatus.CITED,
        citations=[CitationHandle(marker="[1]", scope=CitationScope.SENTENCE)],  # no reference_id
    )
    flags = triage_claims(_set(cited))
    assert _kinds(flags) == [FlagKind.UNRESOLVED_CITATION.value]
    assert flags[0].decidability == Decidability.AGENT.value


def test_consensus_cue_while_uncited_is_flagged_for_humans() -> None:
    flags = triage_claims(_set(_claim(
        "c1", "It is well known that acne is multifactorial.",
        basis=ClaimBasis.BACKGROUND_KNOWLEDGE,  # so only the consensus rule fires
    )))
    assert _kinds(flags) == [FlagKind.CONSENSUS_OVERSTATED.value]
    assert flags[0].decidability == Decidability.HUMAN.value


def test_consensus_cue_while_properly_cited_is_not_flagged() -> None:
    """A settled-sounding claim that is actually cited (and not contradicted) is fine."""
    cited = _claim(
        "c1", "There is consensus that smoking harms health.",
        basis=ClaimBasis.BACKGROUND_KNOWLEDGE, citation_status=CitationStatus.CITED,
        citations=[CitationHandle(marker="[1]", reference_id="PMID:1", scope=CitationScope.SENTENCE)],
    )
    assert triage_claims(_set(cited)) == []


def test_a_caller_may_supply_its_own_cues() -> None:
    flags = triage_claims(
        _set(_claim("c1", "Everyone agrees this works.", basis=ClaimBasis.BACKGROUND_KNOWLEDGE)),
        cues=("everyone agrees",),
    )
    assert _kinds(flags) == [FlagKind.CONSENSUS_OVERSTATED.value]


def _pair(pid: str, a: str, b: str, relation: SemanticRelation) -> AlignedClaimPair:
    return AlignedClaimPair(
        id=pid, claim_a_id=a, claim_b_id=b, claim_a_text="A.", claim_b_text="not A.",
        semantic_relation=relation, evidential_relation=EvidentialRelation.BOTH_SECONDARY,
    )


def _alignment(*pairs: AlignedClaimPair, within: bool, source_id: str = "r.md") -> ClaimAlignment:
    return ClaimAlignment(
        source_a=AlignedSource(id=source_id, source_type=AlignedSourceType.MARKDOWN_REPORT),
        source_b=AlignedSource(id=source_id if within else "b.md",
                               source_type=AlignedSourceType.MARKDOWN_REPORT),
        aligner=AlignerInfo(name="llm-pairwise"), within_source=within, pairs=list(pairs),
    )


def test_within_source_contradiction_is_a_self_contradiction() -> None:
    claim_set = _set(_claim("c1", basis=ClaimBasis.BACKGROUND_KNOWLEDGE),
                     _claim("c2", basis=ClaimBasis.BACKGROUND_KNOWLEDGE))
    alignment = _alignment(_pair("p1", "c1", "c2", SemanticRelation.CONTRADICTORY), within=True)
    flags = triage_claims(claim_set, alignment)
    assert _kinds(flags) == [FlagKind.SELF_CONTRADICTION.value]
    assert flags[0].decidability == Decidability.HUMAN.value
    assert flags[0].pair_id == "p1"
    assert flags[0].claim_ids == ["c1", "c2"]


def test_cross_source_contradiction_is_a_contradiction() -> None:
    claim_set = _set(_claim("c1", basis=ClaimBasis.BACKGROUND_KNOWLEDGE))
    alignment = _alignment(_pair("p1", "c1", "x1", SemanticRelation.CONTRADICTORY), within=False)
    flags = triage_claims(claim_set, alignment)
    assert _kinds(flags) == [FlagKind.CONTRADICTION.value]


def test_non_contradictory_pairs_raise_no_flag() -> None:
    claim_set = _set(_claim("c1", basis=ClaimBasis.BACKGROUND_KNOWLEDGE))
    alignment = _alignment(_pair("p1", "c1", "x1", SemanticRelation.IDENTICAL), within=False)
    assert triage_claims(claim_set, alignment) == []


def test_consensus_cue_on_a_contradicted_claim_is_flagged_even_when_cited() -> None:
    """A cited claim that sounds settled but is contradicted is consensus-overstated."""
    cited = _claim(
        "c1", "There is universal consensus acne is bacterial.",
        basis=ClaimBasis.BACKGROUND_KNOWLEDGE, citation_status=CitationStatus.CITED,
        citations=[CitationHandle(marker="[1]", reference_id="PMID:1", scope=CitationScope.SENTENCE)],
    )
    claim_set = _set(cited)
    alignment = _alignment(_pair("p1", "c1", "x1", SemanticRelation.CONTRADICTORY), within=False)
    kinds = _kinds(triage_claims(claim_set, alignment))
    assert FlagKind.CONSENSUS_OVERSTATED.value in kinds
    assert FlagKind.CONTRADICTION.value in kinds


def test_flag_order_is_claim_level_first_then_contradictions() -> None:
    claim_set = _set(
        _claim("c1", "It is well known X.", basis=ClaimBasis.OBSERVATION),  # unsupported + consensus
        _claim("c2", basis=ClaimBasis.BACKGROUND_KNOWLEDGE),
    )
    alignment = _alignment(_pair("p1", "c1", "c2", SemanticRelation.CONTRADICTORY), within=True)
    kinds = _kinds(triage_claims(claim_set, alignment))
    assert kinds == [
        FlagKind.UNSUPPORTED_CLAIM.value,
        FlagKind.CONSENSUS_OVERSTATED.value,
        FlagKind.SELF_CONTRADICTION.value,
    ]
