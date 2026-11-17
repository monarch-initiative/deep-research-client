"""Candidate generation: which claim pairs are worth a model's time.

Pure functions of the two claim lists (and, for embeddings, a caller-supplied
function), so no model is involved.
"""

import pytest

from deep_research_client.claims import (
    AnchorStatus,
    CitationHandle,
    CitationScope,
    CitationStatus,
    Claim,
    EntityMention,
    candidate_pairs,
    candidate_pairs_within,
)
from deep_research_client.claims.candidates import jaccard


def _claim(cid, text, *, entity=None, ref=None) -> Claim:
    return Claim(
        id=cid, claim_text=text, anchor_status=AnchorStatus.EXACT,
        citation_status=CitationStatus.CITED if ref else CitationStatus.UNCITED,
        entities=[EntityMention(label=entity, id=entity)] if entity else None,
        citations=[CitationHandle(marker=ref, reference_id=ref, scope=CitationScope.SENTENCE)] if ref else None,
    )


def test_a_shared_entity_proposes_a_pair() -> None:
    a = [_claim("a1", "A sentence with no words in common.", entity="HGNC:3603")]
    b = [_claim("b1", "A wholly unlike sentence, truly.", entity="HGNC:3603")]
    pairs = candidate_pairs(a, b)
    assert len(pairs) == 1
    assert pairs[0].signal_list[0].value == "ENTITY"


def test_a_shared_resolved_citation_proposes_a_pair() -> None:
    a = [_claim("a1", "One distinct sentence here.", ref="PMID:7913883")]
    b = [_claim("b1", "Another unrelated sentence there.", ref="PMID:7913883")]
    pairs = candidate_pairs(a, b)
    assert [s.value for s in pairs[0].signal_list] == ["CITATION"]


def test_lexical_overlap_proposes_a_pair_and_the_threshold_gates_it() -> None:
    a = [_claim("a1", "FBN1 variants cause Marfan syndrome in humans.")]
    b = [_claim("b1", "Marfan syndrome in humans is caused by FBN1 variants.")]
    assert candidate_pairs(a, b, min_overlap=0.3)  # clear overlap
    assert not candidate_pairs(a, b, min_overlap=0.99)  # threshold too high


def test_an_unrelated_pair_is_not_proposed() -> None:
    a = [_claim("a1", "FBN1 variants cause Marfan syndrome.", entity="HGNC:3603")]
    b = [_claim("b1", "The mitochondrion is the powerhouse of the cell.", entity="GO:0005739")]
    assert candidate_pairs(a, b) == []


def test_several_signals_accumulate_on_one_pair() -> None:
    a = [_claim("a1", "FBN1 variants cause Marfan syndrome.", entity="HGNC:3603", ref="PMID:1")]
    b = [_claim("b1", "Marfan syndrome is caused by FBN1 variants.", entity="HGNC:3603", ref="PMID:1")]
    pairs = candidate_pairs(a, b)
    assert [s.value for s in pairs[0].signal_list] == ["CITATION", "ENTITY", "LEXICAL"]


def test_pairs_come_back_in_a_to_b_order() -> None:
    a = [_claim("a1", "FBN1 and Marfan.", entity="HGNC:3603"),
         _claim("a2", "Fibrillin and Marfan.", entity="HGNC:3603")]
    b = [_claim("b1", "Marfan and FBN1.", entity="HGNC:3603"),
         _claim("b2", "Marfan and fibrillin.", entity="HGNC:3603")]
    order = [(p.claim_a.id, p.claim_b.id) for p in candidate_pairs(a, b, min_overlap=0.99)]
    assert order == [("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2")]


def test_a_caller_supplied_embedder_proposes_a_paraphrase_with_no_shared_token() -> None:
    """An injected embedder is the optional hook; it is a real function, not a mock."""
    a = [_claim("a1", "Alpha.")]
    b = [_claim("b1", "Omega.")]  # no shared entity, citation or content word

    # A deterministic stand-in embedder the caller controls: identical vectors,
    # so cosine is 1 and the pair clears the threshold.
    def embedder(texts):
        return [[1.0, 0.0] for _ in texts]

    assert candidate_pairs(a, b) == []  # nothing without the embedder
    pairs = candidate_pairs(a, b, embedder=embedder)
    assert [s.value for s in pairs[0].signal_list] == ["EMBEDDING"]


def test_jaccard_is_intersection_over_union() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b", "c"}, {"a"}) == 1 / 3
    assert jaccard(set(), {"a"}) == 0.0


def test_empty_inputs_produce_no_candidates() -> None:
    assert candidate_pairs([], []) == []
    assert candidate_pairs([_claim("a1", "A sentence.")], []) == []


def test_within_generates_unordered_distinct_pairs() -> None:
    """A set against itself: each pair once, i before j, never a claim with itself."""
    claims = [
        _claim("c1", "FBN1 variants cause Marfan syndrome.", entity="HGNC:3603"),
        _claim("c2", "Marfan syndrome is caused by FBN1 variants.", entity="HGNC:3603"),
        _claim("c3", "Marfan syndrome arises from FBN1 variants.", entity="HGNC:3603"),
    ]
    ids = [(p.claim_a.id, p.claim_b.id) for p in candidate_pairs_within(claims)]
    assert ids == [("c1", "c2"), ("c1", "c3"), ("c2", "c3")]


def test_within_embeds_each_text_once() -> None:
    """Within one set the two sides are the same list, so the embedder runs once."""
    claims = [_claim("c1", "alpha beta"), _claim("c2", "gamma delta")]
    calls = []

    def counting(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    pairs = candidate_pairs_within(claims, min_overlap=0.99, embedder=counting, embed_threshold=0.5)
    assert len(calls) == 1  # embedded once, not once per side
    assert [(p.claim_a.id, p.claim_b.id) for p in pairs] == [("c1", "c2")]


def test_within_pairs_nothing_when_claims_do_not_overlap() -> None:
    claims = [
        _claim("c1", "FBN1 variants cause Marfan syndrome.", entity="HGNC:3603"),
        _claim("c2", "The mitochondrion is the powerhouse of the cell.", entity="GO:0005739"),
    ]
    assert candidate_pairs_within(claims) == []


def test_an_embedder_returning_the_wrong_count_is_refused() -> None:
    """A batching bug that drops or adds a vector should fail loudly, not lose pairs."""
    a = [_claim("a1", "Alpha.")]
    b = [_claim("b1", "Omega.")]

    def wrong(texts):
        return [[1.0, 0.0] for _ in range(len(texts) + 1)]  # one too many

    with pytest.raises(ValueError, match="one per text"):
        candidate_pairs(a, b, embedder=wrong)


async def test_aligning_empty_sets_needs_no_model() -> None:
    """No candidates means no classification, so an empty alignment needs no client."""
    from deep_research_client.claims import (
        ClaimSet,
        ExtractorInfo,
        SourceDocument,
        SourceType,
        aalign_claim_sets,
    )

    set_a = ClaimSet(source=SourceDocument(id="a.md", source_type=SourceType.MARKDOWN_REPORT),
                     extractor=ExtractorInfo(name="x"))
    set_b = ClaimSet(source=SourceDocument(id="b.yaml", source_type=SourceType.STRUCTURED_DOCUMENT),
                     extractor=ExtractorInfo(name="y"))
    alignment = await aalign_claim_sets(set_a, set_b, llm_client=None, model="unused")
    assert alignment.pair_list == []
