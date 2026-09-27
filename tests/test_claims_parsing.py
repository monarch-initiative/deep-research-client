"""Anchoring quotes and parsing extractor replies into claims (issue #43).

These are pure functions of text, so they are tested on a report fixture and
a reply in the shape the extraction prompt asks for, with no model call.
"""

import json
from pathlib import Path

import pytest

from deep_research_client.claims.anchoring import locate_quote
from deep_research_client.claims.models import AnchorStatus
from deep_research_client.claims.parsing import (
    TextUnit,
    UnreadableReplyError,
    citation_window,
    claims_from_reply,
)

REPORT = (Path(__file__).parent / "input" / "claims" / "marfan_report.md").read_text(
    encoding="utf-8"
)


@pytest.mark.parametrize(
    "quote,expected_status,expected_text",
    [
        (
            "Marfan syndrome is an autosomal dominant disorder",
            AnchorStatus.EXACT,
            "Marfan syndrome is an autosomal dominant disorder",
        ),
        (
            # Emphasis and the citation marker dropped, as models often do.
            "caused by pathogenic variants in FBN1.",
            AnchorStatus.NORMALIZED,
            "caused by pathogenic variants in **FBN1** [1].",
        ),
        (
            # Link target dropped, label kept.
            "see the 2010 Ghent criteria",
            AnchorStatus.NORMALIZED,
            "see [the 2010 Ghent criteria](https://doi.org/10.1136/jmg.2009.072785)",
        ),
        (
            # Case and reflowed whitespace. The span runs on over the citation
            # marker and full stop that follow, keeping the claim's citation in it.
            "beta-blockers   may slow\naortic root growth",
            AnchorStatus.NORMALIZED,
            "Beta-blockers may slow aortic root growth [3].",
        ),
    ],
    ids=["verbatim", "markup-and-marker", "link", "case-and-whitespace"],
)
def test_a_quote_is_found_and_its_span_reads_as_the_source(quote, expected_status, expected_text):
    """Whatever matched, the span covers the source's own characters."""
    span, status = locate_quote(quote, REPORT)

    assert status == expected_status
    assert span is not None
    assert span.text == expected_text
    assert REPORT[span.start:span.end] == span.text


@pytest.mark.parametrize(
    "source,quote,expected_text",
    [
        (
            "FBN1 variants [1](https://pubmed.ncbi.nlm.nih.gov/1852208/) occur in most patients.",
            "FBN1 variants occur in most patients.",
            "FBN1 variants [1](https://pubmed.ncbi.nlm.nih.gov/1852208/) occur in most patients.",
        ),
        (
            "It is a [fibrillinopathy](https://en.wikipedia.org/wiki/Fibrillin_(protein)) of the aorta.",
            "It is a fibrillinopathy of the aorta",
            "It is a [fibrillinopathy](https://en.wikipedia.org/wiki/Fibrillin_(protein)) of the aorta.",
        ),
    ],
    ids=["linked-numeric-marker", "parenthesised-link-target"],
)
def test_a_link_target_is_ignored_wherever_it_sits(source, quote, expected_text):
    """A quote that leaves out a link's target still anchors, over the source's characters."""
    span, status = locate_quote(quote, source)

    assert status == AnchorStatus.NORMALIZED
    assert span is not None
    assert span.text == expected_text == source[span.start:span.end]


def test_a_quote_the_source_never_says_is_unanchored():
    """A plausible paraphrase is not evidence of anything."""
    span, status = locate_quote("FBN1 mutations are the sole cause of Marfan syndrome", REPORT)

    assert span is None
    assert status == AnchorStatus.UNANCHORED


def test_a_quote_outside_the_unit_is_not_found_inside_it():
    """Search is confined to the unit, so a claim cannot borrow another section's text."""
    management = REPORT.index("## Management")
    span, status = locate_quote("Marfan syndrome is an autosomal dominant disorder", REPORT, management)

    assert span is None
    assert status == AnchorStatus.UNANCHORED


def _genetics_unit() -> TextUnit:
    """The Genetics section of the fixture report, with its bibliography."""
    start = REPORT.index("Marfan syndrome is an autosomal")
    end = REPORT.index("## Clinical features")
    return TextUnit(
        text=REPORT, start=start, end=end, section="Marfan syndrome > Genetics",
        bibliography={
            1: "Dietz HC et al. Nature. 1991. https://doi.org/10.1038/352337a0",
            2: "Loeys BL et al. Nat Genet. 2005. PMID:15731757",
        },
    )


def _reply(*claims: dict) -> str:
    """A reply in the shape the prompt asks for, wrapped as models often wrap it."""
    return "Here are the claims:\n```json\n" + json.dumps({"claims": list(claims)}) + "\n```"


def test_a_reply_becomes_anchored_structured_claims():
    """Structure, provenance, negation and citations all survive parsing."""
    reply = _reply(
        {
            "claim": "Pathogenic variants in FBN1 cause Marfan syndrome.",
            "quote": "caused by pathogenic variants in FBN1",
            "subject": "pathogenic variants in FBN1", "predicate": "causes",
            "object": "Marfan syndrome", "negated": False,
            "entities": ["FBN1", "Marfan syndrome"], "citations": ["[1]"],
        },
        {
            "claim": "Variants in TGFBR2 do not cause Marfan syndrome.",
            "quote": "It is not caused by variants in TGFBR2",
            "subject": "variants in TGFBR2", "predicate": "causes",
            "object": "Marfan syndrome", "negated": True, "citations": [],
        },
    )

    first, second = claims_from_reply(reply, _genetics_unit())

    assert first.anchor_status == AnchorStatus.NORMALIZED
    assert REPORT[first.source_span.start:first.source_span.end] == first.source_span.text
    assert first.section == "Marfan syndrome > Genetics"
    assert (first.subject.label, first.predicate.label, first.object.label) == (
        "pathogenic variants in FBN1", "causes", "Marfan syndrome",
    )
    assert first.negated is False
    assert [(c.marker, c.reference_id) for c in first.citations] == [
        ("[1]", "DOI:10.1038/352337a0"),
    ]
    assert second.negated is True
    assert second.anchor_status == AnchorStatus.NORMALIZED
    assert second.citations is None


def test_a_citation_the_source_does_not_carry_is_dropped():
    """A marker is a fact about the text; the model's word for it is not enough."""
    reply = _reply({
        "claim": "FBN1 variants cause Marfan syndrome.",
        "quote": "caused by pathogenic variants in FBN1",
        "citations": ["[1]", "[5]", "PMID:99999999"],
    })

    (claim,) = claims_from_reply(reply, _genetics_unit())

    assert [c.marker for c in claim.citations] == ["[1]"]


def test_an_unfound_quote_keeps_the_claim_without_a_span():
    """The gap stays visible rather than the claim disappearing."""
    reply = _reply({"claim": "FBN1 is the only cause.", "quote": "FBN1 is the only cause"})

    (claim,) = claims_from_reply(reply, _genetics_unit())

    assert claim.anchor_status == AnchorStatus.UNANCHORED
    assert claim.source_span is None


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        '{"answer": []}',
        '{"claims": "none"}',
        '{"claims": [{"claim": "FBN1 variants cause Marfan syndrome.", '
        '"quote": "caused by pathogenic variants"}, {"claim": "Marfan syn',
    ],
    ids=["prose", "wrong-key", "wrong-type", "cut-off"],
)
def test_an_unreadable_reply_raises_rather_than_reading_as_no_claims(reply):
    """A reply cut off mid-list must not look like a section with no claims."""
    with pytest.raises(UnreadableReplyError, match="Genetics"):
        claims_from_reply(reply, _genetics_unit())


def test_an_empty_claims_list_is_a_unit_with_no_claims():
    """Only an explicit empty list means "no claims"."""
    assert claims_from_reply('{"claims": []}', _genetics_unit()) == []


def test_malformed_entries_are_skipped_and_the_rest_kept():
    """One bad entry does not cost the unit its other claims."""
    reply = _reply(
        "a bare string",
        {"quote": "no claim text"},
        {"claim": "FBN1 variants cause Marfan syndrome.", "quote": "caused by pathogenic variants",
         "negated": "no"},
    )

    (claim,) = claims_from_reply(reply, _genetics_unit())

    assert claim.claim_text == "FBN1 variants cause Marfan syndrome."
    assert claim.negated is None, "only a boolean is taken as a negation flag"


def test_a_curie_entity_is_taken_as_grounded():
    """An identifier in the text is grounding the source already did."""
    reply = _reply({
        "claim": "Marfan syndrome is autosomal dominant.",
        "quote": "Marfan syndrome is an autosomal dominant disorder",
        "entities": ["MONDO:0007947", {"label": "FBN1", "id": "HGNC:3603"}, "Marfan syndrome"],
    })

    (claim,) = claims_from_reply(reply, _genetics_unit())

    assert [(e.label, e.id) for e in claim.entities] == [
        ("MONDO:0007947", "MONDO:0007947"), ("FBN1", "HGNC:3603"), ("Marfan syndrome", None),
    ]


def test_a_marker_from_another_sentence_in_the_section_is_not_attached():
    """[2] belongs to the TGFBR2 sentence, not to the FBN1 one before it."""
    reply = _reply({
        "claim": "Pathogenic variants in FBN1 cause Marfan syndrome.",
        "quote": "caused by pathogenic variants in FBN1",
        "citations": ["[1]", "[2]"],
    })

    (claim,) = claims_from_reply(reply, _genetics_unit())

    assert [c.marker for c in claim.citations] == ["[1]"]


@pytest.mark.parametrize(
    "text,quote,window",
    [
        ("A is B [1]. C is D [2].", "C is D", " C is D [2]."),
        ("A is B.[1] C is D.[2]", "A is B", "A is B.[1]"),
        ("A is B.[1] C is D.[2]", "C is D", " C is D.[2]"),
        ("E is F. [3] G is H.", "E is F", "E is F. [3]"),
        ("E is F. [3] G is H.", "G is H", " G is H."),
        ("E is F.[3](https://x.org/3) G is H.", "E is F", "E is F.[3](https://x.org/3)"),
        (
            "Aortic dilation, e.g. at the root, is common [2]. Next.",
            "Aortic dilation",
            "Aortic dilation, e.g. at the root, is common [2].",
        ),
        (
            "Risk is higher vs. controls in Smith et al. cohorts [4]. Next.",
            "Risk is higher",
            "Risk is higher vs. controls in Smith et al. cohorts [4].",
        ),
        ("See Fig. 2 for dilation [5]. Next.", "See Fig", "See Fig. 2 for dilation [5]."),
        (
            "A is B and wraps\nonto a second line [1]. Next.",
            "A is B",
            "A is B and wraps\nonto a second line [1].",
        ),
        ("- A is B\n- C is D [2]\n", "A is B", "- A is B\n"),
        # The item number's own "." reads as a sentence end; harmless, as it
        # carries no marker.
        ("1. A is B\n2. C is D [2]\n", "A is B", " A is B\n"),
        ("A is B\n\nC is D [2].", "A is B", "A is B\n"),
    ],
    ids=[
        "marker-before-stop", "marker-after-stop", "next-after-stop",
        "spaced-marker-after-stop", "next-after-spaced-marker", "linked-marker-after-stop",
        "e.g.", "vs.-and-et-al.", "Fig.",
        "hard-wrapped-line", "bulleted-item", "numbered-item", "paragraph-break",
    ],
)
def test_the_citation_window_is_the_claims_own_sentence(text, quote, window):
    """A marker belongs to the sentence it closes, wherever the full stop sits."""
    unit = TextUnit(text=text, start=0, end=len(text))
    span, _ = locate_quote(quote, text)
    assert span is not None

    assert citation_window(unit, span) == window


def test_an_unanchored_claim_carries_no_citations():
    """With no sentence to check a marker against, none is taken on trust."""
    reply = _reply({
        "claim": "FBN1 is the only cause.", "quote": "FBN1 is the only cause", "citations": ["[1]"],
    })

    (claim,) = claims_from_reply(reply, _genetics_unit())

    assert claim.anchor_status == AnchorStatus.UNANCHORED
    assert claim.citations is None


def test_a_table_row_is_its_own_citation_window():
    """A marker on one row does not attach to a claim from the next."""
    text = "| Feature | Source |\n|---|---|\n| Aortic dilation | [1] |\n| Ectopia lentis | [2] |\n"
    unit = TextUnit(
        text=text, start=0, end=len(text),
        bibliography={1: "PMID:20591885", 2: "PMID:8166794"},
    )
    reply = _reply({
        "claim": "Ectopia lentis is a feature.", "quote": "| Ectopia lentis |",
        "citations": ["[1]", "[2]"],
    })

    (claim,) = claims_from_reply(reply, unit)

    assert [(c.marker, c.reference_id) for c in claim.citations] == [("[2]", "PMID:8166794")]
