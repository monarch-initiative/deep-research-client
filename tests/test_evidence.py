"""Tests for evidence files and the local check of a report against them."""

import pytest

from deep_research_client.validation.evidence import (
    EvidenceFile,
    check_against_evidence,
    parse_evidence_file,
)

HODGES = """---
key: hodges-1971-clinical-manifestations
identifiers:
  - PMID:5568449
  - DOI:10.1093/ajcn/24.4.432
title: Clinical manifestations of ascorbic acid deficiency in man
source_url: https://pubmed.ncbi.nlm.nih.gov/5568449/
content: abstract
---
Five healthy men were fed a diet devoid of ascorbic acid. The first signs of
scurvy appeared after 29 days, when the body pool had fallen below 300 mg.
"""


@pytest.fixture
def hodges() -> EvidenceFile:
    """A well-formed evidence file."""
    evidence = parse_evidence_file("hodges-1971-clinical-manifestations.md", HODGES)
    assert evidence is not None
    return evidence


def test_parse_evidence_file_reads_frontmatter_and_body(hodges):
    """Fields come from the frontmatter; the body is everything after it."""
    assert hodges.key == "hodges-1971-clinical-manifestations"
    assert hodges.identifiers == ["PMID:5568449", "DOI:10.1093/ajcn/24.4.432"]
    assert hodges.source_url == "https://pubmed.ncbi.nlm.nih.gov/5568449/"
    assert hodges.content == "abstract"
    assert hodges.text.startswith("Five healthy men")
    assert hodges.citation == "PMID:5568449"


@pytest.mark.parametrize(
    "text",
    [
        "No frontmatter at all.",
        "---\nkey: unterminated\nbody with no closing delimiter\n",
        "---\n- a list, not a mapping\n---\nbody\n",
    ],
)
def test_parse_evidence_file_rejects_files_that_are_not_evidence(text):
    """Notes and malformed files are skipped, not half-read."""
    assert parse_evidence_file("notes.md", text) is None


@pytest.mark.parametrize(
    "frontmatter,title",
    [
        # Each of these is invalid YAML, and each came from a real retriever run.
        ("title: Lind and scurvy: 1747 to 1795\nyear: 2005 (J R Soc Med, Nov)", "Lind and scurvy: 1747 to 1795"),
        ("title: Putrid gums and 'Dead Men's Cloaths': James Lind aboard the Salisbury",
         "Putrid gums and 'Dead Men's Cloaths': James Lind aboard the Salisbury"),
        # PubMed brackets a translated title.
        ("title: [Axel Holst and Theodor Frolich--pioneers in the combat of scurvy]",
         "[Axel Holst and Theodor Frolich--pioneers in the combat of scurvy]"),
        ('title: "Clinical manifestations: a quoted title"', "Clinical manifestations: a quoted title"),
    ],
)
def test_free_text_fields_that_break_yaml_do_not_hide_the_source(frontmatter, title):
    """One retriever's unquoted colon must not drop a source's identifiers from the check."""
    text = f"---\nkey: k\n{frontmatter}\nidentifiers:\n  - PMID:16260808\nurl: https://pmc.example/x\n---\nBody.\n"

    evidence = parse_evidence_file("k.md", text)

    assert evidence is not None
    assert (evidence.title, evidence.identifiers, evidence.text) == (title, ["PMID:16260808"], "Body.\n")


@pytest.mark.parametrize(
    "identifiers",
    ["identifiers: [PMID:5568449, 'DOI:10.1093/ajcn/24.4.432']",
     "identifiers:\n  - PMID: 5568449\n  - doi:10.1093/ajcn/24.4.432"],
)
def test_identifier_lists_are_read_in_either_style(identifiers):
    """Inline and block lists, with or without a space after the prefix."""
    evidence = parse_evidence_file("h.md", f"---\n{identifiers}\n---\nBody.\n")
    assert evidence is not None
    assert evidence.identifiers == ["PMID:5568449", "DOI:10.1093/ajcn/24.4.432"]


def test_a_source_without_identifiers_is_cited_by_its_url():
    """Grey literature has no PMID or DOI, only the page it came from."""
    evidence = parse_evidence_file(
        "who-2021-scurvy.md",
        "---\nidentifiers: []\nsource_url: https://www.who.int/scurvy\n---\nText.\n",
    )
    assert evidence is not None
    assert (evidence.key, evidence.citation) == ("who-2021-scurvy", "https://www.who.int/scurvy")


@pytest.mark.parametrize(
    "report,anchored,reason",
    [
        # Verbatim.
        ('Onset was quick: "The first signs of scurvy appeared after 29 days" (PMID:5568449).', 1, None),
        # The same passage reflowed across the source's line break, cited by DOI.
        ('"signs of scurvy appeared after 29 days, when the body pool had fallen" '
         '(DOI:10.1093/ajcn/24.4.432).', 1, None),
        # A quote the source never says.
        ('"the first signs of scurvy appeared after three days" (PMID:5568449).', 0,
         "not found in the cited evidence"),
        # A real passage attributed to a source with no evidence file.
        ('"The first signs of scurvy appeared after 29 days" (PMID:12345678).', 0,
         "no evidence file carries this identifier"),
    ],
)
def test_quotes_are_found_only_in_the_evidence_for_the_identifier_they_cite(hodges, report, anchored, reason):
    """A quote is checked against its own source's text, not any source's."""
    check = check_against_evidence(report, [hodges])

    assert (check.quotes_checked, check.quotes_anchored) == (1, anchored)
    assert [quote.reason for quote in check.unanchored_quotes] == ([reason] if reason else [])


def test_cited_identifiers_without_evidence_are_listed(hodges):
    """An identifier with no evidence file was recalled, not retrieved."""
    report = "Deficiency causes scurvy (PMID:5568449; PMID:99999998).\n\n## References\n- DOI:10.1234/abc.5\n"

    check = check_against_evidence(report, [hodges])

    assert check.cited_identifiers == 3
    assert check.cited_without_evidence == ["DOI:10.1234/abc.5", "PMID:99999998"]
    assert check.evidence_files == 1


def test_a_report_with_no_citations_checks_nothing():
    """Nothing to check is not the same as everything checked out, so counts stay zero."""
    check = check_against_evidence("An essay with no sources.", [])
    assert check.model_dump() == {
        "evidence_files": 0, "cited_identifiers": 0, "cited_without_evidence": [],
        "quotes_checked": 0, "quotes_anchored": 0, "unanchored_quotes": [],
    }
