"""Tests for ontology term validation.

Offline tests exercise the real ``linkml-term-validator`` code paths without
network access, by pre-seeding its on-disk label cache and running the validator
in offline mode, where it never builds an OAK adapter and answers exclusively
from that cache. Tests that need a live ontology service to assert that an
identifier genuinely does not exist, or that a term has been obsoleted, are
marked ``integration``.

The running example throughout is the failure this feature exists for:
``NCIT:C16814`` is a real NCIT term that resolves cleanly, and it denotes
Malaysia rather than the echocardiography a report claimed for it.
"""

import csv
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from deep_research_client.cli import app
from deep_research_client.models import ResearchResult
from deep_research_client.processing import ResultFormatter
from deep_research_client.validation import (
    FoundTerm,
    LabelAgreement,
    TermCheck,
    TermStatus,
    TermValidationReport,
    TermValidator,
    compare_labels,
    extract_terms,
    find_term_ids,
    label_similarity,
    strip_validation_section,
)

# Labels as the ontologies actually give them, used to seed the offline cache.
CACHED_LABELS = {
    "NCIT:C16814": "Malaysia",
    "HP:0001250": "Seizure",
    "HP:0001083": "Ectopia lentis",
    "HP:0002616": "Aortic root aneurysm",
    "MONDO:0007947": "Marfan syndrome",
    "GO:0008543": "fibroblast growth factor receptor signaling pathway",
}


@pytest.fixture
def label_cache(tmp_path: Path) -> Path:
    """A pre-seeded label cache, in the layout the validator reads.

    One CSV per prefix, which is what ``linkml-term-validator`` writes after a
    live run, so an offline test reads exactly what an online one would have
    left behind.
    """
    cache_dir = tmp_path / "terms_cache"
    by_prefix: dict[str, dict[str, str]] = {}
    for curie, label in CACHED_LABELS.items():
        by_prefix.setdefault(curie.split(":", 1)[0], {})[curie] = label

    for prefix, entries in by_prefix.items():
        prefix_dir = cache_dir / prefix.lower()
        prefix_dir.mkdir(parents=True, exist_ok=True)
        with open(prefix_dir / "terms.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, ["curie", "label", "retrieved_at"])
            writer.writeheader()
            for curie in sorted(entries):
                writer.writerow(
                    {
                        "curie": curie,
                        "label": entries[curie],
                        "retrieved_at": "2026-01-01T00:00:00",
                    }
                )
    return cache_dir


@pytest.fixture
def offline_validator(label_cache: Path) -> TermValidator:
    """A validator that answers only from the seeded cache."""
    return TermValidator(cache_dir=label_cache, offline=True, cache_labels=False)


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Marfan syndrome (MONDO:0007947) is inherited.", ["MONDO:0007947"]),
        ("See HP:0001250 and NCIT:C16814.", ["HP:0001250", "NCIT:C16814"]),
        ("MESH:D008382 is cited too.", ["MESH:D008382"]),
        ("http://purl.obolibrary.org/obo/UBERON_0002101", ["UBERON:0002101"]),
        ("https://bioregistry.io/HP:0001250", ["HP:0001250"]),
        ("Taxon NCBITaxon:9606 was studied.", ["NCBITaxon:9606"]),
    ],
)
def test_curies_are_extracted(text: str, expected: list[str]) -> None:
    assert [t.term_id for t in find_term_ids(text)] == expected


@pytest.mark.parametrize(
    "text",
    [
        # Bibliographic namespaces belong to reference validation; extracting
        # them here would report every citation twice.
        "Cited as PMID:7913883.",
        "Cited as DOI:10.1038/ng1234.",
        "Cited as PMC11000121.",
        "Deposited under GEO:GSE68086.",
        # Shapes that look like CURIEs and are not.
        "Recorded at 2024-01-02T10:30:00 in the log.",
        "The ratio was 3:14 across sites.",
        "Note: 2019 was the peak year.",
        "See Table 2 for details.",
        "Read https://example.org/page for more.",
    ],
)
def test_non_terms_are_not_extracted(text: str) -> None:
    assert find_term_ids(text) == []


def test_mentions_are_counted() -> None:
    terms = find_term_ids("HP:0001250 recurs; see HP:0001250 again, and HP:0001250.")

    assert [(t.term_id, t.count) for t in terms] == [("HP:0001250", 3)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("| Echocardiography Test | NCIT:C16814 |", "Echocardiography Test"),
        ("| NCIT:C16814 | Echocardiography Test |", "Echocardiography Test"),
        ("- **Ectopia lentis** (HP:0001083)", "Ectopia lentis"),
        ("- Ectopia lentis (HP:0001083)", "Ectopia lentis"),
        ("HP:0001083: Ectopia lentis", "Ectopia lentis"),
        ("HP:0001083 - Ectopia lentis", "Ectopia lentis"),
        ("HP:0001083 (Ectopia lentis)", "Ectopia lentis"),
        ('"Ectopia lentis" (HP:0001083)', "Ectopia lentis"),
    ],
)
def test_labels_are_read_from_label_positions(text: str, expected: str) -> None:
    assert find_term_ids(text)[0].labels == (expected,)


@pytest.mark.parametrize(
    "text",
    [
        # A clause before a bracket is a clause, not a name. Reading it as one
        # manufactures a label mismatch for a correctly cited term.
        "Patients with aortic root dilation (HP:0002616) need monitoring.",
        # A parenthetical that is plainly not a name.
        "HP:0002616 (seen in 4 of 11 probands)",
        "HP:0002616 (see Table 2)",
        # A link target is not a label.
        "[HP:0002616](https://bioregistry.io/HP:0002616)",
    ],
)
def test_prose_is_not_read_as_a_label(text: str) -> None:
    assert find_term_ids(text)[0].labels == ()


@pytest.mark.parametrize(
    "text, expected",
    [
        # A parenthetical aside after a separator is not part of the name. The
        # bracketed form of the same aside was always handled; this is the
        # separator form held to the same standard.
        ("- HP:0001250 - Seizure (observed in 3 patients)", "Seizure"),
        ("| HP:0001250 - Seizure (observed) | note |", "Seizure"),
        # A comma-led clause that opens with a word no label opens with.
        ("HP:0001250: Seizure, reported in 4 of 11 probands.", "Seizure"),
        ("HP:0001250 - Seizure, seen in most patients", "Seizure"),
        # Commas are ordinary inside real labels, so these survive whole.
        ("HP:0001250: Seizure, generalized", "Seizure, generalized"),
        # Long labels are common and legitimate; no word cap may truncate them.
        (
            "GO:0000122 - negative regulation of transcription by RNA polymerase II",
            "negative regulation of transcription by RNA polymerase II",
        ),
    ],
)
def test_prose_after_a_separator_is_not_read_into_the_label(
    text: str, expected: str
) -> None:
    """Reading an aside as the name reports a correctly cited term as wrong."""
    assert find_term_ids(text)[0].labels == (expected,)


# Real MONDO/OMIM-style names, whose later comma-separated segments open with
# function words. Cutting at those would report a correctly cited term as naming
# something else - and rare-disease reports, which this tool is aimed at, are
# full of them.
LABELS_WITH_FUNCTION_WORD_SEGMENTS = [
    "microcephaly, with or without chorioretinopathy, lymphedema, "
    "or intellectual disability",
    "hypotonia, infantile, with psychomotor retardation and characteristic facies",
    "deafness, autosomal recessive, with or without vestibular dysfunction",
]


def test_the_clause_openers_are_a_subset_of_the_first_word_openers() -> None:
    """The two word lists answer different questions, and one implies the other.

    A word that can appear nowhere in a label certainly cannot appear first, so
    the narrow set belongs inside the wider one. They are maintained by hand, so
    this pins the containment rather than leaving the drift silent - and
    reversing the relation is what produced the truncation bug they were split
    apart to fix.
    """
    from deep_research_client.validation.term_extraction import (
        _NON_LABEL_OPENERS,
        _TRAILING_CLAUSE_OPENERS,
    )

    assert _TRAILING_CLAUSE_OPENERS <= _NON_LABEL_OPENERS


@pytest.mark.parametrize("label", LABELS_WITH_FUNCTION_WORD_SEGMENTS)
@pytest.mark.parametrize("layout", ["table", "separator"])
def test_a_real_name_with_function_word_segments_survives_whole(
    label: str, layout: str
) -> None:
    """Trailing-clause trimming must not truncate a name that simply has commas."""
    line = (
        f"| {label} | MONDO:0013452 |"
        if layout == "table"
        else f"MONDO:0013452 - {label}"
    )

    assert find_term_ids(line)[0].labels == (label,)


@pytest.mark.parametrize("label", LABELS_WITH_FUNCTION_WORD_SEGMENTS)
def test_a_real_name_with_function_word_segments_is_not_a_mismatch(
    label: str, label_cache: Path
) -> None:
    """End to end: the report names the term exactly, so nothing may be flagged."""
    prefix_dir = label_cache / "mondo"
    prefix_dir.mkdir(parents=True, exist_ok=True)
    with open(prefix_dir / "terms.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["curie", "label", "retrieved_at"])
        writer.writeheader()
        writer.writerow(
            {
                "curie": "MONDO:0013452",
                "label": label,
                "retrieved_at": "2026-01-01T00:00:00",
            }
        )
    validator = TermValidator(cache_dir=label_cache, offline=True, cache_labels=False)

    report = validator.validate_markdown(f"| {label} | MONDO:0013452 |")

    assert report.checked_terms[0].agreement == LabelAgreement.MATCH
    assert not report.has_problems


def test_an_aside_after_a_separator_does_not_become_a_mismatch(
    offline_validator: TermValidator,
) -> None:
    """End to end: the term is cited correctly, so nothing may be flagged."""
    report = offline_validator.validate_markdown(
        "- HP:0001250 - Seizure (observed in 3 patients)"
    )

    assert report.checked_terms[0].agreement == LabelAgreement.MATCH
    assert not report.has_problems


def test_a_label_column_that_links_out_is_still_the_label() -> None:
    """The CURIE in a label cell's href must not disqualify that cell."""
    row = "| [Seizure](https://bioregistry.io/HP:0001250) | HP:0001250 | rare |"

    assert find_term_ids(row)[0].labels == ("Seizure",)


def test_a_term_named_two_ways_keeps_both_names() -> None:
    report = "| Seizure | HP:0001250 |\n| Convulsion | HP:0001250 |\n"

    assert find_term_ids(report)[0].labels == ("Seizure", "Convulsion")


def test_citations_are_scanned_alongside_the_body() -> None:
    terms = extract_terms("Body cites HP:0001250.", ["Appendix lists MONDO:0007947"])

    assert [t.term_id for t in terms] == ["HP:0001250", "MONDO:0007947"]


# --------------------------------------------------------------------------
# Label comparison
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reported, canonical, expected",
    [
        ("Marfan syndrome", "Marfan syndrome", LabelAgreement.MATCH),
        ("marfan SYNDROME", "Marfan syndrome", LabelAgreement.MATCH),
        ("seizures", "Seizure", LabelAgreement.MATCH),
        ("Seizure, generalized", "generalized seizure", LabelAgreement.MATCH),
        ("T-cell receptor", "T cell receptor", LabelAgreement.MATCH),
        ("Long QT syndrome", "Long QT syndrome 1", LabelAgreement.VARIANT),
        (
            "fibroblast growth factor receptor signalling pathway",
            "fibroblast growth factor receptor signaling pathway",
            LabelAgreement.VARIANT,
        ),
        ("Echocardiography Test", "Malaysia", LabelAgreement.MISMATCH),
        ("water", "oxygen molecule", LabelAgreement.MISMATCH),
    ],
)
def test_label_agreement(reported: str, canonical: str, expected: LabelAgreement) -> None:
    agreement = compare_labels(reported, canonical).agreement

    assert agreement == expected


@pytest.mark.parametrize(
    "reported, exact, related, expected",
    [
        # An exact synonym is another name for the same thing, so it is a match.
        ("Seizures", ["Seizures"], [], LabelAgreement.MATCH),
        ("seizure", ["Seizures"], [], LabelAgreement.MATCH),
        # A related synonym names something adjacent, not the same thing.
        ("Epilepsy", ["Seizures"], ["Epilepsy"], LabelAgreement.VARIANT),
        # Close to a synonym without being it: still worth a look, not an error.
        ("Long fingers", [], ["Long slender fingers"], LabelAgreement.VARIANT),
        # Synonyms must not rescue a genuinely wrong identifier.
        ("Echocardiography Test", ["Malaysia, Federation of"], [], LabelAgreement.MISMATCH),
    ],
)
def test_synonyms_are_weighed_by_scope(
    reported: str, exact: list[str], related: list[str], expected: LabelAgreement
) -> None:
    """Exact synonyms are the term's own names; other scopes are only adjacent."""
    canonical = "Malaysia" if expected is LabelAgreement.MISMATCH else "Seizure"
    if reported == "Long fingers":
        canonical = "Arachnodactyly"

    assert compare_labels(reported, canonical, exact, related).agreement == expected


def test_a_synonym_match_says_which_synonym() -> None:
    """The reader needs to see what the report was reaching for."""
    comparison = compare_labels("Epilepsy", "Seizure", [], ["Epilepsy"])

    assert comparison.matched_synonym == "Epilepsy"
    assert comparison.agreement == LabelAgreement.VARIANT


def test_the_terms_own_label_wins_over_a_synonym() -> None:
    """A name that is the label is a match on the label, with no synonym cited."""
    comparison = compare_labels("Seizure", "Seizure", ["Seizures"], [])

    assert comparison.agreement == LabelAgreement.MATCH
    assert comparison.matched_synonym is None


@pytest.mark.parametrize(
    "mapping, expected_exact, expected_related",
    [
        (
            {
                "rdfs:label": ["Seizure"],
                "oio:hasExactSynonym": ["Seizures"],
                "oio:hasRelatedSynonym": ["Epilepsy"],
                "oio:hasBroadSynonym": ["Neurologic abnormality"],
            },
            ("Seizures",),
            ("Epilepsy", "Neurologic abnormality"),
        ),
        # The label is not a synonym of itself, and is compared against directly.
        ({"rdfs:label": ["Seizure"]}, (), ()),
    ],
)
def test_an_oak_alias_map_is_split_by_scope(
    mapping: dict, expected_exact: tuple, expected_related: tuple
) -> None:
    """The shape OAK's local adapters serve, and ontogpt reads."""
    from deep_research_client.validation.term_validator import _names_from_alias_map

    names = _names_from_alias_map(mapping)

    assert names.exact == expected_exact
    assert set(names.related) == set(expected_related)


def test_a_missing_label_is_not_judged() -> None:
    assert compare_labels("", "Seizure").agreement == LabelAgreement.NOT_ASSESSED
    assert compare_labels("Seizure", "").agreement == LabelAgreement.NOT_ASSESSED


def test_mismatches_score_far_below_near_misses() -> None:
    """The threshold has margin on both sides, so it is not a knife edge."""
    mismatch = label_similarity("Echocardiography Test", "Malaysia")
    near_miss = label_similarity("Type 2 diabetes", "type 2 diabetes mellitus")

    assert mismatch < 0.3 < 0.7 < near_miss


# --------------------------------------------------------------------------
# Validation, offline
# --------------------------------------------------------------------------


def test_a_correctly_named_term_passes(offline_validator: TermValidator) -> None:
    report = offline_validator.validate_markdown("| Seizure | HP:0001250 |")

    check = report.checked_terms[0]
    assert check.status == TermStatus.VERIFIED
    assert check.agreement == LabelAgreement.MATCH
    assert check.ontology_label == "Seizure"
    assert not report.has_problems


def test_a_real_term_named_as_something_else_is_caught(
    offline_validator: TermValidator,
) -> None:
    """The failure from the issue: the identifier resolves and means Malaysia."""
    report = offline_validator.validate_markdown("| Echocardiography Test | NCIT:C16814 |")

    check = report.checked_terms[0]
    assert check.status == TermStatus.VERIFIED
    assert check.agreement == LabelAgreement.MISMATCH
    assert check.ontology_label == "Malaysia"
    assert check.reported_labels == ["Echocardiography Test"]
    assert report.has_problems
    assert [t.term_id for t in report.mislabelled_terms] == ["NCIT:C16814"]


def test_an_unlabelled_term_is_resolved_but_not_judged(
    offline_validator: TermValidator,
) -> None:
    report = offline_validator.validate_markdown("Discussed at length in MONDO:0007947.")

    check = report.checked_terms[0]
    assert check.status == TermStatus.VERIFIED
    assert check.agreement == LabelAgreement.NOT_ASSESSED
    assert report.labels_checked == 0


def test_the_worst_of_several_names_is_the_verdict(
    offline_validator: TermValidator,
) -> None:
    """One correct name does not excuse an incorrect one for the same term."""
    markdown = "| Seizure | HP:0001250 |\n| Malaysia | HP:0001250 |\n"

    check = offline_validator.validate_markdown(markdown).checked_terms[0]

    assert check.agreement == LabelAgreement.MISMATCH
    assert check.reported_labels == ["Seizure", "Malaysia"]


def test_label_checking_can_be_turned_off(label_cache: Path) -> None:
    validator = TermValidator(
        cache_dir=label_cache, offline=True, cache_labels=False, check_labels=False
    )

    check = validator.validate_markdown("| Echocardiography Test | NCIT:C16814 |").checked_terms[0]

    assert check.status == TermStatus.VERIFIED
    assert check.agreement == LabelAgreement.NOT_ASSESSED


def test_an_unreachable_term_is_unverifiable_not_invented(
    offline_validator: TermValidator,
) -> None:
    """Offline, an uncached term teaches us nothing, so it is not an accusation."""
    report = offline_validator.validate_markdown("Cites HP:0000118.")

    assert report.checked_terms[0].status == TermStatus.UNVERIFIABLE
    assert not report.has_problems


def test_an_unknown_prefix_is_unverifiable(offline_validator: TermValidator) -> None:
    report = offline_validator.validate_markdown("Cites FAKEONT:0001234.")

    assert report.checked_terms[0].status == TermStatus.UNVERIFIABLE
    assert report.unresolvable_prefixes == ["FAKEONT"]


def test_a_skipped_prefix_is_not_looked_up(label_cache: Path) -> None:
    validator = TermValidator(
        cache_dir=label_cache, offline=True, cache_labels=False, skip_prefixes=["NCIT"]
    )

    report = validator.validate_markdown("| Echocardiography Test | NCIT:C16814 |")

    check = report.checked_terms[0]
    assert check.status == TermStatus.UNVERIFIABLE
    assert "skipped" in (check.message or "")
    # A skipped prefix is a choice, not a gap in coverage, so it is not listed
    # among the prefixes nothing could resolve.
    assert report.unresolvable_prefixes == []


def test_a_term_limit_truncates_and_says_so(offline_validator: TermValidator) -> None:
    offline_validator.max_terms = 1

    report = offline_validator.validate_markdown("HP:0001250 and HP:0001083 and MONDO:0007947")

    assert report.total_terms == 1
    assert report.truncated
    assert "stopped early" in report.to_markdown()


def test_a_previous_section_is_not_re_extracted(offline_validator: TermValidator) -> None:
    """Re-validating an annotated report must not read its own output back in."""
    body = "| Echocardiography Test | NCIT:C16814 |\n"
    once = offline_validator.validate_markdown(body)
    annotated = body + "\n" + once.to_markdown()

    twice = offline_validator.validate_markdown(annotated)

    assert [t.term_id for t in twice.checked_terms] == ["NCIT:C16814"]
    assert twice.checked_terms[0].reported_labels == ["Echocardiography Test"]


def test_a_research_result_is_validated(offline_validator: TermValidator) -> None:
    result = ResearchResult(
        query="Marfan syndrome",
        markdown="| Seizure | HP:0001250 |",
        provider="mock",
        citations=["Appendix: MONDO:0007947"],
    )

    report = offline_validator.validate_result(result)

    assert [t.term_id for t in report.checked_terms] == ["HP:0001250", "MONDO:0007947"]


def test_terms_can_be_supplied_directly(offline_validator: TermValidator) -> None:
    """The library entry point below extraction, for callers with their own terms."""
    report = offline_validator.validate_terms(
        [FoundTerm(term_id="NCIT:C16814", prefix="NCIT", labels=("Echocardiography Test",))]
    )

    assert report.checked_terms[0].agreement == LabelAgreement.MISMATCH


def test_the_extra_is_required() -> None:
    from deep_research_client.validation import term_validator_is_available

    assert term_validator_is_available() is True


# --------------------------------------------------------------------------
# Report model and rendering
# --------------------------------------------------------------------------


def test_an_empty_report_says_it_found_nothing() -> None:
    assert "No ontology term identifiers" in TermValidationReport().to_markdown()


def test_counts_exclude_unverifiable_terms_from_the_rate() -> None:
    """Skipping half a term list must not halve its apparent failure rate."""
    report = TermValidationReport(
        terms=[
            TermCheck(term_id="HP:0000001", prefix="HP", status=TermStatus.NOT_FOUND),
            TermCheck(term_id="HP:0000002", prefix="HP", status=TermStatus.VERIFIED),
            TermCheck(term_id="XX:0000003", prefix="XX", status=TermStatus.UNVERIFIABLE),
        ]
    )

    assert report.resolvable_count == 2
    assert report.confabulation_rate == 0.5


def test_a_clean_sweep_of_failures_is_hedged() -> None:
    report = TermValidationReport(
        terms=[
            TermCheck(term_id=f"HP:999999{n}", prefix="HP", status=TermStatus.NOT_FOUND)
            for n in range(3)
        ]
    )

    assert report.all_terms_failed
    assert "ontology service" in report.to_markdown()


def test_one_failure_is_not_hedged_away() -> None:
    """Hedging a single failure would excuse the mistake the check exists for."""
    report = TermValidationReport(
        terms=[TermCheck(term_id="HP:9999999", prefix="HP", status=TermStatus.NOT_FOUND)]
    )

    assert not report.all_terms_failed


def test_the_summary_states_what_the_rate_hides() -> None:
    """A clean resolution rate must not read as an all-clear for labels."""
    report = TermValidationReport(
        terms=[
            TermCheck(
                term_id="NCIT:C16814",
                prefix="NCIT",
                status=TermStatus.VERIFIED,
                ontology_label="Malaysia",
                reported_labels=["Echocardiography Test"],
                agreement=LabelAgreement.MISMATCH,
            )
        ]
    )

    summary = report.summary()

    assert summary["confabulation_rate"] == 0.0
    assert summary["labels_mismatched"] == 1
    assert summary["needs_review"] is True
    # Both names, so the finding is legible without opening the report.
    assert summary["mislabelled_terms"] == [
        {
            "term_id": "NCIT:C16814",
            "reported_labels": ["Echocardiography Test"],
            "ontology_label": "Malaysia",
        }
    ]


def test_obsolete_terms_do_not_fail_a_build_but_are_flagged() -> None:
    report = TermValidationReport(
        terms=[
            TermCheck(
                term_id="GO:0008022",
                prefix="GO",
                status=TermStatus.OBSOLETE,
                ontology_label="obsolete protein C-terminus binding",
                replaced_by="GO:0005515",
            )
        ]
    )

    assert not report.has_problems
    assert report.summary()["needs_review"] is True
    assert report.summary()["obsolete_terms"] == [
        {
            "term_id": "GO:0008022",
            "ontology_label": "obsolete protein C-terminus binding",
            "replaced_by": "GO:0005515",
        }
    ]
    assert "GO:0005515" in report.to_markdown()


def test_an_obsolete_term_without_a_replacement_omits_the_key() -> None:
    """A replacement the ontology does not state is absent, not null."""
    report = TermValidationReport(
        terms=[
            TermCheck(
                term_id="GO:0000005",
                prefix="GO",
                status=TermStatus.OBSOLETE,
                ontology_label="obsolete ribosomal chaperone activity",
            )
        ]
    )

    assert report.summary()["obsolete_terms"] == [
        {
            "term_id": "GO:0000005",
            "ontology_label": "obsolete ribosomal chaperone activity",
        }
    ]


def test_a_term_named_two_ways_keeps_both_in_the_summary() -> None:
    report = TermValidationReport(
        terms=[
            TermCheck(
                term_id="NCIT:C16814",
                prefix="NCIT",
                status=TermStatus.VERIFIED,
                ontology_label="Malaysia",
                reported_labels=["Echocardiography Test", "Echocardiogram"],
                agreement=LabelAgreement.MISMATCH,
            )
        ]
    )

    assert report.summary()["mislabelled_terms"][0]["reported_labels"] == [
        "Echocardiography Test",
        "Echocardiogram",
    ]


def test_variant_labels_are_listed_rather_than_judged() -> None:
    report = TermValidationReport(
        terms=[
            TermCheck(
                term_id="MONDO:0100316",
                prefix="MONDO",
                status=TermStatus.VERIFIED,
                ontology_label="Long QT syndrome 1",
                reported_labels=["Long QT syndrome"],
                agreement=LabelAgreement.VARIANT,
            )
        ]
    )

    assert not report.has_problems
    assert "worth a second look" in report.to_markdown().lower()


def test_inconsistent_naming_is_surfaced() -> None:
    report = TermValidationReport(
        terms=[
            TermCheck(
                term_id="HP:0001250",
                prefix="HP",
                status=TermStatus.VERIFIED,
                ontology_label="Seizure",
                reported_labels=["Seizure", "Seizures"],
                agreement=LabelAgreement.MATCH,
            )
        ]
    )

    assert [t.term_id for t in report.inconsistently_named_terms] == ["HP:0001250"]
    assert "named inconsistently" in report.to_markdown()


def test_the_markdown_contrasts_both_names() -> None:
    report = TermValidationReport(
        terms=[
            TermCheck(
                term_id="NCIT:C16814",
                prefix="NCIT",
                status=TermStatus.VERIFIED,
                ontology_label="Malaysia",
                reported_labels=["Echocardiography Test"],
                agreement=LabelAgreement.MISMATCH,
            )
        ]
    )

    markdown = report.to_markdown()

    assert "Echocardiography Test" in markdown
    assert "Malaysia" in markdown


def test_status_is_stored_as_a_string() -> None:
    """Pins the use_enum_values behaviour inherited from the generated base."""
    check = TermCheck(term_id="HP:11", prefix="HP", status=TermStatus.VERIFIED)

    assert check.status == TermStatus.VERIFIED
    assert check.status == "VERIFIED"
    assert not isinstance(check.status, TermStatus)


def test_report_serialises_with_exclude_none() -> None:
    assert TermValidationReport().model_dump(exclude_none=True) == {"truncated": False}


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def test_both_validation_sections_are_stripped() -> None:
    annotated = (
        "# Report\n\nBody.\n\n"
        "## Reference Validation\n\nRefs.\n\n"
        "## Term Validation\n\nTerms.\n"
    )

    assert strip_validation_section(annotated) == "# Report\n\nBody.\n"


def test_a_section_is_cut_at_its_heading_not_at_a_later_mention() -> None:
    """Locating the cut by searching for the heading text can split a section.

    The halves are then both written back, mangling the file - and this runs on
    the path that overwrites the user's report.
    """
    from deep_research_client.validation import render_with_sections, split_validation_sections

    doc = (
        "# Report\n\nBody text.\n\n"
        "## Reference Validation\n\nRefs.\n\n"
        "## Term Validation\n\n"
        "Superseded; see ## Term Validation in the archive.\n"
    )

    body, sections = split_validation_sections(doc)

    assert body == "# Report\n\nBody text.\n"
    assert [heading for heading, _ in sections] == [
        "## Reference Validation",
        "## Term Validation",
    ]
    # Rewriting the other section must not split the term section in two. The
    # count is of heading *lines*: the section's own prose mentions the heading
    # mid-sentence, which is the whole point of the case.
    rewritten = render_with_sections(
        body, sections, "## Reference Validation", "## Reference Validation\n\nNew refs."
    )
    assert len(re.findall(r"^## Term Validation$", rewritten, re.MULTILINE)) == 1
    assert "Superseded; see ## Term Validation in the archive." in rewritten


def test_a_discussed_heading_is_left_alone() -> None:
    body = (
        "# Report\n\n## Term Validation\n\nWe explain the method.\n\n"
        "## Conclusions\n\nText.\n"
    )

    assert strip_validation_section(body) == body


def test_the_formatter_embeds_a_term_section() -> None:
    result = ResearchResult(query="q", markdown="Body", provider="mock")
    report = TermValidationReport(
        terms=[
            TermCheck(
                term_id="NCIT:C16814",
                prefix="NCIT",
                status=TermStatus.VERIFIED,
                ontology_label="Malaysia",
                reported_labels=["Echocardiography Test"],
                agreement=LabelAgreement.MISMATCH,
            )
        ]
    )

    formatted = ResultFormatter().format_full_markdown(result, term_validation=report)

    assert "## Term Validation" in formatted
    assert "term_validation:" in formatted
    assert "needs_review: true" in formatted


def test_both_validation_sections_can_be_embedded_together() -> None:
    """--validate-references and --validate-terms compose in one report."""
    from deep_research_client.validation import (
        ReferenceCheck,
        ReferenceStatus,
        ReferenceValidationReport,
    )

    result = ResearchResult(query="q", markdown="Body", provider="mock")
    references = ReferenceValidationReport(
        references=[ReferenceCheck(reference_id="PMID:1", status=ReferenceStatus.VERIFIED)]
    )
    terms = TermValidationReport(
        terms=[TermCheck(term_id="HP:0001250", prefix="HP", status=TermStatus.VERIFIED)]
    )

    formatted = ResultFormatter().format_full_markdown(
        result, reference_validation=references, term_validation=terms
    )

    assert "## Reference Validation" in formatted
    assert "## Term Validation" in formatted
    assert "reference_validation:" in formatted
    assert "term_validation:" in formatted
    # Both come off again, so re-validating does not read them back in.
    assert strip_validation_section(formatted).endswith("Body\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.fixture
def report_file(tmp_path: Path) -> Path:
    path = tmp_path / "report.md"
    path.write_text(
        "# Marfan syndrome\n\n"
        "| Assessment | Term |\n"
        "| --- | --- |\n"
        "| Echocardiography Test | NCIT:C16814 |\n"
        "| Seizure | HP:0001250 |\n",
        encoding="utf-8",
    )
    return path


def _offline_args(report_file: Path, label_cache: Path) -> list[str]:
    return [
        "validate-terms",
        str(report_file),
        "--offline",
        "--cache-dir",
        str(label_cache),
    ]


def test_cli_prints_a_report(report_file: Path, label_cache: Path) -> None:
    result = CliRunner().invoke(app, _offline_args(report_file, label_cache))

    assert result.exit_code == 0
    assert "Malaysia" in result.stdout


def test_cli_writes_in_place(report_file: Path, label_cache: Path) -> None:
    result = CliRunner().invoke(app, _offline_args(report_file, label_cache) + ["--in-place"])

    assert result.exit_code == 0
    written = report_file.read_text(encoding="utf-8")
    assert written.startswith("# Marfan syndrome")
    assert "## Term Validation" in written


def test_cli_in_place_is_idempotent(report_file: Path, label_cache: Path) -> None:
    """A second run must replace the section, not stack another one on it."""
    args = _offline_args(report_file, label_cache) + ["--in-place"]
    CliRunner().invoke(app, args)
    CliRunner().invoke(app, args)

    assert report_file.read_text(encoding="utf-8").count("## Term Validation") == 1


def test_cli_in_place_refreshes_a_stale_frontmatter_summary(
    tmp_path: Path, label_cache: Path
) -> None:
    """A summary left by an earlier run must not contradict the fresh section."""
    path = tmp_path / "annotated.md"
    path.write_text(
        "---\n"
        "title: Report\n"
        "term_validation:\n"
        "  total_terms: 99\n"
        "  needs_review: false\n"
        "---\n"
        "| Echocardiography Test | NCIT:C16814 |\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, _offline_args(path, label_cache) + ["--in-place"])

    assert result.exit_code == 0
    written = path.read_text(encoding="utf-8")
    assert "total_terms: 1" in written
    assert "total_terms: 99" not in written
    assert "needs_review: true" in written
    assert "title: Report" in written


def test_cli_in_place_leaves_a_hand_written_frontmatter_alone(
    tmp_path: Path, label_cache: Path
) -> None:
    """A file that never carried a summary is not reformatted by this tool."""
    path = tmp_path / "handwritten.md"
    path.write_text(
        "---\ntitle: Report\nauthor:   Someone\n---\n| Seizure | HP:0001250 |\n",
        encoding="utf-8",
    )

    CliRunner().invoke(app, _offline_args(path, label_cache) + ["--in-place"])

    assert "author:   Someone" in path.read_text(encoding="utf-8")


@pytest.fixture
def report_with_both_sections(tmp_path: Path) -> Path:
    """A report already annotated by both validation commands."""
    path = tmp_path / "both.md"
    path.write_text(
        "---\n"
        "title: Report\n"
        "reference_validation:\n"
        "  total_references: 1\n"
        "term_validation:\n"
        "  total_terms: 1\n"
        "---\n"
        "# Report\n\n"
        "| Seizure | HP:0001250 |\n\n"
        "## Reference Validation\n\n"
        "All extracted references resolved successfully.\n\n"
        "## Term Validation\n\n"
        "Stale section from an earlier run.\n",
        encoding="utf-8",
    )
    return path


def test_cli_in_place_keeps_the_other_commands_section(
    report_with_both_sections: Path, label_cache: Path
) -> None:
    """Stripping both sections to re-extract must not delete the other one.

    Losing it would also leave its frontmatter summary describing a section no
    longer in the file, which is worse than either failure alone.
    """
    result = CliRunner().invoke(
        app, _offline_args(report_with_both_sections, label_cache) + ["--in-place"]
    )

    assert result.exit_code == 0
    written = report_with_both_sections.read_text(encoding="utf-8")
    assert written.count("## Reference Validation") == 1
    assert "All extracted references resolved successfully." in written
    assert "reference_validation:" in written
    # The term section was rewritten, not stacked.
    assert written.count("## Term Validation") == 1
    assert "Stale section from an earlier run." not in written
    # And the reference section keeps its place ahead of the term section.
    assert written.index("## Reference Validation") < written.index("## Term Validation")


def test_cli_validate_references_in_place_keeps_the_term_section(
    report_with_both_sections: Path
) -> None:
    """The mirror image: the reference command must not drop the term section."""
    result = CliRunner().invoke(
        app,
        [
            "validate-references",
            str(report_with_both_sections),
            "--in-place",
            "--no-check-quotes",
            "--skip-prefix",
            "PMID",
            "--skip-prefix",
            "DOI",
        ],
    )

    assert result.exit_code == 0, result.output
    written = report_with_both_sections.read_text(encoding="utf-8")
    assert written.count("## Term Validation") == 1
    assert "Stale section from an earlier run." in written
    assert written.count("## Reference Validation") == 1


def test_cli_writes_json(report_file: Path, label_cache: Path, tmp_path: Path) -> None:
    out = tmp_path / "terms.json"

    result = CliRunner().invoke(app, _offline_args(report_file, label_cache) + ["--json", str(out)])

    assert result.exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["terms"][0]["term_id"] == "NCIT:C16814"
    assert payload["terms"][0]["ontology_label"] == "Malaysia"


def test_cli_fails_the_build_on_a_mislabelled_term(
    report_file: Path, label_cache: Path
) -> None:
    result = CliRunner().invoke(
        app, _offline_args(report_file, label_cache) + ["--fail-on-unresolved"]
    )

    assert result.exit_code == 2


def test_cli_passes_a_clean_report(tmp_path: Path, label_cache: Path) -> None:
    clean = tmp_path / "clean.md"
    clean.write_text("| Seizure | HP:0001250 |\n", encoding="utf-8")

    result = CliRunner().invoke(
        app, _offline_args(clean, label_cache) + ["--fail-on-unresolved"]
    )

    assert result.exit_code == 0


def test_cli_rejects_a_missing_file(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["validate-terms", str(tmp_path / "nope.md")])

    assert result.exit_code == 1


def test_cli_rejects_multiple_files_with_one_output(
    report_file: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other.md"
    other.write_text("HP:0001250\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["validate-terms", str(report_file), str(other), "--output", str(tmp_path / "o.md")],
    )

    assert result.exit_code == 1


def test_cli_research_warns_about_a_term_flag_passed_without_validation(
    monkeypatch,
) -> None:
    """The unused-flag warning compares against each option's default.

    A repeatable option whose framework default is not None would make this fire
    on every plain research run; its absence is covered by the reference suite's
    equivalent test, and this pins the other direction for the term flags.
    """
    monkeypatch.setenv("ENABLE_MOCK_PROVIDER", "true")

    result = CliRunner().invoke(
        app,
        [
            "research",
            "widgets",
            "--provider",
            "mock",
            "--no-cache",
            "--term-skip-prefix",
            "HP",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--term-skip-prefix has no effect" in result.output


def test_lookup_errors_are_reported_as_an_unreachable_service() -> None:
    """A rate-limited or erroring ontology is an outage, not a traceback.

    The resolver raises rather than reporting a term as absent, so the type has
    to be one the CLI catches; catching only OSError and ValueError would let it
    through as a stack trace.
    """
    from deep_research_client.validation import lookup_error_types

    types = lookup_error_types()

    assert types, "the terms extra is installed, so the outage type should resolve"
    assert all(issubclass(t, Exception) for t in types)
    # Precisely the reason it needs naming: it is not already covered.
    assert not any(issubclass(t, (OSError, ValueError)) for t in types)


# --------------------------------------------------------------------------
# Generated model
# --------------------------------------------------------------------------


def test_term_datamodel_matches_linkml_schema() -> None:
    """term_datamodel.py is generated; regenerate it with `just gen-term-datamodel`.

    Guards against the checked-in Pydantic model drifting from the LinkML schema
    that is its source of truth.
    """
    import shutil
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    schema = Path("src/deep_research_client/validation/term_validation.yaml")
    generated = repo_root / "src/deep_research_client/validation/term_datamodel.py"

    gen_pydantic = shutil.which("gen-pydantic", path=str(Path(sys.executable).parent))
    if not gen_pydantic:
        pytest.skip("linkml is not installed; install the dev dependency group to check drift")

    completed = subprocess.run(
        [gen_pydantic, str(schema)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, f"gen-pydantic failed:\n{completed.stderr}"

    assert completed.stdout == generated.read_text(encoding="utf-8"), (
        "term_datamodel.py does not match term_validation.yaml. Either the schema "
        "changed or linkml was upgraded; run `just gen-term-datamodel` and review the diff."
    )


# --------------------------------------------------------------------------
# Integration: these need a live ontology service
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_live_lookup_catches_the_malaysia_case(tmp_path: Path) -> None:
    validator = TermValidator(cache_dir=tmp_path / "cache")

    report = validator.validate_markdown("| Echocardiography Test | NCIT:C16814 |")

    check = report.checked_terms[0]
    assert check.status == TermStatus.VERIFIED
    assert check.ontology_label == "Malaysia"
    assert check.agreement == LabelAgreement.MISMATCH


@pytest.mark.integration
def test_live_synonyms_prevent_a_false_mismatch(tmp_path: Path) -> None:
    """A report naming a term by a real synonym must not be accused.

    "Long fingers" is what HP:0001166 means; against the label "Arachnodactyly"
    alone it scores below the threshold and reads as a mismatch. The ontology
    lists it, so consulting synonyms is what keeps the check honest here.
    """
    validator = TermValidator(cache_dir=tmp_path / "cache")

    report = validator.validate_markdown("| Long fingers | HP:0001166 |")

    check = report.checked_terms[0]
    assert check.status == TermStatus.VERIFIED
    assert check.agreement != LabelAgreement.MISMATCH
    assert check.matched_synonym
    assert not report.has_problems


@pytest.mark.integration
def test_live_synonyms_do_not_rescue_a_wrong_identifier(tmp_path: Path) -> None:
    """The Malaysia case must survive synonym lookup untouched."""
    validator = TermValidator(cache_dir=tmp_path / "cache")

    check = validator.validate_markdown(
        "| Echocardiography Test | NCIT:C16814 |"
    ).checked_terms[0]

    assert check.agreement == LabelAgreement.MISMATCH
    assert check.ontology_label == "Malaysia"


@pytest.mark.integration
def test_live_exact_synonyms_are_matches(tmp_path: Path) -> None:
    """An exact synonym is one of the term's own names, not a near miss."""
    validator = TermValidator(cache_dir=tmp_path / "cache")

    check = validator.validate_markdown("| Seizures | HP:0001250 |").checked_terms[0]

    assert check.agreement == LabelAgreement.MATCH


@pytest.mark.integration
def test_live_lookup_reports_an_invented_identifier(tmp_path: Path) -> None:
    """A live ontology is the only thing that can say a term does not exist."""
    validator = TermValidator(cache_dir=tmp_path / "cache")

    report = validator.validate_markdown("Cites HP:0001250 and HP:9999999.")

    by_id = {t.term_id: t for t in report.checked_terms}
    assert by_id["HP:0001250"].status == TermStatus.VERIFIED
    assert by_id["HP:9999999"].status == TermStatus.NOT_FOUND
    assert report.has_problems


@pytest.mark.integration
def test_live_lookup_reports_an_obsolete_term(tmp_path: Path) -> None:
    validator = TermValidator(cache_dir=tmp_path / "cache")

    check = validator.validate_markdown("Cites GO:0008022.").checked_terms[0]

    assert check.status == TermStatus.OBSOLETE
    assert check.replaced_by == "GO:0005515"


@pytest.mark.integration
def test_live_lookup_caches_labels_for_reuse(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    TermValidator(cache_dir=cache_dir).validate_markdown("Cites HP:0001250.")

    # The second run is offline, so it can only answer from what the first wrote.
    offline = TermValidator(cache_dir=cache_dir, offline=True, cache_labels=False)
    check = offline.validate_markdown("| Seizure | HP:0001250 |").checked_terms[0]

    assert check.status == TermStatus.VERIFIED
    assert check.agreement == LabelAgreement.MATCH
