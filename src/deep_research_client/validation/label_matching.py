"""Comparison of the label a report writes against an ontology term's own label.

Standard library only, so it can be used - and tested - without installing the
optional ``terms`` extra.

The comparison is three-valued, and runs against every name a term carries
rather than its label alone: an ontology's exact synonyms are its own names, and
a report using one has named the term correctly. Scope is kept, because a
*related* synonym is adjacent rather than equivalent.

The three values are deliberate. Demanding an exact string match
would flag "seizures" against HP:0001250's "Seizure" as an error, and a check
that cries wolf on plurals is a check people switch off. Accepting anything
loosely similar would let "Long QT syndrome" pass for
"Long QT syndrome 1", which is a different disease. So a label that differs only
in case, punctuation, word order or plurality is a match; a label that is
recognisably about the same thing but not identical is a variant, reported for a
human to read; and a label with almost nothing in common is a mismatch, which is
the outcome worth acting on.

The threshold between the last two was set against real pairs rather than
guessed. ``NCIT:C16814`` written as "Echocardiography Test" against its actual
label "Malaysia" scores 0.21; the closest genuine near-miss in the same sample,
"Type 2 diabetes" against "type 2 diabetes mellitus", scores 0.77.
"""

import difflib
import re
from typing import NamedTuple, Optional, Sequence

from .term_datamodel import LabelAgreement

__all__ = [
    "VARIANT_SIMILARITY_THRESHOLD",
    "LabelComparison",
    "compare_labels",
    "label_similarity",
    "normalize_label",
]

# Similarity at or above which two labels are treated as recognisably related
# rather than unrelated. Measured pairs cluster well clear of it in both
# directions - see the module docstring - so it is not a knife edge.
VARIANT_SIMILARITY_THRESHOLD = 0.5

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize_label(label: str) -> str:
    """Reduce a label to lowercase words, without punctuation.

    Args:
        label: A label as written.

    Returns:
        The label lowercased, with punctuation replaced by spaces and runs of
        whitespace collapsed.

    Examples:
        >>> normalize_label("Long QT syndrome")
        'long qt syndrome'
        >>> normalize_label("T-cell receptor")
        't cell receptor'
        >>> normalize_label("  Seizure,  generalized ")
        'seizure generalized'
    """
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", label.lower())).strip()


def _fold(token: str) -> str:
    """Fold a regular plural onto its singular.

    Deliberately crude: it exists so "seizures" and "Seizure" compare equal, not
    to stem English. A word of three letters or fewer is left alone, as is one
    ending in a double s, so "loss" and "eyes" are not mangled into "los" and
    "eye" respectively - the second of which would be right, and the first
    wrong, which is the whole reason the rule stays shallow.

    Examples:
        >>> _fold("seizures")
        'seizure'
        >>> _fold("loss")
        'loss'
        >>> _fold("eyes")
        'eyes'
    """
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(label: str) -> list[str]:
    """Normalized, plural-folded tokens of a label, sorted so order does not count.

    Examples:
        >>> _tokens("Seizures, generalized")
        ['generalized', 'seizure']
    """
    return sorted(_fold(token) for token in normalize_label(label).split())


def label_similarity(reported: str, canonical: str) -> float:
    """Score how alike two labels are, from 0 to 1.

    Two measures are combined by taking whichever is more generous: shared
    words, which sees past reordering and inserted qualifiers, and character
    similarity, which sees past spelling differences that leave the words
    themselves different tokens ("signalling" against "signaling").

    Args:
        reported: The label the report wrote.
        canonical: The term's own label.

    Returns:
        Similarity from 0 (nothing in common) to 1 (the same words).

    Examples:
        >>> label_similarity("Marfan syndrome", "Marfan syndrome")
        1.0
        >>> label_similarity("seizures", "Seizure")
        1.0
        >>> round(label_similarity("Echocardiography Test", "Malaysia"), 2)
        0.21
        >>> label_similarity("", "Seizure")
        0.0
    """
    reported_tokens = _tokens(reported)
    canonical_tokens = _tokens(canonical)
    if not reported_tokens or not canonical_tokens:
        return 0.0
    if reported_tokens == canonical_tokens:
        return 1.0

    reported_set, canonical_set = set(reported_tokens), set(canonical_tokens)
    jaccard = len(reported_set & canonical_set) / len(reported_set | canonical_set)
    ratio = difflib.SequenceMatcher(
        None, normalize_label(reported), normalize_label(canonical)
    ).ratio()
    return max(jaccard, ratio)


class LabelComparison(NamedTuple):
    """The verdict on one reported label, and what it rests on.

    Attributes:
        agreement: The verdict.
        similarity: Similarity to the closest name the term carries.
        matched_synonym: The synonym the reported name was recognised as, when
            it was a synonym rather than the term's own label.
    """

    agreement: LabelAgreement
    similarity: float
    matched_synonym: Optional[str] = None


def compare_labels(
    reported: str,
    canonical: str,
    exact_synonyms: Sequence[str] = (),
    related_synonyms: Sequence[str] = (),
) -> LabelComparison:
    """Judge a reported label against every name a term carries.

    A term is not only its label. HP:0001250's label is "Seizure" and its exact
    synonyms include "Seizures" and "Epileptic seizure"; a report writing either
    of those has named the term correctly, and calling that a mismatch is the
    false positive this check most often produces. So exact synonyms count as the
    term's own names.

    Scope is kept, though, because a synonym is not always another way of saying
    the same thing. "Epilepsy" is a *related* synonym of "Seizure" - adjacent,
    not equivalent - so matching one is a variant rather than a match: worth
    reading, not worth failing a build over.

    Args:
        reported: The label the report wrote beside the CURIE.
        canonical: The label the ontology gives the term.
        exact_synonyms: Synonyms the ontology marks as exact, which name the
            same thing as the label.
        related_synonyms: Synonyms of any other scope - related, broad, narrow,
            or of unrecorded scope.

    Returns:
        The agreement verdict, the similarity it rests on, and the synonym that
        carried it, if one did.

    Examples:
        >>> compare_labels("Marfan syndrome", "Marfan syndrome")
        LabelComparison(agreement=<LabelAgreement.MATCH: 'MATCH'>, similarity=1.0, matched_synonym=None)
        >>> compare_labels("seizures", "Seizure").agreement
        <LabelAgreement.MATCH: 'MATCH'>

        An exact synonym is one of the term's own names, and says which:

        >>> comparison = compare_labels(
        ...     "Epileptic seizure", "Seizure", exact_synonyms=["Epileptic seizure"]
        ... )
        >>> comparison.agreement, comparison.matched_synonym
        (<LabelAgreement.MATCH: 'MATCH'>, 'Epileptic seizure')

        A related synonym is adjacent, not equivalent, so it reads as a variant:

        >>> comparison = compare_labels(
        ...     "Epilepsy", "Seizure", related_synonyms=["Epilepsy"]
        ... )
        >>> comparison.agreement, comparison.matched_synonym
        (<LabelAgreement.VARIANT: 'VARIANT'>, 'Epilepsy')

        A different disease in the same family reads as a variant too:

        >>> compare_labels("Long QT syndrome", "Long QT syndrome 1").agreement
        <LabelAgreement.VARIANT: 'VARIANT'>

        The failure this module exists for, which synonyms do not rescue:

        >>> comparison = compare_labels(
        ...     "Echocardiography Test", "Malaysia", exact_synonyms=["Malaysia, Federation of"]
        ... )
        >>> comparison.agreement
        <LabelAgreement.MISMATCH: 'MISMATCH'>

        The similarity is to the closest name the term carries, so it reflects
        the synonym here rather than the label - and is still nowhere near the
        threshold:

        >>> round(comparison.similarity, 2)
        0.23

        A label the report never gave, or a term with none, is not judged:

        >>> compare_labels("", "Seizure")
        LabelComparison(agreement=<LabelAgreement.NOT_ASSESSED: 'NOT_ASSESSED'>, similarity=0.0, matched_synonym=None)
    """
    if not reported.strip() or not canonical.strip():
        return LabelComparison(LabelAgreement.NOT_ASSESSED, 0.0)

    canonical_score = label_similarity(reported, canonical)
    if canonical_score == 1.0:
        return LabelComparison(LabelAgreement.MATCH, canonical_score)

    # Scored once, so the best similarity and the name that produced it come out
    # of the same pass rather than being recomputed and possibly disagreeing.
    exact_scored = [(label_similarity(reported, name), name) for name in exact_synonyms if name]
    related_scored = [
        (label_similarity(reported, name), name) for name in related_synonyms if name
    ]

    for score, name in exact_scored:
        if score == 1.0:
            return LabelComparison(LabelAgreement.MATCH, score, name)

    best_score, best_name = canonical_score, None
    for score, name in exact_scored + related_scored:
        if score > best_score:
            best_score, best_name = score, name

    if best_score >= VARIANT_SIMILARITY_THRESHOLD:
        return LabelComparison(LabelAgreement.VARIANT, best_score, best_name)
    return LabelComparison(LabelAgreement.MISMATCH, best_score, best_name)
