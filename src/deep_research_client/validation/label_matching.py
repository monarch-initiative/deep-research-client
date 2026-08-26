"""Comparison of the label a report writes against an ontology term's own label.

Standard library only, so it can be used - and tested - without installing the
optional ``terms`` extra.

The comparison is deliberately three-valued. Demanding an exact string match
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

from .term_datamodel import LabelAgreement

__all__ = [
    "VARIANT_SIMILARITY_THRESHOLD",
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


def compare_labels(reported: str, canonical: str) -> tuple[LabelAgreement, float]:
    """Judge a reported label against a term's own label.

    Args:
        reported: The label the report wrote beside the CURIE.
        canonical: The label the ontology gives the term.

    Returns:
        The agreement verdict and the similarity it rests on.

    Examples:
        >>> compare_labels("Marfan syndrome", "Marfan syndrome")
        (<LabelAgreement.MATCH: 'MATCH'>, 1.0)
        >>> compare_labels("seizures", "Seizure")
        (<LabelAgreement.MATCH: 'MATCH'>, 1.0)

        A different disease in the same family reads as a variant, not an error:

        >>> agreement, _ = compare_labels("Long QT syndrome", "Long QT syndrome 1")
        >>> agreement
        <LabelAgreement.VARIANT: 'VARIANT'>

        The failure this module exists for:

        >>> agreement, score = compare_labels("Echocardiography Test", "Malaysia")
        >>> agreement
        <LabelAgreement.MISMATCH: 'MISMATCH'>
        >>> round(score, 2)
        0.21

        A label the report never gave, or a term with none, is not judged:

        >>> compare_labels("", "Seizure")
        (<LabelAgreement.NOT_ASSESSED: 'NOT_ASSESSED'>, 0.0)
    """
    if not reported.strip() or not canonical.strip():
        return LabelAgreement.NOT_ASSESSED, 0.0

    score = label_similarity(reported, canonical)
    if score == 1.0:
        return LabelAgreement.MATCH, score
    if score >= VARIANT_SIMILARITY_THRESHOLD:
        return LabelAgreement.VARIANT, score
    return LabelAgreement.MISMATCH, score
