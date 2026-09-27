"""Tie an extractor's quote back to the source text it claims to come from.

An LLM asked for "the passage that states this claim" usually copies it, but
not always byte for byte: it drops markdown emphasis, a link target, a citation
marker, or reflows whitespace. It sometimes returns something the source never
said. So a quote is located in two passes, and only the span that is found is
recorded:

1. verbatim, which is ``EXACT``;
2. after normalising both sides the same way -- whitespace collapsed, markdown
   emphasis and link targets and bracketed citation markers ignored, quotation
   marks and dashes unified, case folded -- which is ``NORMALIZED``. The span
   still covers the original characters, so ``text[start:end]`` is what the
   source actually says.

Anything else is ``UNANCHORED``: the claim keeps no span, because offsets
guessed for a quote that is not there would be worse than none.
"""

import re
from functools import lru_cache
from typing import Optional

from .models import AnchorStatus, TextSpan

__all__ = ["locate_quote"]

#: Characters folded to a common form before comparing.
_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    " ": " ",
})

#: Markdown emphasis and code markers, dropped wherever they occur.
_MARKUP = frozenset("*_`")

#: Punctuation that attaches to the word before it. A space ahead of one is
#: dropped, so "variants [2]." and "variants." compare equal once the marker
#: between them is gone.
_CLOSING = frozenset(".,;:!?)")

#: A link target, allowing one level of balanced parentheses inside it, as in
#: Wikipedia's "Marfan_syndrome_(disease)".
_LINK_TARGET = r"\((?:[^()\s]|\([^()\s]*\))*\)"

#: A bracketed numeric citation marker, such as [3] or [2, 5-7], with its link
#: target when the marker is itself a link ("[1](https://...)"). Shared with
#: the citation window, which lets a sentence end run on over these.
NUMERIC_MARKER = rf"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\](?:{_LINK_TARGET})?"

#: Stretches of markdown that carry no claim text: a bracketed numeric
#: citation marker such as [3] or [2, 5-7], with its link target when the
#: marker is itself a link ("[1](https://...)"), and the target of any other
#: link (keeping its label). Dropped from both sides, so a quote that omits
#: them still matches. The marker comes first, so a linked marker's target is
#: dropped with it rather than left behind.
_IGNORED = re.compile(
    rf"{NUMERIC_MARKER}"                         # [3], [2, 5], [4-6], [1](https://...)
    rf"|\]{_LINK_TARGET}"                       # "](https://...)" after a link label
)


@lru_cache(maxsize=64)
def _normalized(text: str) -> tuple[str, tuple[int, ...]]:
    """Normalise text, remembering which original character each output came from.

    Cached: every claim from a unit that misses an exact match searches the
    same unit text, which would otherwise be normalised again per claim.

    Args:
        text: Text to normalise.

    Returns:
        The normalised text, and for each of its characters the index of the
        source character it came from.

    >>> _normalized("A  **bold**  claim [3].")[0]
    'a bold claim.'
    >>> _normalized("see [the paper](https://x.org/y) here")[0]
    'see the paper here'
    >>> _normalized("FBN1 variants [1](https://x.org/1) cause it")[0]
    'fbn1 variants cause it'
    >>> _normalized("see [Marfan](https://w.org/Marfan_(disease)) here")[0]
    'see marfan here'
    """
    ignored = [False] * len(text)
    for match in _IGNORED.finditer(text):
        # Keep a link label's own "]" out of the output but drop the target.
        for i in range(match.start(), match.end()):
            ignored[i] = True
    # A link's opening "[" pairs with a dropped "](...)"; drop it too.
    for match in re.finditer(r"\[(?=[^\[\]]*\]\()", text):
        ignored[match.start()] = True

    out: list[str] = []
    origin: list[int] = []
    pending_space = False
    for i, raw in enumerate(text):
        if ignored[i]:
            continue
        ch = raw.translate(_FOLD)
        if ch in _MARKUP:
            continue
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space and ch not in _CLOSING:
            out.append(" ")
            origin.append(i)
        pending_space = False
        for folded in ch.casefold():
            out.append(folded)
            origin.append(i)
    return "".join(out), tuple(origin)


def locate_quote(
    quote: str, text: str, start: int = 0, end: Optional[int] = None,
) -> tuple[Optional[TextSpan], AnchorStatus]:
    """Find where a quote occurs in text, within ``text[start:end]``.

    Args:
        quote: The passage an extractor says states a claim.
        text: The whole source text; returned offsets index into it.
        start: Start of the window to search, for example a section.
        end: End of the window (exclusive); the end of ``text`` if omitted.

    Returns:
        The span found, with its verbatim text, and how it was found; or
        ``(None, UNANCHORED)`` when the quote is not there.

    >>> source = "## Genetics\\nMarfan syndrome is caused by **FBN1** variants [2].\\n"
    >>> span, status = locate_quote("Marfan syndrome is caused by", source)
    >>> status.value, span.start, span.text
    ('EXACT', 12, 'Marfan syndrome is caused by')
    >>> span, status = locate_quote("caused by FBN1 variants.", source)
    >>> status.value, span.text
    ('NORMALIZED', 'caused by **FBN1** variants [2].')
    >>> span, status = locate_quote("caused by TGFBR2 variants", source)
    >>> span, status.value
    (None, 'UNANCHORED')
    """
    window_end = len(text) if end is None else end
    quote = quote.strip()
    if not quote:
        return None, AnchorStatus.UNANCHORED

    exact = text.find(quote, start, window_end)
    if exact >= 0:
        return (
            TextSpan(start=exact, end=exact + len(quote), text=quote),
            AnchorStatus.EXACT,
        )

    needle, _ = _normalized(quote)
    haystack, origin = _normalized(text[start:window_end])
    if not needle:
        return None, AnchorStatus.UNANCHORED
    found = haystack.find(needle)
    if found < 0:
        return None, AnchorStatus.UNANCHORED

    span_start = start + origin[found]
    span_end = start + origin[found + len(needle) - 1] + 1
    # A match that ended just before dropped trailing markup or a citation
    # marker is extended over it, so the span reads as the source does.
    span_end = _extend_over_ignored(text, span_end, window_end)
    return (
        TextSpan(start=span_start, end=span_end, text=text[span_start:span_end]),
        AnchorStatus.NORMALIZED,
    )


def _extend_over_ignored(text: str, position: int, limit: int) -> int:
    """Move an end offset past closing markup and citation markers that follow it.

    Args:
        text: The source text.
        position: Current end offset (exclusive).
        limit: Offset not to move past.

    Returns:
        The end offset after any immediately following emphasis markers,
        citation markers and the punctuation that ends the sentence.

    >>> source = "variants** [2]. Next"
    >>> source[:_extend_over_ignored(source, 8, len(source))]
    'variants** [2].'
    """
    while position < limit:
        if text[position] in _MARKUP:
            position += 1
            continue
        marker = _IGNORED.match(text, position)
        if marker is None and text[position] == " ":
            marker = _IGNORED.match(text, position + 1)
            if marker is not None:
                position = marker.end()
                continue
        if marker is not None:
            position = marker.end()
            continue
        if text[position] in ".;":
            position += 1
        break
    return position
