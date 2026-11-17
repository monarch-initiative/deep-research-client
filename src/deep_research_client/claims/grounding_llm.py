"""Ask a model to choose the ontology term a label denotes, among OAK candidates.

The model never invents a CURIE: it is shown the candidates an ontology search
returned (``grounding_oak``) and picks the one that denotes the same concept, or
none. The reply is read by the pure parser in ``grounding_parsing``.

The client is any OpenAI-compatible async client, the same kind extraction and
alignment use, so CBORG, Claude Code and other endpoints work.
"""

from typing import Any, Optional, Sequence

from .grounding import GroundingCandidate, GroundedTerm
from .grounding_parsing import grounding_from_reply

__all__ = [
    "GROUNDING_PROMPT_VERSION",
    "DEFAULT_GROUNDING_MAX_TOKENS",
    "build_grounding_prompt",
    "choose_candidate",
]

#: Bumped whenever the instructions below change, and recorded on every grounded
#: set, so sets grounded with different prompts are not compared unknowingly.
GROUNDING_PROMPT_VERSION = "1"

#: Reply budget per term when the caller names none. A choice is short.
DEFAULT_GROUNDING_MAX_TOKENS = 1024

_INSTRUCTIONS = """\
You ground a term to an ontology. You are given a term as it was written, a
sentence it appears in, and a numbered list of candidate ontology terms, each with
its identifier (a CURIE), its own label, and its definition where there is one.

Choose the one candidate that denotes the same concept as the term, reading it in
the sentence. Match the concept, not the surface string: a candidate whose label or
definition means the same thing is right even if the wording differs, and a
candidate whose string looks similar but means something else is wrong. If no
candidate denotes the term's concept, choose none -- do not stretch to the nearest.

Reply with JSON only: {"id": "<the chosen CURIE, exactly as given>", "reason": "..."}
Use the CURIE of a candidate from the list, or null for none. Never invent a CURIE
that is not in the list."""


def _format_candidates(candidates: Sequence[GroundingCandidate]) -> str:
    """The candidate list as the numbered lines the prompt shows."""
    lines = []
    for i, candidate in enumerate(candidates, 1):
        definition = f" -- {candidate.definition}" if candidate.definition else ""
        lines.append(f"{i}. {candidate.id} ({candidate.label}){definition}")
    return "\n".join(lines)


def build_grounding_prompt(
    label: str, context: str, candidates: Sequence[GroundingCandidate]
) -> list[dict[str, str]]:
    """The messages sent to ground one term.

    Args:
        label: The term as written.
        context: A sentence the term appears in.
        candidates: The ontology terms to choose among.

    Returns:
        Chat messages: the fixed instructions, then the term, its sentence and the
        candidates.

    >>> cands = [GroundingCandidate(id="HP:0000006", label="Autosomal dominant inheritance")]
    >>> messages = build_grounding_prompt("autosomal dominant", "It is autosomal dominant.", cands)
    >>> messages[1]["content"].splitlines()[0]
    'TERM: autosomal dominant'
    """
    body = (
        f"TERM: {label}\n"
        f"SENTENCE: {context}\n\n"
        f"CANDIDATES:\n{_format_candidates(candidates)}"
    )
    return [
        {"role": "system", "content": _INSTRUCTIONS},
        {"role": "user", "content": body},
    ]


async def choose_candidate(
    label: str,
    context: str,
    candidates: Sequence[GroundingCandidate],
    llm_client: Any,
    model: str,
    *,
    max_tokens: int = DEFAULT_GROUNDING_MAX_TOKENS,
) -> tuple[Optional[GroundedTerm], Optional[str]]:
    """Ask the model to ground one term, returning the choice and the model id.

    Args:
        label: The term as written.
        context: A sentence it appears in.
        candidates: The candidates to choose among; must be non-empty.
        llm_client: An ``openai.AsyncOpenAI``-compatible client.
        model: Model name.
        max_tokens: Reply budget.

    Returns:
        The grounded term (or None for "none of these"), and the model id the reply
        named (or None).

    Raises:
        UnreadableReplyError: If the reply was cut off at ``max_tokens``.
    """
    response = await llm_client.chat.completions.create(
        model=model,
        messages=build_grounding_prompt(label, context, candidates),
        temperature=0.0,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        from .parsing import UnreadableReplyError

        raise UnreadableReplyError(
            f"The grounding reply for {label!r} was cut off at max_tokens={max_tokens}; "
            f"raise it (--llm-max-tokens)"
        )
    term = grounding_from_reply(choice.message.content or "", candidates)
    return term, getattr(response, "model", None)
