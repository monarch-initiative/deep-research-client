"""Classify the semantic relation of candidate claim pairs with an LLM.

The model is shown two claims, A and B, with whatever structure extraction gave
them (subject, predicate, object, qualifier, negation), and asked how A's
assertion relates to B's: identical, one subsuming the other, close, consistent,
complementary, contradictory, or unrelated. It judges only the two claim texts;
the evidential axis (observation vs attribution vs background) is derived, not
judged (see ``alignment_models.evidential_relation``).

The client is any OpenAI-compatible async client, the same kind extraction and the
evaluation judges use, so CBORG, Claude Code and other endpoints work.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .align_parsing import RelationVerdict, relation_from_reply
from .candidates import CandidatePair
from .models import Claim

__all__ = [
    "ALIGN_PROMPT_VERSION",
    "DEFAULT_ALIGN_MAX_TOKENS",
    "Classification",
    "build_align_prompt",
    "classify_pairs",
]

#: Bumped whenever the instructions below change, and recorded on every
#: ClaimAlignment, so alignments made with different prompts are not compared
#: unknowingly.
ALIGN_PROMPT_VERSION = "1"

#: Reply budget per pair when the caller names none. A verdict is short.
DEFAULT_ALIGN_MAX_TOKENS = 1024

_INSTRUCTIONS = """\
You compare two claims, A and B, and say how A's assertion relates to B's. Judge
only what the two claims say. Do not use outside knowledge to decide whether
either is true, and do not reward or penalise either for having a citation.

Give exactly one relation, read from A to B:
- "identical": A and B assert the same thing. Wording may differ, but neither
  adds to nor restricts the other.
- "subsumes": A is the more general claim and B is a special case of it. B adds
  nothing A does not already cover, and A says more or applies more broadly.
- "subsumed_by": A is the more specific claim; A is a special case of B. The
  mirror of subsumes.
- "close": A and B are about the same underlying fact but differ in wording,
  precision or scope in a way that is neither identity nor a clean subsumption
  (for example a different reported percentage for the same relationship, or
  overlapping but not nested populations).
- "consistent": A and B make different assertions that can both be true at once.
  They are compatible but do not state the same fact.
- "complementary": like consistent, but A and B are about the same subject and
  each adds something the other lacks, so together they say more than either.
- "contradictory": A and B cannot both be true. One asserts what the other
  denies, or they give incompatible values for the same relationship.
- "unrelated": A and B are about different subject matter, with no meaningful
  relation to classify.

Also give "confidence", a number from 0 to 1, and "rationale", one sentence
naming what made the relation that and not a neighbouring one.

Reply with JSON only, in this form:
{"relation": "close", "confidence": 0.8, "rationale": "..."}"""


def _describe(claim: Claim) -> str:
    """One claim as the lines the prompt shows for it: text, then any structure."""
    lines = [claim.claim_text]
    triple = " ".join(
        part.label for part in (claim.subject, claim.predicate, claim.object) if part is not None
    )
    if triple:
        lines.append(f"  (as a relation: {'NOT ' if claim.negated else ''}{triple})")
    if claim.qualifier:
        lines.append(f"  (qualifier: {claim.qualifier})")
    return "\n".join(lines)


def build_align_prompt(claim_a: Claim, claim_b: Claim) -> list[dict[str, str]]:
    """The messages sent to classify one pair.

    Args:
        claim_a: The A-side claim.
        claim_b: The B-side claim.

    Returns:
        Chat messages: the fixed instructions, then the two claims.

    >>> a = Claim(id="a", claim_text="FBN1 variants cause Marfan syndrome.",
    ...           anchor_status="EXACT", citation_status="UNCITED")
    >>> b = Claim(id="b", claim_text="Marfan syndrome is caused by FBN1 variants.",
    ...           anchor_status="EXACT", citation_status="UNCITED")
    >>> messages = build_align_prompt(a, b)
    >>> messages[1]["content"].splitlines()[0]
    'CLAIM A:'
    """
    body = f"CLAIM A:\n{_describe(claim_a)}\n\nCLAIM B:\n{_describe(claim_b)}"
    return [
        {"role": "system", "content": _INSTRUCTIONS},
        {"role": "user", "content": body},
    ]


@dataclass(frozen=True)
class Classification:
    """What classifying a set of candidate pairs produced.

    Attributes:
        verdicts: One verdict per pair, in the pairs' order.
        models: The model ids the replies named, distinct, in order of first
            appearance; an alias resolves to a full id here, as in extraction.
    """

    verdicts: list[RelationVerdict]
    models: list[str]

    def model_label(self, requested: str) -> str:
        """The model to record: the ids that answered, else the name requested.

        >>> Classification(verdicts=[], models=["claude-sonnet-5"]).model_label("sonnet")
        'claude-sonnet-5'
        >>> Classification(verdicts=[], models=[]).model_label("gpt-4o-mini")
        'gpt-4o-mini'
        """
        answered = [m for m in self.models if m != requested]
        return ", ".join(answered) or requested


async def _classify_one(
    pair: CandidatePair, llm_client: Any, model: str, max_tokens: int, limit: asyncio.Semaphore,
) -> tuple[RelationVerdict, Optional[str]]:
    """Send one pair and read the verdict, with the model the reply names."""
    async with limit:
        response = await llm_client.chat.completions.create(
            model=model,
            messages=build_align_prompt(pair.claim_a, pair.claim_b),
            temperature=0.0,
            max_tokens=max_tokens,
        )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        from .parsing import UnreadableReplyError

        raise UnreadableReplyError(
            f"The relation reply for {pair.claim_a.id} vs {pair.claim_b.id} was cut off at "
            f"max_tokens={max_tokens}; raise it (--llm-max-tokens)"
        )
    return relation_from_reply(choice.message.content or ""), getattr(response, "model", None)


async def classify_pairs(
    pairs: Sequence[CandidatePair],
    llm_client: Any,
    model: str,
    *,
    concurrency: int = 4,
    max_tokens: int = DEFAULT_ALIGN_MAX_TOKENS,
) -> Classification:
    """Classify every candidate pair's semantic relation, keeping their order.

    A failed request, a reply cut off at ``max_tokens`` and an unreadable reply
    all raise: an alignment silently missing the pairs whose replies failed would
    look complete when it is not.

    Args:
        pairs: The candidate pairs to classify, in order.
        llm_client: An ``openai.AsyncOpenAI``-compatible client.
        model: Model name.
        concurrency: Requests in flight at once.
        max_tokens: Reply budget per pair.

    Returns:
        The verdicts, one per pair in order, and the models that answered.

    Raises:
        UnreadableReplyError: If a reply was truncated or named no relation.
    """
    limit = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.ensure_future(_classify_one(pair, llm_client, model, max_tokens, limit))
        for pair in pairs
    ]
    # Mirror decompose_units: on failure, cancel and drain our own tasks before
    # re-raising, so asyncio.run has nothing left to cancel (a cancelled pipe
    # connection to a subprocess client can otherwise hang).
    try:
        per_pair = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return Classification(
        verdicts=[verdict for verdict, _ in per_pair],
        models=list(dict.fromkeys(answered for _, answered in per_pair if answered)),
    )
