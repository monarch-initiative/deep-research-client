"""Decompose prose into atomic claims with an LLM.

The model is asked for claims *and* for the passage of the input that states
each one. The passage is what makes a claim auditable: it is located in the
source afterwards (``anchoring``), and a claim whose passage is not there is
marked rather than trusted. The model is told not to judge whether a claim is
true -- extraction and verification are separate steps.

The client is any OpenAI-compatible async client, the same kind the
evaluation judges use, so CBORG and other compatible endpoints work.
"""

import asyncio
import logging
from typing import Any, Sequence

from .models import Claim
from .parsing import TextUnit, claims_from_reply

__all__ = ["DEFAULT_MODEL", "PROMPT_VERSION", "build_prompt", "decompose_units"]

logger = logging.getLogger(__name__)

#: Model used when the caller names none, matching the evaluation judges.
DEFAULT_MODEL = "gpt-4o-mini"

#: Bumped whenever the instructions below change, and recorded on every
#: ClaimSet, so sets made with different prompts are never compared unknowingly.
PROMPT_VERSION = "1"

_INSTRUCTIONS = """\
You extract the claims a text makes. A claim is one atomic assertion: a single
statement that could on its own be found true or false. Do not judge whether
any claim is true, and do not add knowledge the text does not state.

Rules:
- One assertion per claim. Split conjunctions and lists: "A causes B and C"
  is two claims.
- Write "claim" as a standalone sentence a reader could understand without
  the text: replace pronouns with what they refer to, and fold a table's
  column or row headers into claims made by its cells.
- Keep hedges and conditions ("may", "in adults", "in mouse models") in the
  claim and in "qualifier".
- "quote" must be copied character for character from TEXT: the shortest
  contiguous passage that states the claim. Never paraphrase or join passages.
- Include claims made in tables and lists. Skip headings, questions, and
  statements about the text itself ("this report reviews...").
- "negated" is true only when the text asserts the relationship does not hold.
- "citations": the citation markers attached to the passage, exactly as
  written (for example "[3]", "PMID:12345", a URL); [] if none.
- "subject", "predicate", "object": the assertion's parts when it has that
  shape, else null. "entities": the things the claim mentions.

Reply with JSON only, in this form:
{"claims": [{"claim": "...", "quote": "...", "subject": "...", "predicate": "...",
  "object": "...", "negated": false, "qualifier": null, "subject_qualifier": null,
  "object_qualifier": null, "entities": ["..."], "citations": ["..."]}]}
If the text makes no claims, reply {"claims": []}."""


def build_prompt(unit: TextUnit) -> list[dict[str, str]]:
    """The messages sent to extract claims from one unit.

    Args:
        unit: The text to extract from.

    Returns:
        Chat messages: the fixed instructions, then the unit with its section.

    >>> messages = build_prompt(TextUnit(text="A causes B.", start=0, end=11, section="Intro"))
    >>> messages[1]["content"].splitlines()[:3]
    ['Section: Intro', '', 'TEXT:']
    """
    context = []
    if unit.section:
        context.append(f"Section: {unit.section}")
    if unit.source_path:
        context.append(f"Field: {unit.source_path}")
    header = "\n".join(context)
    body = f"TEXT:\n<<<\n{unit.body}\n>>>"
    return [
        {"role": "system", "content": _INSTRUCTIONS},
        {"role": "user", "content": f"{header}\n\n{body}" if header else body},
    ]


async def _extract_unit(
    unit: TextUnit, llm_client: Any, model: str, max_tokens: int, limit: asyncio.Semaphore,
) -> list[Claim]:
    """Send one unit and parse the reply."""
    async with limit:
        response = await llm_client.chat.completions.create(
            model=model,
            messages=build_prompt(unit),
            temperature=0.0,
            max_tokens=max_tokens,
        )
    reply = response.choices[0].message.content or ""
    claims = claims_from_reply(reply, unit)
    if not claims and reply.strip() and '"claims"' not in reply:
        where = unit.section or unit.source_path or "the source"
        logger.warning("Unreadable claim-extraction reply for %s; no claims recorded", where)
    return claims


async def decompose_units(
    units: Sequence[TextUnit],
    llm_client: Any,
    model: str = DEFAULT_MODEL,
    *,
    concurrency: int = 4,
    max_tokens: int = 4096,
) -> list[Claim]:
    """Extract atomic claims from every unit, keeping document order.

    Claims are numbered ``c1``, ``c2``... across all units. A failed request
    raises: a claim set missing a section it silently skipped would look
    complete when it is not.

    Args:
        units: The units to extract from, in document order.
        llm_client: An ``openai.AsyncOpenAI``-compatible client.
        model: Model name.
        concurrency: Requests in flight at once.
        max_tokens: Reply budget per unit.

    Returns:
        All claims, in unit order and reply order within a unit.
    """
    limit = asyncio.Semaphore(concurrency)
    per_unit = await asyncio.gather(
        *(_extract_unit(unit, llm_client, model, max_tokens, limit) for unit in units)
    )
    claims = [claim for unit_claims in per_unit for claim in unit_claims]
    return [
        claim.model_copy(update={"id": f"c{number}"})
        for number, claim in enumerate(claims, 1)
    ]
