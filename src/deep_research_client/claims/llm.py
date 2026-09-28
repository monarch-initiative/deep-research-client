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
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .models import Claim
from .parsing import TextUnit, UnreadableReplyError, claims_from_reply

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "PROMPT_VERSION",
    "Decomposition",
    "build_prompt",
    "decompose_units",
]

#: Model used when the caller names none, matching the evaluation judges.
DEFAULT_MODEL = "gpt-4o-mini"

#: Reply budget per unit when the caller names none.
DEFAULT_MAX_TOKENS = 4096

#: Bumped whenever the instructions below change, and recorded on every
#: ClaimSet, so sets made with different prompts are never compared unknowingly.
PROMPT_VERSION = "5"

_INSTRUCTIONS = """\
You extract the claims a text makes. A claim is one atomic assertion: a single
statement that could on its own be found true or false. Do not judge whether
any claim is true, and do not add knowledge the text does not state.

Rules:
- One assertion per claim. Split conjunctions and lists: "A causes B and C"
  is two claims. A sentence that states a property and also a cause, or adds
  an appositive or a relative clause ("X, a kinase inhibitor, ..." / "X,
  which causes Y"), makes one claim per assertion.
- Write "claim" as a standalone sentence a reader could understand without
  the text: replace pronouns with what they refer to.
- A table cell's claim must say what its value means: name the row's subject,
  use the column header, and say who or what the value applies to, taken from
  the section or table ("about 60%" in a Frequency column of a table about a
  disease is "... occurs in about 60% of people with <the disease>").
- Keep hedges and conditions ("may", "in adults", "in mouse models") in the
  claim and in "qualifier".
- "quote" must be copied character for character from TEXT: the shortest
  contiguous passage that states the claim. Never paraphrase or join passages.
- Include claims made in tables and lists. Skip headings, questions, and
  statements about how TEXT itself is organised ("this report reviews...").
  Claims about other works (a paper's authors, venue, year, identifiers, or
  what it covers) are claims: keep them, with "about": "work".
- "negated" is true only when the text asserts the relationship does not hold.
- "citations": the citation markers attached to the passage, exactly as
  written (for example "[3]", "PMID:12345", a URL); [] if none. A marker
  belongs only to the claims it supports: when a sentence makes several
  claims, give its marker only to those the cited source is for.
- "subject", "predicate", "object": the assertion's parts when it has that
  shape, else null. "entities": the things the claim mentions.
- "about": "work" when the claim is about a publication or other work as an
  object (its authors, venue, date, identifiers, what it covers); "domain"
  when it is about the subject matter itself.
- "basis", for "domain" claims only (null for "work" claims), how TEXT
  presents the claim: "observation" when TEXT reports it as its own finding
  (its own experiment, analysis, data or case); "secondary_source" when TEXT
  attributes it to another work, by a citation marker or by naming the work
  or its authors; "background_knowledge" when TEXT states it with no citation
  or attribution, as something known in the field. When the Section names a
  cited work (a heading such as "[3] Title"), TEXT reports that work, for
  example its abstract: its domain claims are "secondary_source", never
  "observation".

Example. For TEXT
  Drug Q, an oral kinase inhibitor, reduced tumour size in 40% of patients [4].
the claims are "Drug Q is an oral kinase inhibitor." (quote "Drug Q, an oral
kinase inhibitor", citations [], since [4] reports the result; basis
"background_knowledge") and "Drug Q reduced tumour size in 40% of patients."
(quote "reduced tumour size in 40% of patients", citations ["[4]"]; basis
"secondary_source"). Both are "about": "domain".

Reply with JSON only, in this form:
{"claims": [{"claim": "...", "quote": "...", "subject": "...", "predicate": "...",
  "object": "...", "negated": false, "qualifier": null, "subject_qualifier": null,
  "object_qualifier": null, "entities": ["..."], "citations": ["..."],
  "about": "domain", "basis": "secondary_source"}]}
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


@dataclass(frozen=True)
class Decomposition:
    """What decomposing a source's units produced.

    Attributes:
        claims: Every claim, numbered across the source.
        models: The model ids the replies said answered them, distinct, in
            order of first appearance. An alias such as ``sonnet`` resolves
            to a full id here; an endpoint that names no model adds none.
    """

    claims: list[Claim]
    models: list[str]

    def model_label(self, requested: str) -> str:
        """The model to record: the ids that answered, else the name requested.

        >>> Decomposition(claims=[], models=["claude-sonnet-5"]).model_label("sonnet")
        'claude-sonnet-5'
        >>> Decomposition(claims=[], models=[]).model_label("gpt-4o-mini")
        'gpt-4o-mini'
        """
        return ", ".join(self.models) or requested


async def _extract_unit(
    unit: TextUnit, llm_client: Any, model: str, max_tokens: int, limit: asyncio.Semaphore,
) -> tuple[list[Claim], Optional[str]]:
    """Send one unit and parse the reply; also return the model the reply names."""
    async with limit:
        response = await llm_client.chat.completions.create(
            model=model,
            messages=build_prompt(unit),
            temperature=0.0,
            max_tokens=max_tokens,
        )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        where = unit.section or unit.source_path or "the source"
        raise UnreadableReplyError(
            f"The claim-extraction reply for {where} was cut off at max_tokens={max_tokens}; "
            f"raise max_tokens (--llm-max-tokens) or split the source into smaller units"
        )
    return claims_from_reply(choice.message.content or "", unit), getattr(response, "model", None)


async def decompose_units(
    units: Sequence[TextUnit],
    llm_client: Any,
    model: str = DEFAULT_MODEL,
    *,
    concurrency: int = 4,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Decomposition:
    """Extract atomic claims from every unit, keeping document order.

    Claims are numbered ``c1``, ``c2``... across all units. A failed request,
    a reply cut off at ``max_tokens`` and an unreadable reply all raise: a
    claim set missing a section it silently skipped would look complete when
    it is not.

    Args:
        units: The units to extract from, in document order.
        llm_client: An ``openai.AsyncOpenAI``-compatible client.
        model: Model name.
        concurrency: Requests in flight at once.
        max_tokens: Reply budget per unit.

    Returns:
        All claims, in unit order and reply order within a unit, and the
        models that answered.

    Raises:
        UnreadableReplyError: If a reply was truncated or holds no claims list.
    """
    limit = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.ensure_future(_extract_unit(unit, llm_client, model, max_tokens, limit))
        for unit in units
    ]
    # When one unit fails, gather raises at once and leaves the others
    # running. asyncio.run would then cancel every task in the loop, its own
    # included, and cancelling the task that connects a new subprocess's pipes
    # leaves the process creation waiting forever: the Claude Code client hung
    # there, and the failure was never shown. Cancelling and draining only our
    # own tasks first leaves asyncio.run nothing to cancel.
    try:
        per_unit = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    claims = [claim for unit_claims, _ in per_unit for claim in unit_claims]
    return Decomposition(
        claims=[
            claim.model_copy(update={"id": f"c{number}"})
            for number, claim in enumerate(claims, 1)
        ],
        models=list(dict.fromkeys(answered for _, answered in per_unit if answered)),
    )
