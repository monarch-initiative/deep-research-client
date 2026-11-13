"""Shared scaffolding for CLI-agent research providers.

Some research "providers" are not HTTP APIs but local command-line agents --
Claude Code's ``claude`` binary, Google Antigravity's ``agy`` binary -- driven
in a non-interactive one-shot mode. They differ in their flags, their output
schema, and how they report failure, but they share a surprising amount of
logic once the report text is in hand: the instruction that forces the work to
happen inline, how a report's length is sanity-checked, how its citations are
pulled out, and how a run that answered from memory or declined the task is
recognised as *not a research report* even though the CLI exited 0.

This module holds that shared part -- the constants and the
:class:`CLIAgentResearchProvider` base class -- so each concrete provider
(``claude_code``, ``antigravity``) carries only its own command building,
stream parsing, failure classification and health probe. The base is
deliberately small: it does not try to template the whole ``research()`` flow,
because the per-CLI parsing and provenance capture differ enough that a shared
skeleton would be more conditionals than code.
"""

import asyncio
import logging
import re
from typing import ClassVar, FrozenSet, List, Optional

from . import ResearchProvider
from ..exceptions import ProviderTimeoutError

logger = logging.getLogger(__name__)

#: The line a report opens with when the model could not finish it. The inline
#: directive below asks for it verbatim, and :meth:`_unsuccessful_reason` looks
#: for it, so the instruction and the detection cannot drift apart.
INCOMPLETE_REPORT_MARKER = "**Incomplete report:**"

# Matches the marker at the start of any line: the report may be preceded by
# narration ("Let me search..."). The words are matched rather than the exact
# spelling, because a model reproducing the marker varies its markdown
# ("**Incomplete report**:", "## Incomplete report:", "> **Incomplete
# report:**", a list item, no bold at all), and a missed variant fails quietly:
# the partial report is cached and exits 0. Anchoring to the start of a line,
# before any other text, keeps prose that merely uses the words from matching.
_INCOMPLETE_REPORT_PATTERN = re.compile(
    r"^[ \t>#*_+-]*(?:\d+[.)][ \t]*)?[*_]*incomplete report[*_]*[ \t]*:",
    re.MULTILINE | re.IGNORECASE,
)

# Appended to (or prepended to) every system prompt, the default research
# prompt or a custom one.
#
# Non-interactive one-shot mode has no "later": the process emits one response
# and exits. Local skills/workflows that defer work to a background task would
# be orphaned, leaving us with only a chatty preamble instead of a report, so
# the work has to happen in this run.
#
# Retrieval comes first. Told only to write "a comprehensive report" in one
# response, the model could skip its tools and write the whole thing from
# memory: no sources to curate from, identifiers that cannot be trusted, and
# the large unsourced narrative that output safety classifiers halt. Grounding
# each claim in a fetched source is what the task wants anyway.
#
# There is also a legitimate way out. A run that cannot finish -- a tool fails,
# too little can be retrieved, output is stopped -- is told to return what it
# has, marked incomplete, instead of having no move left but a refusal. It is
# offered as an option, not imposed: the directive does not tell the model not
# to decline. The marker makes the run unsuccessful (output kept, exit code 4),
# so a partial report is flagged for review rather than passed off as finished.
INLINE_REPORT_DIRECTIVE = (
    "IMPORTANT: You are running non-interactively. Do the research now, in this "
    "single run, and write the report inline. Do not defer the work, hand it off "
    "to a background task, workflow or sub-agent, or promise to follow up later. "
    "No one can answer questions during this run, so if the request is ambiguous, "
    "choose a reasonable interpretation and state it at the top of the report.\n\n"
    "Retrieve before you write. Use your research tools (web search, web fetch, "
    "and any others you have been given) to find sources first, and cite the "
    "source of each claim inline, with its URL or identifier, as you write it. "
    "Do not state any specific factual claim, citation or identifier "
    "(PMID, DOI, ontology ID) that you have not retrieved in this session. "
    "If you cannot retrieve enough to support a section, say so in that "
    "section rather than filling it from memory.\n\n"
    "If you cannot complete the report -- a tool fails, too little can be "
    "retrieved, or you are unable to continue -- you may return a partial report "
    "instead of nothing: start it with a line that begins "
    f"'{INCOMPLETE_REPORT_MARKER}' followed by a one-sentence reason, then give "
    "the sourced findings you do have. Output only the report itself in Markdown."
)

# URL pattern for citation extraction (mirrors the HTTP providers).
_URL_PATTERN = re.compile(r"https?://[^\s\)\]>]+")

# Non-URL source identifiers. A run backed by literature tools can cite PMIDs
# or DOIs and no URL, and that is still a sourced report.
_SOURCE_ID_PATTERN = re.compile(
    r"\bPMID:?\s*\d+|\bPMC\d+|\bdoi:?\s*10\.\d{4,9}/\S+|\b10\.\d{4,9}/[^\s,;)\]]+",
    re.IGNORECASE,
)

# How a run that declined to write the report talks about it. Only consulted
# when the text cites nothing, so a sourced report that merely discusses
# classifiers is never read as a refusal.
#
# A refusal verb counts only when its object, within a few words of the same
# clause, is the report itself: "I cannot provide a definitive answer" is
# hedging, and "I cannot provide medical advice. This report ..." or "...
# recommendations, but this report ..." is a disclaimer followed by the report,
# so the gap between verb and noun is spaces only: never a comma, . ; : ! ? or
# a newline. Markdown emphasis around a word ("the **report**") is allowed. The
# noun must stand alone, so "research-backed" is a modifier, not the object.
# Both apostrophes, because models write the curly one.
_REFUSAL_TAIL = (
    r" (?:regenerate|reproduce|produce|write|provide)\b"
    r"(?:[ \t]+[*_]{0,2}[\w'’-]+[*_]{0,2}){0,5}?[ \t]+[*_]{0,2}"
    r"(?:report|narrative|research|write-?up)(?![\w-])"
)
_REFUSAL_PATTERN = re.compile(
    r"safety classifier"
    rf"|\bI(?:['’]m| am) not going to{_REFUSAL_TAIL}"
    rf"|\bI (?:can['’]t|cannot|won['’]t|will not){_REFUSAL_TAIL}",
    re.IGNORECASE,
)


class CLIAgentResearchProvider(ResearchProvider):
    """Base for providers that drive a local agent CLI one-shot.

    Concrete subclasses supply the CLI-specific parts (command building, stream
    parsing, failure classification, health probe) and inherit the retrieval
    bookkeeping and report sanity checks defined here. Every method here is
    pure or depends only on ``self.params`` fields shared by the CLI providers
    (``working_dir``, a timeout), so it can be unit-tested without a subprocess.

    ``self.params`` and ``self.timeout`` are set by each concrete subclass's
    ``__init__`` (as its own params type), so they are not annotated on the base
    -- doing so would widen the subclasses' view of ``self.params`` to the
    common base and hide their provider-specific fields. The two reads in
    ``_run_process`` are marked for the type checker instead.
    """

    #: Tool names that organise or save the agent's own work rather than
    #: retrieve anything, for the default :meth:`_is_retrieval_tool`. A run
    #: whose only calls are these has still answered from memory. Subclasses
    #: that classify tools positively instead (a small known retrieval set)
    #: override :meth:`_is_retrieval_tool` and may leave this empty.
    _NON_RETRIEVAL_TOOLS: ClassVar[FrozenSet[str]] = frozenset()

    @classmethod
    def _is_retrieval_tool(cls, name: str) -> bool:
        """Whether a tool call could have retrieved something.

        The default classifies negatively: anything not on
        :attr:`_NON_RETRIEVAL_TOOLS` counts. A CLI with a large, open-ended
        tool surface is better served by a subclass override that names the
        handful of tools that actually retrieve.
        """
        return name not in cls._NON_RETRIEVAL_TOOLS

    @classmethod
    def _retrieval_calls(
        cls, tool_uses: List[str], permission_denials: Optional[list]
    ) -> int:
        """Count the tool calls that could have retrieved something.

        A denied call retrieved nothing, and neither did a bookkeeping call, so
        a run that tried a tool, was refused and then answered from memory has
        no retrieval calls. Denials are matched to calls by tool name, one
        denial cancelling one call.

        Args:
            tool_uses: Names of the tools the model called, repeats included.
            permission_denials: The run's reported permission denials, each a
                dict with a ``tool_name``, or None.

        Returns:
            How many calls were neither denied nor non-retrieval.

        >>> CLIAgentResearchProvider._retrieval_calls(["a", "b"], None)
        2
        >>> CLIAgentResearchProvider._retrieval_calls(["a"], [{"tool_name": "a"}])
        0
        """
        remaining = list(tool_uses)
        for denial in permission_denials or []:
            dname = denial.get("tool_name") if isinstance(denial, dict) else None
            if dname in remaining:
                remaining.remove(dname)
        return sum(1 for name in remaining if cls._is_retrieval_tool(name))

    @staticmethod
    def _extract_citations(markdown: str) -> List[str]:
        """Extract URL citations from the markdown report, preserving order.

        Args:
            markdown: The markdown research report.

        Returns:
            Deduplicated list of cited URLs.
        """
        citations: List[str] = []
        seen = set()
        for url in _URL_PATTERN.findall(markdown):
            cleaned = url.rstrip(".,;")
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                citations.append(cleaned)
        return citations

    @staticmethod
    def _check_report_length(markdown: str, min_chars: int) -> None:
        """Reject a report too short to be a plausible research result.

        Silent success on an empty run is the costly failure mode: the caller
        gets a well-formed file with real provenance and no content. Failing
        here turns that into a non-zero exit.

        This is an emptiness check, not a quality one: a short "I could not find
        sufficient information" reply passes it.

        Because the run that failed has already been paid for, the rejected text
        is logged in full and previewed in the exception, so the cause is
        diagnosable without a second run.

        Raises:
            ValueError: If ``markdown`` is shorter than ``min_chars``.
        """
        if min_chars <= 0:
            return

        report = markdown.strip()
        if len(report) >= min_chars:
            return

        logger.warning(
            "CLI agent report is below the min_report_chars threshold (%d < %d). "
            "Full text of the rejected report: %s",
            len(report),
            min_chars,
            report or "<empty>",
        )
        preview = report[:200] + ("..." if len(report) > 200 else "")
        raise ValueError(
            f"The CLI returned a {len(report)}-character report, below the "
            f"min_report_chars threshold of {min_chars}. This usually means the "
            "run failed, was cut short, or produced no research. Lower or disable "
            "the threshold (min_report_chars=0) if short answers are expected. "
            f"Report text: {preview!r}"
        )

    @staticmethod
    def _unsuccessful_reason(
        markdown: str,
        citations: List[str],
        retrieval_calls: int,
        web_search_requests: int,
    ) -> Optional[str]:
        """Say why a completed run's text is not a research report, if it is not.

        A one-shot agent CLI exits 0 and reports success whatever the model
        wrote, so a refusal or a from-memory answer would otherwise be written
        out as a valid report. Three shapes are recognised, checked in order:

        - no retrieval call and no web search at all: nothing was retrieved, so
          whatever the text says was written from memory. This holds even when
          the text cites URLs, PMIDs or DOIs: with nothing fetched, those came
          from memory too, and may be hallucinated;
        - a line of the report begins with ``INCOMPLETE_REPORT_MARKER``, in any
          markdown spelling: the model said, as the directive asks, that it
          could not finish. Its sourced findings are kept, but it is not a
          finished report. Any line counts, so a run that writes the marker and
          then recovers is still flagged: the conservative way to be wrong;
        - refusal phrasing aimed at the report itself, in a text that cites
          nothing (no URL, PMID or DOI): the model declined, typically after an
          output safety classifier stopped its first attempt. A cited source
          overrides the phrasing, since judging a sourced report's quality is
          not this check's job.

        Args:
            markdown: The assembled report text.
            citations: URLs extracted from it.
            retrieval_calls: Tool calls that could have retrieved something.
            web_search_requests: Server-side web searches the run reported.

        Returns:
            A one-line reason when the text is not a report, otherwise None.

        >>> CLIAgentResearchProvider._unsuccessful_reason("From memory.", [], 0, 0)
        'no retrieval calls or web searches: the text, and any sources it cites, was written from memory, not researched'
        >>> CLIAgentResearchProvider._unsuccessful_reason(
        ...     "I'm not going to regenerate that narrative.", [], retrieval_calls=3,
        ...     web_search_requests=2,
        ... )
        'the model declined to write the report, and it cites no sources'
        >>> CLIAgentResearchProvider._unsuccessful_reason(
        ...     "Mentions a safety classifier. https://example.org", ["https://example.org"], 1, 0,
        ... ) is None
        True
        >>> CLIAgentResearchProvider._unsuccessful_reason(
        ...     "Searching.\\n\\n**Incomplete report:** the fetch tool failed.\\n\\nOne finding https://example.org",
        ...     ["https://example.org"], retrieval_calls=2, web_search_requests=1,
        ... )
        "the model marked the report incomplete; its reason is on the line that begins 'Incomplete report:'"
        >>> CLIAgentResearchProvider._unsuccessful_reason(
        ...     "As shown (PMID: 12345678), see https://example.org", ["https://example.org"], 0, 0,
        ... )
        'no retrieval calls or web searches: the text, and any sources it cites, was written from memory, not researched'
        >>> CLIAgentResearchProvider._unsuccessful_reason("Researched, uncited.", [], 4, 0) is None
        True
        """
        if retrieval_calls == 0 and web_search_requests == 0:
            return (
                "no retrieval calls or web searches: the text, and any sources it "
                "cites, was written from memory, not researched"
            )
        if _INCOMPLETE_REPORT_PATTERN.search(markdown):
            return (
                "the model marked the report incomplete; its reason is on the line "
                "that begins 'Incomplete report:'"
            )
        if citations or _SOURCE_ID_PATTERN.search(markdown):
            return None
        if _REFUSAL_PATTERN.search(markdown):
            return "the model declined to write the report, and it cites no sources"
        return None

    async def _run_process(
        self, command: List[str], stdin_input: Optional[str] = None
    ) -> tuple[str, str, int]:
        """Run the CLI subprocess, optionally piping ``stdin_input`` to stdin.

        Shared by the CLI providers: one sends the prompt via stdin, the other
        passes it as an argument and sends nothing. Both honour ``self.timeout``
        (a whole-run deadline) and ``self.params.working_dir``.

        Args:
            command: The command argument list.
            stdin_input: Text to pipe to the process's stdin, or None.

        Returns:
            Tuple of (stdout, stderr, return_code).

        Raises:
            ProviderTimeoutError: If the process does not finish within the
                timeout.
        """
        # Set by the concrete subclass's __init__; see the class docstring.
        run_timeout: int = self.timeout  # type: ignore[attr-defined]
        working_dir = getattr(self.params, "working_dir", None)  # type: ignore[attr-defined]

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        payload = stdin_input.encode("utf-8") if stdin_input is not None else None
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=payload),
                timeout=run_timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            # A spent usage allowance can stall a run rather than fail it, and
            # then it arrives here looking like slowness. Say so, so the reader
            # checks the allowance before chasing latency.
            raise ProviderTimeoutError(
                self.name,
                f"the {self.name} run did not finish within {run_timeout}s. If "
                f"this repeats, check the account with `deep-research-client "
                f"providers --check --provider {self.name}` -- a spent usage "
                f"limit can stall a run instead of failing it",
            )

        returncode = -1 if process.returncode is None else process.returncode
        return (
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            returncode,
        )
