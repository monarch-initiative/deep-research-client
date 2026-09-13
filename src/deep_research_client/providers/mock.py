"""Mock provider for testing and development."""

import asyncio
import re
from datetime import datetime
from typing import List, Optional

from . import ResearchProvider
from ..exceptions import (
    ProviderAuthError,
    ProviderBillingError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderQuotaError,
    ProviderRateLimitError,
    ProviderTransientError,
)
from ..models import ResearchResult, ProviderConfig
from ..provider_params import MockParams


#: The failure each ``error_type`` stands for: the class, the status code a
#: real provider would have carried it on, and any extra the constructor takes.
#: Keyed by the same literals MockParams accepts, so the two cannot drift apart
#: silently.
#:
#: The quota entry carries a reset time on purpose. ProviderQuotaError is the
#: one error that folds provider-supplied text into its own remedy, so without
#: it the mock cannot reproduce the case the report's redaction exists for --
#: the comma keeps everything after the time, and none of it reaches the file.
_SIMULATED_ERRORS: dict[str, tuple[type[ProviderError], Optional[int], dict]] = {
    "auth": (ProviderAuthError, 401, {}),
    "billing": (ProviderBillingError, 402, {}),
    # No status code, matching the real thing: a spent allowance is detected
    # from the CLI's own wording, so ProviderQuotaError is built without one.
    # 429 would also read as a contradiction of the documented rule that a 429
    # means wait rather than switch -- that is ProviderRateLimitError's status.
    "quota": (ProviderQuotaError, None, {"resets_at": "3pm, pool quota_pool_7f21"}),
    "rate_limit": (ProviderRateLimitError, 429, {}),
    "transient": (ProviderTransientError, 503, {}),
    # Reproduces the error *type* rather than the path: a real
    # ProviderNotConfiguredError is raised while resolving a provider, before
    # any research call, so nothing outside this table raises one from here.
    # Simulating the type is still what a fallback test needs.
    "not_configured": (ProviderNotConfiguredError, None, {}),
}


#: Lettered option lines in a multiple-choice prompt, e.g. ``C. Thymine``.
_MCQ_OPTION = re.compile(r"^([A-Z])\.\s+(.+?)\s*$", re.MULTILINE)


def _mcq_options(query: str) -> list[tuple[str, str]]:
    r"""The lettered options in a prompt, or [] if it poses no choice.

    Matching the pattern anywhere in the prompt is not enough: a question can
    open with something that looks exactly like an option line. That property,
    that the score an arm should get is computable in advance, is the whole
    reason this provider exists, and a stray match breaks it.

    Two shapes of stray match, and the second is why a run of one is refused:

    - "E. coli grows anaerobically ...?" yields ``("E", "coli grows ...")``
      ahead of the real A/B/C. ``answer_policy="first"`` answers E, a letter
      never offered, and the cell scores EXTRACTION_FAILED.
    - "A. thaliana is a model plant. Which gene ...?" is worse, because the
      stray letter *is* A. It opens a run of exactly one, the blank line after
      it ends that run, and the real list is never reached. Under
      ``answer_policy="last"`` the mock then answers A -- a letter that *is* on
      offer, in the wrong position -- so nothing fails: the cell scores SCORED
      and counts as correct whenever the ideal shuffles into position one. An
      arm promising to decline every question is recorded as choosing.

    So a qualifying run is at least two options, ascending from A with no gaps,
    which is what the renderer emits and what ``present_choices`` guarantees --
    ``degenerate_reason`` refuses anything that would present fewer. Scanning
    continues past a run rather than stopping at the first, so the real list
    wins: it sits last, immediately above the instruction lines.

    >>> _mcq_options("E. coli grows how?\n\nA. Fast\nB. Slow\n")
    [('A', 'Fast'), ('B', 'Slow')]
    >>> _mcq_options("A. thaliana flowers when?\n\nA. FT\nB. CO\n")
    [('A', 'FT'), ('B', 'CO')]
    >>> # A stray run of two is what distinguishes "last wins" from "first wins":
    >>> _mcq_options(
    ...     "A. thaliana is a plant.\nB. subtilis is a bacterium.\n"
    ...     "Which differs?\n\nA. Kingdom\nB. Size\n")
    [('A', 'Kingdom'), ('B', 'Size')]
    >>> _mcq_options("Which base?\n\nA. Thymine\nB. Guanine\n")
    [('A', 'Thymine'), ('B', 'Guanine')]
    >>> _mcq_options("No options here.")
    []
    """
    best: list[tuple[str, str]] = []
    run: list[tuple[str, str]] = []
    for line in query.splitlines():
        match = _MCQ_OPTION.match(line)
        if match and match.group(1) == chr(ord("A") + len(run)):
            run.append((match.group(1), match.group(2)))
            continue
        if len(run) >= 2:
            best = run
        # A line that is not the next letter ends the run -- including a line
        # that restarts at "A", which is how the real list follows a stray one.
        run = [(match.group(1), match.group(2))] if match and match.group(1) == "A" else []
    if len(run) >= 2:
        best = run
    return best


class MockProvider(ResearchProvider):
    """Mock provider that returns fake responses for testing."""

    #: The reports are invented, so this provider is never fallen back to
    #: automatically. ``--fallback-provider mock`` still reaches it.
    produces_real_reports = False

    def __init__(self, config: ProviderConfig, params: Optional[MockParams] = None):
        """Initialize Mock provider.

        Args:
            config: Provider configuration (API key not required for mock)
            params: Mock-specific parameters
        """
        self.params = params or MockParams()
        super().__init__(config, self.params.model)

    def get_default_model(self) -> str:
        """Get default Mock model."""
        return "mock-model-v1"

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True

    async def research(self, query: str) -> ResearchResult:
        """Perform mock research with fake response.

        Args:
            query: The research question

        Returns:
            ResearchResult with mock content and citations

        Raises:
            ProviderError: If error_type names a failure to simulate; the
                subclass matches the name, so a caller can exercise fallback
                (auth/billing/quota/not_configured) or its refusal to fall back
                (rate_limit/transient) without a real outage.
            ValueError: If include_error parameter is True
        """
        # Simulate API delay
        await asyncio.sleep(self.params.response_delay)

        # Simulate error if requested. The typed failure is checked first: it
        # says more than the generic one, so where a caller asked for both,
        # answering with the vaguer error would be throwing information away.
        if self.params.error_type:
            error_class, status_code, extra = _SIMULATED_ERRORS[self.params.error_type]
            raise error_class(
                self.name,
                f"Mock error: simulated {self.params.error_type} failure",
                status_code,
                **extra,
            )
        if self.params.include_error:
            raise ValueError("Mock error: This is a simulated API error for testing")

        # Generate mock content
        if self.params.custom_response:
            markdown_content = self.params.custom_response
        else:
            markdown_content = self._generate_mock_response(query)

        answer = self._mock_answer(query)
        if answer:
            markdown_content = f"{markdown_content}\n\n{answer}"

        # Generate mock citations
        citations = self._generate_mock_citations(query)

        return ResearchResult(
            markdown=markdown_content,
            citations=citations,
            provider=self.name,
            query=query,
            model=self.model,
            start_time=datetime.now(),
            end_time=datetime.now()
        )

    def _mock_answer(self, query: str) -> str:
        """Answer a multiple-choice prompt according to ``answer_policy``.

        The mock has no idea which option is right, so it answers by position.
        That is the point: the score an "always A" arm deserves can be worked
        out independently, which makes the evaluation harness testable end to
        end without calling a real provider.

        Args:
            query: The prompt sent to the provider.

        Returns:
            Text to append to the response, or "" when nothing should be added.
        """
        policy = self.params.answer_policy
        if policy == "none":
            return ""

        options = _mcq_options(query)
        if not options:
            return ""

        letters = [letter for letter, _ in options]
        chosen = letters[0] if policy in ("first", "echo") else letters[-1]

        if policy == "echo":
            quoted = "\n".join(f"{letter}. {text}" for letter, text in options)
            return f"Restating the question:\n\n{quoted}\n\nAnswer: {chosen}"
        return f"Answer: {chosen}"

    def _generate_mock_response(self, query: str) -> str:
        """Generate mock markdown response based on query and parameters."""
        base_response = f"# Mock Research Response\n\nThe user asked: **{query}**\n\n"

        if self.params.response_length == "short":
            content = (
                "This is a short mock response for testing purposes. "
                "The mock provider has simulated a research query and is returning "
                "this fake content to verify the system works correctly."
            )
        elif self.params.response_length == "long":
            content = (
                "This is an extended mock response designed to test how the system "
                "handles longer content. The mock provider is simulating a comprehensive "
                "research response that might include multiple sections, detailed analysis, "
                "and extensive information.\n\n"
                "## Background Information\n\n"
                "Mock providers are essential for testing research systems without making "
                "actual API calls. They allow developers to verify functionality, test "
                "error handling, and ensure proper data flow through the application.\n\n"
                "## Key Findings\n\n"
                "1. Mock responses enable rapid development and testing\n"
                "2. Parameter validation can be tested without external dependencies\n"
                "3. Caching behavior can be verified with consistent mock data\n\n"
                "## Methodology\n\n"
                "The mock provider generates responses based on configurable parameters "
                "including response length, delay simulation, and error injection capabilities. "
                "This approach provides comprehensive testing coverage while maintaining "
                "deterministic behavior for reliable testing scenarios."
            )
        else:  # medium
            content = (
                "This is a medium-length mock response for testing the research system. "
                "The mock provider generates fake but realistic-looking content to simulate "
                "what a real research API might return. This includes structured information, "
                "proper formatting, and citations to verify all components work correctly.\n\n"
                "## Mock Provider Features\n\n"
                "- Configurable response length and delay\n"
                "- Error simulation capabilities\n"
                "- Custom response text support\n"
                "- Proper citation generation"
            )

        return base_response + content

    def _generate_mock_citations(self, query: str) -> List[str]:
        """Generate mock citations for testing."""
        # Always include one basic citation as requested
        citations = [
            f"Mock Citation: Research on '{query}' - https://example.com/mock-citation-1"
        ]

        # Add more citations based on response length
        if self.params.response_length == "medium":
            citations.append("Mock Journal Article - https://example.com/mock-journal-2024")
        elif self.params.response_length == "long":
            citations.extend([
                "Mock Journal Article - https://example.com/mock-journal-2024",
                "Mock Research Database - https://example.com/mock-database",
                "Mock Academic Paper - https://example.com/mock-paper-doi"
            ])

        return citations