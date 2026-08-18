"""Provider-specific parameter models using Pydantic for validation."""

from typing import Optional, Literal, List, Type, Any, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator


class BaseProviderParams(BaseModel):
    """Base provider parameters that all providers can accept."""

    model: Optional[str] = Field(
        default=None, description="Model to use for this provider")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt to override the default research prompt"
    )
    allowed_domains: List[str] = Field(
        default_factory=list,
        description=(
            "Harmonized parameter: Filter web search to specific domains (max 20). "
            "Only include results from these domains. "
            "Example: ['wikipedia.org', 'github.com']. "
            "Use domain names without protocols (http/https)."
        )
    )

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        validate_assignment=True  # Validate on assignment
    )


class PerplexityParams(BaseProviderParams):
    """Parameters specific to Perplexity AI provider.

    Note: Both `allowed_domains` (harmonized) and `search_domain_filter` (Perplexity-specific)
    are supported. If `allowed_domains` is provided and `search_domain_filter` is empty,
    `allowed_domains` will be used. The Perplexity-specific `search_domain_filter` supports
    both allowlist and denylist (prefix with '-' to exclude).
    """

    reasoning_effort: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Reasoning effort level for Perplexity"
    )
    search_recency_filter: Optional[str] = Field(
        default=None,
        description="Filter sources by recency (e.g., 'month', 'week', 'year')"
    )
    search_domain_filter: List[str] = Field(
        default_factory=list,
        description=(
            "Provider-specific alias: Filter search results by domains or URLs. "
            "Supports allowlist (include) and denylist (exclude) modes. "
            "Maximum 20 domains/URLs per request.\n"
            "Examples:\n"
            "  Allowlist: ['wikipedia.org', 'github.com'] - only these domains\n"
            "  Denylist: ['-reddit.com', '-quora.com'] - exclude these domains\n"
            "  Mixed: ['github.com', 'stackoverflow.com', '-reddit.com']\n"
            "Can use domain names (e.g., 'wikipedia.org') or specific URLs.\n"
            "Use simple domain names without protocols (http/https).\n"
            "Note: You can also use the harmonized `allowed_domains` parameter instead."
        )
    )
    return_citations: bool = Field(
        default=True,
        description="Whether to return structured citations"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )

    @model_validator(mode='after')
    def sync_domain_filters(self):
        """Sync allowed_domains with search_domain_filter.

        If allowed_domains is provided but search_domain_filter is empty,
        use allowed_domains as search_domain_filter.
        """
        if self.allowed_domains and not self.search_domain_filter:
            self.search_domain_filter = self.allowed_domains
        return self


class OpenAIParams(BaseProviderParams):
    """Parameters specific to OpenAI provider.

    Supports the harmonized `allowed_domains` parameter (inherited from BaseProviderParams)
    to filter web search results to specific domains (max 20).
    """

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens in response"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter"
    )


class FalconParams(BaseProviderParams):
    """Parameters specific to Falcon provider."""

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum tokens in response"
    )
    max_embedded_images: int = Field(
        default=8,
        ge=0,
        le=100,
        description=(
            "Maximum number of embedded Edison image artifacts to preserve from "
            "verbose message history. Set to 0 to disable recovery of embedded images."
        )
    )


class AstaParams(BaseProviderParams):
    """Parameters specific to the Asta provider."""

    query_char_limit: int = Field(
        default=500,
        ge=50,
        le=5000,
        description=(
            "Maximum number of characters to send to Asta after Markdown sanitization. "
            "Longer queries are truncated near a word boundary."
        )
    )

    paper_limit: int = Field(
        default=50,
        ge=1,
        le=50,
        description="Maximum number of papers to retrieve from Asta relevance search"
    )
    snippet_limit: int = Field(
        default=20,
        ge=1,
        le=20,
        description="Maximum number of evidence snippets to retrieve from Asta"
    )
    snippet_paper_limit: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of retrieved paper IDs to constrain snippet search"
    )
    restrict_snippets_to_papers: bool = Field(
        default=False,
        description=(
            "When true, restrict snippet search to the initially retrieved papers. "
            "Leave false to search the broader corpus and avoid excluding snippet-only sources."
        )
    )
    paper_fields: str = Field(
        default=(
            "title,abstract,authors,year,url,venue,journal,tldr,publicationDate,"
            "citationCount,influentialCitationCount,externalIds"
        ),
        description="Comma-separated Asta paper fields to request"
    )
    publication_date_range: str = Field(
        default="",
        description="Optional Asta publication date filter in <start>:<end> format"
    )
    venues: str = Field(
        default="",
        description="Optional comma-separated venue filter for Asta searches"
    )
    inserted_before: str = Field(
        default="",
        description="Optional snippet cutoff for papers inserted before YYYY[-MM[-DD]]"
    )


class ConsensusParams(BaseProviderParams):
    """Parameters specific to Consensus provider."""

    year_min: Optional[int] = Field(
        default=None,
        description="Minimum publication year for papers"
    )
    year_max: Optional[int] = Field(
        default=None,
        description="Maximum publication year for papers"
    )
    study_types: List[str] = Field(
        default_factory=list,
        description="Filter by study types (e.g., 'RCT', 'Systematic Review')"
    )
    sample_size_min: Optional[int] = Field(
        default=None,
        gt=0,
        description="Minimum sample size for studies"
    )


class MockParams(BaseProviderParams):
    """Parameters specific to Mock provider for testing."""

    response_delay: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Artificial delay in seconds to simulate API call"
    )
    response_length: Literal["short", "medium", "long"] = Field(
        default="medium",
        description="Length of mock response"
    )
    include_error: bool = Field(
        default=False,
        description="Whether to simulate an error response"
    )
    custom_response: Optional[str] = Field(
        default=None,
        description="Custom response text instead of default"
    )


class OpenScientistParams(BaseProviderParams):
    """Parameters specific to OpenScientist research provider.

    OpenScientist runs iterative hypothesis-driven research jobs that
    take 10-60+ minutes to complete. The provider submits a job, polls
    for completion, and downloads the final report.
    """

    max_iterations: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum research iterations per job (API enforces 1-20)"
    )
    use_hypotheses: bool = Field(
        default=False,
        description="Enable hypothesis tracking tools during research"
    )
    investigation_mode: str = Field(
        default="autonomous",
        description="Investigation mode: 'autonomous' (fully automated) or 'coinvestigate' (interactive)"
    )
    poll_interval: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Seconds between status poll requests"
    )
    timeout: int = Field(
        default=3600,
        ge=60,
        le=7200,
        description="Maximum seconds to wait for job completion"
    )
    save_artifacts: bool = Field(
        default=True,
        description=(
            "Download and preserve useful OpenScientist artifacts such as figures, "
            "small structured data files, and rendered reports."
        )
    )
    artifact_max_bytes: int = Field(
        default=5 * 1024 * 1024,
        ge=1,
        le=50 * 1024 * 1024,
        description="Maximum uncompressed bytes to preserve for a single artifact file"
    )


class DeeperMedParams(BaseProviderParams):
    """Parameters specific to the DeepER-Med provider stub.

    There are no provider-specific parameters yet. The upstream system
    (arXiv:2604.15456) is not publicly callable, so its tunable surface is
    unknown; only the inherited base fields are accepted. Fields will be added
    here once an API is published.
    """


class CyberianParams(BaseProviderParams):
    """Parameters specific to Cyberian agent-based research provider.

    Cyberian uses AI agents to perform iterative research workflows,
    unlike API-based providers.
    """

    workflow_file: Optional[str] = Field(
        default=None,
        description="Path to cyberian workflow YAML file (defaults to deep-research.yaml)"
    )
    agent_type: Optional[str] = Field(
        default="claude",
        description="Type of agent to use (claude, aider, cursor, goose, codex)"
    )
    port: Optional[int] = Field(
        default=3284,
        description="Port for agentapi server"
    )
    skip_permissions: bool = Field(
        default=True,
        description="Skip permission checks when starting agents"
    )
    manage_server: bool = Field(
        default=True,
        description=(
            "Start/stop the agentapi server from deep-research-client. "
            "Set false to use an already-running server (e.g., with custom flags)."
        )
    )
    sources: Optional[str] = Field(
        default=None,
        description="Source guidance for the research workflow"
    )
    workdir_base: Optional[str] = Field(
        default=None,
        description=(
            "Base directory for cyberian working dirs. "
            "When set, workspaces are created under this path instead of system temp."
        )
    )
    max_iterations: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Maximum iterations for looping tasks (e.g., iterate subtask). "
            "Useful for testing or limiting long-running workflows. "
            "NOTE: Requires cyberian >= 0.3.0 to take effect."
        )
    )


class ClaudeCodeParams(BaseProviderParams):
    """Parameters specific to the Claude Code research provider.

    Claude Code is invoked as a local command-line tool (the ``claude`` binary)
    rather than via an HTTP API. The provider shells out to it in
    non-interactive "print" mode and lets Claude Code use its own agentic tools
    (web search, file reading, etc.) to perform the research.
    """

    claude_executable: str = Field(
        default="claude",
        description="Path or name of the Claude Code executable to invoke"
    )
    skip_permissions: bool = Field(
        default=False,
        description=(
            "Pass --dangerously-skip-permissions, which bypasses ALL permission "
            "checks and lets the agent use every tool (file edits, shell, etc.). "
            "SECURITY: this overrides allowed_tools entirely, so the allowlist no "
            "longer restricts anything. Defaults to False so the read-only "
            "allowed_tools allowlist governs tool access. Enable only in trusted, "
            "sandboxed environments where running arbitrary tools on an "
            "agent-driven (possibly untrusted) query is acceptable."
        )
    )
    allowed_tools: List[str] = Field(
        default_factory=lambda: ["WebSearch", "WebFetch"],
        description=(
            "Allowlist of Claude Code tool names passed via --allowedTools. In "
            "non-interactive mode, tools not on this list are auto-denied (without "
            "blocking), so this is the primary tool-authority control. Defaults to "
            "a read-only research set (WebSearch, WebFetch) so the out-of-the-box "
            "behavior cannot mutate the filesystem or run shell commands even on an "
            "untrusted query. Widen it for tasks that need more, or set it empty "
            "AND skip_permissions=True to allow all tools. Has no effect when "
            "skip_permissions is True."
        )
    )
    permission_mode: Optional[str] = Field(
        default=None,
        description=(
            "Optional Claude Code --permission-mode value (e.g. 'plan', "
            "'acceptEdits'). Ignored when skip_permissions is true."
        )
    )
    add_dirs: List[str] = Field(
        default_factory=list,
        description="Additional directories to grant Claude Code tool access to (--add-dir)"
    )
    working_dir: Optional[str] = Field(
        default=None,
        description="Working directory in which to run the claude process (defaults to current dir)"
    )
    timeout: int = Field(
        default=1800,
        ge=1,
        description=(
            "Maximum seconds to wait for the claude run before killing it. "
            "Overridden by ProviderConfig.timeout when that is set."
        )
    )
    min_report_chars: int = Field(
        default=200,
        ge=0,
        description=(
            "Minimum plausible length, in characters, of the returned report. A "
            "shorter result raises rather than writing a well-formed file with no "
            "research in it, which is otherwise a silent and expensive no-op. Set "
            "to 0 to disable the check when short answers are expected."
        )
    )
    extra_args: List[str] = Field(
        default_factory=list,
        description=(
            "Extra command-line arguments appended verbatim to the claude invocation. "
            "Escape hatch for flags not otherwise modeled (e.g. ['--max-turns', '20'])."
        )
    )


class BiomniParams(BaseProviderParams):
    """Parameters specific to the Biomni biomedical co-scientist provider.

    Biomni runs locally via the optional ``biomni`` package: it constructs a
    ``biomni.agent.A1`` agent that drives an underlying LLM and executes
    generated code against a large biomedical data lake.

    Note the two distinct model concepts: the harmonized ``model`` field selects
    the *research model card* (``biomni-a1``), while ``llm`` selects the
    *underlying LLM* that Biomni drives (e.g. a Claude model). Leave ``llm`` as
    ``None`` to use Biomni's own default.
    """

    llm: Optional[str] = Field(
        default=None,
        description=(
            "Underlying LLM for the Biomni agent (e.g. 'claude-sonnet-4-20250514'). "
            "Distinct from the harmonized `model` field, which selects the research "
            "model card. Defaults to Biomni's own default when unset."
        )
    )
    source: Optional[str] = Field(
        default=None,
        description=(
            "LLM provider Biomni should use ('Anthropic', 'OpenAI', 'Gemini', "
            "'AzureOpenAI', 'Bedrock', 'Ollama', ...). Defaults to Biomni's "
            "inference from the llm name when unset."
        )
    )
    path: Optional[str] = Field(
        default=None,
        description=(
            "Directory for Biomni's data lake (auto-downloaded, ~11GB on first "
            "run). Defaults to the BIOMNI_DATA_PATH env var, else './biomni_data'."
        )
    )
    timeout: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Per-run timeout in seconds passed to the Biomni agent (A1's "
            "timeout_seconds). When set, takes precedence over "
            "ProviderConfig.timeout; otherwise falls back to it and then to the "
            "provider's default."
        )
    )
    use_tool_retriever: bool = Field(
        default=True,
        description="Let Biomni retrieve the most relevant tools for each task."
    )
    skip_data_lake: bool = Field(
        default=False,
        description=(
            "Skip loading/downloading the large data lake (passes an empty "
            "expected_data_lake_files list to A1). Useful for lightweight tasks "
            "and tests, at the cost of database-backed capabilities."
        )
    )


# Registry mapping provider names to their parameter models
PROVIDER_PARAMS_REGISTRY: dict[str, Type[BaseProviderParams]] = {
    "perplexity": PerplexityParams,
    "openai": OpenAIParams,
    "falcon": FalconParams,
    "asta": AstaParams,
    "consensus": ConsensusParams,
    "mock": MockParams,
    "cyberian": CyberianParams,
    "openscientist": OpenScientistParams,
    "claude_code": ClaudeCodeParams,
    "biomni": BiomniParams,
    "deeper_med": DeeperMedParams,
}


def get_provider_params_class(provider_name: str) -> type[BaseProviderParams]:
    """Get the parameter model class for a provider.

    Args:
        provider_name: Name of the provider

    Returns:
        Parameter model class for the provider

    Raises:
        ValueError: If provider is not found in registry
    """
    params_class = PROVIDER_PARAMS_REGISTRY.get(provider_name)
    if not params_class:
        raise ValueError(
            f"No parameter model found for provider: {provider_name}")
    return params_class


def create_provider_params(
    provider_name: str,
    model: Optional[str] = None,
    provider_params: Optional[Dict[str, Any]] = None
) -> BaseProviderParams:
    """Create and validate provider parameters.

    Args:
        provider_name: Name of the provider
        model: Model to use (overrides provider default)
        provider_params: Provider-specific parameters

    Returns:
        Validated provider parameters instance

    Raises:
        ValueError: If validation fails or provider not found
    """
    params_class = get_provider_params_class(provider_name)

    # Prepare parameter data
    param_data: Dict[str, Any] = {}
    if model:
        param_data["model"] = model
    if provider_params:
        param_data.update(provider_params)

    # Validate and create parameters
    try:
        return params_class(**param_data)
    except Exception as e:
        raise ValueError(f"Invalid parameters for {provider_name}: {e}")
