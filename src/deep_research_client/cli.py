"""CLI interface for deep-research-client."""

import base64
import binascii
from dataclasses import dataclass
import asyncio
import logging
import os
import re
import typer
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List
from typing_extensions import Annotated

if TYPE_CHECKING:  # pragma: no cover - imports only for type checking
    from .validation import (
        ReferenceValidationReport,
        ReferenceValidator,
        TermValidationReport,
        TermValidator,
    )

from .client import DeepResearchClient
from .processing import ResearchProcessor
from .model_cards import (
    DEEPER_MED_ARXIV_ID,
    get_provider_model_cards,
    list_all_models,
    find_models_by_cost,
    find_models_by_capability,
    CostLevel,
    TimeEstimate,
    ModelCapability
)
from .models import ProviderHealth, ResearchResult, sanitize_artifact_filename

# Configure logging
logger = logging.getLogger("deep_research_client")

app = typer.Typer(
    help="deep-research-client: Wrapper for multiple deep research tools")

PROVIDER_CREDENTIAL_HINTS = {
    "openai": ("OPENAI_API_KEY", "OpenAI Deep Research"),
    "falcon": ("EDISON_API_KEY", "Edison Scientific"),
    "asta": ("ASTA_API_KEY", "Asta"),
    "perplexity": ("PERPLEXITY_API_KEY", "Perplexity AI"),
    "consensus": ("CONSENSUS_API_KEY", "Consensus"),
    "openscientist": ("OPENSCIENTIST_API_KEY", "OpenScientist"),
    "claude_code": ("the `claude` CLI on PATH", "Claude Code"),
    "mock": ("ENABLE_MOCK_PROVIDER=true", "Mock provider"),
}

# Providers registered as stubs: the upstream system has no public API yet, so
# they are not merely missing credentials and cannot be enabled by the user.
# Keyed by provider name, valued by a short reason shown in `providers` output.
PROVIDER_STUB_HINTS = {
    "deeper_med": (
        f"DeepER-Med - no public API released yet (arXiv:{DEEPER_MED_ARXIV_ID})"
    ),
}


def setup_logging(verbosity: int) -> None:
    """Set up logging based on verbosity level.

    Args:
        verbosity: Number of -v flags (0=WARNING, 1=INFO, 2=DEBUG, 3+=DEBUG with more detail)
    """
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:  # >= 2
        level = logging.DEBUG

    # Configure format based on verbosity
    if verbosity >= 3:
        # Very verbose: include timestamp, module, and line number
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    elif verbosity >= 2:
        # Debug: include module name
        log_format = '%(levelname)s - %(name)s - %(message)s'
    else:
        # Info/Warning: simple format
        log_format = '%(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=log_format,
        force=True  # Override any existing configuration
    )

    # Set level for our logger
    logger.setLevel(level)

    if verbosity >= 2:
        logger.debug(
            f"Logging configured at {logging.getLevelName(level)} level")


def _collect_noop_research_option_warnings(
    provider: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    use_cborg: bool = False,
    api_key_env: Optional[str] = None,
) -> list[str]:
    """Collect warnings for CLI options that are accepted but ineffective.

    Args:
        provider: The resolved provider name for the research request
        model: Optional model override from CLI
        base_url: Optional custom endpoint override
        use_cborg: Whether the CBORG shortcut was requested
        api_key_env: Optional environment variable override for proxy API keys

    Returns:
        List of warning strings to emit to the user
    """
    warnings: list[str] = []

    if provider == "asta":
        if model:
            warnings.append(
                "Provider 'asta' ignores --model; Asta always uses its fixed retrieval mode."
            )
        if base_url:
            warnings.append(
                "Provider 'asta' ignores --base-url; custom OpenAI-compatible endpoints do not apply to Asta."
            )
        if use_cborg:
            warnings.append(
                "Provider 'asta' ignores --use-cborg; CBORG only applies to the OpenAI provider."
            )
        if api_key_env:
            warnings.append(
                "Provider 'asta' ignores --api-key-env; Asta reads its credential from ASTA_API_KEY."
            )

    return warnings


@dataclass
class _EffectiveResearchOptions:
    """CLI research options after provider-specific no-op pruning."""

    provider_hint: Optional[str]
    model: Optional[str]
    base_url: Optional[str]
    use_cborg: bool
    api_key_env: Optional[str]
    warnings: list[str]


def _resolve_provider_hint(
    provider: Optional[str],
    cache_config,
) -> Optional[str]:
    """Resolve the provider that would be used before proxy overrides are applied."""
    if provider:
        return provider

    preliminary_client = DeepResearchClient(cache_config=cache_config)
    available = preliminary_client.get_available_providers()
    if available:
        return available[0]
    return None


def _effective_research_options(
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    use_cborg: bool,
    api_key_env: Optional[str],
    cache_config,
) -> _EffectiveResearchOptions:
    """Discard provider-specific no-op options before request setup."""
    provider_hint = _resolve_provider_hint(provider, cache_config)
    warnings = _collect_noop_research_option_warnings(
        provider_hint or "",
        model=model,
        base_url=base_url,
        use_cborg=use_cborg,
        api_key_env=api_key_env,
    )

    if provider_hint == "asta":
        return _EffectiveResearchOptions(
            provider_hint=provider_hint,
            model=None,
            base_url=None,
            use_cborg=False,
            api_key_env=None,
            warnings=warnings,
        )

    return _EffectiveResearchOptions(
        provider_hint=provider_hint,
        model=model,
        base_url=base_url,
        use_cborg=use_cborg,
        api_key_env=api_key_env,
        warnings=warnings,
    )


def _unique_artifact_filename(filename: str, used_filenames: set[str], index: int) -> str:
    """Return a filesystem-safe artifact filename unique within one result."""
    safe_name = sanitize_artifact_filename(filename, fallback=f"artifact-{index}")
    if safe_name in {".", ".."}:
        safe_name = f"artifact-{index}"

    candidate = safe_name
    suffix = Path(safe_name).suffix
    stem = Path(safe_name).stem or f"artifact-{index}"
    counter = 2
    while candidate in used_filenames:
        candidate = f"{stem}-{counter}{suffix}"
        counter += 1

    used_filenames.add(candidate)
    return candidate


def _write_result_artifacts(result: ResearchResult, output: Path) -> None:
    """Write research artifacts beside a report and set their relative paths."""
    if not result.artifacts:
        return

    artifact_dir = output.parent / f"{output.stem}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    used_filenames: set[str] = set()
    for index, artifact in enumerate(result.artifacts, 1):
        filename = _unique_artifact_filename(artifact.filename, used_filenames, index)
        artifact_path = artifact_dir / filename
        try:
            content = base64.b64decode(artifact.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid base64 content for artifact {artifact.filename}") from exc
        artifact_path.write_bytes(content)
        artifact.path = artifact_path.relative_to(output.parent).as_posix()


def _echo_no_providers_message() -> None:
    """Tell the user nothing is configured, and what to set.

    A function rather than two inline calls so a test can assert which stream
    it lands on: driving this through CliRunner cannot show that, because the
    pinned click merges stdout and stderr into one buffer.
    """
    typer.echo("No research providers available. Please set API keys:")
    _echo_credential_hints(_settable_credential_hints())


def _settable_credential_hints() -> list[str]:
    """Providers a user can enable by setting an environment variable.

    Under a heading that says "set API keys", a CLI on PATH and a test double
    are not answers. Those entries exist for the `providers` listing, which
    describes what each provider needs rather than telling anyone to set it.

    Returns:
        Provider names whose hint is a variable the reader can export
    """
    return [
        name
        for name, (requirement, _) in PROVIDER_CREDENTIAL_HINTS.items()
        if _ENV_VAR_HINT.match(requirement) and name != "mock"
    ]


def _echo_credential_hints(provider_names: list[str]) -> None:
    """Print credential hints for providers by canonical provider name."""
    for provider_name in provider_names:
        env_var, label = PROVIDER_CREDENTIAL_HINTS[provider_name]
        typer.echo(f"  - {env_var} for {label}")


def _echo_stub_hints() -> None:
    """Print the stub providers, which no credential can enable."""
    if not PROVIDER_STUB_HINTS:
        return
    typer.echo("\nStub providers (not yet callable):")
    for provider_name, reason in PROVIDER_STUB_HINTS.items():
        typer.echo(f"  - {provider_name}: {reason}")


#: A credential hint that names an environment variable, rather than something
#: else the provider needs (a binary on PATH, an installed package).
_ENV_VAR_HINT = re.compile(r"^[A-Z][A-Z0-9_]*(=\S+)?$")


def _why_unconfigured(provider: str) -> str:
    """Say what would make an unregistered provider usable.

    A provider is registered only once whatever it needs is present, so by the
    time we are here the reason is knowable but the provider object is not.
    Reporting "NOT CONFIGURED" without the next step would be the same empty
    answer this command exists to replace.

    Args:
        provider: Canonical provider name.

    Returns:
        Human-readable explanation of what is missing
    """
    if provider in PROVIDER_CREDENTIAL_HINTS:
        requirement, label = PROVIDER_CREDENTIAL_HINTS[provider]
        # Not every entry is a variable to export -- claude_code needs a binary
        # on PATH -- and "set the `claude` CLI on PATH" reads as nonsense.
        if _ENV_VAR_HINT.match(requirement):
            return f"set {requirement} for {label}"
        return f"{label} requires {requirement}"
    if provider in PROVIDER_STUB_HINTS:
        return PROVIDER_STUB_HINTS[provider]
    # Nothing to export: these register only when an optional package imports.
    # The extra is named after the provider, which holds for every name that
    # can reach this branch today.
    return (
        f"registered only when its optional package is installed "
        f"(try `pip install deep-research-client[{provider}]`)"
    )


def _check_provider_health(client: DeepResearchClient, provider: Optional[str]) -> None:
    """Probe providers for live reachability and print the results.

    Args:
        client: Client whose registry holds the providers to probe.
        provider: Probe only this provider, or all configured ones when None.

    Raises:
        typer.Exit: If a named provider is unknown or unconfigured, or any
            provider turned out to be unable to take work.
    """
    from .provider_params import PROVIDER_PARAMS_REGISTRY

    if provider:
        target = client.registry.get_provider(provider)
        if target is None:
            # A provider is only registered once its credential is set, so an
            # absent one is usually an unset key rather than a typo. Saying
            # "unknown" here would send the reader hunting for a spelling
            # mistake instead of exporting a variable.
            if provider in PROVIDER_PARAMS_REGISTRY or provider in PROVIDER_CREDENTIAL_HINTS:
                typer.echo("Provider health:")
                unconfigured = ProviderHealth(
                    provider=provider,
                    configured=False,
                    reachable=False,
                    detail=_why_unconfigured(provider),
                )
                typer.echo(f"  {unconfigured.summary()}")
            else:
                typer.echo(f"Unknown provider: {provider}")
            raise typer.Exit(1)
        targets = [target]
    else:
        # Probing an unconfigured provider only re-reports the missing key.
        targets = client.registry.get_available_providers()
        if not targets:
            # Echoed rather than logged, so the sentence and the hints it
            # introduces land on the same stream. Logging put them on stderr
            # and stdout respectively, so redirecting kept the list and lost
            # the line explaining what it was for.
            typer.echo("No providers are configured, so there is nothing to probe.")
            _echo_credential_hints(list(PROVIDER_CREDENTIAL_HINTS))
            raise typer.Exit(1)

    async def _probe() -> list[ProviderHealth | BaseException]:
        return await asyncio.gather(
            *(t.check_health() for t in targets), return_exceptions=True
        )

    outcomes = asyncio.run(_probe())

    # A probe that raises is still a report. This is the command people run
    # *because* something is already broken, so one provider blowing up must
    # not cost them the lines for all the others.
    reports = [
        outcome
        if isinstance(outcome, ProviderHealth)
        else ProviderHealth(
            provider=target.name,
            configured=True,
            reachable=False,
            detail=f"the probe itself failed: {outcome}",
        )
        for target, outcome in zip(targets, outcomes)
    ]

    typer.echo("Provider health:")
    for report in reports:
        typer.echo(f"  {report.summary()}")

    if any(report.reachable is False for report in reports):
        raise typer.Exit(1)


def _build_reference_validator(
    cache_dir: Optional[Path],
    email: Optional[str],
    full_text: bool,
    max_references: Optional[int],
    skip_prefix: Optional[List[str]] = None,
    rate_limit_delay: Optional[float] = None,
    check_relevance: bool = True,
) -> "ReferenceValidator":
    """Build a ReferenceValidator, exiting with a hint if the extra is missing."""
    from .validation import INSTALL_HINT, ReferenceValidator, validator_is_available

    if not validator_is_available():
        logger.error(INSTALL_HINT)
        raise typer.Exit(1)

    kwargs: dict = {
        "cache_dir": cache_dir,
        "email": email or os.getenv("NCBI_EMAIL"),
        "fetch_full_text": full_text,
        "max_references": max_references,
        "skip_prefixes": list(skip_prefix or []),
        "check_relevance": check_relevance,
    }
    if rate_limit_delay is not None:
        kwargs["rate_limit_delay"] = rate_limit_delay
    return ReferenceValidator(**kwargs)


def _build_term_validator(
    adapter: Optional[str],
    cache_dir: Optional[Path],
    oak_config: Optional[Path],
    offline: bool,
    max_terms: Optional[int],
    skip_prefix: Optional[List[str]] = None,
    check_labels: bool = True,
) -> "TermValidator":
    """Build a TermValidator, exiting with a hint if the extra is missing."""
    from .validation import DEFAULT_ADAPTER, TERM_INSTALL_HINT, TermValidator, term_validator_is_available

    if not term_validator_is_available():
        logger.error(TERM_INSTALL_HINT)
        raise typer.Exit(1)

    return TermValidator(
        adapter=adapter or DEFAULT_ADAPTER,
        cache_dir=cache_dir,
        oak_config=oak_config,
        offline=offline,
        max_terms=max_terms,
        skip_prefixes=list(skip_prefix or []),
        check_labels=check_labels,
    )


def _echo_term_validation_summary(report: "TermValidationReport") -> None:
    """Log a one-line-per-outcome summary of a term validation report."""
    if not report.checked_terms:
        logger.info("No ontology term identifiers found to validate")
        return

    logger.info(
        "Validated %d terms: %d resolved, %d unresolved, %d obsolete, %d unverifiable",
        report.total_terms,
        report.verified_count,
        report.not_found_count,
        report.obsolete_count,
        report.unverifiable_count,
    )
    if report.all_terms_failed:
        logger.warning(
            "Every term failed to resolve, which usually means the ontology service "
            "could not be reached rather than a report full of invented identifiers"
        )
    for check in report.confabulated_terms:
        logger.warning("Unresolved term: %s (%s)", check.term_id, check.message)
    for check in report.obsolete_terms:
        replacement = f", replaced by {check.replaced_by}" if check.replaced_by else ""
        logger.warning("Obsolete term: %s%s", check.term_id, replacement)
    if report.labels_checked:
        logger.info(
            "Checked %d labels: %d match the term, %d name a different one",
            report.labels_checked,
            report.labels_matching,
            len(report.mislabelled_terms),
        )
    for check in report.mislabelled_terms:
        logger.warning(
            "%s is %r, but the report calls it %s",
            check.term_id,
            check.ontology_label,
            ", ".join(repr(label) for label in check.reported_labels or []),
        )
    if report.unresolvable_prefixes:
        logger.info(
            "No resolver covers these prefixes, so their terms were not checked: %s",
            ", ".join(report.unresolvable_prefixes),
        )


def _refresh_validation_frontmatter(
    content: str,
    frontmatter: dict,
    summary: dict,
    key: str = "reference_validation",
) -> str:
    """Bring a stale validation summary in a report's frontmatter up to date.

    A report produced by ``research --validate-references`` or
    ``research --validate-terms`` carries a summary in its frontmatter.
    Re-validating it would otherwise leave that summary contradicting the
    section written below it.

    The frontmatter is only rewritten when such a summary is already present, so
    a hand-written file is never reformatted by a tool that was asked to check
    citations.

    Args:
        content: The report text, with any previous validation section removed.
        frontmatter: Frontmatter already parsed from that text.
        summary: The fresh summary to write under ``key``.
        key: Frontmatter key the summary belongs under.

    Returns:
        The report text, with the summary refreshed if there was one.
    """
    if key not in frontmatter:
        return content

    import yaml

    from .markdown_parser import parse_frontmatter

    updated = dict(frontmatter)
    updated[key] = summary
    _, body = parse_frontmatter(content)
    rendered = yaml.dump(updated, default_flow_style=False, sort_keys=False).rstrip()
    return f"---\n{rendered}\n---\n{body}"


def _echo_validation_summary(report: "ReferenceValidationReport") -> None:
    """Log a one-line-per-outcome summary of a reference validation report."""
    if not report.checked_references:
        logger.info("No PMID or DOI references found to validate")
        return

    logger.info(
        "Validated %d references: %d resolved, %d unresolved, %d unverifiable",
        report.total_references,
        report.verified_count,
        report.not_found_count,
        report.unverifiable_count,
    )
    if report.all_references_failed:
        logger.warning(
            "Every reference failed to resolve, which usually means a network or "
            "rate-limit problem rather than a report full of fabrications"
        )
    for check in report.confabulated_references:
        logger.warning("Unresolved reference: %s (%s)", check.reference_id, check.message)
    if report.quote_checks:
        logger.info(
            "Checked %d quoted claims: %d found in the cited source, %d not",
            report.quotes_checked,
            report.quotes_valid_count,
            len(report.unsupported_quotes),
        )
    for quote_check in report.unsupported_quotes:
        logger.warning(
            "Quote not found in %s: %r", quote_check.reference_id, quote_check.quote[:120]
        )
    if report.unchecked_quotes:
        logger.info(
            "%d quoted claims had nothing to check against", len(report.unchecked_quotes)
        )
    if report.relevance_assessed_count:
        logger.info(
            "Weighed %d references against the report's own vocabulary: %d on topic",
            report.relevance_assessed_count,
            report.on_topic_count,
        )
    for check in report.off_topic_references:
        logger.warning(
            "Reference %s resolves but looks off topic: %s",
            check.reference_id,
            check.title or "(no title)",
        )


@app.callback()
def main_callback(
    verbose: Annotated[int, typer.Option(
        "--verbose", "-v", count=True, help="Increase verbosity (-v, -vv, -vvv)")] = 0,
):
    """Global options for all commands."""
    setup_logging(verbose)


@app.command()
def research(
    query: Annotated[Optional[str], typer.Argument(
        help="Research query or question (not needed if using --template)")] = None,
    provider: Annotated[Optional[str], typer.Option(
        help="Specific provider to use (openai, falcon, asta, perplexity, consensus, openscientist, claude_code, mock)")] = None,
    model: Annotated[Optional[str], typer.Option(
        help="Model to use for the provider (overrides provider default)")] = None,
    output: Annotated[Optional[Path], typer.Option(
        help="Output file path (prints to stdout if not provided)")] = None,
    no_cache: Annotated[bool, typer.Option(
        "--no-cache", help="Disable caching")] = False,
    separate_citations: Annotated[Optional[Path], typer.Option(
        "--separate-citations", help="Save citations to separate file (optional path, defaults to output.citations.md)")] = None,
    cache_dir: Annotated[Optional[Path], typer.Option(
        "--cache-dir", help="Override cache directory (default: ~/.deep_research_cache)")] = None,
    template: Annotated[Optional[Path], typer.Option(
        help="Template file with {variable} placeholders")] = None,
    input_file: Annotated[Optional[Path], typer.Option(
        "--input-file", "-f", help="Read the research query from a text/markdown file")] = None,
    var: Annotated[Optional[List[str]], typer.Option(
        help="Template variable as 'key=value' (can be used multiple times)")] = None,
    param: Annotated[Optional[List[str]], typer.Option(
        help="Provider-specific parameter as 'key=value' (can be used multiple times)")] = None,
    base_url: Annotated[Optional[str], typer.Option(
        "--base-url", help="Custom base URL for API endpoint (for proxies or OpenAI-compatible services)")] = None,
    use_cborg: Annotated[bool, typer.Option(
        "--use-cborg", help="Use CBORG proxy (Berkeley Lab's AI Portal at api.cborg.lbl.gov)")] = False,
    api_key_env: Annotated[Optional[str], typer.Option(
        "--api-key-env", help="Environment variable name to use for API key (e.g., 'CBORG_API_KEY')")] = None,
    # Publication-style metadata options
    title: Annotated[Optional[str], typer.Option(
        "--title", help="Title for the research report")] = None,
    abstract: Annotated[Optional[str], typer.Option(
        "--abstract", help="Abstract or summary for the research")] = None,
    keyword: Annotated[Optional[List[str]], typer.Option(
        "--keyword", help="Keyword/tag for the research (can be used multiple times)")] = None,
    author: Annotated[Optional[str], typer.Option(
        "--author", help="Primary author of the research")] = None,
    contributor: Annotated[Optional[List[str]], typer.Option(
        "--contributor", help="Contributor to the research (can be used multiple times)")] = None,
    # Reference validation options
    validate_references: Annotated[bool, typer.Option(
        "--validate-references", help="Resolve every cited identifier and append a validation section (requires the 'validation' extra)")] = False,
    validation_cache_dir: Annotated[Optional[Path], typer.Option(
        "--validation-cache-dir", help="Directory for cached reference lookups (default: ./references_cache)")] = None,
    validation_email: Annotated[Optional[str], typer.Option(
        "--validation-email", help="Contact email for the NCBI Entrez API (defaults to $NCBI_EMAIL)")] = None,
    validation_full_text: Annotated[bool, typer.Option(
        "--validation-full-text", help="Fetch full text as well as abstracts when validating (~23x slower, better quote checks)")] = False,
    validation_max_references: Annotated[Optional[int], typer.Option(
        "--validation-max-references", min=1, help="Stop after validating this many references")] = None,
    validation_skip_prefix: Annotated[Optional[List[str]], typer.Option(
        "--validation-skip-prefix", help="Identifier prefix to report as unverifiable instead of resolving (repeatable); skipping DOI is the largest saving after caching")] = None,
    validation_rate_limit_delay: Annotated[Optional[float], typer.Option(
        "--validation-rate-limit-delay", min=0.0, help="Seconds to wait between lookups (default: 0.5); lowering it risks rate-limit errors being reported as unresolved references")] = None,
    validation_relevance: Annotated[bool, typer.Option(
        "--validation-relevance/--validation-no-relevance", help="Also weigh each resolved reference against the report's own vocabulary, to flag citations that exist but look off topic (free: no extra lookups)")] = True,
    # Ontology term validation options
    validate_terms: Annotated[bool, typer.Option(
        "--validate-terms", help="Resolve every cited ontology CURIE, check it against the label the report gave it, and append a validation section (requires the 'terms' extra)")] = False,
    term_adapter: Annotated[Optional[str], typer.Option(
        "--term-adapter", help="OAK adapter to resolve terms through (default: ols:; use sqlite:obo: to download each ontology once and answer locally)")] = None,
    term_oak_config: Annotated[Optional[Path], typer.Option(
        "--term-oak-config", help="oak_config.yaml mapping prefixes to adapters, for ontologies the default adapter does not serve")] = None,
    term_cache_dir: Annotated[Optional[Path], typer.Option(
        "--term-cache-dir", help="Directory for cached term labels (default: ./terms_cache)")] = None,
    term_offline: Annotated[bool, typer.Option(
        "--term-offline", help="Resolve terms only from the label cache, never reaching the network")] = False,
    term_max_terms: Annotated[Optional[int], typer.Option(
        "--term-max-terms", min=1, help="Stop after validating this many ontology terms")] = None,
    term_skip_prefix: Annotated[Optional[List[str]], typer.Option(
        "--term-skip-prefix", help="CURIE prefix to report as unverifiable instead of resolving (repeatable)")] = None,
    term_labels: Annotated[bool, typer.Option(
        "--term-labels/--no-term-labels", help="Compare the label written beside each CURIE with the term's own label (free: no extra lookups)")] = True,
    fail_on_unresolved: Annotated[bool, typer.Option(
        "--fail-on-unresolved", help="Exit non-zero if any reference or ontology term fails to resolve, any quote is unsupported, or any term is named as a different term")] = False,
):
    """Perform deep research on a query.

    \b
    Examples:
      # Basic research
      deep-research-client research "What is CRISPR gene editing?"

      # Use specific provider with custom model
      deep-research-client research "Latest AI developments" --provider perplexity --model llama-3.1-sonar-large-128k-online

      # Save to file with separate citations
      deep-research-client research "Climate change impacts" --output report.md --separate-citations

      # Use provider-specific parameters
      deep-research-client research "Medical research" --provider perplexity --param reasoning_effort=high --param search_recency_filter=week

      # Use template with variables
      deep-research-client research --template research_template.md --var topic="machine learning" --var focus="healthcare applications"

      # Read query directly from Markdown/text file
      deep-research-client research --input-file topic.md --provider mock

      # Disable cache and specify custom cache directory
      deep-research-client research "Real-time data" --no-cache --cache-dir ./custom_cache

      # Use CBORG proxy (requires CBORG_API_KEY environment variable)
      deep-research-client research "Quantum computing advances" --use-cborg

      # Use custom OpenAI-compatible endpoint
      deep-research-client research "AI ethics" --base-url https://api.example.com --api-key-env CUSTOM_API_KEY

      # Use CBORG with explicit API key environment variable
      deep-research-client research "Climate models" --use-cborg --api-key-env MY_CBORG_KEY

      # Check every cited PMID/DOI before trusting the report
      deep-research-client research "Statins and myopathy" --validate-references --output report.md
    """
    from .models import CacheConfig

    # Initialize processor
    processor = ResearchProcessor()

    # Load query from file when requested
    if input_file:
        if template:
            logger.error("Cannot combine --input-file with --template")
            raise typer.Exit(1)
        if query:
            logger.error(
                "Provide the query either as an argument or via --input-file, not both")
            raise typer.Exit(1)

        try:
            file_content = input_file.read_text(encoding='utf-8').strip()
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            raise typer.Exit(1)
        except OSError as exc:
            logger.error(f"Unable to read input file {input_file}: {exc}")
            raise typer.Exit(1)

        if not file_content:
            logger.error(f"Input file {input_file} is empty")
            raise typer.Exit(1)

        # Assign stripped content to query so the rest of the pipeline works unchanged
        query = file_content
        logger.info(f"Loaded query from file: {input_file}")

    # Process template if provided
    template_info = None
    if template:
        try:
            # Validate template variables first
            is_valid, error_msg = processor.validate_template_variables(
                template, var)
            if not is_valid:
                logger.error(f"Template error: {error_msg}")
                if error_msg and "requires variables" in error_msg:
                    logger.error("Use --var key=value for each variable")
                raise typer.Exit(1)

            # Process the template
            query, template_info = processor.process_template_file(
                template, var)

            logger.info(f"Using template: {template.name}")
            if template_info['template_variables']:
                var_str = ', '.join(
                    f"{k}={v}" for k, v in template_info['template_variables'].items())
                logger.info(f"Variables: {var_str}")

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Template error: {e}")
            raise typer.Exit(1)

    elif not query:
        logger.error("Either provide a query or use --template")
        raise typer.Exit(1)

    # Parse provider parameters if provided
    provider_params = {}
    if param:
        try:
            for param_str in param:
                if '=' not in param_str:
                    raise ValueError(
                        f"Invalid parameter format: '{param_str}'. Use 'key=value'")
                key, value = param_str.split('=', 1)
                provider_params[key.strip()] = value.strip()
            logger.debug(f"Parsed provider parameters: {provider_params}")
        except ValueError as e:
            logger.error(f"Error parsing parameters: {e}")
            raise typer.Exit(1)

    # Setup cache configuration
    cache_config = CacheConfig(enabled=not no_cache)
    if cache_dir:
        cache_config.directory = str(cache_dir)
        logger.debug(f"Using custom cache directory: {cache_dir}")

    effective_options = _effective_research_options(
        provider=provider,
        model=model,
        base_url=base_url,
        use_cborg=use_cborg,
        api_key_env=api_key_env,
        cache_config=cache_config,
    )

    # Handle proxy/endpoint configuration
    proxy_base_url = None
    proxy_api_key_env = effective_options.api_key_env

    # --use-cborg is a shortcut for CBORG configuration
    if effective_options.use_cborg:
        if effective_options.base_url:
            logger.warning("--use-cborg overrides --base-url")
        proxy_base_url = "https://api.cborg.lbl.gov"
        # Default to CBORG_API_KEY if no specific env var is provided
        if not proxy_api_key_env:
            proxy_api_key_env = "CBORG_API_KEY"
        logger.info(f"Using CBORG proxy at {proxy_base_url}")
    elif effective_options.base_url:
        proxy_base_url = effective_options.base_url
        logger.info(f"Using custom endpoint at {proxy_base_url}")

    # Build provider configs if proxy settings are specified
    provider_configs = None
    if proxy_base_url or proxy_api_key_env:
        from .models import ProviderConfig
        provider_configs = {}

        # Determine API key based on env var
        api_key = None
        if proxy_api_key_env:
            api_key = os.getenv(proxy_api_key_env)
            if not api_key:
                logger.error(
                    f"Environment variable {proxy_api_key_env} not set")
                raise typer.Exit(1)
            logger.debug(f"Using API key from {proxy_api_key_env}")
        else:
            # Use default provider env vars
            if provider == "openai" or not provider:
                api_key = os.getenv("OPENAI_API_KEY")

        # Only configure the selected provider (or openai as default)
        target_provider = provider or "openai"
        if target_provider == "openai":
            provider_configs["openai"] = ProviderConfig(
                name="openai",
                api_key=api_key,
                base_url=proxy_base_url,
                enabled=True
            )

    # Initialize client
    logger.debug("Initializing DeepResearchClient")
    client = DeepResearchClient(
        cache_config=cache_config, provider_configs=provider_configs)

    # Check if any providers are available
    available_providers = client.get_available_providers()
    if not available_providers:
        _echo_no_providers_message()
        raise typer.Exit(1)

    # Show available providers
    if provider:
        if provider not in available_providers:
            logger.error(
                f"Provider '{provider}' not available. Available: {', '.join(available_providers)}")
            raise typer.Exit(1)
        logger.info(f"Using provider: {provider}")
    else:
        logger.info(f"Available providers: {', '.join(available_providers)}")
        logger.info(f"Using: {available_providers[0]}")

    for warning in effective_options.warnings:
        logger.warning(warning)

    # Build publication metadata if any provided
    metadata: Optional[dict] = None
    if title or abstract or keyword or author or contributor:
        metadata = {}
        if title:
            metadata['title'] = title
        if abstract:
            metadata['abstract'] = abstract
        if keyword:
            metadata['keywords'] = keyword
        if author:
            metadata['author'] = author
        if contributor:
            metadata['contributors'] = contributor

    if not validate_references:
        # Compared against the option default rather than truthiness, so that a
        # falsy-but-explicit value such as --validation-email "" still warns.
        unused_validation_flags: tuple[tuple[str, object, object], ...] = (
            ("--validation-cache-dir", validation_cache_dir, None),
            ("--validation-email", validation_email, None),
            ("--validation-full-text", validation_full_text, False),
            ("--validation-max-references", validation_max_references, None),
            ("--validation-skip-prefix", validation_skip_prefix, None),
            ("--validation-rate-limit-delay", validation_rate_limit_delay, None),
            ("--validation-no-relevance", validation_relevance, True),
        )
        for flag_name, flag_value, default in unused_validation_flags:
            if flag_value != default:
                logger.warning(f"{flag_name} has no effect without --validate-references")

    if not validate_terms:
        unused_term_flags: tuple[tuple[str, object, object], ...] = (
            ("--term-adapter", term_adapter, None),
            ("--term-oak-config", term_oak_config, None),
            ("--term-cache-dir", term_cache_dir, None),
            ("--term-offline", term_offline, False),
            ("--term-max-terms", term_max_terms, None),
            ("--term-skip-prefix", term_skip_prefix, None),
            ("--no-term-labels", term_labels, True),
        )
        for flag_name, flag_value, default in unused_term_flags:
            if flag_value != default:
                logger.warning(f"{flag_name} has no effect without --validate-terms")

    if not validate_references and not validate_terms and fail_on_unresolved:
        logger.warning(
            "--fail-on-unresolved has no effect without --validate-references "
            "or --validate-terms"
        )

    # Checked before the provider call: discovering a missing extra after a run
    # that took minutes and real money would be a poor trade.
    if validate_references:
        from .validation import INSTALL_HINT, validator_is_available

        if not validator_is_available():
            logger.error(INSTALL_HINT)
            raise typer.Exit(1)

    if validate_terms:
        from .validation import TERM_INSTALL_HINT, term_validator_is_available

        if not term_validator_is_available():
            logger.error(TERM_INSTALL_HINT)
            raise typer.Exit(1)

    logger.info("Researching...")

    try:
        # Perform research
        logger.debug(f"Starting research with query: {query[:100]}...")
        result = client.research(
            query,
            provider,
            template_info,
            effective_options.model,
            provider_params,
            metadata,
        )

        # Show cache status
        if result.cached:
            logger.info("Result retrieved from cache")
        else:
            logger.info(f"Research completed using {result.provider}")

        # Determine if we're separating citations
        should_separate_citations = separate_citations is not None

        if output:
            _write_result_artifacts(result, output)

        # Format output using processor
        logger.debug("Formatting research result")
        output_content = processor.format_research_result(
            result,
            separate_citations=should_separate_citations,
        )

        # Output result
        if output:
            output.write_text(output_content, encoding='utf-8')
            logger.info(f"Result saved to: {output}")

            # Save separate citations file if requested
            if should_separate_citations and result.citations:
                # Use provided path or default to output.citations.md
                if isinstance(separate_citations, Path):
                    citations_output = separate_citations
                else:
                    citations_output = output.with_suffix('.citations.md')

                citations_content = processor.format_citations_only(result)
                citations_output.write_text(
                    citations_content, encoding='utf-8')
                logger.info(f"Citations saved to: {citations_output}")

            # Show citation count
            if result.citations:
                logger.info(f"Found {len(result.citations)} citations")
        else:
            # For stdout output, handle separate citations differently
            if should_separate_citations and result.citations:
                typer.echo("\n" + "="*60)
                typer.echo(output_content)
                typer.echo("\n" + "="*60)
                typer.echo("CITATIONS:")
                typer.echo("="*60)
                typer.echo(processor.format_citations_only(result))
            else:
                typer.echo("\n" + "="*60)
                typer.echo(output_content)

    except ValueError as exc:
        logger.error(f"Error: {exc}")
        raise typer.Exit(1)
    except OSError as exc:
        logger.error(f"Filesystem error: {exc}")
        logger.debug("Exception details:", exc_info=True)
        raise typer.Exit(1)

    if not validate_references and not validate_terms:
        return

    # Validation runs only after the report has been written or printed. It is
    # network-bound and can fail long after the expensive part of the run has
    # succeeded; losing a report that cost minutes and real money to an NCBI
    # outage would be a poor trade for a citation check.
    validation_report = None
    if validate_references:
        reference_validator = _build_reference_validator(
            cache_dir=validation_cache_dir,
            email=validation_email,
            full_text=validation_full_text,
            max_references=validation_max_references,
            skip_prefix=validation_skip_prefix,
            rate_limit_delay=validation_rate_limit_delay,
            check_relevance=validation_relevance,
        )
        logger.info("Validating references...")
        try:
            validation_report = reference_validator.validate_result(result)
        except (OSError, ValueError) as exc:
            # urllib raises OSError subclasses for network failures. The report is
            # already saved, so report the real cause rather than letting it surface
            # as a filesystem error.
            logger.error(f"Reference validation failed: {exc}")
            logger.debug("Exception details:", exc_info=True)
            raise typer.Exit(3)

        _echo_validation_summary(validation_report)

    term_report = None
    if validate_terms:
        term_validator = _build_term_validator(
            adapter=term_adapter,
            cache_dir=term_cache_dir,
            oak_config=term_oak_config,
            offline=term_offline,
            max_terms=term_max_terms,
            skip_prefix=term_skip_prefix,
            check_labels=term_labels,
        )
        logger.info("Validating ontology terms...")
        try:
            term_report = term_validator.validate_result(result)
        except (OSError, ValueError) as exc:
            logger.error(f"Term validation failed: {exc}")
            logger.debug("Exception details:", exc_info=True)
            raise typer.Exit(3)

        _echo_term_validation_summary(term_report)

    validated_content = processor.format_research_result(
        result,
        separate_citations=should_separate_citations,
        reference_validation=validation_report,
        term_validation=term_report,
    )
    if output:
        try:
            output.write_text(validated_content, encoding='utf-8')
        except OSError as exc:
            # The report without its validation sections is already on disk, so
            # this loses the sections, not the research.
            logger.error(f"Could not add the validation section to {output}: {exc}")
            logger.debug("Exception details:", exc_info=True)
            raise typer.Exit(1)
        logger.info(f"Validation results added to: {output}")
    else:
        for report in (validation_report, term_report):
            if report is not None:
                typer.echo("\n" + "=" * 60)
                typer.echo(report.to_markdown())

    if not fail_on_unresolved:
        return
    if validation_report is not None and validation_report.has_confabulations:
        logger.error("Reference validation found unresolved references or unsupported quotes")
        raise typer.Exit(2)
    if term_report is not None and term_report.has_problems:
        logger.error("Term validation found unresolved or mislabelled terms")
        raise typer.Exit(2)


@app.command(name="validate-references")
def validate_references_command(
    files: Annotated[List[Path], typer.Argument(
        help="Markdown report file(s) to validate")],
    check_quotes: Annotated[bool, typer.Option(
        "--check-quotes/--no-check-quotes",
        help="Also check quoted claims against the text of the reference they cite")] = True,
    check_relevance: Annotated[bool, typer.Option(
        "--check-relevance/--no-check-relevance",
        help="Also weigh each resolved reference against the report's own vocabulary, to flag citations that exist but look off topic (free: no extra lookups)")] = True,
    cache_dir: Annotated[Optional[Path], typer.Option(
        "--cache-dir", help="Directory for cached reference lookups (default: ./references_cache)")] = None,
    email: Annotated[Optional[str], typer.Option(
        "--email", help="Contact email for the NCBI Entrez API (defaults to $NCBI_EMAIL)")] = None,
    full_text: Annotated[bool, typer.Option(
        "--full-text", help="Fetch full text as well as abstracts (~23x slower, better quote checks)")] = False,
    max_references: Annotated[Optional[int], typer.Option(
        "--max-references", min=1, help="Stop after validating this many references per file")] = None,
    skip_prefix: Annotated[Optional[List[str]], typer.Option(
        "--skip-prefix", help="Identifier prefix to report as unverifiable instead of resolving (repeatable)")] = None,
    rate_limit_delay: Annotated[Optional[float], typer.Option(
        "--rate-limit-delay", min=0.0, help="Seconds to wait between lookups (default: 0.5)")] = None,
    in_place: Annotated[bool, typer.Option(
        "--in-place", help="Replace or append the validation section in each input file")] = False,
    output: Annotated[Optional[Path], typer.Option(
        "--output", help="Write the markdown validation report to this file (single input file only)")] = None,
    json_output: Annotated[Optional[Path], typer.Option(
        "--json", help="Write the validation report as JSON to this file (single input file only)")] = None,
    fail_on_unresolved: Annotated[bool, typer.Option(
        "--fail-on-unresolved", help="Exit non-zero if any reference fails to resolve or any quote is unsupported")] = False,
):
    """Check that the references cited in a report actually exist.

    Every PMID and DOI in the report is resolved against PubMed, Crossref and
    DataCite; identifiers that do not resolve are flagged as likely
    confabulations. Quotes attributed to a reference are additionally checked
    against the text of that reference, and every resolved record is weighed
    against the report's own vocabulary so that a citation which exists but is
    about an unrelated subject is flagged too.

    Requires the optional 'validation' extra:
    pip install "deep_research_client[validation]"

    \b
    Examples:
      # Validate a saved report
      deep-research-client validate-references report.md

      # Validate several reports and append the results to each
      deep-research-client validate-references reports/*.md --in-place

      # Fail a pipeline when any citation is fabricated
      deep-research-client validate-references report.md --fail-on-unresolved

      # Existence checks only, no quote checking, capped at 20 references
      deep-research-client validate-references report.md --no-check-quotes --max-references 20
    """
    from .markdown_parser import parse_frontmatter
    from .validation import strip_validation_section

    if not files:
        logger.error("Provide at least one markdown file to validate")
        raise typer.Exit(1)

    if len(files) > 1 and (output or json_output):
        logger.error("--output and --json require exactly one input file")
        raise typer.Exit(1)

    missing = [f for f in files if not f.is_file()]
    if missing:
        for path in missing:
            logger.error(f"File not found: {path}")
        raise typer.Exit(1)

    validator = _build_reference_validator(
        cache_dir=cache_dir,
        email=email,
        full_text=full_text,
        max_references=max_references,
        skip_prefix=skip_prefix,
        rate_limit_delay=rate_limit_delay,
        check_relevance=check_relevance,
    )

    any_problems = False

    for path in files:
        content = strip_validation_section(path.read_text(encoding="utf-8"))
        # Scan the whole body rather than the Output section alone: identifiers
        # routinely appear in the Citations section and in provider-specific
        # sections that sit alongside it.
        frontmatter, body = parse_frontmatter(content)

        logger.info(f"Validating references in {path}")
        try:
            report = validator.validate_markdown(body, check_quotes=check_quotes)
        except (OSError, ValueError) as exc:
            # urllib raises OSError subclasses for network failures. Reported as
            # what it is, rather than as a filesystem problem or a traceback.
            # OSError covers network failures (urllib raises subclasses of it);
            # ValueError covers a malformed cached record. Neither should reach
            # the user as a traceback when every neighbouring path exits cleanly.
            logger.error(f"Reference validation failed: {exc}")
            logger.debug("Exception details:", exc_info=True)
            raise typer.Exit(3)
        _echo_validation_summary(report)
        any_problems = any_problems or report.has_confabulations

        markdown_report = report.to_markdown()

        try:
            if in_place:
                updated = _refresh_validation_frontmatter(
                    content, frontmatter, report.summary()
                )
                path.write_text(updated.rstrip() + "\n\n" + markdown_report, encoding="utf-8")
                logger.info(f"Wrote validation section to {path}")

            if output:
                output.write_text(markdown_report, encoding="utf-8")
                logger.info(f"Validation report written to {output}")

            if json_output:
                json_output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
                logger.info(f"Validation report written to {json_output}")
        except OSError as exc:
            logger.error(f"Filesystem error: {exc}")
            logger.debug("Exception details:", exc_info=True)
            raise typer.Exit(1)

        if not in_place and not output and not json_output:
            typer.echo(markdown_report)

    if fail_on_unresolved and any_problems:
        logger.error("Reference validation found unresolved references or unsupported quotes")
        raise typer.Exit(2)


@app.command(name="validate-terms")
def validate_terms_command(
    files: Annotated[List[Path], typer.Argument(
        help="Markdown report file(s) to validate")],
    check_labels: Annotated[bool, typer.Option(
        "--check-labels/--no-check-labels",
        help="Also compare the label written beside each CURIE with the term's own label (free: no extra lookups)")] = True,
    adapter: Annotated[Optional[str], typer.Option(
        "--adapter", help="OAK adapter to resolve terms through (default: ols:; use sqlite:obo: to download each ontology once and answer locally)")] = None,
    oak_config: Annotated[Optional[Path], typer.Option(
        "--oak-config", help="oak_config.yaml mapping prefixes to adapters, for ontologies the default adapter does not serve")] = None,
    cache_dir: Annotated[Optional[Path], typer.Option(
        "--cache-dir", help="Directory for cached term labels (default: ./terms_cache)")] = None,
    offline: Annotated[bool, typer.Option(
        "--offline", help="Resolve only from the label cache, never reaching the network; uncached terms are reported as unverifiable")] = False,
    max_terms: Annotated[Optional[int], typer.Option(
        "--max-terms", min=1, help="Stop after validating this many terms per file")] = None,
    skip_prefix: Annotated[Optional[List[str]], typer.Option(
        "--skip-prefix", help="CURIE prefix to report as unverifiable instead of resolving (repeatable)")] = None,
    in_place: Annotated[bool, typer.Option(
        "--in-place", help="Replace or append the term validation section in each input file")] = False,
    output: Annotated[Optional[Path], typer.Option(
        "--output", help="Write the markdown validation report to this file (single input file only)")] = None,
    json_output: Annotated[Optional[Path], typer.Option(
        "--json", help="Write the validation report as JSON to this file (single input file only)")] = None,
    fail_on_unresolved: Annotated[bool, typer.Option(
        "--fail-on-unresolved", help="Exit non-zero if any term fails to resolve or is named as a different term")] = False,
):
    """Check that the ontology terms cited in a report are the terms it names.

    Every CURIE in the report is resolved through OAK, and the label the report
    wrote beside it is compared with the term's own label. Identifiers that do
    not resolve are flagged as likely confabulations; identifiers that resolve to
    a term the report calls something else are flagged too, which existence
    checking alone cannot see - NCIT:C16814 is a real term, and it means
    Malaysia.

    Requires the optional 'terms' extra:
    pip install "deep_research_client[terms]"

    \b
    Examples:
      # Validate a saved report
      deep-research-client validate-terms report.md

      # Validate several reports and append the results to each
      deep-research-client validate-terms reports/*.md --in-place

      # Fail a pipeline when a term is invented or mislabelled
      deep-research-client validate-terms report.md --fail-on-unresolved

      # Bulk work: download each ontology once, then answer locally
      deep-research-client validate-terms reports/*.md --adapter sqlite:obo:
    """
    from .markdown_parser import parse_frontmatter
    from .validation import strip_validation_section

    if not files:
        logger.error("Provide at least one markdown file to validate")
        raise typer.Exit(1)

    if len(files) > 1 and (output or json_output):
        logger.error("--output and --json require exactly one input file")
        raise typer.Exit(1)

    missing = [f for f in files if not f.is_file()]
    if missing:
        for path in missing:
            logger.error(f"File not found: {path}")
        raise typer.Exit(1)

    validator = _build_term_validator(
        adapter=adapter,
        cache_dir=cache_dir,
        oak_config=oak_config,
        offline=offline,
        max_terms=max_terms,
        skip_prefix=skip_prefix,
        check_labels=check_labels,
    )

    any_problems = False

    for path in files:
        content = strip_validation_section(path.read_text(encoding="utf-8"))
        frontmatter, body = parse_frontmatter(content)

        logger.info(f"Validating terms in {path}")
        try:
            report = validator.validate_markdown(body)
        except (OSError, ValueError) as exc:
            # OSError covers network failures (urllib raises subclasses of it);
            # ValueError covers a malformed cached record. Neither should reach
            # the user as a traceback when every neighbouring path exits cleanly.
            logger.error(f"Term validation failed: {exc}")
            logger.debug("Exception details:", exc_info=True)
            raise typer.Exit(3)
        _echo_term_validation_summary(report)
        any_problems = any_problems or report.has_problems

        markdown_report = report.to_markdown()

        try:
            if in_place:
                updated = _refresh_validation_frontmatter(
                    content, frontmatter, report.summary(), key="term_validation"
                )
                path.write_text(updated.rstrip() + "\n\n" + markdown_report, encoding="utf-8")
                logger.info(f"Wrote term validation section to {path}")

            if output:
                output.write_text(markdown_report, encoding="utf-8")
                logger.info(f"Validation report written to {output}")

            if json_output:
                json_output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
                logger.info(f"Validation report written to {json_output}")
        except OSError as exc:
            logger.error(f"Filesystem error: {exc}")
            logger.debug("Exception details:", exc_info=True)
            raise typer.Exit(1)

        if not in_place and not output and not json_output:
            typer.echo(markdown_report)

    if fail_on_unresolved and any_problems:
        logger.error("Term validation found unresolved or mislabelled terms")
        raise typer.Exit(2)


@app.command()
def edison_trajectory(
    trajectory_id: Annotated[str, typer.Argument(help="Existing Edison trajectory/task ID")],
    output: Annotated[Optional[Path], typer.Option(
        help="Output file path (prints to stdout if not provided)")] = None,
    separate_citations: Annotated[Optional[Path], typer.Option(
        "--separate-citations", help="Save citations to separate file (optional path, defaults to output.citations.md)")] = None,
):
    """Retrieve an existing Edison trajectory report and artifacts by ID."""
    from .models import ProviderConfig
    from .providers.falcon import FalconProvider

    api_key = os.getenv("EDISON_API_KEY") or os.getenv("FUTUREHOUSE_API_KEY")
    if not api_key:
        logger.error("EDISON_API_KEY is required to retrieve an Edison trajectory")
        raise typer.Exit(1)

    processor = ResearchProcessor()
    provider = FalconProvider(ProviderConfig(name="falcon", api_key=api_key, enabled=True))

    try:
        result = provider.retrieve_trajectory(trajectory_id)
        should_separate_citations = separate_citations is not None

        if output:
            _write_result_artifacts(result, output)

        output_content = processor.format_research_result(
            result,
            separate_citations=should_separate_citations,
        )

        if output:
            output.write_text(output_content, encoding="utf-8")
            logger.info(f"Result saved to: {output}")

            if should_separate_citations and result.citations:
                if isinstance(separate_citations, Path):
                    citations_output = separate_citations
                else:
                    citations_output = output.with_suffix('.citations.md')

                citations_content = processor.format_citations_only(result)
                citations_output.write_text(citations_content, encoding="utf-8")
                logger.info(f"Citations saved to: {citations_output}")

            if result.citations:
                logger.info(f"Found {len(result.citations)} citations")
            if result.artifacts:
                logger.info(f"Recovered {len(result.artifacts)} artifacts")
        else:
            typer.echo("\n" + "=" * 60)
            typer.echo(output_content)
            if should_separate_citations and result.citations:
                typer.echo("\n" + "=" * 60)
                typer.echo("CITATIONS:")
                typer.echo("=" * 60)
                typer.echo(processor.format_citations_only(result))

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug("Exception details:", exc_info=True)
        raise typer.Exit(1)


@app.command()
def providers(
    show_params: Annotated[bool, typer.Option(
        "--show-params", help="Show available parameters for each provider")] = False,
    provider: Annotated[Optional[str], typer.Option(
        help="Show details for specific provider only")] = None,
    check: Annotated[bool, typer.Option(
        "--check",
        help="Probe each configured provider with a cheap live call to see if it is reachable")] = False,
):
    """List available research providers and their parameters."""
    from .provider_params import PROVIDER_PARAMS_REGISTRY

    logger.debug("Initializing client to check providers")
    client = DeepResearchClient()

    if check:
        if show_params:
            # Echoed, not logged: every other line this path emits goes to
            # stdout, and the CLI's logger does not propagate to handlers a
            # caller (or a test) can see.
            typer.echo("Note: --show-params has no effect with --check; ignoring it.")
        _check_provider_health(client, provider)
        return

    available = client.get_available_providers()

    if provider:
        # Show details for specific provider
        if provider not in PROVIDER_PARAMS_REGISTRY:
            logger.error(f"Unknown provider: {provider}")
            logger.error(
                f"Available providers: {', '.join(PROVIDER_PARAMS_REGISTRY.keys())}")
            raise typer.Exit(1)

        is_available = provider in available
        if is_available:
            status = "Available"
        elif provider in PROVIDER_STUB_HINTS:
            # A stub is not credential-blocked; no key would make it work.
            status = "Not available (stub - no upstream API yet)"
        else:
            status = "Not available (missing API key)"
        typer.echo(f"Provider: {provider} - {status}")

        if not is_available:
            if provider in PROVIDER_STUB_HINTS:
                typer.echo(f"Status: {PROVIDER_STUB_HINTS[provider]}")
            elif provider in PROVIDER_CREDENTIAL_HINTS:
                # Show required environment variable
                env_var = PROVIDER_CREDENTIAL_HINTS[provider][0]
                typer.echo(f"Required: {env_var}")

        # Show parameters
        params_class = PROVIDER_PARAMS_REGISTRY[provider]
        typer.echo(f"\nAvailable parameters for {provider}:")
        for field_name, field_info in params_class.model_fields.items():
            if field_name == "model":
                continue  # Skip the base model field

            default_val = field_info.default
            if hasattr(default_val, '__name__'):  # It's a function/factory
                default_str = "(default factory)"
            elif default_val is None:
                default_str = "(optional)"
            else:
                default_str = f"(default: {default_val})"

            typer.echo(
                f"  --param {field_name}=VALUE  {field_info.description} {default_str}")

        return

    if available:
        logger.info(f"Found {len(available)} available providers")
        typer.echo("Available providers:")
        for prov in available:
            typer.echo(f"  {prov}")

        if show_params:
            typer.echo("\nProvider parameters (use --param key=value):")
            for prov in available:
                if prov in PROVIDER_PARAMS_REGISTRY:
                    params_class = PROVIDER_PARAMS_REGISTRY[prov]
                    typer.echo(f"\n  {prov}:")
                    for field_name, field_info in params_class.model_fields.items():
                        if field_name == "model":
                            continue
                        typer.echo(
                            f"    {field_name}: {field_info.description}")

        missing_credential_providers = [
            provider_name
            for provider_name in PROVIDER_CREDENTIAL_HINTS
            if provider_name not in available
        ]
        if missing_credential_providers:
            typer.echo("\nUnavailable providers requiring credentials:")
            _echo_credential_hints(missing_credential_providers)

        _echo_stub_hints()
    else:
        logger.error("No providers available. Please set API keys:")
        _echo_credential_hints(_settable_credential_hints())
        _echo_stub_hints()

    if not show_params and not provider:
        typer.echo(
            "\nUse --show-params to see available parameters for each provider")
        typer.echo(
            "Use --provider <name> to see detailed info for a specific provider")


@app.command()
def clear_cache():
    """Clear all cached research results."""
    logger.debug("Clearing cache")
    client = DeepResearchClient()
    count = client.clear_cache()
    logger.info(f"Cleared {count} cached files")


def _format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def _format_cache_entry(info: dict, detailed: bool = False, show_query: bool = True) -> str:
    """Format a cache entry for display."""
    from datetime import datetime

    lines = []

    # Basic line: filename with provider tag
    provider = info.get("provider", "?")
    name = info["name"]

    # Format modified time
    modified_ts = info.get("modified", 0)
    modified_dt = datetime.fromtimestamp(modified_ts)
    modified_str = modified_dt.strftime("%Y-%m-%d %H:%M")

    size_str = _format_size(info.get("size_bytes", 0))

    if detailed:
        lines.append(f"  [{provider}] {name}")
        lines.append(f"    Modified: {modified_str}  Size: {size_str}")

        if show_query:
            query = info.get("query", "")
            if query:
                # Truncate long queries
                if len(query) > 100:
                    query = query[:100] + "..."
                lines.append(f"    Query: {query}")

        model = info.get("model")
        if model:
            lines.append(f"    Model: {model}")

        duration = info.get("duration_seconds")
        if duration:
            lines.append(f"    Duration: {duration:.1f}s")

        citation_count = info.get("citation_count", 0)
        if citation_count:
            lines.append(f"    Citations: {citation_count}")
    else:
        # Compact format: [provider] filename (size, date)
        lines.append(f"  [{provider}] {name} ({size_str}, {modified_str})")

    return "\n".join(lines)


@app.command()
def list_cache(
    detailed: Annotated[bool, typer.Option(
        "--detailed", "-d", help="Show detailed metadata for each entry")] = False,
    provider_filter: Annotated[Optional[str], typer.Option(
        "--provider", "-p", help="Filter by provider name")] = None,
    limit: Annotated[int, typer.Option(
        "--limit", "-n", help="Limit number of results")] = 0,
):
    """List cached research files with metadata.

    \b
    Examples:
      deep-research-client list-cache                    # List all cached files
      deep-research-client list-cache --detailed         # Show detailed metadata
      deep-research-client list-cache --provider openai  # Filter by provider
      deep-research-client list-cache -n 10              # Show only last 10 entries
    """
    logger.debug("Listing cached files")
    client = DeepResearchClient()
    cache_info = client.get_cache_info()

    if not cache_info:
        logger.info("No cached files found")
        return

    # Apply provider filter
    if provider_filter:
        cache_info = [c for c in cache_info if c.get(
            "provider", "").lower() == provider_filter.lower()]
        if not cache_info:
            logger.info(
                f"No cached files found for provider: {provider_filter}")
            return

    # Apply limit
    if limit > 0:
        cache_info = cache_info[:limit]

    # Calculate total size
    total_size = sum(c.get("size_bytes", 0) for c in cache_info)

    typer.echo(
        f"Found {len(cache_info)} cached files ({_format_size(total_size)}) in ~/.deep_research_cache/:")
    typer.echo()

    for info in cache_info:
        typer.echo(_format_cache_entry(info, detailed=detailed))

    if not detailed:
        typer.echo()
        typer.echo("Use --detailed for more metadata")


@app.command()
def search_cache(
    keyword: Annotated[str, typer.Argument(help="Keyword to search for in queries and content")],
    detailed: Annotated[bool, typer.Option(
        "--detailed", "-d", help="Show detailed metadata for each match")] = False,
    query_only: Annotated[bool, typer.Option(
        "--query-only", "-q", help="Only search in queries, not content")] = False,
    context: Annotated[int, typer.Option(
        "--context", "-c", help="Characters of context around matches")] = 60,
    max_snippets: Annotated[int, typer.Option(
        "--max-snippets", "-m", help="Maximum snippets to show per match")] = 3,
    no_snippets: Annotated[bool, typer.Option(
        "--no-snippets", help="Hide match snippets")] = False,
):
    """Search cached research files by keyword.

    Searches in both queries and content (markdown) by default.
    Shows context snippets around each match.

    \b
    Examples:
      deep-research-client search-cache "BRCA1"              # Find entries with snippets
      deep-research-client search-cache "gene" --detailed    # Show detailed matches
      deep-research-client search-cache "AI" --query-only    # Only search query text
      deep-research-client search-cache "CRISPR" -c 100      # More context around matches
      deep-research-client search-cache "mutation" -m 5      # Show up to 5 snippets
    """
    logger.debug(f"Searching cache for: {keyword}")
    client = DeepResearchClient()
    matches = client.search_cache(
        keyword, context_chars=context, max_snippets=max_snippets)

    if not matches:
        logger.info(f"No cached files found matching: {keyword}")
        return

    # Filter to query-only matches if requested
    if query_only:
        matches = [m for m in matches if m.get("match_in_query", False)]
        if not matches:
            logger.info(f"No queries found matching: {keyword}")
            return

    typer.echo(f"Found {len(matches)} cached files matching '{keyword}':")
    typer.echo()

    for info in matches:
        # Show where the match was found
        match_locations = []
        if info.get("match_in_query"):
            match_locations.append("query")
        if info.get("match_in_content"):
            match_locations.append("content")
        match_str = f" [match in: {', '.join(match_locations)}]"

        typer.echo(_format_cache_entry(info, detailed=detailed) + match_str)

        # Show snippets unless disabled
        if not no_snippets:
            query_snippets = info.get("query_snippets", [])
            content_snippets = info.get("content_snippets", [])

            if query_snippets:
                for snippet in query_snippets:
                    typer.echo(f"      [query] {snippet}")

            if content_snippets:
                for snippet in content_snippets:
                    typer.echo(f"      [content] {snippet}")

            if query_snippets or content_snippets:
                typer.echo()  # Blank line between entries with snippets


# Default schema for browse-cache command
BROWSER_SCHEMA = {
    "title": "Deep Research Cache Browser",
    "description": "Browse and filter cached research results",
    "searchPlaceholder": "Search queries...",
    "searchableFields": ["title", "query_preview", "keywords"],
    "facets": [
        {
            "field": "provider",
            "label": "Provider",
            "type": "string",
            "sortBy": "count"
        },
        {
            "field": "model",
            "label": "Model",
            "type": "string",
            "sortBy": "count"
        },
        {
            "field": "title",
            "label": "Title",
            "type": "string",
            "sortBy": "count"
        },
        {
            "field": "citation_count",
            "label": "Citations",
            "type": "integer",
            "sortBy": "count"
        },
        {
            "field": "word_count",
            "label": "Word Count",
            "type": "integer",
            "sortBy": "count"
        },
        {
            "field": "keywords",
            "label": "Keywords",
            "type": "array",
            "sortBy": "count"
        },
        {
            "field": "template_file",
            "label": "Template",
            "type": "string",
            "sortBy": "count"
        },
        {
            "field": "date",
            "label": "Date",
            "type": "string",
            "sortBy": "alphabetical"
        },
    ],
    "displayFields": [
        {"field": "title", "label": "Title", "type": "string"},
        {"field": "query_preview", "label": "Query", "type": "string"},
        {"field": "provider", "label": "Provider", "type": "string"},
        {"field": "model", "label": "Model", "type": "string"},
        {"field": "template_file", "label": "Template", "type": "string"},
        {"field": "date", "label": "Date", "type": "string"},
        {"field": "size_kb", "label": "Size (KB)", "type": "number"},
        {"field": "citation_count", "label": "Citations", "type": "integer"},
        {"field": "word_count", "label": "Words", "type": "integer"},
        {"field": "keywords", "label": "Keywords", "type": "array"},
    ]
}


def _inject_url_handling(index_html_path: Path) -> None:
    """Inject URL type handling into the generated linkml-browser index.html.

    linkml-browser handles 'curie' type but not 'url' type for links.
    This post-processes the HTML to add URL handling after the curie handling.
    """
    content = index_html_path.read_text(encoding='utf-8')

    # Find the curie handling code and add URL handling after it
    curie_handling = """if (fieldConfig.type === 'curie' && value.includes(':')) {
                                // Create hyperlink for any CURIE (Compact URI)
                                const curieUrl = `https://bioregistry.io/${value}`;
                                displayValue = `<a href="${curieUrl}" target="_blank" style="color: #667eea; text-decoration: none; border-bottom: 1px dashed #667eea;">${displayValue}</a>`;
                            }"""

    url_handling = """if (fieldConfig.type === 'curie' && value.includes(':')) {
                                // Create hyperlink for any CURIE (Compact URI)
                                const curieUrl = `https://bioregistry.io/${value}`;
                                displayValue = `<a href="${curieUrl}" target="_blank" style="color: #667eea; text-decoration: none; border-bottom: 1px dashed #667eea;">${displayValue}</a>`;
                            }

                            // Handle URL type fields as clickable links
                            if (fieldConfig.type === 'url' && value) {
                                displayValue = `<a href="${value}" target="_self" style="color: #667eea; text-decoration: none; border-bottom: 1px dashed #667eea;">${fieldConfig.label}</a>`;
                            }"""

    if curie_handling in content:
        content = content.replace(curie_handling, url_handling)
        index_html_path.write_text(content, encoding='utf-8')


def _generate_individual_pages(
    data: list[dict],
    output_dir: Path,
    template_path: Optional[Path] = None
) -> int:
    """Generate individual HTML pages for each cache entry.

    Returns number of pages generated.
    """
    import markdown as md_lib
    from jinja2 import Environment, FileSystemLoader, PackageLoader

    # Setup Jinja2 environment
    if template_path and template_path.exists():
        env = Environment(loader=FileSystemLoader(template_path.parent))
        template = env.get_template(template_path.name)
    else:
        # Use built-in template
        env = Environment(loader=PackageLoader(
            'deep_research_client', 'templates'))
        template = env.get_template('research_result.html.j2')

    # Setup markdown converter
    md_converter = md_lib.Markdown(extensions=[
        'extra',        # Tables, footnotes, etc.
        'codehilite',   # Syntax highlighting
        'toc',          # Table of contents
        'sane_lists',   # Better list handling
    ])

    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for entry in data:
        if "markdown" not in entry:
            continue

        # Convert markdown to HTML
        md_converter.reset()
        content_html = md_converter.convert(entry["markdown"])

        # Render template
        html_content = template.render(
            id=entry.get("id", ""),
            title=entry.get("title", ""),
            query_preview=entry.get("query_preview", ""),
            provider=entry.get("provider", "unknown"),
            model=entry.get("model", "default"),
            date=entry.get("date", ""),
            duration_seconds=entry.get("duration_seconds"),
            citation_count=entry.get("citation_count", 0),
            keywords=entry.get("keywords", []),
            author=entry.get("author", ""),
            filename=entry.get("filename", ""),
            content_html=content_html,
            citations=entry.get("citations", []),
            template_variables=entry.get("template_variables", {}),
            template_file=entry.get("template_file", ""),
            provider_config=entry.get("provider_config", {}),
        )

        # Write HTML file
        page_file = pages_dir / f"{entry['id']}.html"
        page_file.write_text(html_content, encoding='utf-8')
        count += 1

    return count


@app.command()
def browse_cache(
    output_dir: Annotated[Path, typer.Argument(help="Output directory for browser files")],
    title: Annotated[Optional[str], typer.Option(
        "--title", "-t", help="Browser title")] = None,
    description: Annotated[Optional[str], typer.Option(
        "--description", "-d", help="Browser description")] = None,
    force: Annotated[bool, typer.Option(
        "--force", "-f", help="Overwrite existing directory")] = False,
    export_only: Annotated[bool, typer.Option(
        "--export-only", help="Only export JSON data, don't generate browser")] = False,
    no_pages: Annotated[bool, typer.Option(
        "--no-pages", help="Skip generating individual HTML pages")] = False,
    template: Annotated[Optional[Path], typer.Option(
        "--template", help="Custom Jinja2 template for individual pages")] = None,
):
    """Generate a faceted browser from cached research results.

    Requires the 'browser' optional dependency: pip install deep-research-client[browser]

    Creates a standalone HTML browser with facets for provider, model, keywords, etc.
    Also generates individual HTML pages for each research result (unless --no-pages).

    \b
    Examples:
      deep-research-client browse-cache ./browser           # Generate browser + pages
      deep-research-client browse-cache ./browser -f        # Overwrite existing
      deep-research-client browse-cache ./browser -t "My Research"  # Custom title
      deep-research-client browse-cache ./browser --no-pages  # Skip individual pages
      deep-research-client browse-cache ./data --export-only  # Just export JSON
    """
    import json as json_module

    logger.debug("Exporting cache for browser")
    client = DeepResearchClient()

    # Include content if we're generating pages
    include_content = not no_pages
    data = client.export_cache_for_browser(include_content=include_content)

    if not data:
        logger.error("No cached files found to browse")
        raise typer.Exit(1)

    logger.info(f"Found {len(data)} cached research entries")

    # Handle export-only mode
    if export_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        data_file = output_dir / "cache_data.json"
        schema_file = output_dir / "schema.json"

        # Add href links to data (pages will be in pages/ subdirectory)
        for entry in data:
            entry["href"] = f"pages/{entry['id']}.html"
            # Remove full content from export (too large)
            entry.pop("markdown", None)
            entry.pop("citations", None)

        # Write data
        with open(data_file, 'w', encoding='utf-8') as f:
            json_module.dump(data, f, indent=2)
        logger.info(f"Data exported to: {data_file}")

        # Write schema with href in display fields
        schema = BROWSER_SCHEMA.copy()
        if title:
            schema["title"] = title
        if description:
            schema["description"] = description

        with open(schema_file, 'w', encoding='utf-8') as f:
            json_module.dump(schema, f, indent=2)
        logger.info(f"Schema exported to: {schema_file}")

        typer.echo(f"Exported {len(data)} entries to {output_dir}/")
        typer.echo(
            "Use 'linkml-browser deploy' to generate browser from these files")
        return

    # Check for dependencies
    try:
        from linkml_browser import BrowserGenerator  # type: ignore[import-untyped,import-not-found]
        import markdown as md_lib  # noqa: F401
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(
            e) else "linkml-browser or markdown"
        logger.error(f"{missing} not installed. Install with:")
        logger.error("  pip install deep-research-client[browser]")
        logger.error("  # or: uv add deep-research-client[browser]")
        raise typer.Exit(1)

    # Check if output directory exists
    if output_dir.exists() and not force:
        logger.error(f"Output directory exists: {output_dir}")
        logger.error("Use --force to overwrite")
        raise typer.Exit(1)

    # Add href links to data for browser
    for entry in data:
        entry["href"] = f"pages/{entry['id']}.html"

    # Prepare schema with href link
    schema = BROWSER_SCHEMA.copy()
    if title:
        schema["title"] = title
    if description:
        schema["description"] = description

    # Add href to display fields (as first field for clickable link)
    href_field = {"field": "href", "label": "View", "type": "url"}
    schema["displayFields"] = [href_field] + list(schema["displayFields"])

    # Generate browser first (it clears the directory with force=True)
    # We need to generate pages after this, so save the content first
    logger.info("Generating browser...")

    # Create browser data without full content (too large for JS)
    browser_data = []
    for entry in data:
        browser_entry = {k: v for k, v in entry.items(
        ) if k not in ("markdown", "citations")}
        browser_data.append(browser_entry)

    generator = BrowserGenerator(browser_data, schema)
    generator.generate(output_dir=output_dir, force=force)

    # Post-process index.html to add URL handling for clickable links
    _inject_url_handling(output_dir / "index.html")

    # Now generate individual HTML pages (after browser, so pages/ survives)
    pages_count = 0
    if not no_pages:
        logger.info("Generating individual pages...")
        pages_count = _generate_individual_pages(data, output_dir, template)
        logger.info(f"Generated {pages_count} individual pages")

    typer.echo(f"Browser generated at: {output_dir}/")
    if pages_count > 0:
        typer.echo(
            f"Generated {pages_count} individual pages in {output_dir}/pages/")
    typer.echo(f"Open {output_dir}/index.html in a browser to view")


@app.command()
def models(
    provider: Annotated[Optional[str], typer.Option(
        help="Show models for specific provider")] = None,
    cost: Annotated[Optional[str], typer.Option(
        help="Filter by cost level (low, medium, high, very_high)")] = None,
    capability: Annotated[Optional[str], typer.Option(
        help="Filter by capability (web_search, academic_search, etc.)")] = None,
    detailed: Annotated[bool, typer.Option(
        "--detailed", help="Show detailed model information")] = False
):
    """Show available models and their characteristics.

    \b
    Examples:
      deep-research-client models                    # List all models
      deep-research-client models --provider openai # Show OpenAI models
      deep-research-client models --cost low         # Show low-cost models
      deep-research-client models --detailed         # Show detailed information
    """
    if provider:
        # Show models for specific provider
        logger.debug(f"Fetching models for provider: {provider}")
        cards = get_provider_model_cards(provider)
        if not cards:
            logger.error(f"Provider '{provider}' not found")
            raise typer.Exit(1)

        typer.echo(f"**{cards.provider_name.upper()}** Models")
        typer.echo(f"Default: {cards.default_model}")
        typer.echo()

        for model_name, card in cards.models.items():
            _display_model_card(card, detailed)

    elif cost:
        # Filter by cost level
        try:
            cost_level = CostLevel(cost.lower())
        except ValueError:
            logger.error(
                f"Invalid cost level '{cost}'. Use: low, medium, high, very_high")
            raise typer.Exit(1)

        logger.debug(f"Filtering models by cost level: {cost_level}")
        models_by_cost = find_models_by_cost(cost_level)
        if not models_by_cost:
            logger.info(f"No models found with cost level: {cost}")
            return

        typer.echo(f"**{cost.upper()}** Cost Models")
        typer.echo()

        for provider_name, model_cards_list in models_by_cost.items():
            typer.echo(f"**{provider_name.upper()}:**")
            for card in model_cards_list:
                _display_model_card(card, detailed, indent="  ")
            typer.echo()

    elif capability:
        # Filter by capability
        try:
            cap = ModelCapability(capability.lower())
        except ValueError:
            logger.error(
                f"Invalid capability '{capability}'. Use: web_search, academic_search, scientific_literature, etc.")
            raise typer.Exit(1)

        logger.debug(f"Filtering models by capability: {cap}")
        models_by_cap = find_models_by_capability(cap)
        if not models_by_cap:
            logger.info(f"No models found with capability: {capability}")
            return

        typer.echo(
            f"**{capability.upper().replace('_', ' ')}** Capable Models")
        typer.echo()

        for provider_name, model_cards_list in models_by_cap.items():
            typer.echo(f"**{provider_name.upper()}:**")
            for card in model_cards_list:
                _display_model_card(card, detailed, indent="  ")
            typer.echo()

    else:
        # Show all models by provider
        logger.debug("Listing all models")
        all_models = list_all_models()
        typer.echo("**Available Research Models**")
        typer.echo()

        for provider_name, model_names in all_models.items():
            cards = get_provider_model_cards(provider_name)
            if not cards:
                continue
            typer.echo(
                f"**{provider_name.upper()}** (Default: {cards.default_model}):")

            for model_name in model_names:
                maybe_card = cards.get_model_card(model_name)
                if maybe_card:
                    _display_model_card(maybe_card, detailed, indent="  ")
            typer.echo()


def _display_model_card(card, detailed: bool = False, indent: str = ""):
    """Helper function to display a model card."""
    cost_emoji = {
        CostLevel.LOW: "💚",
        CostLevel.MEDIUM: "💛",
        CostLevel.HIGH: "🧡",
        CostLevel.VERY_HIGH: "❤️"
    }

    time_emoji = {
        TimeEstimate.FAST: "⚡",
        TimeEstimate.MEDIUM: "⏳",
        TimeEstimate.SLOW: "🐌",
        TimeEstimate.VERY_SLOW: "🐢"
    }

    cost_icon = cost_emoji.get(card.cost_level, "❓")
    time_icon = time_emoji.get(card.time_estimate, "❓")

    if detailed:
        typer.echo(f"{indent}**{card.display_name}** ({card.name})")
        if card.aliases:
            typer.echo(f"{indent}  Aliases: {', '.join(card.aliases)}")
        typer.echo(f"{indent}  {card.description}")
        typer.echo(f"{indent}  Cost: {cost_icon} {card.cost_level}")
        typer.echo(f"{indent}  Speed: {time_icon} {card.time_estimate}")

        if card.capabilities:
            caps = ", ".join([cap.replace("_", " ").title()
                             for cap in card.capabilities])
            typer.echo(f"{indent}  Capabilities: {caps}")

        if card.context_window:
            typer.echo(f"{indent}  Context: {card.context_window:,} tokens")

        if card.pricing_notes:
            typer.echo(f"{indent}  Pricing: {card.pricing_notes}")

        if card.use_cases:
            typer.echo(f"{indent}  Use Cases: {', '.join(card.use_cases[:3])}")

        typer.echo()
    else:
        aliases_str = f" ({', '.join(card.aliases)})" if card.aliases else ""
        typer.echo(
            f"{indent}**{card.display_name}**{aliases_str} {cost_icon} {time_icon}")
        typer.echo(
            f"{indent}  {card.description[:100]}{'...' if len(card.description) > 100 else ''}")


@app.command()
def browse_files(
    sources: Annotated[List[Path], typer.Argument(help="Directories or files to include")],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory for browser files")],
    pattern: Annotated[str, typer.Option(
        "--pattern", "-p", help="Glob pattern for finding files in directories")] = "**/*.md",
    title: Annotated[Optional[str], typer.Option(
        "--title", "-t", help="Browser title")] = None,
    description: Annotated[Optional[str], typer.Option(
        "--description", help="Browser description")] = None,
    force: Annotated[bool, typer.Option(
        "--force", "-f", help="Overwrite existing directory")] = False,
    export_only: Annotated[bool, typer.Option(
        "--export-only", help="Only export JSON data, don't generate browser")] = False,
    no_pages: Annotated[bool, typer.Option(
        "--no-pages", help="Skip generating individual HTML pages")] = False,
    template: Annotated[Optional[Path], typer.Option(
        "--template", help="Custom Jinja2 template for individual pages")] = None,
    verbose: Annotated[int, typer.Option(
        "-v", "--verbose", count=True, help="Increase verbosity (-v, -vv, -vvv)")] = 0,
):
    """Generate a faceted browser from markdown research files.

    Unlike browse-cache (which uses cached JSON files), this command parses
    markdown files with YAML frontmatter - the output format from 'research' command.

    Requires the 'browser' optional dependency: pip install deep-research-client[browser]

    \b
    Examples:
      # Browse all markdown files in a directory
      deep-research-client browse-files ./research-outputs -o ./browser

      # Use a specific glob pattern
      deep-research-client browse-files ./docs -o ./browser -p "*.md"

      # Browse a single file
      deep-research-client browse-files ./my-research.md -o ./browser

      # Multiple sources (directories and files)
      deep-research-client browse-files ./dir1 ./dir2 ./extra.md -o ./browser

      # Recursively find files matching pattern
      deep-research-client browse-files ./notes -o ./browser -p "research/**/*.md"
    """
    import json as json_module
    from .markdown_parser import parse_markdown_files

    setup_logging(verbose)

    # Collect files from all sources
    all_files: List[Path] = []
    for source in sources:
        if source.is_file():
            if source.suffix.lower() == '.md':
                all_files.append(source)
                logger.info(f"Added file: {source}")
            else:
                logger.warning(f"Skipping non-markdown file: {source}")
        elif source.is_dir():
            found = list(source.glob(pattern))
            logger.info(
                f"Found {len(found)} files in {source} with pattern '{pattern}'")
            all_files.extend(found)
        else:
            logger.warning(f"Source not found, skipping: {source}")

    if not all_files:
        logger.error("No markdown files found")
        raise typer.Exit(1)

    logger.info(f"Processing {len(all_files)} markdown files")
    data = parse_markdown_files(files=all_files, include_content=not no_pages)

    # Handle export-only mode
    if export_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        data_file = output_dir / "files_data.json"
        schema_file = output_dir / "schema.json"

        # Add href links to data (pages will be in pages/ subdirectory)
        for entry in data:
            entry["href"] = f"pages/{entry['id']}.html"
            # Remove full content from export (too large)
            entry.pop("markdown", None)
            entry.pop("citations", None)
            entry.pop("source_path", None)

        # Write data
        with open(data_file, 'w', encoding='utf-8') as f:
            json_module.dump(data, f, indent=2)
        logger.info(f"Data exported to: {data_file}")

        # Write schema
        schema = BROWSER_SCHEMA.copy()
        if title:
            schema["title"] = title
        if description:
            schema["description"] = description

        with open(schema_file, 'w', encoding='utf-8') as f:
            json_module.dump(schema, f, indent=2)
        logger.info(f"Schema exported to: {schema_file}")

        typer.echo(f"Exported {len(data)} entries to {output_dir}/")
        typer.echo(
            "Use 'linkml-browser deploy' to generate browser from these files")
        return

    # Check for dependencies
    try:
        # type: ignore[import-untyped,import-not-found]
        from linkml_browser import BrowserGenerator
        import markdown as md_lib  # noqa: F401
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(
            e) else "linkml-browser or markdown"
        logger.error(f"{missing} not installed. Install with:")
        logger.error("  pip install deep-research-client[browser]")
        logger.error("  # or: uv add deep-research-client[browser]")
        raise typer.Exit(1)

    # Check if output directory exists
    if output_dir.exists() and not force:
        logger.error(f"Output directory exists: {output_dir}")
        logger.error("Use --force to overwrite")
        raise typer.Exit(1)

    # Add href links to data for browser
    for entry in data:
        entry["href"] = f"pages/{entry['id']}.html"

    # Prepare schema with href link
    schema = BROWSER_SCHEMA.copy()
    if title:
        schema["title"] = title
    if description:
        schema["description"] = description

    # Add href to display fields (as first field for clickable link)
    href_field = {"field": "href", "label": "View", "type": "url"}
    schema["displayFields"] = [href_field] + list(schema["displayFields"])

    # Generate browser first (it clears the directory with force=True)
    logger.info("Generating browser...")

    # Create browser data without full content (too large for JS)
    browser_data = []
    for entry in data:
        browser_entry = {k: v for k, v in entry.items() if k not in (
            "markdown", "citations", "source_path")}
        browser_data.append(browser_entry)

    generator = BrowserGenerator(browser_data, schema)
    generator.generate(output_dir=output_dir, force=force)

    # Post-process index.html to add URL handling for clickable links
    _inject_url_handling(output_dir / "index.html")

    # Now generate individual HTML pages (after browser, so pages/ survives)
    pages_count = 0
    if not no_pages:
        logger.info("Generating individual pages...")
        pages_count = _generate_individual_pages(data, output_dir, template)
        logger.info(f"Generated {pages_count} individual pages")

    typer.echo(f"Browser generated at: {output_dir}/")
    if pages_count > 0:
        typer.echo(
            f"Generated {pages_count} individual pages in {output_dir}/pages/")
    typer.echo(f"Open {output_dir}/index.html in a browser to view")


# ---------------------------------------------------------------------------
# Evaluation commands
# ---------------------------------------------------------------------------

eval_app = typer.Typer(help="Evaluate deep research tools against curated ground truth")
app.add_typer(eval_app, name="eval")


@eval_app.command("load-ground-truth")
def eval_load_ground_truth(
    dismech_dir: Annotated[Optional[Path], typer.Option("--dismech-dir", help="Path to dismech kb/disorders/ directory")] = None,
    gene_review_dir: Annotated[Optional[Path], typer.Option("--gene-review-dir", help="Path to ai-gene-review genes/human/ directory")] = None,
    entity_file: Annotated[Optional[List[Path]], typer.Option("--entity-file", help="Specific YAML file(s) to load")] = None,
    entity_name: Annotated[Optional[List[str]], typer.Option("--entity-name", help="Filter by entity name")] = None,
    max_entities: Annotated[Optional[int], typer.Option("--max", help="Maximum number of entities to load")] = None,
):
    """Load and display ground truth entities from dismech and/or ai-gene-review.

    \b
    Examples:
        deep-research-client eval load-ground-truth --dismech-dir /path/to/dismech/kb/disorders
        deep-research-client eval load-ground-truth --entity-file /path/to/Achondroplasia.yaml
        deep-research-client eval load-ground-truth --gene-review-dir /path/to/genes/human --max 5
    """
    from .evaluation.runner import EvalConfig, load_entities, generate_all_tasks

    config = EvalConfig(
        dismech_dir=dismech_dir,
        gene_review_dir=gene_review_dir,
        entity_files=list(entity_file or []),
        entity_names=list(entity_name or []),
        max_entities=max_entities,
    )
    entities = load_entities(config)
    if not entities:
        typer.echo("No entities loaded. Check your paths.")
        raise typer.Exit(1)

    tasks = generate_all_tasks(entities)

    typer.echo(f"\nLoaded {len(entities)} entities, generated {len(tasks)} evaluation tasks:\n")
    for entity in entities:
        typer.echo(f"  {entity.entity_type.upper()}: {entity.name} ({entity.entity_id})")
        typer.echo(f"    Claims: {len(entity.claims)}, References: {len(entity.all_references)}")
        entity_tasks = [t for t in tasks if t.ground_truth_entity_id == entity.entity_id]
        for t in entity_tasks:
            typer.echo(f"    Task: {t.task_type.value} -> {t.query[:80]}...")
    typer.echo()


@eval_app.command("generate-tasks")
def eval_generate_tasks(
    dismech_dir: Annotated[Optional[Path], typer.Option("--dismech-dir", help="Path to dismech kb/disorders/ directory")] = None,
    gene_review_dir: Annotated[Optional[Path], typer.Option("--gene-review-dir", help="Path to ai-gene-review genes/human/ directory")] = None,
    entity_file: Annotated[Optional[List[Path]], typer.Option("--entity-file", help="Specific YAML file(s) to load")] = None,
    entity_name: Annotated[Optional[List[str]], typer.Option("--entity-name", help="Filter by entity name")] = None,
    max_entities: Annotated[Optional[int], typer.Option("--max", help="Maximum number of entities")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file for tasks (JSON)")] = None,
):
    """Generate evaluation tasks as JSON for use in scoring pipelines.

    \b
    Examples:
        deep-research-client eval generate-tasks --entity-file Achondroplasia.yaml -o tasks.json
    """
    import json as json_mod
    from .evaluation.runner import EvalConfig, load_entities, generate_all_tasks

    config = EvalConfig(
        dismech_dir=dismech_dir,
        gene_review_dir=gene_review_dir,
        entity_files=list(entity_file or []),
        entity_names=list(entity_name or []),
        max_entities=max_entities,
    )
    entities = load_entities(config)
    tasks = generate_all_tasks(entities)

    tasks_json = [t.model_dump(mode="json") for t in tasks]

    if output:
        output.write_text(json_mod.dumps(tasks_json, indent=2))
        typer.echo(f"Wrote {len(tasks)} tasks to {output}")
    else:
        typer.echo(json_mod.dumps(tasks_json, indent=2))


@eval_app.command("score")
def eval_score(
    report: Annotated[Path, typer.Argument(help="Markdown file with DR output to score")],
    entity_file: Annotated[Path, typer.Option("--entity-file", help="Ground truth YAML file")],
    provider_name: Annotated[str, typer.Option("--provider", help="Name of the DR provider that generated the report")] = "unknown",
    task_type: Annotated[Optional[str], typer.Option("--task-type", help="Task type filter (disease_mechanism, gene_function, etc.)")] = None,
    no_fact: Annotated[bool, typer.Option("--no-fact", help="Skip FACT scoring")] = False,
    no_recall: Annotated[bool, typer.Option("--no-recall", help="Skip claim recall scoring")] = False,
    no_race: Annotated[bool, typer.Option("--no-race", help="Skip RACE scoring")] = False,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file for results (JSON)")] = None,
    llm_base_url: Annotated[Optional[str], typer.Option("--llm-base-url", help="Base URL for LLM judge API")] = None,
    llm_api_key_env: Annotated[str, typer.Option("--llm-api-key-env", help="Env var for LLM judge API key")] = "OPENAI_API_KEY",
    llm_model: Annotated[str, typer.Option("--llm-model", help="Model for LLM judge")] = "gpt-4o-mini",
):
    """Score a deep research report against ground truth.

    \b
    Examples:
        deep-research-client eval score report.md --entity-file Achondroplasia.yaml --provider falcon
        deep-research-client eval score report.md --entity-file BRCA1-ai-review.yaml --no-race
    """
    import asyncio
    import json as json_mod
    from openai import AsyncOpenAI
    from .evaluation.runner import (
        EvalConfig, load_entities, generate_all_tasks, parse_dr_output, score_output,
    )
    from .evaluation.models import TaskType

    # Load ground truth and generate tasks
    config = EvalConfig(
        entity_files=[entity_file],
        run_fact=not no_fact,
        run_claim_recall=not no_recall,
        run_race=not no_race,
    )
    entities = load_entities(config)
    if not entities:
        typer.echo("No entities loaded from the ground truth file.")
        raise typer.Exit(1)

    tasks = generate_all_tasks(entities)
    if task_type:
        try:
            tt = TaskType(task_type)
            tasks = [t for t in tasks if t.task_type == tt]
        except ValueError:
            typer.echo(f"Unknown task type: {task_type}. Valid: {[t.value for t in TaskType]}")
            raise typer.Exit(1)

    if not tasks:
        typer.echo("No evaluation tasks generated for the given entity/task-type.")
        raise typer.Exit(1)

    # Read the report
    markdown_text = report.read_text()

    # Set up LLM judge client
    api_key = os.environ.get(llm_api_key_env, "")
    if not api_key:
        typer.echo(f"Warning: {llm_api_key_env} not set. LLM-based scoring will fail.")
    llm_client = AsyncOpenAI(api_key=api_key, base_url=llm_base_url)
    config.llm_model = llm_model

    async def _run():
        results = []
        for eval_task in tasks:
            dr_output = parse_dr_output(eval_task, markdown_text, provider_name)
            result = await score_output(dr_output, eval_task, llm_client, config)
            results.append(result)
        return results

    results = asyncio.run(_run())

    # Display results
    for r in results:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Task: {r.task_id} | Provider: {r.provider}")
        if r.fact_score:
            typer.echo(f"  FACT: accuracy={r.fact_score.citation_accuracy:.2f}, "
                       f"effective_citations={r.fact_score.effective_citations}/{r.fact_score.total_citations}")
        if r.claim_recall_score:
            typer.echo(f"  Claim Recall: {r.claim_recall_score.claim_recall:.2f} "
                       f"({r.claim_recall_score.matched_claims}/{r.claim_recall_score.total_ground_truth_claims})")
        if r.race_score:
            typer.echo(f"  RACE: overall={r.race_score.overall_score:.2f}")
            for d in r.race_score.dimensions:
                typer.echo(f"    {d.dimension}: {d.score:.1f}/5")
        if r.intrinsic_score:
            isc = r.intrinsic_score
            if isc.citation_verifiability:
                cv = isc.citation_verifiability
                typer.echo(f"  Citation Verifiability: {cv.verified_exist}/{cv.total_citations} "
                           f"({cv.verifiability:.2f})")
                if cv.median_year:
                    typer.echo(f"    Median citation year: {cv.median_year}")
            if isc.citation_alignment:
                ca = isc.citation_alignment
                typer.echo(f"  Citation-Claim Alignment: {ca.aligned_count}/{ca.total_checked} "
                           f"({ca.alignment_rate:.2f})")
            if isc.factual_spot_checks:
                sc = isc.factual_spot_checks
                typer.echo(f"  Factual Spot Checks: {sc.correct_count}/{sc.total_checks} correct, "
                           f"{sc.present_count}/{sc.total_checks} present")
            if isc.topic_coverage:
                tc = isc.topic_coverage
                typer.echo(f"  Topic Coverage: {tc.covered_count}/{tc.total_topics} "
                           f"({tc.coverage_rate:.2f})")
        if r.error:
            typer.echo(f"  Errors: {r.error}")

    # Save JSON output
    if output:
        results_json = [r.model_dump(mode="json") for r in results]
        output.write_text(json_mod.dumps(results_json, indent=2))
        typer.echo(f"\nResults written to {output}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
