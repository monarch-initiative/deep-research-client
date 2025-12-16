"""CLI interface for deep-research-client."""

import logging
import os
import typer
from pathlib import Path
from typing import Optional, List
from typing_extensions import Annotated

from .client import DeepResearchClient
from .processing import ResearchProcessor
from .model_cards import (
    get_provider_model_cards,
    list_all_models,
    find_models_by_cost,
    find_models_by_capability,
    CostLevel,
    TimeEstimate,
    ModelCapability
)

# Configure logging
logger = logging.getLogger("deep_research_client")

app = typer.Typer(help="deep-research-client: Wrapper for multiple deep research tools")


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
        logger.debug(f"Logging configured at {logging.getLevelName(level)} level")


@app.callback()
def main_callback(
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity (-v, -vv, -vvv)")] = 0,
):
    """Global options for all commands."""
    setup_logging(verbose)


@app.command()
def research(
    query: Annotated[Optional[str], typer.Argument(help="Research query or question (not needed if using --template)")] = None,
    provider: Annotated[Optional[str], typer.Option(help="Specific provider to use (openai, falcon, perplexity, consensus, mock)")] = None,
    model: Annotated[Optional[str], typer.Option(help="Model to use for the provider (overrides provider default)")] = None,
    output: Annotated[Optional[Path], typer.Option(help="Output file path (prints to stdout if not provided)")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable caching")] = False,
    separate_citations: Annotated[Optional[Path], typer.Option("--separate-citations", help="Save citations to separate file (optional path, defaults to output.citations.md)")] = None,
    cache_dir: Annotated[Optional[Path], typer.Option("--cache-dir", help="Override cache directory (default: ~/.deep_research_cache)")] = None,
    template: Annotated[Optional[Path], typer.Option(help="Template file with {variable} placeholders")] = None,
    input_file: Annotated[Optional[Path], typer.Option("--input-file", "-f", help="Read the research query from a text/markdown file")] = None,
    var: Annotated[Optional[List[str]], typer.Option(help="Template variable as 'key=value' (can be used multiple times)")] = None,
    param: Annotated[Optional[List[str]], typer.Option(help="Provider-specific parameter as 'key=value' (can be used multiple times)")] = None,
    base_url: Annotated[Optional[str], typer.Option("--base-url", help="Custom base URL for API endpoint (for proxies or OpenAI-compatible services)")] = None,
    use_cborg: Annotated[bool, typer.Option("--use-cborg", help="Use CBORG proxy (Berkeley Lab's AI Portal at api.cborg.lbl.gov)")] = False,
    api_key_env: Annotated[Optional[str], typer.Option("--api-key-env", help="Environment variable name to use for API key (e.g., 'CBORG_API_KEY')")] = None,
    # Publication-style metadata options
    title: Annotated[Optional[str], typer.Option("--title", help="Title for the research report")] = None,
    abstract: Annotated[Optional[str], typer.Option("--abstract", help="Abstract or summary for the research")] = None,
    keyword: Annotated[Optional[List[str]], typer.Option("--keyword", help="Keyword/tag for the research (can be used multiple times)")] = None,
    author: Annotated[Optional[str], typer.Option("--author", help="Primary author of the research")] = None,
    contributor: Annotated[Optional[List[str]], typer.Option("--contributor", help="Contributor to the research (can be used multiple times)")] = None,
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
            logger.error("Provide the query either as an argument or via --input-file, not both")
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
            is_valid, error_msg = processor.validate_template_variables(template, var)
            if not is_valid:
                logger.error(f"Template error: {error_msg}")
                if error_msg and "requires variables" in error_msg:
                    logger.error("Use --var key=value for each variable")
                raise typer.Exit(1)

            # Process the template
            query, template_info = processor.process_template_file(template, var)

            logger.info(f"Using template: {template.name}")
            if template_info['template_variables']:
                var_str = ', '.join(f"{k}={v}" for k, v in template_info['template_variables'].items())
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
                    raise ValueError(f"Invalid parameter format: '{param_str}'. Use 'key=value'")
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

    # Handle proxy/endpoint configuration
    proxy_base_url = None
    proxy_api_key_env = api_key_env

    # --use-cborg is a shortcut for CBORG configuration
    if use_cborg:
        if base_url:
            logger.warning("--use-cborg overrides --base-url")
        proxy_base_url = "https://api.cborg.lbl.gov"
        # Default to CBORG_API_KEY if no specific env var is provided
        if not proxy_api_key_env:
            proxy_api_key_env = "CBORG_API_KEY"
        logger.info(f"Using CBORG proxy at {proxy_base_url}")
    elif base_url:
        proxy_base_url = base_url
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
                logger.error(f"Environment variable {proxy_api_key_env} not set")
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
    client = DeepResearchClient(cache_config=cache_config, provider_configs=provider_configs)

    # Check if any providers are available
    available_providers = client.get_available_providers()
    if not available_providers:
        logger.error("No research providers available. Please set API keys:")
        logger.error("  - OPENAI_API_KEY for OpenAI Deep Research")
        logger.error("  - EDISON_API_KEY for Edison Scientific")
        logger.error("  - PERPLEXITY_API_KEY for Perplexity AI")
        raise typer.Exit(1)

    # Show available providers
    if provider:
        if provider not in available_providers:
            logger.error(f"Provider '{provider}' not available. Available: {', '.join(available_providers)}")
            raise typer.Exit(1)
        logger.info(f"Using provider: {provider}")
    else:
        logger.info(f"Available providers: {', '.join(available_providers)}")
        logger.info(f"Using: {available_providers[0]}")

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

    logger.info("Researching...")

    try:
        # Perform research
        logger.debug(f"Starting research with query: {query[:100]}...")
        result = client.research(query, provider, template_info, model, provider_params, metadata)

        # Show cache status
        if result.cached:
            logger.info("Result retrieved from cache")
        else:
            logger.info(f"Research completed using {result.provider}")

        # Determine if we're separating citations
        should_separate_citations = separate_citations is not None

        # Format output using processor
        logger.debug("Formatting research result")
        output_content = processor.format_research_result(result, separate_citations=should_separate_citations)

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
                citations_output.write_text(citations_content, encoding='utf-8')
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

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.debug("Exception details:", exc_info=True)
        raise typer.Exit(1)


@app.command()
def providers(
    show_params: Annotated[bool, typer.Option("--show-params", help="Show available parameters for each provider")] = False,
    provider: Annotated[Optional[str], typer.Option(help="Show details for specific provider only")] = None,
):
    """List available research providers and their parameters."""
    from .provider_params import PROVIDER_PARAMS_REGISTRY

    logger.debug("Initializing client to check providers")
    client = DeepResearchClient()
    available = client.get_available_providers()

    if provider:
        # Show details for specific provider
        if provider not in PROVIDER_PARAMS_REGISTRY:
            logger.error(f"Unknown provider: {provider}")
            logger.error(f"Available providers: {', '.join(PROVIDER_PARAMS_REGISTRY.keys())}")
            raise typer.Exit(1)

        is_available = provider in available
        status = "Available" if is_available else "Not available (missing API key)"
        typer.echo(f"Provider: {provider} - {status}")

        if not is_available:
            # Show required environment variable
            env_vars = {
                "openai": "OPENAI_API_KEY",
                "falcon": "EDISON_API_KEY",
                "perplexity": "PERPLEXITY_API_KEY",
                "consensus": "CONSENSUS_API_KEY",
                "mock": "ENABLE_MOCK_PROVIDER=true"
            }
            if provider in env_vars:
                typer.echo(f"Required: {env_vars[provider]}")

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

            typer.echo(f"  --param {field_name}=VALUE  {field_info.description} {default_str}")

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
                        typer.echo(f"    {field_name}: {field_info.description}")
    else:
        logger.error("No providers available. Please set API keys:")
        typer.echo("  - OPENAI_API_KEY for OpenAI Deep Research")
        typer.echo("  - EDISON_API_KEY for Edison Scientific")
        typer.echo("  - PERPLEXITY_API_KEY for Perplexity AI")
        typer.echo("  - CONSENSUS_API_KEY for Consensus")
        typer.echo("  - ENABLE_MOCK_PROVIDER=true for Mock provider")

    if not show_params and not provider:
        typer.echo("\nUse --show-params to see available parameters for each provider")
        typer.echo("Use --provider <name> to see detailed info for a specific provider")


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
    detailed: Annotated[bool, typer.Option("--detailed", "-d", help="Show detailed metadata for each entry")] = False,
    provider_filter: Annotated[Optional[str], typer.Option("--provider", "-p", help="Filter by provider name")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Limit number of results")] = 0,
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
        cache_info = [c for c in cache_info if c.get("provider", "").lower() == provider_filter.lower()]
        if not cache_info:
            logger.info(f"No cached files found for provider: {provider_filter}")
            return

    # Apply limit
    if limit > 0:
        cache_info = cache_info[:limit]

    # Calculate total size
    total_size = sum(c.get("size_bytes", 0) for c in cache_info)

    typer.echo(f"Found {len(cache_info)} cached files ({_format_size(total_size)}) in ~/.deep_research_cache/:")
    typer.echo()

    for info in cache_info:
        typer.echo(_format_cache_entry(info, detailed=detailed))

    if not detailed:
        typer.echo()
        typer.echo("Use --detailed for more metadata")


@app.command()
def search_cache(
    keyword: Annotated[str, typer.Argument(help="Keyword to search for in queries and content")],
    detailed: Annotated[bool, typer.Option("--detailed", "-d", help="Show detailed metadata for each match")] = False,
    query_only: Annotated[bool, typer.Option("--query-only", "-q", help="Only search in queries, not content")] = False,
    context: Annotated[int, typer.Option("--context", "-c", help="Characters of context around matches")] = 60,
    max_snippets: Annotated[int, typer.Option("--max-snippets", "-m", help="Maximum snippets to show per match")] = 3,
    no_snippets: Annotated[bool, typer.Option("--no-snippets", help="Hide match snippets")] = False,
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
    matches = client.search_cache(keyword, context_chars=context, max_snippets=max_snippets)

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
        env = Environment(loader=PackageLoader('deep_research_client', 'templates'))
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
    title: Annotated[Optional[str], typer.Option("--title", "-t", help="Browser title")] = None,
    description: Annotated[Optional[str], typer.Option("--description", "-d", help="Browser description")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite existing directory")] = False,
    export_only: Annotated[bool, typer.Option("--export-only", help="Only export JSON data, don't generate browser")] = False,
    no_pages: Annotated[bool, typer.Option("--no-pages", help="Skip generating individual HTML pages")] = False,
    template: Annotated[Optional[Path], typer.Option("--template", help="Custom Jinja2 template for individual pages")] = None,
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
        typer.echo("Use 'linkml-browser deploy' to generate browser from these files")
        return

    # Check for dependencies
    try:
        from linkml_browser import BrowserGenerator  # type: ignore[import-untyped,import-not-found]
        import markdown as md_lib  # noqa: F401
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else "linkml-browser or markdown"
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
        browser_entry = {k: v for k, v in entry.items() if k not in ("markdown", "citations")}
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
        typer.echo(f"Generated {pages_count} individual pages in {output_dir}/pages/")
    typer.echo(f"Open {output_dir}/index.html in a browser to view")


@app.command()
def models(
    provider: Annotated[Optional[str], typer.Option(help="Show models for specific provider")] = None,
    cost: Annotated[Optional[str], typer.Option(help="Filter by cost level (low, medium, high, very_high)")] = None,
    capability: Annotated[Optional[str], typer.Option(help="Filter by capability (web_search, academic_search, etc.)")] = None,
    detailed: Annotated[bool, typer.Option("--detailed", help="Show detailed model information")] = False
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
            logger.error(f"Invalid cost level '{cost}'. Use: low, medium, high, very_high")
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
            logger.error(f"Invalid capability '{capability}'. Use: web_search, academic_search, scientific_literature, etc.")
            raise typer.Exit(1)

        logger.debug(f"Filtering models by capability: {cap}")
        models_by_cap = find_models_by_capability(cap)
        if not models_by_cap:
            logger.info(f"No models found with capability: {capability}")
            return

        typer.echo(f"**{capability.upper().replace('_', ' ')}** Capable Models")
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
            typer.echo(f"**{provider_name.upper()}** (Default: {cards.default_model}):")

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
            caps = ", ".join([cap.replace("_", " ").title() for cap in card.capabilities])
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
        typer.echo(f"{indent}**{card.display_name}**{aliases_str} {cost_icon} {time_icon}")
        typer.echo(f"{indent}  {card.description[:100]}{'...' if len(card.description) > 100 else ''}")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
