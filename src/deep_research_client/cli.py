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
    var: Annotated[Optional[List[str]], typer.Option(help="Template variable as 'key=value' (can be used multiple times)")] = None,
    param: Annotated[Optional[List[str]], typer.Option(help="Provider-specific parameter as 'key=value' (can be used multiple times)")] = None,
    base_url: Annotated[Optional[str], typer.Option("--base-url", help="Custom base URL for API endpoint (for proxies or OpenAI-compatible services)")] = None,
    use_cborg: Annotated[bool, typer.Option("--use-cborg", help="Use CBORG proxy (Berkeley Lab's AI Portal at api.cborg.lbl.gov)")] = False,
    api_key_env: Annotated[Optional[str], typer.Option("--api-key-env", help="Environment variable name to use for API key (e.g., 'CBORG_API_KEY')")] = None,
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

    logger.info("Researching...")

    try:
        # Perform research
        logger.debug(f"Starting research with query: {query[:100]}...")
        result = client.research(query, provider, template_info, model, provider_params)

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


@app.command()
def list_cache():
    """List cached research files."""
    logger.debug("Listing cached files")
    client = DeepResearchClient()
    cached_files = client.list_cached_files()

    if not cached_files:
        logger.info("No cached files found")
        return

    logger.info(f"Found {len(cached_files)} cached files in ~/.deep_research_cache/:")
    for file_info in cached_files:
        typer.echo(f"  {file_info['name']}")


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
