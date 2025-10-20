"""CLI interface for deep-research-client."""

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

app = typer.Typer(help="deep-research-client: Wrapper for multiple deep research tools")


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
                typer.echo(f"❌ Template error: {error_msg}", err=True)
                if error_msg and "requires variables" in error_msg:
                    typer.echo("Use --var key=value for each variable", err=True)
                raise typer.Exit(1)

            # Process the template
            query, template_info = processor.process_template_file(template, var)
            
            typer.echo(f"📝 Using template: {template.name}")
            if template_info['template_variables']:
                var_str = ', '.join(f"{k}={v}" for k, v in template_info['template_variables'].items())
                typer.echo(f"🔧 Variables: {var_str}")

        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"❌ Template error: {e}", err=True)
            raise typer.Exit(1)

    elif not query:
        typer.echo("❌ Either provide a query or use --template", err=True)
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
        except ValueError as e:
            typer.echo(f"❌ Error parsing parameters: {e}", err=True)
            raise typer.Exit(1)

    # Setup cache configuration
    cache_config = CacheConfig(enabled=not no_cache)
    if cache_dir:
        cache_config.directory = str(cache_dir)

    # Initialize client
    client = DeepResearchClient(cache_config=cache_config)

    # Check if any providers are available
    available_providers = client.get_available_providers()
    if not available_providers:
        typer.echo("❌ No research providers available. Please set API keys:", err=True)
        typer.echo("  - OPENAI_API_KEY for OpenAI Deep Research", err=True)
        typer.echo("  - FUTUREHOUSE_API_KEY for Falcon", err=True)
        typer.echo("  - PERPLEXITY_API_KEY for Perplexity AI", err=True)
        raise typer.Exit(1)

    # Show available providers
    if provider:
        if provider not in available_providers:
            typer.echo(f"❌ Provider '{provider}' not available. Available: {', '.join(available_providers)}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Using provider: {provider}")
    else:
        typer.echo(f"Available providers: {', '.join(available_providers)}")
        typer.echo(f"Using: {available_providers[0]}")

    typer.echo("⏳ Researching...")

    try:
        # Perform research
        result = client.research(query, provider, template_info, model, provider_params)

        # Show cache status
        if result.cached:
            typer.echo("✅ Result retrieved from cache")
        else:
            typer.echo(f"✅ Research completed using {result.provider}")

        # Determine if we're separating citations
        should_separate_citations = separate_citations is not None

        # Format output using processor
        output_content = processor.format_research_result(result, separate_citations=should_separate_citations)

        # Output result
        if output:
            output.write_text(output_content, encoding='utf-8')
            typer.echo(f"📄 Result saved to: {output}")

            # Save separate citations file if requested
            if should_separate_citations and result.citations:
                # Use provided path or default to output.citations.md
                if isinstance(separate_citations, Path):
                    citations_output = separate_citations
                else:
                    citations_output = output.with_suffix('.citations.md')

                citations_content = processor.format_citations_only(result)
                citations_output.write_text(citations_content, encoding='utf-8')
                typer.echo(f"📚 Citations saved to: {citations_output}")

            # Show citation count
            if result.citations:
                typer.echo(f"📚 Found {len(result.citations)} citations")
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
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def providers(
    show_params: Annotated[bool, typer.Option("--show-params", help="Show available parameters for each provider")] = False,
    provider: Annotated[Optional[str], typer.Option(help="Show details for specific provider only")] = None,
):
    """List available research providers and their parameters."""
    from .provider_params import PROVIDER_PARAMS_REGISTRY

    client = DeepResearchClient()
    available = client.get_available_providers()

    if provider:
        # Show details for specific provider
        if provider not in PROVIDER_PARAMS_REGISTRY:
            typer.echo(f"❌ Unknown provider: {provider}")
            typer.echo(f"Available providers: {', '.join(PROVIDER_PARAMS_REGISTRY.keys())}")
            raise typer.Exit(1)

        is_available = provider in available
        status = "✅ Available" if is_available else "❌ Not available (missing API key)"
        typer.echo(f"Provider: {provider} - {status}")

        if not is_available:
            # Show required environment variable
            env_vars = {
                "openai": "OPENAI_API_KEY",
                "falcon": "FUTUREHOUSE_API_KEY",
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
        typer.echo("Available providers:")
        for prov in available:
            typer.echo(f"  ✅ {prov}")

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
        typer.echo("❌ No providers available. Please set API keys:")
        typer.echo("  - OPENAI_API_KEY for OpenAI Deep Research")
        typer.echo("  - FUTUREHOUSE_API_KEY for Falcon")
        typer.echo("  - PERPLEXITY_API_KEY for Perplexity AI")
        typer.echo("  - CONSENSUS_API_KEY for Consensus")
        typer.echo("  - ENABLE_MOCK_PROVIDER=true for Mock provider")

    if not show_params and not provider:
        typer.echo("\nUse --show-params to see available parameters for each provider")
        typer.echo("Use --provider <name> to see detailed info for a specific provider")


@app.command()
def clear_cache():
    """Clear all cached research results."""
    client = DeepResearchClient()
    count = client.clear_cache()
    typer.echo(f"🗑️  Cleared {count} cached files")


@app.command()
def list_cache():
    """List cached research files."""
    client = DeepResearchClient()
    cached_files = client.list_cached_files()

    if not cached_files:
        typer.echo("📭 No cached files found")
        return

    typer.echo(f"📁 Found {len(cached_files)} cached files in ~/.deep_research_cache/:")
    for file_info in cached_files:
        typer.echo(f"  📄 {file_info['name']}")


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
        cards = get_provider_model_cards(provider)
        if not cards:
            typer.echo(f"❌ Provider '{provider}' not found")
            raise typer.Exit(1)

        typer.echo(f"🔍 **{cards.provider_name.upper()}** Models")
        typer.echo(f"Default: {cards.default_model}")
        typer.echo()

        for model_name, card in cards.models.items():
            _display_model_card(card, detailed)

    elif cost:
        # Filter by cost level
        try:
            cost_level = CostLevel(cost.lower())
        except ValueError:
            typer.echo(f"❌ Invalid cost level '{cost}'. Use: low, medium, high, very_high")
            raise typer.Exit(1)

        models_by_cost = find_models_by_cost(cost_level)
        if not models_by_cost:
            typer.echo(f"📭 No models found with cost level: {cost}")
            return

        typer.echo(f"💰 **{cost.upper()}** Cost Models")
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
            typer.echo(f"❌ Invalid capability '{capability}'. Use: web_search, academic_search, scientific_literature, etc.")
            raise typer.Exit(1)

        models_by_cap = find_models_by_capability(cap)
        if not models_by_cap:
            typer.echo(f"📭 No models found with capability: {capability}")
            return

        typer.echo(f"⚡ **{capability.upper().replace('_', ' ')}** Capable Models")
        typer.echo()

        for provider_name, model_cards_list in models_by_cap.items():
            typer.echo(f"**{provider_name.upper()}:**")
            for card in model_cards_list:
                _display_model_card(card, detailed, indent="  ")
            typer.echo()

    else:
        # Show all models by provider
        all_models = list_all_models()
        typer.echo("🤖 **Available Research Models**")
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
