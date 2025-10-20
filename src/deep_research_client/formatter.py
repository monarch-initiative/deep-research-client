"""Output formatting utilities for research results."""

import yaml
from typing import Dict, Any

from .models import ResearchResult


class ResultFormatter:
    """Formats research results with frontmatter and structured output."""

    def format_full_markdown(
        self,
        result: ResearchResult,
        separate_citations: bool = False
    ) -> str:
        """Format result as markdown with YAML frontmatter.

        Args:
            result: Research result to format
            separate_citations: If True, excludes citations from main content

        Returns:
            Formatted markdown with frontmatter
        """
        # Build frontmatter metadata
        metadata: Dict[str, Any] = {
            "provider": result.provider,
            "model": result.model,
            "cached": result.cached,
        }

        # Add timing information
        if result.start_time:
            metadata["start_time"] = result.start_time.isoformat()
        if result.end_time:
            metadata["end_time"] = result.end_time.isoformat()
        if result.duration_seconds:
            metadata["duration_seconds"] = round(result.duration_seconds, 2)

        # Add template information if used
        if result.template_file:
            metadata["template_file"] = result.template_file
        if result.template_variables:
            metadata["template_variables"] = result.template_variables

        # Add provider configuration
        if result.provider_config:
            metadata["provider_config"] = result.provider_config

        # Add citation count
        if result.citations:
            metadata["citation_count"] = len(result.citations)

        # Convert metadata to YAML frontmatter
        frontmatter_yaml = yaml.dump(metadata, default_flow_style=False, sort_keys=False)

        # Build the markdown content
        parts = []

        # Add YAML frontmatter
        parts.append("---")
        parts.append(frontmatter_yaml.rstrip())
        parts.append("---")
        parts.append("")

        # Add question section
        parts.append("## Question")
        parts.append("")
        parts.append(result.query)
        parts.append("")

        # Add output section
        parts.append("## Output")
        parts.append("")
        parts.append(result.markdown)

        # Add citations section (unless separated)
        if result.citations and not separate_citations:
            parts.append("")
            parts.append("## Citations")
            parts.append("")
            for i, citation in enumerate(result.citations, 1):
                parts.append(f"{i}. {citation}")

        return "\n".join(parts)

    def format_citations_only(self, result: ResearchResult) -> str:
        """Format just the citations as markdown.

        Args:
            result: Research result with citations

        Returns:
            Formatted citations markdown
        """
        if not result.citations:
            return "# Citations\n\nNo citations found in this research result."

        parts = [
            "# Citations for Research Query",
            "",
            f"**Query:** {result.query}",
            f"**Provider:** {result.provider}",
            f"**Generated:** {result.end_time.isoformat() if result.end_time else 'N/A'}",
            "",
        ]

        for i, citation in enumerate(result.citations, 1):
            parts.append(f"{i}. {citation}")

        return "\n".join(parts)