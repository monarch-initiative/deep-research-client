"""Output formatting utilities for research results."""

from collections import Counter
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
            trajectory_id = result.provider_config.get("trajectory_id")
            if trajectory_id:
                metadata["trajectory_id"] = trajectory_id

        # Add run provenance metadata (e.g. actual model(s) used, cost, turns)
        if result.run_metadata:
            metadata["run_metadata"] = result.run_metadata

        # Record a fallback, and only a fallback: `provider` on its own would
        # make a report produced by a stand-in read as a deliberate choice.
        metadata.update(result.fallback_frontmatter())

        # Add citation count
        if result.citations:
            metadata["citation_count"] = len(result.citations)

        if result.artifacts:
            metadata["artifact_count"] = len(result.artifacts)
            metadata["artifact_sources"] = dict(
                Counter((artifact.source or "unknown") for artifact in result.artifacts)
            )
            metadata["artifacts"] = [
                {
                    "filename": artifact.filename,
                    "path": artifact.path or artifact.filename,
                    "media_type": artifact.media_type,
                    "source": artifact.source,
                    "data_storage_id": artifact.data_storage_id,
                    "description": artifact.description,
                }
                for artifact in result.artifacts
            ]

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

        if result.artifacts:
            parts.append("")
            parts.append("## Artifacts")
            parts.append("")
            for artifact in result.artifacts:
                artifact_path = artifact.path or artifact.filename
                label = artifact.description or artifact.filename
                if artifact.is_image:
                    parts.append(f"![{label}]({artifact_path})")
                else:
                    parts.append(f"- [{label}]({artifact_path})")

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
