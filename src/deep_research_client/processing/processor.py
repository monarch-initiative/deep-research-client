"""High-level processing interface for research operations."""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .template_processor import TemplateProcessor
from .result_formatter import ResultFormatter
from ..models import ResearchResult


class ResearchProcessor:
    """High-level processor for research operations including template processing and result formatting."""

    def __init__(self):
        """Initialize the research processor."""
        self.template_processor = TemplateProcessor()
        self.result_formatter = ResultFormatter()

    def process_template_file(
        self, 
        template_path: Path, 
        variables: Optional[List[str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Process a template file with variables.

        Args:
            template_path: Path to template file
            variables: List of variable assignments like ["key=value"]

        Returns:
            Tuple of (rendered_query, template_metadata)

        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template variables are missing or invalid
        """
        # Parse variables if provided
        parsed_variables = {}
        if variables:
            parsed_variables = self.template_processor.parse_variable_list(variables)

        # Process the template
        return self.template_processor.process_template(template_path, parsed_variables)

    def format_research_result(
        self, 
        result: ResearchResult, 
        separate_citations: bool = False
    ) -> str:
        """Format a research result as markdown.

        Args:
            result: Research result to format
            separate_citations: If True, excludes citations from main content

        Returns:
            Formatted markdown string
        """
        return self.result_formatter.format_full_markdown(result, separate_citations)

    def format_citations_only(self, result: ResearchResult) -> str:
        """Format just the citations from a research result.

        Args:
            result: Research result with citations

        Returns:
            Formatted citations markdown
        """
        return self.result_formatter.format_citations_only(result)

    def validate_template_variables(
        self, 
        template_path: Path, 
        variables: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate that all required template variables are provided.

        Args:
            template_path: Path to template file
            variables: List of variable assignments like ["key=value"]

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Load template to check for variables
            template_content = self.template_processor.load_template(template_path)
            template_vars = self.template_processor.get_template_variables(template_content)
            
            if not template_vars:
                return True, None  # No variables needed
            
            if not variables:
                return False, f"Template requires variables: {', '.join(sorted(template_vars))}"
            
            # Parse and validate variables
            parsed_variables = self.template_processor.parse_variable_list(variables)
            
            # Check for missing variables
            missing_vars = template_vars - set(parsed_variables.keys())
            if missing_vars:
                return False, f"Missing template variables: {', '.join(sorted(missing_vars))}"
            
            return True, None
            
        except (FileNotFoundError, ValueError) as e:
            return False, str(e)
