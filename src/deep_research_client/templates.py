"""Template processing for research queries."""

import re
from pathlib import Path
from typing import Dict, Any, List


class TemplateManager:
    """Manages query templates with variable substitution."""

    def __init__(self):
        """Initialize template manager."""
        pass

    def load_template(self, template_path: Path) -> str:
        """Load template from file.

        Args:
            template_path: Path to template file

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        return template_path.read_text(encoding='utf-8')

    def substitute_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in template.

        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variable names to values

        Returns:
            Template with variables substituted

        Examples:
            >>> tm = TemplateManager()
            >>> template = "Research {gene} in {organism}"
            >>> variables = {"gene": "TP53", "organism": "human"}
            >>> tm.substitute_variables(template, variables)
            'Research TP53 in human'
        """
        # Find all template variables
        template_vars = set(re.findall(r'{(\w+)}', template))

        # Check for missing variables
        missing_vars = template_vars - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing template variables: {', '.join(sorted(missing_vars))}")

        # Substitute variables
        result = template
        for var_name, value in variables.items():
            placeholder = f"{{{var_name}}}"
            result = result.replace(placeholder, str(value))

        return result

    def get_template_variables(self, template: str) -> set[str]:
        """Extract all template variables from a template string.

        Args:
            template: Template string

        Returns:
            Set of variable names found in template

        Examples:
            >>> tm = TemplateManager()
            >>> tm.get_template_variables("Research {gene} and {protein} in {organism}")
            {'gene', 'protein', 'organism'}
        """
        return set(re.findall(r'{(\w+)}', template))

    def parse_variable_string(self, var_string: str) -> Dict[str, str]:
        """Parse variable string into dictionary.

        Args:
            var_string: String like "gene=TP53,organism=human,tissue=brain"

        Returns:
            Dictionary of variable names to values

        Examples:
            >>> tm = TemplateManager()
            >>> tm.parse_variable_string("gene=TP53,organism=human")
            {'gene': 'TP53', 'organism': 'human'}
        """
        if not var_string.strip():
            return {}

        variables = {}
        for pair in var_string.split(','):
            pair = pair.strip()
            if '=' not in pair:
                raise ValueError(f"Invalid variable assignment: {pair}. Use format 'name=value'")

            name, value = pair.split('=', 1)
            name = name.strip()
            value = value.strip()

            if not name:
                raise ValueError(f"Empty variable name in: {pair}")

            variables[name] = value

        return variables

    def parse_variable_list(self, var_list: List[str]) -> Dict[str, str]:
        """Parse list of variable strings into dictionary.

        Args:
            var_list: List of strings like ["gene=TP53", "organism=human"]

        Returns:
            Dictionary of variable names to values

        Examples:
            >>> tm = TemplateManager()
            >>> tm.parse_variable_list(["gene=TP53", "organism=human"])
            {'gene': 'TP53', 'organism': 'human'}
        """
        variables = {}

        for var_assignment in var_list:
            var_assignment = var_assignment.strip()
            if '=' not in var_assignment:
                raise ValueError(f"Invalid variable assignment: {var_assignment}. Use format 'name=value'")

            name, value = var_assignment.split('=', 1)
            name = name.strip()
            value = value.strip()

            if not name:
                raise ValueError(f"Empty variable name in: {var_assignment}")

            variables[name] = value

        return variables

    def render_template(self, template_path: Path, variables: Dict[str, Any]) -> str:
        """Load template from file and substitute variables.

        Args:
            template_path: Path to template file
            variables: Dictionary of variables to substitute

        Returns:
            Rendered template string
        """
        template = self.load_template(template_path)
        return self.substitute_variables(template, variables)