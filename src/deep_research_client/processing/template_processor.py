"""Template processing for research queries."""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Literal
from jinja2 import Template as JinjaTemplate


class TemplateProcessor:
    """Manages query templates with variable substitution."""

    def __init__(self):
        """Initialize template processor."""
        pass

    def _parse_frontmatter(self, content: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Parse YAML frontmatter from content if present.

        Args:
            content: Template content possibly containing frontmatter

        Returns:
            Tuple of (frontmatter_dict, content_without_frontmatter)

        Examples:
            >>> tp = TemplateProcessor()
            >>> content = "---\\nformat: jinja\\n---\\nHello {{name}}"
            >>> fm, body = tp._parse_frontmatter(content)
            >>> fm['format']
            'jinja'
            >>> body
            'Hello {{name}}'
        """
        # Check if content starts with frontmatter delimiter
        if not content.startswith('---'):
            return None, content

        # Find the closing delimiter
        lines = content.split('\n')
        end_index = None
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == '---':
                end_index = i
                break

        if end_index is None:
            # No closing delimiter found
            return None, content

        # Extract and parse frontmatter
        frontmatter_lines = lines[1:end_index]
        frontmatter_text = '\n'.join(frontmatter_lines)

        try:
            frontmatter = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError:
            # Invalid YAML, treat as no frontmatter
            return None, content

        # Return frontmatter and remaining content
        body_lines = lines[end_index + 1:]
        body = '\n'.join(body_lines)

        return frontmatter, body

    def _detect_template_format(
        self,
        template_path: Path,
        content: str,
        frontmatter: Optional[Dict[str, Any]]
    ) -> Literal['jinja', 'fstring']:
        """Detect the template format based on file extension and frontmatter.

        Args:
            template_path: Path to the template file
            content: Template content
            frontmatter: Parsed frontmatter if present

        Returns:
            Template format: 'jinja' or 'fstring'

        Examples:
            >>> tp = TemplateProcessor()
            >>> tp._detect_template_format(Path("test.j2"), "", None)
            'jinja'
            >>> tp._detect_template_format(Path("test.jinja"), "", None)
            'jinja'
            >>> tp._detect_template_format(Path("test.md"), "", {"format": "jinja"})
            'jinja'
            >>> tp._detect_template_format(Path("test.md"), "", None)
            'fstring'
        """
        # Check file extension first
        suffix = template_path.suffix.lower()
        stem_parts = template_path.stem.split('.')

        # Check for .j2, .jinja, .jinja2 extensions
        if suffix in ['.j2', '.jinja', '.jinja2']:
            return 'jinja'

        # Check for compound extensions like .md.j2
        if len(stem_parts) > 1:
            second_suffix = '.' + stem_parts[-1]
            if second_suffix in ['.j2', '.jinja', '.jinja2']:
                return 'jinja'

        # Check frontmatter for format specification
        if frontmatter and 'format' in frontmatter:
            format_value = str(frontmatter['format']).lower()
            if format_value in ['jinja', 'jinja2']:
                return 'jinja'
            elif format_value in ['fstring', 'f-string', 'python']:
                return 'fstring'

        # Default to fstring for backward compatibility
        return 'fstring'

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

    def substitute_variables(
        self,
        template: str,
        variables: Dict[str, Any],
        template_format: Literal['jinja', 'fstring'] = 'fstring'
    ) -> str:
        """Substitute variables in template.

        Args:
            template: Template string with {variable} placeholders (fstring) or {{variable}} (jinja)
            variables: Dictionary of variable names to values
            template_format: Template format - 'jinja' or 'fstring'

        Returns:
            Template with variables substituted

        Examples:
            >>> tp = TemplateProcessor()
            >>> template = "Research {gene} in {organism}"
            >>> variables = {"gene": "TP53", "organism": "human"}
            >>> tp.substitute_variables(template, variables, 'fstring')
            'Research TP53 in human'

            >>> jinja_template = "Research {{gene}} in {{organism}}"
            >>> tp.substitute_variables(jinja_template, variables, 'jinja')
            'Research TP53 in human'
        """
        if template_format == 'jinja':
            return self._substitute_jinja(template, variables)
        else:
            return self._substitute_fstring(template, variables)

    def _substitute_fstring(self, template: str, variables: Dict[str, Any]) -> str:
        """Substitute variables using f-string style {variable} placeholders.

        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variable names to values

        Returns:
            Template with variables substituted
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

    def _substitute_jinja(self, template: str, variables: Dict[str, Any]) -> str:
        """Substitute variables using Jinja2 templating.

        Args:
            template: Jinja2 template string
            variables: Dictionary of variable names to values

        Returns:
            Rendered template

        Raises:
            TemplateSyntaxError: If template syntax is invalid
            ValueError: If required variables are missing
        """
        jinja_template = JinjaTemplate(template)

        # Check for missing required variables
        # Get all undefined variables that Jinja would need
        from jinja2 import meta
        from jinja2 import Environment
        env = Environment()
        ast = env.parse(template)
        required_vars = meta.find_undeclared_variables(ast)

        missing_vars = required_vars - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing template variables: {', '.join(sorted(missing_vars))}")

        return jinja_template.render(**variables)

    def get_template_variables(
        self,
        template: str,
        template_format: Literal['jinja', 'fstring'] = 'fstring'
    ) -> set[str]:
        """Extract all template variables from a template string.

        Args:
            template: Template string
            template_format: Template format - 'jinja' or 'fstring'

        Returns:
            Set of variable names found in template

        Examples:
            >>> tp = TemplateProcessor()
            >>> sorted(tp.get_template_variables("Research {gene} and {protein} in {organism}", 'fstring'))
            ['gene', 'organism', 'protein']

            >>> sorted(tp.get_template_variables("Research {{gene}} and {{protein}}", 'jinja'))
            ['gene', 'protein']
        """
        if template_format == 'jinja':
            from jinja2 import meta, Environment
            env = Environment()
            ast = env.parse(template)
            return meta.find_undeclared_variables(ast)
        else:
            return set(re.findall(r'{(\w+)}', template))

    def parse_variable_string(self, var_string: str) -> Dict[str, str]:
        """Parse variable string into dictionary.

        Args:
            var_string: String like "gene=TP53,organism=human,tissue=brain"

        Returns:
            Dictionary of variable names to values

        Examples:
            >>> tp = TemplateProcessor()
            >>> tp.parse_variable_string("gene=TP53,organism=human")
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
            >>> tp = TemplateProcessor()
            >>> tp.parse_variable_list(["gene=TP53", "organism=human"])
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
        # Load raw content
        raw_content = self.load_template(template_path)

        # Parse frontmatter if present
        frontmatter, template_content = self._parse_frontmatter(raw_content)

        # Detect template format
        template_format = self._detect_template_format(template_path, template_content, frontmatter)

        # Substitute variables using detected format
        return self.substitute_variables(template_content, variables, template_format)

    def process_template(
        self,
        template_path: Path,
        variables: Optional[Dict[str, str]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Process a template file with variables and return rendered content and metadata.

        Args:
            template_path: Path to template file
            variables: Optional dictionary of variables to substitute

        Returns:
            Tuple of (rendered_query, template_metadata)

        Raises:
            FileNotFoundError: If template file doesn't exist
            ValueError: If template variables are missing or invalid
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        # Load raw template content
        raw_content = self.load_template(template_path)

        # Parse frontmatter if present
        frontmatter, template_content = self._parse_frontmatter(raw_content)

        # Detect template format
        template_format = self._detect_template_format(template_path, template_content, frontmatter)

        # Get required variables using detected format
        template_vars = self.get_template_variables(template_content, template_format)

        # Use provided variables or empty dict
        variables = variables or {}

        # Check for missing variables
        if template_vars and not variables:
            raise ValueError(f"Template requires variables: {', '.join(sorted(template_vars))}")

        # Render the template using detected format
        rendered_query = self.substitute_variables(template_content, variables, template_format)

        # Create template metadata
        template_metadata = {
            'template_file': str(template_path),
            'template_variables': variables,
            'template_format': template_format
        }

        return rendered_query, template_metadata
