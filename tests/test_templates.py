"""Tests for template functionality."""

import pytest
from pathlib import Path
import tempfile

from deep_research_client.processing import TemplateProcessor


def test_template_processor_creation():
    """Test creating a TemplateProcessor."""
    tp = TemplateProcessor()
    assert tp is not None


def test_substitute_variables():
    """Test variable substitution."""
    tm = TemplateProcessor()
    template = "Research {gene} in {organism}"
    variables = {"gene": "TP53", "organism": "human"}

    result = tm.substitute_variables(template, variables)
    assert result == "Research TP53 in human"


def test_substitute_variables_missing():
    """Test variable substitution with missing variables."""
    tm = TemplateProcessor()
    template = "Research {gene} in {organism}"
    variables = {"gene": "TP53"}  # Missing organism

    with pytest.raises(ValueError, match="Missing template variables: organism"):
        tm.substitute_variables(template, variables)


def test_get_template_variables():
    """Test extracting template variables."""
    tm = TemplateProcessor()
    template = "Research {gene} and {protein} in {organism}"

    variables = tm.get_template_variables(template)
    assert variables == {"gene", "protein", "organism"}


def test_get_template_variables_empty():
    """Test extracting variables from template with no variables."""
    tm = TemplateProcessor()
    template = "This has no variables"

    variables = tm.get_template_variables(template)
    assert variables == set()


def test_parse_variable_string():
    """Test parsing variable string."""
    tm = TemplateProcessor()
    var_string = "gene=TP53,organism=human,tissue=brain"

    variables = tm.parse_variable_string(var_string)
    expected = {"gene": "TP53", "organism": "human", "tissue": "brain"}
    assert variables == expected


def test_parse_variable_string_empty():
    """Test parsing empty variable string."""
    tm = TemplateProcessor()
    variables = tm.parse_variable_string("")
    assert variables == {}

    variables = tm.parse_variable_string("   ")
    assert variables == {}


def test_parse_variable_string_invalid():
    """Test parsing invalid variable string."""
    tm = TemplateProcessor()

    with pytest.raises(ValueError, match="Invalid variable assignment"):
        tm.parse_variable_string("invalid_format")

    with pytest.raises(ValueError, match="Empty variable name"):
        tm.parse_variable_string("=value")


def test_load_template():
    """Test loading template from file."""
    tm = TemplateProcessor()

    # Create temporary template file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        template_content = "Research {gene} in {organism}\n\nDetails about {gene}."
        f.write(template_content)
        temp_path = Path(f.name)

    try:
        loaded_content = tm.load_template(temp_path)
        assert loaded_content == template_content
    finally:
        temp_path.unlink()


def test_load_template_not_found():
    """Test loading non-existent template."""
    tm = TemplateProcessor()
    fake_path = Path("/nonexistent/template.md")

    with pytest.raises(FileNotFoundError):
        tm.load_template(fake_path)


def test_render_template():
    """Test complete template rendering workflow."""
    tm = TemplateProcessor()

    # Create temporary template file
    template_content = """# Gene Research: {gene}

Research the gene {gene} in {organism}.

Focus areas:
1. Function of {gene}
2. Expression in {tissue}
3. Role in {disease}

Gene: {gene}
Organism: {organism}
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(template_content)
        temp_path = Path(f.name)

    try:
        variables = {
            "gene": "BRCA1",
            "organism": "human",
            "tissue": "breast",
            "disease": "cancer"
        }

        result = tm.render_template(temp_path, variables)

        assert "# Gene Research: BRCA1" in result
        assert "Research the gene BRCA1 in human" in result
        assert "Function of BRCA1" in result
        assert "Expression in breast" in result
        assert "Role in cancer" in result

    finally:
        temp_path.unlink()


def test_complex_template_variables():
    """Test template with various variable patterns."""
    tm = TemplateProcessor()
    template = """
    Study {gene} in {organism}.
    Also study {gene} again.
    And {gene} once more.
    Different var: {protein}
    """

    variables = tm.get_template_variables(template)
    assert variables == {"gene", "organism", "protein"}

    # Test substitution
    var_values = {"gene": "TP53", "organism": "mouse", "protein": "p53"}
    result = tm.substitute_variables(template, var_values)

    assert result.count("TP53") == 3  # gene appears 3 times
    assert "mouse" in result
    assert "p53" in result


# Jinja2 template tests


def test_jinja_substitute_variables():
    """Test Jinja2 variable substitution."""
    tm = TemplateProcessor()
    template = "Research {{gene}} in {{organism}}"
    variables = {"gene": "TP53", "organism": "human"}

    result = tm.substitute_variables(template, variables, template_format='jinja')
    assert result == "Research TP53 in human"


def test_jinja_with_conditionals():
    """Test Jinja2 template with conditional logic."""
    tm = TemplateProcessor()
    template = """Research {{gene}}
{% if detail_level == "high" %}
Include detailed information
{% else %}
Brief overview only
{% endif %}"""

    # Test with high detail
    result = tm.substitute_variables(
        template,
        {"gene": "BRCA1", "detail_level": "high"},
        template_format='jinja'
    )
    assert "Include detailed information" in result
    assert "Brief overview only" not in result

    # Test with low detail
    result = tm.substitute_variables(
        template,
        {"gene": "BRCA1", "detail_level": "low"},
        template_format='jinja'
    )
    assert "Brief overview only" in result
    assert "Include detailed information" not in result


def test_jinja_with_loops():
    """Test Jinja2 template with loops."""
    tm = TemplateProcessor()
    template = """Topics:
{% for topic in topics %}
- {{topic}}
{% endfor %}"""

    variables = {
        "topics": ["function", "expression", "mutations"]
    }

    result = tm.substitute_variables(template, variables, template_format='jinja')
    assert "- function" in result
    assert "- expression" in result
    assert "- mutations" in result


def test_jinja_get_template_variables():
    """Test extracting variables from Jinja2 templates."""
    tm = TemplateProcessor()
    template = "Research {{gene}} and {{protein}} in {{organism}}"

    variables = tm.get_template_variables(template, template_format='jinja')
    assert variables == {"gene", "protein", "organism"}


def test_jinja_missing_variables():
    """Test Jinja2 with missing required variables."""
    tm = TemplateProcessor()
    template = "Research {{gene}} in {{organism}}"
    variables = {"gene": "TP53"}  # Missing organism

    with pytest.raises(ValueError, match="Missing template variables: organism"):
        tm.substitute_variables(template, variables, template_format='jinja')


def test_parse_frontmatter():
    """Test parsing YAML frontmatter from template content."""
    tm = TemplateProcessor()

    # Template with frontmatter
    content = """---
format: jinja
description: Test template
---
Research {{gene}}"""

    frontmatter, body = tm._parse_frontmatter(content)

    assert frontmatter is not None
    assert frontmatter['format'] == 'jinja'
    assert frontmatter['description'] == 'Test template'
    assert body.strip() == "Research {{gene}}"


def test_parse_frontmatter_no_frontmatter():
    """Test parsing content without frontmatter."""
    tm = TemplateProcessor()
    content = "Research {gene}"

    frontmatter, body = tm._parse_frontmatter(content)

    assert frontmatter is None
    assert body == content


def test_detect_template_format_extension():
    """Test template format detection by file extension."""
    tm = TemplateProcessor()

    # Jinja extensions
    assert tm._detect_template_format(Path("test.j2"), "", None) == 'jinja'
    assert tm._detect_template_format(Path("test.jinja"), "", None) == 'jinja'
    assert tm._detect_template_format(Path("test.jinja2"), "", None) == 'jinja'
    assert tm._detect_template_format(Path("test.md.j2"), "", None) == 'jinja'

    # Default to fstring
    assert tm._detect_template_format(Path("test.md"), "", None) == 'fstring'
    assert tm._detect_template_format(Path("test.txt"), "", None) == 'fstring'


def test_detect_template_format_frontmatter():
    """Test template format detection via frontmatter."""
    tm = TemplateProcessor()

    # Frontmatter specifies jinja
    frontmatter = {"format": "jinja"}
    assert tm._detect_template_format(Path("test.md"), "", frontmatter) == 'jinja'

    # Frontmatter specifies fstring
    frontmatter = {"format": "fstring"}
    assert tm._detect_template_format(Path("test.md"), "", frontmatter) == 'fstring'

    # No frontmatter, defaults to fstring
    assert tm._detect_template_format(Path("test.md"), "", None) == 'fstring'


def test_render_jinja_template_file():
    """Test rendering a Jinja2 template file with .j2 extension."""
    tm = TemplateProcessor()

    template_content = """Research {{gene}} in {{organism}}.
{% if detail_level == "high" %}
Detailed analysis required.
{% else %}
Brief overview.
{% endif %}"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.j2', delete=False) as f:
        f.write(template_content)
        temp_path = Path(f.name)

    try:
        variables = {
            "gene": "BRCA1",
            "organism": "human",
            "detail_level": "high"
        }

        result = tm.render_template(temp_path, variables)

        assert "Research BRCA1 in human" in result
        assert "Detailed analysis required" in result
        assert "Brief overview" not in result

    finally:
        temp_path.unlink()


def test_render_template_with_frontmatter():
    """Test rendering template with frontmatter specifying format."""
    tm = TemplateProcessor()

    template_content = """---
format: jinja
description: Gene research
---
Research {{gene}}{% if organism %} in {{organism}}{% endif %}."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(template_content)
        temp_path = Path(f.name)

    try:
        variables = {
            "gene": "TP53",
            "organism": "mouse"
        }

        result = tm.render_template(temp_path, variables)

        assert "Research TP53 in mouse" in result
        # Frontmatter should not be in the result
        assert "format: jinja" not in result

    finally:
        temp_path.unlink()


def test_process_template_includes_format():
    """Test that process_template includes format in metadata."""
    tm = TemplateProcessor()

    # Create a Jinja template
    template_content = "Research {{gene}}"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.j2', delete=False) as f:
        f.write(template_content)
        temp_path = Path(f.name)

    try:
        variables = {"gene": "BRCA1"}
        rendered, metadata = tm.process_template(temp_path, variables)

        assert rendered == "Research BRCA1"
        assert metadata['template_format'] == 'jinja'
        assert metadata['template_variables'] == variables

    finally:
        temp_path.unlink()


def test_jinja_filters():
    """Test that Jinja2 filters work correctly."""
    tm = TemplateProcessor()

    template = "Gene: {{gene|upper}}, Organism: {{organism|capitalize}}"
    variables = {"gene": "brca1", "organism": "human"}

    result = tm.substitute_variables(template, variables, template_format='jinja')
    assert "Gene: BRCA1" in result
    assert "Organism: Human" in result