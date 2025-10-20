"""Tests for new template variable list functionality."""

import pytest

from deep_research_client.processing import TemplateProcessor


def test_parse_variable_list():
    """Test parsing list of variable strings."""
    tm = TemplateProcessor()
    var_list = ["gene=TP53", "organism=human", "tissue=brain"]

    variables = tm.parse_variable_list(var_list)
    expected = {"gene": "TP53", "organism": "human", "tissue": "brain"}
    assert variables == expected


def test_parse_variable_list_with_commas():
    """Test parsing variables with commas in values."""
    tm = TemplateProcessor()
    var_list = ["gene=TP53", "organism=Homo sapiens, human", "tissue=brain tissue, neocortex"]

    variables = tm.parse_variable_list(var_list)
    expected = {
        "gene": "TP53",
        "organism": "Homo sapiens, human",
        "tissue": "brain tissue, neocortex"
    }
    assert variables == expected


def test_parse_variable_list_empty():
    """Test parsing empty variable list."""
    tm = TemplateProcessor()
    variables = tm.parse_variable_list([])
    assert variables == {}


def test_parse_variable_list_invalid():
    """Test parsing invalid variable assignments."""
    tm = TemplateProcessor()

    with pytest.raises(ValueError, match="Invalid variable assignment"):
        tm.parse_variable_list(["invalid_format"])

    with pytest.raises(ValueError, match="Empty variable name"):
        tm.parse_variable_list(["=value"])


def test_parse_variable_list_whitespace():
    """Test parsing with extra whitespace."""
    tm = TemplateProcessor()
    var_list = ["  gene=TP53  ", " organism = human ", "tissue=brain "]

    variables = tm.parse_variable_list(var_list)
    expected = {"gene": "TP53", "organism": "human", "tissue": "brain"}
    assert variables == expected


def test_parse_variable_list_equals_in_value():
    """Test parsing when value contains equals sign."""
    tm = TemplateProcessor()
    var_list = ["equation=E=mc^2", "url=https://example.com?a=1&b=2"]

    variables = tm.parse_variable_list(var_list)
    expected = {"equation": "E=mc^2", "url": "https://example.com?a=1&b=2"}
    assert variables == expected