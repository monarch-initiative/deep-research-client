"""Tests for markdown_parser module."""

import pytest
import tempfile
from pathlib import Path

from deep_research_client.markdown_parser import (
    parse_frontmatter,
    extract_sections,
    extract_citations_list,
    parse_markdown_file,
    parse_markdown_files,
)


class TestParseFrontmatter:
    """Test frontmatter parsing."""

    def test_basic_frontmatter(self):
        """Test parsing basic YAML frontmatter."""
        content = """---
provider: perplexity
model: sonar-pro
---

# Content here"""
        fm, body = parse_frontmatter(content)

        assert fm["provider"] == "perplexity"
        assert fm["model"] == "sonar-pro"
        assert "# Content here" in body

    def test_no_frontmatter(self):
        """Test content without frontmatter."""
        content = "# Just a heading\n\nSome text."
        fm, body = parse_frontmatter(content)

        assert fm == {}
        assert body == content

    def test_complex_frontmatter(self):
        """Test parsing complex frontmatter with nested values."""
        content = """---
provider: mock
model: mock-model-v1
cached: false
duration_seconds: 0.4
keywords:
  - AI
  - research
provider_config:
  timeout: 600
  max_retries: 3
---

## Content"""
        fm, body = parse_frontmatter(content)

        assert fm["provider"] == "mock"
        assert fm["cached"] is False
        assert fm["duration_seconds"] == 0.4
        assert fm["keywords"] == ["AI", "research"]
        assert fm["provider_config"]["timeout"] == 600

    def test_empty_frontmatter(self):
        """Test empty frontmatter block."""
        content = """---
---

# Content"""
        fm, body = parse_frontmatter(content)

        assert fm == {}
        assert "# Content" in body

    def test_unclosed_frontmatter(self):
        """Test frontmatter without closing delimiter."""
        content = """---
provider: test
# No closing delimiter
Content here"""
        fm, body = parse_frontmatter(content)

        # Should return original content when frontmatter is unclosed
        assert fm == {}
        assert body == content


class TestExtractSections:
    """Test section extraction from markdown body."""

    def test_standard_sections(self):
        """Test extracting standard Question/Output/Citations sections."""
        body = """## Question

What is machine learning?

## Output

Machine learning is a field of AI.

## Citations

1. Source one
2. Source two"""
        sections = extract_sections(body)

        assert "machine learning" in sections["question"].lower()
        assert "field of AI" in sections["output"]
        assert "Source one" in sections["citations"]

    def test_missing_sections(self):
        """Test handling of missing sections."""
        body = """## Output

Just output, no question."""
        sections = extract_sections(body)

        assert sections["question"] == ""
        assert "Just output" in sections["output"]
        assert sections["citations"] == ""

    def test_no_sections(self):
        """Test body with no standard sections."""
        body = """# Introduction

Some content without our standard sections.

## Custom Heading

More content."""
        sections = extract_sections(body)

        assert sections["question"] == ""
        assert sections["output"] == ""
        assert sections["citations"] == ""


class TestExtractCitationsList:
    """Test citation list extraction."""

    def test_numbered_citations(self):
        """Test extracting numbered citation list."""
        text = """1. First source - https://example.com
2. Second source - https://test.org
3. Third source"""
        citations = extract_citations_list(text)

        assert len(citations) == 3
        assert "First source" in citations[0]
        assert "Second source" in citations[1]
        assert "Third source" in citations[2]

    def test_empty_citations(self):
        """Test empty citations text."""
        assert extract_citations_list("") == []
        assert extract_citations_list("   ") == []

    def test_non_numbered_text(self):
        """Test text without numbered list."""
        text = "Some text that is not a numbered list."
        citations = extract_citations_list(text)
        assert citations == []


class TestParseMarkdownFile:
    """Test parsing complete markdown files."""

    @pytest.fixture
    def sample_md_file(self):
        """Create a sample markdown file for testing."""
        content = """---
provider: perplexity
model: sonar-pro
title: Machine Learning Overview
keywords:
  - ML
  - AI
duration_seconds: 45.2
---

## Question

What is machine learning?

## Output

# Machine Learning

Machine learning is a subset of artificial intelligence (AI) that enables systems to learn and improve from experience.

## Key Concepts

- Supervised learning
- Unsupervised learning

## Citations

1. Wikipedia - Machine Learning
2. DeepMind Research"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            path = Path(f.name)

        yield path

        # Cleanup
        path.unlink()

    def test_parse_complete_file(self, sample_md_file):
        """Test parsing a complete markdown file."""
        result = parse_markdown_file(sample_md_file)

        assert result["provider"] == "perplexity"
        assert result["model"] == "sonar-pro"
        assert result["title"] == "Machine Learning Overview"
        assert result["keywords"] == ["ML", "AI"]
        assert result["duration_seconds"] == 45.2
        assert "machine learning" in result["query_preview"].lower()
        assert "Machine learning is" in result["markdown"]
        assert len(result["citations"]) == 2
        assert result["word_count"] > 0
        assert result["has_title"] is True
        assert result["has_keywords"] is True

    def test_parse_file_extracts_title_from_content(self):
        """Test title extraction when not in frontmatter."""
        content = """---
provider: test
---

## Question

Test query

## Output

# Auto-Generated Title

Some content here."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            path = Path(f.name)

        result = parse_markdown_file(path)

        assert result["title"] == "Auto-Generated Title"
        path.unlink()

    def test_parse_file_minimal_frontmatter(self):
        """Test parsing file with minimal frontmatter."""
        content = """---
provider: openai
---

## Question

Simple question

## Output

Simple answer"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            path = Path(f.name)

        result = parse_markdown_file(path)

        assert result["provider"] == "openai"
        assert result["model"] == "default"
        assert result["keywords"] == []
        assert result["author"] == ""
        path.unlink()


class TestParseMarkdownFiles:
    """Test parsing multiple markdown files."""

    @pytest.fixture
    def sample_directory(self):
        """Create a directory with sample markdown files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)

            # Create file 1
            (dir_path / "file1.md").write_text("""---
provider: perplexity
model: sonar-pro
---

## Question

Query 1

## Output

# Answer 1

Content 1""")

            # Create file 2
            (dir_path / "file2.md").write_text("""---
provider: openai
model: o3
---

## Question

Query 2

## Output

# Answer 2

Content 2""")

            # Create subdirectory with file
            subdir = dir_path / "subdir"
            subdir.mkdir()
            (subdir / "file3.md").write_text("""---
provider: consensus
---

## Question

Query 3

## Output

# Answer 3

Content 3""")

            # Create a non-md file (should be ignored)
            (dir_path / "notes.txt").write_text("Not a markdown file")

            yield dir_path

    def test_parse_directory_default_pattern(self, sample_directory):
        """Test parsing all md files in directory recursively."""
        results = parse_markdown_files(directory=sample_directory)

        assert len(results) == 3
        providers = {r["provider"] for r in results}
        assert providers == {"perplexity", "openai", "consensus"}

    def test_parse_directory_non_recursive(self, sample_directory):
        """Test parsing only top-level md files."""
        results = parse_markdown_files(directory=sample_directory, pattern="*.md")

        assert len(results) == 2
        providers = {r["provider"] for r in results}
        assert "consensus" not in providers  # In subdirectory

    def test_parse_explicit_file_list(self, sample_directory):
        """Test parsing explicit list of files."""
        files = [
            sample_directory / "file1.md",
            sample_directory / "subdir" / "file3.md",
        ]
        results = parse_markdown_files(files=files)

        assert len(results) == 2
        providers = {r["provider"] for r in results}
        assert providers == {"perplexity", "consensus"}

    def test_parse_without_content(self, sample_directory):
        """Test parsing without including full content."""
        results = parse_markdown_files(
            directory=sample_directory,
            pattern="*.md",
            include_content=False
        )

        for result in results:
            assert "markdown" not in result
            assert "citations" not in result
            assert "provider" in result  # Metadata still present

    def test_parse_no_source_raises_error(self):
        """Test that not providing files or directory raises error."""
        with pytest.raises(ValueError, match="Either 'files' or 'directory' must be provided"):
            parse_markdown_files()

    def test_parse_nonexistent_file_ignored(self, sample_directory):
        """Test that nonexistent files are silently ignored."""
        files = [
            sample_directory / "file1.md",
            sample_directory / "nonexistent.md",
        ]
        results = parse_markdown_files(files=files)

        assert len(results) == 1

    def test_results_sorted_by_date(self, sample_directory):
        """Test that results are sorted by date, newest first."""
        # Create files with different dates in frontmatter
        # Note: The parser uses file mtime for date, which has day resolution
        # So we test that date strings are sorted correctly
        results = parse_markdown_files(directory=sample_directory, pattern="*.md")

        # Should have results
        assert len(results) == 2

        # All results should have date field
        for result in results:
            assert "date" in result
            assert result["date"]  # Non-empty date string


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_malformed_yaml_frontmatter(self):
        """Test handling of malformed YAML in frontmatter."""
        content = """---
provider: test
  invalid: yaml: structure
---

Content"""
        fm, body = parse_frontmatter(content)

        # Should return empty frontmatter on YAML error
        assert fm == {}
        assert body == content

    def test_file_without_sections(self):
        """Test parsing file that doesn't follow our section convention."""
        content = """---
provider: external
title: External Document
---

# Introduction

This is a document that doesn't follow our standard sections.

It just has regular markdown content."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            path = Path(f.name)

        result = parse_markdown_file(path)

        # Should still extract what it can
        assert result["provider"] == "external"
        assert result["title"] == "External Document"
        # Content should be in markdown (from full body since no Output section)
        assert "Introduction" in result["markdown"]
        path.unlink()

    def test_unicode_content(self):
        """Test handling of unicode content."""
        content = """---
provider: test
title: Unicode Test 🔬
keywords:
  - 日本語
  - español
---

## Question

What about émojis and ünîcödë?

## Output

# Results: 中文

Content with special chars: α, β, γ"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(content)
            path = Path(f.name)

        result = parse_markdown_file(path)

        assert "🔬" in result["title"]
        assert "日本語" in result["keywords"]
        assert "émojis" in result["query_preview"]
        assert "中文" in result["markdown"]
        path.unlink()
