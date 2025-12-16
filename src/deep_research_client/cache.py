"""Simple file-based caching for research results."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import aiofiles

from .models import ResearchResult, CacheConfig


def preprocess_markdown(markdown: str, query: str = "") -> str:
    """Preprocess markdown content before title extraction or HTML rendering.

    Performs cleanup steps:
    1. Strips the query string if it appears at the start
    2. Strips "Question:" preamble (Falcon provider includes prompt in response)
    3. Removes leading whitespace

    Args:
        markdown: Raw markdown content
        query: Original query string to check for and remove

    Returns:
        Cleaned markdown content

    >>> preprocess_markdown("What is BRCA1?\\n\\n# BRCA1 Gene\\n\\nContent", "What is BRCA1?")
    '# BRCA1 Gene\\n\\nContent'
    >>> preprocess_markdown("# BRCA1 Gene\\n\\nContent", "What is BRCA1?")
    '# BRCA1 Gene\\n\\nContent'
    >>> preprocess_markdown("  \\n\\n# Title\\nBody", "")
    '# Title\\nBody'
    >>> preprocess_markdown("Query text here\\n# Heading", "Query text here")
    '# Heading'
    >>> preprocess_markdown("", "some query")
    ''
    >>> preprocess_markdown("Question: Some prompt text\\n\\n# Real Title\\nContent", "")
    '# Real Title\\nContent'
    >>> preprocess_markdown("Question: Instructions here\\nMore text\\n\\n# Heading\\nBody", "")
    '# Heading\\nBody'
    """
    if not markdown:
        return ""

    result = markdown.strip()

    # Check if the markdown starts with the query (case-insensitive comparison)
    if query:
        query_normalized = query.strip()
        # Check for exact prefix match (with possible whitespace/newline after)
        if result.lower().startswith(query_normalized.lower()):
            # Remove the query prefix
            result = result[len(query_normalized):].lstrip()

    # Check for "Question:" preamble (Falcon includes prompt in response)
    # If the content starts with "Question:" and has a heading later, skip to the heading
    if result.lower().startswith("question:"):
        lines = result.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                # Found a heading - use from here onwards
                result = '\n'.join(lines[i:])
                break

    return result


def extract_title_from_markdown(markdown: str, max_length: int = 120) -> str:
    """Extract a title from markdown content.

    Heuristic:
    1. Remove leading # characters and whitespace
    2. Take everything up to the next # (another heading)
    3. If too long, truncate at first newline
    4. If still too long, truncate at max_length

    Args:
        markdown: Raw markdown content
        max_length: Maximum title length (default 120)

    Returns:
        Extracted title string, or empty string if extraction fails

    >>> extract_title_from_markdown("# Hello World\\n\\nSome content")
    'Hello World'
    >>> extract_title_from_markdown("# My Title\\n## Section 1\\nContent")
    'My Title'
    >>> extract_title_from_markdown("## Subsection\\nContent here")
    'Subsection'
    >>> extract_title_from_markdown("No heading here")
    ''
    >>> extract_title_from_markdown("# " + "A" * 200, max_length=50)
    'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    >>> extract_title_from_markdown("")
    ''
    >>> extract_title_from_markdown("# Title with special: chars & more!\\n\\nBody")
    'Title with special: chars & more!'
    """
    if not markdown or not markdown.strip():
        return ""

    # Find the first heading
    lines = markdown.split('\n')
    title_line = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            # Found a heading - extract the text after the #s
            title_line = stripped.lstrip('#').strip()
            break

    if not title_line:
        return ""

    # If there's a newline in the remaining text, take only the first line
    if '\n' in title_line:
        title_line = title_line.split('\n')[0].strip()

    # Truncate if still too long
    if len(title_line) > max_length:
        title_line = title_line[:max_length].strip()

    return title_line


class CacheManager:
    """Manages file-based caching of research results."""

    def __init__(self, config: CacheConfig):
        """Initialize cache manager with configuration."""
        self.config = config
        # Default to ~/.deep_research_cache if not specified
        cache_dir = config.directory or str(Path.home() / ".deep_research_cache")
        self.cache_dir = Path(cache_dir)
        if config.enabled:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

    def _sanitize_for_filename(self, text: str, max_length: int = 30) -> str:
        """Sanitize text for use in filename."""
        # Convert to lowercase and replace spaces/special chars with hyphens
        sanitized = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
        sanitized = re.sub(r'\s+', '-', sanitized)
        sanitized = re.sub(r'-+', '-', sanitized)  # Remove multiple consecutive hyphens
        sanitized = sanitized.strip('-')  # Remove leading/trailing hyphens

        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length].rstrip('-')

        return sanitized or "query"  # Fallback if empty

    def _get_cache_filename(self, query: str, provider: str, model: Optional[str] = None, provider_params: Optional[dict] = None) -> str:
        """Generate human-readable cache filename."""
        # Create hash for uniqueness including all parameters
        content_parts = [f"{provider}:{query}"]

        if model:
            content_parts.append(f"model:{model}")

        if provider_params:
            # Sort params for consistent hashing
            sorted_params = sorted(provider_params.items())
            params_str = ",".join(f"{k}={v}" for k, v in sorted_params)
            content_parts.append(f"params:{params_str}")

        content = "|".join(content_parts)
        full_hash = hashlib.sha256(content.encode()).hexdigest()
        hash_suffix = full_hash[-8:]  # Last 8 chars of hash

        # Sanitize query for filename
        sanitized_query = self._sanitize_for_filename(query)

        # Combine: provider-query-hash.json
        return f"{provider}-{sanitized_query}-{hash_suffix}.json"

    def _get_cache_path(self, query: str, provider: str, model: Optional[str] = None, provider_params: Optional[dict] = None) -> Path:
        """Get cache file path for given query and provider."""
        filename = self._get_cache_filename(query, provider, model, provider_params)
        return self.cache_dir / filename

    async def get(self, query: str, provider: str, model: Optional[str] = None, provider_params: Optional[dict] = None) -> Optional[ResearchResult]:
        """Get cached result if available."""
        if not self.config.enabled:
            return None

        cache_path = self._get_cache_path(query, provider, model, provider_params)

        if not cache_path.exists():
            return None

        # Load cached result (no expiration check)
        try:
            async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                result = ResearchResult(**data)
                result.cached = True
                return result
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            # If cache file is corrupted, remove it
            if cache_path.exists():
                cache_path.unlink()
            return None

    async def set(self, query: str, provider: str, result: ResearchResult, model: Optional[str] = None, provider_params: Optional[dict] = None) -> None:
        """Cache a research result."""
        if not self.config.enabled:
            return

        cache_path = self._get_cache_path(query, provider, model, provider_params)

        # Create a copy to avoid modifying original
        cached_result = result.model_copy()
        cached_result.cached = False  # Don't store cached flag

        async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
            content = cached_result.model_dump_json(indent=2)
            await f.write(content)

    def clear_cache(self) -> int:
        """Clear all cached files and return count of files removed."""
        if not self.config.enabled or not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    def list_cache_files(self) -> list[Path]:
        """List all cache files."""
        if not self.config.enabled or not self.cache_dir.exists():
            return []

        return list(self.cache_dir.glob("*.json"))

    def get_cache_info(self) -> list[dict[str, Any]]:
        """Get detailed info for all cached files.

        Returns list of dicts with keys:
            - path: Path to cache file
            - name: Filename
            - size_bytes: File size in bytes
            - modified: Last modified datetime
            - query: Original query (from JSON content)
            - provider: Provider name
            - model: Model used (if any)
            - start_time: When research started
            - duration_seconds: How long research took

        >>> from deep_research_client.cache import CacheManager
        >>> from deep_research_client.models import CacheConfig
        >>> cm = CacheManager(CacheConfig(enabled=False))
        >>> cm.get_cache_info()
        []
        """
        if not self.config.enabled or not self.cache_dir.exists():
            return []

        results = []
        for cache_file in self.cache_dir.glob("*.json"):
            info = {
                "path": cache_file,
                "name": cache_file.name,
                "size_bytes": cache_file.stat().st_size,
                "modified": cache_file.stat().st_mtime,
            }

            # Try to load JSON content for additional metadata
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    info["query"] = data.get("query", "")
                    info["provider"] = data.get("provider", "")
                    info["model"] = data.get("model")
                    info["start_time"] = data.get("start_time")
                    info["duration_seconds"] = data.get("duration_seconds")
                    info["citation_count"] = len(data.get("citations", []))
            except (json.JSONDecodeError, IOError):
                info["query"] = "(unable to read)"
                info["provider"] = ""

            results.append(info)

        # Sort by modified time, newest first
        def get_modified(x: dict[str, Any]) -> float:
            return float(x.get("modified") or 0)
        results.sort(key=get_modified, reverse=True)
        return results

    def _extract_snippets(self, text: str, keyword: str, context_chars: int = 60, max_snippets: int = 3) -> list[str]:
        """Extract snippets of text around keyword matches.

        Args:
            text: Text to search in
            keyword: Keyword to find (case-insensitive)
            context_chars: Number of characters before/after match to include
            max_snippets: Maximum number of snippets to return

        Returns:
            List of snippet strings with ... markers

        >>> from deep_research_client.cache import CacheManager
        >>> from deep_research_client.models import CacheConfig
        >>> cm = CacheManager(CacheConfig(enabled=False))
        >>> cm._extract_snippets("The quick brown fox jumps over", "fox", context_chars=10)
        ['...wn fox ju...']
        >>> cm._extract_snippets("No match here", "xyz", context_chars=10)
        []
        """
        if not text or not keyword:
            return []

        text_lower = text.lower()
        keyword_lower = keyword.lower()
        snippets: list[str] = []
        search_start = 0

        while len(snippets) < max_snippets:
            pos = text_lower.find(keyword_lower, search_start)
            if pos == -1:
                break

            # Calculate snippet bounds
            start = max(0, pos - context_chars)
            end = min(len(text), pos + len(keyword) + context_chars)

            # Extract snippet
            snippet = text[start:end]

            # Clean up whitespace (normalize newlines/tabs to spaces)
            snippet = ' '.join(snippet.split())

            # Add ellipsis markers
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            snippets.append(f"{prefix}{snippet}{suffix}")

            # Move past this match
            search_start = pos + len(keyword)

        return snippets

    def search_cache(self, keyword: str, context_chars: int = 60, max_snippets: int = 3) -> list[dict[str, Any]]:
        """Search cached files for keyword in query or content.

        Args:
            keyword: Case-insensitive keyword to search for
            context_chars: Characters of context around matches
            max_snippets: Maximum snippets per field (query/content)

        Returns:
            List of cache info dicts that match the keyword

        >>> from deep_research_client.cache import CacheManager
        >>> from deep_research_client.models import CacheConfig
        >>> cm = CacheManager(CacheConfig(enabled=False))
        >>> cm.search_cache("test")
        []
        """
        if not self.config.enabled or not self.cache_dir.exists():
            return []

        keyword_lower = keyword.lower()
        results = []

        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    data = json.loads(content)

                # Search in query and markdown content
                query = data.get("query", "")
                markdown = data.get("markdown", "")

                match_in_query = keyword_lower in query.lower()
                match_in_content = keyword_lower in markdown.lower()

                if match_in_query or match_in_content:
                    info = {
                        "path": cache_file,
                        "name": cache_file.name,
                        "size_bytes": cache_file.stat().st_size,
                        "modified": cache_file.stat().st_mtime,
                        "query": query,
                        "provider": data.get("provider", ""),
                        "model": data.get("model"),
                        "start_time": data.get("start_time"),
                        "duration_seconds": data.get("duration_seconds"),
                        "citation_count": len(data.get("citations", [])),
                        "match_in_query": match_in_query,
                        "match_in_content": match_in_content,
                        "query_snippets": self._extract_snippets(query, keyword, context_chars, max_snippets) if match_in_query else [],
                        "content_snippets": self._extract_snippets(markdown, keyword, context_chars, max_snippets) if match_in_content else [],
                    }
                    results.append(info)

            except (json.JSONDecodeError, IOError):
                continue

        # Sort by modified time, newest first
        def get_modified(x: dict[str, Any]) -> float:
            return float(x.get("modified") or 0)
        results.sort(key=get_modified, reverse=True)
        return results

    def export_for_browser(self, include_content: bool = False) -> list[dict[str, Any]]:
        """Export cache data in format suitable for linkml-browser.

        Returns a list of dicts with standardized fields for faceted browsing.
        Each entry includes: id, filename, provider, model, title, keywords,
        query_preview, date, size, duration, citation_count.

        Args:
            include_content: If True, include full markdown and citations for page generation

        >>> from deep_research_client.cache import CacheManager
        >>> from deep_research_client.models import CacheConfig
        >>> cm = CacheManager(CacheConfig(enabled=False))
        >>> cm.export_for_browser()
        []
        """
        from datetime import datetime

        if not self.config.enabled or not self.cache_dir.exists():
            return []

        results: list[dict[str, Any]] = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract date from file modification time
                modified_ts = cache_file.stat().st_mtime
                modified_dt = datetime.fromtimestamp(modified_ts)
                date_str = modified_dt.strftime("%Y-%m-%d")

                # Extract query preview (first 200 chars, cleaned)
                query = data.get("query", "")
                query_preview = ' '.join(query.split())[:200]
                if len(query) > 200:
                    query_preview += "..."

                # Preprocess markdown (strips query prefix if present, e.g. Falcon)
                raw_markdown = data.get("markdown", "")
                markdown = preprocess_markdown(raw_markdown, query)

                # Extract title - use stored title, or derive from preprocessed markdown
                title = data.get("title") or ""
                if not title and markdown:
                    title = extract_title_from_markdown(markdown)

                # Calculate word count from markdown
                word_count = len(markdown.split()) if markdown else 0

                # Build browser-friendly record
                record: dict[str, Any] = {
                    "id": cache_file.stem,  # filename without extension
                    "filename": cache_file.name,
                    "provider": data.get("provider", "unknown"),
                    "model": data.get("model") or "default",
                    "title": title,
                    "keywords": data.get("keywords") or [],  # Array for AND faceting
                    "query_preview": query_preview,
                    "date": date_str,
                    "size_kb": round(cache_file.stat().st_size / 1024, 1),
                    "duration_seconds": data.get("duration_seconds"),
                    "citation_count": len(data.get("citations", [])),
                    "word_count": word_count,
                    "has_title": bool(title),
                    "has_keywords": bool(data.get("keywords")),
                    "template_file": data.get("template_file") or "",
                }

                # Add author/contributors if present
                query_metadata = data.get("query_metadata")
                if query_metadata:
                    record["author"] = query_metadata.get("author") or ""
                    record["contributors"] = query_metadata.get("contributors") or []
                else:
                    record["author"] = ""
                    record["contributors"] = []

                # Include full content if requested (for individual page generation)
                if include_content:
                    record["markdown"] = markdown  # Use preprocessed markdown
                    record["citations"] = data.get("citations", [])
                    record["template_variables"] = data.get("template_variables") or {}
                    record["template_file"] = data.get("template_file") or ""
                    record["provider_config"] = data.get("provider_config") or {}

                results.append(record)

            except (json.JSONDecodeError, IOError):
                continue

        # Sort by date, newest first
        results.sort(key=lambda x: str(x.get("date", "")), reverse=True)
        return results