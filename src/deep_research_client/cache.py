"""Simple file-based caching for research results."""

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import aiofiles

from .models import ResearchResult, CacheConfig


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