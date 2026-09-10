"""Configurable selection policy for provider artifact bundles.

Providers that return a bundle of files — an OpenScientist artifacts ZIP, say —
have to decide which members become :class:`~.models.ResearchArtifact` entries
and which are runtime scaffolding. That decision used to be a hardcoded set of
constants inside one provider, which meant a caller who wanted the agent
transcripts (they are provenance, not noise, for some downstream consumers) had
no way to ask for them short of editing the library.

:class:`ArtifactSelectionPolicy` makes the decision data rather than code. The
defaults reproduce the previous curated behaviour exactly; every layer of it can
be widened or narrowed by the caller, and other bundle-shaped providers can
reuse the same policy instead of growing their own constants.

Precedence, highest first:

1. ``exclude_globs`` — an explicit deny always wins.
2. ``max_bytes`` — the size cap always applies, including to explicit includes,
   because it is what keeps a bundle from being read into memory. Raise the cap
   rather than trying to glob around it.
3. ``provider_deny`` — members the provider has already consumed (the markdown
   report it returned as the result body).
4. ``include_globs`` — an explicit allow bypasses every remaining default deny.
5. The default denies: scaffolding directories, archives, and — unless
   ``keep_runtime`` is set — logs, transcripts, and captured stdout/stderr.
6. The extension allowlist (plus ``extra_extensions``, plus any ``image/*``
   media type).

Globs are :mod:`fnmatch` patterns matched against the whole bundle-relative
path, lowercased and with ``/`` separators. ``*`` crosses directory separators,
so ``*.json`` matches ``provenance/iter1_transcript.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fnmatch import fnmatch
import mimetypes
from pathlib import PurePosixPath
from typing import Iterable

# Extensions kept by default: figures, small structured data, rendered reports.
DEFAULT_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".csv",
        ".gif",
        ".htm",
        ".html",
        ".jpeg",
        ".jpg",
        ".json",
        ".md",
        ".pdf",
        ".png",
        ".svg",
        ".tsv",
        ".webp",
    }
)

# Extensions added when ``keep_runtime`` is set, so that switch is coherent on
# its own: without them a kept ``.log`` would survive the noise filter and then
# fall at the extension allowlist.
RUNTIME_EXTENSIONS: frozenset[str] = frozenset({".jsonl", ".log", ".ndjson", ".txt"})

# Never unpacked: a nested archive is not an inspectable artifact.
DEFAULT_ARCHIVE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".7z",
        ".bz2",
        ".gz",
        ".tar",
        ".tgz",
        ".xz",
        ".zip",
    }
)

# Agent working directories and dependency trees, matched as path prefixes.
DEFAULT_SCAFFOLDING_PREFIXES: tuple[str, ...] = (
    ".cache/",
    ".claude/",
    ".codex/",
    ".git/",
    ".ipynb_checkpoints/",
    ".venv/",
    "__macosx/",
    "__pycache__/",
    "cache/",
    "node_modules/",
)

DEFAULT_RUNTIME_SUFFIXES: tuple[str, ...] = (".log", ".tmp")

DEFAULT_RUNTIME_NAME_FRAGMENTS: tuple[str, ...] = (
    "stderr",
    "stdout",
    "transcript",
)


@dataclass(frozen=True)
class ArtifactDecision:
    """Whether one bundle member becomes an artifact, and on what grounds.

    The reason exists so a skipped file can be logged with the rule that
    skipped it — "which knob do I turn to get this file" is the question a
    caller actually has.
    """

    keep: bool
    reason: str

    def __bool__(self) -> bool:
        return self.keep


@dataclass(frozen=True)
class ArtifactSelectionPolicy:
    """A resolved, provider-agnostic answer to "is this member an artifact?".

    Args:
        max_bytes: Largest uncompressed size preserved for a single member.
        allowed_extensions: Extension allowlist, lowercase and dot-prefixed.
        archive_extensions: Extensions refused as nested archives.
        scaffolding_prefixes: Path prefixes treated as agent working state.
        runtime_suffixes: Filename suffixes treated as runtime logs.
        runtime_name_fragments: Substrings in a basename marking runtime output.
        include_globs: Patterns force-kept, bypassing every default deny.
        exclude_globs: Patterns force-dropped, beating everything else.
        keep_runtime: Keep logs, transcripts, and captured stdout/stderr, and
            extend the allowlist with :data:`RUNTIME_EXTENSIONS`.

    Example:
        >>> policy = ArtifactSelectionPolicy(max_bytes=1024)
        >>> bool(policy.decide("provenance/iter1_transcript.json", 100))
        False
        >>> keeping = policy.with_overrides(keep_runtime=True)
        >>> bool(keeping.decide("provenance/iter1_transcript.json", 100))
        True
        >>> bool(keeping.decide("provenance/iter1_transcript.json", 99999))
        False
    """

    max_bytes: int
    allowed_extensions: frozenset[str] = DEFAULT_ALLOWED_EXTENSIONS
    archive_extensions: frozenset[str] = DEFAULT_ARCHIVE_EXTENSIONS
    scaffolding_prefixes: tuple[str, ...] = DEFAULT_SCAFFOLDING_PREFIXES
    runtime_suffixes: tuple[str, ...] = DEFAULT_RUNTIME_SUFFIXES
    runtime_name_fragments: tuple[str, ...] = DEFAULT_RUNTIME_NAME_FRAGMENTS
    include_globs: tuple[str, ...] = ()
    exclude_globs: tuple[str, ...] = ()
    keep_runtime: bool = False

    @classmethod
    def from_params(cls, params: object) -> "ArtifactSelectionPolicy":
        """Build a policy from a params object carrying the artifact knobs.

        Reads the fields defined by
        :class:`~.provider_params.ArtifactSelectionParams` duck-typed, so a
        provider whose params model only sets some of them still works.

        Args:
            params: Provider params exposing ``artifact_*`` attributes.

        Returns:
            The resolved policy.
        """
        max_bytes = getattr(params, "artifact_max_bytes", 5 * 1024 * 1024)
        extra = _normalize_extensions(getattr(params, "artifact_extra_extensions", ()))
        return cls(
            max_bytes=max_bytes,
            allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS | extra,
            include_globs=tuple(getattr(params, "artifact_include_globs", ()) or ()),
            exclude_globs=tuple(getattr(params, "artifact_exclude_globs", ()) or ()),
            keep_runtime=bool(getattr(params, "artifact_keep_runtime", False)),
        )

    def with_overrides(self, **changes: object) -> "ArtifactSelectionPolicy":
        """Return a copy of this policy with the given fields replaced.

        Args:
            **changes: Field names and values to override.

        Returns:
            A new policy; this one is unchanged.
        """
        return replace(self, **changes)  # type: ignore[arg-type]

    @property
    def effective_extensions(self) -> frozenset[str]:
        """Extensions kept, including the runtime set when ``keep_runtime``."""
        if self.keep_runtime:
            return self.allowed_extensions | RUNTIME_EXTENSIONS
        return self.allowed_extensions

    def decide(
        self,
        name: str,
        size: int,
        provider_deny: Iterable[str] = (),
    ) -> ArtifactDecision:
        """Decide whether one bundle member should be preserved.

        Args:
            name: Bundle-relative member path.
            size: Uncompressed size in bytes.
            provider_deny: Member paths the provider has already consumed.

        Returns:
            The decision, carrying the rule that produced it.
        """
        normalized = PurePosixPath(name).as_posix().lstrip("/").lower()
        basename = PurePosixPath(normalized).name

        if _matches_any(normalized, self.exclude_globs):
            return ArtifactDecision(False, "matched artifact_exclude_globs")

        if size > self.max_bytes:
            return ArtifactDecision(
                False,
                f"{size} bytes exceeds artifact_max_bytes ({self.max_bytes})",
            )

        if name in set(provider_deny):
            return ArtifactDecision(False, "already returned as the report body")

        if _matches_any(normalized, self.include_globs):
            return ArtifactDecision(True, "matched artifact_include_globs")

        if normalized.startswith(self.scaffolding_prefixes):
            return ArtifactDecision(False, "agent scaffolding directory")

        suffix = PurePosixPath(normalized).suffix
        if suffix in self.archive_extensions:
            return ArtifactDecision(False, "nested archive")

        if not self.keep_runtime:
            if basename.endswith(self.runtime_suffixes):
                return ArtifactDecision(
                    False, "runtime log (set artifact_keep_runtime to keep)"
                )
            if any(fragment in basename for fragment in self.runtime_name_fragments):
                return ArtifactDecision(
                    False, "runtime transcript (set artifact_keep_runtime to keep)"
                )

        if suffix in self.effective_extensions:
            return ArtifactDecision(True, "allowed extension")

        media_type = mimetypes.guess_type(name)[0]
        if media_type is not None and media_type.startswith("image/"):
            return ArtifactDecision(True, "image media type")

        return ArtifactDecision(
            False,
            f"extension {suffix or '(none)'} not in allowlist "
            "(add it with artifact_extra_extensions)",
        )


def _matches_any(normalized_name: str, patterns: Iterable[str]) -> bool:
    """Return whether a normalized path matches any fnmatch pattern."""
    return any(fnmatch(normalized_name, pattern.lower()) for pattern in patterns)


def _normalize_extensions(extensions: Iterable[str]) -> frozenset[str]:
    """Normalize user-supplied extensions to lowercase, dot-prefixed form.

    Args:
        extensions: Extensions written with or without a leading dot.

    Returns:
        The normalized set, dropping empties.

    Example:
        >>> sorted(_normalize_extensions(["TXT", ".jsonl", ""]))
        ['.jsonl', '.txt']
    """
    normalized = set()
    for raw in extensions or ():
        cleaned = str(raw).strip().lower()
        if not cleaned:
            continue
        normalized.add(cleaned if cleaned.startswith(".") else f".{cleaned}")
    return frozenset(normalized)
