"""Configurable selection policy for provider artifact bundles.

Providers that return a bundle of files — an OpenScientist artifacts ZIP, say —
have to decide which members become :class:`~.models.ResearchArtifact` entries
and which are runtime scaffolding. That decision used to be a hardcoded set of
constants inside one provider, which meant a caller who wanted the agent
transcripts (they are provenance, not noise, for some downstream consumers) had
no way to ask for them short of editing the library.

:class:`ArtifactSelectionPolicy` makes the decision data rather than code. The
defaults reproduce the previous curated behaviour, with two scaffolding
directories added to it (``.codex/`` and ``__pycache__/``, which the hardcoded
list missed); every layer can be widened or narrowed by the caller, and other
bundle-shaped providers can reuse the same policy instead of growing their own
constants.

Precedence, highest first:

1. ``exclude_globs`` — an explicit deny always wins.
2. ``max_bytes`` — the size cap always applies, including to explicit includes,
   because it is what keeps a bundle from being read into memory. Raise the cap
   rather than trying to glob around it, up to the 50 MB ceiling that the
   ``artifact_max_bytes`` field enforces; past that there is no knob, by
   design. The floor belongs to the policy rather than that field, whose own
   minimum is 1: a policy built directly with a cap of 0 or less keeps
   nothing, a zero-byte member included.
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

# Default per-artifact size cap. Declared here and referenced by the pydantic
# field, so the value cannot drift between the policy and the params model.
DEFAULT_MAX_BYTES: int = 5 * 1024 * 1024

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

# Agent working directories and dependency trees. Matched as a path
# *segment*, at any depth: a ``.claude/settings.json`` nested under a
# workspace directory is the same scaffolding as one at the bundle root, and
# both are frequently ``.json``, which would otherwise clear the allowlist.
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


def split_name_list(value: object) -> tuple[str, ...]:
    """Read a list-valued artifact setting, accepting a comma-separated string.

    A bare string is the trap this exists for. ``tuple("*.json")`` is
    ``('*', '.', 'j', 's', 'o', 'n')``, and since ``include_globs`` sits above
    every default deny and ``fnmatch(anything, "*")`` is true, that single
    stray character preserves the entire bundle — archives and transcripts
    included. Splitting instead means a string says what a CLI user means by
    it, and makes the duck-typed door agree with the pydantic one, which
    applies the same rule.

    Args:
        value: A collection of names, a comma-separated string, or None.

    Returns:
        The names, empty for None or anything that is not iterable as names.

    Example:
        >>> split_name_list("*.json")
        ('*.json',)
        >>> split_name_list("a/*, b/*")
        ('a/*', 'b/*')
        >>> split_name_list(["a/*", "b/*"])
        ('a/*', 'b/*')
        >>> split_name_list(None)
        ()
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    return ()


def _with_trailing_slashes(directories: Iterable[str]) -> tuple[str, ...]:
    """Normalize directory names for segment matching.

    Lowercased and slash-terminated, so both sides of the comparison are
    normalized the way every other matcher in this module does it — a caller
    passing ``".Codex/"`` would otherwise match nothing at all.

    Args:
        directories: Directory names, any case, with or without a trailing
            slash. A bare string is rejected rather than iterated as
            characters.

    Returns:
        The normalized names.

    Raises:
        TypeError: If given a single string instead of a collection.
    """
    if isinstance(directories, str):
        raise TypeError(
            "directories must be a collection of names, not a single string; "
            f"pass [{directories!r}] rather than {directories!r}"
        )
    return tuple(
        directory.lower() if directory.endswith("/") else f"{directory.lower()}/"
        for directory in directories
    )


#: The scaffolding names already lowercased and slash-terminated, derived by
#: the same helper the policy normalizes with. Pass this to
#: :func:`is_under_normalized_directory` from a hot loop rather than
#: re-normalizing the defaults on every member. Derived rather than written
#: out, so the constant and the policy cannot come to disagree.
DEFAULT_SCAFFOLDING_DIRECTORIES: tuple[str, ...] = _with_trailing_slashes(
    DEFAULT_SCAFFOLDING_PREFIXES
)


#: Every slug :meth:`ArtifactSelectionPolicy.decide` can put on a decision.
#: Exported so a caller branching on ``rule`` has something to check against,
#: and so the set cannot drift from the returns the way a hand-written list in
#: a docstring did.
ARTIFACT_RULES: frozenset[str] = frozenset(
    {
        "archive",
        "exclude_glob",
        "extension",
        "extension_not_allowed",
        "include_glob",
        "media_type",
        "provider_deny",
        "runtime",
        "scaffolding",
        "size_cap",
    }
)


@dataclass(frozen=True)
class ArtifactDecision:
    """Whether one bundle member becomes an artifact, and on what grounds.

    The reason exists so a skipped file can be logged with the rule that
    skipped it — "which knob do I turn to get this file" is the question a
    caller actually has. ``rule`` is the same thing as a stable slug, so a
    caller can treat one outcome differently without matching on prose — the
    provider does exactly that to log a size-cap skip louder than the rest.

    :data:`ARTIFACT_RULES` is the set a caller may branch on. Note that both
    ``extension`` (a keep) and ``extension_not_allowed`` (the matching deny)
    are in it, and so is ``media_type`` — the rule that keeps an image whose
    suffix is not in the allowlist, which is the interesting keep rather than
    the obvious one.
    """

    keep: bool
    reason: str
    rule: str = ""

    def __post_init__(self) -> None:
        """Reject a slug that is not in :data:`ARTIFACT_RULES`.

        Validated here rather than by a test that enumerates outcomes,
        because such a list is a second hand-maintained copy of the same
        thing — which is how ``media_type`` came to be undocumented. A new
        branch in :meth:`ArtifactSelectionPolicy.decide` now fails the moment
        any test reaches it.

        Raises:
            ValueError: If ``rule`` is neither empty nor a known slug.
        """
        if self.rule and self.rule not in ARTIFACT_RULES:
            raise ValueError(
                f"unknown decision rule {self.rule!r}; add it to ARTIFACT_RULES"
            )

    def __bool__(self) -> bool:
        return self.keep


@dataclass(frozen=True)
class ArtifactSelectionPolicy:
    """A resolved, provider-agnostic answer to "is this member an artifact?".

    Args:
        max_bytes: Largest uncompressed size preserved for a single member.
            0 or less keeps nothing at all, including a zero-byte member.
        allowed_extensions: Extension allowlist, lowercase and dot-prefixed.
        archive_extensions: Extensions refused as nested archives.
        scaffolding_prefixes: Directory names treated as agent working
            state, matched as a whole path segment at any depth rather than
            only at the bundle root. Normalized to lowercase and a trailing
            slash by ``__post_init__``, which :meth:`decide` then relies on —
            a policy reconstructed without ``__init__`` (unpickling,
            ``object.__new__``) would stop matching scaffolding.
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

    def __post_init__(self) -> None:
        """Normalize scaffolding names so segment matching cannot be broken.

        Matching relies on each name ending in a slash: without it,
        ``scaffolding_prefixes=("logs",)`` would drop ``run/logs_summary.csv``
        as a directory. Normalizing here makes that structural rather than a
        convention a caller has to know.

        Raises:
            TypeError: If ``scaffolding_prefixes`` is a single string.
                Iterating one yields characters, which match wrongly rather
                than not at all — see :func:`is_under_directory`.
        """
        object.__setattr__(
            self, "scaffolding_prefixes", _with_trailing_slashes(self.scaffolding_prefixes)
        )

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
        # Explicitly against None rather than falsiness: this is documented
        # as duck-typed, and a params object declaring ``Optional[int] = None``
        # would otherwise reach ``size > None`` at decision time — but a
        # deliberate 0 means "keep nothing", not "use the default".
        max_bytes = getattr(params, "artifact_max_bytes", None)
        if max_bytes is None:
            max_bytes = DEFAULT_MAX_BYTES
        extra = _normalize_extensions(
            split_name_list(getattr(params, "artifact_extra_extensions", ()))
        )
        return cls(
            max_bytes=max_bytes,
            allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS | extra,
            include_globs=split_name_list(
                getattr(params, "artifact_include_globs", ())
            ),
            exclude_globs=split_name_list(
                getattr(params, "artifact_exclude_globs", ())
            ),
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
                Each entry is normalized here before comparison, so one
                differing only in case or a leading ``./`` still matches, and
                passing an already-normalized set is idempotent rather than
                free. Build the set once per bundle (see
                :func:`normalize_member_path`) so at least its construction is
                not repeated per member. Unlike the scaffolding names, these
                are not hoisted onto a normalized fast path: the set holds one
                entry in practice, and a second code path would cost more than
                it saves.

        Returns:
            The decision, carrying the rule that produced it.
        """
        normalized = normalize_member_path(name)
        basename = PurePosixPath(normalized).name

        if _matches_any(normalized, self.exclude_globs):
            return ArtifactDecision(False, "matched artifact_exclude_globs", rule="exclude_glob")

        if self.max_bytes <= 0:
            # Separate branch so the reason is true: at a zero cap a zero-byte
            # member is dropped, and "0 bytes exceeds 0" would be a lie in a
            # string built to be shown to a user.
            return ArtifactDecision(
                False,
                f"artifact_max_bytes is {self.max_bytes}: keeping nothing",
                rule="size_cap",
            )
        if size > self.max_bytes:
            return ArtifactDecision(
                False,
                f"{size} bytes exceeds artifact_max_bytes ({self.max_bytes})",
                rule="size_cap",
            )

        if any(normalize_member_path(denied) == normalized for denied in provider_deny):
            return ArtifactDecision(
                False, "already returned as the report body", rule="provider_deny"
            )

        if _matches_any(normalized, self.include_globs):
            return ArtifactDecision(True, "matched artifact_include_globs", rule="include_glob")

        # ``scaffolding_prefixes`` is normalized by __post_init__, so compare
        # against it directly rather than re-normalizing once per member.
        if is_under_normalized_directory(normalized, self.scaffolding_prefixes):
            return ArtifactDecision(
                False, "agent scaffolding directory", rule="scaffolding"
            )

        suffix = PurePosixPath(normalized).suffix
        if suffix in self.archive_extensions:
            return ArtifactDecision(False, "nested archive", rule="archive")

        if not self.keep_runtime:
            if basename.endswith(self.runtime_suffixes):
                return ArtifactDecision(
                    False,
                    "runtime log (set artifact_keep_runtime to keep)",
                    rule="runtime",
                )
            if any(fragment in basename for fragment in self.runtime_name_fragments):
                return ArtifactDecision(
                    False,
                    "runtime transcript (set artifact_keep_runtime to keep)",
                    rule="runtime",
                )

        if suffix in self.effective_extensions:
            return ArtifactDecision(True, "allowed extension", rule="extension")

        # ``normalized`` like every other rule here; mimetypes lowercases the
        # suffix itself, so this is consistency rather than a fix.
        media_type = mimetypes.guess_type(normalized)[0]
        if media_type is not None and media_type.startswith("image/"):
            return ArtifactDecision(True, "image media type", rule="media_type")

        return ArtifactDecision(
            False,
            f"extension {suffix or '(none)'} not in allowlist "
            "(add it with artifact_extra_extensions)",
            rule="extension_not_allowed",
        )


def is_under_directory(name: str, directories: Iterable[str]) -> bool:
    """Return whether a path lies under one of the named directories.

    Matched on a whole path segment at any depth, so
    ``workspace/.claude/settings.json`` counts as being under ``.claude/``
    while ``run/logs_summary.csv`` does not count as being under ``logs/``.

    Shared by the selection policy and by a provider's report-picking path, so
    the two cannot drift into disagreeing about what scaffolding is.

    Args:
        name: Member path, normalized here if it is not already.
        directories: Directory names, with or without a trailing slash.

    Returns:
        Whether any of them names a directory on the path.

    Raises:
        TypeError: If ``directories`` is a single string. Iterating one yields
            characters, and a lone ``"a"`` or ``"."`` matches any path with a
            matching segment — so a bare string does not fail to match, it
            matches wrongly and only on some paths.

    Example:
        >>> is_under_directory("workspace/.claude/skills/x.md", [".claude/"])
        True
        >>> is_under_directory("run/logs_summary.csv", ["logs"])
        False
        >>> is_under_directory("run/logs/out.csv", ["logs"])
        True
    """
    return is_under_normalized_directory(
        normalize_member_path(name), _with_trailing_slashes(directories)
    )


def is_under_normalized_directory(
    normalized_name: str, normalized_directories: Iterable[str]
) -> bool:
    """Segment match with both sides already normalized.

    The fast path behind :func:`is_under_directory`, for a caller looping over
    a bundle that has normalized the path itself and holds a pre-normalized
    directory tuple such as :data:`DEFAULT_SCAFFOLDING_DIRECTORIES`. Nothing is
    re-derived per call.

    Args:
        normalized_name: Path through :func:`normalize_member_path`.
        normalized_directories: Names already lowercased and slash-terminated.

    Returns:
        Whether any of them names a directory on the path.

    Raises:
        TypeError: If given a single string. Iterating one yields characters,
            and a lone ``"."`` matches any path with a dot-segment — so a bare
            string does not fail to match, it matches wrongly and only on some
            paths.

    Example:
        >>> is_under_normalized_directory("a/.claude/x.md", (".claude/",))
        True
    """
    if isinstance(normalized_directories, str):
        raise TypeError(
            "normalized_directories must be a collection of names, not a "
            f"single string; pass [{normalized_directories!r}] rather than "
            f"{normalized_directories!r}"
        )
    return any(
        normalized_name.startswith(directory) or f"/{directory}" in normalized_name
        for directory in normalized_directories
    )


def normalize_member_path(name: str) -> str:
    """Normalize a bundle member path for comparison.

    Lowercased, no leading slash, ``/`` read as the separator — the form
    every rule in :meth:`ArtifactSelectionPolicy.decide` matches against.
    Exported so a caller can normalize its deny set once per bundle rather
    than once per member.

    Backslashes are *not* converted: they are ordinary characters here, so a
    member written ``workspace\\.claude\\x.json`` would clear every
    scaffolding and glob rule. The ZIP format mandates forward slashes
    (APPNOTE 4.4.17.1), so this does not arise for the bundles in play, and
    :func:`~.models.sanitize_artifact_filename` handles separators on the way
    out regardless.

    Args:
        name: Bundle-relative member path.

    Returns:
        The normalized path.

    Example:
        >>> normalize_member_path("/Provenance/Iter1.JSON")
        'provenance/iter1.json'
    """
    return PurePosixPath(name).as_posix().lstrip("/").lower()


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
