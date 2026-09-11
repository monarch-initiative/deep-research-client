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
from collections.abc import Callable, Collection, Iterable, Mapping
from collections.abc import Set as AbstractSet

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


def _unique(names: Iterable[str]) -> tuple[str, ...]:
    """Drop repeats, keeping the first occurrence's position.

    Normalization creates duplicates the caller did not write: ``{"a", "a/"}``
    both become ``a/``, and ``"*.JSON,*.json"`` both become ``*.json``. Beyond
    the wasted match per member, a duplicate makes two policies that decide
    identically compare unequal — the mirror of the equal-sets property this
    module establishes elsewhere.

    Args:
        names: Names at whatever stage of normalization the caller has
            reached — merely stripped at the first call site, transformed at
            the others. Possibly with repeats.

    Returns:
        The distinct names, in first-seen order.

    Example:
        >>> _unique(["a/", "b/", "a/"])
        ('a/', 'b/')
    """
    return tuple(dict.fromkeys(names))


def require_name_collection(value: object) -> Collection[str]:
    """Type-check a value that must be a re-readable collection of names.

    The shape rule with no cleaning attached, so a caller that must not touch
    its entries can share it. :func:`split_name_list` calls this after
    handling the string case, then strips, dedupes and orders the result;
    :func:`checked_path_collection` calls it after *refusing* the string case
    and hands the collection straight back.

    The value is returned rather than copied, so a caller checking the same
    collection once per bundle member rebuilds nothing.

    Args:
        value: The value to check.

    Returns:
        ``value`` itself, narrowed to a collection.

    Raises:
        TypeError: For a mapping, ``bytes``, a one-shot iterator, or anything
            that is not a collection. :func:`split_name_list` explains why
            each is refused rather than read as best it can be.

    Example:
        >>> require_name_collection(["a", "b"])
        ['a', 'b']
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(
            "expected a list of names or a comma-separated string, not a "
            f"{type(value).__name__}; decode it to str first, because "
            "iterating it yields integers rather than names"
        )
    if isinstance(value, Mapping):
        raise TypeError(
            "expected a list of names or a comma-separated string, not a "
            f"{type(value).__name__}; its keys are unlikely to be the names meant"
        )
    if isinstance(value, Collection):
        return value
    if isinstance(value, Iterable):
        raise TypeError(
            "expected a list of names or a comma-separated string, not a "
            f"{type(value).__name__}; these names must be a re-readable "
            "collection, so wrap it in a list"
        )
    raise TypeError(
        "expected a list of names or a comma-separated string, not a "
        f"{type(value).__name__}: {value!r}"
    )


def _stripped_names(items: Iterable[str]) -> tuple[str, ...]:
    """Strip each name, drop the empties, and drop the repeats.

    Applied to every branch of :func:`split_name_list`, because an empty name
    is not harmless in every matcher that consumes one: ``"" in basename`` is
    true for every member, so a single blank entry in ``runtime_name_fragments``
    drops the whole bundle.

    Args:
        items: Raw names.

    Returns:
        The distinct non-empty names, in the order given.

    Example:
        >>> _stripped_names([" a/* ", "", "  "])
        ('a/*',)
    """
    return _unique(name for name in (item.strip() for item in items) if name)


def split_name_list(value: object) -> tuple[str, ...]:
    """Read a list-valued artifact setting, accepting a comma-separated string.

    A bare string is the trap this exists for. ``tuple("*.json")`` is
    ``('*', '.', 'j', 's', 'o', 'n')``, and since ``include_globs`` sits above
    every default deny and ``fnmatch(anything, "*")`` is true, that single
    stray character preserves the entire bundle — archives and transcripts
    included. Splitting instead means a string says what a CLI user means by
    it, and makes the duck-typed door agree with the pydantic one, which
    applies the same rule.

    A comma is therefore a separator and cannot appear inside a pattern —
    ``"data[a,b]/*"`` becomes two patterns that match nothing. Pass a list to
    use one; ``["data[a,b]/*"]`` is untouched.

    Anything that is neither a string nor a re-readable collection of names
    raises, rather than yielding no names. A setting silently reduced to
    nothing is the worst outcome available here: an empty ``exclude_globs``
    keeps every runtime log the caller asked to drop, and an empty
    ``include_globs`` leaves denied the file they meant to rescue. Three
    inputs therefore raise rather than being read as best they can:

    * A mapping, whose keys are unlikely to be the names meant.
    * ``bytes``, the one remaining string-like that is also an iterable of
      something else: ``tuple(b"*.json")`` is six integers.
    * A one-shot iterator such as a generator. The contract is a re-readable
      collection, so this is refused at every door rather than only where a
      second read is certain: a params object outlives the policy built from
      it, and the second reader would select nothing and say nothing. Refusing
      it uniformly costs a caller who really does read once — such as
      :func:`is_under_directory`, which consumes its argument inside the call
      — one ``list()``, and is the reason there is one rule here instead of a
      permissive door and a strict one. Wrap it in a list.

    Args:
        value: A collection of names, a comma-separated string, or None.
            An unordered collection is sorted, so this function's own return
            value is deterministic rather than dependent on set iteration
            order. It is *not* what makes two policies built from equal sets
            compare equal — every policy field is re-sorted afterwards by
            :func:`_normalized_names` or collapsed into a frozenset by
            :func:`_normalize_extensions`, because a transformation applied
            after this sort can reorder the result. Deleting the sort here
            leaves that equality property intact and breaks only this
            function's documented order.

    Returns:
        The names, stripped of surrounding whitespace, with empty and repeated
        entries dropped — the same answer whichever shape it was handed, so a list
        assembled from ``"stderr,".split(",")`` behaves like the string it
        came from. That matters because an empty name is not inert in every
        matcher: an empty runtime fragment is a substring of every basename.
        Empty for None, for an empty collection, for a collection of blanks,
        and for a string holding nothing but separators and whitespace.

    Raises:
        TypeError: If given a mapping, bytes, a one-shot iterator, or
            anything that is not a collection of names.

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
        return _stripped_names(value.split(","))
    checked = require_name_collection(value)
    # AbstractSet rather than the concrete types: a KeysView or a custom
    # Set is just as unordered, and sorting only real sets would make the
    # promise above true of some unordered collections and not others.
    if isinstance(checked, AbstractSet):
        return tuple(sorted(_stripped_names(str(item) for item in checked)))
    return _stripped_names(str(item) for item in checked)


def checked_path_collection(value: object, parameter: str) -> Collection[str]:
    """Check a collection of paths, without altering any of them.

    The sibling of :func:`split_name_list` for the one kind of name that is
    not a name. A deny entry is a member path, compared verbatim against a
    path that :func:`normalize_member_path` has lowercased and stripped of a
    leading slash — and of nothing else. So every part of the name-cleaning
    rule is wrong here, not just the comma:

    * Splitting ``"a,b/report.md"`` makes two entries that match nothing.
    * Stripping ``" analysis/report.md"`` makes one entry that matches
      nothing, because the member path it names keeps its space.

    Both turn a valid deny into a silent no-op, which is the failure the deny
    exists to prevent. Only the *shape* rule is shared, through
    :func:`require_name_collection`, and the collection is handed back
    untouched — no split, no strip, no dedupe, no sort, and no copy, so a
    caller checking it once per bundle member rebuilds nothing.

    Args:
        value: A collection of paths.
        parameter: Name of the parameter being read, for the message.

    Returns:
        ``value`` itself, narrowed to a collection.

    Raises:
        TypeError: If given a single string, or anything
            :func:`require_name_collection` refuses.

    Example:
        >>> checked_path_collection([" a/b.md", "a/b.md"], "provider_deny")
        [' a/b.md', 'a/b.md']
    """
    if isinstance(value, str):
        raise TypeError(
            f"{parameter} must be a collection of paths, not a single string; "
            f"iterating one yields characters, none of which equals a member "
            f"path, so nothing would be denied. Pass [{value!r}]."
        )
    return require_name_collection(value)


def _normalized_names(
    value: object, transform: Callable[[str], str]
) -> tuple[str, ...]:
    """Read a list setting, transform each name, and re-establish the order.

    The sort inside :func:`split_name_list` runs on the names as given, so any
    transformation applied afterwards can reorder them — lowercasing moves an
    uppercase name below its neighbours, and appending ``/`` (0x2F) moves a
    name above one sharing its prefix. Sorting again here means "the stored
    form is the sorted form" holds for every field rather than for whichever
    wrapper last had the bug.

    The sort condition lives here too, rather than being re-derived per
    wrapper: an unordered collection is sorted and a caller's own list order
    is left alone, which is one rule, not one per transform.

    This is the sort that establishes the policy-level property — two
    policies built from equal sets compare equal — because it runs on the
    stored form. :func:`split_name_list` sorts as well, for its own public
    contract, so a set-shaped setting is sorted twice at construction. That
    is a handful of names once per policy, and collapsing the two would cost
    the public function its documented order.

    Args:
        value: A collection of names or a comma-separated string.
        transform: Applied to each name after splitting.

    Returns:
        The distinct transformed names, sorted when ``value`` was unordered.

    Example:
        >>> _normalized_names({"B/*", "a/*"}, str.lower)
        ('a/*', 'b/*')
        >>> _normalized_names(["z", "a"], str.upper)
        ('Z', 'A')
    """
    names = _unique(transform(name) for name in split_name_list(value))
    return tuple(sorted(names)) if isinstance(value, AbstractSet) else names


def _lowercased(names: object) -> tuple[str, ...]:
    """Read a list setting and lowercase it for matching a normalized path.

    :func:`normalize_member_path` lowercases the path before any matcher sees
    it, so an uppercase suffix or fragment can never match — the same silent
    nothing-matches failure as a dotless extension.

    Args:
        names: A collection of names or a comma-separated string.

    Returns:
        The names, lowercased.

    Example:
        >>> _lowercased("Transcript, LOG")
        ('transcript', 'log')
    """
    return _normalized_names(names, str.lower)


def _as_directory_name(name: str) -> str:
    """Lowercase a directory name and give it the trailing slash."""
    lowered = name.lower()
    return lowered if lowered.endswith("/") else f"{lowered}/"


def _with_trailing_slashes(directories: object) -> tuple[str, ...]:
    """Normalize directory names for segment matching.

    Lowercased and slash-terminated, so both sides of the comparison are
    normalized the way every other matcher in this module does it — a caller
    passing ``".Codex/"`` would otherwise match nothing at all.

    Everything except the bare-string case is delegated to
    :func:`split_name_list`, so this field is not quietly the *loosest* one in
    a module built around being strict. Before that composition it accepted a
    mapping and a one-shot iterator that every other setting refuses, turned
    ``bytes`` into ``AttributeError: 'int' object has no attribute 'endswith'``
    rather than the corrective ``TypeError``, and did not sort an unordered
    collection — so two policies built from equal sets of scaffolding names
    were equal only when CPython happened to iterate them alike.

    Args:
        directories: Directory names, any case, with or without a trailing
            slash. A bare string is rejected rather than iterated as
            characters, and — unlike the user-facing lists — is not
            comma-split either: these names are not a CLI knob.

    Returns:
        The normalized names, distinct, and sorted in their final
        slash-terminated form when given an unordered collection.

    Raises:
        TypeError: If given a single string instead of a collection, or
            anything :func:`split_name_list` refuses.

    Example:
        >>> _with_trailing_slashes({"B", ".Codex/"})
        ('.codex/', 'b/')
        >>> _with_trailing_slashes({"logs", "logs.old"})
        ('logs.old/', 'logs/')
        >>> _with_trailing_slashes(["a", "a/"])
        ('a/',)
    """
    if isinstance(directories, str):
        raise TypeError(
            "directories must be a collection of names, not a single string; "
            f"pass [{directories!r}] rather than {directories!r}"
        )
    return _normalized_names(directories, _as_directory_name)


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

    Every collection field is normalized by ``__post_init__``, so what a field
    *accepts* is wider than what it is annotated as: any of the shapes
    :func:`split_name_list` reads, including a comma-separated string. The
    annotations describe what a *read* returns, which is always the normalized
    tuple or frozenset — widening them would push a union onto :meth:`decide`
    and onto every caller inspecting a policy, to describe a moment that only
    exists inside ``__init__``.

    The cost is that a type checker rejects the wider input at an annotated
    call site: ``ArtifactSelectionPolicy(include_globs="*.json")`` is an
    ``arg-type`` error even though it is exactly what the CLI produces and
    what the tests assert works. Nothing in this repository catches that,
    because mypy skips unannotated test bodies, ``from_params`` reads through
    ``getattr`` (which is ``Any``), and ``with_overrides`` takes
    ``**changes: object``. So treat the wider input as a runtime convenience
    for duck-typed and CLI-shaped callers; pass the annotated type from typed
    Python.

    Args:
        max_bytes: Largest uncompressed size preserved for a single member.
            0 or less keeps nothing at all, including a zero-byte member.
        allowed_extensions: Extension allowlist. Normalized to lowercase and
            dot-prefixed, so ``{"csv"}`` and ``{".CSV"}`` both mean ``.csv``
            rather than matching nothing.
        archive_extensions: Extensions refused as nested archives, normalized
            the same way as ``allowed_extensions``.
        scaffolding_prefixes: Directory names treated as agent working
            state, matched as a whole path segment at any depth rather than
            only at the bundle root. Normalized to lowercase and a trailing
            slash by ``__post_init__``, which :meth:`decide` then relies on —
            a policy reconstructed without ``__init__`` (unpickling,
            ``object.__new__``) would stop matching scaffolding.
        runtime_suffixes: Filename suffixes treated as runtime logs,
            lowercased to match the normalized path.
        runtime_name_fragments: Substrings in a basename marking runtime
            output, lowercased the same way.
        include_globs: Patterns force-kept, bypassing every default deny.
            Read by :func:`split_name_list` in ``__post_init__``, so a
            comma-separated string is a list of patterns here too, not a list
            of characters.
        exclude_globs: Patterns force-dropped, beating everything else. Read
            the same way as ``include_globs``.
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
        """Normalize every collection setting, whichever door it came through.

        A read of any of these fields is a normalized tuple or frozenset, so
        :meth:`decide` can match against them directly. Each matcher relies on
        a different part of that, and each fails silently without it:

        * Scaffolding names end in a slash, or ``scaffolding_prefixes=("logs",)``
          drops ``run/logs_summary.csv`` as a directory.
        * Extensions are lowercase and dot-prefixed, because
          :func:`normalize_member_path` lowercases the path first — so
          ``{"csv"}`` and ``{".CSV"}`` each match nothing at all.
        * Every list is split, or a bare string is iterated as characters.

        That last one is why the settings are all normalized here rather than
        in :meth:`from_params`. Doing it there left the constructor and
        :meth:`with_overrides` unguarded, and they fail in opposite directions
        on the same mistake: ``archive_extensions=".zip"`` turns
        ``suffix in self.archive_extensions`` into a substring test, and since
        an extensionless member has suffix ``""``, every one of them is
        dropped as a nested archive; ``allowed_extensions=".json"`` keeps
        every one of them instead. ``**changes: object`` on ``with_overrides``
        means the type checker sees neither. ``replace()`` re-runs this
        method, so normalizing here covers both doors and leaves the rule one
        home rather than a caller per field that has to remember it.

        Raises:
            TypeError: If ``scaffolding_prefixes`` is a single string —
                iterating one yields characters, which match wrongly rather
                than not at all (see :func:`is_under_directory`), so it is
                refused rather than comma-split like the user-facing lists —
                or if any other collection setting is something
                :func:`split_name_list` refuses.
        """
        object.__setattr__(
            self, "scaffolding_prefixes", _with_trailing_slashes(self.scaffolding_prefixes)
        )
        object.__setattr__(self, "include_globs", _lowercased(self.include_globs))
        object.__setattr__(self, "exclude_globs", _lowercased(self.exclude_globs))
        object.__setattr__(
            self,
            "allowed_extensions",
            _normalize_extensions(split_name_list(self.allowed_extensions)),
        )
        object.__setattr__(
            self,
            "archive_extensions",
            _normalize_extensions(split_name_list(self.archive_extensions)),
        )
        object.__setattr__(self, "runtime_suffixes", _lowercased(self.runtime_suffixes))
        object.__setattr__(
            self, "runtime_name_fragments", _lowercased(self.runtime_name_fragments)
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

        Raises:
            TypeError: If a list-valued setting is something
                :func:`split_name_list` refuses. Raised from ``__post_init__``
                rather than here, so the traceback names the field.
        """
        # Explicitly against None rather than falsiness: this is documented
        # as duck-typed, and a params object declaring ``Optional[int] = None``
        # would otherwise reach ``size > None`` at decision time — but a
        # deliberate 0 means "keep nothing", not "use the default".
        max_bytes = getattr(params, "artifact_max_bytes", None)
        if max_bytes is None:
            max_bytes = DEFAULT_MAX_BYTES
        # Split here only because the extra names have to be separate before
        # they can join the defaults — that is a merge, not a second copy of
        # the rule. Dots, case and the other lists are left to __post_init__.
        extra = frozenset(split_name_list(getattr(params, "artifact_extra_extensions", ())))
        return cls(
            max_bytes=max_bytes,
            allowed_extensions=DEFAULT_ALLOWED_EXTENSIONS | extra,
            include_globs=getattr(params, "artifact_include_globs", ()),
            exclude_globs=getattr(params, "artifact_exclude_globs", ()),
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
                Checked by :func:`checked_path_collection`, not read by
                :func:`split_name_list`: a path is compared verbatim, so
                neither splitting nor stripping it is safe, and the
                collection is handed back untouched.
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

        Raises:
            TypeError: If ``provider_deny`` is a single string, or anything
                :func:`split_name_list` refuses. The string case is the one
                that used to pass silently, denying nothing; the one-shot
                iterator is refused because this argument is reused across
                every member of a bundle, so a generator would fire for the
                first member and be empty for the rest.
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

        denied_paths = checked_path_collection(provider_deny, "provider_deny")
        if any(normalize_member_path(denied) == normalized for denied in denied_paths):
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
        directories: Directory names, any case, with or without a trailing
            slash. Read by :func:`split_name_list`, so surrounding whitespace
            is stripped, blanks and duplicates are dropped, and an unordered
            collection is sorted — none of which changes the answer here, but
            all of which decides what is accepted.

    Returns:
        Whether any of them names a directory on the path.

    Raises:
        TypeError: If ``directories`` is a single string. Iterating one yields
            characters, and a lone ``"a"`` or ``"."`` matches any path with a
            matching segment — so a bare string does not fail to match, it
            matches wrongly and only on some paths.

            Also for anything else :func:`split_name_list` refuses: a mapping,
            ``bytes``, or a one-shot iterator. The last is the one that costs
            something here, since this function consumes its argument inside
            the call and could safely take a generator — it is refused so that
            the same rule holds at every door rather than a caller having to
            know which doors re-read. Pass ``list(names)``.

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
    """Return whether a normalized path matches any fnmatch pattern.

    Both sides arrive lowercased — the path by :func:`normalize_member_path`
    and the patterns by ``__post_init__`` — so this does not lowercase again.
    It used to, once per pattern per member, which was harmless but left the
    stored ``include_globs`` showing a form other than the one matched.
    """
    return any(fnmatch(normalized_name, pattern) for pattern in patterns)


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
