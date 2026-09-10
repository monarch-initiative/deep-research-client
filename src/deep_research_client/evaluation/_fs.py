"""Filesystem helpers shared by the evaluation modules."""

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, as UTF-8.

    Evaluation writes files that are read back by a later run - cell records,
    caches, summary files - and every one of them is written while the run can
    still be interrupted. A truncating in-place write that is interrupted leaves
    a short file, and a short file is worse than a missing one: a half-written
    ``cell.json`` does not resume as an absent cell, it raises a JSON error and
    takes the resumed run down with it. Writing to a sibling and renaming means
    a reader sees either the old file or the new one, never half of either.

    UTF-8 explicitly rather than by locale, because research reports carry
    non-ASCII text and the encoding a file is written with should not depend on
    the machine that wrote it.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        # mkstemp creates 0600 and os.replace keeps the temp file's mode, so
        # without this every file written here would be owner-only - including
        # the benchmark cache, whose whole point is to be shared between runs
        # and, on a cluster, between users. Restore what an ordinary write would
        # have produced under the caller's umask.
        os.chmod(tmp, 0o666 & ~_umask())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _umask() -> int:
    """Read the process umask without leaving it changed.

    There is no way to read it directly, so it has to be set to learn its value
    and then restored.
    """
    current = os.umask(0)
    os.umask(current)
    return current
