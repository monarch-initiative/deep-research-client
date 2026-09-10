"""Filesystem helpers shared by the evaluation modules."""

import os
import stat
import tempfile
from pathlib import Path

#: Cached process umask.
#:
#: There is no way to read the umask without setting it, so learning it opens a
#: window in which the process umask is 0. Doing that once at import is very
#: different from doing it on every write: this module is called several times
#: per cell from a runner built for concurrency, and any *thread* creating a
#: file inside one of those windows - a provider SDK falling back to a thread, an
#: HTTP library's disk cache - would get it created world-writable.
_UMASK = os.umask(0)
os.umask(_UMASK)


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

    The result carries the mode an ordinary write would have left: the
    destination's own mode when it already exists, and the umask default
    otherwise. Neither is what ``mkstemp`` produces - it creates 0600, and
    ``os.replace`` keeps the temporary file's mode - which would quietly make
    every file here owner-only, including a benchmark cache whose whole purpose
    is to be shared between runs and, on a cluster, between users. Preserving an
    existing destination's mode matters for the same reason: a cache someone
    deliberately opened up to a group should not close again on the next refresh.

    Note this reads a cached umask, so a program that calls ``os.umask`` after
    importing this module will not see the change reflected here.
    """
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = 0o666 & ~_UMASK

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
