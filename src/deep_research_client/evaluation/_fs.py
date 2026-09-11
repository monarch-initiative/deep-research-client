"""Filesystem helpers shared by the evaluation modules."""

import os
import stat
import tempfile
from pathlib import Path


def _default_mode(directory: Path) -> int:
    """The mode an ordinary file creation would produce in ``directory``.

    Determined by creating one and looking at it, rather than by reading the
    umask - the only way to read a umask is to set it, which opens a window in
    which any thread creating a file gets it world-writable. This module is
    called several times per cell from a runner built for concurrency, so that
    window is not hypothetical, and probing avoids it entirely rather than
    merely making it rarer. Probing also picks up what umask arithmetic cannot:
    a default ACL, or a setgid directory.

    ``O_EXCL`` on a name nothing else holds, so the probe can only ever report
    the mode of the file it just created - never a mode inherited from whatever
    else happened to appear at that path, which matters most in the one place
    this module writes somewhere the user may not own: a shared cache.
    """
    fd, reserved = tempfile.mkstemp(dir=str(directory), prefix=".mode-probe.")
    os.close(fd)
    probe = Path(f"{reserved}.check")
    try:
        os.close(os.open(str(probe), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
        return stat.S_IMODE(probe.stat().st_mode)
    finally:
        probe.unlink(missing_ok=True)
        Path(reserved).unlink(missing_ok=True)


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
    destination's own mode when it already exists, and otherwise whatever a
    plain file creation in that directory produces. Neither is what ``mkstemp``
    produces - it creates 0600, and ``os.replace`` keeps the temporary file's
    mode - which would quietly make
    every file here owner-only, including a benchmark cache whose whole purpose
    is to be shared between runs and, on a cluster, between users. Preserving an
    existing destination's mode matters for the same reason: a cache someone
    deliberately opened up to a group should not close again on the next refresh.

    The default is probed rather than derived from the umask, so this never
    changes process-global state - see :func:`_default_mode`.
    """
    try:
        mode: int | None = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        # Asking whether the file exists and then stat-ing it would let a file
        # removed between the two raise out of here; this cannot.
        mode = None
    if mode is None:
        # Probed outside the except, so a failure here - a read-only directory,
        # a full filesystem - is reported as itself rather than chained onto the
        # FileNotFoundError that merely told us the destination is new.
        mode = _default_mode(path.parent)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
