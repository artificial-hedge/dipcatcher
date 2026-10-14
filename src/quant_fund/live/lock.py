"""Single-writer lock for the live desk state directory.

Nested locks in one process deadlock, so only the public desk methods acquire
this. The lock file is local state, not a receipt.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def desk_lock(path: Path) -> Iterator[None]:
    """Exclusive lock. Portable across POSIX ``flock`` and Windows ``msvcrt``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        if handle.seek(0, 2) == 0:
            handle.write(" ")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if sys.platform == "win32":
            import msvcrt

            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
