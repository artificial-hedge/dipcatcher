"""On-disk cache of the CI-budget live-optional scorecard blobs.

Pytest sets ``DIP_SCORECARD_BUDGET=ci``. Those benches are seeded and do not
read the research panel, so a second process (another xdist worker, or the
research CLI echo) would otherwise rebuild them. The cache key is the budget
name plus a hash of ``quant_fund`` sources and the locked dependencies.

Production runs leave the env unset and never read or write this cache.
Monkeypatched bench callables must not consult it; the caller checks that.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_STAMP_SALT = b"ci-optional-blobs-v1\0"


def cache_disabled() -> bool:
    return os.environ.get("DIP_DISABLE_SESSION_CACHE") == "1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def source_stamp() -> str:
    """Separate caches when scorecard code or locked dependencies change."""
    root = _repo_root() / "src" / "quant_fund"
    digest = hashlib.sha256()
    digest.update(_STAMP_SALT)
    digest.update(sys.version.encode())
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for name in ("pyproject.toml", "uv.lock"):
        path = _repo_root() / name
        digest.update(name.encode())
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def blob_dir() -> Path:
    override = os.environ.get("DIP_SESSION_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".cache" / "dipcatcher"
    path = base / "ci-optional" / source_stamp()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_budget(budget: str) -> str | None:
    if budget and all(ch.isalnum() or ch in "_-" for ch in budget):
        return budget
    return None


def _blob_path(budget: str) -> Path | None:
    name = _safe_budget(budget)
    if name is None:
        return None
    return blob_dir() / f"{name}.pkl"


@contextmanager
def exclusive_lock() -> Iterator[None]:
    """Exclusive lock so one worker builds the CI blobs and the rest wait."""
    path = blob_dir() / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    locked = False
    try:
        if sys.platform == "win32":
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            if sys.platform == "win32":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def load_blobs(budget: str) -> dict[str, Any] | None:
    if cache_disabled():
        return None
    path = _blob_path(budget)
    if path is None or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def store_blobs(budget: str, blobs: dict[str, Any]) -> bool:
    """Atomically publish ``blobs``. Failures leave any previous file in place."""
    if cache_disabled():
        return False
    path = _blob_path(budget)
    if path is None:
        return False
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            pickle.dump(blobs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except (OSError, pickle.PicklingError, TypeError, AttributeError):
        tmp.unlink(missing_ok=True)
        return False
    return True
