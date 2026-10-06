"""Content-addressed index over the sealed-receipts store.

Shared by the HTTP API (``GET /receipts*``) and the in-process SDK — one
staleness/availability contract on both surfaces. This is a metadata index,
not cryptographic receipt verification.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import stat
import threading
from pathlib import Path

SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_FileStamp = tuple[int, int, int, int, int]
_Snapshot = tuple[tuple[Path, _FileStamp], ...]


def _receipt_sha(path: Path) -> str | None:
    """Read one regular file, ignoring invalid input but not programming errors."""
    try:
        doc = json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError):
        return None
    sha = doc.get("receipt_sha256") if isinstance(doc, dict) else None
    return sha if isinstance(sha, str) and SHA256_HEX.fullmatch(sha) else None


class ReceiptIndex:
    """Lazy sha256 → file index over a receipts directory.

    Scan file identities and nanosecond metadata to detect additions,
    removals, renames and ordinary replacements, even when count and maximum
    mtime stay unchanged. Decode only new or changed files. A missing or
    unreadable directory clears the cache instead of serving stale lookups.
    Symlinks and non-regular files are not receipt blobs. Duplicate digest
    declarations resolve to the lexicographically first filename.

    Metadata caching relies on the immutable-receipt contract; it is not
    an integrity check against a writer capable of preserving all metadata.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._lock = threading.Lock()
        self._key: _Snapshot | None = None
        self._cache: dict[Path, tuple[_FileStamp, str | None]] = {}
        self._by_sha: dict[str, Path] = {}
        self._items: tuple[tuple[str, Path], ...] = ()
        self._available = False

    @property
    def root(self) -> Path:
        return self._root

    def _snapshot(self) -> _Snapshot:
        files: list[tuple[Path, _FileStamp]] = []
        # scandir propagates directory access errors; glob can suppress them.
        with os.scandir(self._root) as entries:
            for entry in entries:
                if not fnmatch.fnmatch(entry.name, "*.json"):
                    continue
                metadata = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                stamp = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                )
                files.append((self._root / entry.name, stamp))
        return tuple(sorted(files, key=lambda item: item[0].name))

    def _scan(self) -> None:
        try:
            key = self._snapshot()
        except OSError:
            self._available = False
            self._by_sha = {}
            self._items = ()
            self._cache = {}
            self._key = None
            return
        if key == self._key:
            return
        cache: dict[Path, tuple[_FileStamp, str | None]] = {}
        by_sha: dict[str, Path] = {}
        for path, stamp in key:
            previous = self._cache.get(path)
            sha = (
                previous[1]
                if previous is not None and previous[0] == stamp
                else _receipt_sha(path)
            )
            cache[path] = (stamp, sha)
            if sha is not None:
                by_sha.setdefault(sha, path)
        # Publish together under the caller's lock after the entire scan.
        self._cache = cache
        self._by_sha = by_sha
        self._items = tuple(by_sha.items())
        self._key = key
        self._available = True

    def available(self) -> bool:
        with self._lock:
            self._scan()
            return self._available

    def lookup(self, sha256: str) -> Path | None:
        with self._lock:
            self._scan()
            return self._by_sha.get(sha256)

    def items(self) -> list[tuple[str, Path]]:
        with self._lock:
            self._scan()
            return list(self._items)
