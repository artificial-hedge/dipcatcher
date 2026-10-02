"""Content-addressed index over the sealed-receipts store.

Shared by the HTTP API (``GET /receipts*``) and the in-process SDK — one
staleness/availability contract on both surfaces.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class ReceiptIndex:
    """Lazy sha256 → file index over a receipts directory.

    Receipts are append-only immutable blobs, so ``(file count, max mtime)``
    is a sufficient staleness key — the index rebuilds only when the store
    changes, and a missing/removed store reports itself unavailable rather
    than serving stale lookups.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._lock = threading.Lock()
        self._key: tuple[int, float] | None = None
        self._by_sha: dict[str, Path] = {}
        self._available = False

    @property
    def root(self) -> Path:
        return self._root

    def _scan(self) -> None:
        if not self._root.is_dir():
            self._available = False
            self._by_sha = {}
            self._key = None
            return
        try:
            files = list(self._root.glob("*.json"))
            key = (len(files), max((f.stat().st_mtime for f in files), default=0.0))
        except OSError:
            self._available = False
            self._by_sha = {}
            self._key = None
            return
        if key == self._key:
            return
        by_sha: dict[str, Path] = {}
        for f in files:
            try:
                doc = json.loads(f.read_bytes())
            except (OSError, ValueError):
                continue
            sha = doc.get("receipt_sha256") if isinstance(doc, dict) else None
            if isinstance(sha, str) and SHA256_HEX.fullmatch(sha):
                by_sha[sha] = f
        self._by_sha = by_sha
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
            return sorted(self._by_sha.items(), key=lambda kv: kv[1].name)
