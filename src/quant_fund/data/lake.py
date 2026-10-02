"""Parquet lake I/O."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import polars as pl

from quant_fund.config.models import AppConfig


def _fsync_dir(path: Path) -> None:
    # Windows cannot open a directory handle, so a directory fsync is not
    # expressible there; os.replace on the same volume is atomic per
    # MoveFileEx semantics, which preserves the crash-safety contract.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Lake:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for part in ("raw", "bronze", "silver", "gold", "metadata"):
            (self.root / part).mkdir(parents=True, exist_ok=True)
        # Concurrent writers to the same canonical path race the final
        # os.replace on Windows (ReplaceFileW denies the destination while
        # another rename is in flight). Serialize the rename step per path so
        # every writer lands; the unique tmp file is still per-writer.
        self._rename_locks: dict[str, threading.Lock] = {}
        self._rename_locks_guard = threading.Lock()

    def _rename_lock(self, rel: str) -> threading.Lock:
        with self._rename_locks_guard:
            lock = self._rename_locks.get(rel)
            if lock is None:
                lock = self._rename_locks[rel] = threading.Lock()
            return lock

    def _resolve(self, rel: str) -> Path:
        # Lake-relative paths only: no absolute paths, no traversal — a bad
        # caller must not be able to write outside the lake root. On Windows,
        # is_absolute() is False for drive-relative rooted paths such as
        # "/abs/x" or "C:x", so the drive/root checks are required as well.
        candidate = Path(rel)
        parts = candidate.parts
        if (
            not rel
            or candidate.is_absolute()
            or bool(candidate.drive)
            or bool(candidate.root)
            or "\\" in rel
            or any(part in (".", "..") for part in parts)
        ):
            raise ValueError(f"lake path must be a clean relative path, got {rel!r}")
        return self.root / candidate

    def write_parquet(self, frame: pl.DataFrame, rel: str) -> Path:
        # Unique tmp + fsync + os.replace + dir fsync: a crash mid-write can
        # never leave a truncated or unflushed parquet at the canonical path,
        # and concurrent writers to the same path cannot share a tmp file.
        path = self._resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            frame.write_parquet(tmp)
            # Windows rejects fsync on a read-only handle (EBADF); open
            # read-write there so the flush still reaches the filesystem.
            fd = os.open(tmp, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            with self._rename_lock(rel):
                os.replace(tmp, path)
            _fsync_dir(path.parent)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def read_parquet(self, rel: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Memory-map a lake parquet file.

        ``columns`` projects before the frame is materialized. Omit it to read
        every stored column, which is what existing callers do.
        """
        return pl.read_parquet(self._resolve(rel), columns=columns, memory_map=True)

    def exists(self, rel: str) -> bool:
        return self._resolve(rel).exists()


def lake_from_config(config: AppConfig) -> Lake:
    return Lake(Path(config.data.root))
