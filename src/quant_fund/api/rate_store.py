"""Shared token buckets for more than one worker.

``DIPCATCHER_RATE_BACKEND`` selects ``memory`` (the default, owned by
``http_controls``), ``sqlite``, or ``file``. The last two persist the bucket
in ``DIPCATCHER_RATE_STORE`` on the shared volume. The wall clock is the
only clock those backends use, because each process has its own monotonic
clock. A store that cannot be read or written fails closed.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

_BACKEND_ENV = "DIPCATCHER_RATE_BACKEND"
_STORE_ENV = "DIPCATCHER_RATE_STORE"
_MAX_BUCKETS = 1024
_MAX_STORE_BYTES = 8_388_608
_STORE_ERRORS = (OSError, UnicodeError, ValueError, TypeError, sqlite3.Error, ImportError)


def rate_backend_env_names() -> tuple[str, ...]:
    """Environment names that select a shared rate-limit store."""
    return (_BACKEND_ENV, _STORE_ENV)


def backend_name() -> str:
    """``memory``, or the raw backend word when it is something else."""
    raw = os.environ.get(_BACKEND_ENV, "memory").strip().lower()
    if raw in {"", "memory"}:
        return "memory"
    return raw


def shared_take(
    kind: str,
    store: str,
    identity: str,
    rate: float,
    burst: float,
    now: float,
) -> bool | None:
    """Take one token from a shared store.

    ``None`` means the backend is unknown or the store could not be used.
    """
    if kind == "sqlite":
        return _guard(_sqlite_take, store, identity, rate, burst, now)
    if kind == "file":
        return _guard(_file_take, store, identity, rate, burst, now)
    return None


def apply_token(
    tokens: float,
    updated: float,
    now: float,
    rate: float,
    burst: float,
) -> tuple[bool, float]:
    """Refill from ``updated`` to ``now`` and take one token when one remains."""
    elapsed = max(0.0, now - updated)
    filled = min(burst, tokens + elapsed * rate)
    if filled < 1.0:
        return False, filled
    return True, filled - 1.0


_Take = Callable[[str, str, float, float, float], bool]


def _guard(
    fn: _Take,
    store: str,
    identity: str,
    rate: float,
    burst: float,
    now: float,
) -> bool | None:
    try:
        return fn(store, identity, rate, burst, now)
    except _STORE_ERRORS:
        return None


def _chmod(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        return


def _sqlite_take(store: str, identity: str, rate: float, burst: float, now: float) -> bool:
    path = Path(store)
    conn = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        _ensure_table(conn)
        conn.execute("BEGIN IMMEDIATE")
        try:
            allowed = _sqlite_update(conn, identity, rate, burst, now)
            conn.execute("COMMIT")
        except _STORE_ERRORS:
            _rollback(conn)
            raise
        return allowed
    finally:
        conn.close()
        _chmod(path)


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS buckets ("
        "identity TEXT PRIMARY KEY, tokens REAL NOT NULL, updated REAL NOT NULL)"
    )


def _rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("ROLLBACK")
    except sqlite3.Error:
        return


def _sqlite_update(
    conn: sqlite3.Connection,
    identity: str,
    rate: float,
    burst: float,
    now: float,
) -> bool:
    row = conn.execute(
        "SELECT tokens, updated FROM buckets WHERE identity = ?",
        (identity,),
    ).fetchone()
    if row is None:
        _drop_oldest_row(conn)
        allowed, tokens = apply_token(burst, now, now, rate, burst)
        conn.execute(
            "INSERT INTO buckets (identity, tokens, updated) VALUES (?, ?, ?)",
            (identity, tokens, now),
        )
        return allowed
    allowed, tokens = apply_token(float(row[0]), float(row[1]), now, rate, burst)
    conn.execute(
        "UPDATE buckets SET tokens = ?, updated = ? WHERE identity = ?",
        (tokens, now, identity),
    )
    return allowed


def _drop_oldest_row(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) FROM buckets").fetchone()
    if count is None or int(count[0]) < _MAX_BUCKETS:
        return
    oldest = conn.execute("SELECT identity FROM buckets ORDER BY updated ASC LIMIT 1").fetchone()
    if oldest is None:
        return
    conn.execute("DELETE FROM buckets WHERE identity = ?", (str(oldest[0]),))


def _file_take(store: str, identity: str, rate: float, burst: float, now: float) -> bool:
    import fcntl

    path = Path(store)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        payload = _read_capped(fd)
        data = _object(payload)
        buckets = data.get("buckets")
        if not isinstance(buckets, dict):
            buckets = {}
            data["buckets"] = buckets
        allowed = _file_update(buckets, identity, rate, burst, now)
        _rewrite(fd, data)
        return allowed
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        _chmod(path)


def _object(payload: str) -> dict[str, Any]:
    if not payload:
        return {"buckets": {}}
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("rate store is not an object")
    return parsed


def _read_capped(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    blob = os.read(fd, _MAX_STORE_BYTES + 1)
    if len(blob) > _MAX_STORE_BYTES:
        raise ValueError("rate store is too large")
    return blob.decode("utf-8")


def _rewrite(fd: int, data: dict[str, Any]) -> None:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > _MAX_STORE_BYTES:
        raise ValueError("rate store is too large")
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, payload)
    os.fsync(fd)


def _file_update(
    buckets: dict[str, Any],
    identity: str,
    rate: float,
    burst: float,
    now: float,
) -> bool:
    row = buckets.get(identity)
    if isinstance(row, dict):
        tokens, updated = _row_state(row, burst, now)
    else:
        _drop_file_oldest(buckets)
        tokens, updated = burst, now
    allowed, filled = apply_token(tokens, updated, now, rate, burst)
    buckets[identity] = {"tokens": filled, "updated": now}
    return allowed


def _row_state(row: dict[str, Any], burst: float, now: float) -> tuple[float, float]:
    return float(row.get("tokens", burst)), float(row.get("updated", now))


def _drop_file_oldest(buckets: dict[str, Any]) -> None:
    if len(buckets) < _MAX_BUCKETS:
        return
    oldest = min(buckets, key=lambda key: _stamp(buckets[key]))
    del buckets[oldest]


def _stamp(row: Any) -> float:
    if isinstance(row, dict):
        try:
            return float(row.get("updated", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0
