"""Durable state for the async-job surface: an append-only hash-chained
JSONL journal the job store writes through.

Why this exists: the wire's job store is an in-process LRU — a process
restart silently drops every queued/running job, its idempotency key, and
any pending webhook. With ``--state-dir`` the store journals every
transition; on boot the journal replays into the same store so clients
can still poll their job ids.

Honesty rules (the contract this module sells):

- Recovered jobs that never reached a terminal state are restored as
  ``failed`` with ``error="process restarted before completion"`` — we
  do not pretend re-execution happens. Their ``Idempotency-Key`` mapping
  is restored so a client retrying the same submission gets the lost
  record back (replayed=true), not a duplicate run.
- ``callback_secret`` is never journaled — secrets don't touch disk.
  Recovered records keep ``callback_url`` for audit but cannot deliver
  a signed webhook post-restart (no secret); ``callback_attempts``
  stays 0.
- Every journal line is ``{"seq", "chain", "sha256", "payload"}`` where
  ``chain`` is the previous line's sha256. Replay verifies each line's
  hash + link and stops at the first bad one: a torn tail from a
  crash-mid-append truncates honestly instead of corrupting state, and
  a mid-journal edit invalidates the rest rather than smuggling.
- Replay never rewrites damaged bytes. A malformed record or missing
  newline blocks further appends until successful replay or explicit
  compaction; new transitions cannot be hidden behind a broken chain.
- ``append`` fsyncs before returning — a record the API has already
  answered with is durable before the worker transitions it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _line_bytes(seq: int, chain: str, payload: dict[str, Any]) -> bytes:
    payload_raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    line = (
        json.dumps(
            {
                "seq": seq,
                "chain": chain,
                "sha256": hashlib.sha256(f"{seq}|{chain}|{payload_raw}".encode()).hexdigest(),
                "payload": json.loads(payload_raw),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    return line


def _verified_payload(raw: bytes, seq: int, chain: str) -> dict[str, Any] | None:
    """Validate one complete journal record without hiding programming errors."""
    if not raw.endswith(b"\n"):
        return None
    try:
        line = json.loads(raw)
    except (ValueError, RecursionError):
        # Includes malformed JSON/UTF-8 and the decoder's depth/integer limits.
        return None
    if not isinstance(line, dict):
        return None
    if type(line.get("seq")) is not int or line["seq"] != seq or line.get("chain") != chain:
        return None
    payload = line.get("payload")
    if not isinstance(payload, dict):
        return None
    payload_raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(f"{seq}|{chain}|{payload_raw}".encode()).hexdigest()
    if line.get("sha256") != expected:
        return None
    return payload


@dataclass
class ReplayResult:
    """What ``replay`` found: verified payloads in order, plus where the
    chain broke (``truncated_at`` is the byte offset of the first bad
    line; ``dropped`` includes the bad line and every subsequent line)."""

    payloads: list[dict[str, Any]]
    truncated_at: int | None = None
    dropped: int = 0
    warnings: list[str] = field(default_factory=list)


class JobJournal:
    """Append-only journal under ``state_dir`` — one file per store name.

    Thread-safe: all mutations hold the internal lock. The file is opened
    lazily on first append so a read-only boot never creates files.
    Replay an existing journal before appending. Locks protect this
    instance only, not independent instances or other processes.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seq = 0
        self._chain = "0" * 64
        self._append_blocked = False

    # ---- write path ----------------------------------------------------

    def append(self, payload: dict[str, Any]) -> None:
        """Append one record; fsync before returning so a confirmed
        transition is durable before the caller moves on."""
        with self._lock:
            if self._append_blocked:
                raise RuntimeError(f"journal {self.path.name}: replay or compact before appending")
            line = _line_bytes(self._seq, self._chain, payload)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # An interrupted write/fsync may already have changed the file.
            # Do not reuse its sequence until recovery establishes the tail.
            self._append_blocked = True
            with self.path.open("ab") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self._chain = hashlib.sha256(line).hexdigest()
            self._seq += 1
            self._append_blocked = False

    # ---- read path -----------------------------------------------------

    def replay(self) -> ReplayResult:
        """Verify records in a single streaming pass, preserving damaged bytes.

        The returned payloads remain materialized for existing store callers;
        raw file contents and split-line copies are no longer held in memory.
        A broken chain leaves appends blocked until recovery or compaction.
        """
        with self._lock:
            self._append_blocked = True
            out = ReplayResult(payloads=[])
            seq = 0
            chain = "0" * 64
            offset = 0
            try:
                fh = self.path.open("rb")
            except FileNotFoundError:
                self._seq = seq
                self._chain = chain
                self._append_blocked = False
                return out
            with fh:
                for raw in fh:
                    payload = _verified_payload(raw, seq, chain)
                    if payload is None:
                        out.truncated_at = offset
                        out.dropped = 1 + sum(1 for _ in fh)
                        out.warnings.append(
                            f"journal {self.path.name}: chain broke at byte "
                            f"{out.truncated_at} ({out.dropped} line(s) dropped)"
                        )
                        break
                    out.payloads.append(payload)
                    chain = hashlib.sha256(raw).hexdigest()
                    seq += 1
                    offset += len(raw)
            self._seq = seq
            self._chain = chain
            self._append_blocked = out.truncated_at is not None
            return out

    # ---- maintenance ---------------------------------------------------

    def compact(self, live: list[dict[str, Any]]) -> None:
        """Rewrite the journal holding only the live records.

        Called on boot after replay (drops dead history) and safe to call
        any time. The replacement file is fsynced before atomic rename.
        Parent-directory fsync and multi-process coordination are not
        provided; this is not a power-loss durability guarantee."""
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            tmp.parent.mkdir(parents=True, exist_ok=True)
            seq = 0
            chain = "0" * 64
            with tmp.open("wb") as fh:
                for payload in live:
                    line = _line_bytes(seq, chain, payload)
                    fh.write(line)
                    chain = hashlib.sha256(line).hexdigest()
                    seq += 1
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            self._seq = seq
            self._chain = chain
            self._appends_since_compact = 0
            self._append_blocked = False


@dataclass
class _Claim:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


class _ClaimLocks:
    """Bounded registry of per-key mutexes for idempotent execution.

    A reservation counts both holders and waiters before they leave the
    registry lock. Only entries without reservations may be evicted: an
    unlocked mutex may still have an acquiring thread or queued waiters.
    Active keys may temporarily exceed the cache bound; the excess is
    reclaimed as claims finish.
    """

    def __init__(self, bound: int) -> None:
        self._guard = threading.Lock()
        self._locks: OrderedDict[str, _Claim] = OrderedDict()
        self._bound = max(1, bound)

    def _prune_idle(self) -> None:
        """Evict idle entries while the registry lock is held."""
        while len(self._locks) > self._bound:
            stale = next((k for k, claim in self._locks.items() if claim.users == 0), None)
            if stale is None:
                break
            del self._locks[stale]

    def _reserve(self, key: str) -> _Claim:
        with self._guard:
            claim = self._locks.get(key)
            if claim is None:
                claim = _Claim()
                self._locks[key] = claim
            claim.users += 1
            self._locks.move_to_end(key)
            self._prune_idle()
            return claim

    def _release(self, claim: _Claim) -> None:
        with self._guard:
            claim.users -= 1
            self._prune_idle()

    @contextmanager
    def hold(self, key: str | None) -> Iterator[None]:
        """Serialize a key's lookup, execution, and insertion; None bypasses."""
        if key is None:
            yield
            return
        claim = self._reserve(key)
        try:
            with claim.lock:
                yield
        finally:
            self._release(claim)

    @asynccontextmanager
    async def ahold(self, key: str | None) -> AsyncIterator[None]:
        """Wait cooperatively without occupying a request worker thread.

        A sync yield dependency can strand its holder: waiting retries
        consume every threadpool token before the holder's handler can
        run. Nonblocking acquisition preserves the shared sync mutex
        while yielding the event loop between attempts. Cancellation
        releases the reservation even when acquisition never succeeds.
        """
        if key is None:
            yield
            return
        from anyio import sleep

        claim = self._reserve(key)
        try:
            while not claim.lock.acquire(blocking=False):
                await sleep(0.01)
            try:
                yield
            finally:
                claim.lock.release()
        finally:
            self._release(claim)
