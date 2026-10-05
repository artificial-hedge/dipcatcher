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
- ``append`` fsyncs before returning — a record the API has already
  answered with is durable before the worker transitions it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
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


@dataclass
class ReplayResult:
    """What ``replay`` found: verified payloads in order, plus where the
    chain broke (``truncated_at`` is the byte offset of the first bad
    line; ``dropped`` counts lines after it).

    ``torn_tail`` marks the benign break shape: the first bad line is an
    unparseable *final* line — the signature of a crash mid-append — so
    the verified prefix is intact and only durability was lost. Any other
    break (a well-formed line that fails verification, or unparseable
    damage with valid lines behind it) is tamper-shaped: the dropped
    suffix may have carried state a security-relevant store cannot
    ignore, and such callers should fail closed rather than trust the
    replayed prefix."""

    payloads: list[dict[str, Any]]
    truncated_at: int | None = None
    dropped: int = 0
    warnings: list[str] = field(default_factory=list)
    torn_tail: bool = False


class JobJournal:
    """Append-only journal under ``state_dir`` — one file per store name.

    Thread-safe: all mutations hold the internal lock. The file is opened
    lazily on first append so a read-only boot never creates files.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seq = 0
        self._chain = "0" * 64

    # ---- write path ----------------------------------------------------

    def append(self, payload: dict[str, Any]) -> None:
        """Append one record; fsync before returning so a confirmed
        transition is durable before the caller moves on."""
        with self._lock:
            # ``seq``/``chain`` must be read under the lock: two appends
            # racing on stale values mint twin seqs and break the chain
            # at replay — a self-inflicted tamper drop.
            line = _line_bytes(self._seq, self._chain, payload)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self._chain = hashlib.sha256(line).hexdigest()
            self._seq += 1

    # ---- read path -----------------------------------------------------

    def replay(self) -> ReplayResult:
        """Verify the chain line-by-line; stop at the first bad line.

        A clean replay returns every payload in order. A torn tail or
        mid-file edit stops at that line — ``truncated_at`` records the
        byte offset so the caller can report the dropped span."""
        out = ReplayResult(payloads=[])
        if not self.path.exists():
            return out
        data = self.path.read_bytes()
        seq = 0
        chain = "0" * 64
        offset = 0
        for raw in data.splitlines(keepends=True):
            offset += len(raw)
            try:
                line = json.loads(raw)
                payload_raw = json.dumps(line["payload"], sort_keys=True, separators=(",", ":"))
                expect = hashlib.sha256(
                    f"{line['seq']}|{line['chain']}|{payload_raw}".encode()
                ).hexdigest()
                ok = line["seq"] == seq and line["chain"] == chain and line["sha256"] == expect
            except Exception:  # noqa: BLE001 — corrupt line, verified shape only
                ok = False
                line = None
            if not ok:
                out.truncated_at = offset - len(raw)
                out.dropped = len(data[out.truncated_at :].splitlines())
                # torn tail = unparseable final line (crash mid-append);
                # everything else is tamper-shaped and fail-close worthy.
                out.torn_tail = line is None and offset >= len(data)
                out.warnings.append(
                    f"journal {self.path.name}: chain broke at byte "
                    f"{out.truncated_at} ({out.dropped} line(s) dropped)"
                )
                break
            out.payloads.append(line["payload"])
            chain = hashlib.sha256(raw).hexdigest()
            seq += 1
        self._seq = seq
        self._chain = chain
        return out

    # ---- maintenance ---------------------------------------------------

    def compact(self, live: list[dict[str, Any]]) -> None:
        """Rewrite the journal holding only the live records.

        Called on boot after replay (drops dead history) and safe to call
        any time — the write is atomic (tmp + rename + fsync) so a crash
        mid-compact leaves the old journal intact."""
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


class _ClaimLocks:
    """Bounded map of per-key mutexes — the single-execution primitive
    for ``Idempotency-Key`` check+insert.

    A route holds its key's lock across the store lookup, the work, and
    the ``put``: same-key racers serialize behind the leader, and the
    second comer replays the stored answer instead of double-executing
    (the check-then-put window is the defect this closes). Entries are
    bounded like the store itself — the oldest *unheld* lock evicts
    first, so a lock is never stolen mid-flight; if every lock is held
    the map overshoots the bound rather than break dedup."""

    def __init__(self, bound: int) -> None:
        self._guard = threading.Lock()
        self._locks: OrderedDict[str, threading.Lock] = OrderedDict()
        self._bound = bound

    @contextmanager
    def hold(self, key: str | None) -> Iterator[None]:
        """Hold ``key``'s mutex for the lookup+execute+put span.

        ``None`` (no idempotency key) is a no-op hold — unsynchronized
        calls keep their plain path."""
        if key is None:
            yield
            return
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            self._locks.move_to_end(key)
            while len(self._locks) > self._bound:
                stale = next((k for k, held in self._locks.items() if not held.locked()), None)
                if stale is None:
                    break
                del self._locks[stale]
        with lock:
            yield
