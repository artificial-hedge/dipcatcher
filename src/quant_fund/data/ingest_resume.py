"""Crash-resumable ingestion journal with an append-only cursor.

Proves the ingestion recovery contract: a run killed mid-ingest can resume from
its journal without losing or double-counting rows. Each committed batch is an
append-only JSONL record carrying a content-addressed ``sha256`` over its body;
re-ingesting a known ``batch_id`` is idempotent (counted as a duplicate and
skipped). Replay is fail-closed: a corrupted or hash-mismatched committed record
raises :class:`IngestJournalError`. Only a syntactically incomplete, unterminated
final record is treated as uncommitted and discarded during crash recovery.

SYNTHETIC rows in tests are correctness fixtures only, never market evidence.
``live_pnl_claim=False`` / ``research_only=True`` throughout.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_fund.schemas.errors import DataContractError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


class IngestJournalError(DataContractError):
    """Corrupted or inconsistent ingestion journal (fail closed)."""


@dataclass(frozen=True)
class Batch:
    """A committed ingest batch: an idempotency key plus its row count."""

    batch_id: str
    rows: int


def _validate_batch(batch: Batch) -> None:
    if not isinstance(batch.batch_id, str) or not batch.batch_id.strip():
        raise IngestJournalError("batch_id must be a non-empty string")
    if type(batch.rows) is not int or batch.rows < 0:
        raise IngestJournalError("rows must be a non-negative integer")


def _has_incomplete_json(raw: bytes) -> bool:
    try:
        json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True
    except (ValueError, RecursionError):
        return False
    return False


class IngestCheckpoint:
    """Append-only JSONL ingest journal with an idempotent, resumable cursor.

    A syntactically incomplete final record without a newline is uncommitted
    and may be discarded. A complete verified record without its newline is
    committed and has the delimiter repaired. Every other corruption is fatal.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._committed: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        data = self.path.read_bytes()
        if not data:
            return
        parts = data.split(b"\n")
        terminated = not parts[-1]
        complete = parts[:-1] if terminated else parts
        tail = None if terminated else complete.pop()
        records = [self._parse_line(raw) for raw in complete]
        repair = None
        if tail:
            try:
                records.append(self._parse_line(tail))
            except IngestJournalError:
                if not _has_incomplete_json(tail):
                    raise
                repair = "truncate"
            else:
                repair = "newline"

        committed: dict[str, int] = {}
        for rec in records:
            batch_id = rec["batch_id"]
            if batch_id in committed:
                raise IngestJournalError(f"duplicate batch_id in ingest journal: {batch_id!r}")
            committed[batch_id] = rec["rows"]
        if repair == "truncate":
            self._repair_tail(data, keep=len(data) - len(tail or b""))
        elif repair == "newline":
            self._repair_tail(data, keep=len(data))
        self._committed = committed

    def _parse_line(self, raw: bytes) -> dict[str, Any]:
        try:
            rec = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise IngestJournalError(f"corrupt ingest journal line: {exc}") from exc
        if not isinstance(rec, dict):
            raise IngestJournalError("ingest journal record is not an object")
        if set(rec) != {"batch_id", "rows", "sha256"}:
            raise IngestJournalError("ingest journal record has invalid fields")
        batch = Batch(batch_id=rec["batch_id"], rows=rec["rows"])
        _validate_batch(batch)
        body: dict[str, Any] = {"batch_id": batch.batch_id, "rows": batch.rows}
        if hash_bytes(canonical_json_bytes(body)) != rec["sha256"]:
            raise IngestJournalError(f"hash mismatch on ingest record {batch.batch_id!r}")
        return body

    def _repair_tail(self, original: bytes, *, keep: int) -> None:
        with self.path.open("r+b") as handle:
            if handle.read() != original:
                raise IngestJournalError("ingest journal changed during crash recovery")
            handle.seek(keep)
            if keep == len(original):
                handle.write(b"\n")
            else:
                handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())

    def _append(self, batch: Batch) -> dict[str, Any]:
        rec: dict[str, Any] = {"batch_id": batch.batch_id, "rows": batch.rows}
        rec["sha256"] = hash_bytes(canonical_json_bytes(rec))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return rec

    def ingest(self, batch: Batch) -> str:
        """Commit a batch idempotently; replaying a batch is a no-op duplicate."""
        _validate_batch(batch)
        prior_rows = self._committed.get(batch.batch_id)
        if prior_rows is not None:
            if prior_rows != batch.rows:
                raise IngestJournalError(
                    f"batch_id {batch.batch_id!r} already committed with {prior_rows} rows, "
                    f"cannot replay with {batch.rows} rows"
                )
            return "duplicate"
        self._append(batch)
        self._committed[batch.batch_id] = batch.rows
        return "committed"

    @property
    def cursor(self) -> int:
        return len(self._committed)

    @property
    def total_rows(self) -> int:
        return sum(self._committed.values())

    @property
    def n_batches(self) -> int:
        return len(self._committed)

    def resume(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "total_rows": self.total_rows,
            "n_batches": self.n_batches,
            "research_only": True,
            "live_pnl_claim": False,
        }


__all__ = [
    "Batch",
    "IngestCheckpoint",
    "IngestJournalError",
]
