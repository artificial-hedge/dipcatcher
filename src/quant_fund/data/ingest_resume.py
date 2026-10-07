"""Crash-resumable ingestion cursor — idempotent batches, no duplicate rows.

Ops-floor recovery surface for the ingestion pipeline. Each ingested batch is
appended to a durable journal with a monotonic ``batch_id`` and a running row
count; a persisted cursor records the last committed batch. After a crash the
pipeline resumes from the cursor and re-ingesting an already-committed
``batch_id`` is a no-op (idempotent), so a restart never double-counts rows.

The journal is a local, append-only JSON with a self-hash per line; it detects
accidental truncation/corruption on reload. It is a correctness/recovery
diagnostic (``research_only=True``, ``live_pnl_claim=False``), never market
evidence.
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
    """Malformed or corrupt ingestion journal."""


@dataclass(frozen=True)
class Batch:
    batch_id: int
    rows: int


class IngestCheckpoint:
    """Durable, idempotent ingestion cursor with crash-resume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._committed: dict[int, int] = {}
        self._load()

    # -- durability --

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rec = self._parse_line(line, lineno)
            self._committed[int(rec["batch_id"])] = int(rec["rows"])

    @staticmethod
    def _parse_line(line: str, lineno: int) -> dict[str, Any]:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IngestJournalError(f"journal line {lineno} is not JSON: {exc}") from exc
        if not isinstance(rec, dict):
            raise IngestJournalError(f"journal line {lineno} is not an object")
        body = {k: v for k, v in rec.items() if k != "sha256"}
        if hash_bytes(canonical_json_bytes(body)) != rec.get("sha256"):
            raise IngestJournalError(f"journal line {lineno} self-hash mismatch")
        return rec

    def _append(self, batch_id: int, rows: int) -> None:
        rec: dict[str, Any] = {"batch_id": batch_id, "rows": rows}
        rec["sha256"] = hash_bytes(canonical_json_bytes(rec))
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    # -- public API --

    def ingest(self, batch: Batch) -> str:
        """Commit a batch idempotently. Returns ``committed`` or ``duplicate``."""
        if batch.batch_id in self._committed:
            return "duplicate"
        self._committed[batch.batch_id] = batch.rows
        self._append(batch.batch_id, batch.rows)
        return "committed"

    @property
    def cursor(self) -> int:
        return max(self._committed) if self._committed else 0

    @property
    def total_rows(self) -> int:
        return sum(self._committed.values())

    @property
    def n_batches(self) -> int:
        return len(self._committed)

    def resume(self) -> dict[str, Any]:
        """Report the resumable cursor after a crash (idempotent reload)."""
        return {
            "cursor": self.cursor,
            "total_rows": self.total_rows,
            "n_batches": self.n_batches,
            "research_only": True,
            "live_pnl_claim": False,
        }


__all__ = ["Batch", "IngestCheckpoint", "IngestJournalError"]
