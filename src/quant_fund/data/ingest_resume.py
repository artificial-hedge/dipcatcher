"""Crash-resumable ingestion journal with an append-only cursor.

Proves the ingestion recovery contract: a run killed mid-ingest can resume from
its journal without losing or double-counting rows. Each committed batch is an
append-only JSONL record carrying a content-addressed ``sha256`` over its body;
re-ingesting a known ``batch_id`` is idempotent (counted as a duplicate and
skipped). Replay is fail-closed: a corrupted or hash-mismatched record raises
:class:`IngestJournalError` rather than silently continuing.

SYNTHETIC rows in tests are correctness fixtures only, never market evidence.
``live_pnl_claim=False`` / ``research_only=True`` throughout.
"""

from __future__ import annotations

import json
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


class IngestCheckpoint:
    """Append-only JSONL ingest journal with an idempotent, resumable cursor."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._committed: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            rec = self._parse_line(raw)
            self._committed[str(rec["batch_id"])] = int(rec["rows"])

    def _parse_line(self, raw: str) -> dict[str, Any]:
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IngestJournalError(f"corrupt ingest journal line: {exc}") from exc
        if not isinstance(rec, dict):
            raise IngestJournalError("ingest journal record is not an object")
        body = {k: v for k, v in rec.items() if k != "sha256"}
        if hash_bytes(canonical_json_bytes(body)) != rec.get("sha256"):
            raise IngestJournalError(f"hash mismatch on ingest record {rec.get('batch_id')!r}")
        return body

    def _append(self, batch: Batch) -> dict[str, Any]:
        rec: dict[str, Any] = {"batch_id": batch.batch_id, "rows": batch.rows}
        rec["sha256"] = hash_bytes(canonical_json_bytes(rec))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def ingest(self, batch: Batch) -> str:
        """Commit a batch idempotently; replaying a batch is a no-op duplicate."""
        if batch.batch_id in self._committed:
            return "duplicate"
        self._append(batch)
        self._committed[batch.batch_id] = int(batch.rows)
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
