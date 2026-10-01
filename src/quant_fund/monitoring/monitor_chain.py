"""monitor_chain — tamper-evident ledger for operational events.

Dashboard verdicts, drift breaches, and kill-switch transitions are the
ops layer's evidence — but today nothing records them. ``MonitorChain``
is an append-only JSONL log where each record carries ``prev_sha256``
and its own ``sha256`` over the canonical payload, so deletion,
reordering, or retro-editing of any event surfaces on ``verify()``.

Records are content-addressed like the receipt corpus: ``sha256`` is
computed over ``canonical_json_bytes`` of the record *before* the field
is added, and ``prev_sha256`` binds the chain. ``verify`` replays the
whole file and reports the first divergence or ``errors: []``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

_GENESIS_PREV = "0" * 64
KINDS = ("dashboard", "drift", "kill_switch", "gate", "other")


def _record(
    seq: int, ts: float, actor: str, kind: str, detail: dict[str, Any], prev: str
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    rec: dict[str, Any] = {
        "seq": seq,
        "ts": ts,
        "actor": actor,
        "kind": kind,
        "detail": detail,
        "prev_sha256": prev,
    }
    rec["sha256"] = hash_bytes(canonical_json_bytes(rec))
    return rec


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"monitor_chain line {line_no} is not JSON: {exc}") from exc
        if not isinstance(rec, dict):
            raise ValueError(f"monitor_chain line {line_no} is not an object")
        out.append(rec)
    return out


class MonitorChain:
    """Append-only hash-chained event log at ``path`` (JSONL)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def tip(self) -> str:
        """sha256 of the newest record, or the genesis sentinel when empty."""
        recs = _read(self.path)
        return recs[-1]["sha256"] if recs else _GENESIS_PREV

    def append(
        self,
        kind: str,
        detail: dict[str, Any],
        *,
        actor: str = "system",
        ts: float | None = None,
    ) -> dict[str, Any]:
        recs = _read(self.path)
        rec = _record(
            seq=len(recs),
            ts=time.time() if ts is None else ts,
            actor=actor,
            kind=kind,
            detail=detail,
            prev=recs[-1]["sha256"] if recs else _GENESIS_PREV,
        )
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        return rec

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        return _read(self.path)[-n:]

    def verify(self) -> dict[str, Any]:
        """Replay the chain; every divergence class is reported."""
        errors: list[str] = []
        try:
            recs = _read(self.path)
        except ValueError as exc:
            return {"ok": False, "n": 0, "errors": [str(exc)], "tip": None}
        prev = _GENESIS_PREV
        for i, rec in enumerate(recs):
            if rec.get("seq") != i:
                errors.append(f"seq_gap@{i}: seq={rec.get('seq')}")
            if rec.get("prev_sha256") != prev:
                errors.append(f"chain_break@{i}")
            body = {k: v for k, v in rec.items() if k != "sha256"}
            if hash_bytes(canonical_json_bytes(body)) != rec.get("sha256"):
                errors.append(f"content_edit@{i}")
            prev = str(rec.get("sha256"))
        return {"ok": not errors, "n": len(recs), "errors": errors, "tip": prev}


__all__ = ["KINDS", "MonitorChain"]
