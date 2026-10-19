"""Load and minimally verify dipcatcher receipts for corpus use."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from quant_fund.utils.receipt import corpus_training_refused


class ReceiptRecord(BaseModel):
    path: str
    sha256: str = Field(description="SHA-256 of the raw receipt file bytes")
    schema_name: str = Field(default="unknown")
    research_only: bool
    live_pnl_claim: bool
    disclaimer: str = ""
    evidence_class: str = Field(default="research")
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.research_only and not self.live_pnl_claim


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eligibility(payload: dict[str, Any]) -> tuple[bool, bool]:
    claim = payload.get("claim")
    research_only = (payload.get("research_only") is True) or (claim == "research_only")
    if "live_pnl_claim" in payload:
        live_pnl_claim = bool(payload["live_pnl_claim"])
    else:
        live_pnl_claim = claim != "research_only"
    return research_only, live_pnl_claim


def load_receipts(
    receipts_dir: str | Path | Iterable[str | Path],
    *,
    repo_root: Path | None = None,
) -> tuple[list[ReceiptRecord], int]:
    if isinstance(receipts_dir, str | Path):
        roots = [Path(receipts_dir)]
    else:
        roots = [Path(d) for d in receipts_dir]
    records: list[ReceiptRecord] = []
    refused = 0
    for root in roots:
        for path in sorted(root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            digest = _sha256(path)
            if corpus_training_refused(path, root, payload, digest, repo_root=repo_root):
                refused += 1
                continue
            research_only, live_pnl_claim = _eligibility(payload)
            records.append(
                ReceiptRecord(
                    path=str(path),
                    sha256=digest,
                    schema_name=str(
                        payload.get("schema", payload.get("schema_version", "unknown"))
                    ),
                    research_only=research_only,
                    live_pnl_claim=live_pnl_claim,
                    disclaimer=str(payload.get("disclaimer", "")),
                    evidence_class=("synthetic" if payload.get("synthetic") else "research"),
                    payload=payload,
                )
            )
    return records, refused
