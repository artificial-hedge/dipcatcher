"""Sealed per-call completion-record receipts.

Every gated call the harness serves lands in the bounded completion log
(``X-Fx1-Completion-Id`` / ``GET /harness/completions``). This module
exports one record as a sealed document — the same
``receipt_sha256``-over-canonical-JSON convention the evidence corpus
uses — so any caller can hand it to ``verify_receipt`` /
``POST /receipts/verify`` and get hash-consistency verification of the
attested fields (prompt/output digests, verdict, usage, latency). The
claim is "these bytes were the recorded record"; it is not a statement
about what any backend actually returned beyond the sealed hashes.
"""

from __future__ import annotations

from typing import Any

__all__ = ["COMPLETION_RECORD_SCHEMA", "completion_record_receipt"]

COMPLETION_RECORD_SCHEMA = "fx1_completion_record.v1"


def completion_record_receipt(record: dict[str, Any]) -> dict[str, Any]:
    """Seal one completion-log record into a verifiable document.

    Deterministic per record content: the same record exports the same
    document (``git_revision`` is fixed within a process), so exports are
    idempotent and replay-safe.
    """
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    doc: dict[str, Any] = {
        "kind": "fx1_completion_record",
        "schema": COMPLETION_RECORD_SCHEMA,
        "git_revision": git_revision(),
        "data_label": "OPS",
        "research_only": True,
        "live_pnl_claim": False,
        "record": dict(record),
    }
    doc["receipt_sha256"] = hash_bytes(canonical_json_bytes(doc))
    return doc
