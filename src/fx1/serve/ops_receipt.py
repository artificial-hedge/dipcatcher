"""Sealed operational receipts — per-call and per-job evidence exports.

Every gated call the harness serves lands in the bounded completion log
(``X-Fx1-Completion-Id`` / ``GET /harness/completions``); every async run
lands in the job ledger (``GET /harness/jobs``). This module exports
either record as a sealed document — the same
``receipt_sha256``-over-canonical-JSON convention the evidence corpus
uses — so any caller can hand it to ``verify_receipt`` /
``POST /receipts/verify`` and get hash-consistency verification of the
attested fields. Content is always digested (``*_sha256``), never
embedded. The claim is "these bytes were the recorded record"; it is not
a statement about what a backend or runner actually produced beyond the
sealed hashes.
"""

from __future__ import annotations

import hashlib
from typing import Any

__all__ = [
    "COMPLETION_RECORD_SCHEMA",
    "JOB_RECORD_SCHEMA",
    "RUN_RESULT_SCHEMA",
    "completion_record_receipt",
    "job_record_receipt",
    "run_result_receipt",
]

COMPLETION_RECORD_SCHEMA = "fx1_completion_record.v1"
JOB_RECORD_SCHEMA = "fx1_job_record.v1"
RUN_RESULT_SCHEMA = "fx1_run_result.v1"

_JOB_PASSTHROUGH = (
    "job_id",
    "status",
    "created_at",
    "finished_at",
    "error",
    "callback_status",
    "callback_attempts",
)
_RESULT_PASSTHROUGH = (
    "command",
    "exit_code",
    "ok",
    "timeout_s",
    "replayed",
    "stdout_truncated",
    "stderr_truncated",
)


def _ops_receipt(kind: str, schema: str, record: dict[str, Any]) -> dict[str, Any]:
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    doc: dict[str, Any] = {
        "kind": kind,
        "schema": schema,
        "git_revision": git_revision(),
        "data_label": "OPS",
        "research_only": True,
        "live_pnl_claim": False,
        "record": record,
    }
    doc["receipt_sha256"] = hash_bytes(canonical_json_bytes(doc))
    return doc


def _digested_run_result(result: dict[str, Any]) -> dict[str, Any]:
    """A run record with its streams digested — evidence, not content."""
    out: dict[str, Any] = {}
    for key in _RESULT_PASSTHROUGH:
        if key in result:
            out[key] = result[key]
    for key in ("stdout", "stderr"):
        if key in result:
            out[f"{key}_sha256"] = hashlib.sha256(result[key].encode("utf-8")).hexdigest()
    if "ok" not in out and "exit_code" in result:
        out["ok"] = result["exit_code"] == 0
    return out


def completion_record_receipt(record: dict[str, Any]) -> dict[str, Any]:
    """Seal one completion-log record into a verifiable document.

    Deterministic per record content: the same record exports the same
    document (``git_revision`` is fixed within a process), so exports are
    idempotent and replay-safe.
    """
    return _ops_receipt("fx1_completion_record", COMPLETION_RECORD_SCHEMA, dict(record))


def run_result_receipt(result: dict[str, Any]) -> dict[str, Any]:
    """Seal one run's outcome — ``fx1_run_result.v1`` with
    ``stdout``/``stderr`` reduced to their sha256 digests."""
    return _ops_receipt("fx1_run_result", RUN_RESULT_SCHEMA, _digested_run_result(result))


def job_record_receipt(job: dict[str, Any]) -> dict[str, Any]:
    """Seal one job-ledger record — ``fx1_job_record.v1``.

    The terminal ``result`` embeds as the digested run record;
    ``callback_url`` (caller-supplied, may carry query material) is
    likewise digested; the callback secret never leaves the record."""
    record: dict[str, Any] = {}
    for key in _JOB_PASSTHROUGH:
        if key in job:
            record[key] = job[key]
    callback_url = job.get("callback_url")
    if callback_url:
        record["callback_url_sha256"] = hashlib.sha256(callback_url.encode("utf-8")).hexdigest()
    result = job.get("result")
    if result is not None:
        record["result"] = _digested_run_result(result)
    return _ops_receipt("fx1_job_record", JOB_RECORD_SCHEMA, record)
