"""opsreceipt_audit — sealed operational-receipt contract battery.

``ops_receipt`` is the module every other audit depends on but none
probes: it turns completion-log rows, job-ledger rows, and bench results
into *sealed documents* — the same ``receipt_sha256``-over-canonical-JSON
convention ``verify_receipt_payload`` checks — so a caller can export a
served record as evidence. The module is the harness's disclosure
boundary: content is always digested (``*_sha256``), never embedded, and
caller-supplied strings that may carry secrets (``callback_url``,
``callback_secret``, ``stdout``/``stderr``) must never survive into the
exported record. These probes pin that contract in-process:

- *Envelope* — every doc carries ``kind``/``schema``/``git_revision``,
  is labelled ``OPS`` + ``research_only`` + ``live_pnl_claim: False``,
  seals under the canonical digest, re-verifies through
  ``verify_receipt_payload``, is deterministic per record, does not
  mutate its input, and fails verification after a one-byte tamper.
- *bench_receipt* — ``fx1_bench_result.v1``; the record passes through
  verbatim (params/metrics nested intact).
- *completion_record_receipt* — ``fx1_completion_record.v1``; verbatim
  passthrough, idempotent re-export.
- *run_result_receipt* — ``fx1_run_result.v1``; only the declared
  passthrough keys survive, ``stdout``/``stderr`` are replaced by their
  sha256 digests, ``ok`` is derived from ``exit_code == 0`` only when the
  runner did not report it, and unlisted keys are dropped.
- *job_record_receipt* — ``fx1_job_record.v1``; ledger passthrough keys
  survive, ``callback_url`` ships only as ``callback_url_sha256``,
  ``callback_secret`` never appears anywhere in the export, and a
  terminal ``result`` embeds pre-digested (streams hashed, not content).

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect it found. This battery is in-process and deterministic: no
served app, no clock dependence, no network.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from fx1.serve import ops_receipt

__all__ = ["opsreceipt_audit", "opsreceipt_audit_bench"]

_STDOUT = "diagnostics ok"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verifies(doc: dict[str, Any]) -> bool:
    from quant_fund.research.receipt_v2 import verify_receipt_payload

    return bool(verify_receipt_payload(doc)["valid"])


def _run_record() -> dict[str, Any]:
    return {
        "command": "doctor",
        "exit_code": 0,
        "ok": True,
        "timeout_s": 30,
        "replayed": False,
        "stdout": _STDOUT,
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "runner_pid": 4242,  # not a declared passthrough key — must drop
    }


def _probe_envelope() -> dict[str, bool]:
    out: dict[str, bool] = {}
    record = {"prompt_sha256": _sha("p"), "metrics": {"tok_per_s": 12.5}}
    doc = ops_receipt.bench_receipt(record)

    out["env_kind"] = doc["kind"] == "fx1_bench_result"
    out["env_schema"] = doc["schema"] == ops_receipt.BENCH_RESULT_SCHEMA
    out["env_ops_label"] = doc["data_label"] == "OPS"
    out["env_research_only"] = doc["research_only"] is True
    out["env_no_live_claim"] = doc["live_pnl_claim"] is False
    out["env_git_revision_str"] = isinstance(doc["git_revision"], str) and bool(doc["git_revision"])
    seal = doc["receipt_sha256"]
    out["env_seal_64hex"] = (
        isinstance(seal, str) and len(seal) == 64 and all(c in "0123456789abcdef" for c in seal)
    )
    out["env_record_passthrough"] = doc["record"] == record
    out["env_verifies"] = _verifies(doc)
    out["env_deterministic"] = ops_receipt.bench_receipt(record)["receipt_sha256"] == seal
    snapshot = copy.deepcopy(record)
    ops_receipt.bench_receipt(record)
    out["env_input_not_mutated"] = record == snapshot
    tampered = copy.deepcopy(doc)
    tampered["record"]["metrics"]["tok_per_s"] = 99.9
    out["env_tamper_breaks_seal"] = not _verifies(tampered)
    return out


def _probe_bench() -> dict[str, bool]:
    out: dict[str, bool] = {}
    record = {
        "prompt_sha256": _sha("bench prompt"),
        "params": {"temperature": 0.0, "max_tokens": 8},
        "metrics": {"tok_per_s": 41.0, "ttft_s": 0.21},
    }
    doc = ops_receipt.bench_receipt(record)
    out["bench_kind_schema"] = (
        doc["kind"] == "fx1_bench_result" and doc["schema"] == "fx1_bench_result.v1"
    )
    out["bench_record_verbatim"] = doc["record"] == record
    out["bench_verifies"] = _verifies(doc)
    return out


def _probe_completion() -> dict[str, bool]:
    out: dict[str, bool] = {}
    record = {
        "completion_id": "cmp-abc",
        "model": "fx-1",
        "created": 1700000000,
        "usage": {"total_tokens": 17},
        "response_sha256": _sha("body"),
    }
    doc = ops_receipt.completion_record_receipt(record)
    out["completion_kind_schema"] = (
        doc["kind"] == "fx1_completion_record" and doc["schema"] == "fx1_completion_record.v1"
    )
    out["completion_record_verbatim"] = doc["record"] == record
    out["completion_deterministic"] = (
        ops_receipt.completion_record_receipt(record)["receipt_sha256"] == doc["receipt_sha256"]
    )
    out["completion_verifies"] = _verifies(doc)
    return out


def _probe_run_result() -> dict[str, bool]:
    out: dict[str, bool] = {}
    record = _run_record()
    doc = ops_receipt.run_result_receipt(record)
    sealed = doc["record"]

    out["run_kind_schema"] = (
        doc["kind"] == "fx1_run_result" and doc["schema"] == "fx1_run_result.v1"
    )
    out["run_stdout_digested"] = "stdout" not in sealed and sealed["stdout_sha256"] == _sha(_STDOUT)
    out["run_stderr_digested"] = "stderr" not in sealed and sealed["stderr_sha256"] == _sha("")
    out["run_passthrough_keys"] = all(
        sealed.get(k) == record[k]
        for k in (
            "command",
            "exit_code",
            "ok",
            "timeout_s",
            "replayed",
            "stdout_truncated",
            "stderr_truncated",
        )
    )
    out["run_nonlisted_dropped"] = "runner_pid" not in sealed

    derived = ops_receipt.run_result_receipt({"command": "x", "exit_code": 0})
    out["run_ok_derived_zero"] = derived["record"]["ok"] is True
    derived_bad = ops_receipt.run_result_receipt({"command": "x", "exit_code": 2})
    out["run_ok_derived_nonzero"] = derived_bad["record"]["ok"] is False
    preserved = ops_receipt.run_result_receipt({"command": "x", "exit_code": 0, "ok": False})
    out["run_ok_explicit_wins"] = preserved["record"]["ok"] is False
    out["run_no_streams_no_digest_keys"] = (
        "stdout_sha256" not in derived["record"] and "stderr_sha256" not in derived["record"]
    )
    out["run_verifies"] = _verifies(doc)
    return out


def _probe_job_record() -> dict[str, bool]:
    out: dict[str, bool] = {}
    url = "https://tenant.example/cb?token=sekrit"
    job = {
        "job_id": "job-1",
        "status": "succeeded",
        "created_at": 1700000000,
        "finished_at": 1700000009,
        "error": None,
        "callback_status": "delivered",
        "callback_attempts": 1,
        "callback_url": url,
        "callback_secret": "whsec_topsecret",
        "internal_note": "drop me",
        "result": _run_record(),
    }
    doc = ops_receipt.job_record_receipt(job)
    sealed = doc["record"]

    out["job_kind_schema"] = (
        doc["kind"] == "fx1_job_record" and doc["schema"] == "fx1_job_record.v1"
    )
    out["job_passthrough_keys"] = all(
        sealed.get(k) == job[k]
        for k in (
            "job_id",
            "status",
            "created_at",
            "finished_at",
            "error",
            "callback_status",
            "callback_attempts",
        )
    )
    out["job_callback_url_digested"] = "callback_url" not in sealed and sealed[
        "callback_url_sha256"
    ] == _sha(url)
    out["job_secret_never_leaks"] = "callback_secret" not in sealed and (
        "whsec_topsecret" not in json.dumps(doc)
    )
    out["job_nonlisted_dropped"] = "internal_note" not in sealed
    res = sealed["result"]
    out["job_result_predigested"] = "stdout" not in res and res["stdout_sha256"] == _sha(_STDOUT)

    no_url = ops_receipt.job_record_receipt({"job_id": "j2", "status": "queued"})
    out["job_no_url_no_digest_key"] = "callback_url_sha256" not in no_url["record"]
    empty_url = ops_receipt.job_record_receipt({"job_id": "j3", "callback_url": ""})
    out["job_empty_url_no_digest_key"] = "callback_url_sha256" not in empty_url["record"]
    no_result = ops_receipt.job_record_receipt({"job_id": "j4", "result": None})
    out["job_null_result_dropped"] = "result" not in no_result["record"]
    out["job_verifies"] = _verifies(doc)
    return out


def _probe_schema_family() -> dict[str, bool]:
    out: dict[str, bool] = {}
    schemas = {
        ops_receipt.BENCH_RESULT_SCHEMA,
        ops_receipt.COMPLETION_RECORD_SCHEMA,
        ops_receipt.JOB_RECORD_SCHEMA,
        ops_receipt.RUN_RESULT_SCHEMA,
    }
    out["schemas_distinct"] = len(schemas) == 4
    docs = [
        ops_receipt.bench_receipt({"a": 1}),
        ops_receipt.completion_record_receipt({"b": 2}),
        ops_receipt.run_result_receipt({"command": "x", "exit_code": 0}),
        ops_receipt.job_record_receipt({"job_id": "j"}),
    ]
    out["all_docs_ops_labelled"] = all(
        d["data_label"] == "OPS" and d["research_only"] is True and d["live_pnl_claim"] is False
        for d in docs
    )
    out["all_docs_verify"] = all(_verifies(d) for d in docs)
    return out


def opsreceipt_audit() -> dict[str, bool]:
    """Every sealed-export contract as literal booleans."""
    out: dict[str, bool] = {}
    out.update(_probe_envelope())
    out.update(_probe_bench())
    out.update(_probe_completion())
    out.update(_probe_run_result())
    out.update(_probe_job_record())
    out.update(_probe_schema_family())
    return out


def opsreceipt_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = opsreceipt_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "opsreceipt_audit",
        "schema": "opsreceipt_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process function calls; no served app, no network",
            "not_verified": [
                "receipt store persistence paths",
                "verify route HTTP wiring (covered by api/serve audits)",
            ],
        },
        "interpretation": (
            "Sealed operational-receipt contract holds: every export carries "
            "kind/schema/git_revision under the OPS + research_only + "
            "no-live-PnL labels, seals under receipt_sha256-over-canonical-"
            "JSON, re-verifies through verify_receipt_payload, and is "
            "deterministic per record; bench and completion records pass "
            "through verbatim; run results drop undeclared keys and ship "
            "stdout/stderr as sha256 digests with ok derived from exit_code "
            "only when unreported; job records ship callback_url as a digest "
            "and never leak callback_secret, with terminal results "
            "pre-digested. Tampering with a sealed record breaks "
            "verification."
            if ok
            else f"OPSRECEIPT AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(opsreceipt_audit_bench(), indent=2, sort_keys=True))
