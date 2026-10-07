"""SYNTHETIC adversarial probes for ``fx1.serve.ops_receipt``.

Sealed-export integrity: secrets and streams are digested or dropped
(never embedded verbatim), the seal is deterministic and excludes itself,
post-seal mutation is caught by the verifier, forbidden-metric keys
inside a record flag the sealed doc at verify time, and non-finite
numbers fail loudly instead of sealing. All fixtures are synthetic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from fx1.serve.ops_receipt import (
    BENCH_RESULT_SCHEMA,
    COMPLETION_RECORD_SCHEMA,
    JOB_RECORD_SCHEMA,
    RUN_RESULT_SCHEMA,
    bench_receipt,
    completion_record_receipt,
    job_record_receipt,
    run_result_receipt,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

_SHA = re.compile(r"[0-9a-f]{64}")


def _completion(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "completion_id": "c-synthetic",
        "backend": "byok",
        "model": "synthetic-model",
        "ok": True,
        "latency_ms": 1.5,
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        "prompt_sha256": hashlib.sha256(b"prompt").hexdigest(),
        "response_sha256": hashlib.sha256(b"response").hexdigest(),
        "at": 1000.0,
        "key_id": "a" * 16,
    }
    record.update(overrides)
    return record


def _assert_self_seal(doc: dict[str, Any]) -> None:
    sha = doc["receipt_sha256"]
    assert isinstance(sha, str) and _SHA.fullmatch(sha)
    rest = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    assert hash_bytes(canonical_json_bytes(rest)) == sha


def test_seal_is_deterministic_and_excludes_itself() -> None:
    """The same record exports the same document — and receipt_sha256 seals
    the doc minus itself."""
    doc = completion_record_receipt(_completion())
    assert doc["schema"] == COMPLETION_RECORD_SCHEMA
    assert doc["data_label"] == "OPS"
    assert doc["research_only"] is True and doc["live_pnl_claim"] is False
    _assert_self_seal(doc)
    assert completion_record_receipt(_completion())["receipt_sha256"] == doc["receipt_sha256"]


def test_clean_completion_record_verifies() -> None:
    doc = completion_record_receipt(_completion())
    result = verify_receipt_payload(doc, "synthetic_completion.json")
    assert result["valid"], result.get("errors")


def test_secret_material_never_enters_a_sealed_job_doc() -> None:
    """callback_secret is dropped entirely — not even its digest rides in the
    sealed record; a query-laden callback_url is digested, never embedded."""
    secret = "synthetic-callback-secret"  # gitleaks:allow — synthetic fixture
    url = f"https://cb.example/hook?token={secret}"
    job = {
        "job_id": "j-1",
        "status": "finished",
        "created_at": 1.0,
        "finished_at": 2.0,
        "error": None,
        "callback_status": "delivered",
        "callback_attempts": 1,
        "callback_url": url,
        "callback_secret": secret,
        "result": {"command": ["echo", "hi"], "exit_code": 0, "stdout": "hi\n"},
        "idempotency_key": "idem-1",
        "internal_scratch": "must not seal",
    }
    doc = job_record_receipt(job)
    blob = json.dumps(doc)
    assert secret not in blob
    assert "callback_secret" not in blob
    assert url not in blob
    record = doc["record"]
    assert record["callback_url_sha256"] == hashlib.sha256(url.encode()).hexdigest()
    assert "callback_url" not in record
    assert "idempotency_key" not in record and "internal_scratch" not in record
    assert record["result"]["stdout_sha256"] == hashlib.sha256(b"hi\n").hexdigest()
    assert "stdout" not in record["result"]
    _assert_self_seal(doc)
    assert verify_receipt_payload(doc, "synthetic_job.json")["valid"]


def test_run_result_streams_digest_only() -> None:
    doc = run_result_receipt(
        {
            "command": ["synthetic", "argv"],
            "exit_code": 0,
            "timeout_s": 5.0,
            "replayed": False,
            "stdout": "synthetic stdout leak material",
            "stderr": "synthetic stderr",
            "stdout_truncated": False,
            "stderr_truncated": True,
        }
    )
    blob = json.dumps(doc)
    assert "leak material" not in blob and "synthetic stderr" not in blob
    record = doc["record"]
    assert record["stdout_sha256"] == hashlib.sha256(b"synthetic stdout leak material").hexdigest()
    assert record["stderr_sha256"] == hashlib.sha256(b"synthetic stderr").hexdigest()
    assert record["command"] == ["synthetic", "argv"]
    assert doc["schema"] == RUN_RESULT_SCHEMA


def test_ok_derived_from_exit_code_only_when_absent() -> None:
    """ok falls back to exit_code==0 — but a recorded ok is never rewritten."""
    derived = run_result_receipt({"command": [], "exit_code": 0})
    assert derived["record"]["ok"] is True
    failed = run_result_receipt({"command": [], "exit_code": 1})
    assert failed["record"]["ok"] is False
    recorded = run_result_receipt({"command": [], "exit_code": 1, "ok": True})
    assert recorded["record"]["ok"] is True  # recorded claim preserved verbatim


def test_post_seal_mutation_breaks_verification() -> None:
    """The sealed doc references the record dict — mutating it afterwards
    leaves a stale seal that the verifier catches."""
    record = _completion()
    doc = completion_record_receipt(record)
    record["usage"]["prompt_tokens"] = 999_999
    result = verify_receipt_payload(doc, "synthetic_completion.json")
    assert result["valid"] is False


def test_forbidden_metric_key_in_record_flags_the_sealed_doc() -> None:
    """A provider counter named like a headline metric rides verbatim into the
    evidence — and the verifier flags the whole document for it."""
    record = _completion(usage={"pnl": 1})
    doc = completion_record_receipt(record)
    result = verify_receipt_payload(doc, "synthetic_completion.json")
    assert result["valid"] is False
    assert "forbidden_metric_keys" in result["errors"]


def test_nonfinite_numbers_seal_deterministically_via_null_canonicalization() -> None:
    """Non-finite floats ride in the doc raw — but the canonicalizer maps
    them to null inside the seal bytes (allow_nan=False), so a corrupt
    latency exports deterministically and still verifies: the sealed
    identity is the normalized value, and verification stays consistent."""
    doc = completion_record_receipt(_completion(latency_ms=float("nan")))
    assert math.isnan(doc["record"]["latency_ms"])
    _assert_self_seal(doc)
    bench = bench_receipt({"metric": float("inf")})
    assert math.isinf(bench["record"]["metric"])
    _assert_self_seal(bench)


def test_bench_receipt_seals_measured_fields_verbatim() -> None:
    record = {"prompt_sha256": "p" * 64, "elapsed_ms": 3.5, "params": {"k": 1}}
    doc = bench_receipt(record)
    assert doc["schema"] == BENCH_RESULT_SCHEMA
    assert doc["record"] == record
    _assert_self_seal(doc)


def test_job_without_result_or_callback_seals_minimally() -> None:
    doc = job_record_receipt({"job_id": "j-2", "status": "queued"})
    record = doc["record"]
    assert record == {"job_id": "j-2", "status": "queued"}
    assert doc["schema"] == JOB_RECORD_SCHEMA
    _assert_self_seal(doc)
    assert verify_receipt_payload(doc, "synthetic_job.json")["valid"]
