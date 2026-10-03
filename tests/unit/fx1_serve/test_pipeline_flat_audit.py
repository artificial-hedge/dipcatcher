"""Pipeline flat-module audit contract tests."""

from __future__ import annotations

from quant_fund.pipeline.flat_audit import pipeline_flat_audit, pipeline_flat_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    results = pipeline_flat_audit()
    failing = {k: v for k, v in results.items() if not v}
    assert not failing, f"probes failed: {sorted(failing)}"


def test_bench_receipt_valid() -> None:
    receipt = pipeline_flat_audit_bench()
    assert receipt["claim"]["ok"] is True
    verdict = verify_receipt_payload(receipt)
    assert verdict["valid"] is True and verdict["errors"] == [], verdict


def test_receipt_deterministic() -> None:
    a = pipeline_flat_audit_bench()
    b = pipeline_flat_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
