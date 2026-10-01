"""Contract tests for the forecast-pipeline audit lane."""

from __future__ import annotations

from quant_fund.pipeline.forecast_audit import (
    forecast_pipeline_audit,
    forecast_pipeline_audit_bench,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    results = forecast_pipeline_audit()
    assert len(results) >= 30
    failed = [name for name, held in results.items() if not held]
    assert failed == []


def test_bench_receipt_verifies() -> None:
    receipt = forecast_pipeline_audit_bench()
    assert receipt["claim"]["ok"] is True
    verdict = verify_receipt_payload(receipt)
    assert verdict["valid"] is True
    assert verdict["errors"] == []


def test_bench_receipt_sha_deterministic() -> None:
    first = forecast_pipeline_audit_bench()
    second = forecast_pipeline_audit_bench()
    assert first["receipt_sha256"] == second["receipt_sha256"]
    assert len(first["receipt_sha256"]) == 64
