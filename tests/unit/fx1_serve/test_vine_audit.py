"""Tests for the pair_vine_copula audit + sealed receipt."""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.models.vine_audit import vine_audit, vine_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_pass() -> None:
    results = vine_audit()
    failed = [k for k, v in results.items() if not v]
    assert failed == [], f"failing probes: {failed}"


def test_bench_receipt_verifies() -> None:
    payload = vine_audit_bench()
    assert payload["claim"]["ok"] is True
    assert payload["schema"] == "vine_audit.v1"
    assert payload["receipt_sha256"]
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []


def test_committed_receipt_still_verifies() -> None:
    path = Path("receipts/vine_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []


def test_receipt_deterministic() -> None:
    a = vine_audit_bench()
    b = vine_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
