"""contract audit: the harness OpenAPI surface is pinned to the golden."""

from __future__ import annotations

import json

from fx1.serve.contract_audit import (
    GOLDEN,
    contract_audit,
    contract_audit_bench,
    normalized_surface,
)
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_all_probes_hold() -> None:
    r = contract_audit()
    bad = {k: v for k, v in r.items() if v is not True}
    assert not bad, bad


def test_live_surface_matches_golden() -> None:
    golden = json.loads(GOLDEN.read_text())
    live = normalized_surface()
    assert live["paths"] == golden["paths"]
    for key in ("openapi", "title", "version"):
        assert live[key] == golden[key]


def test_receipt_verifies() -> None:
    receipt = contract_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "fx1_contract_audit_test.json")
    assert result["valid"], result.get("errors")


def test_receipt_deterministic() -> None:
    a = contract_audit_bench()
    b = contract_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
    assert a["claim"]["results"] == b["claim"]["results"]
