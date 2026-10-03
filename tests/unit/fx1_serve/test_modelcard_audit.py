"""modelcard_audit lane: card contract + ship-gate boundaries pinned."""

from __future__ import annotations

from fx1.modelcard_audit import modelcard_audit, modelcard_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_version_and_scope() -> None:
    r = modelcard_audit()
    assert len(r["bad_versions"]) == 5
    assert r["good_version_ok"] is True
    assert r["live_rejected"] is True
    assert r["nonresearch_rejected"] is True
    assert r["bad_hash_rejected"] is True


def test_ship_gate_boundaries() -> None:
    r = modelcard_audit()
    assert r["ship_baseline"] is True
    assert r["ship_domain_tie_refused"] is True
    assert r["ship_general_regress_refused"] is True
    assert r["ship_dishonest_refused"] is True
    assert r["roundtrip"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = modelcard_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "modelcard_audit_test.json")
    assert result["valid"], result.get("errors")
