"""dip_audit lane: flagship bench contract pinned."""

from __future__ import annotations

from fx1.bench.dip_audit import dip_audit, dip_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_dip_contract() -> None:
    r = dip_audit()
    d, s, h = r["detection"], r["scoring"], r["honesty"]
    assert d["n_events"] == 1
    assert d["boundary_out"] == {"h1": False, "h2": False, "h3": True}
    assert d["rearm"] is True
    assert d["nonfinite_raises"] == "raise:ValueError"
    assert s["baseline_h1"] == 0.5
    assert s["h2_skips_unobservable"] is True
    assert s["oor_prob_raises"] == "raise:ValueError"
    for k in ("sharpe", "pnl_underscored", "pnl_embedded", "camel_sharpe", "nav_suffix"):
        assert h[k] == "raise:ValueError", k
    assert h["clean"] == "accepted"
    assert h["panel_false_positive_guard"] == "accepted"


def test_bench_ok_and_verifies() -> None:
    receipt = dip_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "dip_audit_test.json")
    assert result["valid"], result.get("errors")
