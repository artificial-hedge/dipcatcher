"""dip_audit lane: flagship bench contract pinned."""

from __future__ import annotations

from fx1.bench.dip_audit import dip_audit, dip_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_dip_contract() -> None:
    r = dip_audit()
    d, s, h = r["detection"], r["scoring"], r["honesty"]
    assert d["n_events"] == 1
    assert d["boundary_out"] == {"h1": False, "h2": True, "h3": None}
    assert d["rearm"] is True
    assert d["nonfinite_raises"] == "raise:ValueError"
    assert s["baseline_h1"] == 0.5
    assert s["h2_skips_unobservable"] is True
    assert s["ghost_rejected"] == "raise:ValueError"
    assert s["oor_prob_raises"] == "raise:ValueError"
    outcomes = {c["name"]: c["outcome"] for c in h["cases"]}
    for n in ("ratio_bare", "pl_underscored", "pl_embedded", "camel_ratio", "navlike_suffix"):
        assert outcomes[n] == "raise:ValueError", n
    assert outcomes["clean"] == "accepted"
    assert outcomes["guard_no_false_positive"] == "accepted"


def test_bench_ok_and_verifies() -> None:
    receipt = dip_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "dip_audit_test.json")
    assert result["valid"], result.get("errors")
