"""dispatch_audit lane: backend dispatch contract pinned."""

from __future__ import annotations

from quant_fund.native.dispatch_audit import dispatch_audit, dispatch_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_flag_and_routing_contract() -> None:
    r = dispatch_audit()
    fp = r["flag_parsing"]
    assert fp["auto"] == "auto"
    assert fp["empty"] == "auto"
    assert fp["whitespace"] == "auto"
    assert fp["python_alias"] == "python"
    assert fp["numpy_alias"] == "python"
    assert fp["rust_alias"] == "rust"
    assert fp["garbage"] == "raise:ValueError"
    assert r["backend_routing"]["python_forces_python"] is True
    assert r["rust_forced"]["ok"] is True
    assert r["parity_across_dispatch"]["ok"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = dispatch_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "dispatch_audit_test.json")
    assert result["valid"], result.get("errors")
