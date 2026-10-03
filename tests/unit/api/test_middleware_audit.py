"""middleware_audit lane: auth-middleware contract pinned."""

from __future__ import annotations

from quant_fund.api.middleware_audit import middleware_audit, middleware_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


def test_middleware_contract_edges() -> None:
    r = middleware_audit()
    assert r["remote_no_key_refused"]["status"] == 403
    assert r["remote_no_key_refused"]["remote_on_public_path"] is True
    assert r["empty_key_equals_unset"]["remote_status"] == 403
    assert r["key_required"]["right_key_status"] == 200
    assert r["key_required"]["wrong_key_status"] == 401
    assert r["body_limits"]["oversized_declared"] == 413
    assert r["body_limits"]["malformed_length"] == 400
    assert r["security_headers_on_errors"]["on_status"] == 401
    assert r["security_headers_on_errors"]["csp_present"] is True


def test_bench_ok_and_verifies() -> None:
    receipt = middleware_audit_bench()
    assert receipt["claim"]["ok"] is True
    result = verify_receipt_payload(receipt, "middleware_audit_test.json")
    assert result["valid"], result.get("errors")
