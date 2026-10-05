"""Tests for fx1.serve.observe_audit — the ops-instrumentation battery."""

from __future__ import annotations

from fx1.serve.observe_audit import observe_audit, observe_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

# Probes that pinned a measured defect which has since been fixed —
# kept as the running ledger of what this battery has caught (mirrors
# the convention in the sibling audit tests).
#  - routing-level refusals (unmatched-path 404, wrong-method 405)
#    escaped the {"detail","code"} envelope because the exception
#    handler was registered on fastapi.HTTPException while Starlette
#    raises the base class for routing errors; fixed by registering
#    the handler on starlette.exceptions.HTTPException.
_FORMER_DEFECTS = {"routing_refusals_enveloped"}


def test_all_probes_hold() -> None:
    results = observe_audit()
    assert len(results) >= 60
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_former_defects_now_hold() -> None:
    """Each formerly pinned defect now measures the fixed contract."""
    results = observe_audit()
    for name in sorted(_FORMER_DEFECTS):
        assert results.get(name) is True, f"former defect {name} regressed"


def test_receipt_verifies() -> None:
    blob = observe_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    first = observe_audit_bench()["receipt_sha256"]
    second = observe_audit_bench()["receipt_sha256"]
    assert first == second
