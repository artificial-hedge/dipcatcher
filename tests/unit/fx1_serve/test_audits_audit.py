"""Tests for the meta battery — fx1.audits_audit."""

from __future__ import annotations

import pytest

from fx1 import audits_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_all_probes_hold() -> None:
    results = audit.audits_audit()
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = audit.audits_audit_bench()
    assert blob["claim"]["ok"] is True
    assert verify_receipt_payload(blob)["valid"] is True


def test_receipt_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = {"a": True, "b": True}
    monkeypatch.setattr(audit, "audits_audit", lambda: fixed)
    a = audit.audits_audit_bench()
    b = audit.audits_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
    assert a["claim"]["ok"] is True


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}])
def test_empty_or_nonliteral_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, bool]
) -> None:
    monkeypatch.setattr(audit, "audits_audit", lambda: results)
    blob = audit.audits_audit_bench()
    assert blob["claim"]["ok"] is False
    assert blob["claim"]["results"] == results
    assert verify_receipt_payload(blob)["valid"] is True
    if not results:
        assert "no_probes" in blob["interpretation"]
