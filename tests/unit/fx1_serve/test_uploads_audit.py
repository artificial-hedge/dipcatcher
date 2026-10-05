"""Tests for fx1.serve.uploads_audit — /v1/uploads lifecycle battery."""

from __future__ import annotations

import pytest

from fx1.serve.uploads_audit import uploads_audit, uploads_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_contract_probes_hold() -> None:
    results = uploads_audit()
    assert len(results) == 93
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_receipt_verifies() -> None:
    blob = uploads_audit_bench()
    assert blob["claim"]["ok"] is True
    assert blob["data_label"] == "SYNTHETIC"
    assert blob["research_only"] is True
    assert blob["live_pnl_claim"] is False
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = uploads_audit_bench()
    b = uploads_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_empty_audit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.serve.uploads_audit as audit_module

    monkeypatch.setattr(audit_module, "uploads_audit", lambda: {})
    blob = uploads_audit_bench()
    assert blob["claim"]["defects"] == ["no_probes"]
    assert blob["claim"]["results"] == {}
    assert blob["claim"]["ok"] is False
    assert blob["data_label"] == "SYNTHETIC"
