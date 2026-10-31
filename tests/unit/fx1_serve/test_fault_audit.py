"""Tests for fx1.serve.fault_audit — adversarial-condition battery."""

from __future__ import annotations

import pytest

from fx1.serve.fault_audit import fault_audit, fault_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

# The historical SYNTHETIC receipt records these seven divergences. Each
# now has a focused source repair and passed the complete live battery.
# Keep explicit positive regressions; never rewrite the historical receipt.
_FORMER_DEFECTS = {
    "keystore_warns_on_corrupt_journal",
    "revoked_key_stays_dead_through_corruption",
    "post_corruption_mint_survives_restart",
    "quota_persists_across_restart",
    "wire_deep_json_fails_closed",
    "idem_race_single_execution",
    "idem_race_jobs_single_id",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_contract_probes_hold() -> None:
    results = fault_audit()
    assert len(results) == 49
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_repaired_defect_probes_hold() -> None:
    """Every former negative probe must keep its corrected contract."""
    results = fault_audit()
    assert results.keys() >= _FORMER_DEFECTS
    for name in sorted(_FORMER_DEFECTS):
        assert results[name] is True, f"defect {name} returned"


def test_receipt_verifies() -> None:
    blob = fault_audit_bench()
    assert blob["claim"]["ok"] is True
    assert blob["data_label"] == "SYNTHETIC"
    assert blob["research_only"] is True
    assert blob["live_pnl_claim"] is False
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = fault_audit_bench()
    b = fault_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_empty_audit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.serve.fault_audit as audit_module

    monkeypatch.setattr(audit_module, "fault_audit", lambda: {})
    blob = fault_audit_bench()
    assert blob["claim"]["results"] == {}
    assert blob["claim"]["ok"] is False
    assert blob["data_label"] == "SYNTHETIC"
