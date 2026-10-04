"""Tests for fx1.serve.fault_audit — adversarial-condition battery."""

from __future__ import annotations

import pytest

from fx1.serve.fault_audit import fault_audit, fault_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

# Probes pinned False on live, measured divergences. Each names the
# expected contract; when a fix lands the probe flips True, the failing
# assertion below forces the pin to shrink, and the receipt reseals.
_KNOWN_DEFECTS = {
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
    for name, ok in results.items():
        if name not in _KNOWN_DEFECTS:
            assert ok is True, f"probe {name} failed"


def test_defect_probes_reproduce() -> None:
    """Each pinned defect still measures the live divergence."""
    results = fault_audit()
    assert results.keys() >= _KNOWN_DEFECTS
    for name in sorted(_KNOWN_DEFECTS):
        assert results[name] is False, f"defect {name} no longer reproduces"


def test_receipt_verifies() -> None:
    blob = fault_audit_bench()
    assert blob["claim"]["ok"] is False  # defects are pinned, not hidden
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = fault_audit_bench()
    b = fault_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]
