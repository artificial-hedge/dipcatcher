"""Tests for fx1.serve.quota_audit — the quota/rate-limit boundary battery."""

from __future__ import annotations

import pytest

from fx1.serve.quota_audit import quota_audit, quota_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

# Probes that pinned False as measured divergences before this lane's
# fix. Each names the contract it now holds; the assertions below keep
# them regression-pinned True so a reversion re-breaks the test, not
# the receipt.
#
# The measured defect: the managed-key meters — ``uses``,
# ``tokens_used``, ``last_used_at``, and the rpm window occupancy —
# were deliberately not journaled, so a restart on ``--state-dir``
# laundered spent budget: an exhausted key's declared
# ``max_requests``/``max_tokens`` budget reset to full, its spent
# window reopened, and ``requests_remaining`` reverted to the minted
# bound — contradicting the post-#2748 contract that quota state
# survives restart. The fix journals the counter snapshot inside
# ``authenticate`` and ``charge_tokens`` (the mutation points),
# compacts the journal on boot like ``_JobStore``, and backfills
# ``uses``/``last_used_at`` defaults for records journaled before the
# meters existed.
_FORMER_DEFECTS = {
    "persisted_uses_after_restart",
    "persisted_tokens_after_restart",
    "persisted_window_after_restart",
    "persisted_window_refusal_is_rate_limited",
    "persisted_remaining_budget_enforced",
    "persisted_spenddown_to_refusal",
    "persisted_refusal_is_quota",
    "store_replay_reconstructs_counters",
    "store_replay_window_occupancy",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("FX1_API_KEY", "MOONSHOT_API_KEY", "FX1_CHECKPOINT_DIR"):
        monkeypatch.delenv(name, raising=False)


def test_contract_probes_hold() -> None:
    results = quota_audit()
    assert len(results) >= 60
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_defect_probes_fixed() -> None:
    """Each formerly pinned defect now measures the fixed contract."""
    results = quota_audit()
    for name in sorted(_FORMER_DEFECTS):
        assert results.get(name) is True, f"former defect {name} regressed"


def test_receipt_verifies() -> None:
    blob = quota_audit_bench()
    assert blob["claim"]["ok"] is True
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    first = quota_audit_bench()["receipt_sha256"]
    second = quota_audit_bench()["receipt_sha256"]
    assert first == second


def test_committed_receipt_still_verifies() -> None:
    import json
    from pathlib import Path

    path = Path("receipts/fx1_quota_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
