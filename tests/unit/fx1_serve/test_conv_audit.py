"""Tests for fx1.serve.conv_audit — /v1/conversations lifecycle battery."""

from __future__ import annotations

import pytest

from fx1.serve.conv_audit import _SWEPT_ENVS, conv_audit, conv_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

# Probes that pin the contracts repaired by this lane's defects: unique
# id minting (was positional dup ids), atomic item mutation (was
# read-modify-write races), and the conversations journal (was no
# durability). Each was measured divergent on unfixed main and now holds
# against the repaired store.
_FORMER_DEFECTS = {
    "create_seed_ids_unique",
    "ids_unique_across_appends",
    "seed_append_ids_unique",
    "turn_ids_unique_all",
    "turn_ids_unique_vs_seed",
    "dup_submit_appends_both",
    "item_delete_exact",
    "conc_parallel_adds_all_land",
    "conc_parallel_adds_sorted",
    "conc_parallel_adds_unique_ids",
    "conc_add_delete_atomic",
    "conc_parallel_turns_all_items",
    "conc_parallel_turns_unique_ids",
    "bg_mid_delete_no_resurrect",
    "delete_update_404",
    "dur_journal_file_written",
    "dur_conv_survives",
    "dur_items_survive_order",
    "dur_ids_verbatim",
    "dur_tombstone_survives",
    "dur_restart_appendable",
    "dur_restart_ids_unique",
    "dur_torn_tail_conv_survives",
    "dur_torn_tail_truncates",
    "dur_response_index_not_restored",
    "sdk_dur_journal_written",
    "sdk_dur_conv_survives",
    "sdk_dur_items_survive",
    "sdk_dur_restart_appends",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _SWEPT_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_contract_probes_hold() -> None:
    results = conv_audit()
    assert len(results) == 141
    for name, ok in results.items():
        assert ok is True, f"probe {name} failed"


def test_repaired_defect_probes_hold() -> None:
    """Every repaired-defect probe must keep its corrected contract."""
    results = conv_audit()
    assert results.keys() >= _FORMER_DEFECTS
    for name in sorted(_FORMER_DEFECTS):
        assert results[name] is True, f"defect {name} returned"


def test_receipt_verifies() -> None:
    blob = conv_audit_bench()
    assert blob["claim"]["ok"] is True
    assert blob["data_label"] == "SYNTHETIC"
    assert blob["research_only"] is True
    assert blob["live_pnl_claim"] is False
    verdict = verify_receipt_payload(blob)
    assert verdict["valid"] is True


def test_receipt_deterministic() -> None:
    a = conv_audit_bench()
    b = conv_audit_bench()
    assert a["receipt_sha256"] == b["receipt_sha256"]


def test_empty_audit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.serve.conv_audit as audit_module

    monkeypatch.setattr(audit_module, "conv_audit", lambda: {})
    blob = conv_audit_bench()
    assert blob["claim"]["results"] == {}
    assert blob["claim"]["ok"] is False
    assert blob["data_label"] == "SYNTHETIC"
