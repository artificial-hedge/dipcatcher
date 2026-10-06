"""Tests for fx1.serve.perf_audit — the boundedness battery."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve import perf_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload

_NAMED_PINS = frozenset(
    {
        "envelope_len_never_exceeds_cap",
        "envelope_oldest_evicted_404",
        "stream_refused_at_cap_503",
        "stream_slot_freed_after_release",
        "inflight_gauge_honest_at_cap",
        "bg_jobs_refused_at_cap_503",
        "bg_responses_queued_honestly",
        "bg_responses_drain_to_completed",
        "claim_locks_pruned_after_sweep",
        "claim_locks_wire_bounded",
        "completion_ring_drops_honest",
        "completion_ring_cap_declared",
        "upload_evicted_intent_404",
        "upload_ttl_expires_410",
        "upload_declared_bytes_capped_413",
        "idem_evicted_reexecutes",
        "idem_live_key_replays",
        "header_byok_model_10kb_refused_422",
        "body_5mb_refused_413",
        "body_under_cap_processes",
        "list_absurd_limit_422",
        "cursor_pages_bounded",
        "held_stream_leaves_one_slot",
        "hot_path_p99_bounded",
        "timeout_s_reaches_transport",
        "error_path_burns_no_backend",
        "gate_refusal_burns_no_backend",
        "flood_no_5xx_storm",
        "flood_slots_recovered",
        "memory_growth_bounded",
        "store_dicts_bounded_after_gc",
    }
)


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    return audit.perf_audit()


def test_contract_probes_hold(measured: dict[str, Any]) -> None:
    assert len(measured) == 76
    assert all(value is True for value in measured.values()), measured


def test_named_pins_cover_every_bounded_surface(measured: dict[str, Any]) -> None:
    assert measured.keys() >= _NAMED_PINS
    for name in sorted(_NAMED_PINS):
        assert measured[name] is True, name


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_receipt_refuses_empty_or_nonliteral_success(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "perf_audit", lambda: results)
    receipt = audit.perf_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert receipt["interpretation"].startswith("PERF AUDIT DEFECT:")
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "perf_audit", lambda: measured)
    receipt = audit.perf_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.perf_audit_bench()


def test_committed_receipt_still_verifies() -> None:
    import json
    from pathlib import Path

    path = Path("receipts/fx1_perf_audit.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["claim"]["ok"] is True
    verdict = verify_receipt_payload(payload)
    assert verdict["valid"] is True
    assert verdict["errors"] == []
