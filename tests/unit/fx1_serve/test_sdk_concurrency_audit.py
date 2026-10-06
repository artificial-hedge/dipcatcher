"""sdk-concurrency audit evidence — measured thread-safety verdicts."""

from __future__ import annotations

from typing import Any

import pytest

from fx1.serve import sdk_concurrency_audit as audit
from quant_fund.research.receipt_v2 import verify_receipt_payload


@pytest.fixture(scope="module")
def measured() -> dict[str, bool]:
    return audit.sdk_concurrency_audit()


def test_contract_probes_hold(measured: dict[str, bool]) -> None:
    assert len(measured) == 69
    assert all(value is True for value in measured.values()), {
        k: v for k, v in measured.items() if not v
    }


def test_shared_state_seams(measured: dict[str, bool]) -> None:
    assert measured["log_all_calls_recorded"] is True
    assert measured["log_dropped_accounts_exactly"] is True
    assert measured["hdr_pair_consistent_under_interleave"] is True
    assert measured["bg_cancel_map_drains"] is True
    assert measured["files_cap_holds_under_parallel"] is True
    assert measured["storm_no_faults"] is True


@pytest.mark.parametrize("results", [{}, {"probe": False}, {"probe": 1}, {"probe": None}])
def test_empty_or_nonliteral_audit_never_succeeds(
    monkeypatch: pytest.MonkeyPatch, results: dict[str, Any]
) -> None:
    monkeypatch.setattr(audit, "sdk_concurrency_audit", lambda: results)
    receipt = audit.sdk_concurrency_audit_bench()
    assert receipt["claim"]["ok"] is False
    assert receipt["claim"]["results"] == results
    assert verify_receipt_payload(receipt)["valid"] is True


def test_measured_receipt_verifies_without_rerunning(
    measured: dict[str, bool], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "sdk_concurrency_audit", lambda: measured)
    receipt = audit.sdk_concurrency_audit_bench()
    assert receipt["claim"]["ok"] is True
    assert receipt["data_label"] == "SYNTHETIC"
    assert receipt["live_pnl_claim"] is False
    assert verify_receipt_payload(receipt)["valid"] is True
    assert receipt == audit.sdk_concurrency_audit_bench()


def test_tagged_pairing_arithmetic() -> None:
    """The tagged-uuid check must reject cross-call pairs and accept
    same-call consecutive mints — the arithmetic the torn-pair probe
    relies on."""
    assert (
        audit._pair_consistent(
            {
                "x-request-id": f"{(1 << 64) | 2:032x}",
                "x-fx1-completion-id": f"{(1 << 64) | 1:032x}",
            }
        )
        is True
    )
    # torn: request-id from tag 2, completion-id from tag 1
    assert (
        audit._pair_consistent(
            {
                "x-request-id": f"{(2 << 64) | 2:032x}",
                "x-fx1-completion-id": f"{(1 << 64) | 1:032x}",
            }
        )
        is False
    )
    # same tag but skipped counter — a foreign mint landed between them
    assert (
        audit._pair_consistent(
            {
                "x-request-id": f"{(1 << 64) | 9:032x}",
                "x-fx1-completion-id": f"{(1 << 64) | 1:032x}",
            }
        )
        is False
    )
    assert audit._pair_consistent({"x-request-id": "abc"}) is False
