"""QUALIFYING-flag test matrix for the condition-1 licensed-vendor gate.

SYNTHETIC fixtures only (correctness tests, never market evidence). Covers the
required acceptance matrix: missing field -> non-qualifying, late revision ->
rejected, decision-time leak -> rejected, plus causal-chain, UNAVAILABLE
fail-closed (no synthetic substitution), and hashed PIT receipts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from quant_fund.data.qualifying import (
    QualifyingFeedError,
    evaluate_feed,
    pit_receipt,
    resolve_feed,
    unavailable_feed,
)

NOW = datetime(2024, 1, 6, tzinfo=UTC)
DECISION = datetime(2024, 1, 5, tzinfo=UTC)
ATTEST = {"attested": True, "attestation_id": "universe-att-1"}
ADJ = {"attested": True, "attestation_id": "adj-lineage-1"}


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "security_id": "SEC_A",
        "event_key": "SEC_A:2024-01-02",
        "event_time": "2024-01-02T21:00:00Z",
        "available_time": "2024-01-02T21:05:00Z",
        "ingested_time": "2024-01-02T21:06:00Z",
        "source_id": "vendor_pit_1",
        "revision_id": "rev_0001",
    }
    row.update(overrides)
    return row


def _evaluate(rows: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "decision_time": DECISION,
        "universe_completeness": ATTEST,
        "adjustment_lineage": ADJ,
        "now": NOW,
    }
    params.update(kw)
    return evaluate_feed(rows, **params)


# -- happy path --


def test_complete_consistent_feed_is_qualifying() -> None:
    verdict = _evaluate([_row()])
    assert verdict["qualifying"] is True
    assert verdict["status"] == "QUALIFYING"
    assert verdict["rejected"] is False
    assert verdict["reasons"] == []
    assert verdict["synthetic_substitution"] is False
    assert verdict["live_pnl_claim"] is False


# -- missing field -> non-qualifying (every condition-1 field) --


def test_missing_field_is_non_qualifying() -> None:
    for field in ("event_time", "available_time", "ingested_time", "source_id", "revision_id"):
        row = _row()
        row.pop(field)
        verdict = _evaluate([row])
        assert verdict["qualifying"] is False, field
        assert verdict["status"] == "NON_QUALIFYING", field
        assert verdict["rejected"] is False, field
        assert any(r.endswith(f":{field}") for r in verdict["reasons"]), field


def test_missing_attestation_is_non_qualifying() -> None:
    verdict = _evaluate([_row()], universe_completeness=None)
    assert verdict["qualifying"] is False
    assert verdict["status"] == "NON_QUALIFYING"
    assert "missing_attestation:universe_completeness" in verdict["reasons"]

    verdict = _evaluate([_row()], adjustment_lineage=None)
    assert verdict["qualifying"] is False
    assert "missing_attestation:adjustment_lineage" in verdict["reasons"]

    verdict = _evaluate([_row()], universe_completeness={"attested": False, "attestation_id": "x"})
    assert verdict["qualifying"] is False
    assert "unattested:universe_completeness" in verdict["reasons"]


def test_inconsistent_source_id_is_non_qualifying() -> None:
    verdict = _evaluate(
        [
            _row(security_id="SEC_A", event_key="A:1", source_id="vendor_1"),
            _row(security_id="SEC_B", event_key="B:1", source_id="vendor_2"),
        ]
    )
    assert verdict["qualifying"] is False
    assert "source_id_inconsistent" in verdict["reasons"]


# -- decision-time leak -> rejected --


def test_decision_time_leak_is_rejected() -> None:
    # available_time (21:10) is after decision_time (2024-01-05 is later, so use
    # an available_time strictly after DECISION).
    leak = _row(available_time="2024-01-05T12:00:00Z", ingested_time="2024-01-05T12:01:00Z")
    verdict = _evaluate([leak])
    assert verdict["qualifying"] is False
    assert verdict["rejected"] is True
    assert verdict["status"] == "REJECTED"
    assert any(r.startswith("decision_time_leak") for r in verdict["reasons"])


# -- late revision -> rejected --


def test_late_revision_is_rejected() -> None:
    first = _row(event_key="SEC_A:2024-01-02", revision_id="rev_0001")
    revision = _row(
        event_key="SEC_A:2024-01-02",
        revision_id="rev_0002",
        available_time="2024-01-05T12:00:00Z",
        ingested_time="2024-01-05T12:01:00Z",
    )
    verdict = _evaluate([first, revision])
    assert verdict["qualifying"] is False
    assert verdict["rejected"] is True
    assert verdict["status"] == "REJECTED"
    assert any(r.startswith("late_revision") for r in verdict["reasons"])


def test_out_of_order_revision_is_rejected() -> None:
    first = _row(
        event_key="SEC_A:2024-01-02",
        revision_id="rev_0002",
        available_time="2024-01-03T21:05:00Z",
        ingested_time="2024-01-03T21:06:00Z",
    )
    earlier = _row(event_key="SEC_A:2024-01-02", revision_id="rev_0001")
    verdict = _evaluate([first, earlier])
    assert verdict["rejected"] is True
    assert any(r.startswith("late_revision") for r in verdict["reasons"])


# -- causal chain break -> rejected --


def test_causal_chain_violation_is_rejected() -> None:
    # available_time before event_time breaks event <= available.
    broken = _row(available_time="2024-01-01T21:05:00Z", ingested_time="2024-01-01T21:06:00Z")
    verdict = _evaluate([broken])
    assert verdict["rejected"] is True
    assert any(r.startswith("causal_chain_violation") for r in verdict["reasons"])


def test_future_ingest_is_rejected() -> None:
    verdict = _evaluate(
        [_row(ingested_time="2999-01-01T00:00:00Z", available_time="2999-01-01T00:00:00Z")],
        now=NOW,
    )
    assert verdict["rejected"] is True
    assert any(r.startswith("future_ingest") for r in verdict["reasons"])


# -- UNAVAILABLE fail-closed path: no synthetic substitution --


def test_unavailable_path_has_no_synthetic_substitution() -> None:
    verdict = unavailable_feed("no_entitlement_configured")
    assert verdict["qualifying"] is False
    assert verdict["status"] == "UNAVAILABLE"
    assert verdict["synthetic_substitution"] is False
    assert verdict["entitlement"] == "absent"


def test_resolve_feed_unentitled_is_unavailable_and_ignores_rows() -> None:
    verdict = resolve_feed(entitled=False, rows=[_row()])
    assert verdict["status"] == "UNAVAILABLE"
    assert verdict["qualifying"] is False
    assert verdict["n_rows"] == 0
    assert verdict["synthetic_substitution"] is False


def test_resolve_feed_entitled_requires_decision_time() -> None:
    try:
        resolve_feed(entitled=True, rows=[_row()])
    except QualifyingFeedError:
        pass
    else:
        raise AssertionError("expected QualifyingFeedError when decision_time is missing")


# -- hashed PIT receipts --


def test_pit_receipt_is_deterministic_and_content_addressed() -> None:
    rows = [_row()]
    r1 = pit_receipt(rows)
    r2 = pit_receipt(rows)
    assert r1["receipt_sha256"] == r2["receipt_sha256"]
    assert r1["n_rows"] == 1
    assert len(r1["receipt_sha256"]) == 64

    edited = pit_receipt([_row(close=1.0)])
    assert edited["receipt_sha256"] != r1["receipt_sha256"]


def test_verdict_embeds_pit_receipt() -> None:
    verdict = _evaluate([_row()])
    assert verdict["pit_receipt"]["schema"] == "pit_receipt.v1"
    assert verdict["pit_receipt"]["synthetic_substitution"] is False
