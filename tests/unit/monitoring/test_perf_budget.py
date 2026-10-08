"""Coarse wall-clock perf-budget + exact-economics determinism guards (task 8).

The wall budget is deliberately coarse and pathological (``WALL_BUDGET_S``, a
very generous bound that only trips on a catastrophic >~45x regression) so it is
non-flaky under pytest-xdist. The real, deterministic guarantee is exact:
replaying fills must preserve bit-identical economics (cash and positions),
because idempotent replay never double-applies a fill. SYNTHETIC fixtures only,
never market evidence (live_pnl_claim=False).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from quant_fund.data.qualifying import evaluate_feed
from quant_fund.execution.order_recon import Fill, Order, OrderFillLedger

N_ROWS = 2000
N_FILLS = 2000
WALL_BUDGET_S = 5.0  # coarse pathological guard: only a catastrophic regression trips this

T0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
NOW = datetime(2024, 1, 6, tzinfo=UTC)
DECISION = datetime(2024, 1, 5, tzinfo=UTC)
ATTEST = {"attested": True, "attestation_id": "universe-att-1"}
ADJ = {"attested": True, "attestation_id": "adj-lineage-1"}


def _rows(n: int) -> list[dict]:
    return [
        {
            "security_id": f"SEC_{i}",
            "event_key": f"SEC_{i}:2024-01-02",
            "event_time": "2024-01-02T21:00:00Z",
            "available_time": "2024-01-02T21:05:00Z",
            "ingested_time": "2024-01-02T21:06:00Z",
            "source_id": "vendor_pit_1",
            "revision_id": "rev_0001",
        }
        for i in range(n)
    ]


def test_qualifying_gate_wall_budget() -> None:
    rows = _rows(N_ROWS)
    t0 = time.perf_counter()
    verdict = evaluate_feed(
        rows,
        decision_time=DECISION,
        universe_completeness=ATTEST,
        adjustment_lineage=ADJ,
        now=NOW,
    )
    elapsed = time.perf_counter() - t0
    assert verdict["qualifying"] is True
    assert verdict["n_rows"] == N_ROWS
    assert elapsed < WALL_BUDGET_S


def _build_ledger(with_replays: bool) -> OrderFillLedger:
    ledger = OrderFillLedger(endpoint_kind="paper", starting_cash=1_000_000.0)
    for i in range(N_FILLS):
        side = "BUY" if i % 2 == 0 else "SELL"
        qty = 1.0 + (i % 5)
        price = 10.0 + (i % 3) * 0.5
        order = Order(order_id=f"o{i}", security_id="SEC_A", side=side, quantity=qty)
        fill = Fill(
            fill_id=f"f{i}",
            order_id=f"o{i}",
            security_id="SEC_A",
            quantity=qty,
            price=price,
            fill_time=T0,
            seq=i,
        )
        ledger.submit(order)
        ledger.record_fill(fill)
        if with_replays:
            ledger.submit(order)  # idempotent duplicate order
            ledger.record_fill(fill)  # idempotent duplicate fill
    return ledger


def test_reconciliation_wall_budget() -> None:
    ledger = _build_ledger(with_replays=True)
    t0 = time.perf_counter()
    report = ledger.reconcile()
    elapsed = time.perf_counter() - t0
    assert report["counters"]["fills"] == N_FILLS
    assert report["counters"]["duplicate_fills"] == N_FILLS
    assert elapsed < WALL_BUDGET_S


def test_reference_fast_path_economics_preserved() -> None:
    ref = _build_ledger(with_replays=False).reconcile()
    fast = _build_ledger(with_replays=True).reconcile()
    # Bit-identical economics: idempotent replay never double-applies a fill,
    # so the reference and replay paths agree exactly (not approximately).
    assert ref["computed_cash"] == fast["computed_cash"]
    assert ref["computed_positions"] == fast["computed_positions"]
    assert fast["counters"]["fills"] == N_FILLS
    assert fast["counters"]["orders"] == N_FILLS
