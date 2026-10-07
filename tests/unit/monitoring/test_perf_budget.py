"""Representative-workload perf budgets (deterministic, fast, PR-gate safe).

Two budget families, both SYNTHETIC correctness/performance tests (never market
evidence):

1. **Wall-clock budget** — a representative workload (condition-1 qualifying
   gate; order/fill reconciliation) completes within a generous budget. The
   budget is intentionally loose so the PR gate stays deterministic under load
   while still catching a pathological slowdown.
2. **Reference/fast-path economics preserved** — the idempotent-replay fast path
   leaves cash/positions bit-identical to a single-pass reference, and the
   op-shape budget (work actually applied) is exact despite replays.

The broader ``run_backtest`` reference/fast-path economics remain governed by
``docs/PERF_SWEEP.md`` / ``docs/FAST_REPLAY_P42.md`` and are unchanged here.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from quant_fund.data.qualifying import evaluate_feed
from quant_fund.execution.order_recon import Fill, Order, OrderFillLedger

N_ROWS = 2000
N_FILLS = 2000
# Coarse pathological guard, not a fine-grained regression detector. Measured
# workload wall time is ~0.03–0.11s; this generous ceiling only trips on a large
# (>~45x) regression and stays non-flaky under xdist CPU contention. The
# deterministic guarantee is the exact-economics/op-shape assertions below.
WALL_BUDGET_S = 5.0

BASE = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
DECISION = datetime(2024, 1, 5, tzinfo=UTC)
NOW = datetime(2024, 1, 6, tzinfo=UTC)
ATTEST = {"attested": True, "attestation_id": "u1"}
ADJ = {"attested": True, "attestation_id": "a1"}


def _rows(n: int) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for i in range(n):
        ts = BASE + timedelta(seconds=i)
        out.append(
            {
                "security_id": f"SEC_{i % 7}",
                "event_key": f"SEC_{i % 7}:{i}",
                "event_time": ts.isoformat(),
                "available_time": (ts + timedelta(seconds=1)).isoformat(),
                "ingested_time": (ts + timedelta(seconds=2)).isoformat(),
                "source_id": "vendor_pit_1",
                "revision_id": f"rev_{i:06d}",
            }
        )
    return out


def test_qualifying_gate_wall_budget() -> None:
    rows = _rows(N_ROWS)
    start = time.perf_counter()
    verdict = evaluate_feed(
        rows, decision_time=DECISION, universe_completeness=ATTEST, adjustment_lineage=ADJ, now=NOW
    )
    elapsed = time.perf_counter() - start
    assert verdict["qualifying"] is True
    assert verdict["n_rows"] == N_ROWS
    assert elapsed < WALL_BUDGET_S, f"qualifying gate took {elapsed:.3f}s"


def _build_ledger(with_replays: bool) -> tuple[OrderFillLedger, float]:
    led = OrderFillLedger(endpoint_kind="paper", starting_cash=1_000_000.0)
    cash = 1_000_000.0
    positions = 0.0
    for i in range(N_FILLS):
        side = "BUY" if i % 2 == 0 else "SELL"
        qty = 1.0 + (i % 5)
        price = 10.0 + (i % 3) * 0.5
        order = Order(order_id=f"o{i}", security_id="SEC_X", side=side, quantity=qty)  # type: ignore[arg-type]
        led.submit(order)
        fill = Fill(
            fill_id=f"f{i}",
            order_id=f"o{i}",
            security_id="SEC_X",
            quantity=qty,
            price=price,
            fill_time=BASE + timedelta(seconds=i),
        )
        led.record_fill(fill)
        signed = qty if side == "BUY" else -qty
        cash -= signed * price
        positions += signed
        if with_replays:
            led.submit(order)  # idempotent replay
            led.record_fill(fill)  # idempotent replay
    return led, cash


def test_reconciliation_wall_budget() -> None:
    start = time.perf_counter()
    led, _ = _build_ledger(with_replays=True)
    report = led.reconcile()
    elapsed = time.perf_counter() - start
    assert report["counters"]["fills"] == N_FILLS  # exact op-shape despite replays
    assert report["counters"]["duplicate_fills"] == N_FILLS
    assert elapsed < WALL_BUDGET_S, f"reconciliation took {elapsed:.3f}s"


def test_reference_fast_path_economics_preserved() -> None:
    ref_led, ref_cash = _build_ledger(with_replays=False)
    fast_led, fast_cash = _build_ledger(with_replays=True)
    ref = ref_led.reconcile()
    fast = fast_led.reconcile()
    # cash/positions are bit-identical: replays never change economics.
    assert ref["computed_cash"] == fast["computed_cash"] == ref_cash == fast_cash
    assert ref["computed_positions"] == fast["computed_positions"]
    # the fast path applied each fill exactly once.
    assert fast["counters"]["fills"] == N_FILLS
    assert fast["counters"]["orders"] == N_FILLS
