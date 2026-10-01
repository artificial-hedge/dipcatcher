"""Prefix determinism: state at bar k must not depend on bars after k.

Causality is the load-bearing claim of both engines: a backtest that reads
future bars is worthless. The strictest check is not an index bound — it is
to *truncate the tape* and require that every row up to the cut is
byte-identical to the full run. A non-causal read anywhere in the engine
(leakage, whole-panel normalization, N-dependent RNG) shifts those rows.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.engine import _run_backtest_event_loop
from quant_fund.backtest.fast_replay import run_backtest_fast
from tests.unit.backtest.test_fast_replay import _bars, _cfg, _weights


def _prefix_frame(df: pl.DataFrame, cutoff) -> pl.DataFrame:
    col = "event_time" if "event_time" in df.columns else "fill_time"
    return df.filter(pl.col(col) <= cutoff)


def _row_bytes(df: pl.DataFrame) -> list[tuple]:
    """Row-wise digest for positionally comparing prefix runs."""
    return list(df.sort(df.columns).iter_rows())


def _cutoffs(bars: pl.DataFrame, ks: list[int]) -> list:
    dates = bars.get_column("event_time").unique().sort()
    return [dates[k] for k in ks if k < len(dates)]


def _assert_prefix_equal(full: pl.DataFrame, trunc: pl.DataFrame, cutoff) -> None:
    fp = _prefix_frame(full, cutoff)
    assert _row_bytes(trunc) == _row_bytes(fp), (
        "engine state before the cut differs from the full run — "
        "a future-bar read leaks into the prefix"
    )


@pytest.mark.parametrize("engine", ["event", "fast"])
@pytest.mark.parametrize("missing", [0.0, 0.15])
def test_prefix_determinism(engine: str, missing: float) -> None:
    rng = np.random.default_rng(7)
    sids = ["A", "B", "C"]
    bars = _bars(sids, 40, rng, missing=missing)
    weights = _weights(sids, 40, rng, sparse=0.2)
    cfg = _cfg(commission_bps=8.0, half_spread_bps=2.0, impact_y=0.1)

    run = (
        (lambda b, w: run_backtest_fast(b, w, cfg))
        if engine == "fast"
        else (lambda b, w: _run_backtest_event_loop(b, w, cfg))
    )

    full = run(bars, weights)
    for cutoff in _cutoffs(bars, [5, 12, 19, 27, 34]):
        trunc = run(_prefix_frame(bars, cutoff), _prefix_frame(weights, cutoff))
        _assert_prefix_equal(full.equity, trunc.equity, cutoff)
        _assert_prefix_equal(full.fills, trunc.fills, cutoff)


def test_prefix_determinism_seeded_fuzz() -> None:
    """Sweep seeds/configs — any divergence on any cut is a causality defect."""
    for seed in range(6):
        rng = np.random.default_rng(seed)
        sids = [f"S{i}" for i in range(1 + seed % 3)]
        n = 30 + 5 * seed
        bars = _bars(sids, n, rng, missing=0.1 * (seed % 2))
        weights = _weights(sids, n, rng, sparse=0.15)
        cfg = _cfg(
            commission_bps=5.0 * seed,
            half_spread_bps=1.0,
            impact_y=0.05,
            participation_limit=0.5 + 0.1 * seed,
        )
        full_fast = run_backtest_fast(bars, weights, cfg)
        full_event = _run_backtest_event_loop(bars, weights, cfg)
        for cutoff in _cutoffs(bars, [n // 3, n // 2, (2 * n) // 3]):
            bars_k = _prefix_frame(bars, cutoff)
            weights_k = _prefix_frame(weights, cutoff)
            trunc_fast = run_backtest_fast(bars_k, weights_k, cfg)
            trunc_event = _run_backtest_event_loop(bars_k, weights_k, cfg)
            _assert_prefix_equal(full_fast.equity, trunc_fast.equity, cutoff)
            _assert_prefix_equal(full_fast.fills, trunc_fast.fills, cutoff)
            _assert_prefix_equal(full_event.equity, trunc_event.equity, cutoff)
            _assert_prefix_equal(full_event.fills, trunc_event.fills, cutoff)


def _sorted_rows(df: pl.DataFrame) -> list[tuple]:
    return sorted(df.iter_rows())


@pytest.mark.parametrize("engine", ["event", "fast"])
def test_input_row_order_invariance(engine: str) -> None:
    """Parquet row order is physical layout, not semantics — shuffling the
    input panels must not change the book."""
    rng = np.random.default_rng(11)
    sids = ["A", "B", "C"]
    bars = _bars(sids, 25, rng, missing=0.1)
    weights = _weights(sids, 25, rng, sparse=0.15)
    cfg = _cfg(commission_bps=8.0, half_spread_bps=2.0, impact_y=0.1)
    run = (
        (lambda b, w: run_backtest_fast(b, w, cfg))
        if engine == "fast"
        else (lambda b, w: _run_backtest_event_loop(b, w, cfg))
    )
    base = run(bars, weights)
    for s in range(8):
        result = run(
            bars.sample(fraction=1.0, seed=100 + s, shuffle=True),
            weights.sample(fraction=1.0, seed=200 + s, shuffle=True),
        )
        assert _sorted_rows(base.equity) == _sorted_rows(result.equity)
        assert _sorted_rows(base.fills) == _sorted_rows(result.fills)


@pytest.mark.parametrize("engine", ["event", "fast"])
def test_cost_monotonicity(engine: str) -> None:
    """Raising any cost component can never raise NAV — a negative fee
    credit or sign flip violates this outright."""
    rng = np.random.default_rng(13)
    sids = ["A", "B", "C"]
    bars = _bars(sids, 25, rng, missing=0.05)
    weights = _weights(sids, 25, rng, sparse=0.1)

    def final_nav(cfg) -> float:
        run = (
            (lambda: run_backtest_fast(bars, weights, cfg))
            if engine == "fast"
            else (lambda: _run_backtest_event_loop(bars, weights, cfg))
        )
        equity = run().equity
        return float(equity.sort("event_time").get_column("nav")[-1])

    navs = [
        final_nav(_cfg(commission_bps=c, half_spread_bps=2.0, impact_y=0.1))
        for c in [0.0, 5.0, 20.0, 80.0]
    ]
    assert all(b <= a + 1e-9 for a, b in zip(navs[:-1], navs[1:], strict=True)), (
        f"NAV increased as commission rose: {navs}"
    )
