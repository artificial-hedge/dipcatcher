"""Avellaneda-Stoikov market making tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.avellaneda_stoikov import (
    bench_avellaneda,
    optimal_quotes,
    optimal_spread,
    reservation_price,
    simulate_mm,
)


def test_reservation_price_skew():
    # long inventory lowers reservation price
    assert reservation_price(100, 5, 10, 0.1, 0.02) < 100
    assert reservation_price(100, -5, 10, 0.1, 0.02) > 100
    assert reservation_price(100, 0, 10, 0.1, 0.02) == 100


def test_optimal_spread_positive():
    s = optimal_spread(10.0, 0.1, 0.02, 1.5)
    assert s > 0
    # spread widens with sigma and horizon
    assert optimal_spread(10.0, 0.1, 0.05, 1.5) > s


def test_quotes_straddle_reservation():
    b, a = optimal_quotes(100.0, 0.0, 10.0, 0.1, 0.02, 1.5)
    r = reservation_price(100.0, 0.0, 10.0, 0.1, 0.02)
    assert b < r < a


def test_simulation_runs():
    rng = np.random.default_rng(0)
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 500)))
    out = simulate_mm(mid, seed=0)
    assert np.isfinite(out["pnl_total"])
    assert out["max_abs_inv"] >= 0


def test_bench_avellaneda():
    out = bench_avellaneda()
    assert out["synthetic_reservation_skew"] > 0
