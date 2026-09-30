"""Tests for microstructure/abc_calibrate.py."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.abc_calibrate import (
    ABC_SCHEMA,
    Draw,
    abc_calibrate_bench,
    abc_reject,
    distance,
    propose,
)


def test_propose_in_prior() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = propose(rng)
        assert 0.01 <= d.lam <= 1.0
        assert 0.005 <= d.mu <= 0.5
        assert 3 <= d.band <= 12
        assert 0.0 <= d.density_exponent <= 2.0


def test_distance_zero_and_inf() -> None:
    t = {"sign_lag1": 0.5, "mo_fraction": 0.1, "spread_ticks_median": 2.0, "mid_move_std": 0.3}
    assert distance(t, t) == pytest.approx(0.0)
    bad = dict(t, sign_lag1=float("nan"))
    assert distance(bad, t) == float("inf")


def test_abc_reject_fail_closed() -> None:
    t = {"sign_lag1": 0.1, "mo_fraction": 0.1, "spread_ticks_median": 2.0, "mid_move_std": 0.3}
    with pytest.raises(ValueError, match="n_draws"):
        abc_reject(t, n_draws=2, keep=8)


def test_abc_recovers_easy_target() -> None:
    # Target reachable by the sim class: low sign persistence.
    t = {"sign_lag1": 0.0, "mo_fraction": 0.15, "spread_ticks_median": 2.0, "mid_move_std": 0.3}
    out = abc_reject(t, n_draws=6, keep=2, horizon=400, seed=1)
    assert out["n_draws"] == 6
    assert len(out["accepted"]) == 2
    assert out["min_distance"] <= out["median_draw_distance"]


def test_bench_schema() -> None:
    t = {"sign_lag1": 0.1, "mo_fraction": 0.1, "spread_ticks_median": 2.0, "mid_move_std": 0.3}
    import quant_fund.microstructure.abc_calibrate as m

    orig = m.abc_reject

    def fast(target, *, n_draws=64, keep=8, horizon=3000, seed=0):
        return orig(target, n_draws=4, keep=2, horizon=200, seed=seed)

    import quant_fund.microstructure.sim_real_ledger as srl

    orig_ms = m.measure_sim
    m.abc_reject = fast
    m.measure_sim = lambda cfg, flow, *, horizon: orig_ms(cfg, flow, horizon=200)
    try:
        out = abc_calibrate_bench(t, n_draws=4, keep=2, horizon=200, seed=0)
    finally:
        m.abc_reject = orig
        m.measure_sim = orig_ms
        _ = srl  # silence
    assert out["schema"] == ABC_SCHEMA
    assert "receipt_sha256" in out


def test_draw_config_roundtrip() -> None:
    d = Draw(0.1, 0.1, 0.02, 5, 0.5, True, 0.1, 5, 2.0)
    cfg = d.config(0)
    assert cfg.lam == 0.1
    assert d.flow(0) is not None
    d2 = Draw(0.1, 0.1, 0.02, 5, 0.5, False, 0.1, 5, 2.0)
    assert d2.flow(0) is None
