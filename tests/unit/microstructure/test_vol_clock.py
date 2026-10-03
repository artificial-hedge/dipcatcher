"""Tests for microstructure/vol_clock.py — transaction-time volatility."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.vol_clock import (
    VOL_CLOCK_SCHEMA,
    acf_abs,
    collect_tape,
    ljung_box,
    vol_clock_bench,
)
from quant_fund.microstructure.zi_lob_simulator import ZILobConfig


def test_acf_abs_known_signal() -> None:
    # alternating |r| pattern → strong lag-1 autocorrelation
    ret = np.tile([0.01, -0.01], 500)
    a = acf_abs(ret, 3)
    assert a.shape == (3,)
    # constant |r| → zero autocorrelation
    assert np.allclose(acf_abs(np.ones(100), 3), 0.0)


def test_acf_abs_recovers_planted_clustering() -> None:
    rng = np.random.default_rng(0)
    # |r| clusters: blocks of high vol vs low vol
    vol = np.repeat([0.02, 0.001], 200)
    ret = rng.normal(0, vol)
    a = acf_abs(ret, 5)
    assert a[0] > 0.3  # strong positive lag-1


def test_ljung_box_known_values() -> None:
    # Q = n(n+2) Σ ρ²/(n−k): with rho=[0.5], n=100 → 100*102*0.25/99
    assert ljung_box(np.asarray([0.5]), 100) == pytest.approx(25.7575, abs=1e-3)


def test_collect_tape_deterministic() -> None:
    cfg = ZILobConfig(seed=7, init_depth=8, band=8)
    a = collect_tape(config=cfg, horizon=200.0)
    b = collect_tape(config=cfg, horizon=200.0)
    assert a.n_mo == b.n_mo
    np.testing.assert_array_equal(a.event_rets, b.event_rets)
    np.testing.assert_array_equal(a.calendar_rets, b.calendar_rets)


def test_collect_tape_populates_both_clocks() -> None:
    t = collect_tape(config=ZILobConfig(seed=3, init_depth=8, band=8), horizon=300.0)
    assert t.event_rets.size > 0 and t.calendar_rets.size > 0
    assert t.n_mo > 0


def test_bench_smoke_and_schema() -> None:
    out = vol_clock_bench(n_seeds=2, horizon=200.0)
    assert out["schema"] == VOL_CLOCK_SCHEMA
    for arm in ("calm", "drift"):
        assert "lb_event_mean" in out["arms"][arm]
        assert "lb_calendar_mean" in out["arms"][arm]
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["payload_sha256"]) == 64
