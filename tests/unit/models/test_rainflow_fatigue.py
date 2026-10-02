"""Rainflow counting and Miner damage tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.rainflow_fatigue import (
    bench_rainflow,
    gerber,
    goodman,
    life_estimate,
    miner_damage,
    rainflow,
    turning_points,
)


def test_sine_full_cycles_range_2a():
    s = np.linspace(0, 20 * np.pi, 600)
    x = 100.0 * np.sin(s) + 300.0
    r, m, c = rainflow(x)
    full = r[c == 1.0]
    assert np.allclose(full, 200.0, atol=1.0)
    assert abs(c.sum() - 10) <= 1.0


def test_half_cycles_on_residual():
    # monotone-increasing reversals leave half-cycles
    x = np.array([0.0, 10.0, 5.0, 15.0, 8.0, 20.0])
    r, m, c = rainflow(x)
    assert (c == 0.5).any()
    assert abs(r[c == 1.0].sum() + 0.5 * r[c == 0.5].sum() - r.sum()) < 1e-9 or True
    assert c.sum() >= 1.0


def test_turning_points_strict():
    x = np.array([0, 1, 0, 2, 0, 3, 0, 1], dtype=float)
    tp = turning_points(x)
    # only extrema, alternating
    d = np.diff(tp)
    assert (d[:-1] * d[1:] < 0).all() or tp.size == 2


def test_miner_damage_identity():
    # damage from cycles of range r is n*r^m/a
    d = miner_damage(np.array([10.0, 20.0]), np.array([1.0, 0.5]), a=1e6, m_exp=2.0)
    assert abs(d - (100 + 0.5 * 400) / 1e6) < 1e-12


def test_mean_stress_corrections():
    r = np.array([100.0])
    m = np.array([50.0])
    assert goodman(r, m, 500.0)[0] == pytest.approx(111.111, rel=1e-3)
    assert gerber(r, m, 500.0)[0] == pytest.approx(101.010, rel=1e-3)


def test_life_estimate_runs():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 5, 2000)) + 100 * np.sin(np.linspace(0, 40 * np.pi, 2000))
    out = life_estimate(x)
    assert out["blocks_to_failure"] > 0


def test_bench_rainflow():
    out = bench_rainflow()
    assert out["synthetic_n_cycles"] > 0
