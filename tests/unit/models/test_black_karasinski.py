"""Unit tests for quant_fund.models.black_karasinski."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.black_karasinski import (
    _trinomial_probs,
    bench_black_karasinski,
    bk_bond,
    bk_caplet,
    bk_tree,
)


def _flat_curve(rate: float = 0.05, n: int = 10):
    times = np.linspace(0.05, 1.0, n)
    return times, np.exp(-rate * times)


def test_trinomial_probs_sum_one() -> None:
    for j in (-3.0, -1.0, 0.0, 0.5, 2.0):
        pu, pm, pd, k = _trinomial_probs(j, a=0.1, dt=0.05)
        assert pu + pm + pd == pytest.approx(1.0)
        assert min(pu, pm, pd) >= 0.0


def test_trinomial_probs_moment_matching() -> None:
    # expected next position m = j*(1-a*dt) equals
    # pu*(k+1) + pm*k + pd*(k-1); variance ~ 2/3 in index units.
    j, a, dt = 1.3, 0.1, 0.05
    pu, pm, pd, k = _trinomial_probs(j, a, dt)
    mean_step = pu * (k + 1) + pm * k + pd * (k - 1)
    assert mean_step == pytest.approx(j * (1.0 - a * dt), abs=1e-12)
    var = pu * (k + 1) ** 2 + pm * k * k + pd * (k - 1) ** 2 - mean_step**2
    # index-unit variance = sigma^2 dt / dx^2 = 1/3 by construction.
    assert var == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_tree_reprices_flat_curve() -> None:
    times, df = _flat_curve(0.04)
    tree = bk_tree(a=0.1, sigma=0.01, curve_times=times, df=df, dt=0.05)
    for m in (4, 10, len(tree["ad"]) - 1):
        t = m * 0.05
        log_tgt = np.interp(t, np.concatenate([[0.0], times]), np.concatenate([[0.0], np.log(df)]))
        assert bk_bond(tree, m) == pytest.approx(float(np.exp(log_tgt)), rel=1e-6)


def test_bond_beyond_lattice_rejected() -> None:
    times, df = _flat_curve()
    tree = bk_tree(a=0.1, sigma=0.01, curve_times=times, df=df, dt=0.05)
    with pytest.raises(ValueError):
        bk_bond(tree, len(tree["ad"]) + 5)


def test_caplet_nonnegative_and_increasing_in_sigma() -> None:
    times, df = _flat_curve()
    # near-ATM strike: option value rises with short-rate vol.
    lo = bk_caplet(0.1, 0.005, times, df, expiry=0.9, strike=0.05, dt=0.05)
    hi = bk_caplet(0.1, 0.05, times, df, expiry=0.9, strike=0.05, dt=0.05)
    assert lo >= 0.0 and hi >= 0.0
    assert hi > lo


def test_rates_stay_positive() -> None:
    times, df = _flat_curve(0.03)
    tree = bk_tree(a=0.1, sigma=0.01, curve_times=times, df=df, dt=0.05)
    for r in tree["rates"]:
        assert np.all(np.asarray(r) > 0.0)


def test_bench_black_karasinski_score() -> None:
    out = bench_black_karasinski()
    assert out["score"] == pytest.approx(1.0)
