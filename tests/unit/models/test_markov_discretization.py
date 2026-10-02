"""Markov discretization tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.markov_discretization import (
    bench_markov,
    implied_moments,
    rouwenhorst,
    tauchen,
    tauchen_hussey,
)


def test_rows_sum_one():
    for fn in (tauchen, tauchen_hussey, rouwenhorst):
        _, p = fn(0.9, 0.1, 0.0, 7)
        assert np.allclose(p.sum(1), 1.0)
        assert (p >= 0).all()


def test_rouwenhorst_exact():
    z, p = rouwenhorst(0.95, 0.2, 0.0, 9)
    mom = implied_moments(z, p)
    assert mom["rho"] == pytest.approx(0.95, abs=1e-10)
    assert mom["sd"] == pytest.approx(0.2 / np.sqrt(1 - 0.95**2), abs=1e-10)


def test_tauchen_grid_symmetric():
    z, _ = tauchen(0.8, 0.1, 0.0, 7)
    assert z[0] == pytest.approx(-z[-1])
    assert abs(z[len(z) // 2]) < 1e-12


def test_stationary_mean():
    z, p = tauchen(0.85, 0.1, 0.5, 9)
    mom = implied_moments(z, p)
    assert mom["mean"] == pytest.approx(0.5, abs=0.05)


def test_high_persistence_rw_beats_tauchen():
    # rho=0.99: RW exact; Tauchen has grid-coarseness error
    z_r, p_r = rouwenhorst(0.99, 0.1, 0.0, 9)
    z_t, p_t = tauchen(0.99, 0.1, 0.0, 9)
    e_r = abs(implied_moments(z_r, p_r)["rho"] - 0.99)
    e_t = abs(implied_moments(z_t, p_t)["rho"] - 0.99)
    assert e_r < e_t


def test_bench_markov():
    out = bench_markov()
    assert out["synthetic_rw_rho_err"] < 1e-8
