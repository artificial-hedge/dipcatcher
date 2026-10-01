"""Unit tests for quant_fund.models.libor_market."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.libor_market import (
    bench_libor_market,
    caplet_black,
    discount_curve,
    simulate_forwards,
)


def test_discount_curve_monotone() -> None:
    l0 = np.full(4, 0.03)
    delta = np.full(4, 0.5)
    df = discount_curve(l0, delta)
    assert np.all(np.diff(df) < 0.0)
    assert df[0] == pytest.approx(1.0)
    assert df[-1] == pytest.approx(1.0 / 1.015**4)


def test_caplet_black_atm() -> None:
    v = caplet_black(0.03, 0.03, 0.2, 1.0, 0.5, 0.97)
    # ATM Black caplet ~ delta*df*L*(2 Phi(sd/2)-1)
    approx = 0.5 * 0.97 * 0.03 * (2 * 0.5398 - 1)
    assert v == pytest.approx(approx, rel=0.05)


def test_forward_martingale() -> None:
    # Under the terminal measure the last forward is a martingale.
    n = 4
    l0 = np.full(n, 0.04)
    sigma = np.full(n, 0.2)
    delta = np.full(n, 0.5)
    rho = 0.5 * np.ones((n, n)) + 0.5 * np.eye(n)
    path = simulate_forwards(l0, sigma, delta, rho, np.array([0.5]), 8000, 7)
    last_mean = float(np.mean(path[:, -1, -1]))
    assert last_mean == pytest.approx(0.04, abs=0.003)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        caplet_black(0.03, -0.01, 0.2, 1.0, 0.5, 0.97)
    with pytest.raises(ValueError):
        simulate_forwards(
            np.array([0.03]),
            np.array([0.2]),
            np.array([0.5]),
            np.eye(1),
            np.array([1.0]),
            10,
            0,
        )
    with pytest.raises(ValueError):
        discount_curve(np.array([-0.01]), np.array([0.5]))


def test_bench_score() -> None:
    out = bench_libor_market()
    assert out["score"] == 1.0
    assert out["synthetic_lmm_caplet_err"] < 0.05
