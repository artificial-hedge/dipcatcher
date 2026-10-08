"""Unit tests for quant_fund.models.esscher."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.esscher import (
    bench_esscher,
    esscher_call,
    esscher_tail_prob,
    solve_martingale_theta,
)


def _gauss_cf(u, t, sigma=0.2):
    u = np.asarray(u)
    return np.exp(1j * u * (-0.5 * sigma * sigma * t) - 0.5 * sigma * sigma * t * u * u)


def _gauss_kcgf(u, sigma=0.2):
    u = np.asarray(u)
    return -0.5 * sigma * sigma * u + 0.5 * sigma * sigma * u * u


def test_gaussian_tail_matches() -> None:
    cf = lambda u, t: _gauss_cf(u, t)  # noqa: E731
    kcgf = lambda u: _gauss_kcgf(u)  # noqa: E731
    p = esscher_tail_prob(cf, kcgf, 0.5, 0.0, 0.5)
    assert 0.0 < p < 1.0


def test_martingale_theta_root() -> None:
    cf = lambda u, t: _gauss_cf(u, t)  # noqa: E731
    kcgf = lambda u: _gauss_kcgf(u)  # noqa: E731
    th = solve_martingale_theta(cf, kcgf, 0.05, 0.5)
    # For GBM the Esscher martingale theta solves K(1+th)-K(th)=r:
    # sigma^2 * th = r -> th = r / sigma^2
    assert th == pytest.approx(0.05 / 0.04, abs=1e-6)


def test_esscher_recovers_bs() -> None:
    cf = lambda u, t: _gauss_cf(u, t)  # noqa: E731
    kcgf = lambda u: _gauss_kcgf(u)  # noqa: E731
    s0, k, t, r = 100.0, 110.0, 1.0, 0.05
    th = solve_martingale_theta(cf, kcgf, r, t)
    v = esscher_call(s0, k, t, r, cf, kcgf, th)
    sd = 0.2 * np.sqrt(t)
    d1 = (np.log(s0 / k) + (r + 0.5 * 0.04) * t) / sd
    d2 = d1 - sd
    from scipy.stats import norm

    bs = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    assert v == pytest.approx(bs, rel=0.01)


def test_input_validation() -> None:
    cf = lambda u, t: np.ones_like(np.asarray(u))  # noqa: E731
    kcgf = lambda u: np.asarray(u)  # noqa: E731
    with pytest.raises(ValueError):
        esscher_call(0.0, 100.0, 1.0, 0.05, cf, kcgf, 0.5)


def test_bench_score() -> None:
    out = bench_esscher()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_es_bs_err"] < 0.01
