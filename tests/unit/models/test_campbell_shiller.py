"""Unit tests for quant_fund.models.campbell_shiller."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.campbell_shiller import (
    bench_campbell_shiller,
    var_decompose,
)


def _dgp(n: int = 5000, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = np.zeros(n)
    mu = np.zeros(n)
    for t in range(1, n):
        mu[t] = 0.9 * mu[t - 1] + 0.02 * rng.standard_normal()
        g[t] = 0.02 + mu[t] + 0.10 * rng.standard_normal()
    rho = 0.96
    pd_t = mu * rho / (1.0 - 0.9 * rho) + 0.05 * rng.standard_normal(n)
    r = np.zeros(n)
    r[1:] = -pd_t[:-1] + rho * pd_t[1:] + g[1:]
    return np.column_stack([r[1:], g[1:], pd_t[1:]])


def test_shares_sum_to_one() -> None:
    out = var_decompose(_dgp(), rho=0.96)
    s = out["share_div"] + out["share_disc"] + out["share_cov"]
    assert s == pytest.approx(1.0, abs=1e-6)


def test_dividend_share_dominates_dgp() -> None:
    out = var_decompose(_dgp(), rho=0.96)
    assert 0.3 < out["share_div"] < 0.98
    assert 0.0 < out["share_disc"] < 0.7


def test_stable_var() -> None:
    out = var_decompose(_dgp(), rho=0.96)
    assert out["spec_rad"] < 1.0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        var_decompose(np.zeros((10, 3)))
    with pytest.raises(ValueError):
        var_decompose(np.zeros((100, 2)))
    with pytest.raises(ValueError):
        var_decompose(np.zeros((100, 3)), rho=1.5)


def test_bench_score() -> None:
    out = bench_campbell_shiller()
    assert out["synthetic_score"] == 1.0
