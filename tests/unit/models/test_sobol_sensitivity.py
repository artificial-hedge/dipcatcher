"""Tests for Sobol indices and Morris elementary effects."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.sobol_sensitivity import (
    bench_sobol,
    morris_effects,
    sobol_indices,
)


def _additive(u: np.ndarray) -> np.ndarray:
    return 3.0 * u[:, 0] + u[:, 1]  # x0 dominant


def test_sobol_additive_model():
    res = sobol_indices(_additive, d=2, n=2048, seed=0)
    s1 = res["s1"]
    # var shares: 9*Var(u0) vs Var(u1) -> S0 ~ 0.9
    assert s1[0] > 0.8
    assert s1[1] < 0.15
    assert res["st"][0] > s1[0] - 0.01


def test_sobol_pure_interaction():
    def xor(u: np.ndarray) -> np.ndarray:
        return (u[:, 0] > 0.5).astype(float) * (u[:, 1] > 0.5).astype(float) * (u[:, 1] * 4 - 1.0)

    res = sobol_indices(xor, d=2, n=4096, seed=1)
    assert res["st"][0] > res["s1"][0]


def test_morris_identifies_influential():
    res = morris_effects(_additive, d=2, n_traj=16, seed=2)
    assert res["mu_star"].argmax() == 0
    assert res["sigma"].shape == (2,)


def test_morris_screening_negligible_dim():
    res = morris_effects(_additive, d=3, n_traj=24, seed=3)
    # dim 2 does not appear in _additive -> near-zero mu*
    assert res["mu_star"][2] < 0.05 * res["mu_star"][0]


def test_fail_closed():
    with pytest.raises(ValueError):
        sobol_indices(_additive, d=0, n=128)
    with pytest.raises(ValueError):
        sobol_indices(_additive, d=2, n=8)
    with pytest.raises(ValueError):
        sobol_indices(lambda u: np.zeros(u.shape[0]), d=2, n=128)
    with pytest.raises(ValueError):
        morris_effects(_additive, d=2, n_traj=2)


def test_bench():
    res = bench_sobol()
    assert res["synthetic_score"] == 1.0
    assert res["synthetic_s1_err"] < 0.1
    assert res["synthetic_s3_err"] < 0.1
