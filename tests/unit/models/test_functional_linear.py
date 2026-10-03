"""Tests for functional data analysis (models/functional_linear.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.functional_linear import (
    bench_functional_linear,
    bspline_basis,
    fpca,
    functional_lm,
    synth_flm,
    synth_fpca,
)


@pytest.fixture
def curves():
    return synth_fpca(n=60, t=40, seed=2)


@pytest.fixture
def flm():
    return synth_flm(n=80, t=40, seed=3)


def test_bspline_basis_shape():
    b = bspline_basis(np.linspace(0, 1, 50), n_basis=10)
    assert b.shape == (50, 10)
    # B-spline partition of unity
    assert np.allclose(b.sum(axis=1), 1.0, atol=1e-6)


def test_bspline_zero_before_knots():
    grid = np.linspace(0, 1, 30)
    b = bspline_basis(grid, n_basis=8)
    assert np.all(np.isfinite(b))
    assert b.min() >= 0.0


def test_fpca_recovers_eigenfunctions(curves):
    out = fpca(curves["X"], n_components=2, n_basis=10)
    eigs = np.asarray(out["eigenfunctions"])

    def cong(a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

    assert max(cong(curves["e1"], eigs[:, j]) for j in range(2)) > 0.9
    assert out["eigenvalues"][0] > out["eigenvalues"][1]


def test_fpca_reconstruction(curves):
    out = fpca(curves["X"], n_components=2, n_basis=10)
    assert float(out["rel_err"]) < 0.4
    assert out["fitted"].shape == curves["X"].shape


def test_fpca_determinism(curves):
    a = fpca(curves["X"], n_components=2, n_basis=10)["rel_err"]
    b = fpca(curves["X"], n_components=2, n_basis=10)["rel_err"]
    assert a == b


def test_functional_lm_recovers_beta(flm):
    fit = functional_lm(flm["Y"], flm["x"], n_basis=10, smooth_lambda=1e-2)
    beta = np.asarray(fit["beta"])
    relerr = np.linalg.norm(beta - flm["beta_true"]) / np.linalg.norm(flm["beta_true"])
    assert relerr < 0.2
    assert float(fit["r2"]) > 0.7


def test_functional_lm_shape(flm):
    fit = functional_lm(flm["Y"], flm["x"], n_basis=10)
    assert np.asarray(fit["beta"]).shape == (40,)
    assert np.asarray(fit["fitted"]).shape == flm["Y"].shape


def test_validation():
    with pytest.raises(ValueError):
        fpca(np.zeros((5, 4)))
    with pytest.raises(ValueError):
        functional_lm(np.zeros((20, 15)), np.zeros(19))
    with pytest.raises(ValueError):
        functional_lm(np.zeros((20, 15)), np.zeros(20))  # zero-var x
    with pytest.raises(ValueError):
        bspline_basis(np.linspace(0, 1, 5), n_basis=20)


def test_bench_keys():
    out = bench_functional_linear()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_fpca_relerr"] < 0.3
    assert out["synthetic_eig1_congruence"] > 0.9
    assert out["synthetic_flm_r2"] > 0.8
    assert out["synthetic_beta_relerr"] < 0.1
    assert out["synthetic_determinism"] == 1.0
