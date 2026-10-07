"""Tests for partially-linear regression (models/partial_linear.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.partial_linear import (
    bench_partial_linear,
    cv_bandwidth,
    kernel_smooth,
    robinson_pl,
    series_pl,
    speckman_pl,
    synth_partial_linear,
)


@pytest.fixture
def panel():
    return synth_partial_linear(n=300, corr_xz=0.6, seed=4)


def test_kernel_smooth_recovery(panel):
    z = np.asarray(panel["z"])
    g = np.asarray(panel["g_true"])
    out = kernel_smooth(g, z, bw=0.05)
    assert np.corrcoef(out, g)[0, 1] > 0.95


def test_robinson_beta(panel):
    fit = robinson_pl(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))
    beta = np.asarray(fit["beta"])
    assert abs(beta[0] - 1.5) < 0.15
    se = np.asarray(fit["se"])
    assert se[0] > 0.0


def test_speckman_beta(panel):
    sp = speckman_pl(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))
    assert abs(float(np.asarray(sp["beta"])[0]) - 1.5) < 0.15


def test_series_beta(panel):
    se = series_pl(
        np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]), n_basis=9
    )
    assert abs(float(np.asarray(se["beta"])[0]) - 1.5) < 0.15


def test_cv_bandwidth(panel):
    cv = cv_bandwidth(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))
    bw = float(cv["bw"])
    assert 0.001 < bw < 1.0
    assert np.asarray(cv["mse"]).size == np.asarray(cv["grid"]).size


def test_validation():
    with pytest.raises(ValueError):
        robinson_pl(np.ones(10), np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        robinson_pl(np.ones(40), np.ones(40), np.ones(30))
    with pytest.raises(ValueError):
        kernel_smooth(np.ones(30), np.ones(30), bw=0.0)
    with pytest.raises(ValueError):
        series_pl(np.ones(40), np.ones(40), np.ones(40), n_basis=0)


def test_determinism(panel):
    a = robinson_pl(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))["beta"]
    b = robinson_pl(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))["beta"]
    assert np.allclose(a, b)


def test_bench_keys():
    out = bench_partial_linear()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_beta_err"] < 0.2
    assert out["synthetic_semi_beats_ols"] == 1.0
    assert out["synthetic_g_corr"] > 0.9
    assert out["synthetic_determinism"] == 1.0


def test_robinson_se_matches_manual_sandwich() -> None:
    """EHW se must equal bread^-1 meat bread^-1 (no extra /n or bread^-1)."""
    d = __import__("quant_fund.models.partial_linear", fromlist=["x"]).synth_partial_linear(
        n=300, seed=7
    )
    y, x, z = np.asarray(d["y"]), np.asarray(d["x"]), np.asarray(d["z"])
    out = robinson_pl(y, x, z, bw=0.15, bw_cv=False)
    # independent recomputation
    bw = 0.15
    m_y = kernel_smooth(y, z, bw)
    m_x = np.column_stack([kernel_smooth(x[:, j], z, bw) for j in range(x.shape[1])])
    r_x, resid = x - m_x, (y - m_y) - (x - m_x) @ np.asarray(out["beta"])
    bread = r_x.T @ r_x
    meat = r_x.T @ (r_x * resid[:, None] ** 2)
    expected = np.sqrt(np.diag(np.linalg.inv(bread) @ meat @ np.linalg.inv(bread)))
    assert np.allclose(np.asarray(out["se"]), expected, rtol=1e-8)
    # and not orders of magnitude off the classical scale
    assert np.asarray(out["se"])[0] > 1e-4


def test_cv_is_truly_leave_one_out() -> None:
    """A tiny bandwidth must not win CV — self-fit is excluded."""
    rng = np.random.default_rng(3)
    n = 200
    z = rng.uniform(0, 1, n)
    x = rng.normal(size=(n, 1))
    y = np.sin(4 * np.pi * z) + 0.4 * rng.standard_normal(n) + x[:, 0]
    grid = np.array([0.001, 0.05, 0.2, 0.5])
    out = cv_bandwidth(y, x, z, grid=grid)
    assert out["bw"] != grid[0]  # the undersmoothing endpoint cannot win real LOO CV


def test_kernel_smooth_rejects_nan_bw() -> None:
    with pytest.raises(ValueError, match="bw"):
        kernel_smooth(np.ones(5), np.arange(5.0), bw=float("nan"))


def test_series_pl_rejects_bool_n_basis() -> None:
    d = np.random.default_rng(1)
    n = 40
    y, x, z = d.normal(size=n), d.normal(size=(n, 1)), d.uniform(0, 1, n)
    with pytest.raises(ValueError, match="n_basis"):
        series_pl(y, x, z, n_basis=True)
    with pytest.raises(ValueError, match="n_basis"):
        series_pl(y, x, z, n_basis=2.5)
