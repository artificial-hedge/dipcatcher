"""Tests for christensen_diebold_rudebusch — AFNS curve model."""

import numpy as np
import pytest

from quant_fund.models.christensen_diebold_rudebusch import (
    afns_adjustment,
    bench_christensen_diebold_rudebusch,
    fit_factors,
    fix_lam,
    ns_loadings,
    synth_afns,
)


def test_factor_recovery() -> None:
    y_adj, _, f_true, tau, lam = synth_afns(seed=1)
    f_hat = fit_factors(y_adj, tau, lam)
    for k in range(3):
        assert np.corrcoef(f_true[:, k], f_hat[:, k])[0, 1] > 0.9


def test_level_factor_scale() -> None:
    y_adj, _, f_true, tau, lam = synth_afns(seed=2)
    f_hat = fit_factors(y_adj, tau, lam)
    assert abs(f_hat[:, 0].mean() - f_true[:, 0].mean()) < 0.02


def test_ns_loading_shape() -> None:
    tau = np.array([1.0, 5.0, 10.0])
    x = ns_loadings(tau, 0.5)
    assert x.shape == (3, 3)
    assert np.all(x[:, 0] == 1.0)
    # NS1 decays to 0 at long maturity
    assert x[2, 1] < x[0, 1]


def test_fix_lam_curvature_peak() -> None:
    lam = fix_lam(4.0)
    assert 0.2 < lam < 0.8
    # NS2 at target should exceed NS2 at very short/long ends
    x = ns_loadings(np.array([0.25, 4.0, 20.0]), lam)
    assert x[1, 2] == max(x[:, 2])


def test_adjustment_nonnegative_small() -> None:
    tau = np.geomspace(0.25, 20.0, 10)
    a = afns_adjustment(tau, 0.42, 0.01, 0.01, 0.01)
    assert np.all(np.isfinite(a))
    assert np.all(a >= -1e-12)
    assert np.mean(np.abs(a)) < 0.05


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        ns_loadings(np.array([0.0, 1.0]), 0.5)
    with pytest.raises(ValueError):
        ns_loadings(np.array([1.0, 2.0]), -0.1)
    with pytest.raises(ValueError):
        fit_factors(np.full((10, 5), np.nan), np.arange(5) + 1.0, 0.5)
    with pytest.raises(ValueError):
        fix_lam(-1.0)


def test_determinism() -> None:
    y_adj, _, _, tau, lam = synth_afns(seed=7)
    a = fit_factors(y_adj, tau, lam)
    b = fit_factors(y_adj, tau, lam)
    np.testing.assert_array_equal(a, b)
    assert fix_lam(4.0) == fix_lam(4.0)


def test_bench_schema_and_score() -> None:
    r = bench_christensen_diebold_rudebusch()
    for k in (
        "factor_corr",
        "lam_hat",
        "lam_true",
        "factor_rmse",
        "yield_rmse",
        "adj_mag",
        "score",
    ):
        assert np.isfinite(r[k])
    assert r["score"] == 1.0
