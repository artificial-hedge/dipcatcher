"""Adversarial probes for bvar_minnesota."""

import numpy as np
import pytest

from quant_fund.models import bvar_minnesota as bv


def test_dummies_reject_nonfinite_params():
    with pytest.raises(ValueError, match="lag_decay"):
        bv.minnesota_dummies(3, 1, lag_decay=np.nan)
    with pytest.raises(ValueError, match="lag_decay"):
        bv.minnesota_dummies(3, 1, lag_decay=np.inf)
    with pytest.raises(ValueError, match="mean_own"):
        bv.minnesota_dummies(3, 1, mean_own=np.inf)
    with pytest.raises(ValueError, match="intercept_tight"):
        bv.minnesota_dummies(3, 1, intercept_tight=0.0)
    with pytest.raises(ValueError, match="intercept_tight"):
        bv.minnesota_dummies(3, 1, intercept_tight=np.nan)


def test_dummy_rows_are_equation_scoped():
    yd, xd = bv.minnesota_dummies(3, 2, lambda_=0.2)
    k = 3
    # every lag-dummy row has nonzero Y in AT MOST one column
    lag_rows = yd[1:]
    nonzero_cols = (np.abs(lag_rows) > 0).sum(axis=1)
    assert np.all(nonzero_cols <= 1)
    # exactly k own-lag-1 rows carry the mean_own prior (one per equation,
    # at row 1 + i*k*p + 0*k + i), value mean_own/sd = 1.0/0.2
    p = 2
    for i in range(k):
        row = 1 + i * k * p + i
        assert yd[row, i] == pytest.approx(1.0 / 0.2)
        assert xd[row, 1 + i] == pytest.approx(1.0 / 0.2)
    assert (np.abs(lag_rows) > 0).sum() == k


def test_bvar_shrinks_cross_and_beats_ols():
    d = bv.synth_bvar(n=60, k=3, persistence=0.7, seed=42)
    y = d["y"]
    fit = bv.bvar_estimate(y, p=1, lambda_=0.15)
    B = np.asarray(fit["B"])
    a_bvar = B[1:4].T
    Y, X = bv._design(y, 1)
    bo, *_ = np.linalg.lstsq(X, Y, rcond=None)
    a_ols = bo[1:4].T
    own_err_b = np.abs(np.diag(a_bvar) - np.diag(d["A"])).mean()
    own_err_o = np.abs(np.diag(a_ols) - np.diag(d["A"])).mean()
    assert own_err_b < own_err_o


def test_bvar_rejects_short_sample():
    with pytest.raises(ValueError):
        bv.bvar_estimate(np.random.default_rng(0).normal(size=(3, 3)), p=1)


def test_shrinkage_profile_monotone_direction():
    d = bv.synth_bvar(n=80, k=3, persistence=0.8, seed=7)
    prof = bv.bvar_shrinkage_profile(d["y"], p=1, lambdas=(0.05, 1.0))
    own = np.asarray(prof["own_lag1"])
    # tighter prior -> own-lag-1 closer to mean_own=1.0
    assert np.all(np.abs(own[0] - 1.0) <= np.abs(own[1] - 1.0) + 1e-9)


def test_bench_smoke():
    out = bv.bench_bvar_minnesota()
    assert out["synthetic_determinism"] == 1.0
    assert out["synthetic_tightness_monotone"] == 1.0
