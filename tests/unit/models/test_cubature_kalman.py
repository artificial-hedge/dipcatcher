"""Tests for cubature_kalman — CKF vs EKF tracking bench."""

from __future__ import annotations

import numpy as np

from quant_fund.models.cubature_kalman import _ckf, _sim, bench_cubature_kalman


def test_cubature_cov_equal_weights() -> None:
    # CKF predicted covariance uses equal cubature weights 1/(2n); np.cov's
    # N-1 normalization inflated it by 4/3.
    from quant_fund.models.cubature_kalman import _cub_cov

    cub = np.array([[0.0], [2.0], [4.0], [6.0]])
    xm = cub.mean(0)
    # deviations^2 = 9,1,1,9 -> equal-weight cov = 20/4 = 5, not 20/3
    assert _cub_cov(cub, xm)[0, 0] == 5.0


def test_cub_cov_matches_weighted_sum() -> None:
    from quant_fund.models.cubature_kalman import _cub_cov

    rng = np.random.default_rng(3)
    cub = rng.normal(size=(8, 2))
    xm = cub.mean(0)
    dev = cub - xm
    expect = sum(np.outer(dev[i], dev[i]) for i in range(8)) / 8
    assert np.allclose(_cub_cov(cub, xm), expect)


def test_ckf_tracks() -> None:
    x, y = _sim(0)
    est = _ckf(y)
    assert est.shape == x.shape
    assert np.isfinite(est).all()


def test_bench_cubature_kalman() -> None:
    out = bench_cubature_kalman()
    for key, val in out.items():
        assert key.startswith("synthetic_"), key
        assert np.isfinite(val), key
    assert out["synthetic_ckf_rmse"] > 0
