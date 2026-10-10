import numpy as np
import pytest

from quant_fund.combination.variance_ensemble import (
    dm_test,
    ewma_variance,
    long_run_variance,
    qlike_losses,
    rolling_mad_variance,
)

pytestmark = pytest.mark.synthetic


def _garch(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    r = np.empty(n)
    var = 0.05
    for t in range(n):
        r[t] = np.sqrt(var) * e[t]
        var = 0.00002 + 0.08 * r[t] ** 2 + 0.9 * var
    return r


def test_ewma_tracks_garch_better_than_flat() -> None:
    r = _garch(6000, seed=80)
    v_ewma = ewma_variance(r)
    v_flat = long_run_variance(r)
    losses_ewma = qlike_losses(r, v_ewma)
    losses_flat = qlike_losses(r, v_flat)
    # mean QLIKE comparison on valid tail
    assert np.nanmean(losses_ewma[50:]) < np.nanmean(losses_flat[50:])


def test_rolling_mad_members_finite_and_positive() -> None:
    r = _garch(2000, seed=81)
    v = rolling_mad_variance(r, window=30)
    assert np.all(np.isfinite(v[30:]))
    assert np.all(v[30:] > 0)


def test_qlike_losses_mask_invalid() -> None:
    r = np.array([0.1, np.nan, 0.2])
    v = np.array([0.01, 0.01, -1.0])
    out = qlike_losses(r, v)
    assert np.isfinite(out[0])
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_dm_test_detects_better_member() -> None:
    rng = np.random.default_rng(82)
    n = 3000
    # member A losses ~ N(1, 1); member B ~ N(1.1, 1) → B worse
    la = rng.normal(1.0, 1.0, n)
    lb = rng.normal(1.1, 1.0, n)
    out = dm_test(la, lb)
    assert out["diff"] < 0
    assert out["p"] < 0.05


def test_dm_test_accepts_equal_accuracy() -> None:
    rng = np.random.default_rng(83)
    n = 3000
    la = rng.normal(1.0, 1.0, n)
    lb = rng.normal(1.0, 1.0, n)
    out = dm_test(la, lb)
    assert out["p"] > 0.05


def test_dm_test_hac_positive_autocorr() -> None:
    rng = np.random.default_rng(84)
    n = 3000
    d = np.empty(n)
    e = rng.standard_normal(n)
    d[0] = e[0]
    for t in range(1, n):
        d[t] = 0.7 * d[t - 1] + e[t] + 0.2
    la = d
    lb = np.zeros(n)
    out = dm_test(la, lb, h=10)
    assert out["p"] < 0.05
