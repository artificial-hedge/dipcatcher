import numpy as np
import pytest

from quant_fund.models.kriging import (
    bench_kriging,
    empirical_variogram,
    exponential_variogram,
    fit_variogram,
    ordinary_kriging,
)


def _field(seed=0, n=100):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    y = rng.uniform(0, 10, n)
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    cov = np.exp(-d / 3.0) + 1e-9 * np.eye(n)
    z = rng.multivariate_normal(np.zeros(n), cov)
    return x, y, z


def test_exponential_variogram_limits():
    h = np.linspace(0, 30, 50)
    g = exponential_variogram(h, 9.0, 1.0)
    assert g[0] == pytest.approx(0.0, abs=1e-12)
    assert g[-1] == pytest.approx(1.0, abs=0.05)
    assert np.all(np.diff(g) > 0)


def test_empirical_variogram_bins():
    x, y, z = _field()
    lag, gam = empirical_variogram(x, y, z)
    assert lag.shape == gam.shape
    assert lag.size >= 5
    assert np.all(gam >= 0)


def test_fit_variogram_recovers_range():
    x, y, z = _field(1, n=200)
    lag, gam = empirical_variogram(x, y, z)
    sill, rng_a = fit_variogram(lag, gam)
    assert sill > 0.3
    assert 1.0 < rng_a < 30.0


def test_kriging_exact_at_knots():
    x, y, z = _field(2, n=60)
    out = ordinary_kriging(x, y, z, x[:3], y[:3], rng_a=9.0, sill=1.0)
    pred = np.asarray(out["mean"])
    assert np.abs(pred - z[:3]).max() < 0.05


def test_kriging_shapes_variance():
    x, y, z = _field(3, n=60)
    out = ordinary_kriging(x, y, z, np.array([5.0, 6.0]), np.array([5.0, 6.0]), 9.0, 1.0)
    assert out["mean"].shape == (2,)
    assert np.all(np.asarray(out["variance"]) >= 0)


def test_kriging_input_validation():
    with pytest.raises(ValueError):
        ordinary_kriging(np.ones(3), np.ones(3), np.ones(3), np.ones(1), np.ones(1), 1.0, 1.0)


def test_bench_kriging():
    out = bench_kriging()
    assert out["synthetic_score"] == 1.0
    assert out["synthetic_kriging_rmse_ratio"] < 0.8
