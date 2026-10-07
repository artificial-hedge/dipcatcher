"""ACE (Breiman-Friedman 1985) and AVAS (Tibshirani 1988)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ace_avas import ace, avas


def _nonlinear_fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (200, 2))
    y = np.exp(2 * x[:, 0]) * np.sin(4 * x[:, 1]) + rng.normal(0, 0.05, 200)
    return x, y


def test_ace_finds_transformed_correlation():
    x, y = _nonlinear_fixture()
    out = ace(x, y, max_iter=15)
    # transformed correlation must exceed linear R^2
    xa = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(xa, y, rcond=None)[0]
    lin_r2 = 1 - np.var(y - xa @ b) / np.var(y)
    assert out["r2"] > lin_r2
    assert out["r2"] > 0.5


def test_avas_stabilizes_variance():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.2, 1, 200)
    y = np.sqrt(x) * rng.normal(0, 1, 200)  # sd grows with x
    out = avas(x.reshape(-1, 1), y, max_iter=12)
    assert np.isfinite(out["phi"]).all()
    assert out["r2"] > 0.3


def test_ace_output_shapes():
    x, y = _nonlinear_fixture(3)
    out = ace(x, y)
    assert out["theta"].shape == y.shape
    assert out["phi"].shape == y.shape
    assert np.isfinite(out["theta"]).all()


def test_ace_linear_dgp_keeps_high_r2():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (150, 2))
    y = x[:, 0] + 2 * x[:, 1] + rng.normal(0, 0.1, 150)
    out = ace(x, y)
    assert out["r2"] > 0.9


def test_input_validation():
    with pytest.raises(ValueError):
        ace(np.ones((8, 2)), np.ones(4))
    with pytest.raises(ValueError):
        avas(np.full((30, 1), np.nan), np.ones(30))


def test_ace_vacuous_max_iter_raises():
    x, y = _nonlinear_fixture()
    with pytest.raises(ValueError):
        ace(x, y, max_iter=0)  # would return r2=-inf as a metric
    with pytest.raises(ValueError):
        avas(x, y, max_iter=0)


def test_ace_hostile_params():
    x, y = _nonlinear_fixture()
    with pytest.raises(ValueError):
        ace(x, y, tol=0.0)
    with pytest.raises(ValueError):
        ace(x, y, span=0.0)  # running_mean: span<=0
    with pytest.raises(ValueError):
        ace(x, y, span=np.inf)
    with pytest.raises(ValueError):
        avas(x, y, span=-0.5)


def test_ace_zero_feature_design():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 40)
    x0 = np.zeros((40, 0))  # (n,0) — no features → would div-by-zero r2
    with pytest.raises(ValueError):
        ace(x0, y)


def test_running_mean_guards():
    from quant_fund.models.ace_avas import _running_mean

    xs = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        _running_mean(xs[:0], xs[:0])
    with pytest.raises(ValueError):
        _running_mean(xs, xs[:5])
    with pytest.raises(ValueError):
        _running_mean(xs, xs, span=0.0)


def test_ace_deterministic():
    x, y = _nonlinear_fixture(7)
    a = ace(x, y, max_iter=5)
    b = ace(x, y, max_iter=5)
    assert a["r2"] == b["r2"]
    np.testing.assert_array_equal(a["phi"], b["phi"])


def test_running_mean_sorted_locality():
    from quant_fund.models.ace_avas import _running_mean

    xs = np.arange(10, dtype=np.float64)
    ys = xs**2  # locally increasing
    sm = _running_mean(xs, ys, span=0.3)
    # smoothed value near xs=0 must be below value near xs=9
    assert sm[0] < sm[-1]
    assert np.isfinite(sm).all()
