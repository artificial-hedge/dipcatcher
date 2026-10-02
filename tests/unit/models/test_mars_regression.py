"""MARS (Friedman 1991) multivariate adaptive regression splines."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mars_regression import mars_fit, mars_predict


def _fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (200, 3))
    y = (
        4 * np.maximum(x[:, 0] - 0.5, 0)
        - 3 * np.maximum(0.3 - x[:, 1], 0)
        + 0.5 * x[:, 2]
        + rng.normal(0, 0.05, 200)
    )
    return x, y


def test_mars_beats_linear_on_hinge_dgp():
    x, y = _fixture()
    model = mars_fit(x, y, max_terms=15)
    pred = mars_predict(model, x)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    # linear benchmark
    a = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    lin = np.column_stack([np.ones(len(x)), x]) @ a
    lin_rmse = float(np.sqrt(np.mean((lin - y) ** 2)))
    assert rmse < 0.6 * lin_rmse


def test_mars_recovers_additive_linear_dgp():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, (150, 2))
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(0, 0.02, 150)
    model = mars_fit(x, y, max_terms=10)
    pred = mars_predict(model, x)
    r2 = 1 - np.var(pred - y) / np.var(y)
    assert r2 > 0.95


def test_mars_predict_shape_and_finiteness():
    x, y = _fixture(2)
    model = mars_fit(x, y, max_terms=12)
    rng = np.random.default_rng(9)
    xnew = rng.uniform(0, 1, (17, 3))
    pred = mars_predict(model, xnew)
    assert pred.shape == (17,)
    assert np.isfinite(pred).all()


def test_input_validation():
    with pytest.raises(ValueError):
        mars_fit(np.ones((10, 2)), np.ones(5))
    with pytest.raises(ValueError):
        mars_fit(np.full((30, 2), np.nan), np.ones(30))
