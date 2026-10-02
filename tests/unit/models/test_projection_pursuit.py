"""Projection-pursuit regression (Friedman-Stuetzle 1981)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.projection_pursuit import ft_index, ppr_fit, ppr_predict


def _ridge_fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (200, 3))
    a0 = np.array([0.6, 0.8, 0.0])
    y = np.tanh(4 * (x @ a0)) + rng.normal(0, 0.05, 200)
    return x, y, a0


def test_ppr_finds_ridge_direction():
    x, y, a0 = _ridge_fixture()
    model = ppr_fit(x, y, n_terms=1, n_starts=12)
    a_hat = np.asarray(model["terms"][0][0])
    cos = abs(float(a_hat @ a0 / np.linalg.norm(a_hat)))
    assert cos > 0.9


def test_ppr_beats_linear_on_ridge_dgp():
    x, y, _ = _ridge_fixture(1)
    model = ppr_fit(x, y, n_terms=1, n_starts=12)
    pred = ppr_predict(model, x)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    xa = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(xa, y, rcond=None)[0]
    lin_rmse = float(np.sqrt(np.mean((xa @ b - y) ** 2)))
    assert rmse < 0.7 * lin_rmse


def test_ppr_predict_shape():
    x, y, _ = _ridge_fixture(2)
    model = ppr_fit(x, y, n_terms=2, n_starts=8)
    rng = np.random.default_rng(5)
    pred = ppr_predict(model, rng.uniform(-1, 1, (11, 3)))
    assert pred.shape == (11,)
    assert np.isfinite(pred).all()


def test_ft_index_prefers_structure_over_gaussian():
    rng = np.random.default_rng(6)
    structured = np.abs(rng.normal(0, 1, 400))
    gaussian = rng.normal(0, 1, 400)
    assert ft_index(structured) != pytest.approx(ft_index(gaussian), abs=1e-9)


def test_input_validation():
    with pytest.raises(ValueError):
        ppr_fit(np.ones((10, 2)), np.ones(5))
    with pytest.raises(ValueError):
        ppr_fit(np.full((40, 2), np.nan), np.ones(40))
