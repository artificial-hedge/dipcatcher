"""OLS influence diagnostics tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.influence_diagnostics import (
    bench_influence,
    influence_flags,
    ols_influence,
)


def _design(seed: int = 0, n: int = 50):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2))
    X = np.column_stack([np.ones(n), x])
    y = 0.5 + x @ np.array([1.0, -0.7]) + rng.normal(0, 0.5, n)
    return X, y


def test_hat_sums_to_p():
    X, y = _design()
    inf = ols_influence(X, y)
    assert abs(np.asarray(inf["hat"]).sum() - 3.0) < 1e-9
    assert (np.asarray(inf["hat"]) >= 0).all()


def test_outlier_gets_large_cook():
    X, y = _design()
    y2 = y.copy()
    X2 = X.copy()
    y2[0] = 30.0
    X2[0, 1] = 10.0
    inf = ols_influence(X2, y2)
    cook = np.asarray(inf["cook"])
    assert int(np.argmax(cook)) == 0
    assert cook[0] > 10 * np.median(cook)


def test_studentized_residual_flags():
    X, y = _design()
    y[1] = 15.0
    flags = influence_flags(X, y)
    assert flags["rstudent"][1] == 1.0


def test_dfbetas_shape():
    X, y = _design()
    inf = ols_influence(X, y)
    assert np.asarray(inf["dfbetas"]).shape == (50, 3)


def test_clean_fit_few_flags():
    X, y = _design(1)
    flags = influence_flags(X, y)
    assert flags["cook"].sum() <= 3
    assert flags["cook"].sum() < len(y) * 0.1


def test_input_validation():
    with pytest.raises(ValueError):
        ols_influence(np.ones((5, 4)), np.ones(5))
    with pytest.raises(ValueError):
        influence_flags(np.ones((10, 2)), np.ones(9))


def test_bench_passes():
    out = bench_influence()
    assert out["synthetic_flags_planted"] == 1.0
    assert out["synthetic_max_cook_at_planted"] == 1.0
    assert out["synthetic_clean_flag_sum"] <= 4
    assert out["synthetic_score"] == 1.0
