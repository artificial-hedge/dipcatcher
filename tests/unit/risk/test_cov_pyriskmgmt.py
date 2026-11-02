"""Coverage: pyriskmgmt error and degenerate-regime branches."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.risk.pyriskmgmt import (
    component_var,
    ewma_var_es,
    ewma_variance,
    portfolio_returns,
    portfolio_var_es,
    scale_weights_to_es,
)


def _panel(rows: int = 60, cols: int = 3, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=(rows, cols))


def test_ewma_variance_rejects_short_or_invalid_lambda() -> None:
    assert np.isnan(ewma_variance(np.array([0.01])))
    for lam in (0.0, 1.0, float("nan"), -0.5):
        assert np.isnan(ewma_variance(np.full(10, 0.01), lam=lam))


def test_ewma_variance_is_positive_on_a_moving_series() -> None:
    value = ewma_variance(np.array([0.01, -0.02, 0.015, 0.0]))
    assert np.isfinite(value)
    assert value > 0.0


def test_ewma_var_es_short_series_returns_nan() -> None:
    var, es = ewma_var_es(np.zeros(7))
    assert np.isnan(var)
    assert np.isnan(es)


def test_ewma_var_es_falls_back_when_sigma_is_zero() -> None:
    var, es = ewma_var_es(np.zeros(30), alpha=0.95)
    assert np.isfinite(var)
    assert np.isfinite(es)


def test_ewma_var_es_normal_regime_is_finite() -> None:
    var, es = ewma_var_es(np.random.default_rng(3).normal(0.0, 0.01, 50), alpha=0.95)
    assert np.isfinite(var)
    assert np.isfinite(es)
    assert es >= var


def test_portfolio_returns_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match=r"asset_returns must be \(T, N\)"):
        portfolio_returns(np.zeros(10), np.ones(1))
    with pytest.raises(ValueError, match=r"asset_returns must be \(T, N\)"):
        portfolio_returns(np.zeros((10, 2)), np.ones(3))


def test_portfolio_returns_sanitizes_nonfinite_values() -> None:
    returns = np.array([[0.01, np.nan], [np.inf, 0.02], [-0.01, 0.0]])
    out = portfolio_returns(returns, np.array([1.0, np.nan]))
    assert out.shape == (3,)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("method", ["historical", "gaussian", "ewma", " HISTORICAL "])
def test_portfolio_var_es_supported_methods(method: str) -> None:
    blob = portfolio_var_es(_panel(), np.array([0.4, 0.3, 0.3]), method=method)
    assert blob["method"] == method.strip().lower()
    assert blob["research_only"] is True
    assert blob["execution_claim"] == "paper_backtest"
    assert blob["n"] == 60


def test_portfolio_var_es_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method must be historical, gaussian, or ewma"):
        portfolio_var_es(_panel(), np.array([0.4, 0.3, 0.3]), method="monte-carlo")


def test_component_var_rejects_bad_shapes() -> None:
    assert np.isnan(component_var(np.zeros(10), np.ones(1))).all()
    assert np.isnan(component_var(np.zeros((5, 2)), np.ones(2))).all()
    assert np.isnan(component_var(np.zeros((10, 2)), np.ones(3))).all()


def test_component_var_single_column_covariance_returns_nan() -> None:
    out = component_var(np.random.default_rng(0).normal(size=(30, 1)), np.ones(1))
    assert np.isnan(out).all()


def test_component_var_zero_variance_returns_nan() -> None:
    out = component_var(np.zeros((30, 2)), np.ones(2))
    assert np.isnan(out).all()


def test_component_var_finite_on_a_varying_panel() -> None:
    out = component_var(_panel(), np.array([0.5, 0.3, 0.2]))
    assert np.isfinite(out).all()
    assert out.shape == (3,)


def test_scale_weights_to_es_rejects_bad_limit() -> None:
    for limit in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="es_limit must be finite and positive"):
            scale_weights_to_es(_panel(), np.array([0.4, 0.3, 0.3]), es_limit=limit)


def test_scale_weights_to_es_unmeasurable_es_is_uncapped() -> None:
    zeros = np.zeros((60, 3))
    weights, blob = scale_weights_to_es(zeros, np.array([0.4, 0.3, 0.3]), es_limit=0.006)
    assert blob["scale"] == 1.0
    assert blob["capped"] is False
    assert np.allclose(weights, [0.4, 0.3, 0.3])


def test_scale_weights_to_es_caps_a_risky_book() -> None:
    weights, blob = scale_weights_to_es(
        _panel(), np.array([0.4, 0.3, 0.3]), es_limit=0.0001, alpha=0.95
    )
    assert blob["scale"] < 1.0
    assert blob["capped"] is True
    assert np.allclose(weights, np.array([0.4, 0.3, 0.3]) * blob["scale"])
