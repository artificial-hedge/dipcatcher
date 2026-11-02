"""Coverage: factor/selection/timing decomposition (approximation-labelled)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from quant_fund.portfolio.attribution import factor_selection_timing


def test_decomposition_reports_component_means() -> None:
    portfolio = np.array([0.010, 0.020, -0.005])
    market = np.array([0.004, 0.006, -0.002])
    residual = np.array([0.005, 0.012, -0.004])
    cost = np.array([0.001, 0.002, -0.001])

    out = factor_selection_timing(portfolio, market, residual, cost)

    assert out["market"] == pytest.approx(float(np.mean(market)))
    assert out["selection_residual"] == pytest.approx(float(np.mean(residual)))
    assert out["cost"] == pytest.approx(float(np.mean(cost)))
    assert out["total"] == pytest.approx(float(np.mean(portfolio)))
    assert out["convention"] == 1.0


def test_decomposition_skips_nan_observations() -> None:
    portfolio = np.array([0.01, np.nan, 0.03])
    out = factor_selection_timing(
        portfolio,
        np.array([np.nan, 0.01, 0.02]),
        np.array([0.0, np.nan, 0.01]),
        np.array([0.0, 0.0, np.nan]),
    )
    assert out["market"] == pytest.approx(0.015)
    assert out["selection_residual"] == pytest.approx(0.005)
    assert out["cost"] == pytest.approx(0.0)


def test_decomposition_preserves_nan_when_nothing_is_usable() -> None:
    nan = np.array([np.nan])
    # nanmean of an all-NaN slice warns and returns NaN; that is the contract.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = factor_selection_timing(nan, nan, nan, nan)
    assert np.isnan(out["market"])
    assert np.isnan(out["total"])
