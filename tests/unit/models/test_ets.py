"""Adversarial probes for ets."""

import numpy as np
import pytest

from quant_fund.models import ets


def _y(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    return 5.0 + 0.1 * np.arange(n) + rng.normal(0.0, 0.5, n)


def _seasonal(n: int = 60, period: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10.0 + 0.2 * t + 2.0 * np.sin(2 * np.pi * t / period) + rng.normal(0.0, 0.3, n)


def test_ses_rejects_alpha_out_of_range():
    y = _y()
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ets.ses_fit(y, alpha=2.0)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ets.ses_fit(y, alpha=-0.1)


def test_forecast_rejects_fractional_h():
    fit = ets.ses_fit(_y())
    with pytest.raises(ValueError, match="integer"):
        ets.ets_forecast(fit, h=2.5)


def test_hw_rejects_fractional_period():
    with pytest.raises(ValueError, match="integer"):
        ets.holt_winters_fit(_seasonal(), period=2.5)


def test_hw_forecast_shape_and_damped():
    fit = ets.holt_winters_fit(_seasonal(period=4), period=4, seasonal="add")
    fc = ets.ets_forecast(fit, h=8)
    assert fc.shape == (8,)
    assert np.isfinite(fc).all()


def test_holt_damped_forecast_plateau():
    """phi<1 must produce sublinear trend extrapolation vs phi=1."""
    y = 5.0 + np.arange(80) * 0.4 + np.random.default_rng(1).normal(0, 0.2, 80)
    fd = ets.ets_forecast(ets.holt_fit(y, damped=True), 10)
    fl = ets.ets_forecast(ets.holt_fit(y, damped=False), 10)
    assert fd[-1] < fl[-1]


def test_ses_flat_forecast():
    fit = ets.ses_fit(_y())
    fc = ets.ets_forecast(fit, 5)
    assert np.all(fc == fit.level)
