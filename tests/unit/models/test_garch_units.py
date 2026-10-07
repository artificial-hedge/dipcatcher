"""GARCHVol units contract: variance stays variance, volatility stays volatility.

``forecast(...)['variance']`` is a variance (sigma^2) in decimal-squared units
and ``['sigma']`` is the documented square-root convenience in decimal units.
A variance in produces a variance out and a volatility in produces a volatility
out with NO silent rescale; the only sanctioned conversion is the explicit
``rescale_units``.

SYNTHETIC inputs only — correctness tests, not market evidence.  No
performance or profitability claim is made here.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.vol_eval import (
    VARIANCE_UNITS,
    VOLATILITY_UNITS,
    VolUnitsError,
    coerce_units,
    rescale_units,
)
from quant_fund.models.volatility import GARCHVol

SEED = 20261007


def _returns(n: int = 400, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(rng.normal(0.0, 0.01, size=n))


def test_garch_forecast_dict_declares_both_units_explicitly() -> None:
    model = GARCHVol()
    model.fit_returns(_returns())
    out = model.forecast(horizon=3)
    assert out["variance_units"] == VARIANCE_UNITS
    assert out["sigma_units"] == VOLATILITY_UNITS
    assert VARIANCE_UNITS == "decimal_squared"
    assert VOLATILITY_UNITS == "decimal"


def test_garch_variance_is_sigma_squared_and_not_sigma() -> None:
    model = GARCHVol()
    model.fit_returns(_returns())
    out = model.forecast(horizon=5)
    variance = np.asarray(out["variance"])
    sigma = np.asarray(out["sigma"])
    assert np.allclose(variance, sigma**2)
    # A variance of decimal returns is far below its own sigma in scale;
    # if variance silently held sigma, this inequality would flip.
    assert np.all(variance < sigma)


def test_variance_in_produces_variance_out_with_no_silent_rescale() -> None:
    variance = np.array([4e-4, 1e-4, 2.5e-4])
    coerced = coerce_units(variance, from_units=VARIANCE_UNITS, to_units=VARIANCE_UNITS)
    assert coerced.tobytes() == variance.tobytes()


def test_volatility_in_produces_volatility_out_with_no_silent_rescale() -> None:
    sigma = np.array([0.02, 0.01, 0.015])
    coerced = coerce_units(sigma, from_units=VOLATILITY_UNITS, to_units=VOLATILITY_UNITS)
    assert coerced.tobytes() == sigma.tobytes()
    with pytest.raises(VolUnitsError):
        coerce_units(sigma, from_units=VOLATILITY_UNITS, to_units=VARIANCE_UNITS)
    with pytest.raises(VolUnitsError):
        coerce_units(sigma, from_units=VARIANCE_UNITS, to_units=VOLATILITY_UNITS)


def test_explicit_rescale_round_trips_and_is_the_only_conversion() -> None:
    variance = np.array([4e-4, 1e-4, 2.5e-4])
    sigma = rescale_units(variance, from_units=VARIANCE_UNITS, to_units=VOLATILITY_UNITS)
    assert np.allclose(sigma, np.sqrt(variance))
    back = rescale_units(sigma, from_units=VOLATILITY_UNITS, to_units=VARIANCE_UNITS)
    assert np.allclose(back, variance)
    same = rescale_units(variance, from_units=VARIANCE_UNITS, to_units=VARIANCE_UNITS)
    assert same.tobytes() == variance.tobytes()
