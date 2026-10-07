"""Units preservation for GARCH-family variance forecasts.

``forecast(...)["variance"]`` is a VARIANCE (sigma^2) in decimal-squared return
units — never a volatility — and a variance input to the units helpers comes
back out as the same variance, bit-identical, with NO silent rescale.  The only
sanctioned variance <-> volatility conversion is the explicit
``rescale_units``.  Seeded SYNTHETIC returns only; correctness tests, never
market evidence or a performance claim.
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


def _returns(n: int = 200, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)  # SYNTHETIC
    return rng.normal(0.0, 0.01, n)


def test_garch_forecast_dict_declares_both_units_explicitly() -> None:
    forecast = GARCHVol().fit_returns(_returns()).forecast(horizon=4)

    assert forecast["variance_units"] == VARIANCE_UNITS == "decimal_squared"
    assert forecast["sigma_units"] == VOLATILITY_UNITS == "decimal"


def test_garch_variance_is_sigma_squared_and_not_sigma() -> None:
    forecast = GARCHVol().fit_returns(_returns()).forecast(horizon=4)

    assert np.allclose(forecast["sigma"] ** 2, forecast["variance"])
    assert np.all(forecast["variance"] < forecast["sigma"])  # |sigma| < 1 scale
    assert not np.allclose(forecast["variance"], forecast["sigma"])


def test_variance_in_produces_variance_out_with_no_silent_rescale() -> None:
    forecast = GARCHVol().fit_returns(_returns()).forecast(horizon=4)
    variance = forecast["variance"]

    out = coerce_units(variance, from_units=forecast["variance_units"], to_units=VARIANCE_UNITS)
    assert out.tobytes() == variance.tobytes()
    with pytest.raises(VolUnitsError, match="never silently rescaled"):
        coerce_units(variance, from_units=VARIANCE_UNITS, to_units=VOLATILITY_UNITS)


def test_volatility_in_produces_volatility_out_with_no_silent_rescale() -> None:
    forecast = GARCHVol().fit_returns(_returns()).forecast(horizon=4)
    sigma = forecast["sigma"]

    out = coerce_units(sigma, from_units=forecast["sigma_units"], to_units=VOLATILITY_UNITS)
    assert out.tobytes() == sigma.tobytes()
    with pytest.raises(VolUnitsError, match="never silently rescaled"):
        coerce_units(sigma, from_units=VOLATILITY_UNITS, to_units=VARIANCE_UNITS)


def test_explicit_rescale_round_trips_and_is_the_only_conversion() -> None:
    forecast = GARCHVol().fit_returns(_returns()).forecast(horizon=3)
    variance = forecast["variance"]

    sigma = rescale_units(variance, from_units=VARIANCE_UNITS, to_units=VOLATILITY_UNITS)
    back = rescale_units(sigma, from_units=VOLATILITY_UNITS, to_units=VARIANCE_UNITS)

    assert np.allclose(sigma, forecast["sigma"])
    assert np.allclose(back, variance)
