import numpy as np
import pytest

from quant_fund.decay.half_life import (
    ar1_half_life,
    classify_decay,
    fit_exponential_decay,
)

pytestmark = pytest.mark.synthetic


def test_exponential_decay_recovery() -> None:
    hl_true = np.log(2.0) / -np.log(0.7)  # ≈ 1.94
    lags = np.arange(1, 11, dtype=np.float64)
    ics = 0.3 * 0.7**lags
    out = fit_exponential_decay(lags, ics)
    assert out["half_life"] == pytest.approx(hl_true, rel=0.05)
    assert out["r2"] > 0.99


def test_nondecaying_series_gives_infinite_half_life() -> None:
    lags = np.arange(1, 6, dtype=np.float64)
    out = fit_exponential_decay(lags, np.full(5, 0.2))
    assert out["half_life"] == float("inf")


def test_zero_values_are_dropped() -> None:
    lags = np.array([1.0, 2.0, 3.0, 4.0])
    ics = np.array([0.3, 0.0, 0.15, 0.075])
    out = fit_exponential_decay(lags, ics)
    assert np.isfinite(out["half_life"])
    assert out["n"] == 3.0


def test_ar1_half_life_recovers_phi() -> None:
    rng = np.random.default_rng(40)
    phi = 0.9
    x = np.empty(6000)
    x[0] = 0.0
    e = rng.standard_normal(6000)
    for i in range(1, 6000):
        x[i] = phi * x[i - 1] + e[i]
    out = ar1_half_life(x)
    assert out["phi"] == pytest.approx(phi, abs=0.03)
    assert out["half_life"] == pytest.approx(-np.log(2.0) / np.log(phi), rel=0.05)


def test_ar1_unit_root_gives_very_long_half_life() -> None:
    rng = np.random.default_rng(41)
    x = np.cumsum(rng.standard_normal(2000))
    out = ar1_half_life(x)
    # OLS phi-hat lands just below 1 on a random walk; the implied half-life
    # is very long but finite — the documented behaviour of this estimator
    assert out["phi"] > 0.99
    assert out["half_life"] > 50.0


def test_classify_decay_buckets() -> None:
    assert classify_decay(1.0) == "fast"
    assert classify_decay(3.0) == "medium"
    assert classify_decay(8.0) == "slow"
    assert classify_decay(float("nan")) == "unknown"


def test_fit_needs_three_points() -> None:
    with pytest.raises(ValueError):
        fit_exponential_decay(np.array([1.0, 2.0]), np.array([0.1, 0.05]))
