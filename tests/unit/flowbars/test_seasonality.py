import numpy as np
import pytest

from quant_fund.flowbars.seasonality import (
    deseasonalize_returns,
    seasonal_strength,
    seasonality_curve,
)

pytestmark = pytest.mark.synthetic

N_MIN = 390


def _u_shaped(n_days: int, seed: int):
    rng = np.random.default_rng(seed)
    base = 1.0 + 3.0 * np.exp(-np.linspace(0, 4, N_MIN)) + 3.0 * np.exp(-np.linspace(4, 0, N_MIN))
    base = base / base.mean()
    minute = np.tile(np.arange(N_MIN), n_days)
    vol = np.sqrt(base[minute]) * 0.01
    returns = rng.standard_normal(len(minute)) * vol
    return returns, minute, base


def test_curve_recovers_u_shape() -> None:
    returns, minute, base = _u_shaped(60, seed=1)
    curve = seasonality_curve(returns, minute, n_minutes=N_MIN, bandwidth=4)
    assert curve.shape == (N_MIN,)
    assert float(np.mean(curve)) == pytest.approx(1.0, abs=0.05)
    # opening and closing minutes carry the largest factors
    assert curve[0] > 1.4
    assert curve[-1] > 1.4
    assert np.mean(curve[180:210]) < curve[0]


def test_deseasonalize_preserves_scale() -> None:
    returns, minute, _ = _u_shaped(60, seed=2)
    curve = seasonality_curve(returns, minute, n_minutes=N_MIN, bandwidth=4)
    adjusted = deseasonalize_returns(returns, minute, curve)
    # total variance preserved by construction of the normalised curve
    assert float(np.var(adjusted)) == pytest.approx(float(np.var(returns)), rel=0.05)


def test_seasonal_strength_beats_flat_curve() -> None:
    returns, minute, _ = _u_shaped(60, seed=3)
    curve = seasonality_curve(returns, minute, n_minutes=N_MIN, bandwidth=4)
    flat = np.ones(N_MIN)
    s = seasonal_strength(returns, minute, curve)
    assert 0.0 <= s <= 1.0
    assert s > seasonal_strength(returns, minute, flat)


def test_flat_data_gives_flat_curve() -> None:
    rng = np.random.default_rng(4)
    minute = np.tile(np.arange(N_MIN), 20)
    returns = rng.standard_normal(len(minute)) * 0.01
    curve = seasonality_curve(returns, minute, n_minutes=N_MIN, bandwidth=4)
    assert float(np.std(curve)) < 0.15


def test_validation() -> None:
    with pytest.raises(ValueError):
        seasonality_curve(np.zeros(10), np.full(10, 999))
    with pytest.raises(ValueError):
        deseasonalize_returns(np.zeros(5), np.full(5, 5, dtype=np.int64), np.ones(3))
    with pytest.raises(ValueError):
        seasonal_strength(np.zeros(5), np.full(5, 5, dtype=np.int64), np.ones(2))
