"""Coverage tests for the Almgren-Chriss sizing helpers' validation paths.

Pure numpy, no solver and no IO.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.execution.almgren_chriss import (
    almgren_chriss_trajectory,
    expected_shortfall_ac,
    front_loaded_trajectory,
    slice_trades,
    twap_trajectory,
)

_KWARGS = {"sigma": 1.0, "eta": 1.0, "gamma": 0.0, "risk_aversion": 0.0}


class TestTrajectoryValidation:
    @pytest.mark.parametrize("qty", [math.nan, math.inf, -math.inf])
    def test_rejects_nonfinite_quantity(self, qty: float) -> None:
        with pytest.raises(ValueError, match="quantity must be finite"):
            almgren_chriss_trajectory(qty, 2, **_KWARGS)

    @pytest.mark.parametrize("slices", [0, -1, 2.5, True])
    def test_rejects_bad_slice_count(self, slices: object) -> None:
        with pytest.raises(ValueError, match="n_slices"):
            almgren_chriss_trajectory(10.0, slices, **_KWARGS)  # type: ignore[arg-type]

    @pytest.mark.parametrize("tau", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_bad_tau(self, tau: float) -> None:
        with pytest.raises(ValueError, match="tau"):
            almgren_chriss_trajectory(10.0, 2, tau=tau, **_KWARGS)

    @pytest.mark.parametrize("field", ["sigma", "eta", "gamma", "risk_aversion"])
    def test_rejects_nonfinite_impact_inputs(self, field: str) -> None:
        kwargs = dict(_KWARGS)
        kwargs[field] = math.nan
        with pytest.raises(ValueError, match="must be finite"):
            almgren_chriss_trajectory(10.0, 2, **kwargs)

    @pytest.mark.parametrize(("sigma", "eta"), [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
    def test_rejects_nonpositive_impact(self, sigma: float, eta: float) -> None:
        with pytest.raises(ValueError, match="eta and sigma must be > 0"):
            almgren_chriss_trajectory(10.0, 2, sigma=sigma, eta=eta, gamma=0.0, risk_aversion=0.0)

    def test_rejects_negative_risk_aversion(self) -> None:
        with pytest.raises(ValueError, match="risk_aversion must be >= 0"):
            almgren_chriss_trajectory(10.0, 2, sigma=1.0, eta=1.0, gamma=0.0, risk_aversion=-0.1)


class TestTrajectoryBehaviour:
    def test_flat_quantity_returns_flat_holdings(self) -> None:
        path = almgren_chriss_trajectory(0.0, 3, **_KWARGS)
        assert path.shape == (4,)
        assert np.all(path == 0.0)

    def test_zero_risk_aversion_is_twap(self) -> None:
        path = almgren_chriss_trajectory(10.0, 4, **_KWARGS)
        assert path[0] == pytest.approx(10.0)
        assert path[-1] == 0.0
        assert np.allclose(path, np.linspace(10.0, 0.0, 5))

    @pytest.mark.parametrize("n_slices", [1, 3, 10])
    def test_shape_endpoints_and_monotonicity(self, n_slices: int) -> None:
        path = almgren_chriss_trajectory(
            12.0, n_slices, sigma=0.02, eta=1e-6, gamma=0.0, risk_aversion=1e-2
        )
        assert path.shape == (n_slices + 1,)
        assert path[0] == pytest.approx(12.0)
        assert path[-1] == 0.0
        assert np.all(np.diff(path) <= 1e-12)

    def test_large_kappa_stays_finite(self) -> None:
        # Very high risk aversion / negligible eta: the naive sinh form
        # overflows here; the stable exponential form must not.
        path = almgren_chriss_trajectory(
            1.0, 500, sigma=100.0, eta=1e-12, gamma=0.0, risk_aversion=1e6
        )
        assert np.all(np.isfinite(path))
        assert path[-1] == 0.0
        assert path[1] == pytest.approx(0.0, abs=1e-6)

    def test_negative_quantity_mirrors_long_path(self) -> None:
        kwargs = {"sigma": 0.02, "eta": 1e-6, "gamma": 0.0, "risk_aversion": 1e-2}
        short = almgren_chriss_trajectory(-5.0, 4, **kwargs)
        long = almgren_chriss_trajectory(5.0, 4, **kwargs)
        assert np.allclose(short, -long)

    def test_slice_trades_sums_to_quantity(self) -> None:
        path = twap_trajectory(6.0, 3)
        trades = slice_trades(path)
        assert trades.shape == (3,)
        assert trades.sum() == pytest.approx(6.0)

    def test_front_loaded_starts_faster_than_twap(self) -> None:
        twap = twap_trajectory(10.0, 5)
        front = front_loaded_trajectory(10.0, 5)
        assert front[1] < twap[1]


class TestExpectedShortfall:
    def test_single_slice_diagonal_costs(self) -> None:
        result = expected_shortfall_ac(
            np.array([10.0, 0.0]),
            np.array([10.0]),
            arrival=100.0,
            eta=2.0,
            gamma=1.0,
            sigma=0.5,
            tau=1.0,
        )
        # permanent: gamma * sum(tr * (cumsum(tr) - tr/2)) = 1 * 10 * 5
        assert result["expected_is"] == pytest.approx(50.0 + 2.0 * 100.0)
        assert result["expected_cost"] == result["expected_is"]
        assert result["variance_is"] == pytest.approx(0.5**2 * 1.0 * 0.0)

    def test_flat_book_with_no_trades_costs_nothing(self) -> None:
        result = expected_shortfall_ac(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0]),
            arrival=50.0,
            eta=1.0,
            gamma=1.0,
            sigma=0.1,
        )
        assert result["expected_is"] == 0.0
        assert result["variance_is"] == 0.0

    def test_variance_uses_remaining_holdings(self) -> None:
        holdings = np.array([4.0, 2.0, 1.0])
        trades = np.array([2.0, 1.0])
        result = expected_shortfall_ac(
            holdings, trades, arrival=10.0, eta=0.0, gamma=0.0, sigma=2.0, tau=0.5
        )
        # var = sigma^2 * tau * sum(holdings[1:]**2)
        assert result["variance_is"] == pytest.approx(2.0**2 * 0.5 * (2.0**2 + 1.0**2))

    def test_shape_and_input_validation(self) -> None:
        with pytest.raises(ValueError, match="holdings length"):
            expected_shortfall_ac(
                np.array([1.0]), np.array([]), arrival=1.0, eta=1.0, gamma=1.0, sigma=1.0
            )
        with pytest.raises(ValueError, match="trades length"):
            expected_shortfall_ac(
                np.array([1.0, 0.0]),
                np.array([1.0, 1.0]),
                arrival=1.0,
                eta=1.0,
                gamma=1.0,
                sigma=1.0,
            )

    @pytest.mark.parametrize("tau", [0.0, -1.0, math.nan])
    def test_rejects_bad_tau(self, tau: float) -> None:
        with pytest.raises(ValueError, match="tau"):
            expected_shortfall_ac(
                np.array([1.0, 0.0]),
                np.array([1.0]),
                arrival=1.0,
                eta=1.0,
                gamma=1.0,
                sigma=1.0,
                tau=tau,
            )

    @pytest.mark.parametrize("field", ["eta", "gamma", "sigma", "arrival"])
    def test_rejects_nonfinite_scalars(self, field: str) -> None:
        kwargs = {"arrival": 1.0, "eta": 1.0, "gamma": 1.0, "sigma": 1.0}
        kwargs[field] = math.inf
        with pytest.raises(ValueError, match="must be finite"):
            expected_shortfall_ac(np.array([1.0, 0.0]), np.array([1.0]), **kwargs)
