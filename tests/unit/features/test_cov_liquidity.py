"""features/liquidity.py — estimators and their fail-closed guards.

All series are SYNTHETIC; this pins the arithmetic contracts, not market edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.features import liquidity


def _returns(n: int = 60, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, n)
    returns[::7] = 0.0  # a dead zone so LOT/FHT have zero returns to count
    return returns


def _volumes(n: int = 60, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).uniform(1.0, 100.0, n)


class TestAmivest:
    def test_ratio_is_volume_over_abs_return(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, 0.0, 0.01, 0.02, -0.01, 0.0, 0.01, 0.04])
        volumes = np.ones(10)
        assert liquidity.amivest_ratio(returns, volumes) == pytest.approx(10.0 / 0.15)

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="match returns"):
            liquidity.amivest_ratio(_returns(), _volumes(59))

    def test_negative_volume_rejected(self) -> None:
        volumes = _volumes()
        volumes[3] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            liquidity.amivest_ratio(_returns(), volumes)

    def test_zero_total_absolute_return_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero total absolute return"):
            liquidity.amivest_ratio(np.zeros(20), np.ones(20))

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="length >= 10"):
            liquidity.amivest_ratio(np.zeros(5), np.ones(5))


class TestZeroFrequency:
    def test_counts_only_dead_zone_returns(self) -> None:
        returns = np.array([0.0] * 10 + [0.01] * 10)
        assert liquidity.zero_freq(returns) == pytest.approx(0.5)

    def test_eps_widens_the_dead_zone(self) -> None:
        returns = np.array([0.0] * 10 + [1e-6] * 10)
        assert liquidity.zero_freq(returns) == pytest.approx(0.5)
        assert liquidity.zero_freq(returns, eps=1e-5) == pytest.approx(1.0)

    def test_needs_twenty_observations(self) -> None:
        with pytest.raises(ValueError, match="length >= 20"):
            liquidity.zero_freq(np.zeros(19))


class TestLotSpread:
    def test_half_the_series_zero_gives_zero_spread(self) -> None:
        # z = 0.5 maps to the median, so the half-width is 0.
        out = liquidity.lot_spread(np.array([0.0] * 15 + [0.01, -0.01] * 8))
        assert out["zero_share"] == pytest.approx(15 / 31)
        assert out["sigma_nonzero"] > 0.0
        assert out["spread"] > 0.0

    def test_more_zeros_widen_the_spread(self) -> None:
        few = liquidity.lot_spread(np.array([0.0] * 8 + [0.01, -0.01] * 11))
        many = liquidity.lot_spread(np.array([0.0] * 20 + [0.01, -0.01] * 5))
        assert many["spread"] > few["spread"]

    def test_all_zero_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="no nonzero returns"):
            liquidity.lot_spread(np.zeros(30))

    def test_needs_thirty_observations(self) -> None:
        with pytest.raises(ValueError, match="length >= 30"):
            liquidity.lot_spread(np.zeros(29))


class TestFhtCost:
    def test_zero_share_and_sigma_reported(self) -> None:
        out = liquidity.fht_cost(np.array([0.0] * 20 + [0.01, -0.01] * 10))
        assert set(out) == {"fht", "zero_share", "sigma"}
        assert out["zero_share"] == pytest.approx(0.5)
        assert out["sigma"] > 0.0
        assert out["fht"] > 0.0

    def test_too_few_nonzero_returns_rejected(self) -> None:
        returns = np.zeros(30)
        returns[:9] = 0.01
        with pytest.raises(ValueError, match="too few nonzero returns"):
            liquidity.fht_cost(returns)

    def test_zero_variance_nonzero_returns_rejected(self) -> None:
        # 0.25 is exact in binary, so the nonzero population really has zero std.
        returns = np.zeros(30)
        returns[:10] = 0.25
        with pytest.raises(ValueError, match="zero nonzero-return variance"):
            liquidity.fht_cost(returns)


class TestPastorStambaugh:
    def test_order_flow_reversal_gives_negative_gamma(self) -> None:
        volumes = _volumes()
        base = np.zeros(60)
        base[0] = 0.01
        # Next-bar reversal scaled by signed volume: the illiquidity signature.
        for t in range(1, base.size):
            base[t] = -0.5 * np.sign(base[t - 1]) * volumes[t - 1]
        out = liquidity.pastor_stambaugh_gamma(base, volumes)
        assert set(out) == {"gamma", "t"}
        assert out["gamma"] < 0.0
        assert out["t"] < 0.0

    def test_mismatched_volumes_rejected(self) -> None:
        with pytest.raises(ValueError, match="match returns"):
            liquidity.pastor_stambaugh_gamma(_returns(), _volumes(59))

    def test_negative_volumes_rejected(self) -> None:
        volumes = _volumes()
        volumes[0] = -5.0
        with pytest.raises(ValueError, match="non-negative"):
            liquidity.pastor_stambaugh_gamma(_returns(), volumes)


class TestHasbrouckLambda:
    def test_recovers_a_linear_impact_coefficient(self) -> None:
        signed = np.linspace(-100.0, 100.0, 60)
        signed[::11] = 0.0
        changes = 0.02 * np.sign(signed) * np.sqrt(np.abs(signed))
        out = liquidity.hasbrouck_lambda(changes, signed)
        assert out["lambda"] == pytest.approx(0.02, rel=1e-6)
        assert np.isfinite(out["t"])

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="match price_changes"):
            liquidity.hasbrouck_lambda(np.zeros(30), np.zeros(31))


class TestGlostenHarris:
    def test_transitory_only_without_volumes(self) -> None:
        dp = _returns()
        signs = np.where(np.arange(dp.size) % 2 == 0, 1.0, -1.0)
        out = liquidity.glosten_harris(dp, signs)
        assert set(out) == {"transitory", "t_transitory"}

    def test_adverse_selection_component_with_volumes(self) -> None:
        dp = _returns()
        signs = np.where(np.arange(dp.size) % 2 == 0, 1.0, -1.0)
        out = liquidity.glosten_harris(dp, signs, _volumes())
        assert set(out) == {"transitory", "t_transitory", "adverse", "t_adverse", "spread_est"}
        assert out["spread_est"] == pytest.approx(2.0 * abs(out["transitory"]))

    def test_non_unit_signs_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\+/-1"):
            liquidity.glosten_harris(_returns(), np.zeros(60))

    def test_mismatched_volumes_rejected(self) -> None:
        signs = np.ones(60)
        with pytest.raises(ValueError, match="volumes must match"):
            liquidity.glosten_harris(_returns(), signs, np.ones(59))


class TestEffectiveTick:
    def test_integer_grid_prices_pick_a_one_dollar_tick(self) -> None:
        out = liquidity.effective_tick(np.arange(20.0, 80.0))
        assert out["probs"].shape == (6,)
        assert np.all(out["probs"] == pytest.approx(1.0))
        assert float(out["tick"][0]) == pytest.approx(1.0)

    def test_quarter_grid_prices_pick_a_quarter_tick(self) -> None:
        prices = np.arange(0, 60) / 4.0 + 20.0  # 20.00, 20.25, 20.50, ...
        out = liquidity.effective_tick(prices)
        assert out["probs"][-1] == pytest.approx(1.0)
        # 0.25 from the $1 grid, 0.125 from 50c, 0.125 from 25c, 0.025 from 5c.
        assert float(out["tick"][0]) == pytest.approx(0.525)

    def test_non_positive_price_rejected(self) -> None:
        prices = np.linspace(1.0, 2.0, 30)
        prices[4] = 0.0
        with pytest.raises(ValueError, match="prices must be positive"):
            liquidity.effective_tick(prices)

    def test_short_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="length >= 30"):
            liquidity.effective_tick(np.ones(29))


class TestTurnoverVolatility:
    def test_constant_volume_has_no_dispersion(self) -> None:
        out = liquidity.turnover_volatility(np.full(20, 7.0))
        assert out["turnover_mean"] == pytest.approx(7.0)
        assert out["turnover_cv"] == pytest.approx(0.0)
        assert out["turnover_vol"] == pytest.approx(0.0)

    def test_zero_volumes_are_clipped_in_the_log(self) -> None:
        volumes = np.array([0.0] * 10 + [10.0] * 10)
        out = liquidity.turnover_volatility(volumes)
        assert np.isfinite(out["turnover_vol"])

    def test_negative_volume_rejected(self) -> None:
        volumes = np.ones(20)
        volumes[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            liquidity.turnover_volatility(volumes)

    def test_zero_mean_volume_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero mean volume"):
            liquidity.turnover_volatility(np.zeros(20))
