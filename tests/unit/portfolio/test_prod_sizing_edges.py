from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.config.models import AppConfig
from quant_fund.portfolio.allocators import inverse_volatility, volatility_target
from quant_fund.portfolio.executable_alpha import executable_alpha, should_trade
from quant_fund.portfolio.interval_risk import apply_interval_caps, interval_refs
from quant_fund.portfolio.optimizer import component_risk, optimize_mean_variance
from quant_fund.portfolio.research_allocators import volatility_managed_weights


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_interval_caps_zero_nonfinite_interval_endpoints(bad: float) -> None:
    capped, caps = apply_interval_caps(
        np.array([0.01, -0.01]),
        np.array([bad, -0.02]),
        np.array([0.02, bad]),
        max_weight=0.02,
        width_ref=0.10,
        downside_ref=0.05,
    )

    np.testing.assert_array_equal(caps, np.zeros(2))
    np.testing.assert_array_equal(capped, np.zeros(2))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_interval_caps_zero_nonfinite_weights(bad: float) -> None:
    capped, caps = apply_interval_caps(
        np.array([bad]),
        np.array([-0.02]),
        np.array([0.02]),
        max_weight=0.02,
        width_ref=0.10,
        downside_ref=0.05,
    )

    assert capped[0] == 0.0
    assert np.isfinite(caps[0])


def test_interval_refs_fall_back_when_no_valid_calibration_intervals() -> None:
    width_ref, downside_ref = interval_refs(np.array([math.nan, 0.2]), np.array([0.1, 0.1]))

    assert width_ref > 0.0
    assert downside_ref > 0.0
    assert math.isfinite(width_ref)
    assert math.isfinite(downside_ref)


@pytest.mark.parametrize(
    ("alpha", "sigma", "previous", "message"),
    [
        (np.array([]), np.zeros((0, 0)), np.array([]), "non-empty"),
        (np.array([0.01]), np.eye(2), np.array([0.0]), "sigma"),
        (np.array([math.nan]), np.eye(1) * 0.04, np.array([0.0]), "finite"),
        (np.array([0.01]), np.eye(1) * 0.04, np.array([math.inf]), "w_prev"),
    ],
)
def test_optimizer_rejects_invalid_shapes_and_nonfinite_inputs(
    alpha: np.ndarray, sigma: np.ndarray, previous: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        optimize_mean_variance(alpha, sigma, previous, AppConfig())


def test_component_risk_rejects_nonfinite_covariance() -> None:
    with pytest.raises(ValueError, match="finite"):
        component_risk(np.array([0.5, 0.5]), np.array([[0.04, 0.0], [0.0, math.inf]]))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_inverse_volatility_rejects_nonfinite_covariance(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        inverse_volatility(np.array([[0.04, 0.0], [0.0, bad]]))


@pytest.mark.parametrize("bad", [0.0, -0.1, math.nan, math.inf])
def test_volatility_target_rejects_invalid_target(bad: float) -> None:
    with pytest.raises(ValueError, match="target_vol"):
        volatility_target(np.array([0.5, 0.5]), np.eye(2) * 0.04, bad)


def test_volatility_managed_weights_reject_nonfinite_returns() -> None:
    with pytest.raises(ValueError, match="finite"):
        volatility_managed_weights(np.array([0.01, math.nan, -0.01]), training_scale=0.1, window=2)


def test_volatility_managed_weights_are_zero_during_warmup() -> None:
    weights = volatility_managed_weights(
        np.array([0.01, -0.02, 0.03]), training_scale=0.1, window=2
    )

    np.testing.assert_array_equal(weights[:2], np.zeros(2))
    assert np.isfinite(weights).all()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_executable_alpha_rejects_nonfinite_costs(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        executable_alpha(0.01, bad, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_trade_decision_fails_closed_on_nonfinite_threshold(bad: float) -> None:
    assert should_trade(0.01, bad) is False
    assert should_trade(bad, 0.0) is False
