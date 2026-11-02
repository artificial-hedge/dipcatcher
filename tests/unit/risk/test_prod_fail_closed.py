from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.risk.gates import GateSpec, apply_gate_stack, dd_remaining_leverage, es_leverage
from quant_fund.risk.overlay import BookRiskOverlay
from quant_fund.risk.pyrisk import ExpectedShortfall, ValueAtRisk
from quant_fund.risk.tail_quota import tail_quota_eprocess


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, math.nan, math.inf])
def test_var_and_es_reject_invalid_tail_probability(bad: float) -> None:
    returns = np.array([0.01, -0.02, 0.03])
    with pytest.raises(ValueError, match="tail probability"):
        ValueAtRisk(returns, alpha=bad)
    with pytest.raises(ValueError, match="tail probability"):
        ExpectedShortfall(returns, alpha=bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, math.nan, math.inf])
def test_overlay_rejects_invalid_vol_target(bad: float) -> None:
    with pytest.raises(ValueError, match="vol_target"):
        BookRiskOverlay(vol_target=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, math.nan, math.inf])
def test_overlay_rejects_invalid_drawdown_limit(bad: float) -> None:
    with pytest.raises(ValueError, match="dd_limit"):
        BookRiskOverlay(dd_limit=bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_nonfinite_or_ruined_nav_permanently_halts_overlay(bad: float) -> None:
    overlay = BookRiskOverlay()
    overlay.observe(100.0)
    overlay.observe(bad)
    overlay.observe(120.0)

    assert overlay.ruined is True
    assert overlay.preview_scale() == 0.0


def test_es_gate_drops_size_for_an_unmeasurable_window() -> None:
    returns = np.zeros(12)
    returns[:8] = np.nan

    leverage = es_leverage(returns, lookback=8)

    assert leverage[7] == 0.0
    assert np.isfinite(leverage).all()


def test_remaining_drawdown_gate_ignores_nonfinite_marks_without_ruin() -> None:
    leverage = dd_remaining_leverage(np.array([0.01, math.nan, math.inf, -0.01]), limit=0.05)

    assert np.isfinite(leverage).all()
    assert np.all((leverage >= 0.0) & (leverage <= 1.0))


def test_empty_gate_stack_is_a_finite_empty_result() -> None:
    result = apply_gate_stack(np.array([]), GateSpec())

    assert result.returns.size == 0
    assert result.scale.size == 0
    assert result.snapshot()["mean_scale"] == 1.0
    assert result.snapshot()["last_scale"] == 1.0


@pytest.mark.parametrize("returns", [np.array([]), np.array([[0.0, 0.0]])])
def test_tail_quota_rejects_empty_or_non_1d_returns(returns: np.ndarray) -> None:
    with pytest.raises(ValueError, match="1-d array"):
        tail_quota_eprocess(returns, c=-0.02, alpha=0.05)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_tail_quota_rejects_nonpositive_alarm_threshold(bad: float) -> None:
    with pytest.raises(ValueError, match="alarm_at"):
        tail_quota_eprocess(np.array([0.0, -0.01]), c=-0.02, alpha=0.05, alarm_at=bad)
