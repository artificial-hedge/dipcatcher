"""Compiled and interpreted cash gates must fail closed identically."""

import math

import pytest

from quant_fund.backtest.fast_replay import _exceeds_nb, _funded_nb
from quant_fund.portfolio.risk_gate import exceeds_limit, funded


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1.0, 0.0, 1.0])
@pytest.mark.parametrize("limit", [float("nan"), float("inf"), float("-inf"), -1.0, 0.0, 1.0])
def test_compiled_limit_and_funding_parity(value: float, limit: float) -> None:
    assert bool(_exceeds_nb(value, limit)) == exceeds_limit(value, limit)
    assert bool(_funded_nb(limit, value)) == funded(limit, value)
    if not math.isfinite(value) or not math.isfinite(limit):
        assert bool(_exceeds_nb(value, limit))
        assert not bool(_funded_nb(limit, value))


@pytest.mark.parametrize("limit", [0.0, 1.0, 1_000_000.0])
def test_compiled_limits_preserve_finite_slack(limit: float) -> None:
    for value in (limit, math.nextafter(limit, math.inf)):
        assert not bool(_exceeds_nb(value, limit))
        assert bool(_funded_nb(limit, value))
    breach = limit + max(1e-6, limit * 1e-6)
    assert bool(_exceeds_nb(breach, limit))
    assert not bool(_funded_nb(limit, breach))
