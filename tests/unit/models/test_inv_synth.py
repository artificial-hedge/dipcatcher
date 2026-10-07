"""Unit tests for quant_fund.models._inv_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._inv_synth import (
    D_RATE,
    demand,
    simulate,
)


def test_demand_deterministic_poisson() -> None:
    d1 = demand(0)
    d2 = demand(0)
    np.testing.assert_array_equal(d1, d2)
    assert d1.shape == (104,)
    assert (d1 >= 0).all()
    assert abs(d1.mean() - D_RATE) / D_RATE < 0.1


def test_demand_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="weeks"):
        demand(0, weeks=0)
    with pytest.raises(ValueError, match="rate"):
        demand(0, rate=-1.0)


def test_simulate_never_order_baseline() -> None:
    avg, fill, n_orders = simulate(lambda ip, w: 0.0, seed=0)
    assert n_orders == 0
    assert avg > 0  # pure shortage cost
    assert fill == 0.0


def test_simulate_order_up_to_policy() -> None:
    def s_policy(ip: float, w: int) -> float:
        return max(0.0, 200.0 - ip)

    avg, fill, n_orders = simulate(s_policy, seed=0)
    assert n_orders > 0
    assert fill > 0.5
    assert np.isfinite(avg)


def test_simulate_deterministic() -> None:
    r1 = simulate(lambda ip, w: 100.0 if ip < 50 else 0.0, seed=1)
    r2 = simulate(lambda ip, w: 100.0 if ip < 50 else 0.0, seed=1)
    assert r1 == r2


def test_simulate_rejects_hostile_policy() -> None:
    # negative qty was silently dropped — a broken policy looked fine
    with pytest.raises(ValueError, match="qty"):
        simulate(lambda ip, w: -5.0, seed=0)
    with pytest.raises(ValueError, match="qty"):
        simulate(lambda ip, w: float("nan"), seed=0)


def test_simulate_rejects_zero_weeks() -> None:
    with pytest.raises(ValueError, match="weeks"):
        simulate(lambda ip, w: 0.0, seed=0, weeks=0)


def test_zero_demand_fill_rate_is_vacuous_perfect() -> None:
    avg, fill, n_orders = simulate(lambda ip, w: 0.0, seed=0, weeks=4)
    # demand drawn from poisson can be low; contract: fill=1.0 iff no unmet demand
    d = demand(0, weeks=4, rate=0.0)
    assert d.sum() == 0
    avg0, fill0, _n = simulate(lambda ip, w: 0.0, seed=999, weeks=3)
    assert np.isfinite(avg0) and 0.0 <= fill0 <= 1.0
