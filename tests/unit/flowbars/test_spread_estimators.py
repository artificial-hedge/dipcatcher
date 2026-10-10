import numpy as np
import pytest

from quant_fund.flowbars.spread_estimators import (
    effective_spread_from_quotes,
    roll_implied_spread,
    roll_implied_spread_full,
)

pytestmark = pytest.mark.synthetic


def _roll_tape(n: int, half_spread: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stationary midprice plus bid-ask bounce: trade price = mid ± s."""
    rng = np.random.default_rng(seed)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.02, n))
    side = rng.choice([-1.0, 1.0], size=n)
    trade = mid + side * half_spread
    bid = mid - half_spread
    ask = mid + half_spread
    return trade, bid, ask


def test_roll_recovers_half_spread() -> None:
    half = 0.05
    trade, _, _ = _roll_tape(20000, half, seed=10)
    est = roll_implied_spread(trade)
    assert est == pytest.approx(2 * half, rel=0.15)


def test_roll_full_valid_flag() -> None:
    trade, _, _ = _roll_tape(20000, 0.05, seed=11)
    out = roll_implied_spread_full(trade)
    assert out["valid"] == 1.0
    assert out["gamma1"] < 0
    assert out["spread"] > 0


def test_roll_invalid_for_no_bounce() -> None:
    rng = np.random.default_rng(12)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.02, 20000))
    out = roll_implied_spread_full(mid)
    assert out["valid"] == 0.0
    assert out["spread"] == 0.0


def test_effective_spread_from_quotes() -> None:
    _, bid, ask = _roll_tape(5000, 0.05, seed=13)
    assert effective_spread_from_quotes(bid, ask) == pytest.approx(0.05, rel=0.02)


def test_validation() -> None:
    with pytest.raises(ValueError):
        effective_spread_from_quotes(np.array([2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        roll_implied_spread(np.array([1.0, 2.0]))
