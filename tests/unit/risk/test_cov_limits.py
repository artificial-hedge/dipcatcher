"""Pre-trade risk-gate limit rejections that no earlier test exercised.

The gate is the last check before an order is sized: net, participation and
the date-level market vol overlay must each reject on their own, and a legal
order must still pass. Config values come from the checked-in research config.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_fund.config import load_config
from quant_fund.portfolio.risk_gate import check_order, exceeds_limit, funded
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.orders import Order, OrderSide


def order(qty: float = 1.0, side: OrderSide = OrderSide.BUY) -> Order:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return Order(
        order_id="o1",
        security_id="A",
        symbol="A",
        side=side,
        quantity=qty,
        signal_time=now,
        decision_time=now,
        order_time=now,
    )


def kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "nav": 100.0,
        "price": 1.0,
        "current_weight": 0.0,
        "gross_after": 0.01,
        "net_after": 0.01,
        "participation": 0.01,
        "predicted_vol": 0.1,
    }
    base.update(overrides)
    return base


def test_net_limit_rejects_on_absolute_exposure() -> None:
    cfg = load_config("configs/research.yaml")
    g = cfg.risk_gate
    # |net| is compared: a large short book breaches exactly like a large long one.
    with pytest.raises(RiskGateRejected, match="net"):
        check_order(order(), **kwargs(net_after=-(g.max_net * 1.01)), config=cfg)
    with pytest.raises(RiskGateRejected, match="net"):
        check_order(order(), **kwargs(net_after=g.max_net * 1.01), config=cfg)


def test_participation_limit_rejects() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(RiskGateRejected, match="participation"):
        check_order(
            order(), **kwargs(participation=cfg.risk_gate.max_participation * 1.01), config=cfg
        )


def test_negative_market_overlay_vol_is_rejected() -> None:
    cfg = load_config("configs/research.yaml")
    with pytest.raises(RiskGateRejected, match="predicted volatility cannot be negative"):
        check_order(order(), **kwargs(), config=cfg, market_predicted_vol=-0.01)


def test_market_overlay_vol_is_the_one_compared_to_the_cap() -> None:
    cfg = load_config("configs/research.yaml")
    cap = cfg.risk_gate.max_predicted_vol
    # The name vol is inside the cap; the date-level overlay is not.
    with pytest.raises(RiskGateRejected, match="predicted vol"):
        check_order(
            order(),
            **kwargs(predicted_vol=cap * 0.5),
            config=cfg,
            market_predicted_vol=cap * 1.01,
        )


def test_a_legal_order_passes_every_limit() -> None:
    cfg = load_config("configs/research.yaml")
    assert check_order(order(), **kwargs(), config=cfg) is None


def test_limit_helpers_fail_closed_on_non_finite_numbers() -> None:
    assert exceeds_limit(float("nan"), 1.0) is True
    assert exceeds_limit(1.0, float("inf")) is True
    assert exceeds_limit(1.0, 1.0) is False
    assert funded(float("nan"), 1.0) is False
    assert funded(1.0, float("inf")) is False
    assert funded(1.0, 1.0) is True
