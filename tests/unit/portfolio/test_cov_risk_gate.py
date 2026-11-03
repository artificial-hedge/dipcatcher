"""Coverage: portfolio risk-gate fail-closed branches not hit by the happy path."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_fund.config import load_config
from quant_fund.portfolio.risk_gate import (
    check_order,
    exceeds_limit,
    funded,
    resolve_gate_predicted_vol,
)
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.orders import Order, OrderSide


def _order(qty: float = 1.0, side: OrderSide = OrderSide.BUY) -> Order:
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


def _kwargs() -> dict[str, float]:
    return {
        "nav": 100.0,
        "price": 1.0,
        "current_weight": 0.0,
        "gross_after": 0.01,
        "net_after": 0.01,
        "participation": 0.01,
        "predicted_vol": 0.1,
    }


def _reject(message: str, **overrides: object) -> None:
    cfg = load_config("configs/research.yaml")
    kwargs = {**_kwargs(), **overrides}
    with pytest.raises(RiskGateRejected, match=message):
        check_order(_order(), **kwargs, config=cfg)  # type: ignore[arg-type]


def test_exceeds_limit_is_fail_closed_on_non_finite_operands() -> None:
    assert exceeds_limit(float("nan"), 1.0) is True
    assert exceeds_limit(1.0, float("nan")) is True
    assert exceeds_limit(float("inf"), 1.0) is True


def test_exceeds_limit_tolerates_float_noise_only() -> None:
    assert exceeds_limit(1.0, 1.0) is False
    assert exceeds_limit(1.0, 2.0) is False
    # Within absolute slack.
    assert exceeds_limit(1.0 + 1e-10, 1.0) is False
    # Within relative slack for a large limit.
    assert exceeds_limit(1_000_000.0 + 1e-8, 1_000_000.0) is False
    # A visible breach still rejects.
    assert exceeds_limit(1.0001, 1.0) is True


def test_funded_requires_finite_operands_and_ignores_ulp_shortfall() -> None:
    assert funded(float("nan"), 1.0) is False
    assert funded(1.0, float("nan")) is False
    assert funded(float("inf"), 1.0) is False
    assert funded(1.0, 1.0) is True
    assert funded(1.0, 1.0 + 1e-10) is True
    assert funded(1.0, 1.5) is False


def test_resolve_gate_predicted_vol_handles_zero_and_none() -> None:
    assert resolve_gate_predicted_vol(0.2, None) == pytest.approx(0.2)
    assert resolve_gate_predicted_vol(0.2, 0.0) == 0.0
    assert resolve_gate_predicted_vol(0.9, 0.1) == pytest.approx(0.1)


def test_price_and_model_age_must_be_representable() -> None:
    # math.isfinite overflows on absurd integer ages; must reject, not raise.
    _reject("price age must be finite", price_age_bars=10**400)
    _reject("model age must be finite", model_age_hours=10**400)
    # Unsupported runtime types must also fail closed.
    _reject("price age must be finite", price_age_bars="1")
    _reject("model age must be finite", model_age_hours="1")


def test_negative_ages_reject() -> None:
    _reject("price age cannot be negative", price_age_bars=-1)
    _reject("model age cannot be negative", model_age_hours=-0.5)


def test_nonfinite_pre_trade_metrics_reject() -> None:
    for key in ("nav", "price", "current_weight", "gross_after", "net_after"):
        _reject("non-finite", **{key: float("inf")})
    _reject("non-finite", participation=float("nan"))
    _reject("non-finite", market_predicted_vol=float("nan"))


def test_negative_risk_metrics_reject() -> None:
    _reject("gross exposure cannot be negative", gross_after=-0.01)
    _reject("participation cannot be negative", participation=-0.01)
    _reject("predicted volatility cannot be negative", predicted_vol=-0.01)
    _reject("predicted volatility cannot be negative", market_predicted_vol=-0.01)


def test_notional_name_net_participation_and_vol_caps_reject() -> None:
    cfg = load_config("configs/research.yaml")
    _reject("order notional", nav=1e9, price=1e9)
    _reject("name weight", current_weight=cfg.risk_gate.max_name + 0.01)
    _reject("net", net_after=-(cfg.risk_gate.max_net + 0.01))
    _reject("participation", participation=cfg.risk_gate.max_participation + 0.01)
    _reject("predicted vol", predicted_vol=cfg.risk_gate.max_predicted_vol + 0.01)


def test_sell_order_reduces_name_weight() -> None:
    cfg = load_config("configs/research.yaml")
    # A sell from a flat book would be a short; the abs() name-weight cap sees it.
    with pytest.raises(RiskGateRejected, match="name weight"):
        check_order(_order(qty=1000.0, side=OrderSide.SELL), **_kwargs(), config=cfg)
    check_order(_order(qty=1.0, side=OrderSide.SELL), **_kwargs(), config=cfg)


def test_market_overlay_replaces_name_vol_for_the_gate() -> None:
    cfg = load_config("configs/research.yaml")
    # Name vol breaches but the date-level overlay is calm.
    check_order(
        _order(),
        **{**_kwargs(), "predicted_vol": cfg.risk_gate.max_predicted_vol + 0.05},
        config=cfg,
        market_predicted_vol=0.1,
    )
    # Overlay itself over the cap rejects even with a calm name vol.
    with pytest.raises(RiskGateRejected, match="predicted vol"):
        check_order(
            _order(),
            **_kwargs(),
            config=cfg,
            market_predicted_vol=cfg.risk_gate.max_predicted_vol + 0.05,
        )
