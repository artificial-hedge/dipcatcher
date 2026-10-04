"""SYNTHETIC freshness validation through the gate and paper-broker APIs."""

import math
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from quant_fund.config.models import AppConfig
from quant_fund.execution.simulated_broker import RejectReason, SimulatedBroker
from quant_fund.portfolio.risk_gate import check_order
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus

_INVALID_AGES = [
    pytest.param(10**1000, id="huge-positive-int"),
    pytest.param(-(10**1000), id="huge-negative-int"),
    pytest.param(10**10000, id="int-beyond-string-limit"),
    pytest.param(Decimal("sNaN"), id="signaling-decimal-nan"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
    pytest.param("1", id="numeric-string"),
    pytest.param("invalid", id="invalid-string"),
    pytest.param(1j, id="complex"),
    pytest.param([], id="list"),
    pytest.param({}, id="mapping"),
    pytest.param(object(), id="object"),
]
_AGE_FIELDS = ["price_age_bars", "model_age_hours"]


def _order() -> Order:
    now = datetime(2024, 1, 1, tzinfo=UTC)
    return Order(
        order_id="age-validation",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=1.0,
        signal_time=now,
        decision_time=now,
        order_time=now,
    )


def _check(config, **ages):
    check_order(
        _order(),
        nav=100.0,
        price=1.0,
        current_weight=0.0,
        gross_after=0.01,
        net_after=0.01,
        participation=0.01,
        predicted_vol=0.1,
        config=config,
        **ages,
    )


@pytest.mark.parametrize("field", _AGE_FIELDS)
@pytest.mark.parametrize("age", _INVALID_AGES)
def test_invalid_age_is_a_gate_rejection(field, age):
    with pytest.raises(RiskGateRejected, match="age must be finite"):
        _check(AppConfig(), **{field: age})


@pytest.mark.parametrize("field", _AGE_FIELDS)
@pytest.mark.parametrize("age", _INVALID_AGES)
def test_broker_records_invalid_age_as_risk_rejection(field, age):
    broker = SimulatedBroker(AppConfig())
    cash_before = broker.cash
    rec = broker.submit(_order(), price=1.0, adv_dollars=1e9, **{field: age})
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason is not None
    assert rec.reject_reason.startswith(RejectReason.RISK_GATE.value)
    assert "age must be finite" in rec.reject_reason
    assert rec.fill is None
    assert broker.history == [rec]
    assert broker.reject_count == broker.risk_gate_reject_count == 1
    assert broker.halt_count == 0
    assert broker.cash == cash_before
    assert broker.shares == {}
    assert broker.fills == []


@pytest.mark.parametrize("field", _AGE_FIELDS)
@pytest.mark.parametrize(
    "case", ["omitted", "zero", "fractional", "below", "at", "above", "negative"]
)
def test_finite_age_boundaries_preserve_gate_and_broker_behavior(field, case):
    cfg = AppConfig()
    limit = (
        cfg.risk_gate.stale_price_bars
        if field == "price_age_bars"
        else cfg.risk_gate.stale_model_hours
    )
    values = {
        "zero": 0,
        "fractional": 0.5,
        "below": math.nextafter(limit, -math.inf),
        "at": limit,
        "above": math.nextafter(limit, math.inf),
        "negative": -0.1,
    }
    ages = {} if case == "omitted" else {field: values[case]}
    rejects = case in {"above", "negative"}
    if rejects:
        with pytest.raises(RiskGateRejected, match="is stale|cannot be negative"):
            _check(cfg, **ages)
    else:
        _check(cfg, **ages)
    broker = SimulatedBroker(cfg)
    rec = broker.submit(_order(), price=1.0, adv_dollars=1e9, **ages)
    assert rec.order.status is (OrderStatus.REJECTED if rejects else OrderStatus.FILLED)
    assert broker.risk_gate_reject_count == int(rejects)
    assert len(broker.fills) == int(not rejects)
