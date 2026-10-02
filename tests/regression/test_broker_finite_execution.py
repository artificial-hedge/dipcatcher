"""Public paper-broker inputs must not create unaccounted or nonfinite fills."""

from datetime import UTC, datetime

import pytest

from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus


def _order(side=OrderSide.BUY, quantity=10.0, limit_price=None):
    t = datetime(2024, 1, 2, tzinfo=UTC)
    return Order(
        order_id="finite-execution",
        security_id="A",
        symbol="A",
        side=side,
        quantity=quantity,
        signal_time=t,
        decision_time=t,
        order_time=t,
        limit_price=limit_price,
    )


@pytest.mark.parametrize("cash", [float("nan"), float("inf"), float("-inf"), -1.0])
@pytest.mark.parametrize("allow_capital", [True, False])
def test_initial_cash_must_be_finite_and_nonnegative(cash, allow_capital):
    with pytest.raises(ValueError, match="initial cash must be finite and non-negative"):
        SimulatedBroker(AppConfig(), initial_cash=cash, allow_capital=allow_capital)


@pytest.mark.parametrize("allow_capital", [True, False])
def test_zero_initial_cash_remains_valid(allow_capital):
    broker = SimulatedBroker(AppConfig(), initial_cash=0.0, allow_capital=allow_capital)
    assert broker.cash == 0.0


@pytest.mark.parametrize("cash", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_explicit_nav_cannot_hide_nonfinite_cash(cash, side):
    broker = SimulatedBroker(AppConfig())
    broker.cash = cash
    with pytest.raises(ValueError, match="broker cash must be finite"):
        broker.submit(_order(side), price=100.0, nav=1e6, adv_dollars=1e9)
    assert not broker.shares and not broker.fills and not broker.history


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
@pytest.mark.parametrize(
    "cost", ["commission_bps", "half_spread_bps", "impact_y", "bps_per_turnover"]
)
def test_finite_cost_inputs_that_overflow_leave_book_unchanged(cost, side):
    broker = SimulatedBroker(AppConfig(costs=CostConfig(**{cost: 1e308})))
    before = broker.to_dict()
    with pytest.raises(ValueError, match="execution costs must be finite"):
        broker.submit(_order(side), price=100.0, nav=1e6, adv_dollars=1e9)
    assert broker.to_dict() == before
    assert not broker.fills


def test_resting_sell_cost_overflow_preserves_working_order():
    cfg = AppConfig(costs=CostConfig(commission_bps=1e308))
    broker = SimulatedBroker(cfg)
    broker.submit(_order(OrderSide.SELL, limit_price=100.0), price=100.0, adv_dollars=1e9)
    before = broker.to_dict()
    with pytest.raises(ValueError, match="execution costs must be finite"):
        broker.process_bar("A", bar_open=100.0, bar_high=101.0, bar_low=99.0, adv_dollars=1e9)
    assert broker.to_dict() == before
    assert not broker.fills


def test_finite_sell_proceeds_cannot_overflow_cash():
    cfg = AppConfig(
        costs=CostConfig(frictionless=True),
        risk_gate=RiskGateConfig(max_order_notional=1e308, max_name=0.1, max_participation=0.1),
    )
    broker = SimulatedBroker(cfg, initial_cash=1.7e308)
    before = broker.to_dict()
    with pytest.raises(ValueError, match="post-fill cash must be finite"):
        broker.submit(_order(OrderSide.SELL, quantity=1e307), price=1.0, adv_dollars=1e308)
    assert broker.to_dict() == before
    assert not broker.fills


def test_finite_decision_price_overflow_does_not_mutate_book():
    broker = SimulatedBroker(AppConfig())
    before = broker.to_dict()
    with pytest.raises(ValueError, match="fill costs must be finite"):
        broker.submit(_order(OrderSide.SELL), price=100.0, adv_dollars=1e9, decision_price=1e308)
    assert broker.to_dict() == before
    assert not broker.fills


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
def test_finite_execution_preserves_cash_and_fill_identity(side):
    broker = SimulatedBroker(AppConfig())
    rec = broker.submit(_order(side), price=100.0, adv_dollars=1e9, decision_price=99.0)
    assert rec.order.status is OrderStatus.FILLED
    fill = rec.fill
    assert fill is not None
    signed_quantity = fill.quantity if side is OrderSide.BUY else -fill.quantity
    costs = fill.fee + fill.spread_cost + fill.impact_cost + fill.turnover_cost
    assert broker.cash == pytest.approx(broker.initial_cash - signed_quantity * fill.price - costs)
    assert broker.shares == {"A": signed_quantity}
    assert broker.fills == [fill]


def test_fill_id_failure_does_not_mutate_book():
    def fail_id(kind):
        raise ValueError(f"cannot mint {kind} id")

    broker = SimulatedBroker(AppConfig(), id_factory=fail_id)
    before = broker.to_dict()
    with pytest.raises(ValueError, match="cannot mint fill id"):
        broker.submit(_order(), price=100.0, adv_dollars=1e9)
    assert broker.to_dict() == before
    assert not broker.fills


def test_restored_finite_negative_cash_can_be_repaid_by_sale():
    cfg = AppConfig()
    state = SimulatedBroker(cfg).to_dict()
    state.update(cash=-100.0, shares={"A": 10.0}, last_marks={"A": 100.0})
    broker = SimulatedBroker.from_state(cfg, state)
    rec = broker.submit(_order(OrderSide.SELL), price=100.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.FILLED
    assert broker.cash > 0.0
    assert broker.shares == {"A": 0.0}
    assert broker.fills == [rec.fill]
