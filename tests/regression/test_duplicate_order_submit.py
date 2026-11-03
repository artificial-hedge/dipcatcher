"""A repeated SimulatedBroker order id returns the first acknowledgement."""

from datetime import UTC, datetime

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.formal.order_lifecycle import check_trace
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus


def test_duplicate_order_id_returns_the_original_fill(tmp_path) -> None:
    """A second submit, even with another side and size, does not fill again.

    The first acknowledgement is the idempotency record. Restoring the book
    keeps that reservation. A hand-built trace that submits twice is still
    rejected by the lifecycle spec; the broker no longer produces that trace.
    """
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_net = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    when = datetime(2024, 1, 2, tzinfo=UTC)
    broker.mark({"A": 100.0})
    order = Order(
        order_id="same-id",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=10.0,
        signal_time=when,
        decision_time=when,
        order_time=when,
        status=OrderStatus.NEW,
    )
    first = broker.submit(order, price=100.0, nav=100_000.0, adv_dollars=1e12, sigma=0.0)
    retry = order.model_copy(update={"side": OrderSide.SELL, "quantity": 25.0})
    second = broker.submit(retry, price=50.0, nav=broker.nav(), adv_dollars=1e12, sigma=0.0)
    assert first.fill is not None
    assert second is first
    assert broker.shares["A"] == pytest.approx(10.0)
    assert broker.cash == pytest.approx(99_000.0)
    assert len(broker.fills) == 1
    assert len(broker.history) == 1

    restored = SimulatedBroker.from_state(cfg, broker.to_dict())
    replayed = restored.submit(retry, price=50.0, nav=restored.nav(), adv_dollars=1e12, sigma=0.0)
    assert replayed.fill is not None
    assert replayed.fill.fill_id == first.fill.fill_id
    assert restored.shares["A"] == pytest.approx(10.0)
    assert restored.cash == pytest.approx(99_000.0)
    assert len(restored.fills) == 1
    assert len(restored.history) == 1

    trace = [
        {"op": "submit", "order_id": "same-id", "qty": 10.0},
        {
            "op": "fill",
            "order_id": "same-id",
            "fill_id": first.fill.fill_id,
            "qty": 10.0,
            "rest": False,
        },
        {"op": "submit", "order_id": "same-id", "qty": 25.0},
    ]
    result = check_trace(trace)
    assert not result.ok
    assert any("submit from filled" in item for item in result.violations)


def test_duplicate_after_a_resting_fill_does_not_fill_again(tmp_path) -> None:
    """A bar fill does not free the id, and the retry still returns the first ack."""
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.costs.frictionless = True
    cfg.costs.participation_limit = 1.0
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_net = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    when = datetime(2024, 1, 2, tzinfo=UTC)
    broker.mark({"A": 100.0})
    resting = Order(
        order_id="rest-id",
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=4.0,
        signal_time=when,
        decision_time=when,
        order_time=when,
        status=OrderStatus.NEW,
        limit_price=100.0,
    )
    first = broker.submit(resting, price=100.0, nav=100_000.0, adv_dollars=1e12, sigma=0.0)
    assert first.fill is None
    assert first.order.status is OrderStatus.ACKED
    swept = broker.process_bar(
        "A",
        bar_open=100.0,
        bar_high=101.0,
        bar_low=99.0,
        bar_time=when,
        adv_dollars=1e12,
        sigma=0.0,
    )
    assert len(swept) == 1
    assert swept[0].fill is not None
    assert broker.shares["A"] == pytest.approx(4.0)
    retry = resting.model_copy(
        update={"side": OrderSide.SELL, "quantity": 9.0, "limit_price": None}
    )
    replay = broker.submit(retry, price=100.0, nav=broker.nav(), adv_dollars=1e12, sigma=0.0)
    assert replay is first
    assert replay.fill is None
    assert broker.shares["A"] == pytest.approx(4.0)
    assert len(broker.fills) == 1
    assert broker.cash == pytest.approx(99_600.0)
