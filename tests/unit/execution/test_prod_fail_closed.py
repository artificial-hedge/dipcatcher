"""Fail-closed checks for the paper ledger and the simulated broker.

SYNTHETIC prices are correctness fixtures, never market evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.order_recon import Order, OrderFillLedger, OrderLedgerError
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.monitoring.kill_switch import HALT_NEW_ORDERS
from quant_fund.schemas.errors import KillSwitchActive, RiskGateRejected
from quant_fund.schemas.orders import Order as BrokerOrder
from quant_fund.schemas.orders import OrderSide, OrderStatus

T0 = datetime(2026, 1, 2, tzinfo=UTC)


def _ledger() -> OrderFillLedger:
    return OrderFillLedger(endpoint_kind="paper", starting_cash=1_000.0, accept_orders=True)


def _order(oid: str = "o1", qty: float = 10.0, sec: str = "SEC_A") -> Order:
    return Order(order_id=oid, security_id=sec, side="BUY", quantity=qty)


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.risk_gate.stale_price_bars = 3
    cfg.costs.frictionless = True
    return cfg


def _broker_order(oid: str, qty: float, *, limit: float | None = None) -> BrokerOrder:
    return BrokerOrder(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=qty,
        signal_time=T0,
        decision_time=T0,
        order_time=T0,
        status=OrderStatus.NEW,
        limit_price=limit,
    )


def test_conflicting_order_id_does_not_replace_the_original() -> None:
    ledger = _ledger()
    assert ledger.submit(_order()) == "accepted"
    with pytest.raises(OrderLedgerError, match="conflicts with a different order"):
        ledger.submit(_order(qty=11.0))
    assert ledger.reconcile()["n_orders"] == 1
    assert ledger.submit(_order()) == "duplicate_order"
    assert ledger.reconcile()["counters"]["duplicate_orders"] == 1


def test_kill_wins_over_a_malformed_replay() -> None:
    ledger = _ledger()
    ledger.trip_kill("halt")
    with pytest.raises(KillSwitchActive):
        ledger.submit(_order(qty=float("nan")))


def test_non_finite_declaration_leaves_the_prior_declaration() -> None:
    ledger = _ledger()
    ledger.declare_state(cash=10.0, positions={"SEC_A": 1.0})
    with pytest.raises(OrderLedgerError, match="declared cash must be finite"):
        ledger.declare_state(cash=float("nan"), positions={"SEC_A": 1.0})
    report = ledger.reconcile()
    assert report["drift"]["declared"] is True
    assert report["drift"]["cash_drift"] == pytest.approx(990.0)
    with pytest.raises(OrderLedgerError, match="declared position must be finite"):
        ledger.declare_state(cash=10.0, positions={"SEC_A": float("inf")})
    assert ledger.reconcile()["drift"]["cash_drift"] == pytest.approx(990.0)


def test_filled_order_id_cannot_be_submitted_again(tmp_path) -> None:
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=1_000.0)
    broker.mark({"A": 100.0})
    first = broker.submit(_broker_order("o1", 5.0), price=100.0, adv_dollars=1e9)
    assert first.order.status is OrderStatus.FILLED
    cash = broker.cash
    with pytest.raises(ValueError, match="already submitted"):
        broker.submit(_broker_order("o1", 5.0), price=100.0, adv_dollars=1e9)
    assert broker.cash == pytest.approx(cash)
    assert broker.shares["A"] == pytest.approx(5.0)


def test_amend_honours_kill_and_notional(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_order_notional = 100.0
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_broker_order("o1", 1.0, limit=50.0), price=100.0, adv_dollars=1e9)
    broker.kill.set_state(HALT_NEW_ORDERS)
    with pytest.raises(KillSwitchActive):
        broker.amend_order("o1", quantity=2.0)
    assert broker.open_orders["o1"].quantity == pytest.approx(1.0)
    broker.kill.set_state("ENABLED")
    with pytest.raises(RiskGateRejected, match="order notional"):
        broker.amend_order("o1", quantity=10.0, limit_price=50.0)
    assert broker.open_orders["o1"].quantity == pytest.approx(1.0)


def test_stale_bar_does_not_fill_or_cancel_the_resting_order(tmp_path) -> None:
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_broker_order("o1", 1.0, limit=99.0), price=100.0, adv_dollars=1e9)
    cash = broker.cash
    records = broker.process_bar(
        "A",
        bar_open=100.0,
        bar_high=101.0,
        bar_low=98.0,
        adv_dollars=1e9,
        price_age_bars=100,
    )
    assert len(records) == 1
    assert records[0].order.status is OrderStatus.REJECTED
    assert records[0].reject_reason is not None
    assert "stale" in records[0].reject_reason
    assert "o1" in broker.open_orders
    assert broker.fills == []
    assert broker.cash == pytest.approx(cash)


def test_daily_loss_rejects_before_cash_moves(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.risk_gate.max_daily_loss_fraction = 0.1
    broker = SimulatedBroker(config=cfg, initial_cash=1_000.0)
    broker.mark({"A": 100.0})
    first = broker.submit(_broker_order("o1", 5.0), price=100.0, adv_dollars=1e9)
    assert first.order.status is OrderStatus.FILLED
    broker.mark({"A": 50.0})
    cash = broker.cash
    second = broker.submit(_broker_order("o2", 1.0), price=50.0, adv_dollars=1e9)
    assert second.order.status is OrderStatus.REJECTED
    assert second.reject_reason is not None
    assert "daily loss" in second.reject_reason
    assert broker.cash == pytest.approx(cash)
    assert broker.shares["A"] == pytest.approx(5.0)
