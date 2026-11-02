"""Restart reconciliation must halt when the book would otherwise be guessed."""

from datetime import UTC, datetime

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.live.ledger import LedgerFill, LiveLedger
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    return cfg


def _order(*, limit: float | None = None, qty: float = 10.0, oid: str = "o1") -> Order:
    t = datetime(2024, 1, 2, tzinfo=UTC)
    return Order(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=qty,
        signal_time=t,
        decision_time=t,
        order_time=t,
        status=OrderStatus.NEW,
        limit_price=limit,
    )


def test_from_state_refuses_missing_shares(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(), price=100.0, nav=100_000.0, adv_dollars=1e9)
    state = broker.to_dict()
    state.pop("shares")

    with pytest.raises(ValueError, match="shares"):
        SimulatedBroker.from_state(cfg, state)


def test_from_state_refuses_missing_open_orders(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    state = broker.to_dict()
    state.pop("open_orders")

    with pytest.raises(ValueError, match="open_orders"):
        SimulatedBroker.from_state(cfg, state)


def test_from_state_refuses_missing_marks_for_a_held_name(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(), price=100.0, nav=100_000.0, adv_dollars=1e9)
    state = broker.to_dict()
    state.pop("last_marks")

    with pytest.raises(ValueError, match="last_marks|market mark"):
        SimulatedBroker.from_state(cfg, state)


def test_from_state_refuses_a_dropped_partial_residual(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.participation_limit = 0.1
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(limit=99.0, qty=10.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    recs = broker.process_bar(
        "A",
        bar_open=100.0,
        bar_high=101.0,
        bar_low=98.0,
        bar_time=datetime(2024, 1, 3, tzinfo=UTC),
        adv_dollars=4950.0,
    )
    assert recs[0].fill is not None and recs[0].fill.is_partial
    assert broker.open_orders["o1"].status is OrderStatus.PARTIAL
    state = broker.to_dict()
    state["open_orders"] = []

    with pytest.raises(ValueError, match="partial"):
        SimulatedBroker.from_state(cfg, state)


def test_from_state_refuses_shares_that_disagree_with_fills(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(), price=100.0, nav=100_000.0, adv_dollars=1e9)
    state = broker.to_dict()
    state["shares"] = {"A": 0.0}

    with pytest.raises(ValueError, match="fill history"):
        SimulatedBroker.from_state(cfg, state)


def test_from_state_keeps_a_partial_residual_on_roundtrip(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    cfg.costs.participation_limit = 0.1
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(_order(limit=99.0, qty=10.0), price=100.0, nav=100_000.0, adv_dollars=1e9)
    broker.process_bar(
        "A",
        bar_open=100.0,
        bar_high=101.0,
        bar_low=98.0,
        bar_time=datetime(2024, 1, 3, tzinfo=UTC),
        adv_dollars=4950.0,
    )
    restored = SimulatedBroker.from_state(cfg, broker.to_dict())
    assert restored.open_orders["o1"].status is OrderStatus.PARTIAL
    assert restored.open_orders["o1"].quantity == pytest.approx(5.0)
    assert restored.shares["A"] == pytest.approx(broker.shares["A"])


def test_undeclared_broker_statement_is_not_a_clean_book() -> None:
    ledger = LiveLedger(baseline_cash=1_000.0)
    ledger.note_order("a", expects_fill=True)
    assert ledger.record_fill(LedgerFill("f1", "a", "AAPL", "buy", 1.0, 10.0, 20)) == "accepted"
    report = ledger.reconcile()
    assert report["ok"] is False
    assert report["drift"]["declared"] is False
    assert report["drift"]["alarm"] is True
    assert report["live_pnl_claim"] is False

    ledger.declare(cash=ledger.computed_cash, positions=ledger.computed_positions)
    clean = ledger.reconcile()
    assert clean["ok"] is True
    assert clean["drift"]["alarm"] is False


def test_resting_limit_survives_resume_until_it_is_cancelled(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    broker.submit(
        _order(limit=99.0, qty=10.0),
        price=100.0,
        nav=100_000.0,
        adv_dollars=1e9,
    )
    state = broker.to_dict()
    state["open_orders"] = []
    with pytest.raises(ValueError, match="open order"):
        SimulatedBroker.from_state(cfg, state)

    cancelled = SimulatedBroker.from_state(cfg, broker.to_dict())
    cancelled.cancel_order("o1")
    resumed = SimulatedBroker.from_state(cfg, cancelled.to_dict())
    assert "o1" not in resumed.open_orders
    assert resumed.shares == {}
