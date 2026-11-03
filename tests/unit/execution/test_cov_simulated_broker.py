"""Fail-closed branches of the simulated (paper) broker.

Limit-order bar validation, resting/cancel/amend bookkeeping, expiry sweeps,
and state-restore refusals. All prices are SYNTHETIC. No cash is ever real
and no vendor or network path is touched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quant_fund.config.loader import load_config
from quant_fund.execution.simulated_broker import RejectReason, SimulatedBroker
from quant_fund.monitoring.kill_switch import HALT_NEW_ORDERS
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus

T0 = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def _cfg(tmp_path):
    cfg = load_config("configs/paper.yaml")
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 2.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.risk_gate.max_predicted_vol = 1.0
    cfg.costs.frictionless = True
    return cfg


def _order(
    *,
    oid: str = "o1",
    side: OrderSide = OrderSide.BUY,
    qty: float = 10.0,
    limit: float | None = None,
    expire_time: datetime | None = None,
) -> Order:
    return Order(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=side,
        quantity=qty,
        limit_price=limit,
        expire_time=expire_time,
        signal_time=T0,
        decision_time=T0,
        order_time=T0,
        status=OrderStatus.NEW,
    )


def _broker(tmp_path, *, cash: float = 100_000.0, **kwargs) -> SimulatedBroker:
    broker = SimulatedBroker(config=_cfg(tmp_path), initial_cash=cash, **kwargs)
    broker.mark({"A": 100.0})
    return broker


# ---------------------------------------------------------------------------
# submission guards
# ---------------------------------------------------------------------------


def test_submit_rejects_a_missing_market_price(tmp_path):
    broker = _broker(tmp_path)
    rec = broker.submit(_order(), price=None)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.MISSING_PRICE.value
    assert broker.reject_count == 1
    assert broker.shares.get("A", 0.0) == 0.0


def test_submit_refuses_an_order_id_that_is_already_working(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(_order(limit=50.0), price=100.0)
    assert "o1" in broker.open_orders

    with pytest.raises(ValueError, match="already working"):
        broker.submit(_order(limit=50.0), price=100.0)


def test_buy_beyond_cash_is_rejected_without_moving_the_book(tmp_path):
    broker = _broker(tmp_path, cash=1.0)
    broker.config.costs.participation_limit = 1.0
    rec = broker.submit(_order(qty=10.0), price=100.0, nav=100_000.0, adv_dollars=1e3)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.INSUFFICIENT_CASH.value
    assert broker.cash == 1.0
    assert broker.shares.get("A", 0.0) == 0.0


def test_invalid_decision_price_raises_before_any_state_moves(tmp_path):
    broker = _broker(tmp_path)
    with pytest.raises(ValueError, match="decision_price"):
        broker.submit(_order(), price=100.0, adv_dollars=1e9, decision_price=float("nan"))
    assert broker.cash == 100_000.0
    assert broker.shares.get("A", 0.0) == 0.0
    assert broker.fills == []


# ---------------------------------------------------------------------------
# limit orders — bar validation and resting
# ---------------------------------------------------------------------------


def test_limit_order_without_bar_context_rests_instead_of_filling_blind(tmp_path):
    broker = _broker(tmp_path)
    rec = broker.submit(_order(limit=90.0), price=100.0, adv_dollars=1e9)
    assert rec.order.status is OrderStatus.ACKED
    assert rec.fill is None
    assert broker.open_orders["o1"].status is OrderStatus.ACKED
    assert broker.shares.get("A", 0.0) == 0.0


@pytest.mark.parametrize(
    "bar",
    [
        {"bar_open": 0.0, "bar_high": 100.0, "bar_low": 90.0},
        {"bar_open": 100.0, "bar_high": float("inf"), "bar_low": 90.0},
        {"bar_open": 95.0, "bar_high": 90.0, "bar_low": 80.0},
        {"bar_open": 95.0, "bar_high": 100.0, "bar_low": 96.0},
    ],
)
def test_limit_order_rejects_an_incoherent_execution_bar(tmp_path, bar):
    broker = _broker(tmp_path)
    with pytest.raises(ValueError, match="bar"):
        broker.submit(_order(limit=90.0), price=100.0, adv_dollars=1e9, **bar)
    assert broker.shares.get("A", 0.0) == 0.0
    assert "o1" not in broker.open_orders


def test_buy_limit_gapping_through_fills_at_the_open_not_the_limit(tmp_path):
    broker = _broker(tmp_path)
    rec = broker.submit(
        _order(limit=95.0),
        price=100.0,
        adv_dollars=1e9,
        bar_open=90.0,
        bar_high=100.0,
        bar_low=85.0,
    )
    assert rec.order.status is OrderStatus.FILLED
    assert rec.fill is not None
    assert rec.fill.price == pytest.approx(90.0)


def test_sell_limit_gapping_through_fills_at_the_open(tmp_path):
    broker = _broker(tmp_path)
    broker.shares["A"] = 10.0
    rec = broker.submit(
        _order(side=OrderSide.SELL, limit=105.0),
        price=100.0,
        adv_dollars=1e9,
        bar_open=110.0,
        bar_high=115.0,
        bar_low=100.0,
    )
    assert rec.order.status is OrderStatus.FILLED
    assert rec.fill is not None
    assert rec.fill.price == pytest.approx(110.0)


def test_untouched_limit_order_rests_and_keeps_no_fill(tmp_path):
    broker = _broker(tmp_path)
    rec = broker.submit(
        _order(limit=50.0),
        price=100.0,
        adv_dollars=1e9,
        bar_open=100.0,
        bar_high=110.0,
        bar_low=90.0,
    )
    assert rec.order.status is OrderStatus.ACKED
    assert "o1" in broker.open_orders


# ---------------------------------------------------------------------------
# cancel / amend
# ---------------------------------------------------------------------------


def test_cancel_unknown_order_fails_closed(tmp_path):
    broker = _broker(tmp_path)
    with pytest.raises(ValueError, match="not working"):
        broker.cancel_order("ghost")


def test_cancel_working_order_returns_cancelled_record(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(_order(limit=50.0), price=100.0)
    rec = broker.cancel_order("o1")
    assert rec.order.status is OrderStatus.CANCELLED
    assert "o1" not in broker.open_orders
    # Cancelling twice is a hard error, not a silent no-op.
    with pytest.raises(ValueError, match="not working"):
        broker.cancel_order("o1")


def test_amend_unknown_order_fails_closed(tmp_path):
    broker = _broker(tmp_path)
    with pytest.raises(ValueError, match="not working"):
        broker.amend_order("ghost", quantity=5.0)


def test_amend_revalidates_the_order_schema(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(_order(limit=50.0), price=100.0)
    with pytest.raises(ValueError):
        broker.amend_order("o1", quantity=0.0)
    # The original working order is untouched by the rejected amend.
    assert broker.open_orders["o1"].quantity == pytest.approx(10.0)


def test_amend_updates_quantity_in_place(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(_order(limit=50.0), price=100.0)
    rec = broker.amend_order("o1", quantity=25.0)
    assert rec.order.quantity == pytest.approx(25.0)
    assert broker.open_orders["o1"].quantity == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# process_bar sweeps
# ---------------------------------------------------------------------------


def test_process_bar_fills_a_touched_resting_order(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(_order(limit=95.0), price=100.0)
    records = broker.process_bar(
        "A",
        bar_open=96.0,
        bar_high=100.0,
        bar_low=90.0,
        bar_time=T0 + timedelta(minutes=1),
        adv_dollars=1e9,
    )
    assert len(records) == 1
    assert records[0].order.status is OrderStatus.FILLED
    assert broker.shares["A"] == pytest.approx(10.0)


def test_process_bar_keeps_untouched_orders_resting(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(_order(limit=50.0), price=100.0)
    records = broker.process_bar("A", bar_open=100.0, bar_high=110.0, bar_low=90.0)
    assert records == []
    assert "o1" in broker.open_orders


def test_process_bar_ignores_other_securities(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(_order(limit=50.0), price=100.0)
    records = broker.process_bar("B", bar_open=100.0, bar_high=110.0, bar_low=90.0)
    assert records == []
    assert "o1" in broker.open_orders


def test_process_bar_cancels_expired_orders(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(
        _order(limit=50.0, expire_time=T0 + timedelta(minutes=5)),
        price=100.0,
    )
    records = broker.process_bar(
        "A",
        bar_open=100.0,
        bar_high=110.0,
        bar_low=90.0,
        bar_time=T0 + timedelta(minutes=10),
    )
    assert len(records) == 1
    assert records[0].order.status is OrderStatus.CANCELLED
    assert records[0].reject_reason == "expired"
    assert "o1" not in broker.open_orders


def test_process_bar_halts_fills_under_the_kill_switch_but_keeps_the_order(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(_order(limit=95.0), price=100.0)
    broker.kill.set_state(HALT_NEW_ORDERS)

    records = broker.process_bar("A", bar_open=96.0, bar_high=100.0, bar_low=90.0)

    assert records == []
    assert "o1" in broker.open_orders
    assert broker.shares.get("A", 0.0) == 0.0


def test_process_bar_drops_an_order_rejected_by_the_risk_gate(tmp_path):
    broker = _broker(tmp_path)
    broker.submit(_order(limit=95.0), price=100.0)
    broker.config.risk_gate.max_order_notional = 1.0

    records = broker.process_bar("A", bar_open=96.0, bar_high=100.0, bar_low=90.0, adv_dollars=1e9)

    assert len(records) == 1
    assert records[0].order.status is OrderStatus.REJECTED
    assert "o1" not in broker.open_orders
    assert broker.risk_gate_reject_count == 1


def test_process_bar_rejects_an_incoherent_bar(tmp_path):
    broker = _broker(tmp_path)
    with pytest.raises(ValueError, match="bar"):
        broker.process_bar("A", bar_open=90.0, bar_high=80.0, bar_low=70.0)


def test_shadow_process_bar_never_moves_cash(tmp_path):
    shadow = _broker(tmp_path, cash=0.0, slot="shadow", allow_capital=False)
    shadow.submit(_order(limit=95.0), price=100.0)
    records = shadow.process_bar("A", bar_open=96.0, bar_high=100.0, bar_low=90.0)
    assert records == []
    assert shadow.cash == 0.0
    assert "o1" in shadow.open_orders


# ---------------------------------------------------------------------------
# valuation guards
# ---------------------------------------------------------------------------


def test_nav_requires_a_mark_for_every_held_security(tmp_path):
    broker = _broker(tmp_path)
    broker.shares["B"] = 5.0
    with pytest.raises(ValueError, match="missing market mark"):
        broker.nav()


def test_nav_rejects_nonpositive_marks(tmp_path):
    broker = _broker(tmp_path)
    broker.shares["B"] = 5.0
    with pytest.raises(ValueError, match="valuation marks"):
        broker.nav({"B": 0.0})


def test_capital_free_shadow_broker_reports_zero_exposure(tmp_path):
    shadow = _broker(tmp_path, cash=0.0, slot="shadow", allow_capital=False)
    shadow.shares["A"] = 5.0
    shadow.cash = -500.0  # positions offset cash exactly: NAV is zero
    assert shadow.exposures() == (0.0, 0.0)


def test_mark_skips_none_values_and_stores_the_rest(tmp_path):
    broker = _broker(tmp_path)
    broker.mark({"A": None, "B": 42.0})  # type: ignore[dict-item]
    assert broker.last_marks["A"] == 100.0
    assert broker.last_marks["B"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# from_state refusals
# ---------------------------------------------------------------------------


def test_from_state_requires_a_cash_key(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="cash"):
        SimulatedBroker.from_state(cfg, {"shares": {"A": 1.0}})


def test_from_state_rejects_nonfinite_shares(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="shares"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "shares": {"A": float("nan")}})


def test_from_state_rejects_non_string_kill_state(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="kill switch state"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "kill_state": 3})


def test_from_state_rejects_negative_receipt_baselines(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="receipt counts"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "n_orders": -1})
    with pytest.raises(ValueError, match="receipt counts"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "n_fills": -1})


def test_from_state_rejects_a_non_list_history(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="history must be a list"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "history": {"order": {}}})


def test_from_state_rejects_malformed_history_entries(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="history is invalid"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "history": [{"nope": 1}]})


def test_from_state_rejects_history_count_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    broker = _broker(tmp_path)
    broker.submit(_order(), price=100.0, adv_dollars=1e9)
    state = broker.to_dict()
    state["n_fills"] = 0
    with pytest.raises(ValueError, match="fill count"):
        SimulatedBroker.from_state(cfg, state)


def test_from_state_rejects_non_list_open_orders(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="open_orders must be a list"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "open_orders": {"o1": {}}})


def test_from_state_rejects_invalid_open_orders(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="open_orders are invalid"):
        SimulatedBroker.from_state(cfg, {"cash": 1.0, "open_orders": [{"order_id": "o1"}]})


def test_from_state_round_trips_a_resting_limit_order(tmp_path):
    cfg = _cfg(tmp_path)
    broker = _broker(tmp_path)
    broker.submit(_order(limit=95.0), price=100.0)

    restored = SimulatedBroker.from_state(cfg, broker.to_dict())

    assert "o1" in restored.open_orders
    assert restored.open_orders["o1"].limit_price == pytest.approx(95.0)
