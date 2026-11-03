"""Kill-switch and order-state fail-closed branches of the paper order ledger.

SYNTHETIC fills only: correctness fixtures, never market evidence.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_fund.execution.order_recon import (
    Fill,
    LiveEndpointRefused,
    Order,
    OrderFillLedger,
    OrderLedgerError,
    refuse_live_endpoint,
)
from quant_fund.schemas.errors import KillSwitchActive

T0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 3, 21, 0, tzinfo=UTC)
T2 = datetime(2024, 1, 4, 21, 0, tzinfo=UTC)


def _order(
    oid: str = "o1",
    sec: str = "SEC_A",
    side: str = "BUY",
    qty: float = 10.0,
    expects_fill: bool = True,
) -> Order:
    return Order(
        order_id=oid,
        security_id=sec,
        side=side,  # type: ignore[arg-type]
        quantity=qty,
        expects_fill=expects_fill,
    )


def _fill(
    fid: str = "f1",
    oid: str = "o1",
    sec: str = "SEC_A",
    qty: float = 10.0,
    price: float = 10.0,
    t: datetime = T0,
    seq: int | None = 1,
) -> Fill:
    return Fill(
        fill_id=fid,
        order_id=oid,
        security_id=sec,
        quantity=qty,
        price=price,
        fill_time=t,
        seq=seq,
    )


def _ledger(**kw: float) -> OrderFillLedger:
    return OrderFillLedger(endpoint_kind="paper", accept_orders=True, **kw)


# -- endpoint refusal --


@pytest.mark.parametrize("kind", [" Paper ", "SIMULATED", "paper "])
def test_endpoint_kind_is_normalized(kind: str) -> None:
    ledger = OrderFillLedger(endpoint_kind=kind)
    assert ledger.endpoint_kind in {"paper", "simulated"}


@pytest.mark.parametrize("kind", ["paper-live", "sim_prod", "paper real"])
def test_endpoint_kind_tokens_refuse_live_components(kind: str) -> None:
    with pytest.raises(LiveEndpointRefused):
        refuse_live_endpoint(kind)


@pytest.mark.parametrize("kind", ["", "   "])
def test_endpoint_kind_must_be_non_empty(kind: str) -> None:
    with pytest.raises(OrderLedgerError, match="non-empty string"):
        refuse_live_endpoint(kind)


def test_endpoint_kind_must_be_a_string() -> None:
    with pytest.raises(OrderLedgerError, match="non-empty string"):
        refuse_live_endpoint(3)  # type: ignore[arg-type]


# -- kill switch --


def test_trip_kill_defaults_an_empty_reason_to_manual() -> None:
    led = _ledger()
    assert led.trip_kill("") == "HALT_NEW_ORDERS"
    assert led.kill_state == "HALT_NEW_ORDERS"
    with pytest.raises(KillSwitchActive, match="manual"):
        led.submit(_order())


def test_trip_kill_keeps_the_latest_reason_and_clear_resets_it() -> None:
    led = _ledger()
    led.trip_kill("first")
    led.trip_kill("second")
    with pytest.raises(KillSwitchActive, match="second"):
        led.submit(_order())
    assert led.clear_kill() == "ENABLED"
    assert led.kill_state == "ENABLED"
    assert led.submit(_order()) == "accepted"


def test_kill_switch_blocks_duplicate_replays_too() -> None:
    led = _ledger()
    led.submit(_order())
    led.trip_kill("halt")
    with pytest.raises(KillSwitchActive):
        led.submit(_order())
    assert led.reconcile()["counters"]["duplicate_orders"] == 0


def test_kill_switch_does_not_block_fills_or_cancels() -> None:
    led = _ledger(starting_cash=1000.0)
    led.submit(_order(qty=5.0))
    led.trip_kill("halt")
    assert led.record_fill(_fill(qty=5.0)) == "accepted"
    assert led.reconcile()["kill_state"] == "HALT_NEW_ORDERS"
    assert led.reconcile()["ok"] is True


# -- cancel input guards --


@pytest.mark.parametrize("order_id", ["", "   "])
def test_cancel_rejects_a_blank_order_id(order_id: str) -> None:
    with pytest.raises(OrderLedgerError, match="order_id must be a non-empty string"):
        _ledger().cancel(order_id)


def test_cancel_rejects_a_non_string_order_id() -> None:
    with pytest.raises(OrderLedgerError, match="order_id must be a non-empty string"):
        _ledger().cancel(7)  # type: ignore[arg-type]


# -- fill validation --


@pytest.mark.parametrize("qty", ["10", None, [10.0]])
def test_fill_quantity_must_be_a_number(qty: object) -> None:
    fill = _fill()
    object.__setattr__(fill, "quantity", qty)
    with pytest.raises(OrderLedgerError, match="fill quantity must be a finite number"):
        _ledger().record_fill(fill)


@pytest.mark.parametrize("qty", [True, False])
def test_fill_quantity_must_not_be_a_bool(qty: bool) -> None:
    fill = _fill()
    object.__setattr__(fill, "quantity", qty)
    with pytest.raises(OrderLedgerError, match="fill quantity must be a finite number"):
        _ledger().record_fill(fill)


@pytest.mark.parametrize("price", ["10", None, True])
def test_fill_price_must_be_a_number(price: object) -> None:
    fill = _fill()
    object.__setattr__(fill, "price", price)
    with pytest.raises(OrderLedgerError, match="fill price must be a finite number"):
        _ledger().record_fill(fill)


@pytest.mark.parametrize("price", [float("nan"), float("inf"), 0.0, -1.0])
def test_fill_price_must_be_finite_and_positive(price: float) -> None:
    with pytest.raises(OrderLedgerError, match="fill price must be finite and strictly positive"):
        _ledger().record_fill(_fill(price=price))


def test_fill_ids_must_be_non_empty() -> None:
    for field in ("fill_id", "order_id", "security_id"):
        fill = _fill()
        object.__setattr__(fill, field, "")
        with pytest.raises(OrderLedgerError, match="must be non-empty"):
            _ledger().record_fill(fill)


# -- order state and reconciliation --


def test_order_without_expected_fill_is_never_missing() -> None:
    led = _ledger()
    led.submit(_order(expects_fill=False))
    report = led.reconcile()
    assert report["missing_fills"] == []
    assert report["orders"][0]["status"] == "open"
    assert report["ok"] is True


def test_cancel_removes_the_residual_from_missing_fills() -> None:
    led = _ledger()
    led.submit(_order(qty=10.0))
    led.record_fill(_fill(qty=4.0))
    assert led.reconcile()["missing_fills"] == []
    led.cancel("o1")
    assert led.reconcile()["missing_fills"] == []
    assert led.reconcile()["orders"][0]["status"] == "cancelled"


def test_unknown_expected_fill_ids_are_reported_missing() -> None:
    led = _ledger()
    led.submit(_order(expects_fill=False))
    report = led.reconcile(expected_fill_ids=["ghost", "ghost"])
    assert report["missing_fills"] == ["ghost"]
    assert report["ok"] is False
    assert report["n_orders"] == 1
    assert report["n_fills"] == 0


def test_out_of_order_fill_still_applies_to_positions() -> None:
    led = _ledger(starting_cash=1000.0)
    led.submit(_order(qty=10.0))
    led.record_fill(_fill(fid="f2", qty=4.0, t=T1, seq=2))
    verdict = led.record_fill(_fill(fid="f1", qty=3.0, t=T0, seq=1))
    assert verdict == "out_of_order_fill"
    report = led.reconcile()
    assert report["counters"]["out_of_order_fills"] == 1
    assert report["computed_positions"]["SEC_A"] == 7.0
    assert report["ok"] is False


def test_seq_is_ignored_when_absent_but_time_still_orders() -> None:
    led = _ledger()
    led.submit(_order(qty=10.0))
    assert led.record_fill(_fill(fid="f1", qty=2.0, t=T1, seq=None)) == "accepted"
    assert led.record_fill(_fill(fid="f2", qty=2.0, t=T0, seq=None)) == "out_of_order_fill"


def test_sell_fill_writes_a_negative_position_and_positive_cash() -> None:
    led = _ledger(starting_cash=1000.0)
    led.submit(_order(side="SELL", qty=5.0))
    assert led.record_fill(_fill(qty=5.0, price=10.0)) == "accepted"
    report = led.reconcile()
    assert report["computed_positions"]["SEC_A"] == -5.0
    assert report["computed_cash"] == pytest.approx(1050.0)
    assert report["orders"][0]["status"] == "filled"


def test_declared_state_accepts_string_position_keys() -> None:
    led = _ledger()
    led.declare_state(cash=0.0, positions={"SEC_A": 0})
    report = led.reconcile()
    assert report["drift"]["declared"] is True
    assert report["drift"]["position_drift"] == {"SEC_A": 0.0}
    assert report["drift"]["alarm"] is False


def test_drift_at_the_tolerance_boundary_is_not_an_alarm() -> None:
    led = _ledger(starting_cash=1000.0)
    led.submit(_order(qty=5.0))
    led.record_fill(_fill(qty=5.0, price=10.0))
    led.declare_state(cash=950.0, positions={"SEC_A": 5.0})
    assert led.reconcile()["drift"]["alarm"] is False
    led.declare_state(cash=950.0, positions={"SEC_A": 5.0001})
    assert led.reconcile()["drift"]["alarm"] is True


def test_drift_reports_positions_known_to_one_side_only() -> None:
    led = _ledger(starting_cash=1000.0)
    led.declare_state(cash=1000.0, positions={"SEC_Z": 3.0})
    drift = led.reconcile()["drift"]
    assert drift["position_drift"] == {"SEC_Z": 3.0}
    assert drift["alarm"] is True


def test_reconcile_before_declaration_reports_no_drift() -> None:
    drift = _ledger().reconcile()["drift"]
    assert drift == {"declared": False, "cash_drift": None, "position_drift": {}, "alarm": False}
