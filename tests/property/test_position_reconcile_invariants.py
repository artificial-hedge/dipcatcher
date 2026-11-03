"""Broker-reported positions against the internal book.

A mismatch sets the reconciliation alarm and does not overwrite the ledger.
The paper kill switch then refuses new orders. On the desk, a position book
that no longer matches fills latches the halt. Reports keep
``live_pnl_claim`` false. Fills are simulated correctness records.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.execution.order_recon import Fill, Order, OrderFillLedger
from quant_fund.live.desk import LiveDesk, LiveDeskError
from quant_fund.live.ledger import LedgerFill, LiveLedger
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from quant_fund.schemas.errors import KillSwitchActive
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

_FAST = settings(derandomize=True, max_examples=20, deadline=None)
_T0 = datetime(2024, 1, 2, tzinfo=UTC)
_CASH = 100_000.0
_TOLERANCE = 1e-6


def _desk_config() -> PretradeConfig:
    return PretradeConfig(
        schema_version=1,
        limits=limit_config(),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )


def _open_desk(root: Path) -> LiveDesk:
    return LiveDesk.open(
        root / "desk",
        hmac_key=HMAC_KEY,
        config=_desk_config(),
        cash=1_000_000.0,
        account_value=1_000_000.0,
        ts_ns=TS_NS,
    )


def _paper_order(side: str, qty: int) -> Order:
    return Order(order_id="o1", security_id="A", side=side, quantity=float(qty))


def _paper_fill(qty: int, price: int) -> Fill:
    return Fill(
        fill_id="f1",
        order_id="o1",
        security_id="A",
        quantity=float(qty),
        price=float(price),
        fill_time=_T0 + timedelta(seconds=1),
    )


def _signed(side: str, qty: float) -> float:
    return qty if side == "BUY" else -qty


@given(
    qty=st.integers(1, 12),
    price=st.integers(10, 40),
    side=st.sampled_from(("BUY", "SELL")),
    shares=st.integers(1, 5),
)
@_FAST
def test_broker_position_mismatch_alarms_and_halts_new_orders(
    qty: int, price: int, side: str, shares: int
) -> None:
    ledger = OrderFillLedger(
        endpoint_kind="paper",
        starting_cash=_CASH,
        cash_tolerance=_TOLERANCE,
        position_tolerance=_TOLERANCE,
        accept_orders=True,
    )
    assert ledger.submit(_paper_order(side, qty)) == "accepted"
    assert ledger.record_fill(_paper_fill(qty, price)) == "accepted"
    clean_book = ledger.reconcile()
    cash = float(clean_book["computed_cash"])
    positions = {key: float(value) for key, value in clean_book["computed_positions"].items()}
    ledger.declare_state(cash=cash, positions=positions)
    clean = ledger.reconcile()
    assert clean["ok"] is True
    assert clean["drift"]["alarm"] is False
    assert clean["live_pnl_claim"] is False
    assert clean["research_only"] is True
    assert positions["A"] == pytest.approx(_signed(side, qty))

    drifted = dict(positions)
    drifted["A"] = positions["A"] + shares
    ledger.declare_state(cash=cash, positions=drifted)
    dirty = ledger.reconcile()
    assert dirty["ok"] is False
    assert dirty["drift"]["alarm"] is True
    assert dirty["drift"]["position_drift"]["A"] == pytest.approx(shares)
    assert dirty["computed_positions"]["A"] == pytest.approx(positions["A"])
    assert dirty["kill_state"] == "ENABLED"
    assert dirty["live_pnl_claim"] is False

    assert ledger.trip_kill("broker position drift") == "HALT_NEW_ORDERS"
    halted = ledger.reconcile()
    assert halted["ok"] is False
    assert halted["kill_state"] == "HALT_NEW_ORDERS"
    with pytest.raises(KillSwitchActive):
        ledger.submit(Order(order_id="o2", security_id="A", side="BUY", quantity=1.0))
    assert ledger.cancel("o1") == "already_filled"
    assert ledger.reconcile()["computed_positions"]["A"] == pytest.approx(positions["A"])


@given(
    qty=st.integers(1, 12),
    price=st.integers(10, 40),
    side=st.sampled_from(("BUY", "SELL")),
)
@_FAST
def test_broker_report_inside_tolerance_stays_clean(qty: int, price: int, side: str) -> None:
    ledger = OrderFillLedger(endpoint_kind="paper", starting_cash=_CASH, accept_orders=True)
    ledger.submit(_paper_order(side, qty))
    ledger.record_fill(_paper_fill(qty, price))
    book = ledger.reconcile()
    cash = float(book["computed_cash"])
    positions = {key: float(value) for key, value in book["computed_positions"].items()}
    slack = _TOLERANCE / 10.0
    ledger.declare_state(
        cash=cash + slack,
        positions={"A": positions["A"] - slack},
    )
    report = ledger.reconcile()
    assert report["ok"] is True
    assert report["drift"]["alarm"] is False
    assert report["drift"]["declared"] is True
    assert report["live_pnl_claim"] is False


@given(
    qty=st.integers(1, 12),
    price=st.integers(10, 40),
    side=st.sampled_from(("buy", "sell")),
    shares=st.integers(1, 5),
)
@_FAST
def test_live_ledger_mismatch_alarms_and_keeps_the_internal_book(
    qty: int, price: int, side: str, shares: int
) -> None:
    ledger = LiveLedger(
        baseline_cash=_CASH,
        cash_tolerance=_TOLERANCE,
        position_tolerance=_TOLERANCE,
    )
    ledger.note_order("c1", expects_fill=True)
    fill = LedgerFill(
        fill_id="f1",
        client_order_id="c1",
        symbol="AAPL",
        side=side,
        quantity=float(qty),
        price=float(price),
        fill_time_ns=1,
    )
    assert ledger.record_fill(fill) == "accepted"
    assert ledger.record_fill(fill) == "duplicate_fill"
    signed = qty if side == "buy" else -qty
    ledger.declare(cash=ledger.computed_cash, positions={"AAPL": float(signed)})
    clean = ledger.reconcile()
    assert clean["ok"] is True
    assert clean["drift"]["alarm"] is False
    assert clean["duplicate_fills"] == 1
    assert clean["live_pnl_claim"] is False
    assert clean["research_evidence"] is False
    assert clean["computed_positions"]["AAPL"] == pytest.approx(signed)

    ledger.declare(cash=ledger.computed_cash, positions={"AAPL": float(signed + shares)})
    dirty = ledger.reconcile()
    assert dirty["ok"] is False
    assert dirty["drift"]["alarm"] is True
    assert dirty["drift"]["position_drift"]["AAPL"] == pytest.approx(shares)
    assert dirty["computed_positions"]["AAPL"] == pytest.approx(signed)
    assert dirty["computed_cash"] == pytest.approx(_CASH - signed * price)
    assert dirty["live_pnl_claim"] is False


@given(qty=st.integers(1, 8), price=st.integers(10, 30), shares=st.integers(1, 4))
@_FAST
def test_desk_position_mismatch_latches_the_halt(qty: int, price: int, shares: int) -> None:
    with TemporaryDirectory() as raw:
        desk = _open_desk(Path(raw))
        try:
            desk.arm("loopback", "", TS_NS)
            report = desk.submit(
                "aapl", "BUY", float(qty), float(price), TS_NS + 1, client_order_id="recon-1"
            )
            assert report["status"] == "filled"
            assert report["live_pnl_claim"] is False
            assert report["data_label"] == "SYNTHETIC"
            cash = float(report["cash"])
            desk.positions["AAPL"] = float(qty + shares)
            with pytest.raises(LiveDeskError, match="position book diverged from fills"):
                desk.mark("AAPL", float(price), TS_NS + 2)
            status = desk.status()
            assert status["kill_latched"] is True
            assert status["halt_reason"] == "position book diverged from fills"
            assert status["live_pnl_claim"] is False
            assert status["positions"]["AAPL"] == pytest.approx(qty + shares)
            assert status["cash"] == pytest.approx(cash)
            with pytest.raises(LiveDeskError, match="position book diverged from fills"):
                desk.submit("aapl", "BUY", 1.0, float(price), TS_NS + 3, client_order_id="recon-2")
            assert desk.positions["AAPL"] == pytest.approx(qty + shares)
            assert desk.cash == pytest.approx(cash)
        finally:
            desk.close()
