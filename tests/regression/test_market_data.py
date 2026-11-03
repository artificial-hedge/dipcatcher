"""Regressions for live market-data gaps. SYNTHETIC gateway checks, not evidence.

Bad ticks must not become marks. Sequence holes trip stale_data instead of
being filled. Corporate actions keep account value flat or halt the name.
Unknown broker symbols and stale reference snapshots fail closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.live.brokers import BrokerFill, LiveOrder
from quant_fund.live.desk import LiveDeskError
from quant_fund.live.errors import LiveGateError
from quant_fund.live.market_data import ReferenceSnapshot
from quant_fund.pretrade.config import LimitConfig
from tests.unit.fault.support import TS, FaultBroker, arm, buy, open_desk, open_fault_session

_STEP = 2_100_000_000


def _limits(session: object, **updates: object) -> LimitConfig:
    return session.pretrade.limits.model_copy(update=updates)  # type: ignore[attr-defined, no-any-return]


class _Fills(FaultBroker):
    """Paper venue that prints the whole order on submit. Still no socket."""

    def __init__(self) -> None:
        super().__init__()
        self.fill_on_submit = True

    def submit(self, order: LiveOrder):  # type: ignore[no-untyped-def]
        ack = super().submit(order)
        if not self.fill_on_submit or ack.status != "accepted":
            return ack
        self.deliver(
            order.client_order_id,
            order.quantity,
            order.limit_price,
            order.fill_time_ns,
            f"fill-{order.client_order_id}",
        )
        return self._acks[order.client_order_id]


def test_bad_ticks_never_become_marks_or_authorize_orders(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    session.mark(symbol="AAPL", price=100.0, actor="ada", at_ns=TS)
    for price in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(LiveGateError):
            session.mark(symbol="AAPL", price=price, actor="ada", at_ns=TS + 1)
    with pytest.raises(LiveGateError, match="price_jump"):
        session.mark(symbol="AAPL", price=125.01, actor="ada", at_ns=TS + 2)
    assert session.marks["AAPL"] == 100.0
    session.mark(symbol="AAPL", price=125.0, actor="ada", at_ns=TS + 3)
    session.mark(symbol="AAPL", price=100.0, actor="ada", at_ns=TS + 4)
    quotes = (
        (0.0, 100.0, "bad_bid"),
        (-1.0, 100.0, "bad_bid"),
        (float("nan"), 101.0, "bad_bid"),
        (float("inf"), 101.0, "bad_bid"),
        (100.0, 0.0, "bad_ask"),
        (101.0, 100.0, "crossed"),
        (100.0, 100.0, "locked"),
        (100.0, 110.0, "wide_spread"),
        (130.0, 131.0, "price_jump"),
    )
    for index, (bid, ask, reason) in enumerate(quotes):
        result = session.ingest_quote(
            symbol="AAPL",
            bid=bid,
            ask=ask,
            seq=index + 1,
            actor="ada",
            at_ns=TS + 5 + index,
        )
        assert result["ok"] is False
        assert reason in result["reasons"]
        assert session.marks["AAPL"] == 100.0
        assert result["stale"] is False
    assert session.kill_state["latched"] is False
    jumped = buy(
        session,
        mark=200.0,
        limit_price=100.0,
        client_order_id="jump",
        at_ns=TS + 30,
    )
    assert jumped["broker_called"] is False
    assert "price_jump" in jumped["reasons"]
    assert session.marks["AAPL"] == 100.0
    allowed = buy(session, client_order_id="still", at_ns=TS + 31)
    assert allowed["ok"] is True
    assert allowed["broker_called"] is True
    assert broker.submit_count == 1


def test_sequence_gaps_duplicates_and_reversals_trip_stale_data(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    session.mark(symbol="AAPL", price=100.0, actor="ada", at_ns=TS)
    first = session.ingest_quote(
        symbol="AAPL", bid=100.0, ask=101.0, seq=1, actor="ada", at_ns=TS + 1
    )
    assert first["ok"] is True
    assert session.marks["AAPL"] == pytest.approx(100.5)
    gap = session.ingest_quote(
        symbol="AAPL", bid=100.0, ask=101.0, seq=4, actor="ada", at_ns=TS + 2
    )
    assert gap["ok"] is False
    assert gap["stale"] is True
    assert "sequence_gap" in gap["reasons"]
    assert session.marks["AAPL"] == pytest.approx(100.5)
    denied = buy(session, client_order_id="during-gap", at_ns=TS + 3)
    assert denied["broker_called"] is False
    assert denied["reasons"] == ["stale_data"]
    assert session.marks["AAPL"] == pytest.approx(100.5)
    duplicate = session.ingest_quote(
        symbol="AAPL", bid=100.0, ask=101.0, seq=1, actor="ada", at_ns=TS + 4
    )
    assert "duplicate_sequence" in duplicate["reasons"]
    reversal = session.ingest_quote(
        symbol="AAPL", bid=100.0, ask=101.0, seq=0, actor="ada", at_ns=TS + 5
    )
    assert "out_of_order" in reversal["reasons"]
    assert session.marks["AAPL"] == pytest.approx(100.5)
    assert broker.submit_count == 0
    recovered = session.ingest_quote(
        symbol="AAPL", bid=100.0, ask=101.0, seq=2, actor="ada", at_ns=TS + 6
    )
    assert recovered["ok"] is True
    assert recovered["stale"] is False
    assert buy(session, client_order_id="after", at_ns=TS + 7)["ok"] is True

    fresh = open_fault_session(tmp_path / "snap", FaultBroker())
    arm(fresh)
    fresh.mark(symbol="AAPL", price=100.0, actor="ada", at_ns=TS)
    fresh.ingest_quote(symbol="AAPL", bid=100.0, ask=101.0, seq=1, actor="ada", at_ns=TS + 1)
    fresh.ingest_quote(symbol="AAPL", bid=100.0, ask=101.0, seq=5, actor="ada", at_ns=TS + 2)
    assert buy(fresh, client_order_id="held", at_ns=TS + 3)["reasons"] == ["stale_data"]
    shot = fresh.ingest_quote(
        symbol="AAPL",
        bid=100.0,
        ask=101.0,
        seq=9,
        actor="ada",
        at_ns=TS + 4,
        snapshot=True,
    )
    assert shot["ok"] is True
    assert shot["stale"] is False
    assert buy(fresh, client_order_id="cleared", at_ns=TS + 5)["broker_called"] is True


def test_corporate_actions_do_not_invent_pnl_or_mask_the_loss_latch(tmp_path: Path) -> None:
    broker = _Fills()
    session = open_fault_session(tmp_path / "split", broker)
    arm(session)
    session.update_limits(_limits(session, max_daily_loss_fraction=0.001), actor="ada")
    assert buy(session, quantity=100.0, client_order_id="lot")["ok"] is True
    before = broker.account().equity
    peak = float(session.engine.book.peak)
    split = session.apply_corporate_action(
        kind="split", symbol="AAPL", actor="ada", ratio=2.0, at_ns=TS + _STEP
    )
    assert split["ok"] is True
    assert broker.positions["AAPL"] == pytest.approx(200.0)
    assert session.marks["AAPL"] == pytest.approx(50.0)
    assert broker.account().equity == pytest.approx(before)
    assert session.kill_state["latched"] is False
    assert float(session.engine.book.peak) == pytest.approx(peak)
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(200.0)

    reverse_broker = _Fills()
    reverse = open_fault_session(tmp_path / "reverse", reverse_broker)
    arm(reverse)
    reverse.update_limits(_limits(reverse, max_daily_loss_fraction=0.001), actor="ada")
    assert buy(reverse, quantity=100.0, client_order_id="rev")["ok"] is True
    done = reverse.apply_corporate_action(
        kind="reverse_split", symbol="AAPL", actor="ada", ratio=0.5, at_ns=TS + _STEP
    )
    assert done["ok"] is True
    assert reverse_broker.positions["AAPL"] == pytest.approx(50.0)
    assert reverse.marks["AAPL"] == pytest.approx(200.0)
    assert reverse_broker.account().equity == pytest.approx(before)
    assert reverse.kill_state["latched"] is False

    div_broker = _Fills()
    dividend = open_fault_session(tmp_path / "div", div_broker)
    arm(dividend)
    dividend.update_limits(_limits(dividend, max_daily_loss_fraction=0.001), actor="ada")
    assert buy(dividend, quantity=100.0, client_order_id="div")["ok"] is True
    div_peak = float(dividend.engine.book.peak)
    paid = dividend.apply_corporate_action(
        kind="dividend", symbol="AAPL", actor="ada", cash_per_share=1.0, at_ns=TS + _STEP
    )
    assert paid["ok"] is True
    assert div_broker.cash == pytest.approx(1_000_000.0 - 10_000.0 + 100.0)
    assert dividend.marks["AAPL"] == pytest.approx(99.0)
    assert div_broker.account().equity == pytest.approx(before)
    assert dividend.kill_state["latched"] is False
    assert float(dividend.engine.book.peak) == pytest.approx(div_peak)
    dividend.mark(symbol="AAPL", price=80.0, actor="ada", at_ns=TS + 2 * _STEP)
    assert dividend.kill_state["latched"] is True
    assert dividend.kill_state["reason"] == "daily_loss"

    renamed_broker = _Fills()
    renamed = open_fault_session(tmp_path / "name", renamed_broker)
    arm(renamed)
    assert buy(renamed, symbol="FB", quantity=100.0, client_order_id="fb")["ok"] is True
    change = renamed.apply_corporate_action(
        kind="symbol_change",
        symbol="FB",
        actor="ada",
        new_symbol="META",
        at_ns=TS + _STEP,
    )
    assert change["ok"] is True
    assert renamed_broker.positions["META"] == pytest.approx(100.0)
    assert "FB" not in renamed_broker.positions
    assert renamed.marks["META"] == pytest.approx(100.0)
    assert renamed_broker.account().equity == pytest.approx(before)
    assert renamed.kill_state["latched"] is False

    listed = _Fills()
    halt = open_fault_session(tmp_path / "delist", listed)
    arm(halt)
    assert buy(halt, quantity=40.0, client_order_id="stay")["ok"] is True
    equity = listed.account().equity
    stopped = halt.apply_corporate_action(
        kind="delisting", symbol="AAPL", actor="ada", at_ns=TS + _STEP
    )
    assert stopped["ok"] is False
    assert stopped["reason"] == "delisting"
    assert listed.positions["AAPL"] == pytest.approx(40.0)
    assert halt.marks["AAPL"] == pytest.approx(100.0)
    assert listed.account().equity == pytest.approx(equity)
    assert halt.kill_state["latched"] is False
    blocked = buy(halt, quantity=1.0, client_order_id="after-delist", at_ns=TS + 2 * _STEP)
    assert blocked["broker_called"] is False
    assert "halted" in blocked["reasons"]


def test_working_orders_and_share_caps_halt_until_reconciled(tmp_path: Path) -> None:
    broker = _Fills()
    session = open_fault_session(tmp_path / "work", broker)
    arm(session)
    assert buy(session, quantity=100.0, client_order_id="filled")["ok"] is True
    broker.fill_on_submit = False
    assert buy(session, quantity=1.0, client_order_id="resting", at_ns=TS + 1)["status"] == (
        "accepted"
    )
    held = session.apply_corporate_action(
        kind="split", symbol="AAPL", actor="ada", ratio=2.0, at_ns=TS + 2
    )
    assert held["ok"] is False
    assert held["reason"] == "working_orders"
    assert broker.positions["AAPL"] == pytest.approx(100.0)
    assert session.marks["AAPL"] == 100.0
    assert session.kill_state["latched"] is False
    assert session.cancel(client_order_id="resting", actor="ada", at_ns=TS + 3)["status"] == (
        "canceled"
    )
    applied = session.apply_corporate_action(
        kind="split", symbol="AAPL", actor="ada", ratio=2.0, at_ns=TS + 4
    )
    assert applied["ok"] is True
    assert broker.positions["AAPL"] == pytest.approx(200.0)
    assert session.marks["AAPL"] == pytest.approx(50.0)
    assert session._halted.get("AAPL", False) is False
    resumed = buy(
        session,
        quantity=1.0,
        limit_price=50.0,
        mark=50.0,
        client_order_id="after-split",
        at_ns=TS + _STEP,
    )
    assert resumed["broker_called"] is True

    capped = _Fills()
    book = open_fault_session(tmp_path / "cap", capped)
    arm(book)
    assert buy(book, quantity=10_000.0, limit_price=10.0, mark=10.0, client_order_id="a")["ok"]
    assert buy(
        book,
        quantity=10_000.0,
        limit_price=10.0,
        mark=10.0,
        client_order_id="b",
        at_ns=TS + _STEP,
    )["ok"]
    refused = book.apply_corporate_action(
        kind="split", symbol="AAPL", actor="ada", ratio=2.0, at_ns=TS + 2 * _STEP
    )
    assert refused["ok"] is False
    assert refused["reason"] == "position_limit"
    assert capped.positions["AAPL"] == pytest.approx(20_000.0)
    assert book.marks["AAPL"] == 10.0
    assert book.kill_state["latched"] is False


def test_symbol_map_is_bijective_and_unknown_broker_symbols_fail(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    session.bind_symbol(internal="AAPL", broker="APL", actor="ada")
    with pytest.raises(LiveGateError, match="already bound"):
        session.bind_symbol(internal="AAPL", broker="MSFT", actor="ada")
    with pytest.raises(LiveGateError, match="already bound"):
        session.bind_symbol(internal="IBM", broker="APL", actor="ada")
    with pytest.raises(LiveGateError, match="unknown broker symbol"):
        session.ingest_quote(symbol="ZZZZ", bid=100.0, ask=101.0, seq=1, actor="ada", at_ns=TS)
    assert buy(session, client_order_id="echo")["ok"] is True
    translated = session.ingest_quote(
        symbol="APL", bid=100.0, ask=101.0, seq=1, actor="ada", at_ns=TS + 1
    )
    assert translated["ok"] is True
    assert translated["symbol"] == "AAPL"
    broker._page.append(
        BrokerFill(
            fill_id="via-broker",
            client_order_id="echo",
            symbol="APL",
            side="buy",
            quantity=10.0,
            price=100.0,
            fill_time_ns=TS + 2,
        )
    )
    report = session.reconcile(at_ns=TS + 3)
    assert report["live_pnl_claim"] is False
    assert session.ledger.computed_positions["AAPL"] == pytest.approx(10.0)
    assert "APL" not in session.ledger.computed_positions
    broker._page.append(
        BrokerFill(
            fill_id="via-unknown",
            client_order_id="echo",
            symbol="NOPE",
            side="buy",
            quantity=1.0,
            price=100.0,
            fill_time_ns=TS + 4,
        )
    )
    with pytest.raises(LiveGateError, match="unknown broker symbol NOPE"):
        session.reconcile(at_ns=TS + 5)


def test_stale_reference_snapshot_fails_closed(tmp_path: Path) -> None:
    broker = FaultBroker()
    session = open_fault_session(tmp_path, broker)
    arm(session)
    with pytest.raises(LiveGateError, match="bad_reference_version"):
        session.set_reference(
            ReferenceSnapshot(
                version=0,
                as_of_ns=TS,
                max_age_ns=5_000_000_000,
                lot_size={"AAPL": 1.0},
                tick_size={"AAPL": 0.01},
            ),
            actor="ada",
        )
    session.set_reference(
        ReferenceSnapshot(
            version=3,
            as_of_ns=TS,
            max_age_ns=5_000_000_000,
            lot_size={"AAPL": 1.0},
            tick_size={"AAPL": 0.01},
        ),
        actor="ada",
    )
    odd_lot = buy(session, quantity=10.5, client_order_id="lot")
    assert odd_lot["broker_called"] is False
    assert "lot_size" in odd_lot["reasons"]
    odd_tick = buy(
        session,
        quantity=10.0,
        limit_price=100.005,
        mark=100.005,
        client_order_id="tick",
        at_ns=TS + 1,
    )
    assert odd_tick["broker_called"] is False
    assert "tick_size" in odd_tick["reasons"]
    assert broker.submit_count == 0
    session.set_reference(
        ReferenceSnapshot(
            version=4,
            as_of_ns=TS - 10_000_000_000,
            max_age_ns=1_000,
            lot_size={"AAPL": 1.0},
            tick_size={"AAPL": 0.01},
        ),
        actor="ada",
    )
    stale = buy(session, client_order_id="stale-ref", at_ns=TS + 2)
    assert stale["broker_called"] is False
    assert stale["reasons"] == ["stale_reference"]
    assert broker.submit_count == 0


def test_restart_restores_map_feed_reference_and_one_split(tmp_path: Path) -> None:
    root = tmp_path / "restart"
    broker = _Fills()
    session = open_fault_session(root, broker)
    arm(session)
    session.bind_symbol(internal="AAPL", broker="APL", actor="ada")
    assert buy(session, quantity=100.0, client_order_id="lot")["ok"] is True
    applied = session.apply_corporate_action(
        kind="split", symbol="AAPL", actor="ada", ratio=2.0, at_ns=TS + 1
    )
    assert applied["ok"] is True
    assert session.marks["AAPL"] == pytest.approx(50.0)
    session.ingest_quote(symbol="APL", bid=49.5, ask=50.5, seq=1, actor="ada", at_ns=TS + 2)
    gap = session.ingest_quote(symbol="APL", bid=49.5, ask=50.5, seq=4, actor="ada", at_ns=TS + 3)
    assert gap["stale"] is True
    assert session.marks["AAPL"] == pytest.approx(50.0)
    session.set_reference(
        ReferenceSnapshot(
            version=7,
            as_of_ns=TS - 10_000_000_000,
            max_age_ns=1_000,
            lot_size={"AAPL": 1.0},
            tick_size={"AAPL": 0.01},
        ),
        actor="ada",
    )
    again = open_fault_session(root, broker)
    assert again.ledger.computed_positions["AAPL"] == pytest.approx(200.0)
    assert broker.positions["AAPL"] == pytest.approx(200.0)
    assert again.marks["AAPL"] == pytest.approx(50.0)
    assert broker.account().equity == pytest.approx(1_000_000.0)
    assert again.kill_state["latched"] is False
    assert again._symbol_map["AAPL"] == "APL"
    assert again._feeds["AAPL"][1] is True
    assert again._reference is not None and again._reference.version == 7
    sid = again._sids["AAPL"]
    assert again.engine.book.syms[sid].pos == pytest.approx(200.0)
    cleared = again.ingest_quote(
        symbol="APL",
        bid=49.5,
        ask=50.5,
        seq=8,
        actor="ada",
        at_ns=TS + 4,
        snapshot=True,
    )
    assert cleared["ok"] is True
    assert cleared["stale"] is False
    denied = buy(
        again,
        quantity=1.0,
        limit_price=50.0,
        mark=50.0,
        client_order_id="stale-ref",
        at_ns=TS + 5,
    )
    assert denied["broker_called"] is False
    assert denied["reasons"] == ["stale_reference"]


def test_desk_rejects_bad_ticks_gaps_and_phantom_splits(tmp_path: Path) -> None:
    desk = open_desk(tmp_path / "desk")
    try:
        desk.arm("loopback", "", TS)
        desk.mark("AAPL", 100.0, TS)
        with pytest.raises(LiveDeskError, match="price_jump"):
            desk.mark("AAPL", 140.0, TS + 1)
        assert desk.marks["AAPL"] == 100.0
        desk.bind_symbol("AAPL", "AAPL", TS + 1)
        bad = desk.ingest_quote("AAPL", 101.0, 100.0, 1, TS + 2)
        assert bad["accepted"] is False
        assert "crossed" in bad["reasons"]
        assert desk.marks["AAPL"] == 100.0
        assert desk.engine.book.killed == 0
        good = desk.ingest_quote("AAPL", 100.0, 101.0, 1, TS + 3)
        assert good["accepted"] is True
        gap = desk.ingest_quote("AAPL", 100.0, 101.0, 4, TS + 4)
        assert gap["stale"] is True
        denied = desk.submit("AAPL", "BUY", 10, 100.5, TS + 5, client_order_id="gap")
        assert denied["accepted"] is False
        assert denied["transmitted"] is False
        assert "stale_data" in denied["order"]["reasons"]  # type: ignore[index]
        assert desk._loopback._queued == []
        desk.ingest_quote("AAPL", 100.0, 101.0, 2, TS + 6)
        filled = desk.submit("AAPL", "BUY", 100, 100.0, TS + 7, client_order_id="buy")
        assert filled["status"] == "filled"
        before = float(desk._account_value())
        peak = float(desk.peak)
        split = desk.apply_corporate_action("split", "AAPL", TS + 8, ratio=2.0)
        assert split["accepted"] is True
        assert desk.positions["AAPL"] == pytest.approx(200.0)
        assert desk.marks["AAPL"] == pytest.approx(50.0)
        assert desk._account_value() == pytest.approx(before)
        assert desk.engine.book.killed == 0
        assert float(desk.peak) == pytest.approx(peak)
        with pytest.raises(LiveDeskError, match="unknown broker symbol"):
            desk.ingest_quote("NOPE", 100.0, 101.0, 1, TS + 9)
    finally:
        desk.close()
