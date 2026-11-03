"""Partial fill versus cancel. Seeded order, same result for the same seed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.unit.fault.support import (
    SEEDS,
    STARTING_CASH,
    TS,
    FaultBroker,
    Scheduler,
    arm,
    buy,
    first_effect,
    open_desk,
    open_fault_session,
)
from tests.unit.pretrade.support import TS_NS


def _winner_cash(names: tuple[str, ...]) -> str:
    return first_effect(names)


def test_desk_partial_fill_cancel_race_is_seeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    for seed in SEEDS:
        names = Scheduler(seed).interleave(["fill", "cancel", "dup"])
        assert names == Scheduler(seed).interleave(["fill", "cancel", "dup"])
        _desk_race(tmp_path / f"s{seed}", names)


def _desk_race(root: Path, names: tuple[str, ...]) -> None:
    with open_desk(root) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 10, 50.0, TS_NS + 1, client_order_id="part")
        order = sent["order"]
        assert isinstance(order, dict)
        order_id = str(order["order_id"])
        inbound = root / "desk" / "inbound"
        inbound.mkdir()
        for index, name in enumerate(names):
            if name == "cancel":
                payload: dict[str, object] = {
                    "schema": "live_broker_event.v1",
                    "kind": "cancel",
                    "order_id": order_id,
                    "event_id": "c1",
                }
            else:
                payload = {
                    "schema": "live_broker_event.v1",
                    "kind": "fill",
                    "order_id": order_id,
                    "event_id": "p1",
                    "qty": 4,
                    "price": 50,
                }
            (inbound / f"{index:02d}-{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        report = desk.poll(TS_NS + 2)
    assert report["live_pnl_claim"] is False
    assert report["reserved_cash"] == 0
    if _winner_cash(names) == "fill":
        assert report["positions"] == {"AAPL": 4}
        assert report["cash"] == pytest.approx(999_800.0)
        assert report["kill_latched"] is False
    else:
        assert report["positions"] == {}
        assert report["cash"] == pytest.approx(STARTING_CASH)
        assert report["kill_latched"] is True


def test_session_partial_fill_cancel_race_is_seeded(tmp_path: Path) -> None:
    for seed in SEEDS[:4]:
        names = Scheduler(seed).interleave(["fill", "cancel", "dup"])
        _session_race(tmp_path / f"s{seed}", names, seed)


def _session_race(root: Path, names: tuple[str, ...], seed: int) -> None:
    broker = FaultBroker()
    session = open_fault_session(root, broker, venue="alpaca-paper")
    arm(session)
    client = f"race-{seed}"
    accepted = buy(session, quantity=10.0, client_order_id=client)
    assert accepted["status"] == "accepted"
    assert broker.submit_count == 1
    stamp = TS
    for name in names:
        stamp += 1_000_000
        if name == "cancel":
            session.cancel(client_order_id=client, actor="ada", at_ns=stamp)
        else:
            broker.deliver(client, 4.0, 100.0, stamp, "p1")
            session.reconcile(at_ns=stamp)
    assert broker.submit_count == 1
    position = float(session.ledger.computed_positions.get("AAPL", 0.0))
    if first_effect(names) == "fill":
        assert position == pytest.approx(4.0)
        assert broker.positions["AAPL"] == pytest.approx(4.0)
        assert session.ledger.computed_cash == pytest.approx(broker.cash)
        assert session.kill_state["latched"] is False
    else:
        assert position == pytest.approx(0.0)
        assert broker.positions.get("AAPL", 0.0) == pytest.approx(0.0)
        assert session.ledger.computed_cash == pytest.approx(STARTING_CASH)
        assert session.kill_state["latched"] is True
        blocked = buy(session, quantity=1.0, client_order_id=f"after-{seed}", at_ns=stamp + 1)
        assert blocked["ok"] is False
        assert "kill_switch" in blocked["reasons"]
        assert blocked["broker_called"] is False
        assert broker.submit_count == 1
    assert accepted["live_pnl_claim"] is False
