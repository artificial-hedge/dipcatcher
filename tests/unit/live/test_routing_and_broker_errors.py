"""Order routing, broker error mapping, and a ledger that alarms on drift."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.bridge import DropBridge, LoopbackBridge, parse_event
from quant_fund.live.brokers import (
    AlpacaBroker,
    BrokerAccount,
    BrokerAck,
    BrokerFill,
    FillPage,
    LiveOrder,
    alpaca_credentials,
)
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import BrokerRejected, LiveBrokerRefused, LiveGateError
from quant_fund.live.ledger import LedgerFill, LiveLedger, ledger_fill_from_dict
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def _fill(fill_id: str, order_id: str, side: str = "buy", when: int = 10) -> LedgerFill:
    return LedgerFill(fill_id, order_id, "AAPL", side, 1.0, 10.0, when)


def test_live_ledger_rejects_bad_inputs_and_alarms_on_drift() -> None:
    with pytest.raises(ValueError, match="baseline cash"):
        LiveLedger(baseline_cash=float("nan"))
    with pytest.raises(ValueError, match="baseline position"):
        LiveLedger(baseline_cash=1.0, baseline_positions={"AAPL": float("inf")})
    with pytest.raises(ValueError, match="side"):
        ledger_fill_from_dict(
            {
                "fill_id": "f",
                "client_order_id": "c",
                "symbol": "AAPL",
                "side": "short",
                "quantity": 1,
                "price": 1,
                "fill_time_ns": 1,
            }
        )
    with pytest.raises(ValueError, match="fill_time"):
        ledger_fill_from_dict(
            {
                "fill_id": "f",
                "client_order_id": "c",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 1,
                "price": 1,
                "fill_time_ns": -1,
            }
        )
    with pytest.raises(ValueError, match="quantity"):
        ledger_fill_from_dict(
            {
                "fill_id": "f",
                "client_order_id": "c",
                "symbol": "AAPL",
                "side": "sell",
                "quantity": 0,
                "price": 1,
                "fill_time_ns": 1,
            }
        )

    ledger = LiveLedger(baseline_cash=1_000.0, baseline_positions={"AAPL": 1.0})
    with pytest.raises(ValueError, match="not in the ledger"):
        ledger.set_expects_fill("missing", False)
    ledger.note_order("c1", expects_fill=True)
    ledger.set_expects_fill("c1", True)
    with pytest.raises(ValueError, match="side"):
        ledger.record_fill(LedgerFill("bad", "c1", "AAPL", "short", 1.0, 10.0, 1))

    assert ledger.record_fill(_fill("order:c1", "c1", when=100)) == "accepted"
    replaced = ledger.record_fill(_fill("fill-1", "c1", when=50))
    assert replaced == "replaced_provisional"
    assert ledger.record_fill(_fill("fill-2", "c1", when=40)) == "out_of_order_fill"
    assert ledger.record_fill(_fill("fill-2", "c1", when=40)) == "duplicate_fill"
    assert ledger.record_fill(_fill("orphan", "nobody", "sell", 1)) == "orphan_fill"
    with pytest.raises(ValueError, match="declared cash"):
        ledger.declare(cash=float("nan"), positions={})
    with pytest.raises(ValueError, match="declared position"):
        ledger.declare(cash=1.0, positions={"AAPL": float("nan")})
    ledger.declare(cash=0.0, positions={})
    report = ledger.reconcile()
    assert report["ok"] is False
    assert report["drift"]["alarm"] is True
    assert report["orphan_fills"] == 1
    assert report["out_of_order_fills"] == 1
    assert report["live_pnl_claim"] is False
    assert report["research_evidence"] is False
    assert ledger.computed_positions["AAPL"] > 0

    quiet = LiveLedger(baseline_cash=10.0)
    quiet.note_order("waiting", expects_fill=True)
    waiting = quiet.reconcile()
    assert waiting["missing_fills"] == ["waiting"]
    assert waiting["drift"]["declared"] is False
    assert waiting["ok"] is False


def test_bridge_events_reject_a_malformed_broker_report(tmp_path: Path) -> None:
    def event(**overrides: object) -> dict[str, object]:
        raw: dict[str, object] = {
            "schema": "live_broker_event.v1",
            "kind": "ack",
            "order_id": "o1",
            "event_id": "e1",
        }
        raw.update(overrides)
        return raw

    with pytest.raises(ValueError, match="schema"):
        parse_event(event(schema="nope"), synthetic=False, source="in.json")
    with pytest.raises(ValueError, match="kind"):
        parse_event(event(kind="explode"), synthetic=False, source="in.json")
    with pytest.raises(ValueError, match="order_id"):
        parse_event(event(order_id=""), synthetic=False, source="in.json")
    with pytest.raises(ValueError, match="event_id"):
        parse_event(event(event_id=1), synthetic=False, source="in.json")
    with pytest.raises(ValueError, match="fee"):
        parse_event(event(fee=-1), synthetic=False, source="in.json")
    with pytest.raises(ValueError, match="positive"):
        parse_event(event(kind="fill", qty=0, price=1), synthetic=False, source="in.json")
    with pytest.raises(ValueError, match="reason"):
        parse_event(event(reason=1), synthetic=False, source="in.json")
    with pytest.raises(ValueError, match="number"):
        parse_event(event(qty=True), synthetic=False, source="in.json")
    with pytest.raises(ValueError, match="finite"):
        parse_event(event(qty=float("inf")), synthetic=False, source="in.json")
    parsed = parse_event(event(kind="fill", qty=1, price=2, fee=0.1), synthetic=True, source="loop")
    assert parsed.synthetic is True
    assert parsed.qty == 1

    loop = LoopbackBridge()
    with pytest.raises(ValueError, match="order_id"):
        loop.transmit({"ticket": "order"})
    with pytest.raises(ValueError, match="order or a cancel"):
        loop.transmit({"ticket": "amend", "order_id": "o1"})
    loop.transmit({"ticket": "cancel", "order_id": "o1"})
    loop.transmit({"ticket": "order", "order_id": "o2", "qty": 1, "limit_price": 5})
    kinds = [item.kind for item in loop.poll()]
    assert kinds == ["cancel", "ack", "fill"]
    assert loop.poll() == []

    drop = DropBridge(tmp_path)
    assert drop.poll() == []
    with pytest.raises(ValueError, match="order_id"):
        drop.transmit({"ticket": "order"})
    with pytest.raises(ValueError, match="order or a cancel"):
        drop.transmit({"ticket": "amend", "order_id": "o1"})
    drop.transmit({"ticket": "cancel", "order_id": "o1"})
    assert (tmp_path / "outbound" / "cancel-o1.json").is_file()
    inbound = tmp_path / "inbound"
    inbound.mkdir()
    (inbound / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        drop.poll()
    (inbound / "bad.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        drop.poll()


def _broker(transport):
    return AlpacaBroker("alpaca-paper", ("key-id", "secret"), transport=transport)


def _order_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "ord-1",
        "client_order_id": "client-1",
        "status": "new",
        "filled_qty": "0",
        "symbol": "AAPL",
        "side": "buy",
    }
    payload.update(overrides)
    return payload


def test_alpaca_maps_malformed_account_cancel_and_fill_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY_ID", "bad\nkey")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    with pytest.raises(LiveGateError, match="malformed"):
        alpaca_credentials()
    with pytest.raises(LiveGateError, match="alpaca-paper or alpaca-live"):
        AlpacaBroker("loopback", ("k", "s"))  # type: ignore[arg-type]
    with pytest.raises(LiveBrokerRefused, match="alpaca-live is refused"):
        AlpacaBroker("alpaca-live", ("k", "s")).account()

    def transport(method, url, headers, body, timeout):
        del headers, body, timeout
        if method == "POST":
            return 200, json.dumps(_order_payload(status="not-a-status")).encode()
        if method == "DELETE":
            return 200, b"[]"
        if url.endswith("/v2/account"):
            return 200, b"[]"
        return 200, b"{}"

    broker = _broker(transport)
    ack = broker.submit(LiveOrder("client-1", "AAPL", "buy", 1.0, 10.0, 1))
    assert ack.status == "unmapped"
    assert "unmapped broker status" in ack.reason
    with pytest.raises(BrokerRejected, match="not an object"):
        broker.cancel("client-1", "ord-1")
    with pytest.raises(BrokerRejected, match="malformed"):
        broker.account()

    def cancel_transport(method, url, headers, body, timeout):
        del headers, body, timeout
        if "by_client_order_id" in url:
            return 404, b""
        if method == "DELETE":
            return 204, b""
        return 500, b""

    found = _broker(cancel_transport).cancel("client-1", "ord-1")
    assert found.status == "canceled"
    assert found.broker_order_id == "ord-1"

    def recover_transport(method, url, headers, body, timeout):
        del method, headers, body, timeout
        if "by_client_order_id" in url:
            return 200, json.dumps(
                _order_payload(
                    id="ord-9",
                    status="filled",
                    filled_qty="1",
                    filled_avg_price="10",
                    filled_at="2024-01-02T15:00:00+00:00",
                )
            ).encode()
        return 204, b""

    recovered = _broker(recover_transport).cancel("client-1", "")
    assert recovered.status == "filled"
    assert recovered.fills[0].fill_id.startswith("order:")

    with pytest.raises(BrokerRejected, match="not an object"):
        _broker(
            lambda *args: (200, b"[]") if "by_client_order_id" in args[1] else (200, b"{}")
        ).recover("client-1")

    def account_transport(method, url, headers, body, timeout):
        del method, headers, body, timeout
        if url.endswith("/v2/account"):
            return 200, json.dumps({"cash": "100", "equity": "100"}).encode()
        rows = [
            {"symbol": "", "qty": "1", "current_price": "5"},
            {"symbol": "MSFT", "qty": "0", "market_value": "0"},
            {"symbol": "AAPL", "qty": "2", "current_price": "0", "market_value": "20"},
            {"qty": "1", "current_price": None, "market_value": None},
        ]
        return 200, json.dumps(rows).encode()

    account = _broker(account_transport).account()
    assert account.marks["AAPL"] == pytest.approx(10.0)
    assert "" not in account.positions
    assert account.positions["MSFT"] == 0.0

    with pytest.raises(BrokerRejected, match="position row"):
        _broker(
            lambda method, url, headers, body, timeout: (
                (200, json.dumps({"cash": "1", "equity": "1"}).encode())
                if url.endswith("/v2/account")
                else (200, b"[1]")
            )
        ).account()

    def fills_transport(method, url, headers, body, timeout):
        del method, url, headers, body, timeout
        rows: list[object] = ["nope"]
        rows.append(
            {
                "id": "",
                "order_id": "",
                "side": "buy",
                "qty": "1",
                "price": "1",
                "symbol": "AAPL",
                "transaction_time": "2024-01-02T15:00:00Z",
            }
        )
        rows.append(
            {
                "id": "f-bad-side",
                "order_id": "b1",
                "side": "short",
                "qty": "1",
                "price": "1",
                "transaction_time": "2024-01-02T15:00:00Z",
            }
        )
        rows.append(
            {
                "id": "f-no-time",
                "order_id": "b1",
                "side": "sell",
                "qty": "1",
                "price": "1",
                "transaction_time": 5,
            }
        )
        for index in range(100):
            rows.append(
                {
                    "id": f"f{index}",
                    "order_id": "b1",
                    "side": "buy",
                    "qty": "1",
                    "price": "2",
                    "symbol": "AAPL",
                    "transaction_time": "2024-01-02T15:00:00Z",
                }
            )
        return 200, json.dumps(rows).encode()

    page = _broker(fills_transport).fills(broker_to_client={"b1": "client-1"})
    assert page.truncated is True
    assert page.fills
    assert all(item.side == "buy" for item in page.fills)
    with pytest.raises(BrokerRejected, match="not a list"):
        _broker(lambda *args: (200, b"{}")).fills(broker_to_client={})
    with pytest.raises(BrokerRejected, match="non-JSON"):
        _broker(lambda *args: (200, b"not-json")).submit(
            LiveOrder("client-1", "AAPL", "buy", 1.0, 10.0, 1)
        )
    with pytest.raises(BrokerRejected, match="HTTP 400"):
        _broker(lambda *args: (400, b"[]")).submit(LiveOrder("c", "AAPL", "buy", 1.0, 1.0, 1))
    with pytest.raises(BrokerRejected, match="redacted"):
        _broker(lambda *args: (400, json.dumps({"message": "secret leaked"}).encode())).submit(
            LiveOrder("c", "AAPL", "buy", 1.0, 1.0, 1)
        )
    with pytest.raises(BrokerRejected, match="HTTP 500"):
        _broker(lambda *args: (500, b"")).submit(LiveOrder("c", "AAPL", "buy", 1.0, 1.0, 1))

    guarded = _broker(lambda *args: (200, b"{}"))

    def missing(*args, **kwargs):
        del args, kwargs
        return None, True

    guarded._exchange = missing  # type: ignore[method-assign]
    with pytest.raises(BrokerRejected, match="not found"):
        guarded._request("GET", "/v2/orders", None)


class _Sequence:
    """First account() capture is healthy. The next read is the fault."""

    def __init__(self, accounts: list[BrokerAccount]) -> None:
        self._accounts = accounts
        self.marks: dict[str, float] = {}

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def account(self) -> BrokerAccount:
        if len(self._accounts) > 1:
            return self._accounts.pop(0)
        return self._accounts[0]

    def submit(self, order: LiveOrder) -> BrokerAck:
        del order
        raise AssertionError("submit is past the account check")

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        raise AssertionError("cancel was not expected")

    def recover(self, client_order_id: str) -> BrokerAck | None:
        del client_order_id
        return None

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        del broker_to_client
        return FillPage(fills=(), truncated=False)


class _Routed:
    def __init__(self, account: BrokerAccount, ack: BrokerAck | None = None) -> None:
        self._account = account
        self.ack = ack
        self.page = FillPage(fills=(), truncated=False)
        self.marks: dict[str, float] = {}

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def account(self) -> BrokerAccount:
        return self._account

    def submit(self, order: LiveOrder) -> BrokerAck:
        if self.ack is not None:
            return self.ack
        return BrokerAck(order.client_order_id, "b1", "accepted", 0.0, None, (), "")

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        return BrokerAck(client_order_id, broker_order_id, "canceled", 0.0, None, (), "")

    def recover(self, client_order_id: str) -> BrokerAck | None:
        del client_order_id
        return None

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        del broker_to_client
        return self.page


def _session(tmp: Path, broker: _Routed):
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "alpaca-paper",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(tmp),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    session = open_session(config, PRETRADE, state_dir=tmp, now_ns=lambda: TS, broker=broker)
    session.arm(actor="ada")
    return session


def _good_account(**overrides: object) -> BrokerAccount:
    payload = {"cash": 1_000_000.0, "equity": 1_000_000.0, "positions": {}, "marks": {}}
    payload.update(overrides)
    return BrokerAccount(**payload)  # type: ignore[arg-type]


def test_session_routing_refuses_bad_account_and_projects_partial_fills(tmp_path: Path) -> None:
    negative = _Sequence([_good_account(), _good_account(cash=-1.0)])
    session = _session(tmp_path / "cash", negative)
    with pytest.raises(LiveGateError, match="cash"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1,
            limit_price=10,
            mark=10,
            actor="ada",
            at_ns=TS,
            client_order_id="cash-1",
        )
    broke = _Sequence([_good_account(), _good_account(equity=0.0)])
    session = _session(tmp_path / "equity", broke)
    with pytest.raises(LiveGateError, match="equity"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1,
            limit_price=10,
            mark=10,
            actor="ada",
            at_ns=TS,
            client_order_id="eq-1",
        )
    unmarked = _Sequence([_good_account(), _good_account(positions={"IBM": 1.0}, marks={})])
    session = _session(tmp_path / "mark", unmarked)
    with pytest.raises(LiveGateError, match="no mark"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1,
            limit_price=10,
            mark=10,
            actor="ada",
            at_ns=TS,
            client_order_id="mark-1",
        )

    marked = _Routed(
        _good_account(positions={"AAPL": 5.0}, marks={"AAPL": 10.0}),
        BrokerAck("sell-1", "b-sell", "accepted", 0.0, None, (), ""),
    )
    session = _session(tmp_path / "sell", marked)
    sold = session.submit(
        symbol="AAPL",
        side="sell",
        quantity=1,
        limit_price=10,
        mark=10,
        actor="ada",
        at_ns=TS,
        client_order_id="sell-1",
    )
    assert sold["status"] == "accepted"
    exposure, reserved = session._working_exposure()
    assert exposure["AAPL"] < 0
    assert reserved == 0.0

    unmapped = _Routed(
        _good_account(),
        BrokerAck("unmap-1", "b-u", "not-a-status", 0.0, None, (), ""),
    )
    session = _session(tmp_path / "unmap", unmapped)
    stored = session.submit(
        symbol="AAPL",
        side="buy",
        quantity=1,
        limit_price=10,
        mark=10,
        actor="ada",
        at_ns=TS,
        client_order_id="unmap-1",
    )
    assert stored["status"] == "unmapped"
    assert "unmapped" in str(stored["reason"])

    bad_fill = BrokerFill("f-bad", "side-1", "AAPL", "short", 1.0, 10.0, TS)
    sided = _Routed(
        _good_account(),
        BrokerAck("side-1", "b-side", "filled", 1.0, 10.0, (bad_fill,), ""),
    )
    session = _session(tmp_path / "side", sided)
    with pytest.raises(LiveGateError, match="fill side"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1,
            limit_price=10,
            mark=10,
            actor="ada",
            at_ns=TS,
            client_order_id="side-1",
        )

    partial = BrokerFill("f-part", "part-1", "AAPL", "buy", 1.0, 10.0, TS)
    later = BrokerFill("f-part", "part-1", "AAPL", "buy", 1.0, 10.0, TS)
    routed = _Routed(_good_account(positions={"AAPL": 1.0}, marks={"AAPL": 10.0}))
    routed.page = FillPage(fills=(partial, later), truncated=True)
    session = _session(tmp_path / "part", routed)
    accepted = session.submit(
        symbol="AAPL",
        side="buy",
        quantity=2,
        limit_price=10,
        mark=10,
        actor="ada",
        at_ns=TS,
        client_order_id="part-1",
    )
    assert accepted["status"] == "accepted"
    report = session.reconcile(at_ns=TS + 1)
    assert report["fills_truncated"] is True
    assert report["duplicate_in_batch"] is True
    assert report["ok"] is False
    assert report["live_pnl_claim"] is False
    assert session.orders[0]["status"] == "partially_filled"

    with pytest.raises(LiveGateError, match="actor"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1,
            limit_price=10,
            mark=10,
            actor=" ",
            at_ns=TS,
            client_order_id="actor-1",
        )
    with pytest.raises(LiveGateError, match="symbol"):
        session.submit(
            symbol="bad symbol",
            side="buy",
            quantity=1,
            limit_price=10,
            mark=10,
            actor="ada",
            at_ns=TS,
            client_order_id="sym-1",
        )
    with pytest.raises(LiveGateError, match="non-negative"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1,
            limit_price=10,
            mark=10,
            actor="ada",
            at_ns=-1,
            client_order_id="time-1",
        )
