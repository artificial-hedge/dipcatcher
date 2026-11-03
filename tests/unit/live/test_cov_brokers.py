"""Fail-closed branches of the broker gateways.

Every venue here is scripted in-process: the ``transport`` callable stands in
for the socket, so a disconnect is simulated by raising and a bad response by
returning bytes. Nothing opens a connection and no credentials are real.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from quant_fund.live.brokers import (
    AlpacaBroker,
    BrokerAccount,
    BrokerAck,
    FillPage,
    LiveOrder,
    LoopbackBroker,
    _broker_error,
    _decimal,
    _empty_response,
    _money,
    _ns,
    _parse_broker_response,
    _position_mark,
    _redact,
    alpaca_credentials,
    map_broker_status,
)
from quant_fund.live.errors import (
    BrokerRejected,
    BrokerUnconfirmed,
    LiveBrokerRefused,
    LiveGateError,
)

PAPER = "https://paper-api.alpaca.markets"


def order(**overrides: Any) -> LiveOrder:
    base: dict[str, Any] = {
        "client_order_id": "c1",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10.0,
        "limit_price": 100.0,
        "fill_time_ns": 1_000,
    }
    base.update(overrides)
    return LiveOrder(**base)


class Script:
    """Transport stand-in. Each call pops the next canned ``(status, body)``."""

    def __init__(self, *responses: tuple[int, bytes]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del timeout
        self.calls.append((method, url, headers, body))
        if not self.responses:
            return 404, b"{}"
        return self.responses.pop(0)


def paper(script: Script, *, timeout: float = 10.0) -> AlpacaBroker:
    return AlpacaBroker("alpaca-paper", ("key-id", "secret-key"), transport=script, timeout=timeout)


def ack_payload(**overrides: Any) -> bytes:
    base: dict[str, Any] = {
        "id": "ord-1",
        "client_order_id": "c1",
        "status": "accepted",
        "filled_qty": "0",
        "symbol": "AAPL",
        "side": "buy",
    }
    base.update(overrides)
    return json.dumps(base).encode()


# ------------------------------------------------------------------ loopback


def test_loopback_duplicate_submit_returns_the_first_ack_once() -> None:
    broker = LoopbackBroker(cash=10_000.0)
    first = broker.submit(order())
    second = broker.submit(order())
    assert first.status == "filled"
    assert second == first
    # Only one fill and one cash movement for a replayed client order id.
    assert len(broker.fills(broker_to_client={}).fills) == 1
    assert broker.cash == pytest.approx(9_000.0)


def test_loopback_rejects_an_unknown_side() -> None:
    broker = LoopbackBroker(cash=10_000.0)
    with pytest.raises(BrokerRejected, match="buy or sell"):
        broker.submit(order(side="short"))


def test_loopback_rejects_selling_more_than_held() -> None:
    broker = LoopbackBroker(cash=10_000.0, positions={"AAPL": 1.0})
    ack = broker.submit(order(side="sell", quantity=5.0))
    assert ack.status == "rejected"
    assert ack.reason == "insufficient position"
    assert ack.fills == ()
    # The rejection is remembered, so the retry is the same verdict, not a fill.
    assert broker.submit(order(side="sell", quantity=5.0)) == ack
    assert broker.cash == 10_000.0


def test_loopback_rejects_a_buy_beyond_cash() -> None:
    broker = LoopbackBroker(cash=50.0)
    ack = broker.submit(order(quantity=1.0, limit_price=100.0))
    assert ack.status == "rejected"
    assert ack.reason == "insufficient cash"
    assert broker.cash == 50.0
    assert broker.account().positions == {}


def test_loopback_deferred_order_cancels_and_keeps_prior_verdicts() -> None:
    broker = LoopbackBroker(cash=1_000.0)
    broker.defer_fill = True
    accepted = broker.submit(order())
    assert accepted.status == "accepted"
    assert broker.fills(broker_to_client={}).fills == ()
    canceled = broker.cancel("c1", "loopback:c1")
    assert canceled.status == "canceled"
    # A second cancel returns the stored verdict rather than re-canceling.
    assert broker.cancel("c1", "loopback:c1") == canceled


def test_loopback_cancel_of_a_filled_order_returns_the_fill() -> None:
    broker = LoopbackBroker(cash=1_000.0)
    filled = broker.submit(order(quantity=1.0, limit_price=10.0))
    assert broker.cancel("c1", "") is filled


def test_loopback_cancel_unknown_order_is_refused() -> None:
    broker = LoopbackBroker(cash=1_000.0)
    with pytest.raises(BrokerRejected, match="not at the loopback venue"):
        broker.cancel("nope", "")
    assert broker.recover("nope") is None


def test_loopback_account_marks_equity_and_skips_unmarked_names() -> None:
    broker = LoopbackBroker(cash=100.0, positions={"AAPL": 2.0, "MSFT": 1.0})
    broker.set_mark("AAPL", 10.0)
    account = broker.account()
    assert account.cash == pytest.approx(100.0)
    # MSFT has no mark, so it contributes nothing to equity.
    assert account.equity == pytest.approx(120.0)
    assert account.marks == {"AAPL": 10.0}
    assert account.positions == {"AAPL": 2.0, "MSFT": 1.0}


# ------------------------------------------------------------ credentials/env


def test_alpaca_credentials_absent_incomplete_and_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ALPACA_API_KEY_ID",
        "APCA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert alpaca_credentials() is None
    monkeypatch.setenv("ALPACA_API_KEY_ID", "key")
    with pytest.raises(LiveGateError, match="incomplete"):
        alpaca_credentials()
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    assert alpaca_credentials() == ("key", "secret")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k" * 201)
    with pytest.raises(LiveGateError, match="incomplete"):
        alpaca_credentials()
    monkeypatch.setenv("ALPACA_API_KEY_ID", "ke\ny")
    with pytest.raises(LiveGateError, match="malformed"):
        alpaca_credentials()
    # The APCA_* aliases are accepted when the ALPACA_* names are absent.
    monkeypatch.delenv("ALPACA_API_KEY_ID")
    monkeypatch.delenv("ALPACA_API_SECRET_KEY")
    monkeypatch.setenv("APCA_API_KEY_ID", "apca-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "apca-secret")
    assert alpaca_credentials() == ("apca-key", "apca-secret")


def test_alpaca_broker_refuses_a_venue_that_is_not_alpaca() -> None:
    with pytest.raises(LiveGateError, match="alpaca-paper or alpaca-live"):
        AlpacaBroker("nasdaq", None, transport=Script())  # type: ignore[arg-type]


def test_submit_without_credentials_never_reaches_the_transport() -> None:
    script = Script()
    broker = AlpacaBroker("alpaca-paper", None, transport=script)
    assert broker.credentials_present is False
    with pytest.raises(LiveGateError, match="credentials are not set"):
        broker.submit(order())
    assert script.calls == []


def test_live_venue_is_refused_before_the_transport() -> None:
    script = Script()
    broker = AlpacaBroker("alpaca-live", ("key", "secret"), transport=script)
    with pytest.raises(LiveBrokerRefused, match="honesty contract"):
        broker.submit(order())
    assert script.calls == []


def test_a_disconnect_propagates_instead_of_being_read_as_a_rejection() -> None:
    """A dropped socket must not look like a broker refusal (that would retry blind)."""

    def explode(*_args: Any, **_kwargs: Any) -> tuple[int, bytes]:
        raise ConnectionResetError("peer reset")

    broker = AlpacaBroker("alpaca-paper", ("key", "secret"), transport=explode)
    with pytest.raises(BrokerUnconfirmed, match="unconfirmed after a transport failure"):
        broker.submit(order())
    with pytest.raises(ConnectionResetError):
        broker.account()
    with pytest.raises(ConnectionResetError):
        broker.fills(broker_to_client={})


# ------------------------------------------------------------------- submit


def test_submit_recovers_an_order_it_already_sent() -> None:
    script = Script(
        (400, json.dumps({"message": "client_order_id must be unique"}).encode()),
        (200, ack_payload()),
    )
    broker = paper(script)
    ack = broker.submit(order())
    assert ack.broker_order_id == "ord-1"
    assert ack.status == "accepted"
    assert script.calls[0][0] == "POST"
    assert script.calls[1][0] == "GET"
    assert "client_order_id=c1" in script.calls[1][1]


def test_submit_duplicate_id_without_a_recoverable_order_still_fails() -> None:
    script = Script(
        (400, json.dumps({"message": "client_order_id must be unique"}).encode()),
        (404, b"{}"),
    )
    with pytest.raises(BrokerUnconfirmed, match="unconfirmed after an ambiguous rejection"):
        paper(script).submit(order())


def test_submit_rejection_without_a_client_id_pointer_is_not_retried() -> None:
    script = Script((422, json.dumps({"message": "insufficient buying power"}).encode()))
    with pytest.raises(BrokerRejected, match="insufficient buying power"):
        paper(script).submit(order())
    # Only the POST happened: no lookup for an order the broker never took.
    assert [call[0] for call in script.calls] == ["POST"]


def test_submit_rejects_a_non_object_order_response() -> None:
    script = Script((200, b"[]"))
    with pytest.raises(BrokerUnconfirmed, match="unconfirmed after an incomplete acknowledgement"):
        paper(script).submit(order())


def test_submit_reports_an_unmapped_status_and_creates_no_fill() -> None:
    script = Script((200, ack_payload(status="awaiting_custodian")))
    ack = paper(script).submit(order())
    assert ack.status == "unmapped"
    assert ack.reason == "unmapped broker status awaiting_custodian"
    assert ack.fills == ()


# ------------------------------------------------------------------- cancel


def test_cancel_without_a_broker_id_looks_the_order_up() -> None:
    script = Script((200, ack_payload()), (200, ack_payload(status="canceled")))
    ack = paper(script).cancel("c1", "")
    assert ack.status == "canceled"
    assert ack.broker_order_id == "ord-1"
    assert script.calls[0][0] == "GET"
    assert script.calls[1][0] == "DELETE"
    assert script.calls[1][1].endswith("/v2/orders/ord-1")


def test_cancel_unknown_order_is_refused() -> None:
    with pytest.raises(BrokerRejected, match="order is not at the broker"):
        paper(Script((404, b"{}"))).cancel("c1", "")


def test_cancel_empty_body_synthesizes_a_canceled_ack() -> None:
    script = Script((204, b""), (404, b"{}"))
    ack = paper(script).cancel("c1", "ord-9")
    assert ack.status == "canceled"
    assert ack.broker_order_id == "ord-9"
    assert ack.filled_quantity == 0.0
    assert ack.fills == ()


def test_cancel_empty_body_returns_the_recovered_verdict() -> None:
    script = Script((204, b""), (200, ack_payload(status="filled", filled_qty="10")))
    ack = paper(script).cancel("c1", "ord-9")
    assert ack.status == "filled"
    assert ack.filled_quantity == pytest.approx(10.0)


def test_cancel_rejects_a_non_object_response() -> None:
    script = Script((200, b"[]"))
    with pytest.raises(BrokerRejected, match="cancel response was not an object"):
        paper(script).cancel("c1", "ord-1")


# ------------------------------------------------------------------ recover


def test_recover_returns_none_when_the_broker_has_no_such_order() -> None:
    assert paper(Script((404, b"{}"))).recover("c1") is None


def test_recover_rejects_a_non_object_lookup() -> None:
    with pytest.raises(BrokerRejected, match="broker order lookup was incomplete"):
        paper(Script((200, b"[]"))).recover("c1")


# ------------------------------------------------------------------ account


def test_account_rejects_malformed_books() -> None:
    with pytest.raises(BrokerRejected, match="account response was malformed"):
        paper(Script((200, b"[]"), (200, b"[]"))).account()
    with pytest.raises(BrokerRejected, match="account response was malformed"):
        paper(Script((200, b'{"cash":"1","equity":"1"}'), (200, b'{"not":"a list"}'))).account()
    with pytest.raises(BrokerRejected, match="position row was malformed"):
        paper(Script((200, b'{"cash":"1","equity":"1"}'), (200, b'["AAPL"]'))).account()


def test_account_parses_marks_and_drops_unnamed_rows() -> None:
    positions = json.dumps(
        [
            {"symbol": "AAPL", "qty": "2", "current_price": "10"},
            {"symbol": "MSFT", "qty": "1", "market_value": "-50"},
            {"symbol": "", "qty": "1", "current_price": "3"},
            {"symbol": "ZERO", "qty": "0", "market_value": "0"},
        ]
    ).encode()
    broker = paper(Script((200, b'{"cash":"100","equity":"150"}'), (200, positions)))
    account = broker.account()
    assert account.cash == pytest.approx(100.0)
    assert account.equity == pytest.approx(150.0)
    assert account.positions == {"AAPL": 2.0, "MSFT": 1.0, "ZERO": 0.0}
    # AAPL keeps the broker price; MSFT divides the market value by the qty;
    # ZERO has no usable mark and "" is dropped from both books.
    assert account.marks == {"AAPL": 10.0, "MSFT": 50.0}


def test_account_rejects_a_non_numeric_quantity() -> None:
    positions = json.dumps([{"symbol": "AAPL", "qty": "many"}]).encode()
    with pytest.raises(BrokerRejected, match="not a number"):
        paper(Script((200, b'{"cash":"1","equity":"1"}'), (200, positions))).account()


# -------------------------------------------------------------------- fills


def test_fills_rejects_a_non_list_page() -> None:
    with pytest.raises(BrokerRejected, match="fill list was not a list"):
        paper(Script((200, b"{}"))).fills(broker_to_client={})


def test_fills_skips_rows_it_cannot_use() -> None:
    rows = [
        "not-a-row",
        {"id": "f1", "order_id": "b1", "side": "hold", "qty": "1", "price": "1"},
        {"id": "f2", "order_id": "", "side": "buy", "qty": "1", "price": "1"},
        {"id": "", "order_id": "b1", "side": "buy", "qty": "1", "price": "1"},
        {
            "id": "f3",
            "order_id": "b1",
            "side": "buy",
            "qty": "1",
            "price": "1",
            "transaction_time": 5,
        },
    ]
    script = Script((200, json.dumps(rows).encode()))
    with pytest.raises(BrokerRejected, match="broker fill row was malformed"):
        paper(script).fills(broker_to_client={})


def test_fills_records_mapped_rows_and_marks_a_full_page() -> None:
    rows = [
        {
            "id": f"f{index:03d}",
            "order_id": "broker-1",
            "side": "buy",
            "qty": "1",
            "price": "10",
            "transaction_time": "2024-01-03T20:00:00Z",
        }
        for index in range(100)
    ]
    page = paper(Script((200, json.dumps(rows).encode()))).fills(
        broker_to_client={"broker-1": "c1"}
    )
    assert len(page.fills) == 100
    assert page.truncated is True
    assert page.fills[0].client_order_id == "c1"
    assert page.fills[0].fill_time_ns == 1_704_312_000_000_000_000


def test_fills_rejects_a_timestamp_without_a_timezone() -> None:
    rows = [
        {
            "id": "f1",
            "order_id": "b1",
            "side": "buy",
            "qty": "1",
            "price": "10",
            "transaction_time": "2024-01-03T20:00:00",
        }
    ]
    with pytest.raises(BrokerRejected, match="no timezone"):
        paper(Script((200, json.dumps(rows).encode()))).fills(broker_to_client={})


# ----------------------------------------------------------- http/error text


def test_empty_body_200_is_an_empty_object_and_204_is_too() -> None:
    assert _empty_response(200) == ({}, False)
    assert _empty_response(204) == ({}, False)
    with pytest.raises(BrokerRejected, match="broker HTTP 400"):
        _empty_response(400)


def test_missing_order_on_a_required_call_is_reported() -> None:
    script = Script((404, b""))
    with pytest.raises(BrokerRejected, match="broker HTTP 404"):
        paper(script).account()


def test_non_json_error_body_does_not_echo_the_body() -> None:
    with pytest.raises(BrokerRejected, match="returned non-JSON") as excinfo:
        _parse_broker_response(500, b"<html>secret-key</html>", ("k", "s"), allow_missing=False)
    assert "secret-key" not in str(excinfo.value)


def test_optional_404_is_missing_and_not_an_error() -> None:
    assert _parse_broker_response(404, b"", ("k", "s"), allow_missing=True) == (None, True)
    with pytest.raises(BrokerRejected, match="broker HTTP 404"):
        _parse_broker_response(404, b"", ("k", "s"), allow_missing=False)


def test_broker_error_takes_a_message_or_an_error_key() -> None:
    assert _broker_error(422, {"message": "nope"}, ("k", "s")) == "broker HTTP 422: nope"
    assert _broker_error(422, {"error": "bad"}, ("k", "s")) == "broker HTTP 422: bad"
    assert _broker_error(422, ["ignored"], ("k", "s")) == "broker HTTP 422"
    assert _broker_error(422, {"message": ""}, ("k", "s")) == "broker HTTP 422"


def test_redaction_bounds_the_message_and_removes_every_secret() -> None:
    assert _redact("k and s", ("k", "s", "")) == "[redacted] and [redacted]"
    assert len(_redact("x" * 900, ())) == 500


def test_money_and_timestamp_helpers_fail_closed() -> None:
    with pytest.raises(BrokerRejected, match="was not finite"):
        _money(float("nan"), "cash")
    with pytest.raises(BrokerRejected, match="was not a number"):
        _money(object(), "cash")
    assert _ns("2024-01-03T20:00:00+00:00") == 1_704_312_000_000_000_000


def test_decimal_trims_trailing_zeros_without_losing_precision() -> None:
    assert _decimal(10.0) == "10"
    assert _decimal(0.5) == "0.5"
    assert _decimal(1e-9) == "0.000000001"
    assert _decimal(0.0) == "0"


def test_position_mark_prefers_live_price_then_value_over_quantity() -> None:
    assert _position_mark({"current_price": "10", "avg_entry_price": "5"}) == 10.0
    assert _position_mark({"lastday_price": "7"}) == 7.0
    assert _position_mark({"market_value": "-50", "qty": "2"}) == 25.0
    assert _position_mark({"market_value": "50", "qty": "0"}) is None
    assert _position_mark({"market_value": "50"}) is None
    assert _position_mark({"current_price": ""}) is None


def test_map_broker_status_normalizes_and_keeps_unknown_unmapped() -> None:
    assert map_broker_status(" Filled ") == "filled"
    assert map_broker_status("cancelled") == "canceled"
    assert map_broker_status("mystery") == "unmapped"
    assert map_broker_status("") == "unmapped"


def test_ack_helpers_are_stable_dataclasses() -> None:
    ack = BrokerAck(
        client_order_id="c1",
        broker_order_id="b1",
        status="accepted",
        filled_quantity=0.0,
        fill_price=None,
        fills=(),
        reason="",
    )
    assert ack.status == "accepted"
    account = BrokerAccount(cash=1.0, equity=1.0, positions={})
    assert account.marks == {}
    assert FillPage(fills=(), truncated=False).truncated is False
