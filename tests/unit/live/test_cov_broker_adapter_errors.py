"""Broker-adapter error handling: fail closed instead of guessing.

Every branch here refuses to turn an unexpected broker response into an
accepted order. Credentials never travel without a venue host, a lost POST is
only recovered when the broker named the client order id, malformed account
and fill payloads raise, and the loopback venue rejects impossible fills.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from quant_fund.live.brokers import (
    AlpacaBroker,
    LiveOrder,
    LoopbackBroker,
    _money,
    _ns,
    _parse_broker_response,
    map_broker_status,
)
from quant_fund.live.config import Venue
from quant_fund.live.errors import BrokerRejected, LiveGateError

CREDS = ("unit-key", "unit-secret")


def _order(
    *,
    client: str = "c-1",
    symbol: str = "AAA",
    side: str = "buy",
    quantity: float = 1.0,
    limit_price: float = 10.0,
) -> LiveOrder:
    return LiveOrder(client, symbol, side, quantity, limit_price, 1)


def _broker(responses: dict[str, tuple[int, Any]]) -> tuple[AlpacaBroker, list[str]]:
    """Transport stub keyed by HTTP method; records the methods it saw."""
    seen: list[str] = []

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, timeout
        seen.append(method)
        status, payload = responses[method]
        if isinstance(payload, bytes):
            return status, payload
        return status, json.dumps(payload).encode("utf-8")

    return AlpacaBroker("alpaca-paper", CREDS, transport=transport), seen


def test_non_alpaca_venue_is_refused_at_construction() -> None:
    with pytest.raises(LiveGateError, match="must be alpaca-paper or alpaca-live"):
        AlpacaBroker(cast(Venue, "loopback"), CREDS)


def test_missing_credentials_refuse_before_any_transport_call() -> None:
    broker, seen = _broker({"POST": (200, {})})
    broker = AlpacaBroker("alpaca-paper", None, transport=broker._transport)
    with pytest.raises(LiveGateError, match="credentials are not set"):
        broker.submit(_order())
    assert seen == []


def test_a_path_outside_the_venue_host_is_refused() -> None:
    broker, _ = _broker({})
    with pytest.raises(LiveGateError, match="outside the venue host"):
        broker._exchange("GET", "../v2/orders", None, allow_missing=False)


@pytest.mark.parametrize("value", ["not-a-number", None, "nan", "inf"])
def test_non_numeric_and_non_finite_money_fail_closed(value: object) -> None:
    with pytest.raises(BrokerRejected, match="broker qty"):
        _money(value, "qty")


def test_naive_fill_time_is_rejected() -> None:
    with pytest.raises(BrokerRejected, match="had no timezone"):
        _ns("2024-01-03T15:00:00")


def test_unknown_status_never_becomes_accepted() -> None:
    assert map_broker_status("new") == "accepted"
    assert map_broker_status(" SUSPENDED ") == "suspended"
    assert map_broker_status("something-new") == "unmapped"


def test_account_rejects_malformed_positions() -> None:
    # Both calls return the same mapping; the position call sees a non-list.
    broker, _ = _broker({"GET": (200, {"cash": "1", "equity": "2"})})
    with pytest.raises(BrokerRejected, match="account response was malformed"):
        broker.account()


def test_account_rejects_a_non_object_position_row() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        payload: Any = {"cash": "10", "equity": "10"} if url.endswith("/account") else ["nope"]
        return 200, json.dumps(payload).encode("utf-8")

    broker = AlpacaBroker("alpaca-paper", CREDS, transport=transport)
    with pytest.raises(BrokerRejected, match="position row was malformed"):
        broker.account()


def test_account_marks_come_from_the_broker_and_skip_empty_rows() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        positions = [{"symbol": "AAA", "qty": "1"}, "skip-me"]
        payload: Any = {"cash": "10", "equity": "10"} if url.endswith("/account") else positions
        return 200, json.dumps(payload).encode("utf-8")

    broker = AlpacaBroker("alpaca-paper", CREDS, transport=transport)
    with pytest.raises(BrokerRejected, match="position row was malformed"):
        broker.account()

    def clean_transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        positions = [
            {"symbol": "AAA", "qty": "2", "current_price": "11.5"},
            {"symbol": "BBB", "qty": "4", "market_value": "20"},
            {"symbol": "CCC", "qty": "0", "market_value": "5"},
            {"symbol": "", "qty": "1"},
        ]
        payload: Any = {"cash": "10", "equity": "10"} if url.endswith("/account") else positions
        return 200, json.dumps(payload).encode("utf-8")

    account = AlpacaBroker("alpaca-paper", CREDS, transport=clean_transport).account()
    assert account.positions == {"AAA": 2.0, "BBB": 4.0, "CCC": 0.0}
    assert account.marks == {"AAA": 11.5, "BBB": 5.0}


def test_a_non_numeric_position_mark_fails_the_whole_account() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        payload: Any = (
            {"cash": "10", "equity": "10"}
            if url.endswith("/account")
            else [{"symbol": "AAA", "qty": "1", "current_price": "not-a-price"}]
        )
        return 200, json.dumps(payload).encode("utf-8")

    with pytest.raises(BrokerRejected, match="position mark was not a number"):
        AlpacaBroker("alpaca-paper", CREDS, transport=transport).account()


def test_fill_list_must_be_a_list() -> None:
    broker, _ = _broker({"GET": (200, {"fills": []})})
    with pytest.raises(BrokerRejected, match="fill list was not a list"):
        broker.fills(broker_to_client={})


def test_unusable_fill_rows_are_skipped_not_coerced() -> None:
    rows = [
        "not-an-object",
        {
            "order_id": "o-1",
            "side": "hold",
            "id": "f-1",
            "transaction_time": "2024-01-03T15:00:00Z",
        },
        {"order_id": "", "side": "buy", "id": "f-2", "transaction_time": "2024-01-03T15:00:00Z"},
        {"order_id": "o-2", "side": "buy", "id": "", "transaction_time": "2024-01-03T15:00:00Z"},
        {"order_id": "o-3", "side": "buy", "id": "f-3", "transaction_time": 12345},
        {
            "order_id": "o-4",
            "side": "sell",
            "id": "f-4",
            "transaction_time": "2024-01-03T15:00:00Z",
            "symbol": "AAA",
            "qty": "1",
            "price": "9.5",
        },
    ]
    broker, _ = _broker({"GET": (200, rows)})
    page = broker.fills(broker_to_client={"o-4": "client-4"})
    assert len(page.fills) == 1
    assert page.fills[0].client_order_id == "client-4"
    assert page.fills[0].fill_time_ns == _ns("2024-01-03T15:00:00Z")
    assert page.truncated is False


def test_a_full_page_is_reported_as_truncated() -> None:
    row = {
        "order_id": "o-1",
        "side": "buy",
        "id": "f",
        "transaction_time": "2024-01-03T15:00:00Z",
        "symbol": "AAA",
        "qty": "1",
        "price": "10",
    }
    broker, _ = _broker({"GET": (200, [{**row, "id": f"f-{i}"} for i in range(100)])})
    page = broker.fills(broker_to_client={"o-1": "client-1"})
    assert len(page.fills) == 100
    assert page.truncated is True


def test_a_lost_post_is_recovered_only_when_the_broker_names_the_client_id() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, body, timeout
        if method == "POST":
            return 422, json.dumps({"message": "client_order_id must be unique"}).encode()
        return (
            200,
            json.dumps(
                {
                    "id": "ord-1",
                    "client_order_id": "c-1",
                    "status": "new",
                    "filled_qty": "0",
                    "symbol": "AAA",
                    "side": "buy",
                }
            ).encode(),
        )

    ack = AlpacaBroker("alpaca-paper", CREDS, transport=transport).submit(_order())
    assert ack.status == "accepted"
    assert ack.broker_order_id == "ord-1"


def test_an_unrecoverable_post_is_re_raised() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, body, timeout
        if method == "POST":
            return 422, json.dumps({"message": "client_order_id must be unique"}).encode()
        return 404, b""

    with pytest.raises(BrokerRejected, match="client_order_id"):
        AlpacaBroker("alpaca-paper", CREDS, transport=transport).submit(_order())


def test_an_unrelated_rejection_is_not_retried() -> None:
    broker, seen = _broker({"POST": (403, {"message": "forbidden"})})
    with pytest.raises(BrokerRejected, match="HTTP 403"):
        broker.submit(_order())
    assert seen == ["POST"]


def test_cancel_without_a_broker_id_uses_lookup() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, body, timeout
        if method == "GET":
            return (
                200,
                json.dumps(
                    {
                        "id": "ord-9",
                        "client_order_id": "c-1",
                        "status": "new",
                        "filled_qty": "0",
                        "symbol": "AAA",
                        "side": "buy",
                    }
                ).encode(),
            )
        return 204, b""

    ack = AlpacaBroker("alpaca-paper", CREDS, transport=transport).cancel("c-1", "")
    assert ack.broker_order_id == "ord-9"
    assert ack.status == "accepted"


def test_cancel_of_an_unknown_order_is_refused() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body, timeout
        return 404, b""

    with pytest.raises(BrokerRejected, match="order is not at the broker"):
        AlpacaBroker("alpaca-paper", CREDS, transport=transport).cancel("c-1", "")


def test_an_empty_delete_response_is_a_cancel_without_a_recovered_order() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        if url.endswith("/v2/orders/ord-9"):
            return 204, b""
        return 404, b""

    ack = AlpacaBroker("alpaca-paper", CREDS, transport=transport).cancel("c-1", "ord-9")
    assert ack.status == "canceled"
    assert ack.filled_quantity == 0.0


def test_a_non_object_lookup_response_is_refused() -> None:
    broker, _ = _broker({"GET": (200, ["not-an-object"])})
    with pytest.raises(BrokerRejected, match="order lookup was not an object"):
        broker.recover("c-1")


def test_a_non_object_cancel_response_is_refused() -> None:
    broker, _ = _broker({"DELETE": (200, ["not-an-object"])})
    with pytest.raises(BrokerRejected, match="cancel response was not an object"):
        broker.cancel("c-1", "ord-1")


def test_a_2xx_empty_body_parses_to_an_empty_object() -> None:
    assert _parse_broker_response(204, b"", CREDS, allow_missing=False) == ({}, False)


def test_loopback_rejects_impossible_orders() -> None:
    broker = LoopbackBroker(cash=100.0, positions={"AAA": 1.0})
    with pytest.raises(BrokerRejected, match="side must be buy or sell"):
        broker.submit(_order(side="hold"))
    with pytest.raises(BrokerRejected, match="order is not at the loopback venue"):
        broker.cancel("missing", "missing")

    assert broker.submit(_order(client="too-big", quantity=1_000.0)).status == "rejected"
    assert broker.submit(_order(client="too-many", side="sell", quantity=5.0)).status == "rejected"
    broker.set_mark("AAA", 10.0)
    account = broker.account()
    assert account.equity == pytest.approx(110.0)


def test_loopback_is_idempotent_on_client_order_id() -> None:
    broker = LoopbackBroker(cash=100.0)
    first = broker.submit(_order(client="repeat"))
    second = broker.submit(_order(client="repeat"))
    assert first == second
    assert broker.recover("repeat") == first
    assert broker.recover("never-seen") is None
    assert broker.cancel("repeat", "loopback:repeat").status == "filled"
    assert len(broker.fills(broker_to_client={}).fills) == 1
