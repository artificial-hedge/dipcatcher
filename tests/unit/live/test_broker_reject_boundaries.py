"""Loopback reject boundaries, HTTP 400, and the paper timeout default.

A sell of exactly the held quantity plus the 1e-9 slack still fills, and a
buy that spends exactly cash plus that slack still fills. HTTP 400 is a
reject. The paper adapter's default timeout is 10 seconds. No live venue
is contacted.
"""

from __future__ import annotations

import pytest

from quant_fund.live.brokers import (
    AlpacaBroker,
    LiveOrder,
    LoopbackBroker,
    _parse_broker_response,
)
from quant_fund.live.errors import BrokerRejected


def _order(side: str, quantity: float, price: float, client_order_id: str = "c") -> LiveOrder:
    return LiveOrder(client_order_id, "AAPL", side, quantity, price, 1)


def test_sell_at_held_plus_slack_still_fills() -> None:
    held = 1.0
    broker = LoopbackBroker(cash=100.0, positions={"AAPL": held})
    ack = broker.submit(_order("sell", held + 1e-9, 10.0))
    assert ack.status == "filled"
    assert ack.reason == ""


def test_buy_at_cash_plus_slack_still_fills() -> None:
    cash = 100.0
    broker = LoopbackBroker(cash=cash)
    ack = broker.submit(_order("buy", 1.0, cash + 1e-9))
    assert ack.status == "filled"
    assert ack.reason == ""


def test_http_400_json_is_rejected() -> None:
    with pytest.raises(BrokerRejected, match="broker HTTP 400"):
        _parse_broker_response(
            400,
            b'{"message":"bad request"}',
            ("k", "s"),
            allow_missing=False,
        )


def test_paper_broker_uses_the_ten_second_timeout() -> None:
    seen: dict[str, float] = {}

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body
        seen["timeout"] = timeout
        return 404, b""

    broker = AlpacaBroker("alpaca-paper", ("k", "s"), transport=transport)
    assert broker.recover("missing") is None
    assert seen["timeout"] == 10.0
