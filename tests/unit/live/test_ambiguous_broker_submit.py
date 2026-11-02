"""Ambiguous broker acks halt. A retry reconciles and does not submit again."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.brokers import AlpacaBroker, LiveOrder
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import BrokerRejected, BrokerUnconfirmed, LiveGateError
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def _order(client: str = "client-1") -> LiveOrder:
    return LiveOrder(client, "SYNTH", "buy", 1.0, 10.0, 1)


def _accepted(client: str, *, status: str = "accepted", order_id: str = "ord-1") -> dict[str, str]:
    return {
        "id": order_id,
        "client_order_id": client,
        "status": status,
        "filled_qty": "0",
        "symbol": "SYNTH",
        "side": "buy",
    }


class _Script:
    def __init__(self, steps: list[Any]) -> None:
        self.steps = list(steps)
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del headers, body, timeout
        self.calls.append((method, url))
        if not self.steps:
            raise AssertionError(f"unexpected {method} {url}")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        status, payload = step
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return int(status), raw


def _broker(script: _Script) -> AlpacaBroker:
    return AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=script)


def _posts(script: _Script) -> list[tuple[str, str]]:
    return [call for call in script.calls if call[0] == "POST"]


def test_timeout_reconciles_and_a_retry_does_not_post_again() -> None:
    found = _accepted("client-1")
    script = _Script([TimeoutError("timed out"), (200, found), (200, found)])
    broker = _broker(script)

    ack = broker.submit(_order())
    assert ack.broker_order_id == "ord-1"
    assert ack.status == "accepted"
    assert len(_posts(script)) == 1

    again = broker.submit(_order())
    assert again.broker_order_id == "ord-1"
    assert len(_posts(script)) == 1
    assert any("by_client_order_id" in url for _method, url in script.calls)


def test_timeout_with_no_order_halts_and_is_not_submitted_again() -> None:
    script = _Script([TimeoutError("timed out"), (404, b""), (404, b"")])
    broker = _broker(script)
    order = _order()

    with pytest.raises(BrokerUnconfirmed, match="transport failure") as first:
        broker.submit(order)
    message = str(first.value)
    assert "client-1" in message
    assert "could not be confirmed" in message
    assert "not submitted again" in message
    assert len(_posts(script)) == 1

    with pytest.raises(BrokerUnconfirmed, match="a retry"):
        broker.submit(order)
    assert len(_posts(script)) == 1


def test_lookup_failure_after_timeout_halts_without_a_second_post() -> None:
    script = _Script([TimeoutError("timed out"), TimeoutError("lookup timed out")])
    broker = _broker(script)

    with pytest.raises(BrokerUnconfirmed, match="transport failure") as caught:
        broker.submit(_order())
    assert caught.value.__cause__ is not None
    assert len(_posts(script)) == 1


def test_incomplete_ack_reconciles_before_it_is_trusted() -> None:
    partial = {"client_order_id": "client-1", "status": "accepted", "filled_qty": "1"}
    script = _Script([(200, partial), (200, _accepted("client-1", status="partially_filled"))])
    broker = _broker(script)

    ack = broker.submit(_order())
    assert ack.status == "partially_filled"
    assert len(_posts(script)) == 1
    assert script.calls[1][0] == "GET"


def test_incomplete_ack_with_no_order_halts() -> None:
    partial = {"status": "accepted"}
    script = _Script([(200, partial), (404, b""), (404, b"")])
    broker = _broker(script)

    with pytest.raises(BrokerUnconfirmed, match="incomplete acknowledgement"):
        broker.submit(_order())
    with pytest.raises(BrokerUnconfirmed, match="a retry"):
        broker.submit(_order())
    assert len(_posts(script)) == 1


def test_ambiguous_http_error_reconciles_instead_of_posting_again() -> None:
    script = _Script(
        [
            (503, {"message": "unavailable"}),
            (200, _accepted("client-1")),
        ]
    )
    broker = _broker(script)

    ack = broker.submit(_order())
    assert ack.status == "accepted"
    assert len(_posts(script)) == 1


def test_definite_rejection_is_not_treated_as_unconfirmed() -> None:
    script = _Script([(422, {"message": "qty must be positive"})])
    broker = _broker(script)

    with pytest.raises(BrokerRejected, match="broker HTTP 422"):
        broker.submit(_order())
    assert len(script.calls) == 1
    assert script.calls[0][0] == "POST"


def test_duplicate_client_order_id_is_recovered_not_posted_twice() -> None:
    script = _Script(
        [
            (403, {"message": "client_order_id must be unique"}),
            (200, _accepted("client-1")),
        ]
    )
    broker = _broker(script)

    ack = broker.submit(_order())
    assert ack.broker_order_id == "ord-1"
    assert len(_posts(script)) == 1


def test_identified_unknown_status_is_returned_without_a_lookup() -> None:
    script = _Script([(200, _accepted("client-1", status="halted_for_review"))])
    broker = _broker(script)

    ack = broker.submit(_order())
    assert ack.status == "unmapped"
    assert ack.broker_order_id == "ord-1"
    assert len(script.calls) == 1


def test_session_lost_submit_stays_inflight_and_is_not_posted_again(tmp_path: Path) -> None:
    class _SessionTransport:
        def __init__(self) -> None:
            self.posts = 0

        def __call__(
            self,
            method: str,
            url: str,
            headers: dict[str, str],
            body: bytes | None,
            timeout: float,
        ) -> tuple[int, bytes]:
            del headers, body, timeout
            if method == "POST":
                self.posts += 1
                raise TimeoutError("timed out")
            if url.endswith("/v2/account"):
                return 200, json.dumps({"cash": "1000000", "equity": "1000000"}).encode()
            if url.endswith("/v2/positions"):
                return 200, b"[]"
            if "by_client_order_id" in url:
                return 404, b""
            raise AssertionError(url)

    transport = _SessionTransport()
    broker = AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=transport)
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "alpaca-paper",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(tmp_path),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    session = open_session(config, PRETRADE, state_dir=tmp_path, now_ns=lambda: TS, broker=broker)
    session.arm(actor="ada")

    with pytest.raises(LiveGateError, match="stays inflight"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            limit_price=100.0,
            mark=100.0,
            actor="ada",
            client_order_id="dc-lost-1",
            at_ns=TS,
        )
    assert transport.posts == 1
    assert session.orders[0]["status"] == "inflight"

    replayed = session.submit(
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        client_order_id="dc-lost-1",
        at_ns=TS,
    )
    assert replayed["replayed"] is True
    assert replayed["status"] == "inflight"
    assert transport.posts == 1
