"""A cancel that races a fill, or an unknown cancel status, does not assume success."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.brokers import AlpacaBroker
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import LiveGateError
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def _row(status: str, *, qty: str = "0", price: str | None = None) -> dict[str, str]:
    payload = {
        "id": "ord-1",
        "client_order_id": "client-1",
        "status": status,
        "filled_qty": qty,
        "symbol": "SYNTH",
        "side": "buy",
    }
    if price is not None:
        payload["filled_avg_price"] = price
        payload["filled_at"] = "2024-01-03T20:00:00+00:00"
    return payload


class _Script:
    def __init__(self, steps: list[Any]) -> None:
        self.steps = list(steps)
        self.calls: list[str] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, body, timeout
        self.calls.append(method)
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        status, payload = step
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        return int(status), raw


def _broker(script: _Script) -> AlpacaBroker:
    return AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=script)


def test_empty_cancel_does_not_invent_success_when_lookup_misses() -> None:
    script = _Script([(204, b""), (404, b"")])
    broker = _broker(script)

    with pytest.raises(LiveGateError, match="could not be confirmed"):
        broker.cancel("client-1", "ord-1")
    assert script.calls == ["DELETE", "GET"]


def test_empty_cancel_returns_the_fill_that_won_the_race() -> None:
    script = _Script([(204, b""), (200, _row("filled", qty="2", price="10"))])
    broker = _broker(script)

    ack = broker.cancel("client-1", "ord-1")
    assert ack.status == "filled"
    assert ack.filled_quantity == pytest.approx(2.0)
    assert ack.fills[0].quantity == pytest.approx(2.0)
    assert script.calls == ["DELETE", "GET"]


def test_confirmed_cancel_keeps_the_partial_fill() -> None:
    body = _row("canceled", qty="1", price="10")
    script = _Script([(200, body)])
    broker = _broker(script)

    ack = broker.cancel("client-1", "ord-1")
    assert ack.status == "canceled"
    assert ack.filled_quantity == pytest.approx(1.0)
    assert len(ack.fills) == 1
    assert script.calls == ["DELETE"]


def test_unknown_cancel_status_reconciles_and_halts_when_still_unknown() -> None:
    unknown = _row("halted_for_review")
    script = _Script([(200, unknown), (200, unknown)])
    broker = _broker(script)

    with pytest.raises(LiveGateError, match="could not be confirmed"):
        broker.cancel("client-1", "ord-1")
    assert script.calls == ["DELETE", "GET"]


def test_unknown_cancel_status_uses_a_confirmed_lookup() -> None:
    script = _Script(
        [
            (200, _row("halted_for_review")),
            (200, _row("partially_filled", qty="1", price="10")),
        ]
    )
    broker = _broker(script)

    ack = broker.cancel("client-1", "ord-1")
    assert ack.status == "partially_filled"
    assert ack.fills
    assert script.calls == ["DELETE", "GET"]


def test_cancel_timeout_reconciles_the_partial_and_does_not_assume_cancel() -> None:
    script = _Script(
        [TimeoutError("timed out"), (200, _row("partially_filled", qty="1", price="10"))]
    )
    broker = _broker(script)

    ack = broker.cancel("client-1", "ord-1")
    assert ack.status == "partially_filled"
    assert script.calls == ["DELETE", "GET"]


def test_cancel_and_lookup_both_failing_halts() -> None:
    script = _Script([TimeoutError("timed out"), TimeoutError("lookup timed out")])
    broker = _broker(script)

    with pytest.raises(LiveGateError, match="do not assume the cancel succeeded") as caught:
        broker.cancel("client-1", "ord-1")
    assert caught.value.__cause__ is not None
    assert script.calls == ["DELETE", "GET"]


def test_reject_during_cancel_reconciles_instead_of_dropping_the_fill() -> None:
    script = _Script(
        [
            (422, {"message": "order is already filled"}),
            (200, _row("filled", qty="2", price="10")),
        ]
    )
    broker = _broker(script)

    ack = broker.cancel("client-1", "ord-1")
    assert ack.status == "filled"
    assert script.calls == ["DELETE", "GET"]


def test_session_cancel_race_records_the_fill(tmp_path: Path) -> None:
    class _Route:
        def __init__(self) -> None:
            self.deletes = 0

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
                return 200, json.dumps(_row("accepted")).encode()
            if method == "DELETE":
                self.deletes += 1
                return 204, b""
            if url.endswith("/v2/account"):
                return 200, json.dumps({"cash": "1000000", "equity": "1000000"}).encode()
            if url.endswith("/v2/positions"):
                return 200, b"[]"
            if "by_client_order_id" in url:
                return 200, json.dumps(_row("filled", qty="2", price="10")).encode()
            if "activities" in url:
                return 200, b"[]"
            raise AssertionError(f"{method} {url}")

    route = _Route()
    broker = AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=route)
    session = open_session(
        LiveConfig.model_validate(
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
        ),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=lambda: TS,
        broker=broker,
    )
    session.arm(actor="ada")
    opened = session.submit(
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        limit_price=10.0,
        mark=10.0,
        actor="ada",
        client_order_id="client-1",
        at_ns=TS,
    )
    assert opened["status"] == "accepted"
    canceled = session.cancel(client_order_id="client-1", actor="ada", at_ns=TS + 1)
    assert canceled["status"] == "filled"
    assert canceled["filled_quantity"] == pytest.approx(2.0)
    assert route.deletes == 1


def test_session_unknown_cancel_halts_without_marking_canceled(tmp_path: Path) -> None:
    class _Route:
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
                return 200, json.dumps(_row("new")).encode()
            if method == "DELETE":
                return 200, json.dumps(_row("halted_for_review")).encode()
            if url.endswith("/v2/account"):
                return 200, json.dumps({"cash": "1000000", "equity": "1000000"}).encode()
            if url.endswith("/v2/positions"):
                return 200, b"[]"
            if "by_client_order_id" in url:
                return 404, b""
            if "activities" in url:
                return 200, b"[]"
            raise AssertionError(f"{method} {url}")

    broker = AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=_Route())
    session = open_session(
        LiveConfig.model_validate(
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
        ),
        PRETRADE,
        state_dir=tmp_path / "unk",
        now_ns=lambda: TS,
        broker=broker,
    )
    session.arm(actor="ada")
    session.submit(
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        limit_price=10.0,
        mark=10.0,
        actor="ada",
        client_order_id="client-1",
        at_ns=TS,
    )
    with pytest.raises(LiveGateError, match="could not be confirmed"):
        session.cancel(client_order_id="client-1", actor="ada", at_ns=TS + 1)
    assert session.orders[0]["status"] == "accepted"
    assert session.orders[0]["status"] != "canceled"
