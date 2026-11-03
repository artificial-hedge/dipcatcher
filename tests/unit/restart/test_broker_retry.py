"""Bounded retry for a degraded broker. SYNTHETIC, no sockets.

429, 5xx, timeouts, and dropped connections retry with backoff. The client
order id is recovered before another submit. The budget fails closed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.unit.fault.support import STARTING_CASH, FaultBroker, arm, buy, open_fault_session

from quant_fund.live.brokers import AlpacaBroker, BrokerAck, LiveOrder
from quant_fund.live.errors import BrokerRejected, LiveGateError


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quant_fund.live.session._retry_sleep", lambda _delay: None)


class _Throttle(FaultBroker):
    def __init__(self, faults: list[BaseException]) -> None:
        super().__init__()
        self.faults = list(faults)
        self.posts: list[str] = []

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.posts.append(order.client_order_id)
        if self.faults:
            raise self.faults.pop(0)
        return super().submit(order)


class _TimeoutAfterStore(FaultBroker):
    def __init__(self) -> None:
        super().__init__()
        self.posts: list[str] = []
        self._raise = True

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.posts.append(order.client_order_id)
        ack = super().submit(order)
        if self._raise:
            self._raise = False
            raise TimeoutError("timeout after accept")
        return ack


@pytest.mark.parametrize(
    "fault",
    [
        BrokerRejected("broker HTTP 429: slow down"),
        BrokerRejected("broker HTTP 500"),
        BrokerRejected("broker HTTP 502"),
        BrokerRejected("broker HTTP 503"),
        BrokerRejected("broker HTTP 504"),
        TimeoutError("timed out"),
        ConnectionError("connection reset"),
    ],
)
def test_transient_fault_retries_then_accepts_once(tmp_path: Path, fault: BaseException) -> None:
    broker = _Throttle([fault, fault])
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    accepted = buy(session, quantity=1.0, client_order_id="retry-1")
    assert accepted["status"] == "accepted"
    assert accepted["live_pnl_claim"] is False
    assert broker.posts == ["retry-1", "retry-1", "retry-1"]
    assert broker.submit_count == 1
    assert list(broker.orders) == ["retry-1"]
    assert session.ledger.computed_cash == pytest.approx(STARTING_CASH)


def test_retry_sleeps_on_a_bounded_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []
    monkeypatch.setattr("quant_fund.live.session._retry_sleep", delays.append)
    broker = _Throttle([BrokerRejected("broker HTTP 429")] * 2)
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    assert buy(session, quantity=1.0, client_order_id="back")["status"] == "accepted"
    assert delays == [0.01, 0.02]


def test_timeout_after_accept_does_not_submit_again(tmp_path: Path) -> None:
    broker = _TimeoutAfterStore()
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    accepted = buy(session, quantity=2.0, client_order_id="once")
    assert accepted["status"] == "accepted"
    assert broker.posts == ["once"]
    assert broker.submit_count == 1
    assert session.ledger.computed_positions == {}


def test_budget_fails_closed_without_a_synthetic_fill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    delays: list[float] = []
    monkeypatch.setattr("quant_fund.live.session._retry_sleep", delays.append)
    broker = _Throttle([BrokerRejected("broker HTTP 429")] * 5)
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    with pytest.raises(LiveGateError, match="stays inflight"):
        buy(session, quantity=3.0, client_order_id="budget")
    assert broker.posts == ["budget", "budget", "budget"]
    assert delays == [0.01, 0.02]
    assert broker.orders == {}
    assert session.orders[0]["status"] == "inflight"
    assert session.ledger.computed_cash == pytest.approx(STARTING_CASH)
    assert session.ledger.computed_positions == {}
    stored = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert stored["orders"][0]["status"] == "inflight"
    assert stored["live_pnl_claim"] is False


def test_hard_reject_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr("quant_fund.live.session._retry_sleep", delays.append)
    broker = _Throttle([BrokerRejected("broker HTTP 400: no buying power")])
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    rejected = buy(session, quantity=1.0, client_order_id="hard")
    assert rejected["status"] == "rejected"
    assert rejected["broker_called"] is True
    assert "HTTP 400" in str(rejected["reason"])
    assert broker.posts == ["hard"]
    assert delays == []
    assert session.ledger.computed_cash == pytest.approx(STARTING_CASH)


def test_alpaca_timeout_after_accept_recovers_by_client_id(tmp_path: Path) -> None:
    """Scripted paper transport. The live host is never contacted."""
    script = _AlpacaScript()
    broker = AlpacaBroker("alpaca-paper", ("key", "secret"), transport=script)
    session = open_fault_session(tmp_path, broker, venue="alpaca-paper")
    arm(session)
    accepted = buy(session, quantity=1.0, client_order_id="alpaca-1")
    assert accepted["status"] == "accepted"
    assert accepted["broker_order_id"] == "ord-1"
    assert script.posts == 1
    assert any("by_client_order_id" in url for url in script.gets)
    assert session.venue == "alpaca-paper"
    assert accepted["live_pnl_claim"] is False


class _AlpacaScript:
    def __init__(self) -> None:
        self.posts = 0
        self.gets: list[str] = []
        self.stored: dict[str, object] | None = None

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del headers, timeout
        if method == "POST":
            self.posts += 1
            self.stored = json.loads(body or b"{}")
            raise TimeoutError("timeout after accept")
        self.gets.append(url)
        if "by_client_order_id" in url and self.stored is not None:
            payload = {
                "id": "ord-1",
                "client_order_id": self.stored["client_order_id"],
                "status": "new",
                "filled_qty": "0",
                "symbol": self.stored["symbol"],
                "side": self.stored["side"],
            }
            return 200, json.dumps(payload).encode()
        if url.endswith("/v2/account"):
            return 200, json.dumps({"cash": "1000000", "equity": "1000000"}).encode()
        if url.endswith("/v2/positions"):
            return 200, b"[]"
        if "activities" in url:
            return 200, b"[]"
        return 404, b"{}"
