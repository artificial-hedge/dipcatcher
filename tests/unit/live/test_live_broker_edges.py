from __future__ import annotations

from typing import Any

import pytest

from quant_fund.live.brokers import AlpacaBroker, BrokerAck
from quant_fund.live.errors import BrokerRejected


def _ack(**overrides: Any) -> BrokerAck:
    payload: dict[str, Any] = {
        "id": "broker-order-1",
        "client_order_id": "client-order-1",
        "symbol": "SYNTH",
        "side": "buy",
        "qty": "10",
        "filled_qty": "0",
        "status": "accepted",
    }
    payload.update(overrides)
    return AlpacaBroker("alpaca-paper", None)._ack(payload)


@pytest.mark.parametrize("status", ["partially_filled", "filled", "canceled", "rejected"])
def test_ack_emits_cumulative_fill_for_partial_and_terminal_statuses(status: str) -> None:
    ack = _ack(
        status=status,
        filled_qty="2.5",
        filled_avg_price="101.25",
        updated_at="2024-01-03T20:00:00+00:00",
    )

    assert ack.status == status
    assert ack.filled_quantity == 2.5
    assert ack.fill_price == 101.25
    assert len(ack.fills) == 1
    fill = ack.fills[0]
    assert fill.fill_id == "order:broker-order-1:cum"
    assert fill.client_order_id == "client-order-1"
    assert fill.symbol == "SYNTH"
    assert fill.side == "buy"
    assert fill.quantity == 2.5
    assert fill.price == 101.25
    assert fill.fill_time_ns == 1_704_312_000_000_000_000


@pytest.mark.parametrize("status", ["canceled", "rejected"])
def test_terminal_ack_without_executions_has_no_provisional_fill(status: str) -> None:
    ack = _ack(status=status, filled_avg_price=None)

    assert ack.filled_quantity == 0.0
    assert ack.fill_price is None
    assert ack.fills == ()


def test_duplicate_ack_decodes_to_the_same_provisional_fill() -> None:
    payload = {
        "status": "partially_filled",
        "filled_qty": "3",
        "filled_avg_price": "100.5",
        "filled_at": "2024-01-03T20:00:00Z",
    }

    assert _ack(**payload) == _ack(**payload)


@pytest.mark.parametrize("filled_qty", ["-0.01", "nan", "inf", "-inf"])
def test_ack_rejects_negative_or_non_finite_filled_quantity(filled_qty: str) -> None:
    with pytest.raises(BrokerRejected, match="filled quantity"):
        _ack(filled_qty=filled_qty, filled_avg_price="100")


def test_ack_rejects_cumulative_fill_larger_than_order() -> None:
    with pytest.raises(BrokerRejected, match="exceeded order quantity"):
        _ack(filled_qty="10.0001", filled_avg_price="100")


@pytest.mark.parametrize("filled_avg_price", [None, "", "0", "-1", "nan", "inf"])
def test_ack_rejects_missing_or_invalid_price_for_an_execution(
    filled_avg_price: object,
) -> None:
    with pytest.raises(BrokerRejected, match="average price"):
        _ack(filled_qty="1", filled_avg_price=filled_avg_price)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("client_order_id", ""),
        ("symbol", ""),
        ("side", ""),
        ("side", "hold"),
    ],
)
def test_ack_rejects_incomplete_order_identity(field: str, value: str) -> None:
    with pytest.raises(BrokerRejected, match="order identity was incomplete"):
        _ack(**{field: value})


@pytest.mark.parametrize("qty", ["0", "-1", "nan", "inf"])
def test_ack_rejects_invalid_order_quantity(qty: str) -> None:
    with pytest.raises(BrokerRejected, match="order quantity"):
        _ack(qty=qty)


def test_empty_cancel_response_with_missing_recovery_stays_pending() -> None:
    calls: list[str] = []

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del url, headers, body, timeout
        calls.append(method)
        if method == "DELETE":
            return 204, b""
        assert method == "GET"
        return 404, b""

    broker = AlpacaBroker(
        "alpaca-paper",
        ("unit-test-key", "unit-test-secret"),
        transport=transport,
    )
    ack = broker.cancel("client-order-1", "broker-order-1")

    assert calls == ["DELETE", "GET"]
    assert ack.status == "pending_cancel"
    assert ack.client_order_id == "client-order-1"
    assert ack.broker_order_id == "broker-order-1"
    assert ack.fill_price is None
    assert ack.fills == ()
    assert "cancellation and execution state are unconfirmed" in ack.reason
