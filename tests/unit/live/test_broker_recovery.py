from __future__ import annotations

import json
import random
from typing import Any

import pytest

from quant_fund.live.brokers import AlpacaBroker, LiveOrder
from quant_fund.live.errors import BrokerRejected

_Response = tuple[int, bytes]
_Outcome = _Response | Exception


class _FakeTransport:
    def __init__(self, outcomes: list[_Outcome]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> _Response:
        del headers, body, timeout
        self.calls.append((method, url))
        if not self.outcomes:
            raise AssertionError("unexpected broker request")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _order_response(client_order_id: object = "client-1") -> _Response:
    payload = {
        "id": "broker-order-1",
        "client_order_id": client_order_id,
        "status": "accepted",
        "filled_qty": "0",
        "symbol": "SYNTH",
        "side": "buy",
    }
    return 200, json.dumps(payload).encode("utf-8")


def _broker(
    transport: _FakeTransport,
    sleeps: list[float],
    *,
    retries: int = 2,
    backoff_s: float = 0.25,
    max_backoff_s: float = 1.0,
    seed: int = 17,
) -> AlpacaBroker:
    return AlpacaBroker(
        "alpaca-paper",
        ("test-key", "test-secret"),
        transport=transport,
        recover_retries=retries,
        recover_backoff_s=backoff_s,
        recover_max_backoff_s=max_backoff_s,
        recover_sleeper=sleeps.append,
        recover_rng=random.Random(seed),
    )


def test_recover_retries_transient_lookup_with_bounded_jittered_backoff() -> None:
    transport = _FakeTransport(
        [
            TimeoutError("first lookup timed out"),
            OSError("second lookup disconnected"),
            TimeoutError("third lookup timed out"),
            _order_response(),
        ]
    )
    sleeps: list[float] = []
    broker = _broker(
        transport,
        sleeps,
        retries=3,
        backoff_s=0.5,
        max_backoff_s=1.0,
        seed=23,
    )

    ack = broker.recover("client-1")

    assert ack is not None
    assert ack.client_order_id == "client-1"
    assert [method for method, _ in transport.calls] == ["GET"] * 4
    expected_rng = random.Random(23)
    caps = (0.5, 1.0, 1.0)
    expected = [expected_rng.uniform(0.0, cap) for cap in caps]
    assert sleeps == pytest.approx(expected)
    for sleep, cap in zip(sleeps, caps, strict=True):
        assert 0.0 <= sleep <= cap


def test_recover_exhaustion_surfaces_the_final_transient_error() -> None:
    final_error = OSError("final lookup failed")
    transport = _FakeTransport(
        [TimeoutError("first lookup failed"), OSError("second lookup failed"), final_error]
    )
    sleeps: list[float] = []
    broker = _broker(transport, sleeps, retries=2)

    with pytest.raises(OSError) as caught:
        broker.recover("client-1")

    assert caught.value is final_error
    assert len(transport.calls) == 3
    assert len(sleeps) == 2
    assert transport.outcomes == []


def test_uncertain_post_is_not_replayed_when_recovery_retries() -> None:
    transport = _FakeTransport(
        [
            TimeoutError("POST response lost"),
            OSError("first recovery lookup failed"),
            _order_response(),
        ]
    )
    sleeps: list[float] = []
    broker = _broker(transport, sleeps, retries=1)
    order = LiveOrder("client-1", "SYNTH", "buy", 1.0, 10.0, 1)

    with pytest.raises(TimeoutError, match="POST response lost"):
        broker.submit(order)
    ack = broker.recover("client-1")

    assert ack is not None
    assert ack.client_order_id == "client-1"
    assert [method for method, _ in transport.calls] == ["POST", "GET", "GET"]
    assert transport.calls[0][1].endswith("/v2/orders")
    assert len(sleeps) == 1
    assert transport.outcomes == []


def test_recover_404_does_not_retry() -> None:
    transport = _FakeTransport([(404, b"")])
    sleeps: list[float] = []
    broker = _broker(transport, sleeps, retries=3)

    assert broker.recover("client-1") is None
    assert len(transport.calls) == 1
    assert sleeps == []


def test_recover_non_transient_reject_does_not_retry() -> None:
    transport = _FakeTransport([(422, b'{"message":"invalid client_order_id"}')])
    sleeps: list[float] = []
    broker = _broker(transport, sleeps, retries=3)

    with pytest.raises(BrokerRejected, match="broker HTTP 422"):
        broker.recover("client-1")

    assert len(transport.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("returned_id", ["different-client", None, 42])
def test_recover_rejects_foreign_or_malformed_client_order_ids(returned_id: object) -> None:
    transport = _FakeTransport([_order_response(returned_id)])
    sleeps: list[float] = []
    broker = _broker(transport, sleeps, retries=3)

    with pytest.raises(BrokerRejected, match="client_order_id"):
        broker.recover("client-1")

    assert len(transport.calls) == 1
    assert sleeps == []


@pytest.mark.parametrize("retries", [-1, 1.5, True])
def test_invalid_recover_retry_count_fails_closed(retries: Any) -> None:
    with pytest.raises(ValueError, match="recover_retries"):
        AlpacaBroker(
            "alpaca-paper",
            ("test-key", "test-secret"),
            transport=_FakeTransport([]),
            recover_retries=retries,
        )


@pytest.mark.parametrize("option", ["recover_backoff_s", "recover_max_backoff_s"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "invalid"])
def test_invalid_recover_backoff_fails_closed(option: str, value: Any) -> None:
    with pytest.raises(ValueError, match="recover backoff"):
        AlpacaBroker(
            "alpaca-paper",
            ("test-key", "test-secret"),
            transport=_FakeTransport([]),
            **{option: value},
        )
