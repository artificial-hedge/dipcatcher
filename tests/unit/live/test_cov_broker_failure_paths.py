"""Broker transport failures: timeouts, refused redirects, malformed errors.

These paths must never synthesise an acknowledgement. A timeout or a
transport error propagates out of the adapter, a refused redirect surfaces as
a non-2xx status, and error bodies are parsed (and redacted) rather than
treated as success.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any, cast

import pytest

from quant_fund.live import brokers
from quant_fund.live.brokers import (
    AlpacaBroker,
    BrokerAck,
    LiveOrder,
    _parse_broker_response,
    _RefuseRedirect,
    alpaca_credentials,
    urllib_transport,
)
from quant_fund.live.errors import BrokerRejected, LiveGateError

CREDS = ("unit-key", "unit-secret")


def _order(*, client: str = "c-1") -> LiveOrder:
    return LiveOrder(client, "AAA", "buy", 1.0, 10.0, 1)


def _alpaca(transport: Any, *, timeout: float = 10.0) -> AlpacaBroker:
    return AlpacaBroker("alpaca-paper", CREDS, transport=transport, timeout=timeout)


class _FakeResponse:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


class _FakeOpener:
    def __init__(self, *, response: Any = None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, float | None]] = []

    def open(self, request: Any, timeout: float | None = None) -> Any:
        self.calls.append((request.get_method(), timeout))
        if self.error is not None:
            raise self.error
        return self.response


def test_urllib_transport_returns_the_status_and_body(monkeypatch: pytest.MonkeyPatch) -> None:
    opener = _FakeOpener(response=_FakeResponse(200, b'{"ok":true}'))
    monkeypatch.setattr(brokers.urllib.request, "build_opener", lambda *a, **k: opener)

    status, payload = urllib_transport(
        "POST", "https://paper-api.alpaca.markets/v2/orders", {"A": "b"}, b"{}", 3.5
    )

    assert (status, payload) == (200, b'{"ok":true}')
    assert opener.calls == [("POST", 3.5)]


def test_urllib_transport_turns_an_http_error_into_a_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b'{"message":"rate limited"}'
    error = urllib.error.HTTPError(
        "https://paper-api.alpaca.markets/v2/orders",
        429,
        "Too Many Requests",
        None,
        io.BytesIO(body),
    )
    monkeypatch.setattr(
        brokers.urllib.request, "build_opener", lambda *a, **k: _FakeOpener(error=error)
    )

    status, payload = urllib_transport(
        "GET", "https://paper-api.alpaca.markets/v2/account", {}, None, 1.0
    )

    assert status == 429
    assert payload == body


def test_a_refused_redirect_is_never_followed() -> None:
    handler = _RefuseRedirect()
    request = brokers.urllib.request.Request("https://paper-api.alpaca.markets/v2/orders")
    assert handler.redirect_request(request, None, 302, "Found", {}, "https://evil.test/") is None


def test_a_refused_redirect_with_an_empty_body_is_an_http_error() -> None:
    # urllib raises HTTPError when the redirect handler declines, so the
    # transport hands back the 3xx status and the empty redirect body.
    with pytest.raises(BrokerRejected, match="broker HTTP 302"):
        _parse_broker_response(302, b"", CREDS, allow_missing=False)


def test_a_refused_redirect_does_not_become_an_accepted_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = urllib.error.HTTPError("u", 301, "Moved", None, io.BytesIO(b""))
    opener = _FakeOpener(error=error)
    monkeypatch.setattr(brokers.urllib.request, "build_opener", lambda *a, **k: opener)

    broker = AlpacaBroker("alpaca-paper", CREDS)
    with pytest.raises(BrokerRejected, match="broker HTTP 301"):
        broker.submit(_order())
    assert len(opener.calls) == 1


def test_a_socket_timeout_propagates_and_is_not_retried() -> None:
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
        raise TimeoutError("read timed out")

    with pytest.raises(TimeoutError, match="read timed out"):
        _alpaca(transport).submit(_order())
    assert calls == ["POST"]


def test_a_url_error_propagates_without_an_ack() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body, timeout
        raise urllib.error.URLError("connection refused")

    with pytest.raises(urllib.error.URLError):
        _alpaca(transport).account()


def test_the_configured_timeout_reaches_the_transport() -> None:
    seen: list[float] = []

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body
        seen.append(timeout)
        return 200, json.dumps({"id": "o-1", "client_order_id": "c-1", "status": "new"}).encode()

    _alpaca(transport, timeout=2.5).submit(_order())
    assert seen == [2.5]


def test_a_200_non_json_body_is_refused() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body, timeout
        return 200, b"<html>gateway</html>"

    with pytest.raises(BrokerRejected, match="HTTP 200 returned non-JSON"):
        _alpaca(transport).submit(_order())


def test_an_error_body_that_is_not_an_object_keeps_the_status_only() -> None:
    with pytest.raises(BrokerRejected, match=r"^broker HTTP 500$"):
        _parse_broker_response(500, b'["boom"]', CREDS, allow_missing=False)


def test_an_error_body_uses_the_error_key_when_message_is_absent() -> None:
    with pytest.raises(BrokerRejected, match="HTTP 401: bad key"):
        _parse_broker_response(401, b'{"error":"bad key"}', CREDS, allow_missing=False)


def test_an_error_body_redacts_the_credentials() -> None:
    body = json.dumps({"message": f"key {CREDS[1]} is disabled"}).encode()
    with pytest.raises(BrokerRejected) as excinfo:
        _parse_broker_response(403, body, CREDS, allow_missing=False)
    message = str(excinfo.value)
    assert CREDS[1] not in message
    assert "[redacted]" in message


def test_an_empty_error_body_still_carries_the_status() -> None:
    with pytest.raises(BrokerRejected, match="broker HTTP 502"):
        _parse_broker_response(502, b"   ", CREDS, allow_missing=False)


def test_a_missing_order_on_a_required_lookup_is_refused() -> None:
    with pytest.raises(BrokerRejected, match="broker HTTP 404"):
        _parse_broker_response(404, b"", CREDS, allow_missing=False)


def test_cancel_parses_an_order_body_from_the_delete_response() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body, timeout
        return (
            200,
            json.dumps(
                {
                    "id": "ord-1",
                    "client_order_id": "c-1",
                    "status": "pending_cancel",
                    "filled_qty": "0",
                    "symbol": "AAA",
                    "side": "buy",
                }
            ).encode(),
        )

    ack = _alpaca(transport).cancel("c-1", "ord-1")
    assert isinstance(ack, BrokerAck)
    assert ack.status == "pending_cancel"
    assert ack.broker_order_id == "ord-1"


def test_a_zero_priced_position_is_not_used_as_a_mark() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        if url.endswith("/account"):
            return 200, json.dumps({"cash": "10", "equity": "10"}).encode()
        return (
            200,
            json.dumps([{"symbol": "ZERO", "qty": "5", "market_value": "0"}]).encode(),
        )

    account = _alpaca(transport).account()
    assert account.positions == {"ZERO": 5.0}
    assert account.marks == {}


def test_credentials_accept_both_documented_names_and_trim_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "  key  ")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret\n")
    assert alpaca_credentials() == ("key", "secret")

    monkeypatch.setenv("ALPACA_API_KEY_ID", "k" * 201)
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    with pytest.raises(LiveGateError, match="credentials are incomplete"):
        alpaca_credentials()


def test_alpaca_live_refuses_before_the_transport_is_called() -> None:
    calls: list[str] = []

    def transport(*args: Any) -> tuple[int, bytes]:
        calls.append(cast(str, args[0]))
        return 200, b"{}"

    broker = AlpacaBroker("alpaca-live", CREDS, transport=transport)
    with pytest.raises(LiveGateError, match="honesty contract"):
        broker.submit(_order())
    assert calls == []
