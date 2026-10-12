"""A dead host fails at the connect timeout, not after the whole read timeout."""

from __future__ import annotations

import http.client
from typing import Any

import pytest

from fx1.interactive import model_client
from fx1.interactive.model_client import _Box, _dial_factory


class _Sock:
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def settimeout(self, value: float | None) -> None:
        self.timeouts.append(value)


def _dial_with(monkeypatch: pytest.MonkeyPatch, timeout: float | None) -> tuple[Any, _Sock]:
    seen: dict[str, Any] = {}
    sock = _Sock()

    def fake_dial(address: Any, connect: Any, *_rest: Any) -> _Sock:
        seen["connect"] = connect
        return sock

    monkeypatch.setattr(model_client, "_dial", fake_dial)
    conn = http.client.HTTPConnection("127.0.0.1", 9)
    dial = _dial_factory(conn, _Box())
    returned = dial(("127.0.0.1", 9), timeout)
    assert returned is sock
    return seen["connect"], sock


def test_long_read_timeout_gets_a_short_connect_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    connect, sock = _dial_with(monkeypatch, 90.0)
    assert connect == model_client._CONNECT_TIMEOUT_S
    assert sock.timeouts == [90.0]


def test_short_read_timeout_is_not_lengthened(monkeypatch: pytest.MonkeyPatch) -> None:
    connect, sock = _dial_with(monkeypatch, 2.0)
    assert connect == 2.0
    assert sock.timeouts == [2.0]


def test_no_timeout_stays_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    connect, sock = _dial_with(monkeypatch, None)
    assert connect is None
    assert sock.timeouts == []
