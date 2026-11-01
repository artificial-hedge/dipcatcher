"""DNS, body-read, and retry bounds. No live sockets."""

from __future__ import annotations

import threading

import pytest

from quant_fund.data.concurrent_io import call_with_retry, pooled_request
from quant_fund.data.sources.base import HttpClient
from quant_fund.utils import net_timeout


def test_non_finite_timeout_is_rejected() -> None:
    for value in (float("nan"), float("inf"), 0, -1, True, "5"):
        with pytest.raises(ValueError, match="positive finite"):
            net_timeout.require_timeout(value)


def test_retry_count_is_capped() -> None:
    assert net_timeout.require_retries(0) == 0
    assert net_timeout.require_retries(8) == 8
    for value in (-1, 9, True, 1.5):
        with pytest.raises(ValueError, match="0..8"):
            net_timeout.require_retries(value)
    with pytest.raises(ValueError, match="0..8"):
        call_with_retry(lambda: None, retries=9)


def test_http_client_rejects_non_finite_timeout() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        HttpClient(timeout=float("nan"))
    with pytest.raises(ValueError, match="0..8"):
        HttpClient(retries=9)


def test_pooled_request_rejects_non_finite_timeout_before_connect() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        pooled_request("GET", "http://example.invalid/x", timeout=float("inf"), max_bytes=16)


def test_dns_lookup_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def hang(*_args: object, **_kwargs: object) -> list[object]:
        release.wait(2.0)
        return []

    monkeypatch.setattr(net_timeout.socket, "getaddrinfo", hang)
    try:
        with pytest.raises(TimeoutError, match="DNS lookup"):
            net_timeout.bounded_getaddrinfo("example.test", 80, 0.05)
    finally:
        release.set()


def test_read_bounded_stops_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter((0.0, 0.0, 5.0))
    monkeypatch.setattr(net_timeout.time, "monotonic", lambda: next(clock))

    class _Slow:
        def read(self, _n: int = -1) -> bytes:
            return b"x"

    with pytest.raises(TimeoutError, match="read deadline"):
        net_timeout.read_bounded(_Slow(), timeout=1.0, max_bytes=100)


def test_read_bounded_caps_a_single_shot_body() -> None:
    class _Once:
        def read(self) -> bytes:
            return b"abcdef"

    with pytest.raises(ValueError, match="exceeded 4 bytes"):
        net_timeout.read_bounded(_Once(), timeout=1.0, max_bytes=4)
    assert net_timeout.read_bounded(_Once(), timeout=1.0, max_bytes=6) == b"abcdef"


def test_read_bounded_assembles_chunks() -> None:
    class _Chunks:
        def __init__(self) -> None:
            self._body = b"hello"

        def read(self, n: int = -1) -> bytes:
            chunk, self._body = self._body[:n], self._body[n:]
            return chunk

    assert net_timeout.read_bounded(_Chunks(), timeout=1.0, max_bytes=5) == b"hello"
