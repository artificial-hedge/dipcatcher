"""Bounded search transport and one-budget fallback regressions."""

import io

from fx1.webresearch.search import (
    DEFAULT_MAX_RESPONSE_BYTES,
    DuckDuckGoLiteSearcher,
)


class _Response:
    def __init__(self, body: bytes) -> None:
        self._stream = io.BytesIO(body)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self._stream.read(size)

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        self._stream.close()


def test_search_response_has_a_hard_byte_limit() -> None:
    responses: list[_Response] = []

    def open_fn(_request: object, _timeout_s: float) -> _Response:
        response = _Response(b"x" * (DEFAULT_MAX_RESPONSE_BYTES + 1))
        responses.append(response)
        return response

    result = DuckDuckGoLiteSearcher(open_fn=open_fn).search("test")
    assert result.hits == []
    assert "transport fault" in result.error
    assert len(responses) == 2
    assert all(
        response.read_sizes == [DEFAULT_MAX_RESPONSE_BYTES + 1] for response in responses
    )


def test_fallback_uses_only_remaining_search_budget() -> None:
    clock = [100.0]
    timeouts: list[float] = []

    def open_fn(_request: object, timeout_s: float) -> _Response:
        timeouts.append(timeout_s)
        if len(timeouts) == 1:
            clock[0] += 4.5
        return _Response(b"<html></html>")

    result = DuckDuckGoLiteSearcher(open_fn=open_fn, now_fn=lambda: clock[0]).search(
        "test", timeout_s=5.0
    )
    assert len(timeouts) == 2
    assert abs(timeouts[0] - 5.0) < 1e-9
    assert abs(timeouts[1] - 0.5) < 1e-9
    assert result.hits == []


def test_fallback_stops_when_first_endpoint_exhausts_budget() -> None:
    clock = [100.0]
    calls = 0

    def open_fn(_request: object, _timeout_s: float) -> _Response:
        nonlocal calls
        calls += 1
        clock[0] += 5.0
        return _Response(b"<html></html>")

    result = DuckDuckGoLiteSearcher(open_fn=open_fn, now_fn=lambda: clock[0]).search(
        "test", timeout_s=5.0
    )
    assert calls == 1
    assert result.error == "search time budget exhausted"


def test_nonpositive_timeout_does_not_open_network() -> None:
    def open_fn(_request: object, _timeout_s: float) -> _Response:
        raise AssertionError("network should not open")

    result = DuckDuckGoLiteSearcher(open_fn=open_fn).search("test", timeout_s=0)
    assert result.error == "search time budget exhausted"

