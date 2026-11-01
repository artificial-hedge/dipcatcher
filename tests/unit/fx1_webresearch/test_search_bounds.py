"""Search reads stop at a byte cap. No live network."""

from __future__ import annotations

from fx1.webresearch.search import DuckDuckGoLiteSearcher


class _Huge:
    def __init__(self) -> None:
        self._body = b"x" * (512 * 1024 + 1)

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = len(self._body)
        chunk, self._body = self._body[:n], self._body[n:]
        return chunk

    def __enter__(self) -> _Huge:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_search_body_over_the_cap_is_a_transport_fault() -> None:
    def open_fn(_request: object, _timeout: float) -> _Huge:
        return _Huge()

    result = DuckDuckGoLiteSearcher(open_fn=open_fn).search("inflation", timeout_s=1.0)
    assert result.hits == []
    assert "ValueError" in result.error
