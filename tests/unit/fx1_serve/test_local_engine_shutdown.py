"""Local engine signal shutdown without sockets or model artifacts."""

from __future__ import annotations

import signal
from typing import Any

from fx1.serve import local_engine


class _TerminatingServer:
    def __init__(self) -> None:
        self.closed = False
        self.engine: Any = None

    def serve_forever(self) -> None:
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)

    def server_close(self) -> None:
        self.closed = True


def test_sigterm_closes_server_and_restores_previous_handler() -> None:
    previous = signal.getsignal(signal.SIGTERM)
    server = _TerminatingServer()
    original_server = local_engine._EngineHTTPServer
    local_engine._EngineHTTPServer = lambda *args: server  # type: ignore[assignment]
    try:
        assert local_engine.serve(object(), "127.0.0.1", 0, watch_orphan=False) == 0  # type: ignore[arg-type]
        assert server.closed is True
        assert signal.getsignal(signal.SIGTERM) == previous
    finally:
        local_engine._EngineHTTPServer = original_server
        signal.signal(signal.SIGTERM, previous)
