"""Local HTTP ``/metrics`` endpoint for Prometheus. Started only on request."""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from quant_fund.observe.metrics import render_prometheus
from quant_fund.utils.api_auth import credential_matches, presented_api_key

_SERVER: ThreadingHTTPServer | None = None
_THREAD: threading.Thread | None = None
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
_UNAUTHORIZED = b'{"detail":"invalid or missing X-API-Key"}\n'


class _Handler(BaseHTTPRequestHandler):
    allow_unauthenticated = False
    key_env = ""

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path.split("?", 1)[0] != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        if not self._authorized():
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(_UNAUTHORIZED)))
            self.send_header("WWW-Authenticate", "Bearer")
            self.end_headers()
            self.wfile.write(_UNAUTHORIZED)
            return
        body = render_prometheus().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if type(self).allow_unauthenticated:
            return True
        name = type(self).key_env
        expected = os.environ.get(name, "").strip() if name else ""
        if not expected:
            return False
        presented = presented_api_key({str(key): str(value) for key, value in self.headers.items()})
        return credential_matches(presented, expected)

    def log_message(self, format: str, *args: object) -> None:
        return


def start_metrics_server(
    port: int = 9464,
    host: str = "127.0.0.1",
    *,
    allow_unauthenticated: bool = False,
    key_env: str = "",
) -> int:
    """Bind ``host:port`` and serve ``/metrics``. Returns the actual port.

    ``allow_unauthenticated`` is the loopback scrape opt-out. It is refused
    when ``host`` is not loopback.
    """
    global _SERVER, _THREAD
    if allow_unauthenticated and host not in _LOOPBACK:
        raise ValueError("unauthenticated /metrics is loopback-only")
    if _SERVER is not None:
        return int(_SERVER.server_address[1])
    _Handler.allow_unauthenticated = allow_unauthenticated
    _Handler.key_env = key_env
    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, name="dipcatcher-metrics", daemon=True)
    thread.start()
    _SERVER = server
    _THREAD = thread
    return int(server.server_address[1])


def stop_metrics_server() -> None:
    global _SERVER, _THREAD
    server = _SERVER
    thread = _THREAD
    _SERVER = None
    _THREAD = None
    if server is not None:
        server.shutdown()
        server.server_close()
    if thread is not None:
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError("metrics server thread did not stop within 2 seconds")
    _Handler.allow_unauthenticated = False
    _Handler.key_env = ""
