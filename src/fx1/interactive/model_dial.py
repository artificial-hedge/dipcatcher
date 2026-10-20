"""Connect budget for the hosted-model HTTP client.

TCP, a proxy ``CONNECT``, and the TLS handshake share one short budget.
The socket is then restored to the caller's read timeout. A test that
replaces ``HTTPConnection.connect`` still wins: the stock method is called
when it is no longer the one captured at import.
"""

from __future__ import annotations

import contextlib
import errno
import http.client
import socket
import ssl
import sys

_STOCK_HTTP_CONNECT = http.client.HTTPConnection.connect
_STOCK_HTTPS_CONNECT = http.client.HTTPSConnection.connect


def open_https(
    host: str,
    port: int,
    timeout: float,
    proxy: tuple[str, int] | None,
    *,
    connect_cap: float,
) -> http.client.HTTPConnection:
    context = ssl.create_default_context()
    if proxy is None:
        return _BudgetHTTPS(host, port, timeout=timeout, context=context, connect_cap=connect_cap)
    conn = _BudgetHTTPS(
        proxy[0], proxy[1], timeout=timeout, context=context, connect_cap=connect_cap
    )
    conn.set_tunnel(host, port)
    return conn


def open_http(
    host: str,
    port: int,
    timeout: float,
    proxy: tuple[str, int] | None,
    *,
    connect_cap: float,
) -> http.client.HTTPConnection:
    if proxy is None:
        return _BudgetHTTP(host, port, timeout=timeout, connect_cap=connect_cap)
    conn = _BudgetHTTP(proxy[0], proxy[1], timeout=timeout, connect_cap=connect_cap)
    conn.set_tunnel(host, port)
    return conn


class _BudgetHTTP(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, *, timeout: float, connect_cap: float) -> None:
        self._connect_cap = connect_cap
        super().__init__(host, port, timeout=timeout)

    def connect(self) -> None:
        if http.client.HTTPConnection.connect is not _STOCK_HTTP_CONNECT:
            http.client.HTTPConnection.connect(self)
            return
        _open_capped(self, tls=False)


class _BudgetHTTPS(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout: float,
        context: ssl.SSLContext,
        connect_cap: float,
    ) -> None:
        self._connect_cap = connect_cap
        super().__init__(host, port, timeout=timeout, context=context)

    def connect(self) -> None:
        if http.client.HTTPSConnection.connect is not _STOCK_HTTPS_CONNECT:
            http.client.HTTPSConnection.connect(self)
            return
        _open_capped(self, tls=True)


def _open_capped(conn: http.client.HTTPConnection, *, tls: bool) -> None:
    sys.audit("http.client.connect", conn, conn.host, conn.port)
    sock = _dial(conn)
    conn.sock = sock
    _nodelay(sock)
    _handshake(conn, sock, tls=tls)


def _dial(conn: http.client.HTTPConnection) -> socket.socket:
    creator = getattr(conn, "_create_connection", None)
    if not callable(creator):
        raise OSError("missing connection factory")
    timeout = _dial_timeout(conn.timeout, _cap_of(conn))
    sock = creator((conn.host, conn.port), timeout, getattr(conn, "source_address", None))
    if not isinstance(sock, socket.socket):
        raise OSError("connection factory did not return a socket")
    return sock


def _handshake(conn: http.client.HTTPConnection, sock: socket.socket, *, tls: bool) -> None:
    budget = _budget(conn.timeout, _cap_of(conn))
    if budget is not None:
        sock.settimeout(budget)
    try:
        _tunnel_and_wrap(conn, tls=tls)
    finally:
        _restore(conn)


def _tunnel_and_wrap(conn: http.client.HTTPConnection, *, tls: bool) -> None:
    if _tunnel_host(conn) is not None:
        _call_tunnel(conn)
    if tls:
        _wrap(conn)


def _call_tunnel(conn: http.client.HTTPConnection) -> None:
    tunnel = getattr(conn, "_tunnel", None)
    if not callable(tunnel):
        raise OSError("missing proxy tunnel")
    tunnel()


def _wrap(conn: http.client.HTTPConnection) -> None:
    sock = conn.sock
    if sock is None:
        raise OSError("missing socket")
    context = getattr(conn, "_context", None)
    if not isinstance(context, ssl.SSLContext):
        raise OSError("missing TLS context")
    host = _tunnel_host(conn)
    name = host if host is not None else conn.host
    conn.sock = context.wrap_socket(sock, server_hostname=name)


def _tunnel_host(conn: http.client.HTTPConnection) -> str | None:
    raw = getattr(conn, "_tunnel_host", None)
    if isinstance(raw, str) and raw:
        return raw
    return None


def _dial_timeout(timeout: object, cap: float) -> object:
    budget = _budget(timeout, cap)
    if budget is None:
        return timeout
    return budget


def _budget(timeout: object, cap: float) -> float | None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return None
    return min(float(timeout), cap)


def _restore(conn: http.client.HTTPConnection) -> None:
    sock = conn.sock
    timeout = conn.timeout
    if sock is None or isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return
    # A failed handshake may already have closed the socket. settimeout must
    # not replace the SSLError (or connect error) that the caller classifies.
    with contextlib.suppress(OSError):
        sock.settimeout(float(timeout))


def _cap_of(conn: http.client.HTTPConnection) -> float:
    raw = getattr(conn, "_connect_cap", 15.0)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 15.0
    return float(raw)


def _nodelay(sock: socket.socket) -> None:
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError as exc:
        if exc.errno != errno.ENOPROTOOPT:
            raise
