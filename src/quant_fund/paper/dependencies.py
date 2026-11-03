"""Paper-mode probes for DNS, TLS, and the broker port.

Unset ``DIPCATCHER_PAPER_BROKER_URL`` leaves the simulated book local.
A set URL is checked before an order is journaled. Failure returns a code
and does not submit. ``.invalid`` names are a DNS failure. HTTPS uses the
default trust store plus ``DIPCATCHER_PAPER_BROKER_CA`` when that file is set.
"""

from __future__ import annotations

import os
import socket
import ssl
from urllib.parse import urlsplit

from quant_fund.observe.order_guard import clear_dependency, dependency_code, note_dependency

_URL = "DIPCATCHER_PAPER_BROKER_URL"
_TIMEOUT = "DIPCATCHER_PAPER_BROKER_TIMEOUT"
_CA = "DIPCATCHER_PAPER_BROKER_CA"


def probe_broker() -> str | None:
    """Return a failure code, or ``None`` when the probe is off or healthy."""
    raw = os.environ.get(_URL, "").strip()
    if not raw:
        return None
    return _dispatch(raw, _timeout())


def dependency_refusal() -> str | None:
    """Latch a failed probe. A later success clears it. Unset keeps a latch."""
    raw = os.environ.get(_URL, "").strip()
    if not raw:
        return dependency_code()
    code = probe_broker()
    if code is None:
        clear_dependency()
        return None
    note_dependency(code)
    return code


def _timeout() -> float:
    raw = os.environ.get(_TIMEOUT, "").strip()
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    if value <= 0 or value > 30:
        return 1.0
    return value


def _dispatch(raw: str, timeout: float) -> str | None:
    parsed = urlsplit(raw)
    host = parsed.hostname or ""
    if parsed.scheme == "dns" or host.endswith(".invalid"):
        return _dns(host) or "dns"
    if parsed.scheme == "https":
        port = _port(raw, 443)
        if port is None:
            return "broker_down"
        return _tls(host, port, timeout)
    if parsed.scheme == "http":
        port = _port(raw, 80)
        if port is None:
            return "broker_down"
        return _http(host, port, timeout)
    return "broker_down"


def _port(raw: str, default: int) -> int | None:
    try:
        port = urlsplit(raw).port
    except ValueError:
        return None
    if port is None:
        return default
    return port


def _dns(host: str) -> str | None:
    if not host:
        return "dns"
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(2.0)
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError:
        return "dns"
    finally:
        socket.setdefaulttimeout(previous)
    return None


def _context() -> ssl.SSLContext | None:
    context = ssl.create_default_context()
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    ca = os.environ.get(_CA, "").strip()
    if not ca:
        return context
    try:
        context.load_verify_locations(cafile=ca)
    except (OSError, ssl.SSLError):
        return None
    return context


def _tls(host: str, port: int, timeout: float) -> str | None:
    failed = _dns(host)
    if failed is not None:
        return failed
    context = _context()
    if context is None:
        return "tls"
    try:
        with (
            socket.create_connection((host, port), timeout=timeout) as raw,
            context.wrap_socket(raw, server_hostname=host),
        ):
            raw.settimeout(timeout)
            return None
    except TimeoutError:
        return "broker_slow"
    except ssl.SSLError:
        return "tls"
    except OSError as exc:
        return _os_code(exc)


def _http(host: str, port: int, timeout: float) -> str | None:
    failed = _dns(host)
    if failed is not None:
        return failed
    try:
        with socket.create_connection((host, port), timeout=timeout) as raw:
            raw.settimeout(timeout)
            raw.sendall(_request(host))
            payload = _read_some(raw)
    except TimeoutError:
        return "broker_slow"
    except OSError as exc:
        return _os_code(exc)
    return _status(payload)


def _request(host: str) -> bytes:
    safe = host.encode("ascii", "ignore") or b"localhost"
    return b"GET / HTTP/1.1\r\nHost: " + safe + b"\r\nConnection: close\r\n\r\n"


def _read_some(raw: socket.socket) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total < 512:
        piece = raw.recv(256)
        if not piece:
            break
        chunks.append(piece)
        total += len(piece)
    return b"".join(chunks)


def _status(payload: bytes) -> str | None:
    line = payload.split(b"\r\n", 1)[0]
    parts = line.split()
    if len(parts) < 2 or not parts[1].isdigit():
        return "broker_down"
    if parts[1] == b"200":
        return None
    return "broker_down"


def _os_code(exc: OSError) -> str:
    if isinstance(exc, TimeoutError):
        return "broker_slow"
    if "timed out" in str(exc).lower():
        return "broker_slow"
    return "broker_down"
