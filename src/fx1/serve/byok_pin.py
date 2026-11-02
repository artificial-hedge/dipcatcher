"""BYOK HTTP pinned to one vetted numeric address.

The connection classes stay importable from ``fx1.serve.backends``.
``_open_byok_pinned`` reads them back from that module so a test patch on
the historical seam is the class that dials.
"""

from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, NoReturn, cast

from quant_fund.utils.net_timeout import bounded_getaddrinfo
from quant_fund.utils.outbound import (
    PRIVATE_ALLOW_CIDRS_ENV,
    OutboundRefused,
    address_permitted,
    parse_allow_cidrs,
    vetted_addresses,
)


def _byok_private_networks_allowed() -> bool:
    """Whether ``FX1_PRIVATE_ALLOW_CIDRS`` names any extra BYOK networks."""
    return bool(os.environ.get(PRIVATE_ALLOW_CIDRS_ENV, "").strip())


class _ByokRequest(urllib.request.Request):
    """Request carrying a BYOK destination-policy snapshot to the wire seam."""

    _fx1_byok_public_only: bool = False
    _fx1_byok_allow_private: bool = False


def _apply_byok_destination_policy(request: _ByokRequest) -> None:
    """Snapshot the process policy onto a request for the transport seam."""
    allow_private = _byok_private_networks_allowed()
    request._fx1_byok_public_only = not allow_private
    request._fx1_byok_allow_private = allow_private


def _normalized_address(raw: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    address = ipaddress.ip_address(raw.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _byok_address_allowed(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address, *, allow_private: bool
) -> bool:
    """Public unicast, plus CIDRs from ``FX1_PRIVATE_ALLOW_CIDRS`` when opted in.

    The allowlist never admits link-local metadata, multicast, unspecified,
    reserved, documentation, benchmarking, carrier-grade NAT, or legacy
    relay/site-local space.
    """
    raw = os.environ.get(PRIVATE_ALLOW_CIDRS_ENV, "") if allow_private else ""
    return address_permitted(address, parse_allow_cidrs(raw))


def _raise_byok_resolution(host: str, exc: OutboundRefused) -> NoReturn:
    if str(exc) == "allowlist CIDR is invalid":
        raise exc
    cause = exc.__cause__
    if isinstance(cause, TimeoutError):
        raise TimeoutError(str(cause)) from exc
    if isinstance(cause, OSError) or str(exc) in {
        "cannot resolve host",
        "destination host did not resolve to an IP address",
    }:
        raise OSError(f"BYOK endpoint host {host!r} did not resolve") from exc
    raise ValueError("BYOK endpoint resolved to a prohibited network address") from exc


def _byok_resolved_addresses(
    host: str,
    port: int,
    *,
    allow_private: bool,
    timeout: float = 10.0,
    local_literals: bool = False,
) -> tuple[str, ...]:
    """Classify a literal before DNS, then return numeric addresses to dial."""

    def resolve(name: str, number: int) -> list[tuple[Any, ...]]:
        return bounded_getaddrinfo(name, number, timeout, type=socket.SOCK_STREAM)

    try:
        if local_literals:
            return vetted_addresses(
                host,
                port,
                None,
                allow_loopback=True,
                allow_private=True,
                resolve_fn=resolve,
            )
        raw = os.environ.get(PRIVATE_ALLOW_CIDRS_ENV, "") if allow_private else ""
        return vetted_addresses(host, port, parse_allow_cidrs(raw), resolve_fn=resolve)
    except OutboundRefused as exc:
        _raise_byok_resolution(host, exc)


class _ByokPinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection pinned to a previously validated numeric address."""

    source_address: tuple[str, int] | None

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._address, self.port), self.timeout, self.source_address
        )


class _ByokPinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS pinned to a numeric address while authenticating the URL host."""

    source_address: tuple[str, int] | None
    _context: ssl.SSLContext

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        context = ssl.create_default_context()
        # HTTPSConnection only sets HTTP/1.1 ALPN when no context is supplied.
        context.set_alpn_protocols(["http/1.1"])
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._address = address

    def connect(self) -> None:
        raw_sock = socket.create_connection(
            (self._address, self.port), self.timeout, self.source_address
        )
        try:
            self.sock = self._context.wrap_socket(raw_sock, server_hostname=self.host)
        except BaseException:
            raw_sock.close()
            raise


def _pinned_connection_cls(scheme: str) -> Callable[..., http.client.HTTPConnection]:
    """Use the class bound on ``backends`` so existing test patches dial."""
    name = "_ByokPinnedHTTPSConnection" if scheme == "https" else "_ByokPinnedHTTPConnection"
    return cast(
        Callable[..., http.client.HTTPConnection],
        getattr(sys.modules["fx1.serve.backends"], name),
    )


def _open_byok_pinned(
    request: urllib.request.Request,
    *,
    timeout_s: float,
    allow_private: bool,
    local_literals: bool = False,
) -> tuple[http.client.HTTPResponse, http.client.HTTPConnection]:
    """Open one direct request pinned to a vetted numeric address.

    Redirects and proxies are not used. ``local_literals`` is the unflagged
    engine policy: loopback and RFC1918/ULA literals stay dialable, and any
    other name must resolve entirely to public unicast.
    """
    parsed = urllib.parse.urlparse(request.full_url)
    host = parsed.hostname
    if host is None:  # construction validation should make this unreachable
        raise urllib.error.URLError(ValueError("BYOK endpoint has no host"))
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = _byok_resolved_addresses(
            host,
            port,
            allow_private=allow_private,
            timeout=timeout_s,
            local_literals=local_literals,
        )
    except (OSError, ValueError) as exc:
        raise urllib.error.URLError(exc) from exc
    target = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
    # urllib's Request normalizes names such as ``Content-Type`` to
    # ``Content-type`` internally. Restore conventional wire casing so this
    # transport remains behavior-compatible with the prior urllib path.
    headers = {
        "-".join(part.capitalize() for part in name.split("-")): value
        for name, value in request.header_items()
    }
    connection_cls = _pinned_connection_cls(parsed.scheme)
    # A credentialed POST is never replayed across DNS answers: a provider
    # may have accepted a body before a read-side transport failure. Resolve
    # and validate the whole answer set, then make exactly one attempt.
    connection = connection_cls(host, port, addresses[0], timeout_s)
    try:
        connection.request(request.get_method(), target, body=request.data, headers=headers)
        return connection.getresponse(), connection
    except (OSError, http.client.HTTPException) as exc:
        connection.close()
        raise urllib.error.URLError(exc) from exc
