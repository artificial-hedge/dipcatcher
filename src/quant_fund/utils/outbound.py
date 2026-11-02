"""Pinned outbound HTTP and path confinement for untrusted inputs.

Untrusted hostnames must resolve entirely to public unicast addresses.
TCP connects to that numeric address, so a later DNS answer cannot rebind
the socket. ``allow_loopback`` and ``allow_private`` apply to IP literals
(and to the name ``localhost`` only when every answer is loopback). A
name is never allowed to resolve into loopback, RFC1918, or link-local
space. Link-local, multicast, reserved, unspecified, carrier-grade NAT,
192.88.99.0/24, and fec0::/10 stay blocked even when private literals
are opted in.

``FX1_PRIVATE_ALLOW_CIDRS`` is a comma-separated CIDR allowlist. It adds
only those networks on top of public unicast. Metadata, link-local,
multicast, reserved, documentation, benchmark, and the legacy ranges
above stay blocked when a listed network contains them. 6to4, Teredo,
well-known NAT64, and IPv4-translated addresses stay blocked too.
Decimal, octal, hex, and compatibility-folded IPv4 literals are
classified before DNS, so a resolver cannot launder them.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

__all__ = [
    "PRIVATE_ALLOW_CIDRS_ENV",
    "OutboundRefused",
    "address_allowed",
    "address_permitted",
    "directory_child",
    "is_public_ip",
    "literal_destination_refusal",
    "open_pinned",
    "parse_allow_cidrs",
    "pinned_connection",
    "public_addresses",
    "read_http",
    "read_http_redirects",
    "repo_file",
    "repo_relative",
]

PRIVATE_ALLOW_CIDRS_ENV = "FX1_PRIVATE_ALLOW_CIDRS"

ResolveFn = Callable[..., Any]

_LEGACY_SPECIAL = (
    ipaddress.ip_network("192.88.99.0/24"),
    ipaddress.ip_network("fec0::/10"),
)
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_RFC1918 = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)
_ULA = (ipaddress.ip_network("fc00::/7"),)
# Documentation, benchmarking, and other special-purpose ranges that Python
# does not mark ``is_reserved``. An allowlist entry must not admit them.
_NEVER_ALLOW = _LEGACY_SPECIAL + (
    _CGNAT,
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("2001:db8::/32"),
    ipaddress.ip_network("2001:10::/28"),
)
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_TRANSLATED = ipaddress.ip_network("::ffff:0:0/96")
_INET_CHARS = frozenset("0123456789abcdefABCDEFxX.")
_DOT_FOLDS = str.maketrans({"。": ".", "．": ".", "｡": ".", "﹒": "."})


class OutboundRefused(ValueError):
    """The destination or path was refused before a byte was read."""


def _unwrap(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _compatible_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """Deprecated IPv4-compatible form ``::a.b.c.d``. ``::`` and ``::1`` are not it."""
    if address.ipv4_mapped is not None or int(address) <= 1:
        return None
    if int(address) >> 32 != 0:
        return None
    return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)


def _transitional(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Tunnel and translation forms an allowlist must not reopen."""
    if not isinstance(address, ipaddress.IPv6Address) or address.ipv4_mapped is not None:
        return False
    if address.sixtofour is not None or address.teredo is not None:
        return True
    if address in _NAT64 or address in _IPV4_TRANSLATED:
        return True
    return _compatible_ipv4(address) is not None


def _host_key(host: str) -> str:
    """Strip brackets, a zone id, and one trailing dot."""
    text = host.strip()
    if text.startswith("[") and "]" in text:
        text = text[1 : text.index("]")]
    text = text.split("%", 1)[0]
    if text.endswith("."):
        text = text[:-1]
    return text


def _ip_or_none(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return _unwrap(ipaddress.ip_address(text))
    except ValueError:
        return None


def _inet_aton_literal(text: str) -> ipaddress.IPv4Address | None:
    """OS integer and dotted-octal/hex forms. Other text is not an address."""
    if not text or any(char not in _INET_CHARS for char in text):
        return None
    try:
        packed = socket.inet_aton(text)
    except OSError:
        return None
    return ipaddress.IPv4Address(packed)


def _parse_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    key = _host_key(host)
    parsed = _ip_or_none(key) or _inet_aton_literal(key)
    if parsed is not None:
        return parsed
    folded = unicodedata.normalize("NFKC", key).translate(_DOT_FOLDS)
    if folded == key:
        return None
    return _ip_or_none(folded) or _inet_aton_literal(folded)


def _legacy_or_cgnat(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if any(address in network for network in _LEGACY_SPECIAL):
        return True
    return isinstance(address, ipaddress.IPv4Address) and address in _CGNAT


def _blocked_special(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    )


def _rfc1918_or_ula(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    networks = _RFC1918 if isinstance(address, ipaddress.IPv4Address) else _ULA
    return any(address in network for network in networks)


def is_public_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for global unicast after unwrapping IPv4-mapped IPv6."""
    address = _unwrap(address)
    if _transitional(address) or _legacy_or_cgnat(address):
        return False
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


def address_allowed(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_loopback: bool = False,
    allow_private: bool = False,
) -> bool:
    """Literal policy. Private opt-in is RFC1918 and IPv6 ULA only."""
    address = _unwrap(address)
    if _transitional(address) or _legacy_or_cgnat(address):
        return False
    if address.is_loopback:
        return allow_loopback
    if _blocked_special(address):
        return False
    if _rfc1918_or_ula(address):
        return allow_private
    return is_public_ip(address)


def _always_blocked(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Ranges an allowlist must not reopen. Loopback is reserved on IPv6."""
    if _transitional(address):
        return True
    if address.is_link_local or address.is_multicast or address.is_unspecified:
        return True
    if address.is_reserved and not address.is_loopback:
        return True
    return any(address in network for network in _NEVER_ALLOW)


def address_permitted(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    networks: Iterable[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    """Public unicast, plus addresses that fall inside ``networks``.

    Metadata, link-local, multicast, unspecified, reserved, documentation,
    benchmark, carrier-grade NAT, 192.88.99.0/24, and fec0::/10 stay blocked
    even when a network contains them. Loopback and RFC1918/ULA are allowed
    only when a listed network contains them.
    """
    address = _unwrap(address)
    if _always_blocked(address):
        return False
    if is_public_ip(address):
        return True
    return any(address in network for network in networks)


def parse_allow_cidrs(
    raw: str,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Parse a comma-separated CIDR allowlist. Blank items are skipped.

    A bare address is a single-host network. Invalid text fails closed
    without echoing the token.
    """
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            networks.append(ipaddress.ip_network(text, strict=False))
        except ValueError:
            raise OutboundRefused("allowlist CIDR is invalid") from None
    return tuple(networks)


def literal_destination_refusal(
    host: str, *, allow_loopback: bool, allow_private: bool
) -> str | None:
    """Refuse a disallowed IP literal. Hostnames return None."""
    literal = _parse_literal(host)
    if literal is None:
        return None
    if address_allowed(literal, allow_loopback=allow_loopback, allow_private=allow_private):
        return None
    return "destination address is not allowed"


def _localhost_name(host: str) -> bool:
    return host.lower().rstrip(".") == "localhost"


def _resolve(host: str, port: int, resolve_fn: ResolveFn) -> Any:
    try:
        return resolve_fn(host, port)
    except OSError as exc:
        raise OutboundRefused("cannot resolve host") from exc


def _answer_address(info: object) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not isinstance(info, tuple) or len(info) < 5:
        return None
    family, socktype = info[0], info[1]
    sockaddr = info[4]
    if family not in (socket.AF_INET, socket.AF_INET6) or socktype != socket.SOCK_STREAM:
        return None
    if not isinstance(sockaddr, tuple) or not sockaddr:
        return None
    try:
        return _unwrap(ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0]))
    except ValueError:
        return None


def _name_ok(address: ipaddress.IPv4Address | ipaddress.IPv6Address, *, localhost: bool) -> bool:
    if localhost:
        return bool(address.is_loopback)
    return is_public_ip(address)


def _filter_answers(infos: Any, *, localhost: bool) -> list[str]:
    found: list[str] = []
    for info in infos:
        address = _answer_address(info)
        if address is None:
            continue
        if not _name_ok(address, localhost=localhost):
            raise OutboundRefused("destination host resolved to a non-public address")
        normalized = str(address)
        if normalized not in found:
            found.append(normalized)
    return found


def _literal_addresses(
    literal: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_loopback: bool,
    allow_private: bool,
) -> tuple[str, ...]:
    if not address_allowed(literal, allow_loopback=allow_loopback, allow_private=allow_private):
        raise OutboundRefused("destination address is not allowed")
    return (str(literal),)


def _resolved_public(
    host: str,
    port: int,
    *,
    allow_loopback: bool,
    resolve_fn: ResolveFn,
) -> tuple[str, ...]:
    infos = _resolve(host, port, resolve_fn)
    localhost = allow_loopback and _localhost_name(host)
    addresses = _filter_answers(infos, localhost=localhost)
    if not addresses:
        raise OutboundRefused("destination host did not resolve to an IP address")
    return tuple(addresses)


def _resolver(resolve_fn: ResolveFn | None) -> ResolveFn:
    """Pick the resolver at call time so tests can patch ``socket.getaddrinfo``."""
    return socket.getaddrinfo if resolve_fn is None else resolve_fn


def public_addresses(
    host: str,
    port: int,
    *,
    allow_loopback: bool = False,
    allow_private: bool = False,
    resolve_fn: ResolveFn | None = None,
) -> tuple[str, ...]:
    """Numeric addresses for ``host``. Any disallowed answer refuses the set."""
    literal = _parse_literal(host)
    if literal is not None:
        return _literal_addresses(
            literal, allow_loopback=allow_loopback, allow_private=allow_private
        )
    return _resolved_public(
        host, port, allow_loopback=allow_loopback, resolve_fn=_resolver(resolve_fn)
    )


def _require_http(parsed: urllib.parse.ParseResult) -> str:
    if parsed.scheme not in {"http", "https"}:
        raise OutboundRefused("scheme is not http(s)")
    if parsed.username is not None or parsed.password is not None:
        raise OutboundRefused("url must not embed credentials")
    if not parsed.hostname:
        raise OutboundRefused("url names no host")
    return parsed.hostname


def _port_for(parsed: urllib.parse.ParseResult) -> int:
    if parsed.port is not None:
        return int(parsed.port)
    if parsed.scheme == "https":
        return 443
    return 80


def _request_target(parsed: urllib.parse.ParseResult) -> str:
    path = parsed.path or "/"
    if parsed.params:
        path = f"{path};{parsed.params}"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return path


def _wire_headers(request: urllib.request.Request) -> dict[str, str]:
    return {
        "-".join(part.capitalize() for part in name.split("-")): value
        for name, value in request.header_items()
    }


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection whose TCP destination is a validated numeric address."""

    source_address: tuple[str, int] | None

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._address, self.port), self.timeout, self.source_address
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS pinned to a numeric address while authenticating the URL host."""

    source_address: tuple[str, int] | None
    _context: ssl.SSLContext

    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        context = ssl.create_default_context()
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


def pinned_connection(
    scheme: str, host: str, port: int, address: str, timeout: float
) -> http.client.HTTPConnection:
    """Connection that dials ``address`` and sends ``host`` as the Host/SNI name."""
    if scheme == "https":
        return _PinnedHTTPSConnection(host, port, address, timeout)
    if scheme == "http":
        return _PinnedHTTPConnection(host, port, address, timeout)
    raise OutboundRefused("scheme is not http(s)")


class _PinnedHandle:
    def __init__(
        self,
        *,
        status: int,
        reason: str,
        headers: http.client.HTTPMessage,
        response: http.client.HTTPResponse,
        url: str,
        connection: http.client.HTTPConnection,
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers
        self._response = response
        self._url = url
        self._connection = connection
        self._closed = False

    def read(self, amt: int | None = None) -> bytes:
        if amt is None:
            return self._response.read()
        return self._response.read(amt)

    def geturl(self) -> str:
        return self._url

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._response.close()
        self._connection.close()


def _send(
    connection: http.client.HTTPConnection,
    request: urllib.request.Request,
    parsed: urllib.parse.ParseResult,
) -> http.client.HTTPResponse:
    connection.request(
        request.get_method(),
        _request_target(parsed),
        body=request.data,
        headers=_wire_headers(request),
    )
    return connection.getresponse()


@contextmanager
def open_pinned(
    request: urllib.request.Request,
    *,
    timeout: float,
    allow_loopback: bool = False,
    allow_private: bool = False,
    resolve_fn: ResolveFn | None = None,
) -> Iterator[_PinnedHandle]:
    """Open one direct request. Redirects are left to the caller."""
    parsed = urllib.parse.urlparse(request.full_url)
    host = _require_http(parsed)
    port = _port_for(parsed)
    addresses = public_addresses(
        host,
        port,
        allow_loopback=allow_loopback,
        allow_private=allow_private,
        resolve_fn=resolve_fn,
    )
    connection = pinned_connection(parsed.scheme, host, port, addresses[0], float(timeout))
    try:
        response = _send(connection, request, parsed)
    except (OSError, http.client.HTTPException, ValueError, TypeError, RuntimeError):
        connection.close()
        raise
    handle = _PinnedHandle(
        status=int(response.status),
        reason=str(response.reason or ""),
        headers=response.headers,
        response=response,
        url=request.full_url,
        connection=connection,
    )
    try:
        yield handle
    finally:
        handle.close()


def _http_error(handle: _PinnedHandle, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        handle.geturl(),
        int(handle.status),
        handle.reason or "error",
        handle.headers,
        BytesIO(body),
    )


def read_http(
    request: urllib.request.Request,
    *,
    timeout: float,
    allow_loopback: bool = False,
    allow_private: bool = False,
    resolve_fn: ResolveFn | None = None,
) -> bytes:
    """Read a response body. Redirects are refused. HTTP errors raise ``HTTPError``."""
    with open_pinned(
        request,
        timeout=timeout,
        allow_loopback=allow_loopback,
        allow_private=allow_private,
        resolve_fn=resolve_fn,
    ) as handle:
        body = handle.read()
        if 300 <= int(handle.status) < 400:
            raise OutboundRefused("redirects are not followed")
        if int(handle.status) >= 400:
            raise _http_error(handle, body)
        return body


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_REDIRECT_HEADERS = frozenset({"user-agent", "accept", "content-type"})


def _redirect_headers(request: urllib.request.Request, *, drop_body: bool) -> dict[str, str]:
    """Copy only safe headers onto the next hop. Host is never forwarded."""
    headers: dict[str, str] = {}
    for name, value in request.header_items():
        lowered = name.lower()
        if lowered not in _REDIRECT_HEADERS:
            continue
        if drop_body and lowered == "content-type":
            continue
        headers[name] = value
    return headers


def _redirect_request(
    request: urllib.request.Request, location: str, status: int
) -> urllib.request.Request:
    """Next hop. 301/302/303 become GET; 307/308 keep the method and body."""
    drop_body = status in {301, 302, 303}
    method = "GET" if drop_body else request.get_method()
    data = None if drop_body else request.data
    return urllib.request.Request(
        urllib.parse.urljoin(request.full_url, location.strip()),
        data=data,
        headers=_redirect_headers(request, drop_body=drop_body),
        method=method,
    )


def _finish_http(handle: _PinnedHandle, body: bytes) -> bytes:
    if int(handle.status) >= 400:
        raise _http_error(handle, body)
    return body


def read_http_redirects(
    request: urllib.request.Request,
    *,
    timeout: float,
    max_redirects: int = 5,
    allow_loopback: bool = False,
    allow_private: bool = False,
    resolve_fn: ResolveFn | None = None,
) -> bytes:
    """Follow redirects by re-pinning every hop. Other 3xx responses are bodies."""
    current = request
    followed = 0
    while True:
        with open_pinned(
            current,
            timeout=timeout,
            allow_loopback=allow_loopback,
            allow_private=allow_private,
            resolve_fn=resolve_fn,
        ) as handle:
            body = handle.read()
            status = int(handle.status)
            if status not in _REDIRECT_STATUSES:
                return _finish_http(handle, body)
            location = handle.headers.get("Location")
        if followed >= max_redirects:
            raise OutboundRefused("too many redirects")
        if not location or not str(location).strip():
            raise OutboundRefused("redirect missing Location")
        current = _redirect_request(current, str(location), status)
        followed += 1


def repo_relative(root: Path, relative: str | Path) -> Path:
    """Resolve ``relative`` and require it to stay inside ``root``."""
    root_resolved = Path(root).resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise OutboundRefused("path escapes the repository root")
    resolved = (root_resolved / candidate).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise OutboundRefused("path escapes the repository root") from exc
    return resolved


def repo_file(root: str | Path, relative: str | Path) -> tuple[Path, Path]:
    """Return ``(resolved root, confined file)``. Raises ``ValueError`` on escape."""
    root_resolved = Path(root).resolve()
    try:
        target = repo_relative(root_resolved, relative)
    except OutboundRefused as exc:
        raise ValueError(str(exc)) from exc
    return root_resolved, target


def directory_child(directory: Path, name: str) -> Path:
    """One path component inside ``directory``, after resolving symlinks."""
    if not name or name in {".", ".."} or any(sep in name for sep in ("/", "\\", "\x00")):
        raise OutboundRefused("name must be a single path component")
    if Path(name).name != name:
        raise OutboundRefused("name must be a single path component")
    parent = Path(directory).resolve()
    child = (parent / name).resolve()
    if child.parent != parent:
        raise OutboundRefused("name escapes the directory")
    return child
