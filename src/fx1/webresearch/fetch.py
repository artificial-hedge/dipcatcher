"""Web research — stdlib HTTP fetching with strict hygiene.

No new dependencies: ``urllib`` does the transport. Hygiene rules are
structural, not advisory:

- http/https only; ``file://`` and other schemes are refused.
- Private, loopback, link-local, and reserved addresses are refused even
  when a public URL redirects to them — research fetches can never become
  an SSRF lane into the operator's network.
- Responses are capped at ``max_bytes``; oversized bodies are truncated and
  flagged, never silently swallowed.
- Redirects are capped; exceeding the cap fails the fetch loudly.
- ``robots_allowed`` implements the conservative subset of robots.txt that a
  polite crawler needs (user-agent groups + prefix ``Disallow`` lines),
  best-effort: an unreachable robots.txt degrades to allowed, a present one
  is honored.

Every network seam is injectable (``open_fn`` / ``fetch_fn`` / ``resolve_fn``)
so the engine and its tests never need a live socket.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

DEFAULT_USER_AGENT = "dipcatcher-research/0.1 (+https://github.com/artificial-hedge/dipcatcher)"
DEFAULT_MAX_BYTES = 512 * 1024
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MAX_REDIRECTS = 5
ACCEPT = "text/html, text/plain;q=0.9, */*;q=0.1"

OpenFn = Callable[[urllib.request.Request, float], Any]  # -> context manager over response
ResolveFn = Callable[..., list[tuple[Any, ...]]]  # socket.getaddrinfo shape


class FetchPolicyError(Exception):
    """The fetch was refused by policy (scheme, address, robots) — not a network fault."""


@dataclass
class FetchResult:
    """One fetched page: what arrived, and what policy/transport stopped."""

    url: str
    final_url: str = ""
    status: int | None = None
    content_type: str = ""
    title: str = ""
    text: str = ""
    links: list[str] = field(default_factory=list)
    size_bytes: int = 0
    truncated: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 400 and not self.error


# ---------------------------------------------------------------------------
# Address policy
# ---------------------------------------------------------------------------


def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return ip.is_global


def _public_address(host: str, port: int, resolve_fn: ResolveFn) -> tuple[str | None, str]:
    """Choose one address only after checking every DNS answer for private IPs."""
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _ip_is_public(literal):
            return str(literal), ""
        return None, f"address {host} is not a public internet address"
    try:
        infos = resolve_fn(host, port)
    except OSError as exc:
        return None, f"cannot resolve host ({type(exc).__name__})"
    selected: str | None = None
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if not _ip_is_public(ip):
            return None, f"host {host} resolves to non-public address {addr}"
        selected = selected or str(ip)
    if selected is None:
        return None, f"host {host} has no public address"
    return selected, ""


def host_is_public(host: str, *, resolve_fn: ResolveFn = socket.getaddrinfo) -> tuple[bool, str]:
    """(allowed, reason) for fetching ``host``. Literal IPs need no resolution."""
    address, reason = _public_address(host, 80, resolve_fn)
    return address is not None, reason


def _url_host_port(url: str) -> tuple[str | None, int | None, str | None]:
    """Validate the URL authority without looking up its host."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None, None, "url has an invalid host or port"
    if parsed.scheme not in {"http", "https"}:
        return None, None, f"scheme {parsed.scheme!r} is not http(s)"
    if parsed.username is not None or parsed.password is not None:
        return None, None, "url must not embed credentials"
    if not host:
        return None, None, "url names no host"
    return host, port if port is not None else (443 if parsed.scheme == "https" else 80), None


def url_policy_error(url: str, *, resolve_fn: ResolveFn = socket.getaddrinfo) -> str | None:
    """Why *url* may not be fetched; ``None`` when it may. Never echoes secrets."""
    host, port, error = _url_host_port(url)
    if error is not None:
        return error
    assert host is not None and port is not None
    address, reason = _public_address(host, port, resolve_fn)
    if address is None:
        return reason
    return None


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """Connect to the checked IP; retain the URL hostname for HTTP Host."""

    def __init__(self, host: str, *, pinned_ip: str, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = self._create_connection(
            (self._pinned_ip, self.port), self.timeout, self.source_address
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Pin TCP while keeping the original host for SNI and certificate checks."""

    def __init__(self, host: str, *, pinned_ip: str, **kwargs: Any) -> None:
        super().__init__(host, **kwargs)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = self._create_connection(
            (self._pinned_ip, self.port), self.timeout, self.source_address
        )
        # ProxyHandler({}) disables proxies; retain the standard HTTPS identity
        # when wrapping the socket, rather than treating the IP as the peer name.
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


def _approved_request_ip(req: urllib.request.Request, resolve_fn: ResolveFn) -> str:
    host, port, error = _url_host_port(req.full_url)
    if error is not None:
        raise FetchPolicyError(f"refused by fetch policy: {error}")
    assert host is not None and port is not None
    # This lookup is the one used for the socket, including on redirects.
    address, reason = _public_address(host, port, resolve_fn)
    if address is None:
        raise FetchPolicyError(f"refused by fetch policy: {reason}")
    return address


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, resolve_fn: ResolveFn) -> None:
        super().__init__()
        self._resolve_fn = resolve_fn

    def http_open(self, req: urllib.request.Request) -> Any:
        address = _approved_request_ip(req, self._resolve_fn)
        return self.do_open(
            lambda host, **kwargs: _PinnedHTTPConnection(host, pinned_ip=address, **kwargs),
            req,
        )


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, resolve_fn: ResolveFn) -> None:
        super().__init__()
        self._resolve_fn = resolve_fn

    def https_open(self, req: urllib.request.Request) -> Any:
        address = _approved_request_ip(req, self._resolve_fn)
        return self.do_open(
            lambda host, **kwargs: _PinnedHTTPSConnection(host, pinned_ip=address, **kwargs),
            req,
            context=self._context,
        )


class _LimitedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Cap redirects, fail loudly beyond the cap, and re-check fetch policy
    on every hop — a public URL that redirects to a private address must not
    become a lane into the operator's network."""

    def __init__(self, max_redirects: int, resolve_fn: ResolveFn) -> None:
        super().__init__()
        self._max = max_redirects
        self._hops = 0
        self._resolve_fn = resolve_fn

    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        self._hops += 1
        if self._hops > self._max:
            raise urllib.error.HTTPError(
                req.full_url, code, f"too many redirects (>{self._max})", headers, fp
            )
        policy = url_policy_error(newurl, resolve_fn=self._resolve_fn)
        if policy is not None:
            raise FetchPolicyError(f"redirect refused by fetch policy: {policy}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# ---------------------------------------------------------------------------
# HTML -> text
# ---------------------------------------------------------------------------

_SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "tr",
    "br",
    "blockquote",
    "pre",
    "table",
    "ul",
    "ol",
    "header",
    "footer",
}


class _TextExtractor(HTMLParser):
    """Collects title, anchor hrefs, and visible text (blocks separated by newlines)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.links: list[str] = []
        self._chunks: list[str] = []
        self._in_title = False
        self._skip_depth = 0
        self._seen_hrefs: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href and href not in self._seen_hrefs:
                self._seen_hrefs.add(href)
                self.links.append(href)
        if tag in _BLOCK_TAGS and self._chunks and self._chunks[-1] != "\n":
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS and self._chunks and self._chunks[-1] != "\n":
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
        else:
            self._chunks.append(data)

    def text(self) -> str:
        raw = "".join(self._chunks)
        lines = [" ".join(line.split()) for line in raw.split("\n")]
        return "\n".join(line for line in lines if line).strip()


def html_to_text(html: str, *, base_url: str = "") -> tuple[str, str, list[str]]:
    """(title, visible text, absolute links) from one HTML document."""
    extractor = _TextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception:  # noqa: BLE001 — a broken page must still yield what parsed
        pass
    links: list[str] = []
    for href in extractor.links:
        if base_url:
            href = urllib.parse.urljoin(base_url, href)
        parsed = urllib.parse.urlparse(href)
        if parsed.scheme in {"http", "https"}:
            links.append(href)
    return extractor.title.strip(), extractor.text(), links


# ---------------------------------------------------------------------------
# robots.txt (conservative subset)
# ---------------------------------------------------------------------------

_robots_cache: dict[str, str | None] = {}
_robots_lock = threading.Lock()


def _robots_disallowed(robots_text: str, user_agent: str, path: str) -> bool:
    """Prefix ``Disallow`` matching for the group claiming *user_agent* (or ``*``)."""
    ua_lower = user_agent.lower()
    group_match = False
    patterns: list[str] = []
    for raw_line in robots_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "user-agent":
            agent = value.lower()
            group_match = agent == "*" or agent in ua_lower or ua_lower in agent
        elif key == "disallow" and group_match and value:
            patterns.append(value)
    for pattern in patterns:
        if pattern == "/":
            return True
        if "*" in pattern:
            import re

            regex = re.escape(pattern).replace(r"\*", ".*")
            if re.fullmatch(regex + r".*", path):
                return True
        elif path.startswith(pattern):
            return True
    return False


def robots_allowed(
    url: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    fetch_fn: Callable[..., FetchResult] | None = None,
    timeout_s: float = 5.0,
    max_bytes: int = 64 * 1024,
) -> bool:
    """Whether robots.txt permits fetching *url*. Unreachable robots → allowed."""
    parsed = urllib.parse.urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    with _robots_lock:
        if origin in _robots_cache:
            cached = _robots_cache[origin]
            return cached is None or not _robots_disallowed(cached, user_agent, parsed.path or "/")
    if fetch_fn is None:

        def fetch_fn(url: str, **kwargs: Any) -> FetchResult:
            return fetch_page(url, timeout_s=timeout_s, max_bytes=max_bytes)

    robots = fetch_fn(f"{origin}/robots.txt")
    robots_text = robots.text if robots.ok else None
    with _robots_lock:
        _robots_cache[origin] = robots_text
    return robots_text is None or not _robots_disallowed(
        robots_text, user_agent, parsed.path or "/"
    )


def clear_robots_cache() -> None:
    with _robots_lock:
        _robots_cache.clear()


# ---------------------------------------------------------------------------
# fetch_page
# ---------------------------------------------------------------------------


def fetch_page(
    url: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    user_agent: str = DEFAULT_USER_AGENT,
    open_fn: OpenFn | None = None,
    resolve_fn: ResolveFn = socket.getaddrinfo,
    check_policy: bool = True,
) -> FetchResult:
    """Fetch one URL with policy, size, and redirect limits. Never raises for
    policy or transport reasons — problems land in ``FetchResult.error``."""
    result = FetchResult(url=url)
    if check_policy:
        policy = url_policy_error(url, resolve_fn=resolve_fn)
        if policy is not None:
            result.error = f"refused by fetch policy: {policy}"
            return result
    if not url.isascii():
        # IRIs must be percent-encoded before hitting the wire; already-encoded
        # triplets survive because % is in the safe set.
        url = urllib.parse.quote(url, safe=":/?&=%+#@")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": ACCEPT})

    def open_response() -> Any:
        if open_fn is not None:
            return open_fn(request, timeout_s)
        # Do not honor ambient HTTP(S)_PROXY: proxy resolution would bypass
        # the validated address and put the network policy outside this process.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _PinnedHTTPHandler(resolve_fn),
            _PinnedHTTPSHandler(resolve_fn),
            _LimitedRedirectHandler(max_redirects, resolve_fn),
        )
        return opener.open(request, timeout=timeout_s)

    try:
        with open_response() as response:
            result.status = getattr(response, "status", 200)
            result.final_url = response.geturl()
            result.content_type = response.headers.get("Content-Type", "")
            if result.content_type and not (
                result.content_type.startswith("text/html")
                or result.content_type.startswith("text/plain")
            ):
                result.error = f"unsupported content type {result.content_type.split(';')[0]!r}"
                return result
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    chunks.append(chunk[: max_bytes - sum(len(c) for c in chunks)])
                    result.truncated = True
                    break
                chunks.append(chunk)
            body = b"".join(chunks)
            result.size_bytes = len(body)
            try:
                decoded = body.decode("utf-8", errors="replace")
            except UnicodeError:
                result.error = "body is not decodable text"
                return result
            if result.content_type.startswith("text/html") or "<html" in decoded[:512].lower():
                title, text, links = html_to_text(decoded, base_url=result.final_url or url)
                result.title = title
                result.text = text
                result.links = links
            else:
                result.text = decoded
    except urllib.error.HTTPError as exc:
        result.status = exc.code
        result.error = f"HTTP {exc.code}"
    except FetchPolicyError as exc:
        result.error = str(exc)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        result.error = f"transport fault ({type(exc).__name__})"
    return result
