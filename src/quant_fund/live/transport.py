"""HTTPS transport restricted to the Alpaca trading and data hosts.

Redirects are refused. Credentials in the URL are refused. The transport
does not log headers.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, OpenerDirector, Request, build_opener

from quant_fund.live.errors import LiveRefused, TransportError

MAX_BODY_BYTES = 1_048_576
PAPER_TRADING_HOST = "paper-api.alpaca.markets"
LIVE_TRADING_HOST = "api.alpaca.markets"
DATA_HOST = "data.alpaca.markets"
PAPER_TRADING_BASE = f"https://{PAPER_TRADING_HOST}"
LIVE_TRADING_BASE = f"https://{LIVE_TRADING_HOST}"
DATA_BASE = f"https://{DATA_HOST}"


def trading_base(environment: str) -> str:
    """Return the only trading origin allowed for this environment."""
    if environment == "paper":
        return PAPER_TRADING_BASE
    if environment == "live":
        return LIVE_TRADING_BASE
    raise LiveRefused("environment must be paper or live")


def assert_broker_url(url: str, *, environment: str) -> None:
    """Fail closed unless the URL is https on the environment's allowlist."""
    parsed = urlsplit(url)
    if parsed.scheme != "https":
        raise LiveRefused("broker URL must be https")
    if parsed.username or parsed.password or parsed.query.find("@") >= 0:
        raise LiveRefused("broker URL must not carry credentials")
    host = parsed.hostname
    allowed = {urlsplit(trading_base(environment)).hostname, DATA_HOST}
    if host not in allowed:
        raise LiveRefused(f"broker host {host!r} is not allowed for {environment}")
    if parsed.port not in (None, 443):
        raise LiveRefused("broker URL port is not allowed")


class _RefuseRedirect(HTTPRedirectHandler):
    """A redirect would leave the host allowlist unchecked."""

    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> Request | None:
        raise LiveRefused(f"broker redirect refused ({code})")


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


class HttpTransport(Protocol):
    """Keyword-only request. Implementations must not persist headers."""

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> HttpResponse: ...


class UrllibTransport:
    """Stdlib HTTPS client. Host allowlist is checked before the socket opens."""

    def __init__(self, *, environment: str, timeout_s: float = 10.0) -> None:
        self.environment = environment
        self.timeout_s = float(timeout_s)
        trading_base(environment)

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_s: float,
    ) -> HttpResponse:
        assert_broker_url(url, environment=self.environment)
        opener: OpenerDirector = build_opener(_RefuseRedirect)
        request = Request(url, data=body, headers=dict(headers), method=method)
        try:
            with opener.open(request, timeout=timeout_s) as response:
                status = int(response.status)
                payload = response.read(MAX_BODY_BYTES + 1)
        except LiveRefused:
            raise
        except HTTPError as exc:
            status = int(exc.code)
            payload = exc.read(MAX_BODY_BYTES + 1)
        except (URLError, TimeoutError, OSError) as exc:
            raise TransportError("broker request did not complete") from exc
        if len(payload) > MAX_BODY_BYTES:
            raise TransportError("broker response exceeds 1 MiB")
        return HttpResponse(status=status, body=payload)


def redact(text: str, *secrets: str) -> str:
    """Remove credential material from a broker message before it is stored."""
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


def response_message(body: bytes, *secrets: str) -> str:
    """Short broker message with secrets stripped. Never the raw headers."""
    text = body[:500].decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and "message" in parsed:
        text = str(parsed["message"])
    return redact(text, *secrets)
