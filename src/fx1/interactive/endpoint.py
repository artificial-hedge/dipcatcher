"""Normalize a pasted model endpoint into a chat-completions URL.

The service origin (scheme, host, port) is kept distinct from the completion
route. No provider hostname is invented here: the caller supplies the host,
and :func:`fx1.serve.backends.hosted_api_url_problem` decides whether that
URL may carry a credential.

``route_added`` describes this call. Re-normalizing ``completion_url`` is
stable for origin, prefix, completion URL, and host; ``route_added`` is then
false because the route is already present.
"""

from __future__ import annotations

import re
import unicodedata
import urllib.parse
from dataclasses import dataclass

from fx1.serve.backends import hosted_api_url_problem

_COMPLETION_ROUTE = "/chat/completions"
_DEFAULT_PREFIX = "/v1"
_MAX_ENDPOINT_CHARS = 2048
_EMBEDDED_REJECT = ("\x00", "\n", "\r")


class EndpointError(ValueError):
    """The endpoint text could not be turned into a completion URL.

    The message is a reason only. It never includes the supplied text,
    userinfo, query, or token.
    """


@dataclass(frozen=True)
class EndpointSpec:
    """A validated completion endpoint.

    ``origin`` is ``scheme://host[:port]`` with scheme and host lower-cased
    and no trailing slash. ``api_prefix`` is the path before
    ``/chat/completions`` (``""`` when the route sits at the root) and never
    ends with ``/``.
    """

    origin: str
    api_prefix: str
    completion_url: str
    host: str
    route_added: bool

    def describe(self) -> str:
        """``POST`` plus the completion URL, for display."""
        return f"POST {self.completion_url}"


def normalize_endpoint(raw: str) -> EndpointSpec:
    """Return the canonical completion endpoint for *raw*.

    A missing path or ``/`` becomes the ``/v1`` prefix. A path that already
    ends in ``/chat/completions`` keeps the prefix before that suffix. Any
    other path is the prefix and ``/chat/completions`` is appended. Duplicate
    slashes collapse; trailing slashes are removed; scheme and host are
    lower-cased; path case is preserved. Percent-encoding, backslashes,
    dot-segments, and non-canonical hosts are rejected.
    """
    cleaned = _clean_input(raw)
    _reject_escapes(cleaned)
    parsed = _parse_http_url(cleaned)
    _reject_decorations(cleaned, parsed)
    host = _host_name(parsed)
    origin = _origin(parsed, host)
    prefix, route_added = _split_prefix(parsed.path)
    completion = _completion_url(origin, prefix)
    _reject_absurd_length(completion)
    _require_safe(completion)
    return EndpointSpec(
        origin=origin,
        api_prefix=prefix,
        completion_url=completion,
        host=host,
        route_added=route_added,
    )


def _clean_input(raw: str) -> str:
    if any(ch in raw for ch in _EMBEDDED_REJECT):
        raise EndpointError("endpoint must not contain newlines or NUL bytes")
    _reject_absurd_length(raw)
    cleaned = "".join(ch for ch in raw if not _is_ignorable(ch))
    if not cleaned:
        raise EndpointError("endpoint must not be empty")
    return cleaned


def _is_ignorable(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch) in {"Cc", "Cf"}


def _reject_absurd_length(text: str) -> None:
    if len(text) > _MAX_ENDPOINT_CHARS:
        raise EndpointError("endpoint must be at most 2048 characters")


def _reject_escapes(cleaned: str) -> None:
    # ``%`` is not decoded. A loopback prefix plus a zone or ``%00`` would
    # otherwise satisfy the literal-address check.
    if "%" in cleaned or "\\" in cleaned:
        raise EndpointError("endpoint must not contain percent-encoding or backslashes")


def _parse_http_url(cleaned: str) -> urllib.parse.ParseResult:
    try:
        parsed = urllib.parse.urlparse(cleaned)
    except ValueError:
        raise EndpointError("must be a well-formed http(s) URL") from None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise EndpointError("must be an http(s) URL")
    return parsed


def _reject_decorations(cleaned: str, parsed: urllib.parse.ParseResult) -> None:
    if parsed.username is not None or parsed.password is not None:
        raise EndpointError("must not embed userinfo credentials")
    if parsed.params or parsed.query or "?" in cleaned:
        raise EndpointError("must not carry params, query, or fragment")
    if parsed.fragment or "#" in cleaned:
        raise EndpointError("must not carry params, query, or fragment")


def _host_name(parsed: urllib.parse.ParseResult) -> str:
    host = parsed.hostname
    if not host:
        raise EndpointError("must name a host")
    lowered = host.lower()
    if not lowered.isascii():
        raise EndpointError("hostname must be ASCII")
    labels = lowered.split(".")
    if any(label.startswith("xn--") for label in labels):
        raise EndpointError("punycode hostnames are not accepted")
    if any(not label for label in labels):
        raise EndpointError("hostname must be a canonical ASCII name")
    return lowered


def _origin(parsed: urllib.parse.ParseResult, host: str) -> str:
    try:
        port = parsed.port
    except ValueError:
        raise EndpointError("port must be 1-65535") from None
    shown = f"[{host}]" if ":" in host else host
    scheme = parsed.scheme.lower()
    if port is None:
        return f"{scheme}://{shown}"
    return f"{scheme}://{shown}:{port}"


def _split_prefix(path: str) -> tuple[str, bool]:
    collapsed = _collapse_path(path)
    _reject_dot_segments(collapsed)
    if collapsed in {"", "/"}:
        return _DEFAULT_PREFIX, True
    if collapsed.endswith(_COMPLETION_ROUTE):
        return collapsed[: -len(_COMPLETION_ROUTE)], False
    return collapsed, True


def _reject_dot_segments(path: str) -> None:
    if any(segment in {".", ".."} for segment in path.split("/")):
        raise EndpointError("endpoint path must not contain dot segments")


def _collapse_path(path: str) -> str:
    if not path:
        return ""
    collapsed = re.sub(r"/{2,}", "/", path)
    if collapsed != "/":
        collapsed = collapsed.rstrip("/")
    return collapsed


def _completion_url(origin: str, prefix: str) -> str:
    return f"{origin}{prefix}{_COMPLETION_ROUTE}"


def _require_safe(url: str) -> None:
    problem = hosted_api_url_problem(url)
    if problem is None:
        return
    if url in problem:
        raise EndpointError("endpoint was rejected")
    raise EndpointError(problem)
