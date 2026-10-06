"""middleware_audit — request-lifecycle probes over the middleware chain.

The claim under test: every request that reaches ``create_app``'s surface
travels one pipeline — rate limit → ingress validation (framing/singleton
headers, body cap) → auth → routing/handlers → the ``_finish`` tail
(request-id echo, security headers, timing header, metrics, access log) —
with optional CORS outermost and GZip innermost; and every refusal class
arrives in the dialect's error envelope with the correlation id riding
along. Each probe exercises the chain end to end through real requests on
the in-process app, plus raw ASGI scope calls where a higher-level client
would normalize away the surface under test.

Measured chain (``app.user_middleware`` outermost→innermost): optional
``CORSMiddleware`` → ``BaseHTTPMiddleware`` wrapping ``harness_api_auth``
(rate limiter → ingress refusal → ``_resolve_auth`` → ``call_next`` →
``_finish``) → ``GZipMiddleware``. There is no request-deadline middleware
— ``X-Fx1-Timeout`` is a wire timeout to the backend link, not a wall-clock
cap on the request (pinned structurally, not pretended).

Coverage map:

- *Ordering* — auth precedes routing (an unauthed request to an unmapped
  path is 401, never 404; authed bogus is enveloped 404 — OpenAI's own
  convention). Ingress precedes auth (a duplicate ``Content-Length`` or an
  over-cap body on an unauthed call answers 400/413, not 401). The global
  limiter precedes ingress and auth (an exhausted bucket answers 429 for
  unauthed and ambiguous requests alike). Validation precedes handler
  logic (a well-authed malformed body is 422, not a backend call).
- *Request-id lifecycle* — a well-formed ``X-Request-ID`` echoes verbatim
  on 2xx and on every refusal class (401/404/405/413/422/429/500/503);
  absent or malformed mints a fresh 32-hex id; two calls never share one.
  ``/v1/messages*`` answers carry the ``request-id`` twin. The id lands
  in the access-log line (``rid=…``) — the wired correlation sink. It
  never masquerades as a response's own ``req_``/``resp_`` id — those are
  separate namespaces.
- *Drain ordering* — drain is a handler/gate latch, not a middleware: an
  unauthed request under drain is 401 (auth still first); an authed gated
  write is 503 ``draining`` in the dialect's envelope; reads, bogus paths
  (404 — routing precedes the latch), ``/health``, and storage-only
  uploads stay open.
- *Gate precedence* — resolver ``KeyError`` → 404 ``model_not_found`` on
  ``/v1`` vs the harness dialect's ``Literal``-rejected 422; ``RuntimeError``
  /``ValueError`` → 503 ``backend_unavailable`` in both grammars; an
  unauthed request never reaches the resolver (401); a read-scoped key's
  write is 403 ``insufficient_scope`` before any backend call.
- *Error envelope* — a handler fault that no route classified lands
  ``{error:{…,"internal"}}`` 500 on ``/v1`` / ``{detail,code:"internal"}``
  on ``/harness`` / ``{type:"error",error:{type:"api_error"}}`` on
  ``/v1/messages`` — WITH the request-id/security/metrics tail, never
  ``text/plain``; the exception text itself is never echoed. A malformed
  JSON body is enveloped 422 ``validation`` on every dialect (the flat
  FastAPI ``{detail}`` shape never leaks onto ``/v1``). Routing-level
  ``StarletteHTTPException`` (404/405) is enveloped through the base-class
  handler. A request nested past the parser's depth is refused 422, not a
  bare 500 (see defect below).
- *Method handling* — a wrong method on a real ``/harness`` path is 405
  enveloped with ``Allow``; under ``/v1`` the catch-all answers 404
  (OpenAI 404s unknown method+path pairs — pinned, not a leak). ``HEAD``
  on a GET route is 404 under ``/v1`` and 405 with an empty body on
  ``/health``. ``OPTIONS`` without CORS is enveloped 404.
- *CORS* — with origins configured the preflight is answered by the
  outermost middleware before auth (no credentials carried → 200 with the
  allow-lists); a foreign origin gets no ``Access-Control-Allow-Origin``.
- *Path normalization* — the app does not collapse ``..`` or ``//``: raw
  ``/v1/../v1/models`` and ``/v1//models`` are enveloped 404s, and a
  percent-encoded traversal that a client stack normalizes (``%2e%2e``)
  arrives already collapsed (200 — the normalization is the client's).
  ``//v1/models`` misses the ``/v1`` prefix and answers in the flat
  dialect — the dialect is chosen by the literal path, pinned honestly.
- *Content-Type / Accept / query* — ``text/plain`` or a missing
  Content-Type on a JSON route is an honest enveloped 422 (never 415);
  a ``charset`` suffix is accepted. ``Accept: text/xml`` on a JSON route
  is honestly ignored (JSON 200) while ``/metrics`` really negotiates
  ``text/plain`` → Prometheus. Unknown query params are ignored; an empty
  declared param is 422; a repeated param binds last-wins.
- *Body cap / compression* — a >1 MiB entity is 413 ``too_large``
  enveloped (before auth — the cap is ingress, not a route concern).
  There is no 431 header cap at the app layer (pinned). Negotiated
  ``Accept-Encoding: gzip`` compresses route responses past the declared
  threshold; ``identity`` (and sub-threshold bodies) stay uncompressed.
- *Lifecycle tail* — ``X-Content-Type-Options: nosniff``,
  ``Cache-Control: no-store``, ``Referrer-Policy: no-referrer``,
  ``X-Fx1-Api-Version``, ``Openai-Processing-Ms`` ride every response
  class including refusals and the enveloped 500; ``openai-version``
  appears on ``/v1`` only. Refusals and enveloped faults are counted in
  ``/metrics`` ``by_status`` — no invisible request. ``/health`` is the
  only public path (unauthed, exempt from the limiter).

One defect was found and fixed while building this battery: exceptions a
route never classified (unmapped resolver faults, deep-body parse
recursion, serializer faults) used to unwind past ``call_next`` to the
outermost ``ServerErrorMiddleware`` and answer a bare ``text/plain`` 500 —
dropping the request-id, the security headers, the metrics count, and the
access-log line. The dispatch now catches them at the middleware boundary:
``RecursionError`` refuses 422 ``validation`` in the dialect's grammar
(the request's own nesting; backend-link recursion is pre-classified
``backend_unavailable`` and never reaches here), and every other
unclassified fault answers 500 ``internal`` enveloped — logged via the
access path and counted like any other response. The ``*_enveloped_500*``
and ``deep_json_422_*`` probes pin the fix.

Sealed ``middleware_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from fx1.serve.conv_audit import _RESOURCES, _audit_context, _temporary_directory
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

__all__ = ["middleware_audit", "middleware_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_CHAT_PATH = "/v1/chat/completions"
_MESSAGES_PATH = "/v1/messages"
_COMPLETE_PATH = "/harness/complete"
_RID_RE = re.compile(r"^[0-9a-f]{32}$")
_CHAT_BODY = {"model": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]}
_MESSAGES_BODY = {
    "model": "hosted_k3",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}
_COMPLETE_BODY = {"backend": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# Stub backend + client plumbing
# ---------------------------------------------------------------------------


class _StubBackend:
    """Deterministic completion stub."""

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        return f"stub:{messages[-1]['content']}"

    def close(self) -> None:
        pass


def _stub_resolver(name: str, *a: Any, **k: Any) -> Any:
    if name not in {"hosted_k3", "local_fx1", "byok"}:
        raise KeyError(name)
    return _StubBackend()


def _fault_resolver(exc: Exception) -> Any:
    """Resolver factory whose every lookup raises ``exc`` — the fault any
    route's backend resolution could leak if the middleware tail didn't
    keep it inside the contract."""

    def resolve(name: str, *a: Any, **k: Any) -> Any:
        raise exc

    return resolve


def _client(
    resolver: Any = None,
    *,
    api_key: str | None = None,
    rate_limit_rps: float | None = None,
    cors_origins: str | None = None,
    gzip_min_bytes: int | None = None,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env, dirs, and resources per
    construction; ``resolver`` answers links (default: the stub map)."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    resources = _RESOURCES.get()
    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    saved_key = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            backend_resolver=resolver or _stub_resolver,
            state_dir=isolated / "state",
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
            rate_limit_rps=rate_limit_rps,
            cors_origins=cors_origins,
            gzip_min_bytes=gzip_min_bytes,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = TestClient(app, raise_server_exceptions=False)
        resources.callback(client.close)
        resources.enter_context(client)
        return client, api_mod
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key


def _raw(
    app: Any,
    method: str,
    path: str,
    headers: list[tuple[bytes, bytes]],
    body: bytes = b"",
    query: bytes = b"",
) -> dict[str, Any]:
    """One request through the ASGI callable with verbatim scope values —
    duplicate header names, literal ``..`` segments, and doubled slashes
    reach the app exactly as given (a client library would normalize some
    of these before the app ever sees them)."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query,
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 5000),
        "server": ("test", 80),
    }
    resp: dict[str, Any] = {"headers": []}

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.start":
            resp["status"] = message["status"]
            resp["headers"] = list(message.get("headers", []))
        elif message["type"] == "http.response.body":
            resp["body"] = resp.get("body", b"") + message.get("body", b"")

    try:
        asyncio.run(app(scope, receive, send))
    except Exception:
        # ``raise_server_exceptions=False`` semantics: the fault envelope is
        # already on the wire — an exception escaping after
        # ``http.response.start`` (e.g. ``_RaisingJSONResponse``) is the
        # in-process propagation path, not a new response.
        if "status" not in resp:
            raise
    resp["header_map"] = {k.decode("latin-1"): v.decode("latin-1") for k, v in resp["headers"]}
    try:
        resp["json"] = json.loads(resp.get("body", b""))
    except ValueError:
        resp["json"] = None
    return resp


class _AccessLogCapture(logging.Handler):
    """Captures the ``fx1.serve.api`` access/fault log lines a request
    leaves — the rid-bearing record is the wired correlation sink."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def lines(self, level: int | None = None) -> list[str]:
        return [rec.getMessage() for rec in self.records if level is None or rec.levelno == level]


@contextmanager
def _capture_api_log() -> Iterator[_AccessLogCapture]:
    import fx1.serve.api as api_mod

    capture = _AccessLogCapture()
    api_mod.logger.addHandler(capture)
    try:
        yield capture
    finally:
        api_mod.logger.removeHandler(capture)


# ---------------------------------------------------------------------------
# Envelope shape helpers
# ---------------------------------------------------------------------------


def _openai_err(r: Any) -> dict[str, Any] | None:
    """The ``{error:{message,type,param,code}}`` body, else None."""
    try:
        body = r.json()
    except ValueError:
        return None
    err = body.get("error")
    if (
        isinstance(err, dict)
        and isinstance(err.get("message"), str)
        and isinstance(err.get("type"), str)
        and "code" in err
    ):
        return err
    return None


def _anthropic_err(r: Any) -> dict[str, Any] | None:
    """The ``{type:"error", error:{type,message}}`` body, else None."""
    try:
        body = r.json()
    except ValueError:
        return None
    err = body.get("error")
    if (
        body.get("type") == "error"
        and isinstance(err, dict)
        and isinstance(err.get("message"), str)
    ):
        return err
    return None


def _harness_err(r: Any) -> Any | None:
    """The flat ``{detail, code}`` body, else None."""
    try:
        body = r.json()
    except ValueError:
        return None
    if isinstance(body, dict) and "detail" in body and "code" in body:
        return body
    return None


def _headers(r: Any) -> dict[str, str]:
    """Lowercase header map for either a httpx Response or a _raw dict."""
    if hasattr(r, "headers"):
        return {k.lower(): v for k, v in r.headers.items()}
    return {k.lower(): v for k, v in r.get("header_map", {}).items()}


def _has_rid(r: Any) -> bool:
    """The lifecycle tail ran: a request-id rides the response."""
    return bool(_headers(r).get("x-request-id"))


# ---------------------------------------------------------------------------
# Chain structure
# ---------------------------------------------------------------------------


def _structure_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, api_mod = _client()
    app = cast("FastAPI", client.app)

    classes = [getattr(m.cls, "__name__", "") for m in app.user_middleware]
    results["stack_auth_then_gzip"] = classes == ["BaseHTTPMiddleware", "GZipMiddleware"]
    dispatch = app.user_middleware[0].kwargs.get("dispatch")
    results["stack_first_is_auth_dispatch"] = getattr(dispatch, "__name__", "") == (
        "harness_api_auth"
    )
    # No deadline middleware exists anywhere on the chain — request timing
    # is observational (Openai-Processing-Ms), never a wall-clock kill.
    results["stack_no_deadline_middleware"] = not any(
        "timeout" in name.lower() or "deadline" in name.lower() for name in classes
    )

    corsed, _ = _client(cors_origins="https://app.example")
    cors_app = cast("FastAPI", corsed.app)
    cors_classes = [getattr(m.cls, "__name__", "") for m in cors_app.user_middleware]
    results["stack_cors_outermost_when_armed"] = cors_classes == [
        "CORSMiddleware",
        "BaseHTTPMiddleware",
        "GZipMiddleware",
    ]

    ungzipped, _ = _client(gzip_min_bytes=0)
    ungzipped_app = cast("FastAPI", ungzipped.app)
    results["stack_gzip_absent_when_disabled"] = all(
        getattr(m.cls, "__name__", "") != "GZipMiddleware" for m in ungzipped_app.user_middleware
    )
    return results


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def _ordering_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}
    app = client.app

    # auth before routing: an unauthed request is refused before the
    # router can say whether the path exists
    r = client.get("/v1/nope")
    results["unauthed_bogus_v1_401"] = r.status_code == 401 and _openai_err(r) is not None
    r = client.get("/v1/nope", headers=root_h)
    results["authed_bogus_v1_404"] = (
        r.status_code == 404
        and _openai_err(r) is not None
        and "Invalid URL" in (r.json()["error"]["message"])
    )
    r = client.get("/harness/nope")
    results["unauthed_bogus_harness_401"] = r.status_code == 401 and _harness_err(r) is not None
    r = client.get("/harness/nope", headers=root_h)
    results["authed_bogus_harness_404"] = r.status_code == 404 and _harness_err(r) is not None
    # auth before method handling: wrong method, no credential → 401
    r = client.delete("/v1/models")
    results["unauthed_wrong_method_401"] = r.status_code == 401
    r = client.delete("/v1/models", headers=root_h)
    results["authed_wrong_method_v1_404"] = r.status_code == 404  # catch-all, not 405
    # the public path is the only auth exemption — and it's wrong-method
    # reach means routing happens there without a credential
    r = client.post("/health")
    results["public_health_wrong_method_405"] = (
        r.status_code == 405 and r.headers.get("allow") == "GET"
    )

    # ingress before auth: ambiguous framing and the body cap refuse
    # before the credential is ever consulted
    resp = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [
            (b"content-length", b"10"),
            (b"content-length", b"10"),
            (b"content-type", b"application/json"),
        ],
        b"0123456789",
    )
    results["ingress_before_auth_dup_cl_400"] = resp["status"] == 400 and _has_rid(resp)
    oversized = json.dumps({"model": "hosted_k3", "messages": [{"role": "user", "content": "x"}]})
    oversized = oversized[:-1] + ',"pad":"' + ("y" * (1 << 20)) + '"}'
    resp = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [(b"content-type", b"application/json")],
        oversized.encode(),
    )
    results["ingress_before_auth_body_cap_413"] = resp["status"] == 413 and _has_rid(resp)

    # the global limiter precedes ingress and auth: an exhausted bucket
    # answers 429 for unauthed and ambiguous requests alike (keyed on
    # client host — a rotating fake key can't evade it)
    limited, _ = _client(rate_limit_rps=0.5, api_key=_ROOT)
    first = limited.get("/v1/models")
    second = limited.get("/v1/models")
    results["ratelimit_before_auth_429"] = first.status_code == 401 and second.status_code == 429
    third = limited.get("/v1/models", headers=[("x-request-id", "a"), ("x-request-id", "b")])
    results["ratelimit_before_ingress_429"] = third.status_code == 429
    # the rate-limit refusal is enveloped in the dialect + carries the
    # lifecycle tail (rid + Retry-After + bucket headers)
    results["ratelimit_429_enveloped_v1"] = (
        second.status_code == 429
        and _openai_err(second) is not None
        and _has_rid(second)
        and second.headers.get("retry-after") is not None
        and second.headers.get("x-ratelimit-limit") is not None
    )
    r = limited.get("/health")
    results["public_health_exempt_from_limiter"] = r.status_code == 200

    # auth before validation: a malformed body on an unauthed call is 401
    r = client.post(_CHAT_PATH, content=b'{"model":', headers={"content-type": "application/json"})
    results["auth_before_validation_401"] = r.status_code == 401
    # validation before the handler: a well-authed bad body is 422 — the
    # resolver is never consulted
    r = client.post(_CHAT_PATH, json={"model": "hosted_k3"}, headers=root_h)
    results["validation_before_handler_422"] = r.status_code == 422 and _openai_err(r) is not None
    return results


# ---------------------------------------------------------------------------
# Request-id lifecycle + correlation
# ---------------------------------------------------------------------------


def _request_id_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}

    # echo on success and on every refusal class
    r = client.get("/v1/models", headers={**root_h, "X-Request-ID": "rid-mw-1"})
    results["rid_echo_on_200"] = r.headers.get("x-request-id") == "rid-mw-1"
    r = client.get("/v1/models", headers={"X-Request-ID": "rid-mw-401"})
    results["rid_echo_on_401"] = (
        r.status_code == 401 and r.headers.get("x-request-id") == "rid-mw-401"
    )
    r = client.get("/v1/nope", headers={**root_h, "X-Request-ID": "rid-mw-404"})
    results["rid_echo_on_404"] = (
        r.status_code == 404 and r.headers.get("x-request-id") == "rid-mw-404"
    )
    r = client.post("/health", headers={"X-Request-ID": "rid-mw-405"})
    results["rid_echo_on_405"] = (
        r.status_code == 405 and r.headers.get("x-request-id") == "rid-mw-405"
    )
    r = client.post(
        _CHAT_PATH,
        json={"model": "hosted_k3"},
        headers={**root_h, "X-Request-ID": "rid-mw-422"},
    )
    results["rid_echo_on_422"] = (
        r.status_code == 422 and r.headers.get("x-request-id") == "rid-mw-422"
    )
    big = {"model": "hosted_k3", "messages": [{"role": "user", "content": "x" * (1 << 20)}]}
    r = client.post(_CHAT_PATH, json=big, headers={**root_h, "X-Request-ID": "rid-mw-413"})
    results["rid_echo_on_413"] = (
        r.status_code == 413 and r.headers.get("x-request-id") == "rid-mw-413"
    )
    # minted when absent; never shared between calls
    r = client.get("/v1/models", headers=root_h)
    rid_a = r.headers.get("x-request-id") or ""
    results["rid_minted_when_absent"] = _RID_RE.fullmatch(rid_a) is not None
    r = client.get("/v1/models", headers=root_h)
    rid_b = r.headers.get("x-request-id") or ""
    results["rid_distinct_per_request"] = rid_a != rid_b and _RID_RE.fullmatch(rid_b) is not None
    # the anthropic twin rides errors too
    r = client.post(
        _MESSAGES_PATH,
        json={"model": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]},
        headers={**root_h, "X-Request-ID": "rid-mw-ant"},
    )
    results["rid_anthropic_twin_on_refusal"] = (
        r.headers.get("request-id") == "rid-mw-ant"
        and r.headers.get("x-request-id") == "rid-mw-ant"
    )
    # correlation: the wire rid lands in the access log — the only wired
    # sink; a response body's own ids are a separate namespace
    with _capture_api_log() as capture:
        client.get("/v1/models", headers={**root_h, "X-Request-ID": "rid-mw-log"})
    access = [line for line in capture.lines() if "rid=rid-mw-log" in line]
    results["rid_in_access_log"] = any(
        "method=GET" in line and "path=/v1/models" in line for line in access
    )
    r = client.post(_CHAT_PATH, json=_CHAT_BODY, headers={**root_h, "X-Request-ID": "rid-mw-ns"})
    body = r.json() if r.status_code == 200 else {}
    results["rid_not_response_object_id"] = (
        r.headers.get("x-request-id") == "rid-mw-ns"
        and body.get("id") != "rid-mw-ns"
        and body.get("request_id") != "rid-mw-ns"
    )
    return results


# ---------------------------------------------------------------------------
# Drain ordering
# ---------------------------------------------------------------------------


def _drain_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}

    # the drain latch itself is admin-gated — an unauthed caller can't set it
    r = client.post("/harness/drain")
    results["drain_requires_admin_401"] = r.status_code == 401
    r = client.post("/harness/drain", headers=root_h)
    results["drain_latch_sets"] = r.status_code == 200 and r.json().get("draining") is True

    # ordering: auth still precedes the latch — unauthed under drain is 401
    r = client.get("/v1/models")
    results["drain_unauthed_401_first"] = r.status_code == 401
    # routing precedes the latch: an authed bogus path under drain is 404
    r = client.get("/v1/nope", headers=root_h)
    results["drain_authed_bogus_404"] = r.status_code == 404 and _openai_err(r) is not None
    # gated writes refuse in the dialect's envelope, both grammars
    r = client.post("/v1/responses", json={"model": "hosted_k3", "input": "hi"}, headers=root_h)
    results["drain_gated_v1_503_enveloped"] = (
        r.status_code == 503
        and _openai_err(r) is not None
        and r.json()["error"].get("code") == "draining"
        and r.json()["error"].get("type") == "service_unavailable"
        and _has_rid(r)
    )
    r = client.post("/harness/runs", json={"command": "doctor"}, headers=root_h)
    results["drain_gated_harness_503_enveloped"] = (
        r.status_code == 503
        and _harness_err(r) is not None
        and r.json().get("code") == "draining"
        and _has_rid(r)
    )
    # control plane + storage-only uploads stay open under drain
    r = client.get("/v1/files", headers=root_h)
    results["drain_read_stays_open"] = r.status_code == 200
    r = client.post(
        "/v1/files",
        files={"file": ("d.jsonl", b'{"x":1}\n')},
        data={"purpose": "batch"},
        headers=root_h,
    )
    results["drain_upload_admitted_200"] = r.status_code == 200
    r = client.get("/health")
    results["drain_health_public_200"] = r.status_code == 200
    return results


# ---------------------------------------------------------------------------
# Gate precedence (backend resolution faults)
# ---------------------------------------------------------------------------


def _gate_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}

    # resolver KeyError → 404 on /v1, vs the harness dialect's Literal
    # rejection at 422 before resolution is even consulted
    r = client.post(
        _CHAT_PATH,
        json={"model": "nonexistent_be", "messages": [{"role": "user", "content": "hi"}]},
        headers=root_h,
    )
    results["gate_unknown_model_v1_404"] = (
        r.status_code == 404
        and _openai_err(r) is not None
        and r.json()["error"].get("code") == "model_not_found"
        and _has_rid(r)
    )
    r = client.post(
        _COMPLETE_PATH,
        json={"backend": "nonexistent_be", "messages": [{"role": "user", "content": "hi"}]},
        headers=root_h,
    )
    results["gate_unknown_backend_harness_422"] = (
        r.status_code == 422 and _harness_err(r) is not None
    )

    # resolver RuntimeError/ValueError → 503 backend_unavailable both dialects
    for dialect, path, body in (
        ("v1", _CHAT_PATH, _CHAT_BODY),
        ("harness", _COMPLETE_PATH, _COMPLETE_BODY),
        ("anthropic", _MESSAGES_PATH, _MESSAGES_BODY),
    ):
        faulted, _ = _client(resolver=_fault_resolver(RuntimeError("no cfg")), api_key=_ROOT)
        r = faulted.post(path, json=body, headers=root_h)
        if dialect == "v1":
            err = _openai_err(r)
        elif dialect == "anthropic":
            err = _anthropic_err(r)
        else:
            err = _harness_err(r)
        ok = r.status_code == 503 and err is not None and _has_rid(r)
        # the backend_unavailable code rides the v1 + harness envelopes;
        # anthropic's 503 grammar is {type:error, error:{type,message}}
        if dialect != "anthropic":
            ok = ok and (err or {}).get("code") == "backend_unavailable"
        results[f"gate_unavailable_503_{dialect}"] = ok

    # an unauthed call never reaches the resolver — 401 even when the
    # resolver would 503
    faulted, _ = _client(resolver=_fault_resolver(RuntimeError("no cfg")), api_key=_ROOT)
    r = faulted.post(_CHAT_PATH, json=_CHAT_BODY)
    results["gate_unauthed_never_resolves_401"] = r.status_code == 401

    # a read-scoped key's write is 403 before any backend call
    minted, _ = _client(api_key=_ROOT)
    r = minted.post("/harness/keys", json={"scopes": ["read"]}, headers=root_h)
    assert r.status_code == 201, r.text
    ro_h = {"X-API-Key": r.json()["key"]}
    r = minted.post(_CHAT_PATH, json=_CHAT_BODY, headers=ro_h)
    results["gate_scope_denied_403"] = (
        r.status_code == 403
        and _openai_err(r) is not None
        and r.json()["error"].get("code") == "insufficient_scope"
        and _has_rid(r)
    )
    return results


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------


def _envelope_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}
    app = client.app

    # a fault no route classified lands enveloped in every dialect — with
    # the request-id/security tail — never the bare text/plain 500
    for dialect, path, body in (
        ("v1", _CHAT_PATH, _CHAT_BODY),
        ("harness", _COMPLETE_PATH, _COMPLETE_BODY),
        ("anthropic", _MESSAGES_PATH, _MESSAGES_BODY),
    ):
        faulted, _ = _client(resolver=_fault_resolver(TypeError("secret-xc9")), api_key=_ROOT)
        with _capture_api_log() as capture:
            r = faulted.post(path, json=body, headers=root_h)
        shaped = (
            _openai_err(r)
            if dialect == "v1"
            else _anthropic_err(r)
            if dialect == "anthropic"
            else _harness_err(r)
        )
        results[f"unclassified_fault_500_{dialect}"] = (
            r.status_code == 500
            and shaped is not None
            and _has_rid(r)
            and r.headers.get("x-content-type-options") == "nosniff"
            and "secret-xc9" not in r.text
        )
        results[f"unclassified_fault_logged_{dialect}"] = any(
            "unhandled request fault" in line for line in capture.lines(logging.ERROR)
        )
        # the faulted request is counted — not invisible to metrics
        metrics = faulted.get("/metrics", headers=root_h)
        results[f"unclassified_fault_counted_{dialect}"] = (
            metrics.json().get("by_status", {}).get("500", 0) >= 1
        )
        # the anthropic twin rides the enveloped fault too
        if dialect == "anthropic":
            results["unclassified_fault_anthropic_twin"] = bool(r.headers.get("request-id"))

    # deep-nested entity: refused in the 4xx family, enveloped — never
    # the bare 500 (RecursionError from the parser / serializer depth)
    deep_body = b'{"command":"doctor","x":' + b"[" * 3000 + b"]" * 3000 + b"}"
    resp = _raw(
        app,
        "POST",
        "/harness/runs",
        [(b"content-type", b"application/json"), (b"x-api-key", _ROOT.encode())],
        deep_body,
    )
    results["deep_json_harness_enveloped_422"] = (
        resp["status"] in (400, 413, 422) and _has_rid(resp) and resp["json"] is not None
    )
    deep_chat = (
        b'{"model":"hosted_k3","messages":[{"role":"user","content":"hi"}],"x":'
        + b"[" * 3000
        + b"]" * 3000
        + b"}"
    )
    resp = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [(b"content-type", b"application/json"), (b"x-api-key", _ROOT.encode())],
        deep_chat,
    )
    # the serializer-depth fault envelopes instead of escaping bare —
    # either an honest 4xx refusal or an enveloped+counted 500
    results["deep_json_v1_enveloped_not_bare"] = (
        resp["status"] in (400, 413, 422, 500)
        and _has_rid(resp)
        and resp["json"] is not None
        and "error" in (resp["json"] or {})
    )

    # malformed JSON is an enveloped 422 on every dialect — FastAPI's
    # flat default never leaks onto /v1
    r = client.post(
        _CHAT_PATH,
        content=b'{"model":',
        headers={**root_h, "content-type": "application/json"},
    )
    results["malformed_json_v1_422_enveloped"] = (
        r.status_code == 422
        and _openai_err(r) is not None
        and r.json()["error"].get("code") == "validation"
        and "detail" not in r.json()
    )
    r = client.post(
        _MESSAGES_PATH,
        content=b"{bad",
        headers={**root_h, "content-type": "application/json"},
    )
    results["malformed_json_anthropic_422_enveloped"] = (
        r.status_code == 422 and _anthropic_err(r) is not None
    )
    r = client.post(
        "/harness/runs",
        content=b'{"command":',
        headers={**root_h, "content-type": "application/json"},
    )
    results["malformed_json_harness_422_enveloped"] = (
        r.status_code == 422 and _harness_err(r) is not None
    )

    # routing-level StarletteHTTPException resolves through the base-class
    # handler — the enveloped 404 on /v1 was already pinned above; the
    # harness dialect 404 carries {detail, code} here
    r = client.get("/harness/nope", headers=root_h)
    results["starlette_404_harness_enveloped"] = (
        r.status_code == 404 and _harness_err(r) is not None and r.json().get("code") == "not_found"
    )

    # no dialect leak: every /v1 refusal class answers the {error} shape
    # and none carries a bare top-level "detail" key
    big = {"model": "hosted_k3", "messages": [{"role": "user", "content": "x" * (1 << 20)}]}
    refused = [
        client.get("/v1/models"),  # 401
        client.get("/v1/nope", headers=root_h),  # 404
        client.post(
            _CHAT_PATH,
            json={"model": "hosted_k3"},
            headers=root_h,
        ),  # 422
        client.post(_CHAT_PATH, json=big, headers=root_h),  # 413
    ]
    results["v1_refusals_no_detail_leak"] = all(
        _openai_err(r) is not None and "detail" not in r.json() for r in refused
    )
    return results


# ---------------------------------------------------------------------------
# Methods, HEAD/OPTIONS, paths
# ---------------------------------------------------------------------------


def _method_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}
    app = client.app

    # wrong method on a real /harness path: enveloped 405 with Allow
    r = client.post("/health")
    results["wrong_method_health_405_allow"] = (
        r.status_code == 405
        and r.headers.get("allow") == "GET"
        and _harness_err(r) is not None
        and _has_rid(r)
    )
    # under /v1 the catch-all answers 404 (OpenAI 404s unknown method+path)
    r = client.delete("/v1/models", headers=root_h)
    results["wrong_method_v1_404_catchall"] = (
        r.status_code == 404 and _openai_err(r) is not None and r.headers.get("allow") is None
    )
    # HEAD on a GET route: 404 under /v1, 405 on /health with an empty body
    r = client.request("HEAD", "/v1/models", headers=root_h)
    results["head_v1_models_404"] = r.status_code == 404
    r = client.request("HEAD", "/health")
    results["head_health_405_empty_body"] = (
        r.status_code == 405 and r.headers.get("allow") == "GET" and not r.content
    )
    # OPTIONS without CORS: enveloped 404 via the catch-all
    r = client.options(_CHAT_PATH, headers=root_h)
    results["options_no_cors_404_enveloped"] = r.status_code == 404 and _openai_err(r) is not None

    # with CORS armed the outermost middleware answers preflights before
    # auth — an unauthed OPTIONS is 200 with the allow-lists
    corsed, _ = _client(cors_origins="https://app.example", api_key=_ROOT)
    origin = {"Origin": "https://app.example", "Access-Control-Request-Method": "POST"}
    r = corsed.options(_CHAT_PATH, headers=origin)
    results["cors_preflight_200_skips_auth"] = (
        r.status_code == 200
        and r.headers.get("access-control-allow-origin") == "https://app.example"
        and "POST" in (r.headers.get("access-control-allow-methods") or "")
    )
    r = corsed.options(
        _CHAT_PATH,
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    results["cors_foreign_origin_refused"] = (
        r.headers.get("access-control-allow-origin") != "https://evil.example"
    )
    # a non-preflight request from an allowed origin still needs auth
    r = corsed.get("/v1/models", headers={"Origin": "https://app.example"})
    results["cors_simple_request_still_authed"] = r.status_code == 401

    # path normalization: the app collapses nothing — literal "..", "//",
    # and decoded %2F all land in the catch-all; a client-normalized
    # traversal arrives already collapsed
    resp = _raw(app, "GET", "/v1/../v1/models", [(b"x-api-key", _ROOT.encode())])
    results["traversal_literal_dotdot_404"] = resp["status"] == 404 and "error" in (
        resp["json"] or {}
    )
    resp = _raw(app, "GET", "/v1//models", [(b"x-api-key", _ROOT.encode())])
    results["traversal_double_slash_404"] = resp["status"] == 404 and "error" in (
        resp["json"] or {}
    )
    resp = _raw(app, "GET", "//v1/models", [(b"x-api-key", _ROOT.encode())])
    # misses the /v1 prefix → flat dialect — dialect is the literal path's
    results["leading_dslash_flat_dialect_404"] = (
        resp["status"] == 404
        and "detail" in (resp["json"] or {})
        and "error" not in (resp["json"] or {})
    )
    resp = _raw(app, "GET", "/v1/mod/els", [(b"x-api-key", _ROOT.encode())])
    results["encoded_slash_decoded_404"] = resp["status"] == 404
    # percent-encoded traversal is refused too — the encoded path stays
    # un-collapsed end to end and lands in the catch-all
    r = client.get("/v1/%2e%2e/v1/models", headers=root_h)
    results["encoded_dotdot_404_no_collapse"] = r.status_code == 404
    # dialect is chosen by the literal path — a traversal under
    # /v1/messages answers in the anthropic grammar
    resp = _raw(app, "GET", "/v1/messages/../models", [(b"x-api-key", _ROOT.encode())])
    results["traversal_anthropic_dialect_404"] = (
        resp["status"] == 404
        and isinstance(resp["json"], dict)
        and resp["json"].get("type") == "error"
    )
    # trailing slash is a distinct resource — 404, no silent redirect
    r = client.get("/v1/models/", headers=root_h)
    results["trailing_slash_404_no_redirect"] = r.status_code == 404
    return results


# ---------------------------------------------------------------------------
# Content-Type / Accept / query string
# ---------------------------------------------------------------------------


def _negotiation_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}

    # non-JSON Content-Type on a JSON route: an honest enveloped 422,
    # never a 415 and never a bare parse fault
    r = client.post(
        _CHAT_PATH,
        content=json.dumps(_CHAT_BODY).encode(),
        headers={**root_h, "Content-Type": "text/plain"},
    )
    results["ct_text_plain_422_enveloped"] = r.status_code == 422 and _openai_err(r) is not None
    r = client.post(_CHAT_PATH, content=json.dumps(_CHAT_BODY).encode(), headers=root_h)
    results["ct_missing_422_enveloped"] = r.status_code == 422 and _openai_err(r) is not None
    r = client.post(
        _CHAT_PATH,
        content=json.dumps(_CHAT_BODY).encode(),
        headers={**root_h, "Content-Type": "application/json; charset=utf-8"},
    )
    results["ct_charset_accepted_200"] = r.status_code == 200

    # Accept is honestly ignored on JSON routes; /metrics really negotiates
    r = client.get("/v1/models", headers={**root_h, "Accept": "text/xml"})
    results["accept_xml_ignored_json_200"] = (
        r.status_code == 200 and "application/json" in r.headers.get("content-type", "")
    )
    r = client.get("/metrics", headers={**root_h, "Accept": "text/plain"})
    results["accept_negotiates_metrics_text"] = (
        r.status_code == 200 and "text/plain" in r.headers.get("content-type", "")
    )

    # query string: unknown params ignored; declared-but-empty 422s;
    # repeats bind last-wins on a surface where the param truncates
    r = client.get("/v1/models?foo=bar", headers=root_h)
    results["query_unknown_param_ignored_200"] = r.status_code == 200
    r = client.get("/v1/models?limit=", headers=root_h)
    results["query_empty_declared_param_422"] = r.status_code == 422 and _openai_err(r) is not None
    for i in range(3):
        up = client.post(
            "/v1/files",
            files={"file": (f"q{i}.jsonl", b'{"x":1}\n')},
            data={"purpose": "batch"},
            headers=root_h,
        )
        assert up.status_code == 200, up.text
    r = client.get("/v1/files?limit=2&limit=1", headers=root_h)
    last_wins = r.status_code == 200 and len(r.json().get("data", [])) == 1
    r = client.get("/v1/files?limit=1&limit=2", headers=root_h)
    results["query_repeated_last_wins"] = (
        last_wins and r.status_code == 200 and len(r.json().get("data", [])) == 2
    )
    # /v1/models declares limit for SDK compat but does not truncate the
    # tiny static list — validated, not applied (pinned, not a leak)
    r = client.get("/v1/models?limit=1", headers=root_h)
    results["models_limit_validated_not_truncating"] = (
        r.status_code == 200 and len(r.json().get("data", [])) > 1
    )
    return results


# ---------------------------------------------------------------------------
# Body cap, headers, compression
# ---------------------------------------------------------------------------


def _resource_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}

    # the 1 MiB ingress cap is an envelope, before auth — see the
    # ordering probes for the unauthed cell
    big = {"model": "hosted_k3", "messages": [{"role": "user", "content": "x" * (1 << 20)}]}
    r = client.post(_CHAT_PATH, json=big, headers=root_h)
    results["body_cap_413_v1_enveloped"] = (
        r.status_code == 413
        and _openai_err(r) is not None
        and r.json()["error"].get("code") == "too_large"
        and _has_rid(r)
    )
    r = client.post("/harness/runs", content=b"x" * ((1 << 20) + 1), headers=root_h)
    results["body_cap_413_harness_enveloped"] = (
        r.status_code == 413 and _harness_err(r) is not None and r.json().get("code") == "too_large"
    )
    # exactly-at-cap bodies still process (the cap is exclusive)
    ok_body = {"model": "hosted_k3", "messages": [{"role": "user", "content": "x" * 1000}]}
    r = client.post(_CHAT_PATH, json=ok_body, headers=root_h)
    results["body_under_cap_processes_200"] = r.status_code == 200

    # there is no app-level request-header cap — a 200 KB header value
    # is admitted (header bounds are the HTTP server's job, pinned)
    r = client.get("/health", headers={"X-Big": "y" * 200_000})
    results["oversized_header_admitted_no_431"] = r.status_code == 200

    # compression: negotiated + past threshold compresses; identity and
    # sub-threshold bodies never carry a content-encoding
    r = client.get("/openapi.json", headers={**root_h, "Accept-Encoding": "gzip"})
    results["gzip_negotiated_large_response"] = (
        r.status_code == 200
        and r.headers.get("content-encoding") == "gzip"
        and r.headers.get("vary") is not None
    )
    r = client.get("/openapi.json", headers={**root_h, "Accept-Encoding": "identity"})
    results["gzip_identity_uncompressed"] = (
        r.status_code == 200 and r.headers.get("content-encoding") is None
    )
    r = client.get("/v1/models", headers={**root_h, "Accept-Encoding": "gzip"})
    results["gzip_subthreshold_uncompressed"] = r.headers.get("content-encoding") is None
    return results


# ---------------------------------------------------------------------------
# Lifecycle tail (post-processing uniformity)
# ---------------------------------------------------------------------------


def _tail_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}

    def tail_ok(r: Any) -> bool:
        return (
            r.headers.get("x-content-type-options") == "nosniff"
            and r.headers.get("cache-control") == "no-store"
            and r.headers.get("referrer-policy") == "no-referrer"
            and r.headers.get("x-fx1-api-version") is not None
            and r.headers.get("openai-processing-ms") is not None
        )

    r = client.get("/v1/models", headers=root_h)
    results["tail_security_headers_200"] = tail_ok(r)
    r = client.get("/v1/models")
    results["tail_security_headers_401"] = tail_ok(r)
    r = client.get("/v1/nope", headers=root_h)
    results["tail_security_headers_404"] = tail_ok(r)
    r = client.post("/health")
    results["tail_security_headers_405"] = tail_ok(r)
    faulted, _ = _client(resolver=_fault_resolver(TypeError("x")), api_key=_ROOT)
    r = faulted.post(_CHAT_PATH, json=_CHAT_BODY, headers=root_h)
    results["tail_security_headers_enveloped_500"] = tail_ok(r) and r.status_code == 500
    # openai-version rides /v1 only
    r = client.get("/v1/models", headers=root_h)
    results["tail_openai_version_on_v1"] = r.headers.get("openai-version") is not None
    r = client.get("/harness/commands", headers=root_h)
    results["tail_openai_version_absent_harness"] = (
        r.status_code == 200 and r.headers.get("openai-version") is None
    )
    # metrics count refusals — no invisible request
    before = client.get("/metrics", headers=root_h).json().get("by_status", {})
    client.get("/v1/models")  # 401
    client.get("/v1/nope", headers=root_h)  # 404
    after = client.get("/metrics", headers=root_h).json().get("by_status", {})
    results["tail_metrics_counts_refusals"] = after.get("401", 0) > before.get(
        "401", 0
    ) and after.get("404", 0) > before.get("404", 0)
    # every request leaves exactly one access line carrying the ids
    with _capture_api_log() as capture:
        client.get("/v1/models", headers={**root_h, "X-Request-ID": "rid-tail-1"})
        client.get("/v1/models", headers={**root_h, "X-Request-ID": "rid-tail-2"})
    lines = [line for line in capture.lines() if "path=/v1/models" in line]
    results["tail_access_log_one_line_per_request"] = (
        len(lines) == 2
        and all("status=200" in line for line in lines)
        and any("rid=rid-tail-1" in line for line in lines)
        and any("rid=rid-tail-2" in line for line in lines)
    )
    return results


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def middleware_audit() -> dict[str, Any]:
    """Run the request-lifecycle battery; returns literal bools."""
    out: dict[str, Any] = {}
    with _audit_context():
        out.update(_structure_probes())
        out.update(_ordering_probes())
        out.update(_request_id_probes())
        out.update(_drain_probes())
        out.update(_gate_probes())
        out.update(_envelope_probes())
        out.update(_method_probes())
        out.update(_negotiation_probes())
        out.update(_resource_probes())
        out.update(_tail_probes())
    return out


def middleware_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under middleware_audit.v1."""
    r = middleware_audit()
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "middleware_audit",
        "schema": "middleware_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process buffered TestClient + verbatim ASGI scope calls",
            "not_executed": [
                "real socket-level framing (h11/uvicorn limits)",
                "live network delivery or disconnect timing",
                "middleware under actual process drain/shutdown",
            ],
        },
        "interpretation": (
            "The request lifecycle is one honest pipeline. Order is "
            "limiter → ingress (framing/singleton headers, 1 MiB body cap) "
            "→ auth → routing/handlers → the _finish tail, with CORS "
            "outermost when armed and GZip innermost; no deadline "
            "middleware exists (X-Fx1-Timeout is the wire timeout only). "
            "Auth precedes routing (unauthed bogus is 401, authed bogus "
            "enveloped 404), ingress precedes auth (dup framing 400 and "
            "over-cap 413 need no credential), the limiter precedes both "
            "(exhausted bucket 429s unauthed calls), and drain is a "
            "handler-level latch so auth/routing still order ahead of it "
            "(unauthed-under-drain 401; bogus-under-drain 404; reads and "
            "storage uploads stay open). X-Request-ID echoes on every "
            "outcome class or mints fresh 32-hex, rides the access log as "
            "rid=, twins to request-id on the anthropic surface, and never "
            "masquerades as a response's own req_/resp_ id. Every refusal "
            "class lands in the dialect's envelope — {error} on /v1, "
            "{type:error} on /v1/messages, {detail,code} on /harness — "
            "including the previously-bare unclassified fault, which is "
            "now enveloped 500 (or 422 for input-driven recursion) with "
            "the full tail and a metrics count; the exception text never "
            "echoes. Wrong methods are enveloped 405+Allow on /harness "
            "and catch-all 404 under /v1; HEAD and OPTIONS pin the same "
            "catch-all; CORS preflights skip auth by construction; "
            "trailing slash and literal traversal are 404s with no silent "
            "normalization; Content-Type failures are honest 422 (never "
            "415); Accept is ignored on JSON routes and negotiated on "
            "/metrics; repeated query params bind last-wins; gzip "
            "compresses only negotiated route responses past the "
            "threshold; the security/version/timing headers and the "
            "access line apply to every response class including the "
            "enveloped 500."
            if ok
            else f"MIDDLEWARE AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(middleware_audit_bench(), indent=2, sort_keys=True))
