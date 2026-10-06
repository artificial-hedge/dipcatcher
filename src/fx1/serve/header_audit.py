"""header_audit — adversarial probes on the wire's header contract.

The claim under test: the request-header channel is *parseable, honest,
and fail-closed*, and the response-header channel reports exactly what
was emitted — every ``X-Fx1-*`` extension header, plus the generic
``Idempotency-Key`` / ``X-Request-ID`` / ``X-API-Key`` /
``Authorization`` / ``anthropic-*`` / ``X-RateLimit-*`` / CORS surface,
exercised end-to-end through real requests on the in-process app.

Coverage map:

- *Timeout parsing* — ``X-Fx1-Timeout`` accepts ``(0, 3600]`` seconds in
  any ``float()``-parseable spelling (whitespace-padded, exponential,
  ``1_0``) and fails closed 400 on zero, negatives, NaN, inf, overflow,
  empty, and non-numeric values — never a silent default, never a bare
  500. On ``/v1/messages`` the same refusal lands in the Anthropic
  ``{type: "error", error}`` grammar.
- *Timeout plumbing* — the value reaches the resolver as ``timeout_s``;
  body ``fx1.timeout_s`` wins over the header (the body extension is the
  per-request override). On the ``/harness`` surface the header is
  ignored — that dialect binds ``timeout_s`` in the body only.
- *Timeout enforcement* — a BYOK link aimed at an endpoint that accepts
  the socket then sleeps past the deadline refuses ``502
  backend_failure`` inside a second or so, not after the sleeper's own
  delay — the header deadline is the wire timeout, not a decoration.
  ``stream=true`` fails the same way: the gate runs before SSE starts,
  so the refusal is a JSON envelope, never a partial event stream. On a
  batch submit the header applies per-line while a line's own
  ``fx1.timeout_s`` overrides it. A well-formed HTTP answer whose body
  is not JSON, and a read-side disconnect, fail closed the same way —
  every transport fault is ``backend_failure``, never a bare 500
  (defect #4).
- *Checkpoint dir* — ``X-Fx1-Checkpoint-Dir`` binds only on a
  ``local_fx1`` link (422 everywhere else), an empty value refuses, a
  nonexistent path fails closed, and the header overrides
  ``FX1_CHECKPOINT_DIR`` on ``local_fx1`` only. On ``/harness`` the
  header is ignored; the body field refuses non-``local_fx1`` chains.
- *BYOK headers* — ``X-Fx1-Byok-{Base-Url,Api-Key,Model}`` resolve to
  the same kwargs as the ``fx1.byok`` body block; the body block wins on
  conflict; a base-url without an api-key refuses 400; an api-key or
  model without a base-url is ignored (the channel arms on the URL); a
  base-url that fails ``ByokOverride`` validation refuses 422 enveloped
  (see defect #1); ``byok``-marked credentials never appear in refusal
  text; the override on a non-``byok`` link refuses 422; a server with
  ``byok_override=False`` refuses 422 ``byok_override_disabled``.
- *Receipt headers* — ``X-Fx1-Receipt-Sha256`` on completion surfaces is
  the sha256 of the sealed ``fx1_completion_record.v1`` receipt for the
  cited ``X-Fx1-Completion-Id`` (re-derived through
  ``/harness/completions/{id}/receipt``); both are absent on ungated
  surfaces. ``X-Fx1-Receipt-Hashes`` fails closed 400 on a malformed
  digest, 422 ``receipt_not_found`` on a well-formed but unresolvable
  one, accepts a resolvable digest, and treats an all-empty list as
  absent. ``fx1.receipt_hashes`` wins over the header on conflict.
- *Request id* — a valid caller ``X-Request-ID`` echoes verbatim on
  success and refusal; an absent, malformed, over-long, or CRLF-bearing
  value mints a fresh hex id instead of shipping (log-injection cannot
  mint a response header); ambiguous duplicates refuse before
  Starlette's first-wins and the translator's last-wins views can
  disagree. Anthropic surfaces emit the twin
  ``request-id``.
- *Idempotency* — ``Idempotency-Key`` empty or whitespace-only is
  honestly ignored (two identical calls both run); padding is stripped
  so a padded key replays an unpinned-looking twin; >256 chars refuses
  400 enveloped; an obs-text byte is an opaque dictionary key and
  dedups, while control characters refuse. Neither is re-emitted. A replay carries
  ``X-Fx1-Idempotent-Replay: true`` byte-identically; key+body mismatch
  is 409. ``/v1/batches`` submissions dedup on the same key (one batch
  is minted).
- *Anthropic headers* — ``anthropic-version`` switches a ``/v1`` payload
  to the Anthropic grammar by *presence* (the value is not validated —
  pinned); ``anthropic-beta`` is tolerated and ignored; the SDK's
  ``x-api-key`` and ``Authorization: Bearer`` both authenticate
  ``/v1/messages``; ``x-should-retry`` lands on retryable statuses only.
- *Rate-limit headers* — with the global limiter armed,
  ``X-RateLimit-{Limit,Remaining,Reset}`` ride admitted *and* refused
  responses, honestly counting the consumed slot (defect #2);
  ``Retry-After`` lands only on a window refusal, never a hard quota.
  A managed key's own window reports the ``*-Requests`` family on top.
- *Envelope* — every header-parse refusal lands in the dialect's error
  shape: ``{error:{message,type,code}}`` on ``/v1``,
  ``{type:"error",error:{...}}`` on ``/v1/messages``, ``{detail,code}``
  on ``/harness`` — and malformed/ambiguous framing answers in the
  path's grammar too (defect #3). The actual ASGI body is capped without
  trusting ``Content-Length``. Never a bare 400/500.
- *CORS* — with origins configured, a preflight gets the allow-lists
  (``X-Fx1-*`` request headers are declared), a foreign origin is
  refused, and a no-CORS app answers OPTIONS 405 enveloped.

Four defects were found and fixed while building this battery — the
``X-Fx1-Byok-Base-Url`` smuggle to a bare 500, the dropped
``X-RateLimit-*`` headers on managed-key refusals, the flat error shape
on ``Content-Length`` refusals under ``/v1``, and read-side transport
faults (read timeouts, disconnects, malformed JSON payloads) escaping
the ``RuntimeError`` convention to a bare 500 on every backend wire
call (defect #4, pinned by the enforcement and garbage-payload probes).
All four are pinned green below.

Sealed ``header_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fx1.serve.conv_audit import _RESOURCES, _audit_context, _temporary_directory
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["header_audit", "header_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_CHAT_PATH = "/v1/chat/completions"
_MESSAGES_PATH = "/v1/messages"
_COMPLETE_PATH = "/harness/complete"
# Deliberately-insecure literal: BYOK overrides accept http:// for local
# stacks — this URL is never dialed (the injected resolver returns stubs).
_BYOK_BASE_URL = "http://127.0.0.1:9/v1"  # NOSONAR(S5332) — never dialed
_RID_RE = re.compile(r"^[0-9a-f]{32}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Stub backend + client plumbing
# ---------------------------------------------------------------------------


class _StubBackend:
    """Deterministic completion stub — echoes the last user turn."""

    def __init__(self, model: str = "hdr-stub-0") -> None:
        self._model = model
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        self.calls += 1
        return f"stub:{messages[-1]['content']}"

    def embeddings(self, input: Any, *, model: str, **extra: Any) -> Any:  # noqa: A002
        """Echo-shaped ``EmbeddingResult`` for the embeddings surface."""
        from fx1.serve.backends import EmbeddingResult

        del extra
        count = len(input) if isinstance(input, (list, tuple)) else 1
        return EmbeddingResult(
            data=tuple(
                {"object": "embedding", "index": i, "embedding": [0.1, 0.2]} for i in range(count)
            ),
            model=model,
            usage={"prompt_tokens": 1, "total_tokens": 1},
        )

    def close(self) -> None:
        pass


class _SpyResolver:
    """``backend_resolver`` spy — records the ``(name, kwargs)`` each call
    resolved with, so a probe pins exactly which header value reached the
    link layer. Missing names raise ``KeyError`` like ``get_backend``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, name: str, *a: Any, **k: Any) -> Any:
        kwargs = dict(k)
        if a:
            kwargs["checkpoint_dir"] = a[0]
        self.calls.append((name, kwargs))
        if name not in {"hosted_k3", "local_fx1", "byok"}:
            raise KeyError(name)
        return _StubBackend(f"{name}-stub")


def _client(
    resolver: Any = None,
    *,
    api_key: str | None = None,
    rate_limit_rps: float | None = None,
    cors_origins: str | None = None,
    byok_override: bool | None = None,
    real_backends: bool = False,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env, dirs, and resources per
    construction; ``resolver`` (or the module spy) answers links unless
    ``real_backends`` routes resolution to the genuine ``get_backend``."""
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
        kwargs: dict[str, Any] = {}
        if not real_backends:
            kwargs["backend_resolver"] = resolver or _SpyResolver()
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            state_dir=isolated / "state",
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
            rate_limit_rps=rate_limit_rps,
            cors_origins=cors_origins,
            byok_override=byok_override,
            **kwargs,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        app.state.hdr_audit_receipts = receipts
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
) -> dict[str, Any]:
    """One request through the ASGI callable with *verbatim* headers —
    duplicate names, exotic casing, and bytes a browser client refuses to
    send all reach the app unchanged, which is exactly what the header
    contract must survive."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
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

    asyncio.run(app(scope, receive, send))
    resp["header_map"] = {k.decode("latin-1"): v.decode("latin-1") for k, v in resp["headers"]}
    return resp


class _TcpEndpoint:
    """A TCP listener with two behaviors: ``sleep_s`` accepts a connection
    then sleeps past any sane deadline (the link's ``timeout_s`` is what
    must cut it), or ``fixed_body`` answers one canned HTTP response —
    the malformed-payload path's fixture."""

    def __init__(self, sleep_s: float = 30.0, fixed_body: bytes | None = None) -> None:
        self.sleep_s = sleep_s
        self._fixed_body = fixed_body
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._closed = threading.Event()
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                conn.recv(65536)
                if self._fixed_body is not None:
                    head = (
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(self._fixed_body)}\r\n".encode()
                        + b"\r\n"
                    )
                    conn.sendall(head + self._fixed_body)
                else:
                    time.sleep(self.sleep_s)
            except OSError:
                pass
            finally:
                conn.close()

    def close(self) -> None:
        self._closed.set()
        self._sock.close()


def _sleep_endpoint() -> _TcpEndpoint:
    ep = _TcpEndpoint()
    _RESOURCES.get().callback(ep.close)
    return ep


def _garbage_endpoint(body: bytes) -> _TcpEndpoint:
    ep = _TcpEndpoint(fixed_body=body)
    _RESOURCES.get().callback(ep.close)
    return ep


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _chat(client: TestClient, *, headers: dict[str, str] | None = None, **extra: Any) -> Any:
    body = {"model": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return client.post(_CHAT_PATH, json=body, headers=headers or {})


def _messages(client: TestClient, *, headers: dict[str, str] | None = None, **extra: Any) -> Any:
    body = {
        "model": "hosted_k3",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(extra)
    return client.post(_MESSAGES_PATH, json=body, headers=headers or {})


def _complete(client: TestClient, *, headers: dict[str, str] | None = None, **extra: Any) -> Any:
    body = {"backend": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return client.post(_COMPLETE_PATH, json=body, headers=headers or {})


def _mint(client: TestClient, root_h: dict[str, str], **policy: Any) -> tuple[str, str]:
    r = client.post("/harness/keys", json=policy, headers=root_h)
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _openai_err(resp: Any) -> dict[str, Any] | None:
    """The `{error:{message,type,code}}` object, or None when absent."""
    try:
        err = resp.json().get("error")
    except ValueError:
        return None
    return err if isinstance(err, dict) else None


def _anthropic_err(resp: Any) -> dict[str, Any] | None:
    """The Anthropic `{type:"error", error:{type,message}}` object."""
    try:
        body = resp.json()
    except ValueError:
        return None
    if body.get("type") != "error" or not isinstance(body.get("error"), dict):
        return None
    return cast(dict[str, Any], body["error"])


def _flat_err(resp: Any) -> dict[str, Any] | None:
    """The `/harness` `{detail, code}` shape."""
    try:
        body = resp.json()
    except ValueError:
        return None
    # detail is a string, or a pydantic error list on 422s — presence of
    # ``detail`` + ``code`` is the contract; ``detail``'s shape is route-specific
    return body if isinstance(body.get("detail"), (str, list)) and "code" in body else None


# ---------------------------------------------------------------------------
# Timeout: parsing boundary
# ---------------------------------------------------------------------------


def _timeout_parse_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client()

    def chat_timeout(value: str) -> Any:
        return _chat(client, headers={"X-Fx1-Timeout": value})

    # admitted spellings — every float()-parseable value in (0, 3600]
    ok_spellings = {
        "timeout_admits_integer": "5",
        "timeout_admits_decimal": "0.5",
        "timeout_admits_ceiling_3600": "3600",
        "timeout_admits_whitespace_padded": " 7 ",
        "timeout_admits_exponential": "1e2",
        "timeout_admits_plus_sign": "+3",
        "timeout_admits_leading_dot": ".5",
        "timeout_admits_underscore_digit": "1_0",
    }
    for name, raw in ok_spellings.items():
        r = chat_timeout(raw)
        results[name] = r.status_code == 200 and "error" not in r.text.lower()

    # refused spellings — fail closed 400 in the OpenAI envelope
    bad_spellings = {
        "timeout_refuses_zero": "0",
        "timeout_refuses_negative": "-1",
        "timeout_refuses_non_numeric": "abc",
        "timeout_refuses_nan": "NaN",
        "timeout_refuses_inf": "inf",
        "timeout_refuses_negative_inf": "-inf",
        "timeout_refuses_huge": "1e9",
        "timeout_refuses_above_ceiling": "3600.0001",
        "timeout_refuses_empty": "",
        "timeout_refuses_whitespace_only": "   ",
        "timeout_refuses_hex": "0x10",
        "timeout_refuses_comma_decimal": "1,5",
    }
    for name, raw in bad_spellings.items():
        r = chat_timeout(raw)
        err = _openai_err(r)
        results[name] = r.status_code == 400 and err is not None
    results["timeout_refusal_names_header"] = "X-Fx1-Timeout" in str(
        _openai_err(chat_timeout("abc")) or {}
    )
    results["timeout_admitted_never_enveloped_error"] = _openai_err(chat_timeout("5")) is None

    # the same refusal lands in the Anthropic grammar on /v1/messages
    r = _messages(client, headers={"X-Fx1-Timeout": "abc"})
    results["timeout_anthropic_refusal_enveloped"] = (
        r.status_code == 400 and _anthropic_err(r) is not None
    )

    # responses + embeddings surfaces share the same translator path
    r = client.post(
        "/v1/responses",
        json={"model": "hosted_k3", "input": "hi"},
        headers={"X-Fx1-Timeout": "abc"},
    )
    results["timeout_responses_refusal"] = r.status_code == 400 and _openai_err(r) is not None
    r = client.post(
        "/v1/embeddings",
        json={"model": "hosted_k3", "input": "hi"},
        headers={"X-Fx1-Timeout": "abc"},
    )
    results["timeout_embeddings_refusal"] = r.status_code == 400 and _openai_err(r) is not None

    # count_tokens shares the resolver too (the anthropic dialect surface)
    r = client.post(
        "/v1/messages/count_tokens",
        json={"model": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Fx1-Timeout": "abc"},
    )
    results["timeout_count_tokens_refusal"] = r.status_code == 400 and _anthropic_err(r) is not None

    # on the /harness dialect the header is out of band — it is ignored,
    # never parsed: timeout binds only through the body's timeout_s field
    spy = _SpyResolver()
    hclient, _ = _client(spy)
    r = _complete(hclient, headers={"X-Fx1-Timeout": "7"})
    results["timeout_harness_header_ignored"] = (
        r.status_code == 200 and spy.calls[-1][1].get("timeout_s") is None
    )
    r = _complete(hclient, timeout_s=7)
    results["timeout_harness_body_binds"] = (
        r.status_code == 200 and spy.calls[-1][1].get("timeout_s") == 7
    )
    r = _complete(hclient, headers={"X-Fx1-Timeout": "abc"})
    results["timeout_harness_bad_header_not_refused"] = r.status_code == 200

    # stream surface: a malformed header refuses before SSE opens —
    # the refusal is JSON, never a partial event stream
    r = _chat(client, headers={"X-Fx1-Timeout": "abc"}, stream=True)
    results["timeout_stream_refusal_is_json"] = (
        r.status_code == 400
        and "text/event-stream" not in r.headers.get("content-type", "")
        and _openai_err(r) is not None
    )
    return results


# ---------------------------------------------------------------------------
# Timeout + backend + fallbacks: precedence and plumbing
# ---------------------------------------------------------------------------


def _precedence_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    spy = _SpyResolver()
    client, _api = _client(spy)

    def last() -> tuple[str, dict[str, Any]]:
        assert spy.calls
        return spy.calls[-1]

    # body fx1.timeout_s wins over X-Fx1-Timeout
    r = _chat(client, headers={"X-Fx1-Timeout": "7"}, fx1={"timeout_s": 2.5})
    results["timeout_body_overrides_header"] = (
        r.status_code == 200 and last()[1].get("timeout_s") == 2.5
    )
    spy.calls.clear()
    r = _chat(client, headers={"X-Fx1-Timeout": "7"})
    results["timeout_header_binds_without_body"] = (
        r.status_code == 200 and last()[1].get("timeout_s") == 7.0
    )

    # X-Fx1-Backend vs fx1.backend — the body extension wins
    spy.calls.clear()
    r = _chat(client, headers={"X-Fx1-Backend": "local_fx1"}, fx1={"backend": "hosted_k3"})
    results["backend_body_overrides_header"] = r.status_code == 200 and last()[0] == "hosted_k3"
    spy.calls.clear()
    r = _chat(
        client,
        headers={
            "X-Fx1-Backend": "local_fx1",
            "X-Fx1-Checkpoint-Dir": "some/dir",
        },
        model="x",
    )
    results["backend_header_binds_without_body"] = r.status_code == 200 and last()[0] == "local_fx1"
    # model naming a backend fills when neither body nor header speaks
    spy.calls.clear()
    r = _chat(
        client,
        model="byok",
        fx1={"byok": {"base_url": _BYOK_BASE_URL, "api_key": "k", "model": "m"}},
    )
    results["backend_model_names_link"] = r.status_code == 200 and last()[0] == "byok"
    # unknown header value fails closed 400 enveloped
    spy.calls.clear()
    r = _chat(client, headers={"X-Fx1-Backend": "nope"})
    results["backend_unknown_refuses_400"] = (
        r.status_code == 400 and _openai_err(r) is not None and "nope" in str(_openai_err(r))
    )
    results["backend_header_is_case_sensitive"] = (
        _chat(client, headers={"X-Fx1-Backend": "HOSTED_K3"}).status_code == 400
    )
    results["backend_empty_header_ignored"] = (
        _chat(client, headers={"X-Fx1-Backend": ""}).status_code == 200
    )
    results["backend_header_ignored_on_harness"] = (
        _complete(client, headers={"X-Fx1-Backend": "local_fx1"}).status_code == 200
        and spy.calls[-1][0] == "hosted_k3"
    )

    # X-Fx1-Fallbacks — header fills only when the body is silent;
    # body list wins; a bad name in the header refuses via request
    # validation (fallbacks are Literal-typed on CompleteRequest)
    spy.calls.clear()
    r = _chat(client, headers={"X-Fx1-Fallbacks": "local_fx1,byok"})
    results["fallbacks_header_binds"] = r.status_code == 200 and last()[0] == "hosted_k3"
    r = _chat(client, headers={"X-Fx1-Fallbacks": "bogus"})
    err = _openai_err(r)
    results["fallbacks_bad_name_refuses"] = r.status_code in (400, 422) and err is not None
    # exhausted fallback chain records attempts honestly
    results["fallbacks_header_dead_chain_enveloped"] = (
        _chat(client, headers={"X-Fx1-Backend": "nope,nope"}).status_code == 400
    )

    # receipt-hash precedence — body fx1.receipt_hashes wins on conflict:
    # the unresolvable citation named in the refusal is the body's value
    body_hash = "a" * 64
    hdr_hash = "b" * 64
    r = _chat(
        client,
        headers={"X-Fx1-Receipt-Hashes": hdr_hash},
        fx1={"receipt_hashes": [body_hash]},
    )
    msg = str((_openai_err(r) or {}).get("message", ""))
    results["receipt_hashes_body_overrides_header"] = (
        r.status_code == 422 and body_hash in msg and hdr_hash not in msg
    )
    return results


# ---------------------------------------------------------------------------
# Timeout: enforcement on a sleeping endpoint + per-line batch
# ---------------------------------------------------------------------------


def _timeout_enforcement_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    # real backend resolution — the byok link dials a socket that accepts
    # then sleeps past the deadline
    client, _api = _client(real_backends=True)
    ep = _sleep_endpoint()
    byok = {
        "base_url": f"http://127.0.0.1:{ep.port}/v1",  # NOSONAR(S5332) — local sleeper
        "api_key": "k",
        "model": "m",
    }
    started = time.monotonic()
    r = _chat(client, model="byok", fx1={"byok": byok, "timeout_s": 0.5})
    elapsed = time.monotonic() - started
    err = _openai_err(r)
    results["timeout_enforced_refusal_502"] = (
        r.status_code == 502 and err is not None and err.get("code") == "backend_failure"
    )
    results["timeout_enforced_fast"] = elapsed < ep.sleep_s / 2

    started = time.monotonic()
    r = _chat(client, model="byok", fx1={"byok": byok}, headers={"X-Fx1-Timeout": "0.5"})
    elapsed = time.monotonic() - started
    results["timeout_header_enforced"] = r.status_code == 502 and elapsed < ep.sleep_s / 2

    # /harness dialect — body timeout_s reaches the same wire timeout
    started = time.monotonic()
    r = _complete(client, backend="byok", byok=byok, timeout_s=0.5)
    elapsed = time.monotonic() - started
    results["timeout_harness_enforced"] = (
        r.status_code == 502 and _flat_err(r) is not None and elapsed < ep.sleep_s / 2
    )

    # stream=true — the gate runs before SSE: a timed-out link answers a
    # JSON refusal and never emits a data frame
    r = _chat(client, model="byok", fx1={"byok": byok, "timeout_s": 0.5}, stream=True)
    results["timeout_stream_cut_is_json_502"] = (
        r.status_code == 502
        and "text/event-stream" not in r.headers.get("content-type", "")
        and not r.text.startswith("data:")
    )

    # a syntactically fine HTTP 200 whose body is not JSON is the same
    # fault class — malformed payload, enveloped 502, never a bare 500
    gep = _garbage_endpoint(b"not json at all")
    gbyok = {
        "base_url": f"http://127.0.0.1:{gep.port}/v1",  # NOSONAR(S5332) — local fixture
        "api_key": "k",
        "model": "m",
    }
    r = _chat(client, model="byok", fx1={"byok": gbyok})
    err = _openai_err(r)
    results["timeout_malformed_payload_502"] = (
        r.status_code == 502 and err is not None and err.get("code") == "backend_failure"
    )

    # batch submit — X-Fx1-Timeout on the submit applies to every line;
    # a line's own fx1.timeout_s overrides it per line
    spy = _SpyResolver()
    bclient, _ = _client(spy)
    lines = "\n".join(
        [
            json.dumps(
                {
                    "custom_id": "a",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "hosted_k3",
                        "messages": [{"role": "user", "content": "one"}],
                        "fx1": {"timeout_s": 9},
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "b",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "hosted_k3",
                        "messages": [{"role": "user", "content": "two"}],
                    },
                }
            ),
        ]
    )
    upload = bclient.post(
        "/v1/files",
        files={"file": ("lines.jsonl", lines.encode(), "application/jsonl")},
        data={"purpose": "batch"},
    )
    assert upload.status_code == 200, upload.text
    file_id = upload.json()["id"]
    r = bclient.post(
        "/v1/batches",
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        headers={"X-Fx1-Timeout": "4"},
    )
    assert r.status_code == 200, r.text
    batch_id = r.json()["id"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        st = bclient.get(f"/v1/batches/{batch_id}")
        if st.json().get("status") in {"completed", "failed", "cancelled", "expired"}:
            break
        time.sleep(0.05)
    got = [k.get("timeout_s") for _n, k in spy.calls]
    results["timeout_batch_per_line_header"] = 4.0 in got
    results["timeout_batch_line_body_overrides"] = 9.0 in got
    # a malformed submit-level header fails each line closed, not the batch
    spy.calls.clear()
    upload = bclient.post(
        "/v1/files",
        files={"file": ("l2.jsonl", lines.encode(), "application/jsonl")},
        data={"purpose": "batch"},
    )
    r = bclient.post(
        "/v1/batches",
        json={
            "input_file_id": upload.json()["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        headers={"X-Fx1-Timeout": "abc"},
    )
    assert r.status_code == 200, r.text
    bid = r.json()["id"]
    deadline = time.monotonic() + 30
    status = ""
    stj: dict[str, Any] = {}
    while time.monotonic() < deadline:
        stj = bclient.get(f"/v1/batches/{bid}").json()
        status = stj.get("status", "")
        if status in {"completed", "failed", "cancelled", "expired"}:
            break
        time.sleep(0.05)
    # a malformed submit header fails closed per line — the line whose
    # own fx1.timeout_s overrides the bad header still completes (body
    # wins, the same precedence rule as the request path); the
    # header-dependent line is the one refused
    results["timeout_batch_bad_header_fails_lines_closed"] = (
        status == "completed"
        and stj["request_counts"]["failed"] == 1
        and stj["request_counts"]["completed"] == 1
        and spy.calls == [("hosted_k3", {"timeout_s": 9.0})]
    )
    return results


# ---------------------------------------------------------------------------
# Checkpoint dir
# ---------------------------------------------------------------------------


def _checkpoint_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    spy = _SpyResolver()
    client, _api = _client(spy)
    ckpt = _temporary_directory() / "ckpt"

    # binds on local_fx1 — the resolver sees the header value
    r = _chat(
        client,
        headers={
            "X-Fx1-Backend": "local_fx1",
            "X-Fx1-Checkpoint-Dir": str(ckpt),
        },
    )
    results["ckpt_binds_on_local_fx1"] = r.status_code == 200 and spy.calls[-1][1].get(
        "checkpoint_dir"
    ) == str(ckpt)
    # refuses on every other link (post-#2812)
    for link in ("hosted_k3", "byok"):
        r = _chat(
            client,
            headers={
                "X-Fx1-Backend": link,
                "X-Fx1-Checkpoint-Dir": str(ckpt),
            },
            **(
                {"fx1": {"byok": {"base_url": _BYOK_BASE_URL, "api_key": "k", "model": "m"}}}
                if link == "byok"
                else {}
            ),
        )
        err = _openai_err(r)
        results[f"ckpt_refuses_on_{link}"] = (
            r.status_code == 422 and err is not None and "checkpoint" in str(err)
        )
    # model-named link is no loophole
    r = _chat(client, model="hosted_k3", headers={"X-Fx1-Checkpoint-Dir": str(ckpt)})
    results["ckpt_refuses_model_named_link"] = r.status_code == 422
    # empty refuses closed — min_length 1, no silent fallback to env
    r = _chat(
        client,
        headers={
            "X-Fx1-Backend": "local_fx1",
            "X-Fx1-Checkpoint-Dir": "",
        },
    )
    results["ckpt_empty_refuses"] = r.status_code == 422 and _openai_err(r) is not None
    # nonexistent dir fails closed at resolution (real get_backend →
    # FileNotFoundError → enveloped 422)
    rclient, _ = _client(real_backends=True)
    r = _chat(
        rclient,
        headers={
            "X-Fx1-Backend": "local_fx1",
            "X-Fx1-Checkpoint-Dir": str(ckpt / "missing"),
        },
    )
    results["ckpt_nonexistent_fails_closed"] = r.status_code == 422 and _openai_err(r) is not None
    # overlong refuses — max_length 4096 on the translated field
    r = _chat(
        client,
        headers={
            "X-Fx1-Backend": "local_fx1",
            "X-Fx1-Checkpoint-Dir": "/" + "x" * 5000,
        },
    )
    results["ckpt_overlong_refuses"] = r.status_code == 422
    # header overrides FX1_CHECKPOINT_DIR on local_fx1 — and only there
    spy.calls.clear()
    os.environ["FX1_CHECKPOINT_DIR"] = str(_temporary_directory() / "env-ckpt")
    try:
        r = _chat(
            client,
            headers={
                "X-Fx1-Backend": "local_fx1",
                "X-Fx1-Checkpoint-Dir": str(ckpt),
            },
        )
        results["ckpt_header_overrides_env"] = r.status_code == 200 and spy.calls[-1][1].get(
            "checkpoint_dir"
        ) == str(ckpt)
        spy.calls.clear()
        r = _chat(client, headers={"X-Fx1-Backend": "local_fx1"})
        results["ckpt_env_fills_when_header_absent"] = (
            r.status_code == 200
            and spy.calls[-1][1].get("checkpoint_dir") == os.environ["FX1_CHECKPOINT_DIR"]
        )
    finally:
        os.environ.pop("FX1_CHECKPOINT_DIR", None)
    # on /harness the header is ignored; the body field refuses
    # non-local_fx1 chains there too
    spy.calls.clear()
    r = _complete(client, headers={"X-Fx1-Checkpoint-Dir": str(ckpt)})
    results["ckpt_header_ignored_on_harness"] = (
        r.status_code == 200 and spy.calls[-1][1].get("checkpoint_dir") is None
    )
    r = _complete(client, checkpoint_dir=str(ckpt))
    results["ckpt_body_refuses_non_local_on_harness"] = (
        r.status_code == 422 and _flat_err(r) is not None
    )
    return results


# ---------------------------------------------------------------------------
# BYOK header channel
# ---------------------------------------------------------------------------


def _byok_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    spy = _SpyResolver()
    client, _api = _client(spy)

    def last() -> dict[str, Any]:
        assert spy.calls
        return spy.calls[-1][1]

    # the full header set resolves exactly like the body block
    r = _chat(
        client,
        model="gpt-x",
        headers={
            "X-Fx1-Byok-Base-Url": _BYOK_BASE_URL,
            "X-Fx1-Byok-Api-Key": "sk-hdr",
            "X-Fx1-Byok-Model": "upstream-m",
        },
    )
    kw = last()
    results["byok_header_full_set_resolves"] = (
        r.status_code == 200
        and spy.calls[-1][0] == "byok"
        and kw.get("base_url") == _BYOK_BASE_URL
        and kw.get("api_key") == "sk-hdr"
        and kw.get("model") == "upstream-m"
    )
    # model falls back to body.model when the header omits it (any
    # non-backend name is forwarded as the upstream model)
    spy.calls.clear()
    r = _chat(
        client,
        model="gpt-x",
        headers={
            "X-Fx1-Byok-Base-Url": _BYOK_BASE_URL,
            "X-Fx1-Byok-Api-Key": "sk-hdr",
        },
    )
    results["byok_model_falls_back_to_body_model"] = (
        r.status_code == 200 and last().get("model") == "gpt-x"
    )
    # base-url without api-key refuses 400 enveloped
    spy.calls.clear()
    r = _chat(client, model="gpt-x", headers={"X-Fx1-Byok-Base-Url": _BYOK_BASE_URL})
    err = _openai_err(r)
    results["byok_url_without_key_refuses_400"] = (
        r.status_code == 400 and err is not None and "Api-Key" in str(err)
    )
    # Partial credential headers refuse rather than silently selecting a
    # different provider when an intermediary drops the base-url header.
    spy.calls.clear()
    r = _chat(client, headers={"X-Fx1-Byok-Api-Key": "sk-lonely"})
    results["byok_key_without_url_refuses_400"] = (
        r.status_code == 400 and _openai_err(r) is not None and not spy.calls
    )
    spy.calls.clear()
    r = _chat(client, headers={"X-Fx1-Byok-Model": "m-lonely"})
    results["byok_model_without_url_refuses_400"] = (
        r.status_code == 400 and _openai_err(r) is not None and not spy.calls
    )
    # body wins on conflict — the resolver sees the body's credentials
    spy.calls.clear()
    r = _chat(
        client,
        model="byok",
        headers={
            "X-Fx1-Byok-Base-Url": "http://127.0.0.1:8/v1",  # NOSONAR(S5332) — never dialed
            "X-Fx1-Byok-Api-Key": "sk-hdr",
            "X-Fx1-Byok-Model": "hdr-model",
        },
        fx1={"byok": {"base_url": _BYOK_BASE_URL, "api_key": "sk-body", "model": "body-model"}},
    )
    kw = last()
    results["byok_body_overrides_header"] = (
        r.status_code == 200 and kw.get("api_key") == "sk-body" and kw.get("model") == "body-model"
    )
    # invalid base_url — post-fix the refusal is enveloped 422 (was a bare
    # 500 pre-fix: the ValidationError escaped the OpenAICompatError map)
    for name, url in (
        ("byok_bad_scheme_refuses_422", "ftp://evil"),
        ("byok_missing_netloc_refuses_422", "http://"),
    ):
        r = _chat(
            client,
            headers={
                "X-Fx1-Byok-Base-Url": url,
                "X-Fx1-Byok-Api-Key": "k",
                "X-Fx1-Byok-Model": "m",
            },
        )
        results[name] = r.status_code == 422 and _openai_err(r) is not None
    # over-length url/key — same enveloped refusal
    r = _chat(
        client,
        headers={
            "X-Fx1-Byok-Base-Url": "http://x/" + "a" * 3000,
            "X-Fx1-Byok-Api-Key": "k",
            "X-Fx1-Byok-Model": "m",
        },
    )
    results["byok_overlong_url_refuses_422"] = r.status_code == 422 and _openai_err(r) is not None
    # Pydantic's default ValidationError string embeds rejected input;
    # header re-enveloping must expose only the safe validation message.
    url_secret = "url-user-secret-marker"
    r = _chat(
        client,
        headers={
            "X-Fx1-Byok-Base-Url": f"ftp://u:{url_secret}@evil/v1",
            "X-Fx1-Byok-Api-Key": "k",
            "X-Fx1-Byok-Model": "m",
        },
    )
    results["byok_invalid_url_value_redacted"] = (
        r.status_code == 422
        and url_secret not in r.text
        and (_openai_err(r) or {}).get("code") == "invalid_byok_headers"
    )
    key_secret = "key-secret-marker"
    r = _chat(
        client,
        headers={
            "X-Fx1-Byok-Base-Url": _BYOK_BASE_URL,
            "X-Fx1-Byok-Api-Key": key_secret + "x" * 4096,
            "X-Fx1-Byok-Model": "m",
        },
    )
    results["byok_invalid_key_value_redacted"] = (
        r.status_code == 422
        and key_secret not in r.text
        and (_openai_err(r) or {}).get("code") == "invalid_byok_headers"
    )
    # anthropic dialect: same refusal in the anthropic grammar
    r = _messages(
        client,
        headers={
            "X-Fx1-Byok-Base-Url": "ftp://evil",
            "X-Fx1-Byok-Api-Key": "k",
            "X-Fx1-Byok-Model": "m",
        },
    )
    results["byok_bad_url_anthropic_envelope"] = (
        r.status_code == 422 and _anthropic_err(r) is not None
    )
    # the override on a non-byok link refuses 422
    r = _chat(
        client,
        headers={
            "X-Fx1-Backend": "hosted_k3",
            "X-Fx1-Byok-Base-Url": _BYOK_BASE_URL,
            "X-Fx1-Byok-Api-Key": "k",
            "X-Fx1-Byok-Model": "m",
        },
    )
    results["byok_on_non_byok_link_refuses_422"] = (
        r.status_code == 422 and _openai_err(r) is not None
    )
    # credentials never leak: a refused byok call's body/headers must not
    # contain the api key anywhere
    r = _chat(
        client,
        headers={
            "X-Fx1-Backend": "hosted_k3",
            "X-Fx1-Byok-Base-Url": _BYOK_BASE_URL,
            "X-Fx1-Byok-Api-Key": "sk-secret-probe",
            "X-Fx1-Byok-Model": "m",
        },
    )
    results["byok_key_never_emitted"] = "sk-secret-probe" not in r.text and all(
        "sk-secret-probe" not in v for v in r.headers.values()
    )
    # disabled server refuses the channel closed
    dclient, _ = _client(byok_override=False)
    r = _chat(
        dclient,
        model="byok",
        headers={
            "X-Fx1-Byok-Base-Url": _BYOK_BASE_URL,
            "X-Fx1-Byok-Api-Key": "k",
            "X-Fx1-Byok-Model": "m",
        },
    )
    results["byok_disabled_refuses_422"] = (
        r.status_code == 422 and (_openai_err(r) or {}).get("code") == "byok_override_disabled"
    )
    # embeddings surface shares the channel
    spy.calls.clear()
    r = client.post(
        "/v1/embeddings",
        json={"model": "up-embed", "input": "hi"},
        headers={
            "X-Fx1-Byok-Base-Url": _BYOK_BASE_URL,
            "X-Fx1-Byok-Api-Key": "k",
        },
    )
    results["byok_embeddings_channel"] = (
        r.status_code == 200 and spy.calls[-1][0] == "byok" and last().get("api_key") == "k"
    )
    return results


# ---------------------------------------------------------------------------
# Emitted receipt / completion headers
# ---------------------------------------------------------------------------


def _emitted_header_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client()

    for name, fn in (
        ("receipt_sha_on_complete", lambda: _complete(client)),
        ("receipt_sha_on_chat", lambda: _chat(client)),
        ("receipt_sha_on_messages", lambda: _messages(client)),
    ):
        r = fn()
        cid = r.headers.get("x-fx1-completion-id")
        rsha = r.headers.get("x-fx1-receipt-sha256")
        rec = client.get(f"/harness/completions/{cid}/receipt") if cid else None
        results[name] = (
            r.status_code == 200
            and cid is not None
            and rsha is not None
            and _SHA_RE.fullmatch(rsha) is not None
            and rec is not None
            and rec.status_code == 200
            and rec.json().get("receipt_sha256") == rsha
        )
    # ungated surfaces never emit completion headers
    for name, r in (
        ("no_completion_headers_on_health", client.get("/health")),
        ("no_completion_headers_on_models", client.get("/v1/models")),
        ("no_completion_headers_on_commands", client.get("/harness/commands")),
        ("no_completion_headers_on_metrics", client.get("/metrics")),
    ):
        results[name] = (
            r.status_code == 200
            and "x-fx1-completion-id" not in r.headers
            and "x-fx1-receipt-sha256" not in r.headers
        )
    # the API version stamp rides every response
    results["api_version_stamped_on_refusal"] = (
        "x-fx1-api-version" in _chat(client, headers={"X-Fx1-Timeout": "abc"}).headers
    )
    results["api_version_stamped_on_ok"] = "x-fx1-api-version" in client.get("/health").headers

    # request-side X-Fx1-Receipt-Hashes
    fake = "c" * 64
    r = _chat(client, headers={"X-Fx1-Receipt-Hashes": fake})
    err = _openai_err(r)
    results["receipt_hashes_unresolvable_422"] = (
        r.status_code == 422
        and err is not None
        and err.get("code") == "receipt_not_found"
        and fake in str(err)
    )
    r = _chat(client, headers={"X-Fx1-Receipt-Hashes": "zzz"})
    results["receipt_hashes_malformed_400"] = r.status_code == 400 and _openai_err(r) is not None
    r = _chat(client, headers={"X-Fx1-Receipt-Hashes": f"{fake},zzz"})
    results["receipt_hashes_mixed_400"] = r.status_code == 400
    # empty list is honest absence — no citations demanded
    results["receipt_hashes_empty_ignored"] = (
        _chat(client, headers={"X-Fx1-Receipt-Hashes": " , ,"}).status_code == 200
    )
    return results


def _resolvable_receipt_probe() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client()
    # plant a doc whose receipt_sha256 is a well-formed digest into the
    # receipts store this app is mounted on, so the citation resolves
    from fastapi import FastAPI

    root = cast(FastAPI, client.app).state.hdr_audit_receipts
    receipt = {
        "kind": "hdr_probe",
        "schema": "hdr_probe.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
    }
    receipt["receipt_sha256"] = hash_bytes(canonical_json_bytes(receipt))
    doc = Path(root) / "planted.json"
    doc.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    sha = str(receipt["receipt_sha256"])
    r = _chat(client, headers={"X-Fx1-Receipt-Hashes": sha})
    results["receipt_hashes_resolvable_200"] = r.status_code == 200
    # the fetched receipt endpoint serves the same sealed file
    fetched = client.get(f"/receipts/{sha}")
    results["receipt_fetch_matches_sealed"] = (
        fetched.status_code == 200 and fetched.json() == receipt
    )
    # conditional GET honors If-None-Match
    r = client.get(f"/receipts/{sha}", headers={"If-None-Match": f'"{sha}"'})
    results["receipt_fetch_etag_304"] = r.status_code == 304
    r = client.get(f"/receipts/{sha}", headers={"If-None-Match": '"d" * 0 + "e" * 64'})
    results["receipt_fetch_other_etag_200"] = r.status_code == 200
    r = client.get("/receipts/ZZZZ")
    results["receipt_fetch_bad_sha_422"] = r.status_code == 422
    return results


# ---------------------------------------------------------------------------
# X-Request-ID
# ---------------------------------------------------------------------------


def _request_id_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client()
    app = client.app

    # echoed verbatim on success and refusal
    r = client.get("/health", headers={"X-Request-ID": "rid-ok-1"})
    results["rid_echoed_on_success"] = r.headers.get("x-request-id") == "rid-ok-1"
    r = _chat(client, headers={"X-Request-ID": "rid-ok-2", "X-Fx1-Timeout": "abc"})
    results["rid_echoed_on_refusal"] = (
        r.status_code == 400 and r.headers.get("x-request-id") == "rid-ok-2"
    )
    # full charset + 64-char boundary echo
    rid64 = "A" * 60 + "._-" + "z"
    r = client.get("/health", headers={"X-Request-ID": rid64})
    results["rid_charset_boundary_echoes"] = r.headers.get("x-request-id") == rid64
    # minted when absent — 32 lowercase hex
    r = client.get("/health")
    rid = r.headers.get("x-request-id") or ""
    results["rid_minted_when_absent"] = _RID_RE.fullmatch(rid) is not None
    # malformed or over-long — minted, never echoed
    for name, value in (
        ("rid_overlong_mints", "x" * 65),
        ("rid_bad_charset_mints", "has space"),
    ):
        r = client.get("/health", headers={"X-Request-ID": value})
        results[name] = r.headers.get("x-request-id") != value and (
            _RID_RE.fullmatch(r.headers.get("x-request-id") or "") is not None
        )
    # CRLF can never mint a second header — raw ASGI sends what httpx won't
    resp = _raw(
        app,
        "GET",
        "/health",
        [(b"x-request-id", b"abc\r\nX-Injected: 1")],
    )
    hm = resp["header_map"]
    results["rid_crlf_never_injects"] = (
        resp["status"] == 200
        and "x-injected" not in hm
        and _RID_RE.fullmatch(hm.get("x-request-id", "")) is not None
    )
    # Duplicate request ids are ambiguous across HTTP stacks and refuse.
    resp = _raw(
        app,
        "GET",
        "/health",
        [(b"x-request-id", b"first-rid"), (b"x-request-id", b"second-rid")],
    )
    results["rid_duplicate_refuses_400"] = resp["status"] == 400
    # anthropic twin — request-id mirrors the value on /v1/messages
    r = _messages(client, headers={"X-Request-ID": "rid-anth-1"})
    results["rid_anthropic_twin"] = (
        r.headers.get("request-id") == "rid-anth-1"
        and r.headers.get("x-request-id") == "rid-anth-1"
    )
    # even refusals carry the twin
    r = _messages(client, headers={"X-Request-ID": "rid-anth-2", "X-Fx1-Timeout": "abc"})
    results["rid_anthropic_refusal_twin"] = r.headers.get("request-id") == "rid-anth-2"
    return results


# ---------------------------------------------------------------------------
# Idempotency-Key
# ---------------------------------------------------------------------------


def _idempotency_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client()
    app = client.app
    body = {"model": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]}

    # empty / whitespace-only — honestly ignored (not refused, not a key)
    for name, value in (
        ("idem_empty_ignored", ""),
        ("idem_whitespace_ignored", "   "),
    ):
        r1 = client.post(_CHAT_PATH, json=body, headers={"Idempotency-Key": value})
        r2 = client.post(_CHAT_PATH, json=body, headers={"Idempotency-Key": value})
        results[name] = (
            r1.status_code == 200
            and r2.status_code == 200
            and r2.headers.get("x-fx1-idempotent-replay") != "true"
            and r1.json()["id"] != r2.json()["id"]
        )
    # padded key strips to the inner value — a retry with the stripped
    # spelling replays the padded call
    r1 = client.post(_CHAT_PATH, json=body, headers={"Idempotency-Key": "  pad-key  "})
    r2 = client.post(_CHAT_PATH, json=body, headers={"Idempotency-Key": "pad-key"})
    results["idem_padding_normalizes"] = (
        r2.headers.get("x-fx1-idempotent-replay") == "true" and r2.json() == r1.json()
    )
    # 256 ok, 257 refuses closed
    r = client.post(_CHAT_PATH, json=body, headers={"Idempotency-Key": "k" * 256})
    results["idem_256_admitted"] = r.status_code == 200
    r = client.post(_CHAT_PATH, json=body, headers={"Idempotency-Key": "k" * 257})
    results["idem_257_refuses_400"] = r.status_code == 400 and _openai_err(r) is not None
    # binary/CRLF key — an opaque dict key that dedups and is never echoed
    resp1 = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [(b"content-type", b"application/json"), (b"idempotency-key", "k\xffy".encode("latin-1"))],
        json.dumps(body).encode(),
    )
    resp2 = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [(b"content-type", b"application/json"), (b"idempotency-key", "k\xffy".encode("latin-1"))],
        json.dumps(body).encode(),
    )
    results["idem_binary_key_dedups"] = (
        resp1["status"] == 200
        and resp2["status"] == 200
        and resp2["header_map"].get("x-fx1-idempotent-replay") == "true"
        and resp1["body"] == resp2["body"]
    )
    resp3 = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [
            (b"content-type", b"application/json"),
            (b"idempotency-key", b"crlf\r\nX-Inject: 1"),
        ],
        json.dumps(body).encode(),
    )
    results["idem_crlf_key_refuses_400"] = (
        resp3["status"] == 400 and "x-inject" not in resp3["header_map"]
    )
    # key reuse with a different body fails closed 409
    r1 = client.post(_CHAT_PATH, json=body, headers={"Idempotency-Key": "conflict-1"})
    r2 = client.post(
        _CHAT_PATH,
        json={**body, "messages": [{"role": "user", "content": "different"}]},
        headers={"Idempotency-Key": "conflict-1"},
    )
    results["idem_key_body_conflict_409"] = r2.status_code == 409 and _openai_err(r2) is not None
    # replay is byte-identical and flagged, on JSON and on the harness
    # dialect's replayed field
    r1 = _complete(client, headers={"Idempotency-Key": "harn-1"})
    r2 = _complete(client, headers={"Idempotency-Key": "harn-1"})
    results["idem_harness_replay_flagged"] = (
        r2.status_code == 200 and r2.json().get("replayed") is True
    )
    # /v1/batches dedups on the submit key — one batch minted
    up = client.post(
        "/v1/files",
        files={
            "file": (
                "l.jsonl",
                b'{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"hosted_k3","messages":[{"role":"user","content":"x"}]}}',
                "application/jsonl",
            )
        },
        data={"purpose": "batch"},
    )
    breq = {
        "input_file_id": up.json()["id"],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
    }
    r1 = client.post("/v1/batches", json=breq, headers={"Idempotency-Key": "batch-1"})
    r2 = client.post("/v1/batches", json=breq, headers={"Idempotency-Key": "batch-1"})
    results["idem_batch_submit_dedups"] = (
        r1.status_code == 200
        and r2.json().get("id") == r1.json().get("id")
        and r2.headers.get("x-fx1-idempotent-replay") == "true"
    )
    # Last-Event-ID resume contract — header-parse edges
    r = _chat(client, headers={"Last-Event-ID": "3"})
    results["lei_non_stream_400"] = r.status_code == 400 and _openai_err(r) is not None
    r = _chat(client, stream=True, headers={"Last-Event-ID": "abc", "Idempotency-Key": "k"})
    results["lei_non_integer_400"] = r.status_code == 400
    r = _chat(client, stream=True, headers={"Last-Event-ID": "-1", "Idempotency-Key": "k"})
    results["lei_negative_400"] = r.status_code == 400
    r = _chat(client, stream=True, headers={"Last-Event-ID": "1"})
    results["lei_needs_idem_key"] = r.status_code == 400
    r = _chat(client, stream=True, headers={"Last-Event-ID": "1", "Idempotency-Key": "never"})
    results["lei_unknown_key_409"] = r.status_code == 409
    return results


# ---------------------------------------------------------------------------
# Casing + duplicates
# ---------------------------------------------------------------------------


def _casing_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client()
    app = client.app
    body = json.dumps(
        {"model": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()

    # header names are case-insensitive — every spelling binds
    for name, hdr in (
        ("case_lower_binds", b"x-fx1-timeout"),
        ("case_upper_binds", b"X-FX1-TIMEOUT"),
        ("case_mixed_binds", b"X-Fx1-Timeout"),
    ):
        resp = _raw(
            app,
            "POST",
            _CHAT_PATH,
            [(b"content-type", b"application/json"), (hdr, b"abc")],
            body,
        )
        results[name] = resp["status"] == 400
    # Duplicates refuse before the translator can collapse them last-wins.
    resp = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [
            (b"content-type", b"application/json"),
            (b"x-fx1-timeout", b"junk"),
            (b"x-fx1-timeout", b"5"),
        ],
        body,
    )
    results["dup_xfx1_refuses_400"] = resp["status"] == 400
    resp = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [
            (b"content-type", b"application/json"),
            (b"x-fx1-timeout", b"5"),
            (b"x-fx1-timeout", b"junk"),
        ],
        body,
    )
    results["dup_xfx1_reverse_refuses_400"] = resp["status"] == 400
    resp = _raw(
        app,
        "GET",
        "/harness/commands",
        [(b"x-api-key", _ROOT.encode()), (b"x-api-key", b"wrong")],
    )
    results["dup_api_key_refuses_400"] = resp["status"] == 400
    resp = _raw(
        app,
        "GET",
        "/harness/commands",
        [(b"x-api-key", _ROOT.encode()), (b"authorization", f"Bearer {_ROOT}".encode())],
    )
    results["mixed_auth_headers_refuse_400"] = resp["status"] == 400
    return results


# ---------------------------------------------------------------------------
# Anthropic headers
# ---------------------------------------------------------------------------


def _anthropic_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(_SpyResolver(), api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}

    # anthropic-version — presence flips the dialect; the value is not
    # validated (pinned, on /v1/models where the shapes differ)
    r = client.get("/v1/models", headers=root_h)
    results["anthropic_openai_shape_without_header"] = r.json().get("object") == "list"
    for name, value in (
        ("anthropic_version_any_value", "2023-06-01"),
        ("anthropic_version_garbage", "garbage"),
        ("anthropic_version_empty", ""),
    ):
        r = client.get("/v1/models", headers={**root_h, "anthropic-version": value})
        results[name] = r.status_code == 200 and isinstance(r.json().get("data"), list)
    # /v1/messages is always the anthropic dialect — header or not
    r = client.post(
        _MESSAGES_PATH,
        json={
            "model": "hosted_k3",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={**root_h, "X-Fx1-Timeout": "abc"},
    )
    results["anthropic_path_forces_dialect"] = (
        r.status_code == 400 and _anthropic_err(r) is not None
    )
    # anthropic-beta tolerated, ignored
    r = client.post(
        _MESSAGES_PATH,
        json={
            "model": "hosted_k3",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={**root_h, "anthropic-beta": "bogus-feature-2099"},
    )
    results["anthropic_beta_tolerated"] = r.status_code == 200
    # x-api-key (SDK spelling, lowercase-insensitive) authenticates
    r = client.post(
        _MESSAGES_PATH,
        json={
            "model": "hosted_k3",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"x-api-key": _ROOT},
    )
    results["x_api_key_authenticates_anthropic"] = r.status_code == 200
    # Authorization: Bearer works on the /v1 surface (incl. /v1/messages)
    r = client.post(
        _MESSAGES_PATH,
        json={
            "model": "hosted_k3",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {_ROOT}"},
    )
    results["bearer_authenticates_v1"] = r.status_code == 200
    # Bearer on /harness is NOT accepted — X-API-Key only there
    r = client.get("/harness/commands", headers={"Authorization": f"Bearer {_ROOT}"})
    results["bearer_refused_on_harness"] = r.status_code == 401
    r = client.get("/harness/commands", headers=root_h)
    results["x_api_key_authenticates_harness"] = r.status_code == 200
    # empty X-API-Key is absent, not a credential
    r = client.get("/harness/commands", headers={"X-API-Key": ""})
    results["empty_x_api_key_refuses"] = r.status_code == 401
    # x-should-retry lands on retryable statuses, absent on a 200
    r = client.post(
        _MESSAGES_PATH,
        json={
            "model": "hosted_k3",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=root_h,
    )
    results["x_should_retry_absent_on_ok"] = r.headers.get("x-should-retry") is None
    return results


# ---------------------------------------------------------------------------
# Rate-limit response headers
# ---------------------------------------------------------------------------


def _rate_limit_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(_SpyResolver(), api_key=_ROOT, rate_limit_rps=100.0)
    root_h = {"X-API-Key": _ROOT}

    def rl(resp: Any) -> dict[str, str]:
        return {k: v for k, v in resp.headers.items() if k.lower().startswith("x-ratelimit-")}

    # admitted call — the bucket reports its post-consumption state
    r = client.get("/harness/commands", headers=root_h)
    got = rl(r)
    results["rl_headers_on_admitted"] = (
        r.status_code == 200
        and got.get("x-ratelimit-limit") == "100"
        and got.get("x-ratelimit-remaining") == "99"
        and "x-ratelimit-reset" in got
    )
    # the counter counts down honestly — pin it on a bucket that cannot
    # refill within the window (1 rps): the first call empties it, the
    # second is refused while still reporting the armed headers
    c1rps, _ = _client(_SpyResolver(), api_key=_ROOT, rate_limit_rps=1.0)
    h1 = {"X-API-Key": _ROOT}
    r1 = c1rps.get("/harness/commands", headers=h1)
    r2 = c1rps.get("/harness/commands", headers=h1)
    results["rl_remaining_decrements"] = (
        r1.status_code == 200
        and rl(r1).get("x-ratelimit-remaining") == "0"
        and r2.status_code == 429
        and "x-ratelimit-limit" in rl(r2)
    )
    # an auth refusal consumed a slot and reports it (401 path applies
    # the headers already)
    r = client.get("/harness/commands", headers={"X-API-Key": "nope"})
    results["rl_headers_on_401"] = "x-ratelimit-limit" in rl(r)
    # managed-key refusals report the global bucket too — the defect-fix
    # pin: the KeyStoreError branch must merge the armed headers
    _key, key_id = _mint(client, root_h, max_requests=1)
    h = {"X-API-Key": _key}
    client.get("/harness/commands", headers=h)
    r = client.get("/harness/commands", headers=h)
    results["rl_headers_on_quota_refusal"] = (
        r.status_code == 429
        and "x-ratelimit-limit" in rl(r)
        and "x-ratelimit-remaining" in rl(r)
        and "retry-after" not in r.headers  # a hard budget never retries
    )
    scope_key, _ = _mint(client, root_h, scopes=["read"])
    r = client.post("/harness/keys", json={}, headers={"X-API-Key": scope_key})
    results["rl_headers_on_scope_refusal"] = r.status_code == 403 and "x-ratelimit-limit" in rl(r)
    rpm_key, _ = _mint(client, root_h, rpm=1)
    hk = {"X-API-Key": rpm_key}
    client.get("/harness/commands", headers=hk)
    r = client.get("/harness/commands", headers=hk)
    got = rl(r)
    results["rl_headers_on_key_window_429"] = (
        r.status_code == 429
        and r.headers.get("retry-after") is not None
        and "x-ratelimit-limit-requests" in got
        and "x-ratelimit-limit" in got  # global bucket merged post-fix
    )
    # global window refusal — limiter saturated
    tight, _api2 = _client(_SpyResolver(), api_key=_ROOT, rate_limit_rps=1.0)
    tight.get("/harness/commands", headers={"X-API-Key": _ROOT})
    r = tight.get("/harness/commands", headers={"X-API-Key": _ROOT})
    got = rl(r)
    results["rl_headers_on_global_429"] = (
        r.status_code == 429
        and r.headers.get("retry-after") is not None
        and {"x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"} <= set(got)
    )
    results["rl_enveloped_openai_shape"] = (
        _openai_err(
            tight.post(
                _CHAT_PATH,
                json={
                    "model": "hosted_k3",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"X-API-Key": _ROOT},
            )
        )
        is not None
        or tight.get("/health").status_code == 200  # health is exempt
    )
    # limiter off — no headers at all
    quiet, _api3 = _client(_SpyResolver(), api_key=_ROOT)
    r = quiet.get("/harness/commands", headers={"X-API-Key": _ROOT})
    results["rl_absent_when_disarmed"] = not rl(r)
    # public paths never consume or report the bucket
    r = client.get("/health")
    results["rl_absent_on_public_path"] = not rl(r)
    return results


# ---------------------------------------------------------------------------
# Envelope + CORS
# ---------------------------------------------------------------------------


def _envelope_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client()
    app = client.app
    body = json.dumps(
        {"model": "hosted_k3", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()

    # every header-parse refusal lands in the dialect's error shape
    results["envelope_openai_400"] = (
        _openai_err(_chat(client, headers={"X-Fx1-Timeout": "abc"})) is not None
    )
    results["envelope_anthropic_400"] = (
        _anthropic_err(_messages(client, headers={"X-Fx1-Timeout": "abc"})) is not None
    )
    r = _complete(client, timeout_s=-1)  # body validation on /harness
    results["envelope_harness_422_flat"] = r.status_code == 422
    # malformed Content-Length — post-fix the /v1 path answers the openai
    # grammar, /v1/messages the anthropic one, /harness the flat shape
    resp = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [(b"content-type", b"application/json"), (b"content-length", b"bogus")],
        body,
    )
    try:
        parsed = json.loads(resp.get("body", b""))
    except ValueError:
        parsed = {}
    results["envelope_bad_content_length_v1"] = resp["status"] == 400 and isinstance(
        parsed.get("error"), dict
    )
    resp = _raw(
        app,
        "POST",
        _MESSAGES_PATH,
        [(b"content-type", b"application/json"), (b"content-length", b"bogus")],
        body,
    )
    try:
        parsed = json.loads(resp.get("body", b""))
    except ValueError:
        parsed = {}
    results["envelope_bad_content_length_anthropic"] = (
        resp["status"] == 400
        and parsed.get("type") == "error"
        and isinstance(parsed.get("error"), dict)
    )
    resp = _raw(
        app,
        "POST",
        _COMPLETE_PATH,
        [(b"content-type", b"application/json"), (b"content-length", b"bogus")],
        body,
    )
    try:
        parsed = json.loads(resp.get("body", b""))
    except ValueError:
        parsed = {}
    results["envelope_bad_content_length_harness"] = (
        resp["status"] == 400 and "detail" in parsed and "code" in parsed
    )
    # Security-sensitive duplicates and conflicting framing refuse before
    # framework layers can disagree about which value controls the body.
    secured, _ = _client(api_key=_ROOT)
    secured_app = secured.app
    auth = [(b"x-api-key", _ROOT.encode())]
    small = json.dumps(
        {"model": "hosted_k3", "messages": [{"role": "user", "content": "x"}]}
    ).encode()
    for name, extra in (
        ("negative_content_length_refuses", [(b"content-length", b"-1")]),
        (
            "duplicate_content_length_refuses",
            [(b"content-length", str(len(small)).encode()), (b"content-length", b"1")],
        ),
        (
            "content_length_transfer_encoding_refuses",
            [(b"content-length", str(len(small)).encode()), (b"transfer-encoding", b"chunked")],
        ),
    ):
        framed = _raw(
            secured_app,
            "POST",
            _CHAT_PATH,
            [(b"content-type", b"application/json"), *auth, *extra],
            small,
        )
        results[name] = framed["status"] == 400
    oversized = json.dumps(
        {
            "model": "hosted_k3",
            "messages": [{"role": "user", "content": "x" * ((1 << 20) + 1)}],
        }
    ).encode()
    for name, extra in (
        ("missing_length_oversize_refuses_413", []),
        ("underdeclared_oversize_refuses_413", [(b"content-length", b"1")]),
    ):
        large = _raw(
            secured_app,
            "POST",
            _CHAT_PATH,
            [(b"content-type", b"application/json"), *auth, *extra],
            oversized,
        )
        results[name] = large["status"] == 413
    # over-cap body — 413 in the path's grammar
    resp = _raw(
        app,
        "POST",
        _CHAT_PATH,
        [
            (b"content-type", b"application/json"),
            (b"content-length", str((1 << 20) + 10).encode()),
        ],
        body,
    )
    try:
        parsed = json.loads(resp.get("body", b""))
    except ValueError:
        parsed = {}
    results["envelope_too_large_v1"] = resp["status"] == 413 and isinstance(
        parsed.get("error"), dict
    )
    # unknown /v1 path — the catch-all answers the provider grammar
    r = client.post("/v1/definitely-not-a-route", json={})
    results["envelope_v1_catchall"] = r.status_code == 404 and _openai_err(r) is not None
    r = client.post("/v1/messages/unknown", json={})
    results["envelope_anthropic_catchall"] = r.status_code == 404 and _anthropic_err(r) is not None
    return results


def _cors_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client(cors_origins="https://app.example.com")
    origin = {"Origin": "https://app.example.com"}

    r = client.options(
        _CHAT_PATH,
        headers={
            **origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-fx1-timeout,idempotency-key",
        },
    )
    allow_headers = r.headers.get("access-control-allow-headers", "").lower()
    results["cors_preflight_admitted"] = (
        r.status_code == 200
        and r.headers.get("access-control-allow-origin") == "https://app.example.com"
        and "x-fx1-timeout" in allow_headers
        and "idempotency-key" in allow_headers
    )
    r = client.options(
        _CHAT_PATH,
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    results["cors_foreign_origin_refused"] = (
        r.headers.get("access-control-allow-origin") != "https://evil.example"
    )
    # preflight never authenticates — no credentials on an OPTIONS
    r = client.options(
        _CHAT_PATH,
        headers={**origin, "Access-Control-Request-Method": "POST"},
    )
    results["cors_preflight_skips_auth"] = r.status_code == 200
    # no CORS configured — OPTIONS on an API path is enveloped 405/404
    noclient, _ = _client()
    r = noclient.options(_CHAT_PATH, headers=origin)
    results["cors_off_options_enveloped"] = r.status_code in (404, 405)
    return results


def _misc_probes() -> dict[str, bool]:
    results: dict[str, bool] = {}
    client, _api = _client()

    # Accept negotiation on /metrics — prometheus text on demand
    r = client.get("/metrics", headers={"Accept": "text/plain"})
    results["metrics_accept_prom_text"] = r.status_code == 200 and "text/plain" in r.headers.get(
        "content-type", ""
    )
    r = client.get("/metrics?format=prom")
    results["metrics_format_param_prom"] = "text/plain" in r.headers.get("content-type", "")
    # a sane default stays JSON
    r = client.get("/metrics")
    results["metrics_default_json"] = "application/json" in r.headers.get("content-type", "")

    # security headers stamp every response
    r = client.get("/health")
    results["security_headers_stamped"] = (
        r.headers.get("x-content-type-options") == "nosniff"
        and r.headers.get("referrer-policy") == "no-referrer"
    )
    return results


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def header_audit() -> dict[str, Any]:
    """Run the header-surface battery; returns literal bools."""
    out: dict[str, Any] = {}
    with _audit_context():
        # Timeout/garbage probes use owned loopback BYOK fixtures.
        os.environ["FX1_BYOK_ALLOW_PRIVATE_NETWORKS"] = "1"
        out.update(_timeout_parse_probes())
        out.update(_precedence_probes())
        out.update(_timeout_enforcement_probes())
        out.update(_checkpoint_probes())
        out.update(_byok_probes())
        out.update(_emitted_header_probes())
        out.update(_resolvable_receipt_probe())
        out.update(_request_id_probes())
        out.update(_idempotency_probes())
        out.update(_casing_probes())
        out.update(_anthropic_probes())
        out.update(_rate_limit_probes())
        out.update(_envelope_probes())
        out.update(_cors_probes())
        out.update(_misc_probes())
    return out


def header_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under header_audit.v1."""
    r = header_audit()
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "header_audit",
        "schema": "header_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "The header contract is fail-closed and honest end to end. "
            "X-Fx1-Timeout parses any float() spelling in (0, 3600] and "
            "refuses everything else 400 enveloped; the value reaches the "
            "resolver as timeout_s (body fx1.timeout_s wins on conflict), "
            "is enforced as the wire deadline on real links (a sleeping "
            "endpoint refuses 502 backend_failure inside the deadline, "
            "JSON even on stream=true), applies per-line on batch submits "
            "with line fx1.timeout_s overriding, and is ignored on the "
            "/harness dialect. X-Fx1-Checkpoint-Dir binds only on "
            "local_fx1 (422 elsewhere, empty/nonexistent fail closed, "
            "overrides FX1_CHECKPOINT_DIR on that link only). "
            "X-Fx1-Byok-* resolves to the same kwargs as fx1.byok, body "
            "wins on conflict, a url-less key/model refuses, a keyless "
            "url refuses 400, an invalid url refuses enveloped 422, the "
            "override on a non-byok link refuses 422, creds never appear "
            "in refusals, and byok_override=False refuses "
            "byok_override_disabled. Emitted headers tell the truth: "
            "X-Fx1-Receipt-Sha256 is the sha of the cited completion "
            "record's sealed receipt, completion headers never appear on "
            "ungated surfaces, X-Request-ID echoes verbatim within its "
            "charset (duplicates refuse; CRLF mints a fresh id — no "
            "injection), and duplicate X-Fx1/auth/framing headers refuse. "
            "Idempotency-Key: empty/whitespace honestly ignored, padding "
            "normalized, >256 chars and controls refused 400, obs-text "
            "keys dedup opaquely, conflict is 409, replays flagged and "
            "byte-identical including batch submits. anthropic-version "
            "flips the dialect by presence (value unvalidated, pinned), "
            "anthropic-beta tolerated, x-api-key and Bearer both "
            "authenticate /v1/messages while /harness takes X-API-Key "
            "only. Armed X-RateLimit-* headers report the consumed bucket "
            "on admitted AND refused calls (managed-key refusals merge "
            "the global headers post-fix; hard quotas omit Retry-After). "
            "Every header-parse refusal lands in the dialect's envelope — "
            "{error} on /v1, {type:error} on /v1/messages, {detail,code} "
            "on /harness — including malformed/ambiguous framing; the "
            "actual ASGI body is capped without trusting Content-Length. "
            "CORS preflights declare the X-Fx1-* allow-list to configured "
            "origins only."
            if ok
            else f"HEADER AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(header_audit_bench(), indent=2, sort_keys=True))
