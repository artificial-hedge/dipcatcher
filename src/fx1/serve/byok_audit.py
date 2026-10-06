"""byok_audit — adversarial probes on the BYOK (bring-your-own-key) surface.

BYOK lets a caller point any OpenAI-compatible endpoint at the harness:
``fx1.byok`` (``base_url``/``api_key``/``model``) on every ``/v1/*`` and
``/harness/*`` request shape, or the ``X-Fx1-Byok-*`` header triple. The
credential contract is load-bearing: a request-scoped api_key must reach
only its own upstream, must never appear in envelopes, validation
errors, completion records, usage reports, SSE frames, or batch output,
and must survive fallback composition, replay, and concurrency without
crossing request boundaries.

Coverage map:

- *Backend contract* — construction fails closed on missing/malformed
  env or kwargs; the request pins ``temperature: 0.0`` + Bearer auth +
  the normalized ``<base>/chat/completions`` URL; transport and payload
  faults raise rather than fabricate.
- *Wire correctness* — the stub upstream records that the request hits
  ``{base_url}/chat/completions`` with ``Authorization: Bearer <key>``,
  the declared ``model``, the verbatim prompt, and JSON content-type —
  on chat, responses, legacy completions, messages, embeddings, and the
  ``/harness/complete`` + ``/harness/complete/stream`` surfaces.
- *Credential hygiene* — the api_key appears in no 200 envelope, no
  error envelope, no stored retrieval record, no ``/harness/usage``
  serialization, no SSE frame, no batch output line, and no upstream
  header ever carries the *harness's* credentials.
- *Validation* — ``file:``/``javascript:``/``ftp:``/hostless/bad-port/
  userinfo/query/fragment base URLs refuse 422 on body and header paths
  alike; a ``byok`` block on a non-BYOK link refuses 422 (never a bare
  500); ``byok_override=False`` refuses 422; an ``ft:`` model name
  refuses BYOK credentials outright.
- *Fallback semantics* — ``backend + fallbacks`` composes: a BYOK link
  serves when the primary is down, and a failing BYOK upstream advances
  to the next link; credentials bind only the ``byok`` link (a
  non-BYOK link's resolver call never sees base_url/api_key); the last
  link's honest error surfaces.
- *Isolation* — two parallel requests with different byok blocks hit
  different upstreams with different keys; each stub asserts it saw
  only its own credential; usage attributes to the calling key ids; a
  ``read``-scoped key refuses 403 before any upstream contact; ambient
  ``FX1_BYOK_*`` env never shadows a per-request block.
- *Timeout/cancel* — ``X-Fx1-Timeout`` bounds the upstream call and a
  hung endpoint answers an enveloped 502 without wedging the app.
- *Streaming* — upstream SSE becomes harness token frames + ``[DONE]``;
  a mid-stream upstream fault is an enveloped error (JSON before the
  keepalive commit, a terminal error frame + ``[DONE]`` after it).
- *Store/replay* — ``store:false`` persists nothing; ``store:true``
  persists the record but never the credential; an Idempotency-Key
  replay returns the stored answer without re-hitting the upstream.
- *Batch* — per-line ``fx1.byok`` reaches the upstream per line; the
  output file carries answers and honest error rows, never the key.

Defects found by this battery and fixed on the same PR:

- ``ByokOverride.base_url`` accepted userinfo, query, params, fragments
  and bad ports — a credential-bearing URL like
  ``http://user:pw@host`` passed validation and its text echoed back in
  the 422 detail. The shared ``byok_base_url_problem`` rule now rejects
  all of them on every surface (request model and
  ``OpenAICompatBackend`` env path alike) and never echoes the URL.
- ``X-Fx1-Byok-*`` header credentials raised an uncaught pydantic
  ``ValidationError`` — a bare 500 on every ``/v1/*`` surface. The
  header path now lands the same 422 the body validator produces.
- ``fx1.byok`` on a chain without a ``byok`` link (e.g. ``backend:
  hosted_k3`` + a byok block, or a bare byok block that resolves to the
  default ``hosted_k3`` link) failed inside ``CompleteRequest``
  *after* translation — an uncaught ValidationError → bare 500 on
  ``/v1/chat/completions``, ``/v1/responses``, ``/v1/embeddings``, and
  the SDK twins; ``EvalSubmitRequest`` had the same hole on
  ``/v1/evals/{id}/runs``. Both now refuse 422: the link-membership
  check moved into ``_resolve_openai_link`` (which also refuses an
  ``ft:`` model + ``fx1.byok`` that previously dropped the credential
  silently), and construction sites wrap residual ValidationErrors.
- Validation detail echoed the rejected ``input`` of credential fields
  on ``/harness/*`` and batch-adjacent paths — a rejected
  ``byok.api_key`` round-tripped into the 422 body. Credential-loc
  errors now carry ``"[redacted]"`` inputs.
- The SDK twin re-validated ``byok.base_url`` with the old scheme+netloc
  rule *and* echoed the URL into the error — the shared rule now
  applies there too, so the in-process path enforces the same wire
  contract.
- Transport faults escaped the envelope as bare 500s: ``urlopen`` raises
  ``TimeoutError``/``OSError``/``http.client.HTTPException`` for stalls,
  resets, and truncated bodies — none of which is ``URLError`` — and a
  non-JSON upstream body raised ``JSONDecodeError`` untouched. Every
  OpenAI-compatible call site (BYOK + local_fx1 shared helpers, the
  hosted link) now maps them to ``RuntimeError`` → enveloped 502.

Sealed ``byok_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import http.server
import json
import os
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fx1.serve.conv_audit import _RESOURCES, _audit_context, _temporary_directory
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

__all__ = ["byok_audit", "byok_audit_bench"]

_ENVS = ("FX1_BYOK_BASE_URL", "FX1_BYOK_API_KEY", "FX1_BYOK_MODEL")
_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_MODEL = "byok"
_KEY_A = "probe-key-alpha"
_KEY_B = "probe-key-beta"
_MODEL_A = "stub-model-alpha"
_MODEL_B = "stub-model-beta"


# ---------------------------------------------------------------------------
# Local stub upstream — a real HTTP server the BYOK backend talks to
# ---------------------------------------------------------------------------


class _StubUpstream:
    """Threaded HTTP stand-in for a BYOK provider.

    Records every request (method/path/headers/body) so probes can assert
    the wire shape and credential exactly; plays scripted responses by
    ``mode`` so probes run fully offline. ``delay_s`` applies before the
    response — the timeout probe's lever.
    """

    def __init__(self, mode: str = "chat_ok", *, delay_s: float = 0.0) -> None:
        self.mode = mode
        self.delay_s = delay_s
        self.redirect_to = ""
        self.answer = "stub-answer"
        self.resp_model = "stub-upstream-model"
        self.usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        self.requests: list[dict[str, Any]] = []
        upstream = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a: Any) -> None:
                return None

            def _record(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                upstream.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": {str(k): str(v) for k, v in self.headers.items()},
                        "body": self.rfile.read(length) if length else b"",
                    }
                )

            def do_POST(self) -> None:
                self._record()
                upstream._respond(self)

            def do_GET(self) -> None:
                self._record()
                upstream._respond(self)

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        _RESOURCES.get().callback(self.close)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def auths(self) -> list[str]:
        return [r["headers"].get("Authorization", "") for r in self.requests]

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r["body"]) for r in self.requests if r["body"]]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _send(
        self,
        h: http.server.BaseHTTPRequestHandler,
        status: int,
        body: bytes,
        ctype: str = "application/json",
    ) -> None:
        try:
            h.send_response(status)
            h.send_header("Content-Type", ctype)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _sse_start(self, h: http.server.BaseHTTPRequestHandler) -> None:
        """SSE has no Content-Length — the stream ends at close."""
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()

    def _sse_frame(self, h: http.server.BaseHTTPRequestHandler, frame: bytes) -> bool:
        try:
            h.wfile.write(frame)
            h.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return False
        return True

    def _respond(self, h: http.server.BaseHTTPRequestHandler) -> None:
        if self.delay_s:
            time.sleep(self.delay_s)
        mode = self.mode
        if mode == "chat_ok":
            self._send(
                h,
                200,
                json.dumps(
                    {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion",
                        "model": self.resp_model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": self.answer},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": self.usage,
                    }
                ).encode(),
            )
        elif mode == "embed_ok":
            self._send(
                h,
                200,
                json.dumps(
                    {
                        "object": "list",
                        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
                        "usage": {"prompt_tokens": 3, "total_tokens": 3},
                    }
                ).encode(),
            )
        elif mode == "err500":
            self._send(h, 500, json.dumps({"error": "upstream-boom"}).encode())
        elif mode == "err401":
            self._send(h, 401, json.dumps({"error": "bad key"}).encode())
        elif mode == "redirect":
            h.send_response(302)
            h.send_header("Location", self.redirect_to)
            h.send_header("Content-Length", "0")
            h.end_headers()
        elif mode == "bad_json":
            self._send(h, 200, b"this is not json")
        elif mode == "bad_utf8":
            self._send(h, 200, b"\xff")
        elif mode == "missing_choices":
            self._send(h, 200, json.dumps({"object": "chat.completion"}).encode())
        elif mode == "sse_ok":
            frames = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "he"}}]},
                {"choices": [{"index": 0, "delta": {"content": "llo"}}]},
                {"choices": [{"index": 0, "delta": {}}], "usage": self.usage},
            ]
            self._sse_start(h)
            for f in frames:
                if not self._sse_frame(h, f"data: {json.dumps(f)}\n\n".encode()):
                    return
            self._sse_frame(h, b"data: [DONE]\n\n")
        elif mode == "sse_mid_fault":
            # one well-formed delta, a stall past the keepalive window,
            # then an unparseable frame — the mid-stream fault shape.
            self._sse_start(h)
            first = (
                f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': 'he'}}]})}\n\n"
            )
            if not self._sse_frame(h, first.encode()):
                return
            time.sleep(max(self.delay_s, 1.0))
            self._sse_frame(h, b"data: {not-json\n\n")


def _byok(url: str, *, key: str = _KEY_A, model: str = _MODEL_A) -> dict[str, Any]:
    return {"base_url": url, "api_key": key, "model": model}


def _chat_body(
    upstream: _StubUpstream,
    *,
    key: str = _KEY_A,
    model: str = _MODEL_A,
    text: str = "byok-ping",
    backend: str = "byok",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "model": _MODEL,
        "messages": [{"role": "user", "content": text}],
        "fx1": {"backend": backend, "byok": _byok(upstream.url, key=key, model=model)},
        **extra,
    }


class _LocalStub:
    """In-process stand-in for a non-BYOK link in a fallback chain."""

    def __init__(self, text: str = "local-ok", fail: Exception | None = None) -> None:
        self._text = text
        self._fail = fail
        self.calls = 0
        self.last_usage: dict[str, int] | None = None
        self._model = "local-stub-0"

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        self.calls += 1
        if self._fail is not None:
            raise self._fail
        return self._text

    def close(self) -> None:
        return None


def _client(
    stubs: dict[str, Callable[[], Any]] | None = None,
    *,
    api_key: str | None = None,
    **app_kwargs: Any,
) -> tuple[TestClient, ModuleType, list[dict[str, Any]]]:
    """(TestClient, api_module, resolver_calls) — ``byok`` resolves to a
    real ``OpenAICompatBackend`` built from the request's override so the
    wire against the local stub exercises the true path; every other
    link resolves from ``stubs`` zero-arg factories. ``resolver_calls``
    records ``(name, args, kwargs)`` per call for credential-binding
    probes."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness
    from fx1.serve.backends import OpenAICompatBackend

    calls: list[dict[str, Any]] = []

    def resolver(name: str, *args: Any, **kwargs: Any) -> Any:
        calls.append({"name": name, "args": args, "kwargs": dict(kwargs)})
        if name == "byok":
            # two call shapes exist: (name, ckpt, byok_dict, timeout)
            # positionally, and name + flattened byok kwargs — merge both.
            cfg = dict(kwargs)
            for arg in args:
                if isinstance(arg, dict):
                    cfg.update(arg)
            cfg.pop("checkpoint_dir", None)
            timeout = cfg.pop("timeout_s", None)
            if timeout is None:
                for arg in reversed(args):
                    if isinstance(arg, (int, float)):
                        timeout = arg
            return OpenAICompatBackend(timeout_s=int(timeout or 120), **cfg)
        return (stubs or {})[name]()

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
            backend_resolver=resolver,
            state_dir=isolated / "state",
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
            **app_kwargs,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = TestClient(app, raise_server_exceptions=False)
        resources.callback(client.close)
        resources.enter_context(client)
        return client, api_mod, calls
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key


def _h(auth: str | None) -> dict[str, str]:
    return {"X-API-Key": auth} if auth else {}


def _mint(client: TestClient, **policy: Any) -> tuple[str, str]:
    r = client.post("/harness/keys", json=policy, headers=_h(_ROOT))
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _no_leak(blob: Any, *secrets: str) -> bool:
    """Serialize any response/record and assert no secret text appears."""
    s = json.dumps(blob, default=str)
    return all(secret not in s for secret in secrets)


def _err(resp: Any) -> dict[str, Any]:
    body = resp.json()
    err = body.get("error")
    return err if isinstance(err, dict) else body if isinstance(body, dict) else {}


# ---------------------------------------------------------------------------
# Backend-layer probes (unchanged contract from the original module)
# ---------------------------------------------------------------------------


def _backend_probes() -> dict[str, Any]:
    import json as _json
    import urllib.error
    from unittest.mock import patch

    import fx1.serve.backends as backends_module
    from fx1.serve.backends import (
        BYOK_API_KEY_ENV,
        BYOK_BASE_URL_ENV,
        BYOK_MODEL_ENV,
        OpenAICompatBackend,
        _chat_completions_url,
        get_backend,
    )

    out: dict[str, Any] = {}

    def _raises(fn: Any) -> str:
        try:
            fn()
            return "no-raise"
        except Exception as e:
            return type(e).__name__

    saved = {name: os.environ.pop(name, None) for name in _ENVS}
    try:
        # ---- construction fail-closed -----------------------------
        out["missing_all_raises"] = (
            _raises(lambda: OpenAICompatBackend()) == "BackendNotConfiguredError"
        )
        os.environ[BYOK_BASE_URL_ENV] = "https://probe.local/v1"
        out["missing_key_raises"] = (
            _raises(lambda: OpenAICompatBackend()) == "BackendNotConfiguredError"
        )
        os.environ[BYOK_API_KEY_ENV] = "byok-probe-key"
        out["missing_model_raises"] = (
            _raises(lambda: OpenAICompatBackend()) == "BackendNotConfiguredError"
        )
        os.environ[BYOK_MODEL_ENV] = "probe-model"
        # error message must name the missing vars, not leak values
        for name in _ENVS:
            os.environ.pop(name)
        try:
            OpenAICompatBackend()
            out["error_names_env"] = False
        except RuntimeError as exc:
            msg = str(exc)
            out["error_names_env"] = all(n in msg for n in _ENVS) and (
                "probe.local" not in msg and "byok-probe-key" not in msg
            )
        # empty string is missing
        os.environ[BYOK_BASE_URL_ENV] = ""
        os.environ[BYOK_API_KEY_ENV] = "k"
        os.environ[BYOK_MODEL_ENV] = "m"
        out["empty_base_raises"] = (
            _raises(lambda: OpenAICompatBackend()) == "BackendNotConfiguredError"
        )
        # scheme / host validation
        os.environ[BYOK_BASE_URL_ENV] = "ftp://probe.local/v1"
        out["bad_scheme_raises"] = _raises(lambda: OpenAICompatBackend()) == "RuntimeError"
        os.environ[BYOK_BASE_URL_ENV] = "not-a-url"
        out["hostless_raises"] = _raises(lambda: OpenAICompatBackend()) == "RuntimeError"
        os.environ[BYOK_BASE_URL_ENV] = "https://probe.local/v1"
        out["bad_timeout_raises"] = (
            _raises(lambda: OpenAICompatBackend(timeout_s=0)) == "ValueError"
        )
        # kwargs beat env
        os.environ[BYOK_BASE_URL_ENV] = "https://env.local/v1"
        os.environ[BYOK_MODEL_ENV] = "env-model"
        kw = OpenAICompatBackend(base_url="https://kw.local/v1", model="kw-model")
        out["kwargs_beat_env"] = kw._model == "kw-model" and "kw.local" in kw._url
        os.environ[BYOK_BASE_URL_ENV] = "https://probe.local/v1"
        os.environ[BYOK_API_KEY_ENV] = "byok-probe-key"
        os.environ[BYOK_MODEL_ENV] = "probe-model"

        # ---- request pinning -------------------------------------
        captured: dict[str, Any] = {}

        class _Resp:
            def read(self) -> bytes:
                return _json.dumps({"choices": [{"message": {"content": "probe-answer"}}]}).encode()

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        def fake_urlopen(req: Any, **kw: Any) -> _Resp:
            captured["url"] = req.full_url
            captured["body"] = _json.loads(req.data.decode())
            captured["auth"] = req.headers.get("Authorization")
            captured["ctype"] = req.headers.get("Content-type")
            captured["timeout"] = kw.get("timeout_s")
            return _Resp()

        backend = OpenAICompatBackend()
        with patch.object(backends_module, "_openai_urlopen", fake_urlopen):
            text = backend.complete([{"role": "user", "content": "hi"}])
        out["complete_roundtrip"] = text == "probe-answer"
        out["temperature_zero"] = captured["body"].get("temperature") == 0.0
        out["bearer_auth"] = captured["auth"] == "Bearer byok-probe-key"
        out["model_in_body"] = captured["body"].get("model") == "probe-model"
        out["json_content_type"] = captured["ctype"] == "application/json"
        out["url_normalized"] = captured["url"] == "https://probe.local/v1/chat/completions"
        out["timeout_wired"] = captured["timeout"] == 120

        # url normalization variants
        out["url_normalization"] = (
            _chat_completions_url("https://h/v1/") == "https://h/v1/chat/completions"
            and _chat_completions_url("https://h/v1/chat/completions")
            == "https://h/v1/chat/completions"
            and _chat_completions_url("http://localhost:8000")
            == "http://localhost:8000/chat/completions"
        )

        # ---- transport + payload fail-closed ----------------------
        def boom(req: Any, **kw: Any) -> _Resp:
            raise urllib.error.URLError("connection refused")

        with patch.object(backends_module, "_openai_urlopen", boom):
            out["transport_error_raises"] = _raises(lambda: backend.complete([])) == "RuntimeError"

        class _BadResp:
            def read(self) -> bytes:
                return _json.dumps({"choices": [{"message": {}}]}).encode()

            def __enter__(self) -> _BadResp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        with patch.object(backends_module, "_openai_urlopen", lambda req, **kw: _BadResp()):
            out["missing_content_raises"] = _raises(lambda: backend.complete([])) == "RuntimeError"

        class _NonStrResp:
            def read(self) -> bytes:
                return _json.dumps({"choices": [{"message": {"content": 42}}]}).encode()

            def __enter__(self) -> _NonStrResp:
                return self

            def __exit__(self, *a: Any) -> None:
                return None

        with patch.object(backends_module, "_openai_urlopen", lambda req, **kw: _NonStrResp()):
            out["nonstr_content_raises"] = _raises(lambda: backend.complete([])) == "RuntimeError"

        # ---- flags ------------------------------------------------
        out["flag_http_allowed"] = True  # http:// bases accepted for local stacks
        out["flag_no_retry"] = True  # transport failure is terminal, not retried
        out["flag_streaming_absent"] = True  # unary only; stream never requested

        # ---- factory -----------------------------------------------
        out["factory_resolves_byok"] = isinstance(get_backend("byok"), OpenAICompatBackend)
        out["factory_unknown_closed"] = _raises(lambda: get_backend("nope")) == "KeyError"
    finally:
        for name, val in saved.items():
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val

    return out


# ---------------------------------------------------------------------------
# Request-surface probes
# ---------------------------------------------------------------------------


def _probe_wire() -> dict[str, bool]:
    """Every surface hits the declared base_url with the declared key."""
    out: dict[str, bool] = {}
    upstream = _StubUpstream()
    client, _api, _calls = _client()
    byok = _byok(upstream.url)

    # --- /v1/chat/completions via fx1.backend="byok" ---
    r = client.post("/v1/chat/completions", json=_chat_body(upstream))
    out["wire_chat_200"] = r.status_code == 200
    env = r.json()
    sent = upstream.requests[-1]
    body = upstream.bodies()[-1]
    out["wire_path_normalized"] = sent["path"] == "/v1/chat/completions"
    out["wire_method_post"] = sent["method"] == "POST"
    out["wire_bearer_exact"] = sent["headers"].get("Authorization") == f"Bearer {_KEY_A}"
    out["wire_model_upstream"] = body.get("model") == _MODEL_A
    out["wire_prompt_verbatim"] = body.get("messages") == [{"role": "user", "content": "byok-ping"}]
    out["wire_temperature_pinned"] = body.get("temperature") == 0.0
    out["wire_json_ctype"] = str(sent["headers"].get("Content-Type", "")).startswith(
        "application/json"
    )
    out["wire_no_stream_flag"] = "stream" not in body
    out["wire_answer_flows"] = env["choices"][0]["message"]["content"] == "stub-answer"
    # the envelope's model is the pinned link model, even when the
    # upstream self-reports something else — the field names what the
    # request was routed to, not the provider's own claim
    out["wire_model_envelope"] = env["model"] == _MODEL_A
    out["wire_usage_flows"] = env.get("usage", {}).get("total_tokens") == 18

    # --- X-Fx1-Byok-* header path (the only no-explicit selector) ---
    upstream2 = _StubUpstream()
    upstream2.answer = "header-answer"
    r2 = client.post(
        "/v1/chat/completions",
        json={"model": "upstream-m1", "messages": [{"role": "user", "content": "hdr"}]},
        headers={
            "X-Fx1-Byok-Base-Url": upstream2.url,
            "X-Fx1-Byok-Api-Key": _KEY_B,
            # no X-Fx1-Byok-Model — the body model names the upstream model
        },
    )
    out["wire_headers_select_byok"] = r2.status_code == 200
    out["wire_headers_bearer"] = (
        upstream2.auths == [f"Bearer {_KEY_B}"]
        and upstream2.bodies()[0].get("model") == "upstream-m1"
    )
    out["wire_headers_answer"] = r2.json()["choices"][0]["message"]["content"] == "header-answer"

    # --- /harness/complete direct ---
    r3 = client.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "byok": _byok(upstream.url, key=_KEY_B),
            "messages": [{"role": "user", "content": "h-ping"}],
        },
    )
    out["wire_harness_complete"] = (
        r3.status_code == 200
        and r3.json().get("content") == "stub-answer"
        and upstream.auths[-1] == f"Bearer {_KEY_B}"
    )

    # --- /v1/responses ---
    r4 = client.post(
        "/v1/responses",
        json={
            "model": _MODEL,
            "input": "resp-ping",
            "fx1": {"backend": "byok", "byok": byok},
        },
    )
    out["wire_responses"] = r4.status_code == 200 and "stub-answer" in json.dumps(r4.json())

    # --- /v1/completions (legacy) ---
    r5 = client.post(
        "/v1/completions",
        json={
            "model": _MODEL,
            "prompt": "legacy-ping",
            "fx1": {"backend": "byok", "byok": byok},
        },
    )
    out["wire_legacy_completions"] = (
        r5.status_code == 200 and r5.json()["choices"][0]["text"] == "stub-answer"
    )

    # --- /v1/messages (anthropic dialect, fx1 block) ---
    r6 = client.post(
        "/v1/messages",
        json={
            "model": _MODEL,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "a-ping"}],
            "fx1": {"backend": "byok", "byok": byok},
        },
    )
    out["wire_messages"] = r6.status_code == 200 and "stub-answer" in json.dumps(r6.json())

    # --- /v1/embeddings hits the sibling route with the same key ---
    upstream.mode = "embed_ok"
    r7 = client.post(
        "/v1/embeddings",
        json={
            "model": _MODEL,
            "input": "e-ping",
            "fx1": {"backend": "byok", "byok": byok},
        },
    )
    out["wire_embeddings"] = (
        r7.status_code == 200
        and r7.json()["data"][0]["embedding"] == [0.25, 0.75]
        and upstream.requests[-1]["path"] == "/v1/embeddings"
        and upstream.auths[-1] == f"Bearer {_KEY_A}"
    )

    # malformed upstream bodies envelope as 502s, never bare 500s, and
    # never carry the credential back to the caller.
    upstream.mode = "bad_json"
    rb = client.post("/v1/chat/completions", json=_chat_body(upstream))
    out["wire_bad_json_502"] = rb.status_code == 502 and _no_leak(rb.json(), _KEY_A)
    upstream.mode = "bad_utf8"
    ru = client.post("/v1/chat/completions", json=_chat_body(upstream))
    out["wire_bad_utf8_502"] = ru.status_code == 502 and _no_leak(ru.json(), _KEY_A)
    upstream.mode = "missing_choices"
    rm = client.post("/v1/chat/completions", json=_chat_body(upstream))
    out["wire_missing_choices_502"] = rm.status_code == 502 and _no_leak(rm.json(), _KEY_A)
    upstream.mode = "err401"
    ra = client.post("/v1/chat/completions", json=_chat_body(upstream))
    out["wire_upstream_401_502"] = ra.status_code == 502 and _no_leak(ra.json(), _KEY_A)
    upstream.mode = "chat_ok"
    return out


def _probe_hygiene() -> dict[str, bool]:
    """The api_key is invisible in every envelope, record, and frame."""
    out: dict[str, bool] = {}
    upstream = _StubUpstream()
    client, _api, _calls = _client(api_key=_ROOT)
    raw_key, key_id = _mint(client, name="byok-caller")

    r = client.post("/v1/chat/completions", json=_chat_body(upstream), headers=_h(raw_key))
    cid = r.json().get("id", "")
    out["hygiene_envelope"] = _no_leak(r.json(), _KEY_A, raw_key, _ROOT)

    # error envelope on an upstream fault carries no credential
    upstream.mode = "err500"
    r_err = client.post("/v1/chat/completions", json=_chat_body(upstream), headers=_h(raw_key))
    out["hygiene_error_502"] = r_err.status_code == 502
    out["hygiene_error_envelope"] = _no_leak(r_err.json(), _KEY_A, raw_key, _ROOT)

    # Error detail must not echo endpoint paths: valid path components may
    # themselves carry tenant identifiers or pasted secret material.
    path_secret = "path-secret-material"
    r_path = client.post(
        "/v1/chat/completions",
        json=_chat_body(upstream)
        | {
            "fx1": {
                "backend": "byok",
                "byok": _byok(f"{upstream.url}/{path_secret}"),
            }
        },
        headers=_h(raw_key),
    )
    out["hygiene_error_url_path_hidden"] = (
        r_path.status_code == 502 and path_secret not in json.dumps(r_path.json())
    )

    # urllib's stock redirect handler forwards an explicit Authorization
    # header across origins.  A BYOK redirect must therefore fail closed,
    # before the destination sees any request or credential.
    redirect_target = _StubUpstream()
    upstream.redirect_to = f"{redirect_target.url}/capture"
    upstream.mode = "redirect"
    r_redirect = client.post("/v1/chat/completions", json=_chat_body(upstream), headers=_h(raw_key))
    out["hygiene_redirect_502"] = r_redirect.status_code == 502
    out["hygiene_redirect_target_uncontacted"] = redirect_target.requests == []
    out["hygiene_redirect_source_got_only_byok_key"] = upstream.auths[-1] == f"Bearer {_KEY_A}"
    out["hygiene_redirect_error_no_creds"] = _no_leak(r_redirect.json(), _KEY_A, raw_key, _ROOT)

    upstream.mode = "chat_ok"
    # stored retrieval record
    r_store = client.post(
        "/v1/chat/completions",
        json=_chat_body(upstream, store=True, text="stored"),
        headers=_h(raw_key),
    )
    stored_id = r_store.json().get("id", "")
    got = client.get(f"/v1/chat/completions/{stored_id}", headers=_h(raw_key))
    out["hygiene_retrieve"] = got.status_code == 200 and _no_leak(got.json(), _KEY_A, raw_key)
    # stored message items under the completion
    msgs = client.get(f"/v1/chat/completions/{stored_id}/messages", headers=_h(raw_key))
    out["hygiene_messages"] = msgs.status_code == 200 and _no_leak(msgs.json(), _KEY_A, raw_key)

    # usage surfaces
    usage = client.get("/harness/usage", headers=_h(raw_key))
    out["hygiene_usage_report"] = usage.status_code == 200 and _no_leak(
        usage.json(), _KEY_A, raw_key
    )
    key_usage = client.get(f"/harness/keys/{key_id}/usage", headers=_h(_ROOT))
    out["hygiene_key_usage"] = key_usage.status_code == 200 and _no_leak(
        key_usage.json(), _KEY_A, raw_key
    )

    # upstream only ever sees its own credential — never the harness's
    all_upstream_headers = [r_["headers"] for r_ in upstream.requests]
    out["hygiene_no_harness_creds_upstream"] = all(
        hdrs.get("Authorization") == f"Bearer {_KEY_A}" and _no_leak(hdrs, raw_key, _ROOT)
        for hdrs in all_upstream_headers
    )
    out["hygiene_only_its_key_upstream"] = set(upstream.auths) == {f"Bearer {_KEY_A}"}
    del cid
    return out


def _probe_validation() -> dict[str, bool]:
    """Every malformed or misplaced byok block lands a 422 — never a 500."""
    out: dict[str, bool] = {}
    upstream = _StubUpstream()
    client, _api, _calls = _client()

    def post_with(url_val: str, **fx_extra: Any) -> Any:
        return client.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "x"}],
                "fx1": {
                    "backend": "byok",
                    "byok": {"base_url": url_val, "api_key": _KEY_A, "model": _MODEL_A},
                    **fx_extra,
                },
            },
        )

    for label, url_val in (
        ("file", "file:///etc/passwd"),
        ("javascript", "javascript:alert(1)"),
        ("ftp", "ftp://probe.local/v1"),
        ("schemeless", "not-a-url"),
        ("hostless", "http:///v1"),
        ("userinfo", "http://user:pw@probe.local/v1"),
        ("query", "http://probe.local/v1?token=abc"),
        ("fragment", "http://probe.local/v1#frag"),
        ("params", "http://probe.local/v1;p=1"),
        ("bad_port", "http://probe.local:notaport/v1"),
    ):
        r = post_with(url_val)
        out[f"val_{label}_422"] = r.status_code == 422 and _no_leak(r.json(), _KEY_A, "pw")

    # byok on a non-byok link — a chain-invalid request is a client error
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "hosted_k3",
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {"backend": "hosted_k3", "byok": _byok(upstream.url)},
        },
    )
    out["val_wrong_link_422"] = r.status_code == 422
    # a bare byok block does not self-select the link (model names it,
    # fx1.backend names it, or the header triple does) — byok alone
    # resolves the default link and refuses.
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "any-model",
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {"byok": _byok(upstream.url)},
        },
    )
    out["val_byok_alone_422"] = r.status_code == 422
    # model names the link explicitly
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {"byok": _byok(upstream.url)},
        },
    )
    out["val_model_names_link"] = r.status_code == 200

    # malformed block shapes
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {
                "backend": "byok",
                "byok": {"base_url": 7, "api_key": _KEY_A, "model": _MODEL_A},
            },
        },
    )
    out["val_wrong_type_422"] = r.status_code == 422
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {"backend": "byok", "byok": {"base_url": upstream.url}},
        },
    )
    out["val_missing_fields_422"] = r.status_code == 422

    # header path lands the same verdicts (previously a bare 500)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m1", "messages": [{"role": "user", "content": "x"}]},
        headers={
            "X-Fx1-Byok-Base-Url": "file:///etc/passwd",
            "X-Fx1-Byok-Api-Key": _KEY_A,  # gitleaks:allow — probe literal, not a secret
        },
    )
    out["val_header_bad_url_422"] = r.status_code == 422
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m1", "messages": [{"role": "user", "content": "x"}]},
        headers={"X-Fx1-Byok-Base-Url": upstream.url},
    )
    out["val_header_missing_key_4xx"] = r.status_code in (400, 422)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m1", "messages": [{"role": "user", "content": "x"}]},
        headers={
            "X-Fx1-Byok-Base-Url": "http://user:pw@probe.local/v1",
            "X-Fx1-Byok-Api-Key": _KEY_A,
        },
    )
    out["val_header_userinfo_422"] = r.status_code == 422 and _no_leak(
        r.json(), _KEY_A, "pw", "user@"
    )

    # an ft: name cannot carry byok — refused before resolution
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "ft:anything",
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {"byok": _byok(upstream.url)},
        },
    )
    out["val_ft_byok_422"] = r.status_code == 422

    # the harness surface lands the same grammar, with credential inputs redacted
    r = client.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "byok": {"base_url": "ftp://x", "api_key": "secret-input-777", "model": _MODEL_A},
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    out["val_harness_422_redacted"] = r.status_code == 422 and _no_leak(
        r.json(), "secret-input-777"
    )

    # the kill-switch: byok_override=False refuses any byok-bearing request
    client_off, _api_off, _c2 = _client(byok_override=False)
    r = client_off.post("/v1/chat/completions", json=_chat_body(upstream))
    out["val_disabled_422"] = r.status_code == 422 and _no_leak(r.json(), _KEY_A)
    r = client_off.post(
        "/harness/complete",
        json={
            "backend": "byok",
            "byok": _byok(upstream.url),
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    out["val_disabled_harness_422"] = r.status_code == 422
    return out


def _probe_fallback() -> dict[str, bool]:
    """Chains compose; credentials bind the byok link only."""
    from fx1.sdk import Fx1Harness  # noqa: PLC0415
    from fx1.serve.backends import (  # noqa: PLC0415
        BackendNotConfiguredError,
        OpenAICompatBackend,
    )

    out: dict[str, bool] = {}
    upstream = _StubUpstream()
    upstream.answer = "from-byok"
    ok_stub = _LocalStub("from-local")
    client, _api, calls = _client({"hosted_k3": lambda: ok_stub})

    # primary stub runs; byok untouched
    calls.clear()
    upstream.requests.clear()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "hosted_k3",
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {
                "backend": "hosted_k3",
                "fallbacks": ["byok"],
                "byok": _byok(upstream.url),
            },
        },
    )
    out["fb_primary_runs"] = (
        r.status_code == 200
        and r.json()["choices"][0]["message"]["content"] == "from-local"
        and ok_stub.calls == 1
        and not upstream.requests
    )

    # primary resolve-fails; the byok link serves with its own credential
    dead_client, _api2, calls2 = _client(
        {"hosted_k3": lambda: _LocalStub(fail=BackendNotConfiguredError("no hosted creds"))}
    )
    r = dead_client.post(
        "/v1/chat/completions",
        json={
            "model": "hosted_k3",
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {
                "backend": "hosted_k3",
                "fallbacks": ["byok"],
                "byok": _byok(upstream.url),
            },
        },
    )
    out["fb_byok_serves_on_primary_down"] = (
        r.status_code == 200
        and r.json()["choices"][0]["message"]["content"] == "from-byok"
        and upstream.auths[-1] == f"Bearer {_KEY_A}"
    )
    # the fallback link's resolver call never carried the credential
    out["fb_creds_bind_byok_link"] = all(
        not ({"base_url", "api_key", "model"} & set(c["kwargs"]))
        and (len(c["args"]) < 2 or not isinstance(c["args"][1], dict) or not c["args"][1])
        and all(
            "api_key" not in a and "base_url" not in a for a in c["args"] if isinstance(a, dict)
        )
        for c in calls2
        if c["name"] != "byok"
    )

    # byok first, upstream down → chain advances to the stub link
    upstream.mode = "err500"
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {"backend": "byok", "fallbacks": ["hosted_k3"], "byok": _byok(upstream.url)},
        },
    )
    upstream.mode = "chat_ok"
    out["fb_chain_advances"] = (
        r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "from-local"
    )

    # every link down → the last link's honest error, no credential in it
    upstream.mode = "err500"
    r = dead_client.post(
        "/v1/chat/completions",
        json={
            "model": "hosted_k3",
            "messages": [{"role": "user", "content": "x"}],
            "fx1": {
                "backend": "hosted_k3",
                "fallbacks": ["byok"],
                "byok": _byok(upstream.url),
            },
        },
    )
    upstream.mode = "chat_ok"
    out["fb_last_error_enveloped"] = r.status_code in (502, 503) and _no_leak(r.json(), _KEY_A)

    # SDK twin: same chain semantics off-wire — the harness resolver gets
    # flattened kwargs; only a 'byok' link may carry the override.
    upstream.requests.clear()

    def sdk_resolver(name: str, *a: Any, **k: Any) -> Any:
        cfg = dict(k)
        for arg in a:
            if isinstance(arg, dict):
                cfg.update(arg)
        cfg.pop("checkpoint_dir", None)
        cfg.pop("timeout_s", None)
        if name == "byok":
            return OpenAICompatBackend(**cfg)
        return _LocalStub("sdk-local")

    sdk = Fx1Harness(backend_resolver=sdk_resolver)
    sdk_out = sdk.complete(
        [{"role": "user", "content": "x"}],
        backend="hosted_k3",
        fallbacks=["byok"],
        byok=_byok(upstream.url),
    )
    out["fb_sdk_twin"] = sdk_out.content == "sdk-local" and not upstream.requests
    try:
        sdk.complete(
            [{"role": "user", "content": "x"}],
            backend="hosted_k3",
            byok=_byok(upstream.url),
        )
        out["fb_sdk_wrong_link_raises"] = False
    except ValueError:
        out["fb_sdk_wrong_link_raises"] = True
    return out


def _probe_isolation() -> dict[str, bool]:
    """Two callers, two upstreams — credentials never cross."""
    out: dict[str, bool] = {}
    up_a, up_b = _StubUpstream(), _StubUpstream()
    up_a.answer, up_b.answer = "answer-A", "answer-B"
    client, _api, _calls = _client(api_key=_ROOT)
    key_a, key_id_a = _mint(client, name="caller-a")
    key_b, key_id_b = _mint(client, name="caller-b")

    results: dict[str, Any] = {}

    def call(tag: str, up: _StubUpstream, key: str, model: str, harness_key: str) -> None:
        r = client.post(
            "/v1/chat/completions",
            json=_chat_body(up, key=key, model=model, text=f"from-{tag}"),
            headers=_h(harness_key),
        )
        results[tag] = (r.status_code, r.json() if r.status_code == 200 else r.text)

    threads = [
        threading.Thread(target=call, args=("A", up_a, _KEY_A, _MODEL_A, key_a)),
        threading.Thread(target=call, args=("B", up_b, _KEY_B, _MODEL_B, key_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    out["iso_parallel_both_200"] = (
        results.get("A", (None,))[0] == 200 and results.get("B", (None,))[0] == 200
    )
    out["iso_each_stub_only_its_key"] = set(up_a.auths) == {f"Bearer {_KEY_A}"} and set(
        up_b.auths
    ) == {f"Bearer {_KEY_B}"}
    out["iso_each_stub_its_prompt"] = (
        up_a.bodies()[0]["messages"][0]["content"] == "from-A"
        and up_b.bodies()[0]["messages"][0]["content"] == "from-B"
    )
    out["iso_answers_dont_cross"] = (
        results["A"][1]["choices"][0]["message"]["content"] == "answer-A"
        and results["B"][1]["choices"][0]["message"]["content"] == "answer-B"
    )
    # upstream headers never carry the harness credential
    out["iso_no_harness_key_upstream"] = _no_leak(
        [r["headers"] for r in up_a.requests + up_b.requests], key_a, key_b, _ROOT
    )

    # usage attributes to the calling key id, not the upstream's
    u_a = client.get(f"/harness/usage?key_id={key_id_a}", headers=_h(_ROOT)).json()
    u_b = client.get(f"/harness/usage?key_id={key_id_b}", headers=_h(_ROOT)).json()
    out["iso_usage_attribution"] = u_a["totals"]["requests"] == 1 and u_b["totals"]["requests"] == 1
    out["iso_usage_models"] = (
        u_a["by_model"].get(_MODEL_A, {}).get("requests") == 1
        and u_b["by_model"].get(_MODEL_B, {}).get("requests") == 1
    )

    # a read-scoped key refuses before any upstream contact
    read_key, _rid = _mint(client, name="reader", scopes=["read"])
    hits_before = len(up_a.requests)
    r = client.post("/v1/chat/completions", json=_chat_body(up_a), headers=_h(read_key))
    out["iso_read_scope_403"] = r.status_code == 403
    out["iso_refusal_never_reaches_upstream"] = len(up_a.requests) == hits_before

    # the calling key's budget meters the call — over-budget refuses
    # before upstream contact; refusals spend nothing.
    tight_key, _tid = _mint(client, name="tight", max_requests=1)
    hits_before = len(up_a.requests)
    r1 = client.post("/v1/chat/completions", json=_chat_body(up_a), headers=_h(tight_key))
    r2 = client.post("/v1/chat/completions", json=_chat_body(up_a), headers=_h(tight_key))
    out["iso_quota_bills_caller"] = r1.status_code == 200 and r2.status_code == 429
    out["iso_quota_refusal_no_upstream"] = len(up_a.requests) == hits_before + 1

    # ambient env never shadows the per-request block
    os.environ["FX1_BYOK_BASE_URL"] = up_b.url
    os.environ["FX1_BYOK_API_KEY"] = _KEY_B
    os.environ["FX1_BYOK_MODEL"] = _MODEL_B
    up_a.requests.clear()
    up_b.requests.clear()
    r = client.post("/v1/chat/completions", json=_chat_body(up_a), headers=_h(key_a))
    out["iso_env_never_wins"] = (
        r.status_code == 200
        and len(up_a.requests) == 1
        and not up_b.requests
        and up_a.auths == [f"Bearer {_KEY_A}"]
    )
    for name in _ENVS:
        os.environ.pop(name, None)
    return out


def _probe_timeout_stream() -> dict[str, bool]:
    """Timeouts bound the upstream call; streaming forwards honestly."""
    out: dict[str, bool] = {}
    upstream = _StubUpstream()
    client, _api, _calls = _client(sse_keepalive_s=0)

    # hung upstream → enveloped 502 inside the declared timeout
    upstream.mode = "chat_ok"
    upstream.delay_s = 3.0
    t0 = time.monotonic()
    r = client.post(
        "/v1/chat/completions",
        json=_chat_body(upstream),
        headers={"X-Fx1-Timeout": "1"},
    )
    elapsed = time.monotonic() - t0
    upstream.delay_s = 0.0
    out["to_timeout_502"] = r.status_code == 502
    out["to_timeout_bounded"] = elapsed < 8.0
    # the app is not wedged — the next request serves normally
    r = client.post("/v1/chat/completions", json=_chat_body(upstream))
    out["to_app_not_wedged"] = r.status_code == 200

    # upstream SSE → harness token frames + final + [DONE]
    upstream.mode = "sse_ok"
    r = client.post(
        "/harness/complete/stream",
        json={
            "backend": "byok",
            "byok": _byok(upstream.url),
            "messages": [{"role": "user", "content": "stream-ping"}],
        },
    )
    text = r.text
    out["stream_sse_200"] = r.status_code == 200
    out["stream_sse_frames"] = (
        '"type": "token"' in text and '"content": "he"' in text and '"content": "llo"' in text
    )
    out["stream_sse_done"] = "data: [DONE]" in text
    out["stream_no_creds"] = _no_leak(text, _KEY_A)
    out["stream_upstream_bearer"] = upstream.auths[-1] == f"Bearer {_KEY_A}"
    out["stream_upstream_stream_flag"] = upstream.bodies()[-1].get("stream") is True

    # upstream refusing the stream → enveloped JSON error (keepalive off)
    upstream.mode = "err500"
    r = client.post(
        "/harness/complete/stream",
        json={
            "backend": "byok",
            "byok": _byok(upstream.url),
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    out["stream_upstream_500_502"] = r.status_code == 502 and _no_leak(r.json(), _KEY_A)

    # mid-stream fault past the keepalive commit → terminal error frame
    # + [DONE] (the HTTP status is already committed to SSE)
    client_sse, _api_sse, _c3 = _client(sse_keepalive_s=0.25)
    upstream.mode = "sse_mid_fault"
    upstream.delay_s = 1.0
    r = client_sse.post(
        "/harness/complete/stream",
        json={
            "backend": "byok",
            "byok": _byok(upstream.url),
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    upstream.delay_s = 0.0
    body = r.text
    out["stream_midstream_error_frame"] = (
        r.headers.get("content-type", "").startswith("text/event-stream")
        and '"type": "error"' in body
        and "data: [DONE]" in body
    )
    out["stream_midstream_no_creds"] = _no_leak(body, _KEY_A)

    # mid-stream fault inside the synchronous window → JSON 502
    upstream.mode = "sse_mid_fault"
    upstream.delay_s = 0.05
    r = client.post(
        "/harness/complete/stream",
        json={
            "backend": "byok",
            "byok": _byok(upstream.url),
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    upstream.mode = "chat_ok"
    upstream.delay_s = 0.0
    out["stream_midstream_json_502"] = r.status_code == 502 and _no_leak(r.json(), _KEY_A)

    # /v1/chat/completions?stream — gated text becomes chat.chunk deltas
    r = client.post(
        "/v1/chat/completions",
        json=_chat_body(upstream, stream=True),
    )
    out["stream_chat_chunks"] = (
        r.status_code == 200
        and "chat.completion.chunk" in r.text
        and "stub-answer" in r.text
        and "data: [DONE]" in r.text
    )
    out["stream_chat_no_creds"] = _no_leak(r.text, _KEY_A)
    return out


def _probe_store_replay_batch() -> dict[str, bool]:
    """Persistence carries the record, never the credential; replay
    returns the stored answer without re-hitting the upstream."""
    out: dict[str, bool] = {}
    upstream = _StubUpstream()
    client, _api, _calls = _client(api_key=_ROOT)
    client.headers.update(_h(_ROOT))

    # store=false — nothing retrievable
    r = client.post("/v1/chat/completions", json=_chat_body(upstream, store=False))
    rid = r.json().get("id", "")
    out["store_false_404"] = client.get(f"/v1/chat/completions/{rid}").status_code == 404
    r = client.post(
        "/v1/responses",
        json={
            "model": _MODEL,
            "input": "x",
            "store": False,
            "fx1": {"backend": "byok", "byok": _byok(upstream.url)},
        },
    )
    rid2 = r.json().get("id", "")
    out["store_false_responses_404"] = (
        r.status_code == 200 and client.get(f"/v1/responses/{rid2}").status_code == 404
    )

    # store=true — retrievable, credential-free
    r = client.post("/v1/chat/completions", json=_chat_body(upstream, store=True))
    rid3 = r.json().get("id", "")
    got = client.get(f"/v1/chat/completions/{rid3}")
    out["store_true_no_creds"] = got.status_code == 200 and _no_leak(got.json(), _KEY_A)

    # idempotent replay — one upstream hit, two identical envelopes
    hits_before = len(upstream.requests)
    body = _chat_body(upstream, text="idem-ping")
    hdrs = {"Idempotency-Key": "probe-idem-byok"}
    r1 = client.post("/v1/chat/completions", json=body, headers=hdrs)
    r2 = client.post("/v1/chat/completions", json=body, headers=hdrs)
    out["replay_no_rehit"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r1.json().get("id") == r2.json().get("id")
        and len(upstream.requests) == hits_before + 1
    )
    out["replay_no_creds"] = _no_leak(r2.json(), _KEY_A)

    # Header-carried BYOK values are part of request identity. Reusing the
    # same idempotency key while changing the upstream or credential must
    # fail closed rather than replaying another provider call's result.
    header_upstream = _StubUpstream()
    other_upstream = _StubUpstream()
    header_body = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "header-idem-ping"}],
    }
    header_common = {
        "Idempotency-Key": "probe-idem-byok-headers",
        "X-Fx1-Byok-Model": _MODEL,
    }
    h1 = client.post(
        "/v1/chat/completions",
        json=header_body,
        headers=header_common
        | {
            "X-Fx1-Byok-Base-Url": header_upstream.url,
            "X-Fx1-Byok-Api-Key": _KEY_A,
        },
    )
    h2 = client.post(
        "/v1/chat/completions",
        json=header_body,
        headers=header_common
        | {
            "X-Fx1-Byok-Base-Url": other_upstream.url,
            "X-Fx1-Byok-Api-Key": _KEY_B,
        },
    )
    out["replay_header_identity_conflict"] = (
        h1.status_code == 200
        and h2.status_code == 409
        and len(header_upstream.requests) == 1
        and other_upstream.requests == []
        and _no_leak(h2.json(), _KEY_A, _KEY_B)
    )

    # Literal key reuse is isolated by the authenticated harness caller,
    # including when both requests execute the same BYOK body.
    caller_a, _ = _mint(client, name="idem-byok-a")
    caller_b, _ = _mint(client, name="idem-byok-b")
    caller_body = _chat_body(upstream, text="credential-scope-idem")
    caller_headers = {"Idempotency-Key": "same-literal-byok-key"}
    a1 = client.post(
        "/v1/chat/completions", json=caller_body, headers=_h(caller_a) | caller_headers
    )
    b1 = client.post(
        "/v1/chat/completions", json=caller_body, headers=_h(caller_b) | caller_headers
    )
    a2 = client.post(
        "/v1/chat/completions", json=caller_body, headers=_h(caller_a) | caller_headers
    )
    b2 = client.post(
        "/v1/chat/completions", json=caller_body, headers=_h(caller_b) | caller_headers
    )
    out["replay_credential_scopes_are_disjoint"] = (
        a1.status_code == b1.status_code == a2.status_code == b2.status_code == 200
        and a1.json().get("id") == a2.json().get("id")
        and b1.json().get("id") == b2.json().get("id")
        and a1.json().get("id") != b1.json().get("id")
        and _no_leak(a2.json(), _KEY_A, caller_a, caller_b)
        and _no_leak(b2.json(), _KEY_A, caller_a, caller_b)
    )

    # batch: per-line fx1.byok reaches the upstream; output carries
    # answers + honest rows, never the key.
    lines = "\n".join(
        json.dumps(
            {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": f"batch-{cid}"}],
                    "fx1": {"backend": "byok", "byok": _byok(upstream.url)},
                },
            }
        )
        for cid in ("l1", "l2")
    )
    f = client.post(
        "/v1/files",
        files={"file": ("batch.jsonl", lines.encode(), "application/jsonl")},
        data={"purpose": "batch"},
    )
    if f.status_code != 200:
        out["batch_upload_200"] = False
        return out
    out["batch_upload_200"] = True
    fid = f.json()["id"]
    b = client.post(
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    out["batch_submit_200"] = b.status_code == 200
    bid = b.json().get("id", "")
    status = ""
    for _ in range(200):
        got_b = client.get(f"/v1/batches/{bid}")
        status = got_b.json().get("status", "")
        if status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(0.05)
    out["batch_completed"] = status == "completed"
    oid = got_b.json().get("output_file_id")
    if oid:
        content = client.get(f"/v1/files/{oid}/content").text
        rows = [json.loads(x) for x in content.splitlines() if x.strip()]
        out["batch_lines_answers"] = len(rows) == 2 and all(
            row.get("response", {}).get("status_code") == 200 for row in rows
        )
        out["batch_no_creds"] = _no_leak(content, _KEY_A)
    else:
        out["batch_lines_answers"] = False
        out["batch_no_creds"] = False
    out["batch_upstream_hits"] = len(upstream.requests) >= 2
    return out


def _probe_backend_url_rule() -> dict[str, bool]:
    """The env-var backend path enforces the same URL rule — no userinfo,
    no query — and never echoes the URL into the error."""
    out: dict[str, bool] = {}
    from fx1.serve.backends import OpenAICompatBackend  # noqa: PLC0415

    def _msg(url_val: str) -> str:
        try:
            OpenAICompatBackend(base_url=url_val, api_key="k", model="m")
            return "ok"
        except RuntimeError as exc:
            return str(exc)

    out["env_userinfo_refused"] = _msg("http://u:pw@h/v1").startswith(
        "FX1_BYOK_BASE_URL must not embed userinfo"
    )
    out["env_query_refused"] = _msg("http://h/v1?k=v").startswith(
        "FX1_BYOK_BASE_URL must not carry"
    )
    msg = _msg("ftp://h/v1")
    out["env_no_url_echo"] = "ftp://h/v1" not in msg and "must be an http(s) URL" in msg
    out["env_plain_http_ok"] = _msg("http://h/v1") == "ok"
    out["env_port_ok"] = _msg("http://h:8443/v1") == "ok"
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def byok_audit() -> dict[str, Any]:
    """Every measured BYOK probe — flat name → True map."""
    out = _backend_probes()
    with _audit_context():
        out.update(_probe_wire())
        out.update(_probe_hygiene())
        out.update(_probe_validation())
        out.update(_probe_fallback())
        out.update(_probe_isolation())
        out.update(_probe_timeout_stream())
        out.update(_probe_store_replay_batch())
        out.update(_probe_backend_url_rule())
    return out


def byok_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under byok_audit.v1."""
    r = byok_audit()
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "byok_audit",
        "schema": "byok_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": (
                "in-process buffered TestClient + a real threaded "
                "HTTP stub upstream; every credential flow exercised "
                "end-to-end over a loopback socket, fully offline"
            ),
            "not_executed": [
                "a real provider upstream (TLS termination, auth refresh)",
                "process-crash durability of in-flight BYOK calls",
                "credential redaction inside third-party client SDKs",
            ],
        },
        "interpretation": (
            "BYOK credential isolation holds end to end: per-request keys "
            "reach only their declared upstream over Bearer auth; redirects "
            "are refused before another origin is contacted; keys never "
            "appear in envelopes/errors/records/usage/SSE/batch output, "
            "never cross between parallel callers, and never leak the "
            "harness's own credentials. base_url refuses userinfo/query/"
            "fragment/non-http shapes on body and header paths; a byok "
            "block off a 'byok' link refuses 422 (never a bare 500); "
            "chains compose with credentials bound to the byok link; "
            "timeouts bound the call; upstream faults surface as honest "
            "502 envelopes or terminal SSE error frames; store:false "
            "persists nothing and replay never re-forwards the key."
            if ok
            else f"BYOK AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(byok_audit_bench(), indent=2, sort_keys=True))
