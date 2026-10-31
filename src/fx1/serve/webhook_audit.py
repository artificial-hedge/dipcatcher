"""webhook_audit — signed-delivery contract battery for the fx1 webhooks.

``api_audit`` pins the wire contract's happy paths; this battery attacks
the *delivery* contract every ``callback_url`` surface shares —
``/harness/jobs``, ``/harness/evals``, ``/v1/fine_tuning/jobs``,
``/v1/batches``, and ``/v1/messages/batches`` — against a real HTTP sink
on loopback (``http.server.ThreadingHTTPServer``; nothing leaves the
box):

- *Signature correctness* — the ``X-Fx1-Webhook-*`` pair recomputes from
  the declared secret over ``<ts>.<raw body>``; the wrong secret, a
  tampered body, a tampered timestamp, or a replayed-stale timestamp all
  fail :func:`fx1.serve.webhooks.verify_webhook`; unsigned deliveries
  carry no signature headers at all.
- *Fire-once* — exactly one POST per terminal transition across all five
  surfaces: completion, queued-cancel, repeated DELETE (the idempotent
  re-cancel must not re-fire), repeated GET polling, and batch
  cancel paths.
- *Retry semantics* — 5xx retries bounded at ``WEBHOOK_MAX_ATTEMPTS``,
  4xx is definitive (single attempt), connection-refused and read-timeout
  give up loudly into ``callback_status='failed'`` with an error string —
  never silent, never unbounded.
- *Payload* — the delivered body is the same shape GET returns (job
  record, eval record, ``fine_tuning.job``, ``batch``, ``message_batch``)
  with a terminal status, the create-time ``callback_url``, and a
  populated ``finished_at`` — the delivered record must be the final
  record, not a pre-terminal snapshot.
- *Failure visibility* — delivered and failed verdicts both surface on
  the record's GET projection (``callback_status`` / ``callback_attempts``
  / ``callback_error``), on batches too.
- *Security* — ``file:``/``gopher:``/``ftp:``/empty-netloc/userinfo URLs
  fail closed 422 at create on every surface; ``callback_secret`` without
  a URL refuses; the secret never serializes onto GET responses, the
  delivered body, or the state-dir journal; an unresolvable host does
  not hang the request path.

Probes are literal bools: ``True`` pins a contract that holds; ``False``
pins a measured divergence — the sealed receipt names every defect by
probe name so the finding survives byte-for-byte.

Honesty: every verdict is measured against a live local sink; no probe
stubs, mocks, or asserts its way past the network boundary.

Sealed ``webhook_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hmac as hmac_mod
import json
import os
import socket
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.harness import Harness
from fx1.serve.finetune import FTJobOutcome
from fx1.serve.webhooks import (
    WEBHOOK_MAX_ATTEMPTS,
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_TIMESTAMP_HEADER,
    deliver_signed,
    sign_webhook,
    verify_webhook,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

__all__ = ["webhook_audit", "webhook_audit_bench"]

_SECRET = "whsec-audit"  # NOSONAR — loopback-only test key, not a real credential
_JOB_TERMINAL = ("succeeded", "failed", "cancelled")
_WAIT_S = 15.0

# intentionally insecure callback URLs — every surface must refuse them
_URL_FILE = "file:///etc/passwd"  # NOSONAR — intentionally insecure scheme
_URL_GOPHER = "gopher://x/hook"  # NOSONAR — intentionally insecure scheme
_URL_FTP = "ftp://x/hook"  # NOSONAR — intentionally insecure scheme
_URL_EMPTY_NETLOC = "http:///hook"  # NOSONAR — intentionally malformed URL
_URL_JS = "javascript:alert(1)"  # NOSONAR — intentionally insecure scheme
_URL_USERINFO = "http://user:pass@127.0.0.1/hook"  # NOSONAR — intentionally insecure URL
_URL_NXHOST = "http://nonexistent.invalid./hook"  # NOSONAR — intentionally unresolvable

_ENV_KEYS = (
    "FX1_API_KEY",
    "MOONSHOT_API_KEY",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_CHECKPOINT_DIR",
    "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS",
)

_FT_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'


class _Hit:
    """One captured delivery: path, headers, and the raw body bytes."""

    __slots__ = ("body", "headers", "path")

    def __init__(self, path: str, headers: dict[str, str], body: bytes) -> None:
        self.path = path
        self.headers = headers
        self.body = body


class _Sink:
    """A real loopback HTTP webhook sink.

    ``ThreadingHTTPServer`` on ``127.0.0.1:0``; each POST lands as a
    ``_Hit``. Path selects the verdict: ``/fail`` 500s, ``/reject`` 404s,
    ``/flaky`` 500s twice then 200s, ``/stall`` parks past the caller's
    read timeout, ``/redir`` 307s onto ``/hook``; everything else 200s.
    """

    def __init__(self) -> None:
        self.hits: list[_Hit] = []
        self.path_n: dict[str, int] = {}
        self._lock = threading.Lock()
        self.stall_s = 1.0
        sink = self

        class _H(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server name
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                with sink._lock:  # noqa: SLF001 — same-module closure state
                    sink.hits.append(_Hit(self.path, dict(self.headers.items()), raw))
                    seen = sink.path_n.get(self.path, 0) + 1
                    sink.path_n[self.path] = seen
                if self.path == "/fail":
                    code = 500
                elif self.path == "/reject":
                    code = 404
                elif self.path == "/flaky":
                    code = 500 if seen < 3 else 200
                elif self.path == "/stall":
                    time.sleep(sink.stall_s)
                    code = 200
                elif self.path == "/redir":
                    self.send_response(307)
                    self.send_header("Location", "/hook")
                    self.end_headers()
                    return
                else:
                    code = 200
                self.send_response(code)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._srv.server_address[1])

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)


@dataclass
class _Ctx:
    """One app's test surface: client, app, sink."""

    client: TestClient
    app: FastAPI
    sink: _Sink


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


class _StubBackend:
    """A completions stub — every gated surface resolves to it."""

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        return f"clean:{messages[-1]['content']}"

    def close(self) -> None:
        pass


def _ft_runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
    emit("info", "bench runner")
    return FTJobOutcome(fine_tuned_model=None)


def _app(
    workdir: Path,
    *,
    runner: Callable[[list[str], int], tuple[int, str, str]] = _fast_runner,
    max_inflight: int = 4,
    state_dir: Path | None = None,
) -> FastAPI:
    import fx1.serve.api as api_mod  # noqa: PLC0415

    return api_mod.create_app(
        harness=Harness(runner=runner),
        backend_resolver=lambda *a, **k: _StubBackend(),
        ft_runner=_ft_runner,
        ft_dir=workdir / "ft",
        state_dir=state_dir,
        max_inflight=max_inflight,
    )


def _make_ctx(
    workdir: Path,
    *,
    sink: _Sink,
    runner: Callable[[list[str], int], tuple[int, str, str]] = _fast_runner,
    state_dir: Path | None = None,
) -> _Ctx:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    app = _app(workdir, runner=runner, state_dir=state_dir)
    return _Ctx(client=TestClient(app, raise_server_exceptions=False), app=app, sink=sink)


def _wait_job(client: TestClient, job_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    """Poll the job record until terminal (or callback verdict set)."""
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = client.get(f"/harness/jobs/{job_id}").json()
        if st.get("status") in _JOB_TERMINAL and st.get("callback_status"):
            return st
        time.sleep(0.05)
    return st


def _wait_hits(sink: _Sink, n: int, timeout: float = _WAIT_S) -> None:
    end = time.monotonic() + timeout
    while len(sink.hits) < n and time.monotonic() < end:
        time.sleep(0.05)


def _busy_executor(app: FastAPI, slots: int, sleep_s: float = 1.5) -> None:
    """Occupy every worker thread with a sleeper so a submitted record
    stays 'queued' — the deterministic way to reach the queued-cancel
    path. All five surfaces share ``jobs_executor``."""
    executor = app.state.jobs_executor
    release = threading.Event()
    for _ in range(slots):
        executor.submit(lambda: release.wait(timeout=sleep_s))
    time.sleep(0.1)


def _submit_job(
    client: TestClient,
    callback_url: str | None = None,
    *,
    secret: str | None = None,
    idem: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"command": "doctor"}
    if callback_url is not None:
        body["callback_url"] = callback_url
    if secret is not None:
        body["callback_secret"] = secret
    headers = {"Idempotency-Key": idem} if idem else {}
    r = client.post("/harness/jobs", json=body, headers=headers)
    assert r.status_code == 202, f"job submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _submit_eval(client: TestClient, callback_url: str) -> dict[str, Any]:
    """An eval with no resolvable backend fails terminal fast — the
    deterministic way to reach the terminal-webhook path."""
    r = client.post(
        "/harness/evals",
        json={"suite": "calibration", "backend": "hosted_k3", "callback_url": callback_url},
    )
    assert r.status_code == 202, f"eval submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _batch_create(
    client: TestClient, callback_url: str | None, *, secret: str | None = None
) -> dict[str, Any]:
    line = {
        "custom_id": "r1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "fx1", "messages": [{"role": "user", "content": "hi"}]},
    }
    up = client.post(
        "/v1/files",
        files={"file": ("in.jsonl", (json.dumps(line) + "\n").encode(), "application/jsonl")},
        data={"purpose": "batch"},
    )
    assert up.status_code == 200, up.text
    body: dict[str, Any] = {
        "input_file_id": up.json()["id"],
        "endpoint": "/v1/chat/completions",
    }
    if callback_url is not None:
        body["callback_url"] = callback_url
    if secret is not None:
        body["callback_secret"] = secret
    r = client.post("/v1/batches", json=body)
    assert r.status_code == 200, f"batch submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _abatch_create(
    client: TestClient, callback_url: str | None, *, secret: str | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "requests": [
            {
                "custom_id": "a",
                "params": {
                    "model": "fx1",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            }
        ]
    }
    if callback_url is not None:
        body["callback_url"] = callback_url
    if secret is not None:
        body["callback_secret"] = secret
    r = client.post("/v1/messages/batches", json=body)
    assert r.status_code == 200, f"abatch submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _ft_create(
    client: TestClient, callback_url: str | None, *, secret: str | None = None
) -> dict[str, Any]:
    up = client.post(
        "/v1/files",
        files={"file": ("c.jsonl", _FT_CORPUS, "application/jsonl")},
        data={"purpose": "fine-tune"},
    )
    assert up.status_code == 200, up.text
    body: dict[str, Any] = {"model": "fx1", "training_file": up.json()["id"]}
    if callback_url is not None:
        body["callback_url"] = callback_url
    if secret is not None:
        body["callback_secret"] = secret
    r = client.post("/v1/fine_tuning/jobs", json=body)
    assert r.status_code == 200, f"ft submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _wait_batch(client: TestClient, batch_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    b: dict[str, Any] = {}
    while time.monotonic() < end:
        b = client.get(f"/v1/batches/{batch_id}").json()
        if b.get("status") in ("completed", "failed", "expired", "cancelled"):
            return b
        time.sleep(0.05)
    return b


def _wait_abatch(client: TestClient, batch_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    b: dict[str, Any] = {}
    while time.monotonic() < end:
        b = client.get(f"/v1/messages/batches/{batch_id}").json()
        if b.get("processing_status") == "ended":
            return b
        time.sleep(0.05)
    return b


def _wait_verdict(client: TestClient, path: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    """Poll a record until its callback verdict is populated — the
    terminal status lands before the delivery verdict does."""
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = client.get(path).json()
        if st.get("callback_status"):
            return st
        time.sleep(0.05)
    return st


def _wait_ft(client: TestClient, job_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    j: dict[str, Any] = {}
    while time.monotonic() < end:
        j = client.get(f"/v1/fine_tuning/jobs/{job_id}").json()
        if j.get("status") in _JOB_TERMINAL:
            return j
        time.sleep(0.05)
    return j


def _probe_signature(ctx: _Ctx) -> dict[str, bool]:
    """HMAC signing/verification over the raw body + timestamp."""
    out: dict[str, bool] = {}
    jid = _submit_job(ctx.client, ctx.sink.url("/signed"), secret=_SECRET)["job_id"]
    _wait_job(ctx.client, jid)
    _wait_hits(ctx.sink, 1)
    hit = ctx.sink.hits[-1]
    sig = hit.headers.get(WEBHOOK_SIGNATURE_HEADER)
    ts = hit.headers.get(WEBHOOK_TIMESTAMP_HEADER)

    out["sig_header_shape"] = isinstance(sig, str) and sig.startswith("sha256=")
    out["sig_timestamp_present_fresh"] = (
        isinstance(ts, str) and ts.isdigit() and abs(time.time() - float(ts)) < 60
    )
    recomputed = sign_webhook(_SECRET, ts or "", hit.body)
    out["sig_recomputed_matches"] = bool(sig) and hmac_mod.compare_digest(recomputed, sig or "")
    out["sig_verifies_at_sink"] = verify_webhook(_SECRET, ts, sig, hit.body)
    out["sig_wrong_secret_fails"] = not verify_webhook("whsec-other", ts, sig, hit.body)
    out["sig_covers_body"] = not verify_webhook(_SECRET, ts, sig, hit.body + b" ")
    out["sig_covers_timestamp"] = not verify_webhook(
        _SECRET, str(int(ts or "0") + 1) if ts else None, sig, hit.body
    )
    out["sig_replay_stale_refused"] = not verify_webhook(
        _SECRET,
        ts,
        sig,
        hit.body,
        now=time.time() + 600.0,  # 10 min past delivery > 300s tolerance
    )
    out["sig_freshness_explicit_disable"] = verify_webhook(
        _SECRET, ts, sig, hit.body, tolerance_s=-1.0, now=time.time() + 10**9
    )
    out["sig_malformed_inputs_fail"] = (
        not verify_webhook(_SECRET, ts, "nonsense", hit.body)
        and not verify_webhook(_SECRET, ts, None, hit.body)
        and not verify_webhook(_SECRET, None, sig, hit.body)
        and not verify_webhook("", ts, sig, hit.body)
    )
    out["content_type_json"] = hit.headers.get("Content-Type") == "application/json"
    return out


def _probe_unsigned(ctx: _Ctx) -> dict[str, bool]:
    """A secret-less delivery carries no signature headers."""
    n0 = len(ctx.sink.hits)
    jid = _submit_job(ctx.client, ctx.sink.url("/unsigned"))["job_id"]
    _wait_job(ctx.client, jid)
    _wait_hits(ctx.sink, n0 + 1)
    hit = ctx.sink.hits[-1]
    return {
        "unsigned_no_signature_header": WEBHOOK_SIGNATURE_HEADER not in hit.headers
        and WEBHOOK_TIMESTAMP_HEADER not in hit.headers,
        "unsigned_still_delivered": hit.path == "/unsigned",
    }


def _probe_fire_once(ctx: _Ctx) -> dict[str, bool]:
    """Exactly one POST per terminal transition, on every surface."""
    client, sink = ctx.client, ctx.sink
    out: dict[str, bool] = {}

    n0 = len(sink.hits)
    _submit_job(client, sink.url("/j1"))
    _wait_hits(sink, n0 + 1)
    out["fire_job_success_once"] = sink.path_n.get("/j1") == 1

    # queued-cancel: occupy every worker, submit, DELETE while queued
    _busy_executor(ctx.app, 4)
    qjob = _submit_job(client, sink.url("/jcancel"))
    qid = qjob["job_id"]
    n0 = len(sink.hits)
    cxl = client.delete(f"/harness/jobs/{qid}")
    _wait_hits(sink, n0 + 1)
    out["fire_job_cancel_queued_once"] = (
        cxl.status_code == 200
        and cxl.json().get("status") == "cancelled"
        and sink.path_n.get("/jcancel") == 1
        and json.loads(sink.hits[-1].body).get("status") == "cancelled"
    )
    # repeated DELETE is a 200 idempotent re-read — it must NOT re-fire
    client.delete(f"/harness/jobs/{qid}")
    time.sleep(0.3)
    out["fire_job_repeated_cancel_once"] = sink.path_n.get("/jcancel") == 1

    # cancel on a running (or by then terminal) job 409s; the completion
    # still fires exactly once
    _busy_executor(ctx.app, 4)
    sjob = _submit_job(client, sink.url("/jslow"))
    slow_id = sjob["job_id"]
    end = time.monotonic() + _WAIT_S
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = client.get(f"/harness/jobs/{slow_id}").json()
        if st.get("status") in ("running", *_JOB_TERMINAL):
            break
        time.sleep(0.05)
    out["fire_job_cancel_running_409_once"] = (
        client.delete(f"/harness/jobs/{slow_id}").status_code == 409
        and _wait_job(client, slow_id).get("callback_status") == "delivered"
        and sink.path_n.get("/jslow") == 1
    )

    # repeated GET polling never re-fires
    client.get(f"/harness/jobs/{slow_id}")
    client.get(f"/harness/jobs/{slow_id}")
    time.sleep(0.2)
    out["fire_job_gets_no_refire"] = sink.path_n.get("/jslow") == 1

    # evals: an unresolvable backend fails terminal; cancel is the same
    n0 = len(sink.hits)
    evid = _submit_eval(client, sink.url("/ev"))["eval_id"]
    _wait_hits(sink, n0 + 1)
    evst = client.get(f"/harness/evals/{evid}").json()
    out["fire_eval_terminal_once"] = (
        evst.get("status") in _JOB_TERMINAL and sink.path_n.get("/ev") == 1
    )
    _busy_executor(ctx.app, 4)
    ev2id = _submit_eval(client, sink.url("/evcancel"))["eval_id"]
    n0 = len(sink.hits)
    cxl2 = client.delete(f"/harness/evals/{ev2id}")
    _wait_hits(sink, n0 + 1)
    out["fire_eval_cancel_queued_once"] = (
        cxl2.status_code == 200 and sink.path_n.get("/evcancel") == 1
    )
    client.delete(f"/harness/evals/{ev2id}")
    time.sleep(0.3)
    out["fire_eval_repeated_cancel_once"] = sink.path_n.get("/evcancel") == 1

    # fine-tuning jobs
    n0 = len(sink.hits)
    ft = _ft_create(client, sink.url("/ft"))
    ftst = _wait_ft(client, ft["id"])
    _wait_hits(sink, n0 + 1)
    out["fire_ft_success_once"] = (
        ftst.get("status") == "succeeded"
        and sink.path_n.get("/ft") == 1
        and json.loads(sink.hits[-1].body).get("status") == "succeeded"
    )
    _busy_executor(ctx.app, 4)
    ft2 = _ft_create(client, sink.url("/ftcancel"))
    n0 = len(sink.hits)
    client.post(f"/v1/fine_tuning/jobs/{ft2['id']}/cancel")
    _wait_hits(sink, n0 + 1)
    out["fire_ft_cancel_queued_once"] = (
        sink.path_n.get("/ftcancel") == 1
        and json.loads(sink.hits[-1].body).get("status") == "cancelled"
    )

    # /v1/batches
    n0 = len(sink.hits)
    b = _batch_create(client, sink.url("/b1"))
    bst = _wait_batch(client, b["id"])
    _wait_hits(sink, n0 + 1)
    out["fire_batch_completed_once"] = (
        bst.get("status") == "completed" and sink.path_n.get("/b1") == 1
    )
    client.get(f"/v1/batches/{b['id']}")
    client.get(f"/v1/batches/{b['id']}")
    time.sleep(0.2)
    out["fire_batch_gets_no_refire"] = sink.path_n.get("/b1") == 1

    # executor busy keeps the next batch 'validating' so the cancel lands
    # deterministically before the first line runs
    _busy_executor(ctx.app, 4)
    b2 = _batch_create(client, sink.url("/b2cancel"))
    n0 = len(sink.hits)
    client.post(f"/v1/batches/{b2['id']}/cancel")
    b2st = _wait_batch(client, b2["id"])
    _wait_hits(sink, n0 + 1)
    out["fire_batch_cancel_once"] = (
        b2st.get("status") == "cancelled" and sink.path_n.get("/b2cancel") == 1
    )

    # /v1/messages/batches
    n0 = len(sink.hits)
    ab = _abatch_create(client, sink.url("/ab1"))
    abst = _wait_abatch(client, ab["id"])
    _wait_hits(sink, n0 + 1)
    out["fire_abatch_ended_once"] = (
        abst.get("processing_status") == "ended" and sink.path_n.get("/ab1") == 1
    )
    client.get(f"/v1/messages/batches/{ab['id']}")
    client.get(f"/v1/messages/batches/{ab['id']}")
    time.sleep(0.2)
    out["fire_abatch_gets_no_refire"] = sink.path_n.get("/ab1") == 1

    _busy_executor(ctx.app, 4)
    ab2 = _abatch_create(client, sink.url("/ab2cancel"))
    n0 = len(sink.hits)
    client.post(f"/v1/messages/batches/{ab2['id']}/cancel")
    ab2st = _wait_abatch(client, ab2["id"])
    _wait_hits(sink, n0 + 1)
    out["fire_abatch_cancel_once"] = (
        ab2st.get("processing_status") == "ended"
        and sink.path_n.get("/ab2cancel") == 1
        and json.loads(sink.hits[-1].body).get("processing_status") == "ended"
    )
    return out


def _probe_retry(ctx: _Ctx) -> dict[str, bool]:
    """5xx retried bounded, 4xx definitive, transport faults loud."""
    client, sink = ctx.client, ctx.sink
    out: dict[str, bool] = {}

    jid = _submit_job(client, sink.url("/fail"))["job_id"]
    st = _wait_job(client, jid)
    out["retry_5xx_bounded_3"] = (
        sink.path_n.get("/fail") == WEBHOOK_MAX_ATTEMPTS
        and st.get("callback_status") == "failed"
        and st.get("callback_attempts") == WEBHOOK_MAX_ATTEMPTS
        and "500" in (st.get("callback_error") or "")
    )

    jid = _submit_job(client, sink.url("/reject"))["job_id"]
    st = _wait_job(client, jid)
    out["retry_4xx_definitive_1"] = (
        sink.path_n.get("/reject") == 1
        and st.get("callback_status") == "failed"
        and st.get("callback_attempts") == 1
        and "404" in (st.get("callback_error") or "")
    )

    # dead port: connection refused retries then gives up loudly
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    dead_port = dead.getsockname()[1]
    dead.close()
    jid = _submit_job(client, f"http://127.0.0.1:{dead_port}/hook")["job_id"]
    st = _wait_job(client, jid)
    out["retry_conn_refused_loud"] = (
        st.get("callback_status") == "failed"
        and st.get("callback_attempts") == WEBHOOK_MAX_ATTEMPTS
        and bool(st.get("callback_error"))
    )

    jid = _submit_job(client, sink.url("/flaky"))["job_id"]
    st = _wait_job(client, jid)
    out["retry_flaky_delivers"] = (
        sink.path_n.get("/flaky") == 3
        and st.get("callback_status") == "delivered"
        and st.get("callback_attempts") == 3
    )

    # a 307 is not followed for POST — urllib refuses to re-send a body
    # across a redirect, so the signed payload can never silently land on
    # a different path: 3 bounded retries, then a loud 'failed' verdict
    jid = _submit_job(client, sink.url("/redir"))["job_id"]
    st = _wait_job(client, jid)
    out["retry_redirect_307_refused_loud"] = (
        st.get("callback_status") == "failed"
        and st.get("callback_attempts") == WEBHOOK_MAX_ATTEMPTS
        and sink.path_n.get("/redir") == WEBHOOK_MAX_ATTEMPTS
        and sink.path_n.get("/hook", 0) == 0
    )

    # sink that accepts then stalls past the read timeout: unit-level
    # deliver_signed with a short timeout — retried, bounded, loud error
    sink.stall_s = 1.0
    t0 = time.monotonic()
    ok, err, attempts = deliver_signed(
        sink.url("/stall"), None, b"{}", timeout_s=0.2, backoff_s=0.01
    )
    elapsed = time.monotonic() - t0
    out["retry_timeout_retried_loud"] = (
        ok is False
        and attempts == WEBHOOK_MAX_ATTEMPTS
        and bool(err)
        and elapsed < 10.0
        and sink.path_n.get("/stall", 0) == WEBHOOK_MAX_ATTEMPTS
    )
    sink.stall_s = 0.0
    return out


def _probe_payload(ctx: _Ctx) -> dict[str, bool]:
    """The delivered body is the final record, not a stale snapshot."""
    client, sink = ctx.client, ctx.sink
    out: dict[str, bool] = {}

    seen_statuses: list[str] = []
    n0 = len(sink.hits)
    jid = _submit_job(client, sink.url("/pj"), secret=_SECRET)["job_id"]
    end = time.monotonic() + _WAIT_S
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = client.get(f"/harness/jobs/{jid}").json()
        if not seen_statuses or seen_statuses[-1] != st.get("status"):
            seen_statuses.append(str(st.get("status")))
        if st.get("status") in _JOB_TERMINAL and st.get("callback_status"):
            break
        time.sleep(0.05)
    _wait_hits(sink, n0 + 1)
    hit = sink.hits[-1]
    body = json.loads(hit.body)
    out["payload_job_parses_json"] = isinstance(body, dict)
    # the delivered body is serialized before the verdict exists —
    # callback_status is null in-flight by design; the verdict fields
    # surface on GET (pinned by verdict_delivered_visible)
    out["payload_job_shape"] = (
        body.get("job_id") == jid
        and body.get("status") in _JOB_TERMINAL
        and body.get("callback_url") == sink.url("/pj")
        and body.get("result", {}).get("ok") is True
    )
    out["payload_job_finished_at_set"] = body.get("finished_at") is not None
    out["payload_terminal_consistent"] = body.get("status") == st.get("status")
    # transitions never move backwards
    order = {"queued": 0, "running": 1, "succeeded": 2, "failed": 2, "cancelled": 2}
    out["order_status_monotone"] = seen_statuses == sorted(
        seen_statuses, key=lambda s: order.get(str(s), 0)
    )

    n0 = len(sink.hits)
    b = _batch_create(client, sink.url("/pb"))
    _wait_batch(client, b["id"])
    _wait_hits(sink, n0 + 1)
    bbody = json.loads(sink.hits[-1].body)
    out["payload_batch_envelope"] = (
        bbody.get("object") == "batch"
        and bbody.get("id") == b["id"]
        and bbody.get("status") == "completed"
        and bbody.get("request_counts", {}).get("completed") == 1
        and bbody.get("callback_url") == sink.url("/pb")
    )

    n0 = len(sink.hits)
    ab = _abatch_create(client, sink.url("/pab"))
    _wait_abatch(client, ab["id"])
    _wait_hits(sink, n0 + 1)
    abody = json.loads(sink.hits[-1].body)
    out["payload_abatch_envelope"] = (
        abody.get("type") == "message_batch"
        and abody.get("id") == ab["id"]
        and abody.get("processing_status") == "ended"
        and abody.get("request_counts", {}).get("succeeded") == 1
        and abody.get("callback_url") == sink.url("/pab")
    )

    n0 = len(sink.hits)
    ft = _ft_create(client, sink.url("/pft"))
    _wait_ft(client, ft["id"])
    _wait_hits(sink, n0 + 1)
    fbody = json.loads(sink.hits[-1].body)
    out["payload_ft_shape"] = (
        fbody.get("object") == "fine_tuning.job"
        and fbody.get("id") == ft["id"]
        and fbody.get("status") == "succeeded"
        and fbody.get("callback_url") == sink.url("/pft")
        and fbody.get("finished_at") is not None
    )

    n0 = len(sink.hits)
    evid = _submit_eval(client, sink.url("/pev"))["eval_id"]
    _wait_hits(sink, n0 + 1)
    ebody = json.loads(sink.hits[-1].body)
    evst = client.get(f"/harness/evals/{evid}").json()
    out["payload_eval_shape"] = (
        ebody.get("eval_id") == evid
        and ebody.get("status") in _JOB_TERMINAL
        and ebody.get("callback_url") == sink.url("/pev")
        and ebody.get("status") == evst.get("status")
        and ebody.get("finished_at") is not None
    )
    return out


def _probe_visibility(ctx: _Ctx) -> dict[str, bool]:
    """GET surfaces the delivery verdict on every record type."""
    client, sink = ctx.client, ctx.sink
    out: dict[str, bool] = {}

    jid = _submit_job(client, sink.url("/vok"))["job_id"]
    st = _wait_job(client, jid)
    out["verdict_delivered_visible"] = (
        st.get("callback_status") == "delivered"
        and st.get("callback_attempts") == 1
        and st.get("callback_error") is None
    )

    jid = _submit_job(client, sink.url("/fail"))["job_id"]
    st = _wait_job(client, jid)
    out["verdict_failed_visible"] = (
        st.get("callback_status") == "failed"
        and st.get("callback_attempts") == WEBHOOK_MAX_ATTEMPTS
        and isinstance(st.get("callback_error"), str)
        and "500" in str(st.get("callback_error"))
    )

    b = _batch_create(client, sink.url("/fail"))
    bst = _wait_verdict(client, f"/v1/batches/{b['id']}")
    out["verdict_failed_visible_batch"] = (
        bst.get("callback_status") == "failed"
        and bst.get("callback_attempts") == WEBHOOK_MAX_ATTEMPTS
        and bool(bst.get("callback_error"))
    )

    ab = _abatch_create(client, sink.url("/fail"))
    abst = _wait_verdict(client, f"/v1/messages/batches/{ab['id']}")
    out["verdict_failed_visible_abatch"] = (
        abst.get("callback_status") == "failed"
        and abst.get("callback_attempts") == WEBHOOK_MAX_ATTEMPTS
        and bool(abst.get("callback_error"))
    )
    return out


def _probe_secret_never_serializes(ctx: _Ctx, workdir: Path) -> dict[str, bool]:
    """``callback_secret`` never reaches a wire response, the delivered
    body, or the on-disk journal."""
    client, sink = ctx.client, ctx.sink
    out: dict[str, bool] = {}
    n0 = len(sink.hits)
    jid = _submit_job(client, sink.url("/s"), secret=_SECRET)["job_id"]
    st = _wait_job(client, jid)
    _wait_hits(sink, n0 + 1)
    got = client.get(f"/harness/jobs/{jid}")
    lst = client.get("/harness/jobs")
    out["secret_not_in_get"] = "callback_secret" not in st and _SECRET not in got.text
    out["secret_not_in_list"] = _SECRET not in lst.text
    out["secret_not_in_delivery"] = _SECRET.encode() not in sink.hits[-1].body

    # the state-dir journal persists records — the secret must not land
    jdir = workdir / "state"
    jctx = _make_ctx(workdir / "w2", sink=sink, state_dir=jdir)
    n0 = len(sink.hits)
    sjid = _submit_job(jctx.client, sink.url("/sj"), secret=_SECRET)["job_id"]
    _wait_job(jctx.client, sjid)
    _wait_hits(sink, n0 + 1)
    journal_bytes = b""
    jpath = jdir / "jobs.jsonl"
    if jpath.exists():
        journal_bytes = jpath.read_bytes()
    out["secret_not_in_journal"] = bool(journal_bytes) and _SECRET.encode() not in journal_bytes
    return out


def _probe_security(ctx: _Ctx) -> dict[str, bool]:
    """URL smuggling fails closed at create on every surface."""
    client, sink = ctx.client, ctx.sink
    out: dict[str, bool] = {}

    def _job_create(url: str) -> int:
        return client.post(
            "/harness/jobs", json={"command": "doctor", "callback_url": url}
        ).status_code

    def _eval_create(url: str) -> int:
        return client.post(
            "/harness/evals",
            json={
                "suite": "calibration",
                "backend": "hosted_k3",
                "callback_url": url,
            },
        ).status_code

    def _ft_create_code(url: str) -> int:
        return client.post(
            "/v1/fine_tuning/jobs",
            json={"model": "fx1", "training_file": "file-x", "callback_url": url},
        ).status_code

    def _batch_create_code(url: str) -> int:
        return client.post(
            "/v1/batches",
            json={
                "input_file_id": "file-x",
                "endpoint": "/v1/chat/completions",
                "callback_url": url,
            },
        ).status_code

    def _abatch_create_code(url: str) -> int:
        return client.post(
            "/v1/messages/batches",
            json={
                "requests": [
                    {
                        "custom_id": "a",
                        "params": {
                            "model": "fx1",
                            "max_tokens": 8,
                            "messages": [{"role": "user", "content": "x"}],
                        },
                    }
                ],
                "callback_url": url,
            },
        ).status_code

    surfaces = (
        _job_create,
        _eval_create,
        _ft_create_code,
        _batch_create_code,
        _abatch_create_code,
    )

    out["security_file_scheme_refused"] = all(fn(_URL_FILE) == 422 for fn in surfaces)
    out["security_gopher_ftp_refused"] = all(
        fn(_URL_GOPHER) == 422 and fn(_URL_FTP) == 422 for fn in surfaces
    )
    out["security_empty_netloc_refused"] = all(fn(_URL_EMPTY_NETLOC) == 422 for fn in surfaces)
    out["security_javascript_refused"] = all(fn(_URL_JS) == 422 for fn in surfaces)
    # userinfo smuggle: credentials inside the authority must refuse at
    # create — the validator rejects userinfo, not just bad schemes
    out["security_userinfo_refused"] = all(fn(_URL_USERINFO) == 422 for fn in surfaces)
    out["security_secret_requires_url"] = (
        client.post("/harness/jobs", json={"command": "doctor", "callback_secret": "x"}).status_code
        == 422
        and client.post(
            "/v1/fine_tuning/jobs",
            json={"model": "fx1", "training_file": "f", "callback_secret": "x"},
        ).status_code
        == 422
        and client.post(
            "/v1/batches",
            json={
                "input_file_id": "f",
                "endpoint": "/v1/chat/completions",
                "callback_secret": "x",
            },
        ).status_code
        == 422
    )

    # unresolvable host: the *request path* must not hang — submit returns
    # 202 promptly and the record lands a loud 'failed' verdict
    t0 = time.monotonic()
    r = client.post(
        "/harness/jobs",
        json={
            "command": "doctor",
            "callback_url": _URL_NXHOST,
        },
    )
    submit_s = time.monotonic() - t0
    jid = r.json()["job_id"]
    st = _wait_job(client, jid, timeout=45.0)
    out["security_unresolvable_no_hang"] = (
        r.status_code == 202
        and submit_s < 5.0
        and st.get("callback_status") == "failed"
        and bool(st.get("callback_error"))
    )

    # the create-time URL is the only delivery target — a same-key replay
    # with a swapped URL conflicts, never silently retargets
    r1 = client.post(
        "/harness/jobs",
        json={"command": "doctor", "callback_url": sink.url("/idem-a")},
        headers={"Idempotency-Key": "wa-idem"},
    )
    jid_a = r1.json()["job_id"]
    r2 = client.post(
        "/harness/jobs",
        json={"command": "doctor", "callback_url": sink.url("/idem-b")},
        headers={"Idempotency-Key": "wa-idem"},
    )
    _wait_job(client, jid_a)
    time.sleep(0.2)
    out["security_url_immutable_via_idem"] = (
        r2.status_code == 409 and sink.path_n.get("/idem-b", 0) == 0
    )
    return out


def webhook_audit() -> dict[str, Any]:
    """Run every probe against a real loopback sink; literal bools out."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    # The audit's real HTTP sink is deliberately loopback-only. Production
    # callback delivery remains public-network-only unless explicitly opted in.
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
    sink = _Sink()
    out: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            ctx = _make_ctx(Path(td) / "app", sink=sink)
            out.update(_probe_signature(ctx))
            out.update(_probe_unsigned(ctx))
            out.update(_probe_fire_once(ctx))
            out.update(_probe_retry(ctx))
            out.update(_probe_payload(ctx))
            out.update(_probe_visibility(ctx))
            out.update(_probe_secret_never_serializes(ctx, Path(td)))
            out.update(_probe_security(ctx))
    finally:
        sink.close()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def webhook_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = webhook_audit()
    ok = all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True)
    out: dict[str, Any] = {
        "kind": "webhook_audit",
        "schema": "webhook_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "Webhook delivery contract holds: HMAC-SHA256 signatures over "
            "<timestamp>.<body> verify against the declared secret and "
            "refuse wrong-secret, tampered, and replayed-stale deliveries; "
            "every terminal transition fires exactly once across jobs, "
            "evals, fine-tuning jobs, and both batch surfaces — repeated "
            "cancels and repeated GETs never re-fire; 5xx retries bound "
            "at three attempts with capped backoff, 4xx is definitive, "
            "dead and stalling endpoints give up loudly into a recorded "
            "'failed' verdict; the delivered body is the final record "
            "shape (finished_at populated, terminal status consistent "
            "with GET); callback verdicts surface on every record's GET; "
            "callback_secret never serializes to GETs, deliveries, or the "
            "journal; scheme/userinfo smuggling and secret-without-URL "
            "all refuse 422 at create; and an unresolvable host never "
            "hangs the request path."
            if ok
            else f"WEBHOOK AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(webhook_audit_bench(), indent=2, sort_keys=True))
