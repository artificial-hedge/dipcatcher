"""webhookdel_audit — webhook *dispatcher* internals battery for fx1 serve.

``webhook_audit`` pins the registration contract — ``callback_url``
validation, secret handling, and the fire-once guarantee each surface
inherits. This battery goes a level deeper, into
:func:`fx1.serve.webhooks.deliver_signed` itself and the app's
``_deliver_callback`` wiring, against a real loopback sink
(``http.server.ThreadingHTTPServer`` on ``127.0.0.1:0`` — nothing leaves
the box):

- *Verdict table* — every response class measured end-to-end: 2xx
  delivers on attempt 1; 3xx retries but is *never* followed (the
  redirect target sees zero requests); 4xx is definitive — one attempt,
  including 429 carrying ``Retry-After`` (pinned as the measured
  contract); 5xx retries to the ``WEBHOOK_MAX_ATTEMPTS`` ceiling; the
  error string names the concrete verdict.
- *Backoff* — inter-attempt gaps measured at the sink: the schedule
  doubles exponentially (0.5s then 1.0s under the default), a zero
  backoff is honored, each attempt opens a fresh TCP connection,
  re-resolves DNS, and re-mints the signature pair.
- *Sink faults* — unresolvable DNS, connection refused, read timeout,
  and TLS mismatch each classify honestly in the error string and retry
  bounded; a private-address literal refuses before any attempt, a
  private-resolving hostname refuses at resolution; a dead resolved
  address fails over to a live sibling within the same attempt.
- *Timing* — submit returns before delivery; a terminal record is
  GETable while its verdict is still in flight; a queued-cancel DELETE
  returns only after the cancelled delivery lands; a failing delivery
  holds its inflight slot through the whole retry schedule.
- *Ordering* — a single-worker executor delivers strictly FIFO; the
  default pool dispatches concurrently (inter-hit gap below the stall
  window); cancel-path deliveries run on request threads — bounded by
  the caller, not the worker pool.
- *Ledger* — under ``--state-dir`` the terminal mark journals the
  delivery verdict atomically; a restart restores the record *with* its
  verdict and never re-fires; a recovered queued job fails closed as
  ``failed`` with zero delivery attempts (``callback_secret`` never
  touches disk, so nothing can sign) and its idempotency key still
  replays.
- *Drain* — the latch refuses new work with ``503 draining`` but never
  freezes in-flight or queued work; shutdown flips queued jobs to
  ``cancelled`` and fires the webhook exactly once.
- *Envelope* — every refusal arrives in the path's own error grammar.

Probes are literal bools: ``True`` pins a contract that holds; ``False``
pins a measured divergence — the sealed receipt names every defect by
probe name so the finding survives byte-for-byte.

Honesty: every verdict is measured against a live loopback sink, the
real store, and the real journal; no probe stubs its way past the
network boundary. Timing probes use bounds at ~10% under the nominal
sleep so scheduling jitter cannot flake the battery.

Sealed ``webhookdel_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

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

from fx1.serve.webhook_audit import (
    _JOB_TERMINAL,
    _WAIT_S,
    _abatch_create,
    _app,
    _batch_create,
    _busy_executor,
    _fast_runner,
    _ft_create,
    _submit_eval,
    _submit_job,
    _wait_abatch,
    _wait_batch,
    _wait_ft,
    _wait_job,
    _wait_verdict,
)
from fx1.serve.webhooks import (
    WEBHOOK_MAX_ATTEMPTS,
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_TIMESTAMP_HEADER,
    check_callback_url,
    deliver_signed,
    verify_webhook,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

__all__ = ["webhookdel_audit", "webhookdel_audit_bench"]

_SECRET_A = "whsec-del-a"  # NOSONAR — loopback-only test key, not a real credential
_SECRET_B = "whsec-del-b"  # NOSONAR — loopback-only test key, not a real credential
_PRIV_ENV = "FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"

_URL_NXHOST = "http://nonexistent.invalid./hook"  # NOSONAR — intentionally unresolvable
_URL_BAD_FILE = "file:///etc/passwd"  # NOSONAR — intentionally insecure scheme
_URL_BAD_GOPHER = "gopher://x/hook"  # NOSONAR — intentionally insecure scheme
_URL_BAD_FTP = "ftp://x/hook"  # NOSONAR — intentionally insecure scheme
_URL_BAD_JS = "javascript:alert(1)"  # NOSONAR — intentionally insecure scheme
_URL_BAD_DATA = "data:text/plain,x"  # NOSONAR — intentionally insecure scheme
_URL_BAD_NETLOC = "http:///hook"  # NOSONAR — intentionally malformed URL
_URL_BAD_USERINFO = "http://user:pass@127.0.0.1/hook"  # NOSONAR — intentionally insecure URL
_URL_BAD_PORT0 = "http://127.0.0.1:0/hook"  # NOSONAR — intentionally invalid port

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
    _PRIV_ENV,
)

_JOB_POST = "/harness/jobs"


class _Hit:
    """One captured delivery: path, headers, body, arrival time, peer port."""

    __slots__ = ("body", "headers", "path", "peer_port", "t")

    def __init__(
        self, path: str, headers: dict[str, str], body: bytes, peer_port: int, t: float
    ) -> None:
        self.path = path
        self.headers = headers
        self.body = body
        self.peer_port = peer_port
        self.t = t


class _Sink:
    """A real loopback HTTP webhook sink with a path-driven verdict table.

    ``ThreadingHTTPServer`` on ``127.0.0.1:0``; each POST lands as a
    ``_Hit`` stamped with ``time.monotonic()`` and the client-side port
    (fresh-connection probing). Path grammar:

    - ``/sNNN`` — respond with status NNN verbatim (``/s429`` adds a
      ``Retry-After`` header so the pin covers it),
    - ``/rNNN`` — respond NNN with ``Location: /landing``,
    - ``/stall*`` — park ``stall_s`` before answering 200,
    - ``/flaky`` — 500 twice then 200,
    - anything else — 200.
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
                path = self.path.split("?", 1)[0]
                with sink._lock:  # noqa: SLF001 — same-module closure state
                    sink.hits.append(
                        _Hit(
                            self.path,
                            dict(self.headers.items()),
                            raw,
                            int(self.client_address[1]),
                            time.monotonic(),
                        )
                    )
                    seen = sink.path_n.get(path, 0) + 1
                    sink.path_n[path] = seen
                if path.startswith("/stall"):
                    time.sleep(sink.stall_s)
                    code = 200
                elif path == "/flaky":
                    code = 500 if seen < 3 else 200
                elif path.startswith("/r") and path[2:].isdigit():
                    self.send_response(int(path[2:]))
                    self.send_header("Location", "/landing")
                    self.end_headers()
                    return
                elif path == "/s429":
                    self.send_response(429)
                    self.send_header("Retry-After", "1")
                    self.end_headers()
                    return
                elif path.startswith("/s") and path[2:].isdigit():
                    code = int(path[2:])
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

    def hits_on(self, path: str) -> list[_Hit]:
        return [h for h in self.hits if h.path == path or h.path.startswith(f"{path}?")]

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)


class _V6Sink:
    """A sink bound to ``::1`` only — the multi-address failover target:

    ``localhost`` resolves to both ``::1`` and ``127.0.0.1`` on this
    platform; whichever the resolver lists first, the delivery must
    reach the live address."""

    def __init__(self) -> None:
        class _V6Server(ThreadingHTTPServer):
            address_family = socket.AF_INET6

        self.hits: list[_Hit] = []

        class _H(BaseHTTPRequestHandler):
            sink = self

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                self.sink.hits.append(
                    _Hit(self.path, dict(self.headers.items()), raw, 0, time.monotonic())
                )
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        self._srv = _V6Server(("::1", 0), _H)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._srv.server_address[1])

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)


@dataclass
class _Ctx:
    """One app's test surface: client, app, sink."""

    client: TestClient
    app: FastAPI


def _boom_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    raise RuntimeError("synthetic runner fault")


def _make_ctx(
    workdir: Path,
    *,
    runner: Callable[[list[str], int], tuple[int, str, str]] = _fast_runner,
    max_inflight: int = 4,
    state_dir: Path | None = None,
) -> _Ctx:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    app = _app(workdir, runner=runner, max_inflight=max_inflight, state_dir=state_dir)
    return _Ctx(client=TestClient(app, raise_server_exceptions=False), app=app)


def _wait_status(client: TestClient, job_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    """Poll until terminal — regardless of the callback verdict's state."""
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = client.get(f"/harness/jobs/{job_id}").json()
        if st.get("status") in _JOB_TERMINAL:
            return st
        time.sleep(0.05)
    return st


def _wait_hits(sink: _Sink, n: int, timeout: float = _WAIT_S) -> None:
    end = time.monotonic() + timeout
    while len(sink.hits) < n and time.monotonic() < end:
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Unit-level dispatcher probes — deliver_signed against the live sink
# ---------------------------------------------------------------------------


def _probe_verdict_table(sink: _Sink) -> dict[str, bool]:
    """The full response-class table, measured end-to-end."""
    out: dict[str, bool] = {}
    body = b"{}"

    two_xx = [deliver_signed(sink.url(f"/s{c}"), None, body, backoff_s=0) for c in (200, 201, 204)]
    out["verdict_2xx_delivered_once"] = all(r == (True, None, 1) for r in two_xx) and all(
        sink.path_n.get(f"/s{c}") == 1 for c in (200, 201, 204)
    )

    five_xx = [deliver_signed(sink.url(f"/s{c}"), None, body, backoff_s=0) for c in (500, 502, 503)]
    out["verdict_5xx_retried_bounded"] = all(
        ok is False and att == WEBHOOK_MAX_ATTEMPTS for ok, _e, att in five_xx
    ) and all(sink.path_n.get(f"/s{c}") == WEBHOOK_MAX_ATTEMPTS for c in (500, 502, 503))
    out["verdict_error_names_status"] = all(
        f"returned {c}" in (err or "")
        for (c, (_ok, err, _a)) in zip((500, 502, 503), five_xx, strict=True)
    )

    four_xx = [
        (c, deliver_signed(sink.url(f"/s{c}"), None, body, backoff_s=0))
        for c in (400, 404, 410, 422)
    ]
    out["verdict_4xx_definitive_once"] = all(
        ok is False and att == 1 and f"returned {c}" in (err or "") for c, (ok, err, att) in four_xx
    ) and all(sink.path_n.get(f"/s{c}") == 1 for c in (400, 404, 410, 422))

    # 429 + Retry-After is *definitive* under the 4xx contract — the
    # dispatcher does not honor the hint; pinned as the measured verdict.
    ok, err, att = deliver_signed(sink.url("/s429"), None, body, backoff_s=0)
    out["verdict_429_definitive_despite_retry_after"] = (
        ok is False and att == 1 and "429" in (err or "") and sink.path_n.get("/s429") == 1
    )

    # redirects are never followed: each code retries bounded, and the
    # Location target never receives a request
    land0 = sink.path_n.get("/landing", 0)
    redir = [
        (c, deliver_signed(sink.url(f"/r{c}"), None, body, backoff_s=0))
        for c in (301, 302, 307, 308)
    ]
    out["verdict_3xx_retried_never_followed"] = (
        all(ok is False and att == WEBHOOK_MAX_ATTEMPTS for _c, (ok, _e, att) in redir)
        and all(sink.path_n.get(f"/r{c}") == WEBHOOK_MAX_ATTEMPTS for c, _r in redir)
        and sink.path_n.get("/landing", 0) == land0
        and all(f"returned {c}" in (err or "") for c, (_o, err, _a) in redir)
    )
    return out


def _probe_backoff(sink: _Sink) -> dict[str, bool]:
    """The retry schedule is measured, not assumed."""
    out: dict[str, bool] = {}
    body = b'{"probe":"backoff"}'

    n0 = len(sink.hits)
    t0 = time.monotonic()
    ok, err, att = deliver_signed(sink.url("/s503"), _SECRET_A, body, backoff_s=0.5)
    elapsed = time.monotonic() - t0
    hs = [h for h in sink.hits[n0:] if h.path == "/s503"]
    gaps = [hs[i + 1].t - hs[i].t for i in range(len(hs) - 1)]
    # schedule: sleep(0.5) before attempt 2, sleep(1.0) before attempt 3;
    # bounds sit ~10% under nominal so jitter cannot flake the pin
    out["backoff_exponential_doubling"] = (
        att == WEBHOOK_MAX_ATTEMPTS
        and len(gaps) == 2
        and gaps[0] >= 0.45
        and gaps[1] >= 0.9
        and gaps[1] >= 1.8 * gaps[0]
    )
    out["backoff_total_bounded"] = ok is False and 1.4 <= elapsed < 20.0
    out["backoff_fresh_connection_each_attempt"] = len({h.peer_port for h in hs}) == len(hs)
    out["backoff_sig_verified_each_attempt"] = all(
        verify_webhook(
            _SECRET_A,
            h.headers.get(WEBHOOK_TIMESTAMP_HEADER),
            h.headers.get(WEBHOOK_SIGNATURE_HEADER),
            h.body,
        )
        for h in hs
    )
    out["backoff_payload_identical_each_attempt"] = all(h.body == body for h in hs)

    # zero backoff is honored — three attempts land back-to-back
    t0 = time.monotonic()
    deliver_signed(sink.url("/s500"), None, body, backoff_s=0.0)
    out["backoff_zero_no_sleep"] = (time.monotonic() - t0) < 0.5

    # DNS re-resolves every attempt — instrumentation wraps the real
    # resolver (call-through, never faked) and counts hostname lookups;
    # create_connection's internal lookups use the already-numeric
    # result, so only resolution-layer calls match the filter
    from unittest.mock import patch  # noqa: PLC0415

    calls = 0
    real_getaddrinfo = socket.getaddrinfo

    def _counting(host: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        if host == "localhost":
            calls += 1
        return real_getaddrinfo(host, *args, **kwargs)

    with patch.object(socket, "getaddrinfo", _counting):
        deliver_signed(
            sink.url("/s500").replace("127.0.0.1", "localhost"),
            None,
            body,
            backoff_s=0,
        )
    out["dns_reresolved_per_attempt"] = calls == WEBHOOK_MAX_ATTEMPTS
    return out


def _probe_faults(sink: _Sink) -> dict[str, bool]:
    """Every transport fault classifies loudly and retries bounded."""
    out: dict[str, bool] = {}
    body = b"{}"

    # connection refused on a dead port
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    dead_port = int(sock.getsockname()[1])
    sock.close()
    ok, err, att = deliver_signed(f"http://127.0.0.1:{dead_port}/hook", None, body, backoff_s=0.01)
    out["fault_conn_refused_retried_loud"] = (
        ok is False and att == WEBHOOK_MAX_ATTEMPTS and "refused" in (err or "").lower()
    )

    # unresolvable DNS — bounded, classified, never hanging
    t0 = time.monotonic()
    ok, err, att = deliver_signed(_URL_NXHOST, None, body, backoff_s=0.01, timeout_s=3.0)
    dns_elapsed = time.monotonic() - t0
    out["fault_dns_unresolvable_loud"] = (
        ok is False
        and att == WEBHOOK_MAX_ATTEMPTS
        and bool(err)
        and dns_elapsed < 30.0
        and ("gaierror" in (err or "") or "OSError" in (err or "") or "getaddrinfo" in (err or ""))
    )

    # read timeout — the sink accepts then stalls past the deadline
    sink.stall_s = 0.8
    prev_stall = sink.path_n.get("/stall", 0)
    t0 = time.monotonic()
    ok, err, att = deliver_signed(sink.url("/stall"), None, body, backoff_s=0.05, timeout_s=0.2)
    to_elapsed = time.monotonic() - t0
    sink.stall_s = 1.0
    out["fault_read_timeout_retried"] = (
        ok is False
        and att == WEBHOOK_MAX_ATTEMPTS
        and sink.path_n.get("/stall", 0) - prev_stall == WEBHOOK_MAX_ATTEMPTS
        and to_elapsed < 10.0
        and ("timed out" in (err or "") or "Timeout" in (err or ""))
    )

    # TLS mismatch — https onto a plain-HTTP port fails the handshake
    tls_url = sink.url("/hook").replace("http://", "https://", 1)
    ok, err, att = deliver_signed(tls_url, None, body, backoff_s=0.01, timeout_s=3.0)
    out["fault_tls_mismatch_retried"] = (
        ok is False and att == WEBHOOK_MAX_ATTEMPTS and "SSL" in (err or "")
    )

    # private-address refusals — env opt-out scoped to this probe only
    saved = os.environ.pop(_PRIV_ENV, None)
    try:
        ok, err, att = deliver_signed("http://127.0.0.1:9/hook", None, body, backoff_s=0)
        out["fault_private_literal_zero_attempts"] = (
            ok is False and att == 0 and "private" in (err or "")
        )
        ok2, err2, att2 = deliver_signed("http://localhost:9/hook", None, body, backoff_s=0)
        out["fault_private_resolved_refused"] = (
            ok2 is False
            and att2 == WEBHOOK_MAX_ATTEMPTS
            and "private or special-use" in (err2 or "")
        )
    finally:
        if saved is not None:
            os.environ[_PRIV_ENV] = saved

    # multi-address failover — ::1-only sink reached via "localhost":
    # whichever address family the resolver lists first, the dead
    # sibling is skipped inside the same attempt
    v6 = _V6Sink()
    try:
        ok, _err, att = deliver_signed(
            f"http://localhost:{v6.port}/hook", None, body, backoff_s=0.05, timeout_s=3.0
        )
        out["fault_multiaddr_delivers"] = ok is True and att == 1 and len(v6.hits) == 1
    finally:
        v6.close()
    return out


def _probe_validation(ctx: _Ctx, sink: _Sink) -> dict[str, bool]:
    """The delivery-side URL gate: schemes, userinfo, port zero, and the
    private-network opt-in — at the validator and at submit."""
    out: dict[str, bool] = {}
    refused = []
    for bad in (
        _URL_BAD_FILE,
        _URL_BAD_GOPHER,
        _URL_BAD_FTP,
        _URL_BAD_JS,
        _URL_BAD_DATA,
        _URL_BAD_NETLOC,
        _URL_BAD_USERINFO,
        _URL_BAD_PORT0,
    ):
        try:
            check_callback_url(bad)
            refused.append(False)
        except ValueError:
            refused.append(True)
    out["validate_refuses_bad_urls"] = all(refused)
    out["validate_accepts_http_https"] = (
        check_callback_url("http://example.com/hook") == "http://example.com/hook"
        and check_callback_url("https://example.com/hook") == "https://example.com/hook"
    )

    # deliver_signed never raises into its caller, even on garbage input
    never_raised = True
    for bad in ("", "not a url", _URL_BAD_FILE, _URL_BAD_PORT0):
        try:
            ok, _err, att = deliver_signed(bad, None, b"{}", backoff_s=0)
            never_raised = never_raised and ok is False and att == 0
        except Exception:  # noqa: BLE001 — the probe asserts the opposite
            never_raised = False
    out["validate_deliver_never_raises"] = never_raised

    # the private-network opt-in is a real gate: literal loopback refuses
    # at submit with env unset, accepts with env set
    saved = os.environ.pop(_PRIV_ENV, None)
    try:
        r = ctx.client.post(
            _JOB_POST,
            json={"command": "doctor", "callback_url": "http://127.0.0.1:9/hook"},
        )
        refused_code = r.status_code
    finally:
        os.environ[_PRIV_ENV] = saved or "1"
    r_on = ctx.client.post(
        _JOB_POST,
        json={"command": "doctor", "callback_url": sink.url("/gated")},
    )
    out["validate_loopback_gate_env"] = refused_code == 422 and r_on.status_code == 202

    out["validate_secret_requires_url_422"] = (
        ctx.client.post(_JOB_POST, json={"command": "doctor", "callback_secret": "x"}).status_code
        == 422
        and ctx.client.post(
            "/v1/batches",
            json={
                "input_file_id": "f",
                "endpoint": "/v1/chat/completions",
                "callback_secret": "x",
            },
        ).status_code
        == 422
    )
    return out


def _probe_signing(sink: _Sink) -> dict[str, bool]:
    """Per-delivery signature semantics: names, format, rotation."""
    out: dict[str, bool] = {}
    n0 = len(sink.hits)
    ok_a, _e, _a = deliver_signed(sink.url("/sig-a"), _SECRET_A, b'{"k":"a"}', backoff_s=0)
    ok_b, _e2, _a2 = deliver_signed(sink.url("/sig-b"), _SECRET_B, b'{"k":"b"}', backoff_s=0)
    ok_u, _e3, _a3 = deliver_signed(sink.url("/sig-u"), None, b'{"k":"u"}', backoff_s=0)
    hs = sink.hits[n0:]
    ha = next((h for h in hs if h.path == "/sig-a"), None)
    hb = next((h for h in hs if h.path == "/sig-b"), None)
    hu = next((h for h in hs if h.path == "/sig-u"), None)

    out["sig_delivered_both_secrets"] = ok_a is True and ok_b is True and ok_u is True
    out["sig_header_names_exact"] = (
        ha is not None
        and ha.headers.get(WEBHOOK_SIGNATURE_HEADER) is not None
        and ha.headers.get(WEBHOOK_TIMESTAMP_HEADER) is not None
        and WEBHOOK_SIGNATURE_HEADER == "X-Fx1-Webhook-Signature"
        and WEBHOOK_TIMESTAMP_HEADER == "X-Fx1-Webhook-Timestamp"
    )
    sig_a = ha.headers.get(WEBHOOK_SIGNATURE_HEADER) if ha else None
    ts_a = ha.headers.get(WEBHOOK_TIMESTAMP_HEADER) if ha else None
    out["sig_format_sha256_hex"] = (
        isinstance(sig_a, str)
        and sig_a.startswith("sha256=")
        and len(sig_a) == 7 + 64
        and all(c in "0123456789abcdef" for c in sig_a[7:])
    )
    out["sig_timestamp_unix_fresh"] = (
        isinstance(ts_a, str) and ts_a.isdigit() and abs(time.time() - float(ts_a)) < 60.0
    )
    out["sig_verifies_each_hit"] = (
        ha is not None
        and hb is not None
        and verify_webhook(_SECRET_A, ts_a, sig_a, ha.body)
        and verify_webhook(
            _SECRET_B,
            hb.headers.get(WEBHOOK_TIMESTAMP_HEADER),
            hb.headers.get(WEBHOOK_SIGNATURE_HEADER),
            hb.body,
        )
    )
    # rotation isolation: each signature verifies only under its own secret
    sig_b = hb.headers.get(WEBHOOK_SIGNATURE_HEADER) if hb else None
    ts_b = hb.headers.get(WEBHOOK_TIMESTAMP_HEADER) if hb else None
    out["sig_rotation_secret_isolated"] = (
        not verify_webhook(_SECRET_B, ts_a, sig_a, ha.body if ha else b"")
        and not verify_webhook(_SECRET_A, ts_b, sig_b, hb.body if hb else b"")
        and sig_a != sig_b
    )
    out["sig_unsigned_headers_absent"] = (
        hu is not None
        and WEBHOOK_SIGNATURE_HEADER not in hu.headers
        and WEBHOOK_TIMESTAMP_HEADER not in hu.headers
    )
    # the query string is part of the delivered target — pinned path+query
    deliver_signed(sink.url("/sig-q?x=1&y=2"), None, b"{}", backoff_s=0)
    hq = sink.hits[-1]
    out["sig_target_preserves_query"] = "?x=1&y=2" in hq.path
    return out


# ---------------------------------------------------------------------------
# App-level probes — _deliver_callback as wired into the harness surface
# ---------------------------------------------------------------------------


def _probe_timing(td: Path, sink: _Sink) -> dict[str, bool]:
    """Synchronous-vs-async dispatch, measured on the wire."""
    out: dict[str, bool] = {}
    ctx = _make_ctx(td / "timing", max_inflight=1)
    sink.stall_s = 1.2

    # submit returns before the delivery verdict can possibly exist —
    # the stalled sink makes the post-submit window ~stall_s wide
    r = ctx.client.post(
        _JOB_POST, json={"command": "doctor", "callback_url": sink.url("/stall-sub")}
    )
    jid = r.json()["job_id"]
    st0 = ctx.client.get(f"/harness/jobs/{jid}").json()
    out["timing_submit_returns_before_delivery"] = (
        r.status_code == 202 and st0.get("callback_status") is None
    )

    # the terminal status is GETable while the verdict is mid-flight
    st: dict[str, Any] = {}
    pre_verdict = False
    end = time.monotonic() + _WAIT_S
    while time.monotonic() < end:
        st = ctx.client.get(f"/harness/jobs/{jid}").json()
        if st.get("status") == "succeeded":
            pre_verdict = st.get("callback_status") is None
            break
        time.sleep(0.02)
    _wait_job(ctx.client, jid)
    out["timing_terminal_before_verdict"] = st.get("status") == "succeeded" and pre_verdict

    # a failing delivery holds its inflight slot for the whole retry
    # schedule: a second submit is refused 503 while the worker sleeps
    # between attempts, and only admitted once the verdict lands
    j1 = _submit_job(ctx.client, sink.url("/s500"))["job_id"]
    r_mid = ctx.client.post(
        _JOB_POST, json={"command": "doctor", "callback_url": sink.url("/held")}
    )
    out["timing_slot_held_submit_refused"] = (
        r_mid.status_code == 503 and r_mid.json().get("code") == "over_capacity"
    )
    st1 = _wait_job(ctx.client, j1)
    j2 = _submit_job(ctx.client, sink.url("/held"))["job_id"]
    st2 = _wait_job(ctx.client, j2)
    out["timing_slot_held_through_retries"] = (
        st1.get("status") == "succeeded"
        and st1.get("callback_status") == "failed"
        and st1.get("callback_attempts") == WEBHOOK_MAX_ATTEMPTS
        and st2.get("status") == "succeeded"
        and st2.get("callback_status") == "delivered"
    )

    # a runner fault is job data — the failed record still fires signed
    ctxf = _make_ctx(td / "timing-fail", runner=_boom_runner, max_inflight=1)
    fjid = _submit_job(ctxf.client, sink.url("/jfail"), secret=_SECRET_A)["job_id"]
    fst = _wait_job(ctxf.client, fjid)
    fhits = sink.hits_on("/jfail")
    fbody = json.loads(fhits[-1].body) if fhits else {}
    fsig = fhits[-1].headers.get(WEBHOOK_SIGNATURE_HEADER) if fhits else None
    fts = fhits[-1].headers.get(WEBHOOK_TIMESTAMP_HEADER) if fhits else None
    out["timing_failed_job_fires_signed_failed"] = (
        fst.get("status") == "failed"
        and fst.get("callback_status") == "delivered"
        and len(fhits) == 1
        and fbody.get("status") == "failed"
        and "synthetic runner fault" in str(fbody.get("error"))
        and verify_webhook(_SECRET_A, fts, fsig, fhits[-1].body)
    )

    # a queued-cancel DELETE returns only after the cancelled delivery
    # lands — the last ctx probe: its queued record's semaphore releases
    # whenever the executor frees, which is fine to leave dangling here
    _busy_executor(ctx.app, 1, sleep_s=2.5)
    qid = _submit_job(ctx.client, sink.url("/stall-cancel"))["job_id"]
    n0 = len(sink.hits)
    t0 = time.monotonic()
    rc = ctx.client.delete(f"/harness/jobs/{qid}")
    cancel_elapsed = time.monotonic() - t0
    chits = sink.hits_on("/stall-cancel")
    out["timing_cancel_blocks_until_delivered"] = (
        rc.status_code == 200
        and rc.json().get("status") == "cancelled"
        and cancel_elapsed >= sink.stall_s * 0.9
        and len(sink.hits) >= n0 + 1
        and len(chits) == 1
    )
    sink.stall_s = 1.0
    return out


def _probe_ordering(td: Path, sink: _Sink) -> dict[str, bool]:
    """FIFO under a single worker; concurrent dispatch under a pool; the
    cancel path dispatches on request threads beyond the pool's bound."""
    out: dict[str, bool] = {}

    # serial executor — deliveries land strictly in submit order
    ctx1 = _make_ctx(td / "ord-serial", max_inflight=1)
    j1 = _submit_job(ctx1.client, sink.url("/ord-a"))["job_id"]
    j2 = _submit_job(ctx1.client, sink.url("/ord-b"))["job_id"]
    _wait_job(ctx1.client, j1)
    _wait_job(ctx1.client, j2)
    seq = [h.path for h in sink.hits if h.path in ("/ord-a", "/ord-b")]
    out["ordering_serial_fifo"] = seq == ["/ord-a", "/ord-b"]

    # a pool dispatches concurrently — two stalled deliveries overlap
    ctx2 = _make_ctx(td / "ord-par", max_inflight=2)
    sink.stall_s = 1.0
    ja = _submit_job(ctx2.client, sink.url("/par-a"))["job_id"]
    jb = _submit_job(ctx2.client, sink.url("/par-b"))["job_id"]
    _wait_job(ctx2.client, ja)
    _wait_job(ctx2.client, jb)
    ha = sink.hits_on("/par-a")
    hb = sink.hits_on("/par-b")
    out["ordering_parallel_dispatch"] = (
        len(ha) == 1 and len(hb) == 1 and abs(ha[0].t - hb[0].t) < sink.stall_s
    )

    # cancel-path deliveries run on request threads — every pool worker
    # is asleep, both queued records are admitted, and the two cancels
    # still deliver concurrently (never through the executor)
    ctx3 = _make_ctx(td / "ord-cancel", max_inflight=2)
    _busy_executor(ctx3.app, 2, sleep_s=3.0)
    q1 = _submit_job(ctx3.client, sink.url("/stall-cq-a"))["job_id"]
    q2 = _submit_job(ctx3.client, sink.url("/stall-cq-b"))["job_id"]
    codes: list[int] = []

    def _del(job_id: str) -> None:
        codes.append(ctx3.client.delete(f"/harness/jobs/{job_id}").status_code)

    threads = [threading.Thread(target=_del, args=(j,)) for j in (q1, q2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ca = sink.hits_on("/stall-cq-a")
    cb = sink.hits_on("/stall-cq-b")
    out["ordering_cancel_parallel_request_threads"] = (
        sorted(codes) == [200, 200]
        and len(ca) == 1
        and len(cb) == 1
        and abs(ca[0].t - cb[0].t) < sink.stall_s
    )
    sink.stall_s = 1.0
    return out


def _probe_fire_once(ctx: _Ctx, sink: _Sink) -> dict[str, bool]:
    """Exactly one delivery per terminal transition, per surface, and a
    second terminal transition never re-fires."""
    client = ctx.client
    out: dict[str, bool] = {}

    n0 = len(sink.hits)
    jid = _submit_job(client, sink.url("/fo-job"))["job_id"]
    st = _wait_job(client, jid)
    _wait_hits(sink, n0 + 1)
    out["fire_job_success_once"] = (
        st.get("status") == "succeeded" and sink.path_n.get("/fo-job") == 1
    )
    # a cancel on the terminal record 409s and never re-fires
    d = client.delete(f"/harness/jobs/{jid}")
    time.sleep(0.2)
    out["fire_job_cancel_terminal_409"] = d.status_code == 409 and sink.path_n.get("/fo-job") == 1
    # repeated GET polling never re-fires
    client.get(f"/harness/jobs/{jid}")
    client.get(f"/harness/jobs/{jid}")
    time.sleep(0.2)
    out["fire_job_gets_no_refire"] = sink.path_n.get("/fo-job") == 1

    # queued-cancel: occupy every worker so the delete lands pre-start
    _busy_executor(ctx.app, 4, sleep_s=3.0)
    qjob = _submit_job(client, sink.url("/fo-jc"))
    n0 = len(sink.hits)
    cxl = client.delete(f"/harness/jobs/{qjob['job_id']}")
    _wait_hits(sink, n0 + 1)
    jchits = sink.hits_on("/fo-jc")
    out["fire_job_queued_cancel_once"] = (
        cxl.status_code == 200
        and len(jchits) == 1
        and json.loads(jchits[-1].body).get("status") == "cancelled"
    )
    # the idempotent re-cancel 200s but never re-fires
    cxl2 = client.delete(f"/harness/jobs/{qjob['job_id']}")
    time.sleep(0.25)
    out["fire_job_repeated_cancel_no_refire"] = (
        cxl2.status_code == 200 and sink.path_n.get("/fo-jc") == 1
    )

    n0 = len(sink.hits)
    evid = _submit_eval(client, sink.url("/fo-ev"))["eval_id"]
    _wait_hits(sink, n0 + 1)
    evst = client.get(f"/harness/evals/{evid}").json()
    out["fire_eval_terminal_once"] = (
        evst.get("status") in _JOB_TERMINAL and sink.path_n.get("/fo-ev") == 1
    )

    n0 = len(sink.hits)
    ft = _ft_create(client, sink.url("/fo-ft"))
    ftst = _wait_ft(client, ft["id"])
    _wait_hits(sink, n0 + 1)
    out["fire_ft_success_once"] = (
        ftst.get("status") == "succeeded"
        and sink.path_n.get("/fo-ft") == 1
        and json.loads(sink.hits[-1].body).get("status") == "succeeded"
    )

    n0 = len(sink.hits)
    b = _batch_create(client, sink.url("/fo-b"))
    bst = _wait_batch(client, b["id"])
    _wait_hits(sink, n0 + 1)
    out["fire_batch_completed_once"] = (
        bst.get("status") == "completed" and sink.path_n.get("/fo-b") == 1
    )
    client.get(f"/v1/batches/{b['id']}")
    client.get(f"/v1/batches/{b['id']}")
    time.sleep(0.2)
    out["fire_batch_gets_no_refire"] = sink.path_n.get("/fo-b") == 1

    n0 = len(sink.hits)
    ab = _abatch_create(client, sink.url("/fo-ab"))
    abst = _wait_abatch(client, ab["id"])
    _wait_hits(sink, n0 + 1)
    out["fire_abatch_ended_once"] = (
        abst.get("processing_status") == "ended" and sink.path_n.get("/fo-ab") == 1
    )
    client.get(f"/v1/messages/batches/{ab['id']}")
    client.get(f"/v1/messages/batches/{ab['id']}")
    time.sleep(0.2)
    out["fire_abatch_gets_no_refire"] = sink.path_n.get("/fo-ab") == 1
    return out


def _probe_ledger(td: Path, sink: _Sink) -> dict[str, bool]:
    """``--state-dir`` journaling of the delivery verdict, and the
    honest-abandon recovery contract for never-delivered callbacks."""
    out: dict[str, bool] = {}
    jdir = td / "state"
    ctx = _make_ctx(td / "w1", state_dir=jdir)
    n0 = len(sink.hits)
    jid = _submit_job(ctx.client, sink.url("/led"), secret=_SECRET_A)["job_id"]
    st = _wait_job(ctx.client, jid)
    _wait_hits(sink, n0 + 1)

    jpath = jdir / "jobs.jsonl"
    journal_bytes = jpath.read_bytes() if jpath.exists() else b""
    last: dict[str, Any] = {}
    for raw in journal_bytes.splitlines():
        if not raw.strip():
            continue
        line = json.loads(raw)
        job = line.get("payload", {}).get("job")
        if isinstance(job, dict) and job.get("job_id") == jid:
            last = job
    out["ledger_terminal_verdict_journaled"] = (
        st.get("callback_status") == "delivered"
        and last.get("status") == "succeeded"
        and last.get("callback_status") == "delivered"
        and last.get("callback_attempts") == 1
        and last.get("callback_url") == sink.url("/led")
    )
    out["ledger_secret_never_journaled"] = (
        bool(journal_bytes) and _SECRET_A.encode() not in journal_bytes
    )

    # restart: a fresh app over the same state dir restores the record
    # WITH its verdict and never re-fires — the delivered flag is durable
    ctx2 = _make_ctx(td / "w2", state_dir=jdir)
    st2 = ctx2.client.get(f"/harness/jobs/{jid}").json()
    n_before = len(sink.hits)
    ctx2.client.get(f"/harness/jobs/{jid}")
    d = ctx2.client.delete(f"/harness/jobs/{jid}")
    time.sleep(0.3)
    out["ledger_restart_restores_verdict"] = (
        st2.get("status") == "succeeded"
        and st2.get("callback_status") == "delivered"
        and st2.get("callback_attempts") == 1
        and "callback_secret" not in st2
    )
    out["ledger_restart_no_refire"] = len(sink.hits) == n_before and d.status_code == 409

    # recovered queued job: no secret journaled means nothing can sign —
    # the honest verdict is 'failed' with zero delivery attempts
    jdir2 = td / "state2"
    ctxa = _make_ctx(td / "wA", state_dir=jdir2)
    _busy_executor(ctxa.app, 4, sleep_s=6.0)
    qjob = _submit_job(ctxa.client, sink.url("/led-queued"), secret=_SECRET_A, idem="wdel-1")
    n0 = len(sink.hits)
    ctxb = _make_ctx(td / "wB", state_dir=jdir2)
    qst = ctxb.client.get(f"/harness/jobs/{qjob['job_id']}").json()
    out["ledger_recovered_queued_fails_closed"] = (
        qst.get("status") == "failed"
        and "restarted" in str(qst.get("error"))
        and qst.get("callback_status") is None
        and qst.get("callback_attempts") == 0
        and sink.path_n.get("/led-queued", 0) == 0
        and len(sink.hits) == n0
    )
    # the idempotency mapping survives the restart — a same-key retry
    # replays the lost record instead of duplicating the run
    r2 = ctxb.client.post(
        _JOB_POST,
        json={
            "command": "doctor",
            "callback_url": sink.url("/led-queued"),
            "callback_secret": _SECRET_A,
        },
        headers={"Idempotency-Key": "wdel-1"},
    )
    out["ledger_idem_survives_restart"] = (
        r2.status_code == 202
        and r2.json().get("replayed") is True
        and r2.json().get("job_id") == qjob["job_id"]
    )
    return out


def _probe_drain(td: Path, sink: _Sink) -> dict[str, bool]:
    """The drain latch and the shutdown path behave honestly toward
    pending deliveries."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    out: dict[str, bool] = {}
    ctx = _make_ctx(td / "drain", max_inflight=1)
    _busy_executor(ctx.app, 1, sleep_s=2.5)
    qid = _submit_job(ctx.client, sink.url("/dq"))["job_id"]

    d = ctx.client.post("/harness/drain")
    r = ctx.client.post(_JOB_POST, json={"command": "doctor"})
    mid = ctx.client.get(f"/harness/jobs/{qid}").json()
    out["drain_latch_refuses_new_work"] = (
        d.status_code == 200 and r.status_code == 503 and r.json().get("code") == "draining"
    )
    # the latch alone never touches queued work or its callback
    out["drain_latch_leaves_queued_pending"] = (
        mid.get("status") == "queued" and sink.path_n.get("/dq", 0) == 0
    )
    # queued work continues under the latch once a worker frees —
    # drain refuses *new* work, it doesn't freeze admitted work
    st = _wait_job(ctx.client, qid)
    out["drain_queued_completes_and_fires"] = (
        st.get("status") == "succeeded"
        and st.get("callback_status") == "delivered"
        and sink.path_n.get("/dq") == 1
    )

    # shutdown: the lifespan's cancel_pending flips queued jobs to
    # 'cancelled' and fires the webhook exactly once — measured through
    # the real lifespan, not a re-implementation
    app = _app(td / "drain-off", max_inflight=1)
    n0 = len(sink.hits)
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.jobs_executor.submit(lambda: time.sleep(2.5))
        time.sleep(0.15)
        r = client.post(_JOB_POST, json={"command": "doctor", "callback_url": sink.url("/dq2")})
        q2 = r.json()["job_id"]
        queued = client.get(f"/harness/jobs/{q2}").json().get("status") == "queued"
    _wait_hits(sink, n0 + 1)
    hits = sink.hits_on("/dq2")
    body = json.loads(hits[-1].body) if hits else {}
    out["drain_shutdown_cancels_queued_once"] = (
        queued and r.status_code == 202 and len(hits) == 1 and body.get("status") == "cancelled"
    )
    return out


def _probe_cancel_race(td: Path, sink: _Sink) -> dict[str, bool]:
    """A cancel landing while the success delivery is mid-flight loses
    cleanly: the terminal record is already immutable, the DELETE 409s,
    and the in-flight delivery completes."""
    out: dict[str, bool] = {}
    ctx = _make_ctx(td / "crace", max_inflight=2)
    sink.stall_s = 1.0
    jid = _submit_job(ctx.client, sink.url("/stall-race"))["job_id"]

    in_flight = False
    end = time.monotonic() + _WAIT_S
    while time.monotonic() < end:
        st = ctx.client.get(f"/harness/jobs/{jid}").json()
        if st.get("status") == "succeeded":
            in_flight = st.get("callback_status") is None
            break
        time.sleep(0.02)
    d = ctx.client.delete(f"/harness/jobs/{jid}")
    st = _wait_job(ctx.client, jid)
    hits = sink.hits_on("/stall-race")
    out["cancel_mid_delivery_409"] = in_flight and d.status_code == 409 and len(hits) == 1
    out["cancel_mid_delivery_completes"] = (
        st.get("status") == "succeeded" and st.get("callback_status") == "delivered"
    )
    sink.stall_s = 1.0
    return out


def _probe_records(ctx: _Ctx, sink: _Sink) -> dict[str, bool]:
    """The delivery verdict surfaces honestly on each record's GET."""
    client = ctx.client
    out: dict[str, bool] = {}

    jid = _submit_job(client, sink.url("/rec"), secret=_SECRET_A)["job_id"]
    st = _wait_job(client, jid)
    out["record_fields_honest_job"] = (
        st.get("callback_url") == sink.url("/rec")
        and st.get("callback_status") == "delivered"
        and st.get("callback_attempts") == 1
        and st.get("callback_error") is None
        and "callback_secret" not in st
    )
    # failed delivery: the verdict names the class, never silent
    jid2 = _submit_job(client, sink.url("/s500"))["job_id"]
    st2 = _wait_job(client, jid2)
    out["record_fields_failed_verdict"] = (
        st2.get("callback_status") == "failed"
        and st2.get("callback_attempts") == WEBHOOK_MAX_ATTEMPTS
        and isinstance(st2.get("callback_error"), str)
        and "500" in str(st2.get("callback_error"))
    )
    # no-callback records never pretend a delivery happened
    jid3 = _submit_job(client)["job_id"]
    st3 = _wait_status(client, jid3)
    out["record_fields_no_callback_honest"] = (
        st3.get("status") == "succeeded"
        and st3.get("callback_url") is None
        and st3.get("callback_status") is None
        and st3.get("callback_attempts") == 0
        and st3.get("callback_error") is None
    )

    b = _batch_create(client, sink.url("/recb"))
    bst = _wait_verdict(client, f"/v1/batches/{b['id']}")
    out["record_fields_batch"] = (
        bst.get("callback_status") == "delivered"
        and bst.get("callback_attempts") == 1
        and bst.get("callback_url") == sink.url("/recb")
    )

    ft = _ft_create(client, sink.url("/recf"))
    fst = _wait_verdict(client, f"/v1/fine_tuning/jobs/{ft['id']}")
    out["record_fields_ft"] = (
        fst.get("callback_status") == "delivered"
        and fst.get("callback_attempts") == 1
        and fst.get("callback_url") == sink.url("/recf")
    )
    return out


def _probe_envelope(ctx: _Ctx, sink: _Sink) -> dict[str, bool]:
    """Every refusal arrives in the path's own error grammar."""
    client = ctx.client
    out: dict[str, bool] = {}

    r404 = client.get("/harness/jobs/does-not-exist")
    out["envelope_404_harness"] = (
        r404.status_code == 404
        and r404.json().get("code") == "not_found"
        and "detail" in r404.json()
    )

    jid = _submit_job(client, sink.url("/env-ok"))["job_id"]
    _wait_job(client, jid)
    d = client.delete(f"/harness/jobs/{jid}")
    out["envelope_409_cancel_terminal"] = (
        d.status_code == 409 and d.json().get("code") == "conflict"
    )

    r = client.post(
        _JOB_POST,
        json={"command": "doctor", "callback_url": _URL_BAD_FILE},
    )
    body = r.json()
    out["envelope_422_harness_validation"] = (
        r.status_code == 422 and body.get("code") == "validation" and "detail" in body
    )

    rv = client.post(
        "/v1/batches",
        json={
            "input_file_id": "f",
            "endpoint": "/v1/chat/completions",
            "callback_url": _URL_BAD_FILE,
        },
    )
    vbody = rv.json()
    out["envelope_422_v1_validation"] = (
        rv.status_code == 422
        and isinstance(vbody.get("error"), dict)
        and vbody["error"].get("code") == "validation"
        and "message" in vbody["error"]
    )

    client.post("/harness/drain")
    rd = client.post(_JOB_POST, json={"command": "doctor"})
    out["envelope_503_draining"] = rd.status_code == 503 and rd.json().get("code") == "draining"
    return out


def webhookdel_audit() -> dict[str, Any]:
    """Run every probe against a real loopback sink; literal bools out."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    # The audit's real HTTP sink is deliberately loopback-only. Production
    # callback delivery remains public-network-only unless explicitly opted in.
    os.environ[_PRIV_ENV] = "1"
    sink = _Sink()
    out: dict[str, Any] = {}
    try:
        out.update(_probe_verdict_table(sink))
        out.update(_probe_backoff(sink))
        out.update(_probe_faults(sink))
        out.update(_probe_signing(sink))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out.update(_probe_timing(root, sink))
            out.update(_probe_ordering(root, sink))
            out.update(_probe_fire_once(_make_ctx(root / "main"), sink))
            out.update(_probe_ledger(root, sink))
            out.update(_probe_drain(root, sink))
            out.update(_probe_cancel_race(root, sink))
            out.update(_probe_records(_make_ctx(root / "rec"), sink))
            out.update(_probe_envelope(_make_ctx(root / "env"), sink))
            out.update(_probe_validation(_make_ctx(root / "val"), sink))

    finally:
        try:
            sink.close()
        finally:
            # Env restore must survive a close() failure — the opt-in is
            # process-wide and must never leak past the battery.
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return out


def webhookdel_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = webhookdel_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True)
    out: dict[str, Any] = {
        "kind": "webhookdel_audit",
        "schema": "webhookdel_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": [
                "webhooks.deliver_signed",
                "webhooks.check_callback_url",
                "webhooks.verify_webhook",
                "api._deliver_callback",
                "harness.jobs.submit/cancel",
                "harness.evals.submit",
                "v1.fine_tuning.jobs",
                "v1.batches",
                "v1.messages.batches",
                "harness.drain",
                "lifespan.shutdown",
            ],
            "not_verified": [
                "real public-network delivery (loopback sink only)",
                "delivery across a true process kill mid-retry",
                "TLS to a certificate-valid sink (private-network gate "
                "makes public https unroutable in-band)",
            ],
            "not_executed": [
                "webhook re-delivery tuning knobs (none exist — the schedule is fixed)",
            ],
        },
        "interpretation": (
            "Webhook delivery contract holds: the verdict table is exact — "
            "2xx delivers on attempt 1, 4xx (including 429 with "
            "Retry-After) is definitive in a single attempt, 5xx and 3xx "
            "retry to the WEBHOOK_MAX_ATTEMPTS ceiling and redirects are "
            "never followed; the backoff schedule doubles exponentially "
            "(0.5s then 1.0s measured), re-mints the signature pair, "
            "re-resolves DNS, and opens a fresh connection per attempt; "
            "DNS failure, connection refused, read timeout, and TLS "
            "mismatch all classify loudly and boundedly; private targets "
            "refuse (literal at validation, resolved at delivery), and a "
            "dead resolved address fails over within the attempt; "
            "delivery is synchronous in the caller — submit returns "
            "first, the terminal record is GETable before the verdict "
            "lands, a queued-cancel DELETE returns only after the "
            "cancelled delivery lands, and a failing delivery holds its "
            "inflight slot for the whole retry schedule; a single-worker "
            "executor delivers FIFO while a pool dispatches concurrently "
            "and cancel-path deliveries run on request threads; every "
            "terminal transition fires exactly once per surface and "
            "second transitions never re-fire; under --state-dir the "
            "terminal mark journals the verdict atomically, a restart "
            "restores it without re-firing, a recovered queued job fails "
            "closed as 'failed' with zero attempts (secrets never touch "
            "disk), and idempotency mappings survive; the drain latch "
            "refuses new work without freezing admitted work while "
            "shutdown cancels queued jobs and fires 'cancelled' once; "
            "callback verdicts surface honestly on every record's GET; "
            "every refusal is enveloped in the path's own grammar."
            if ok
            else f"WEBHOOKDEL AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(webhookdel_audit_bench(), indent=2, sort_keys=True))
