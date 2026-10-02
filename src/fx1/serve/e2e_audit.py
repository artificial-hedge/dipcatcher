"""End-to-end audit — the whole harness over real sockets, both directions.

The parity audit proves the SDK, the ASGI app, and ``HarnessClient`` agree
inside one process. This audit closes the last unmeasured layer: the
transport itself. It stands up TWO real loopback servers —

1. a stub OpenAI-compatible engine (stdlib ``ThreadingHTTPServer``)
   answering ``/v1/chat/completions`` in both non-streaming and SSE
   ``stream: true`` modes — the stand-in for a BYOK endpoint, and
2. ``uvicorn`` serving ``create_app()`` with the production middleware
   and the real ``get_backend`` resolver (BYOK env pointed at the stub) —

then drives the entire lifecycle through ``HarnessClient``'s default
urllib transport: health → registry → run → gated completion → SSE stream
→ batch → receipt verify, plus the failure classes (wrong key, missing
key, oversized body, gate refusal echoing through the wire, unknown
command/backend). Every hop crosses a real TCP socket: a byte that
survives this lane survives the network.

Sealed ``e2e_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    import uvicorn

__all__ = ["e2e_audit", "e2e_audit_bench"]

_E2E_ENV = (
    "FX1_API_KEY",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "MOONSHOT_API_KEY",
)
_API_KEY = "e2e-probe-key"  # a probe string, not a credential


class _StubChat(BaseHTTPRequestHandler):
    """Deterministic OpenAI-compatible engine: echoes the last user message."""

    protocol_version = "HTTP/1.1"
    server_version = "stub-engine"

    def do_GET(self) -> None:  # noqa: N802 — stdlib hook name
        if self.path != "/v1/models":
            self.send_error(404)
            return
        payload = json.dumps(
            {"object": "list", "data": [{"id": "stub-v0", "object": "model"}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802 — stdlib hook name
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            content = body["messages"][-1]["content"]
        except Exception:  # noqa: BLE001 — hostile input fails closed
            self.send_error(400)
            return
        if body.get("stream"):
            frames = (
                b'data: {"choices":[{"delta":{"content":"stub:"}}]}\n\n'
                + f'data: {{"choices":[{{"delta":{{"content":{json.dumps(content)}}}}}]}}\n\n'.encode()
                + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(frames)))
            self.end_headers()
            self.wfile.write(frames)
            return
        payload = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"stub:{content}",
                        }
                    }
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # keep test output quiet
        return None


def _free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _serve_uvicorn(app: Any) -> tuple[uvicorn.Server, threading.Thread, int]:
    """uvicorn on a real loopback socket; returns (server, thread, port)."""
    import uvicorn

    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="critical", server_header=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    return server, thread, port


def _raises(fn: Any) -> str:
    """Exception class name; "" when no raise."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — probe captures the class
        return type(exc).__name__
    return ""


def _raw_headers(url: str, key: str | None) -> tuple[int, dict[str, str]]:
    req = urllib.request.Request(url)
    if key is not None:
        req.add_header("X-API-Key", key)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — loopback probe URL built by this module  # nosec B310
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in (exc.headers or {}).items()}


def e2e_audit() -> dict[str, bool]:
    out: dict[str, bool] = {}
    saved = {k: os.environ.get(k) for k in _E2E_ENV}
    stub = ThreadingHTTPServer(("127.0.0.1", 0), _StubChat)
    stub_thread = threading.Thread(target=stub.serve_forever, daemon=True)
    stub_thread.start()
    server: uvicorn.Server | None = None
    server_thread: threading.Thread | None = None
    server2: uvicorn.Server | None = None
    server2_thread: threading.Thread | None = None
    try:
        stub_port = int(stub.server_address[1])
        os.environ.update(
            {
                "FX1_API_KEY": _API_KEY,
                "FX1_BYOK_BASE_URL": f"http://127.0.0.1:{stub_port}/v1",
                "FX1_BYOK_API_KEY": "stub-engine-key",
                "FX1_BYOK_MODEL": "stub-v0",
            }
        )
        os.environ.pop("MOONSHOT_API_KEY", None)

        from fx1.harness import Harness
        from fx1.serve import api as api_mod
        from fx1.serve.client import (
            HarnessAuthError,
            HarnessClient,
            HarnessTransportError,
        )

        def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
            return 0, f"ran:{' '.join(argv)}", ""

        app = api_mod.create_app(harness=Harness(runner=fake_runner))
        server, server_thread, port = _serve_uvicorn(app)
        out["e2e_server_boots"] = bool(server.started and thread_is_alive(server_thread))

        base = f"http://127.0.0.1:{port}"
        remote = HarnessClient(base, api_key=_API_KEY, timeout_s=15.0)
        msg = [{"role": "user", "content": "ping"}]
        receipt = "a" * 64

        # -- lifecycle over the wire --------------------------------------
        health = remote.health()
        out["e2e_health"] = health.backends.get("byok") is True
        cmds = remote.commands()
        out["e2e_commands"] = len(cmds) > 0
        name = sorted(cmds)[0]
        run = remote.run(name)
        out["e2e_run_executes"] = run.exit_code == 0 and run.ok
        done = remote.complete(msg, backend="byok", receipt_hashes=[receipt])
        out["e2e_complete_gated"] = (
            done.content.startswith("stub:ping")
            and done.receipt_hashes == (receipt,)
            and done.model == "stub-v0"
            and "Evidence:" in done.content  # footer appended
        )
        chunks = remote.stream_complete(msg, backend="byok", receipt_hashes=[receipt])
        out["e2e_stream_sse"] = len(chunks) >= 3 and "".join(chunks) == done.content
        batch = remote.complete_many(
            [msg, [{"role": "user", "content": "pong"}]],
            backend="byok",
            receipt_hashes=[receipt],
            max_workers=2,
        )
        out["e2e_batch"] = (
            len(batch) == 2
            and batch[0].content == done.content
            and batch[1].content.startswith("stub:pong")
        )
        receipt_path = Path(__file__).resolve().parents[3] / "receipts" / "fx1_parity_audit.json"
        if receipt_path.exists():
            verdict = remote.verify_receipt(json.loads(receipt_path.read_text()))
            out["e2e_verify_sealed"] = verdict.valid is True
            tampered = json.loads(receipt_path.read_text())
            tampered["receipt_sha256"] = "0" * 64
            out["e2e_verify_tamper"] = remote.verify_receipt(tampered).valid is False
        else:  # receipt absent (isolated checkout) — record honestly
            out["e2e_verify_sealed"] = False
            out["e2e_verify_tamper"] = False

        # -- fault classes over the wire ----------------------------------
        no_auth = HarnessClient(base, timeout_s=15.0)
        out["e2e_missing_key_401"] = (
            _raises(lambda: no_auth.complete(msg, backend="byok")) == HarnessAuthError.__name__
        )
        wrong = HarnessClient(base, api_key="wrong", timeout_s=15.0)
        out["e2e_wrong_key_401"] = (
            _raises(lambda: wrong.complete(msg, backend="byok")) == HarnessAuthError.__name__
        )
        out["e2e_gate_refusal"] = (
            _raises(
                lambda: remote.complete(
                    [{"role": "user", "content": "total Sharpe 9.9 NAV"}],
                    backend="byok",
                )
            )
            == "Fx1HonestyError"
        )
        out["e2e_404_maps"] = _raises(lambda: remote.run("no-such-command")) == "KeyError"
        out["e2e_422_maps"] = _raises(lambda: remote.complete(msg, backend="bogus")) == "ValueError"
        big = "x" * (2 * 1024 * 1024)
        out["e2e_body_cap_413"] = (
            _raises(lambda: remote.complete([{"role": "user", "content": big}], backend="byok"))
            == HarnessTransportError.__name__
        )
        status, headers = _raw_headers(f"{base}/health", _API_KEY)
        out["e2e_security_headers"] = (
            status == 200
            and "server" not in headers
            and headers.get("x-content-type-options") == "nosniff"
        )

        # the CLI front door over the real wire — `fx1 harness --remote`
        from typer.testing import CliRunner

        from fx1.cli import app as cli_app

        cr = CliRunner().invoke(
            cli_app,
            ["harness", "health", "--remote", base, "--api-key", _API_KEY],
        )
        out["e2e_cli_remote_health"] = cr.exit_code == 0 and '"byok": true' in cr.output
        cr2 = CliRunner().invoke(
            cli_app, ["harness", "list", "--remote", base, "--api-key", _API_KEY]
        )
        out["e2e_cli_remote_list"] = cr2.exit_code == 0 and sorted(cmds)[0] in cr2.output

        # -- resilience over the real wire: cap 503 -> Retry-After -> recover
        capped_app = api_mod.create_app(harness=Harness(runner=fake_runner), max_inflight=1)
        server2, server2_thread, port2 = _serve_uvicorn(capped_app)
        base2 = f"http://127.0.0.1:{port2}"
        slots = capped_app.state.inflight_slots
        held = slots.acquire(blocking=False)
        out["e2e_cap_slot_held"] = held is True
        released = {"done": False}

        def _release_sleep(_s: float) -> None:
            if not released["done"]:
                released["done"] = True
                slots.release()

        resilient = HarnessClient(
            base2,
            api_key=_API_KEY,
            timeout_s=15.0,
            max_retries=3,
            retry_backoff_s=1e-9,
            retry_writes=True,
            sleep=_release_sleep,
        )
        try:
            sat = resilient.complete(msg, backend="byok")
            out["e2e_retry_recovers_under_cap"] = sat.content.startswith("stub:ping")
        except Exception:  # noqa: BLE001 — probe records, never crashes
            out["e2e_retry_recovers_under_cap"] = False
            if not released["done"]:
                released["done"] = True
                slots.release()

        # ops surface over the wire — counters reflect this audit's calls
        m = remote.metrics()
        out["e2e_metrics_counts"] = (
            m.requests_total >= 10
            and m.errors_total >= 2
            and m.by_status.get("200", 0) >= 5
            and "401" in m.by_status
            and m.inflight == 0
            and m.inflight_watermark >= 1
            and m.max_inflight == 16
        )
        m2 = resilient.metrics()
        out["e2e_metrics_capped_server"] = m2.max_inflight == 1 and m2.requests_total >= 1

        # idempotency over the real wire: same key -> the command runs once
        # and replays are served from the store even while draining.
        k = "e2e-idem-key"
        first = resilient.run("doctor", idempotency_key=k)
        d = resilient.drain()
        out["e2e_drain_response"] = d["draining"] is True and isinstance(d["inflight"], int)
        try:
            resilient.run("doctor")  # no key -> fresh work -> refused
            out["e2e_drain_blocks_new_work"] = False
        except Exception as exc:  # noqa: BLE001 — probe records the class
            out["e2e_drain_blocks_new_work"] = type(
                exc
            ).__name__ == "BackendNotConfiguredError" and "draining" in str(exc)
        # keyed retry of an already-executed submission is served, not refused
        second = resilient.run("doctor", idempotency_key=k)
        out["e2e_idem_replay_under_drain"] = (
            second.command == first.command and second.exit_code == first.exit_code
        )
        out["e2e_drain_metrics_flag"] = resilient.metrics().draining is True
        out["e2e_drain_health_still_up"] = resilient.health().status == "ok"

        # async jobs over the real wire: submit, poll to terminal, then drain
        # and verify a keyed resubmit replays instead of refusing.
        server3, server3_thread, port3 = _serve_uvicorn(
            api_mod.create_app(harness=Harness(runner=fake_runner))
        )
        try:
            jremote = HarnessClient(f"http://127.0.0.1:{port3}", api_key=_API_KEY, timeout_s=15.0)
            j1 = jremote.submit_run("doctor", idempotency_key="e2e-job-key")
            res = jremote.wait_run(j1, poll_s=0.05, timeout_s=15.0)
            out["e2e_job_roundtrip"] = res.command == "doctor" and res.ok
            st = jremote.job_status(j1)
            out["e2e_job_status_fields"] = (
                st["status"] == "succeeded" and st["result"]["command"] == "doctor"
            )
            out["e2e_job_unknown_404"] = _raises(lambda: jremote.job_status("nope")) == "KeyError"
            page = jremote.list_jobs()
            out["e2e_job_list"] = page["total"] >= 1 and any(
                j["job_id"] == j1 for j in page["jobs"]
            )
            out["e2e_job_list_filter"] = (
                jremote.list_jobs(status="succeeded")["jobs"][0]["status"] == "succeeded"
            )
            out["e2e_job_cancel_terminal_409"] = (
                _raises(lambda: jremote.cancel_job(j1)) == "HarnessTransportError"
            )
            rdy = jremote.ready()
            out["e2e_ready_200"] = rdy["ready"] is True and isinstance(rdy["inflight"], int)
            ver = jremote.server_version()
            out["e2e_version_route"] = ver["api_version"] == "1" and bool(ver["fx1_version"])
            out["e2e_api_version_header"] = jremote.last_api_version == "1"
            jremote.drain()
            out["e2e_ready_under_drain"] = (
                _raises(lambda: jremote.ready()) == "BackendNotConfiguredError"
            )
            try:
                jremote.submit_run("doctor")
                out["e2e_error_code_draining"] = False
            except Exception as exc:
                out["e2e_error_code_draining"] = getattr(exc, "code", None) == "draining"
            out["e2e_job_replay_under_drain"] = (
                jremote.submit_run("doctor", idempotency_key="e2e-job-key") == j1
            )
            out["e2e_job_submit_under_drain"] = (
                _raises(lambda: jremote.submit_run("doctor")) == "BackendNotConfiguredError"
            )
        finally:
            server3.should_exit = True
            server3_thread.join(timeout=15)

        # blocking drain over the real wire: a job holds a slot, drain
        # wait_s=0 reports not drained, wait_s=10 blocks until it empties.
        def _slow_runner(a: list[str], t: float) -> tuple[int, str, str]:
            time.sleep(0.5)
            return (0, "ran:" + " ".join(a), "")

        server4, server4_thread, port4 = _serve_uvicorn(
            api_mod.create_app(harness=Harness(runner=_slow_runner))
        )
        try:
            wremote = HarnessClient(f"http://127.0.0.1:{port4}", api_key=_API_KEY, timeout_s=15.0)
            wremote.submit_run("doctor")
            d0 = wremote.drain(wait_s=0.01)
            out["e2e_drain_wait_timeout"] = d0["drained"] is False
            d1 = wremote.drain(wait_s=10.0)
            out["e2e_drain_wait_blocks"] = d1["drained"] is True and d1["inflight"] == 0
        finally:
            server4.should_exit = True
            server4_thread.join(timeout=15)

        # cooperative cancel over the real wire: occupy both executor
        # workers without slots (slots == workers) so jobs stay queued.
        server5_app = api_mod.create_app(harness=Harness(runner=_slow_runner), max_inflight=2)
        server5, server5_thread, port5 = _serve_uvicorn(server5_app)
        try:
            cremote = HarnessClient(f"http://127.0.0.1:{port5}", api_key=_API_KEY, timeout_s=15.0)
            server5_app.state.jobs_executor.submit(lambda: time.sleep(4.0))
            server5_app.state.jobs_executor.submit(lambda: time.sleep(4.0))
            c1 = cremote.submit_run("doctor")
            c2 = cremote.submit_run("doctor")
            out["e2e_job_queued_while_workers_busy"] = (
                cremote.job_status(c2)["status"] == "queued"
                and cremote.job_status(c1)["status"] == "queued"
            )
            out["e2e_job_cancel_queued"] = cremote.cancel_job(c2)["status"] == "cancelled"
            out["e2e_job_wait_cancelled_raises"] = (
                _raises(lambda: cremote.wait_run(c2, poll_s=0.05, timeout_s=15.0))
                == "HarnessJobError"
            )
            # the surviving queued job runs once a worker frees
            dl = time.monotonic() + 15.0
            st = cremote.job_status(c1)
            while st["status"] == "queued" and time.monotonic() < dl:
                time.sleep(0.1)
                st = cremote.job_status(c1)
            out["e2e_job_survivor_runs"] = st["status"] in ("running", "succeeded")
        finally:
            server5.should_exit = True
            server5_thread.join(timeout=15)
            server4_thread.join(timeout=15)

        # rate limiting over the real wire: the client retries a 429 with
        # Retry-After transparently, and the denial is metered.
        server6_app = api_mod.create_app(harness=Harness(runner=fake_runner), rate_limit_rps=3.0)
        server6, server6_thread, port6 = _serve_uvicorn(server6_app)
        try:
            rl = HarnessClient(
                f"http://127.0.0.1:{port6}",
                api_key=_API_KEY,
                timeout_s=15.0,
                max_retries=4,
                retry_backoff_s=0.05,
            )
            results = [_raises(lambda: rl.health()) or "ok" for _ in range(6)]
            m = rl.metrics()
            out["e2e_rate_limit_retries_succeed"] = (
                results == ["ok"] * 6 and m.rate_limited_total >= 1
            )
            # Prometheus scrape over the real socket — content-negotiated
            # exposition, parseable lines, live counters.
            prom_text = rl.metrics_text()
            prom_series = {
                ln.split()[0].split("{")[0]
                for ln in prom_text.splitlines()
                if ln and not ln.startswith("#")
            }
            out["e2e_metrics_prometheus"] = (
                prom_text.startswith("# HELP")
                and "# TYPE fx1_requests_total counter" in prom_text
                and {"fx1_requests_total", "fx1_inflight", "fx1_rate_limited_total"} <= prom_series
                and any(
                    ln.startswith("fx1_rate_limited_total ") and float(ln.split()[-1]) >= 1
                    for ln in prom_text.splitlines()
                )
            )
        finally:
            server6.should_exit = True
            server6_thread.join(timeout=15)

        # job-completion webhook over the real wire: a submitted job POSTs
        # its terminal record to the caller's callback_url.
        hook_hits: list[dict[str, Any]] = []
        hook_raw: list[bytes] = []
        hook_hdrs: list[dict[str, str]] = []

        class _JobHook(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — stdlib hook name
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                hook_raw.append(raw)
                hook_hdrs.append(dict(self.headers.items()))
                hook_hits.append(json.loads(raw))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        hook_srv = ThreadingHTTPServer(("127.0.0.1", 0), _JobHook)
        hook_thread = threading.Thread(target=hook_srv.serve_forever, daemon=True)
        hook_thread.start()
        hook_url = f"http://127.0.0.1:{hook_srv.server_address[1]}/hook"
        server7_app = api_mod.create_app(harness=Harness(runner=fake_runner))
        server7, server7_thread, port7 = _serve_uvicorn(server7_app)
        try:
            hc = HarnessClient(f"http://127.0.0.1:{port7}", api_key=_API_KEY, timeout_s=15.0)
            jid = hc.submit_run("doctor", callback_url=hook_url, callback_secret="whsec-e2e")
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and not hook_hits:
                time.sleep(0.05)
            st = hc.job_status(jid)
            out["e2e_job_webhook_delivered"] = (
                len(hook_hits) == 1
                and hook_hits[0]["job_id"] == jid
                and hook_hits[0]["status"] == "succeeded"
                and st.get("callback_status") == "delivered"
            )
            from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415 — inside the served block

            out["e2e_job_webhook_signed"] = verify_webhook(
                "whsec-e2e",
                hook_hdrs[0].get("X-Fx1-Webhook-Timestamp"),
                hook_hdrs[0].get("X-Fx1-Webhook-Signature"),
                hook_raw[0],
            )
            frames = hc.stream_job(jid)
            out["e2e_job_events_terminal"] = (
                len(frames) >= 1
                and frames[-1]["job_id"] == jid
                and frames[-1]["status"] == "succeeded"
            )
            out["e2e_wait_run_stream"] = hc.wait_run_stream(jid).command == "doctor"
        finally:
            server7.should_exit = True
            server7_thread.join(timeout=15)
            hook_srv.shutdown()
            hook_srv.server_close()
            hook_thread.join(timeout=5)
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=15)
        if server2 is not None:
            server2.should_exit = True
        if server2_thread is not None:
            server2_thread.join(timeout=15)
        stub.shutdown()
        stub.server_close()
        stub_thread.join(timeout=5)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def thread_is_alive(thread: threading.Thread | None) -> bool:
    return bool(thread is not None and thread.is_alive())


def e2e_audit_bench() -> dict[str, Any]:
    r = e2e_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "e2e_audit",
        "schema": "e2e_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "end-to-end verified over real loopback sockets: HarnessClient "
            "drove health/registry/run/gated-complete/SSE-stream/batch/"
            "receipt-verify against uvicorn + the production middleware, "
            "with BYOK resolving to a real stub OpenAI engine — auth "
            "faults, gate refusals, size caps, and error classes all "
            "survive the wire. No Server header leaks the stack."
            if ok
            else f"E2E AUDIT DEFECT: {r}"
        ),
    }
    body = dict(out)
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return out
