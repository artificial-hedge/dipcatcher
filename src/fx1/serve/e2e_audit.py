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
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=15)
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
