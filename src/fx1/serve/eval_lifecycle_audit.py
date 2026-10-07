"""eval_lifecycle_audit — probe battery over the eval surface's state machine.

The claim under test: an eval submission is one durable state machine —
every entry leg (HTTP ``POST /harness/evals``, ``Fx1Harness.run_eval``,
``HarnessClient.submit_eval``, and ``fx1 harness eval`` — in-process and
``--remote``) produces the same record shape and walks the same
``queued → running → succeeded|failed|cancelled`` path; idempotency keys,
decode pins, backend attribution, and sealed receipts survive process
restarts; and every refusal lands as a fail-closed wire code, never a
half-run.

Coverage map:

- *Four legs, one lifecycle* — the record a raw POST returns is the record
  the SDK twin returns is the record ``HarnessClient.wait_eval`` returns is
  the record the CLI prints: same field set, same terminal state, same
  ``temperature=0`` sampling pin. The remote CLI leg runs against a real
  uvicorn socket; the in-process leg drives a real env-resolved BYOK
  backend against a stub OpenAI engine.
- *Idempotency* — same key + same body replays the record without a second
  model call; same key + different body 409s; distinct keys run
  independently; ``/v1/evals`` scopes the key per spec; a replay survives
  drain mode; LRU eviction drops the mapping with the record.
- *Transitions* — only ``queued→running→terminal``; cancel flips queued→
  cancelled and the worker's atomic claim can never resurrect it (the
  cancel wins the race under the store lock); cancel on running or
  terminal 409s; ``wait_eval`` raises ``HarnessJobError`` on failed/
  cancelled and ``HarnessTransportError`` on timeout without killing the
  eval; drain refuses new work 503 (a keyed replay still answers); a
  saturated slot refuses ``over_capacity`` with ``Retry-After``.
- *Store* — ``EvalStore.start`` is the worker's atomic queued→running
  claim; ``mark`` journals every hop; replay restores terminal records
  verbatim and turns interrupted work into restart-failed — the idem key
  still resolves to the lost record so a retried submission never
  duplicates work.
- *Receipt* — terminal records export sealed ``fx1_eval_record.v1`` docs
  that verify under the receipt chain; a flipped byte fails closed; the
  same record seals identical bytes; a non-terminal receipt request 409s
  while a cancelled (terminal) record still exports; the sealed
  ``record`` is exactly the served record.
- *Backend attribution* — the resolved link lands on ``record.backend``;
  a fallback chain orders its ``attempts``; model calls meter under
  ``eval:{suite}:{backend}``, judge calls under
  ``eval:{suite}:judge:{backend}``; a judge on a non-judge suite 422s at
  submit; a per-request BYOK override reaches the resolver and never
  lands on the record.
- *Concurrency* — N parallel evals produce N distinct records with the
  submitted suite/seed on each — no cross-contamination — and the list
  surface's count agrees.
- *Durability* — ``--state-dir`` journals let a fresh app process answer
  GETs for terminal records, mark interrupted evals restart-failed, and
  replay idem keys to lost records; the SDK twin binds the same journals.
- */v1/evals* — spec lifecycle (create → get → update → list → delete →
  tombstone), item_schema validated at create, the datasource frozen
  against updates, runs bind ``eval_spec``/``eval_model``, ``model``
  resolves link names / ``fx1`` / ``ft:`` and fails closed on anything
  else, scoped idempotency replays per spec, output_items page per-task
  verdicts with cursors that round-trip and fail closed when foreign or
  malformed.
- *Boundary* — unknown suite/backend, malformed bodies, and every knob
  violation answer their mapped wire code at submit (nothing queues);
  auth scopes gate the surface (a read key cannot submit or cancel; a
  write key cannot mint keys).
- *Callbacks* — a terminal webhook POSTs the full record shape, the
  secret signs but is never echoed, a dead endpoint records
  ``callback_status='failed'`` without touching the record's status, and
  a queued cancel fires it too.
- *Diff* — same-suite/same-seed pairs report a comparable verdict;
  cross-seed pairs serve ``comparable: false`` / ``verdict: 'unknown'``
  rather than a fabricated verdict; a diff against a non-terminal record
  fails closed.

Six defects and one evidence gap were found and fixed while building
this battery: the worker's queued→running transition in ``_submit_eval``
was a non-atomic read-then-write outside the store lock, so a cancel
landing between the check and the write let a cancelled eval run to
``succeeded`` — the record resurrected out of a terminal state (fixed
via ``EvalStore.start``'s atomic claim); the store ``put`` happened
*after* the executor hand-off, so a fast-scheduled worker could dequeue
a record the store didn't know yet and drop it — permanently ``queued``
(fixed by storing before ``jobs_executor.submit``, with a tombstone
delete if the submit itself fails); ``output_items`` accepted a
malformed ``after`` cursor by silently restarting at index 0 and
accepted another run's item id by paging this run's rows at its index
(now 400 ``invalid_cursor`` unless the cursor is a well-formed item id
of this run); the spec list treated an unknown ``after`` as
end-of-list, hiding client paging bugs (now 400 ``invalid_cursor``,
matching the run list); a run's ``eval_spec``/``eval_model`` binding was
journaled only after ``_submit_eval`` returned, so a fast eval could
fire its terminal callback carrying an unbound record (now bound at
record construction, before the executor sees it); and the run list
rejected the ``evalrun_``-prefixed id shape its own wire hands out as a
cursor (now accepted via the same id it serves). The evidence gap: a
dead fallback chain dropped its ``attempts`` — the failed record said
503 but kept nothing about which links were tried (the worker now seals
the chain attempts on both failure paths).

Sealed ``eval_lifecycle_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, cast

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi.testclient import TestClient

    from fx1.serve.backends import SamplingParams

__all__ = ["eval_lifecycle_audit", "eval_lifecycle_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "k3y-material"
_SWEPT_ENVS = (
    _API_KEY_ENV,
    "MOONSHOT_API_KEY",
    "FX1_API_STATE_DIR",
    "FX1_SDK_STATE_DIR",
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
# Deliberately-insecure literal: BYOK overrides accept http:// for local
# stacks — this URL is never dialed (the injected resolver returns stubs).
_BYOK_BASE_URL = "http://127.0.0.1:9/v1"  # NOSONAR(S5332) — never dialed
_BYOK_CREDS = {
    "base_url": _BYOK_BASE_URL,
    "api_key": "bk-audit-secret",
    "model": "fx1-ft:pilot-9",
}
# Nonexistent dir for the wrong-link checkpoint probe (no hardcoded /tmp).
_MISSING_DIR = os.path.join(tempfile.gettempdir(), "fx1-ela-no-such-ckpt")

_EVALS_PATH = "/harness/evals"
_TERMINAL = {"succeeded", "failed", "cancelled"}


class _EvalBackend:
    """Suite-driving stub — one counted ``complete`` call per ModelFn hit.
    ``gate`` blocks inside the call (running-state probes) until set."""

    def __init__(
        self,
        reply: str = "0.5",
        usage: dict[str, int] | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self._reply = reply
        self._usage = usage
        self._gate = gate
        self.calls = 0
        self.last_usage: dict[str, int] | None = None
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        with self._lock:
            self.calls += 1
        if self._gate is not None:
            self._gate.wait(30)
        self.last_usage = dict(self._usage) if self._usage else None
        return self._reply

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _FailBackend:
    """Never reached the provider: raises before any model work."""

    def complete(
        self,
        messages: list[dict[str, str]],  # NOSONAR(S1172) — protocol signature
        *,
        sampling: SamplingParams | None = None,  # NOSONAR(S1172)
    ) -> str:
        from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

        raise BackendNotConfiguredError("no credentials configured")

    def close(self) -> None:
        """No resources to release — the stub holds nothing."""


class _CliChat(BaseHTTPRequestHandler):
    """Deterministic OpenAI-compatible engine for the CLI legs — echoes the
    last user message, counts completions."""

    calls = 0
    _lock = threading.Lock()
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 — stdlib hook name
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 — stdlib hook name
        with _CliChat._lock:  # noqa: SLF001 — class-level probe counter
            _CliChat.calls += 1
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
        payload = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": f"stub:{content}"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        pass


class _Hit:
    """One webhook delivery landing on ``_Sink``."""

    __slots__ = ("body", "headers", "path")

    def __init__(self, path: str, headers: dict[str, str], body: bytes) -> None:
        self.path = path
        self.headers = headers
        self.body = body


class _Sink:
    """A real loopback webhook sink; path selects the verdict (``/fail``
    500s, everything else 200s)."""

    def __init__(self) -> None:
        self.hits: list[_Hit] = []
        self._lock = threading.Lock()
        sink = self

        class _H(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — http.server name
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                with sink._lock:  # noqa: SLF001 — same-module closure state
                    sink.hits.append(_Hit(self.path, dict(self.headers.items()), raw))
                code = 500 if self.path == "/fail" else 200
                self.send_response(code)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                pass

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{int(self._srv.server_address[1])}{path}"

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)


def _client(
    backend_map: dict[str, Any],
    api_key: str | None = None,
    *,
    resolver: Any | None = None,
    **app_kw: Any,
) -> tuple[TestClient, ModuleType]:
    """(TestClient, api_module) — isolated env per construction; backends
    resolve from ``backend_map[name]`` zero-arg factories, or ``resolver``
    verbatim when given (capturing-resolver probes)."""
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    saved = {k: os.environ.get(k) for k in _SWEPT_ENVS}
    try:
        for k in _SWEPT_ENVS:
            os.environ.pop(k, None)
        if api_key is not None:
            os.environ[_API_KEY_ENV] = api_key
        resolve = resolver or (lambda name, *a, **k: backend_map[name]())
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            backend_resolver=resolve,
            **app_kw,
        )
        return TestClient(app, raise_server_exceptions=False), api_mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _jobs_exec(client: TestClient) -> ThreadPoolExecutor:
    """The app's bounded eval executor — `TestClient.app` is typed as the
    raw ASGI callable, so cast back to the FastAPI app the client wraps."""
    from fastapi import FastAPI  # noqa: PLC0415

    app = cast(FastAPI, client.app)
    return cast(ThreadPoolExecutor, app.state.jobs_executor)


def _mint(client: TestClient, root_h: dict[str, str], **policy: Any) -> tuple[str, str]:
    """Mint a managed key → (raw, key_id)."""
    r = client.post("/harness/keys", json=policy, headers=root_h)
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), str(body["id"])


def _submit(
    client: TestClient,
    headers: dict[str, str] | None = None,
    *,
    key: str | None = None,
    **body: Any,
) -> Any:
    hdrs = dict(headers or {})
    if key is not None:
        hdrs["Idempotency-Key"] = key
    return client.post(
        _EVALS_PATH,
        json={"suite": "tooluse", "backend": "byok", "seed": 0, **body},
        headers=hdrs,
    )


def _record(client: TestClient, eval_id: str, headers: dict[str, str] | None = None) -> Any:
    return client.get(f"{_EVALS_PATH}/{eval_id}", headers=headers or {})


def _wait(
    client: TestClient,
    eval_id: str,
    headers: dict[str, str] | None = None,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Poll GET until terminal; assert it lands inside the window."""
    deadline = time.monotonic() + timeout_s
    while True:
        r = _record(client, eval_id, headers)
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in _TERMINAL:
            return dict(body)
        assert time.monotonic() < deadline, f"eval {eval_id} never went terminal"
        time.sleep(0.02)


def _wait_running(client: TestClient, eval_id: str, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while _record(client, eval_id).json()["status"] != "running":
        assert time.monotonic() < deadline, f"eval {eval_id} never ran"
        time.sleep(0.01)


def _tc_transport(client: TestClient) -> Any:
    """Adapt HarnessClient's transport contract to a TestClient."""

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Any, bytes]:
        p = urllib.parse.urlparse(url)
        path = p.path + (f"?{p.query}" if p.query else "")
        if method == "GET":
            resp = client.get(path, headers=headers)
        elif method == "DELETE":
            resp = client.delete(path, headers=headers)
        elif isinstance(payload, bytes):
            resp = client.post(path, content=payload, headers=headers)
        else:
            resp = client.post(path, json=payload, headers=headers)
        return resp.status_code, dict(resp.headers), resp.content

    return send


def _sdk(backend_map: dict[str, Any], **kw: Any) -> Any:
    """Fx1Harness over the same stub resolver as the wire twin."""
    from fx1.harness import Harness
    from fx1.sdk import Fx1Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    return Fx1Harness(
        harness=Harness(runner=fake_runner),
        backend_resolver=lambda name, *a, **k: backend_map[name](),
        **kw,
    )


def _serve_uvicorn(app: Any) -> tuple[Any, threading.Thread, int]:
    """uvicorn on a real loopback socket; returns (server, thread, port)."""
    import socket  # noqa: PLC0415

    import uvicorn

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = int(s.getsockname()[1])
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


def _cli(argv: list[str], env: dict[str, str] | None = None) -> Any:
    """Invoke the fx1 CLI under a swept env plus ``env`` overrides."""
    from typer.testing import CliRunner

    from fx1.cli import app as cli_app

    extra = env or {}
    saved = {k: os.environ.get(k) for k in {*_SWEPT_ENVS, *extra}}
    try:
        for k in _SWEPT_ENVS:
            os.environ.pop(k, None)
        os.environ.update(extra)
        return CliRunner().invoke(cli_app, argv)
    finally:
        for k in extra:
            if k not in _SWEPT_ENVS:
                os.environ.pop(k, None)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _json_docs(stdout: str) -> list[dict[str, Any]]:
    """Parse the back-to-back indented JSON docs a CLI leg prints."""
    dec = json.JSONDecoder()
    docs: list[dict[str, Any]] = []
    rest = stdout.strip()
    while rest:
        obj, end = dec.raw_decode(rest)
        docs.append(obj)
        rest = rest[end:].strip()
    return docs


def _legs_probes() -> dict[str, Any]:
    """Four legs, one record shape — HTTP, client, SDK, CLI."""
    out: dict[str, Any] = {}
    backend = _EvalBackend()
    client, _api = _client({"byok": lambda: backend, "local_fx1": lambda: backend})

    # --- HTTP leg -------------------------------------------------------------
    sub = _submit(client)
    out["http_submit_202_queued_shape"] = (
        sub.status_code == 202
        and sub.headers.get("location") == f"{_EVALS_PATH}/{sub.json()['eval_id']}"
        and sub.json()["status"] in ("queued", "running", "succeeded")
        and sub.json()["replayed"] is False
    )
    eval_id = sub.json()["eval_id"]
    wire_rec = _wait(client, eval_id)
    out["http_terminal_succeeded"] = wire_rec["status"] == "succeeded" and isinstance(
        wire_rec.get("report"), dict
    )

    # --- HarnessClient leg ----------------------------------------------------
    from fx1.serve.client import HarnessClient

    hc = HarnessClient("http://testserver", transport=_tc_transport(client))
    sub2 = hc.submit_eval("tooluse", backend="byok", seed=1)
    cli_rec = hc.wait_eval(sub2["eval_id"], timeout_s=15.0)
    out["client_leg_same_shape"] = (
        set(cli_rec) == set(wire_rec)
        and cli_rec["status"] == "succeeded"
        and cli_rec["eval_id"] == sub2["eval_id"]
    )

    # --- SDK leg --------------------------------------------------------------
    sdk = _sdk({"byok": lambda: _EvalBackend(), "local_fx1": lambda: _EvalBackend()})
    sdk_rec = sdk.run_eval("tooluse", backend="byok", seed=2)
    sdk_dump = sdk_rec.model_dump(mode="json")
    out["sdk_leg_same_shape"] = (
        set(sdk_dump) == set(wire_rec)
        and sdk_rec.status == "succeeded"
        and sdk.eval_record(sdk_rec.eval_id).model_dump(mode="json") == sdk_dump
    )

    # --- CLI in-process leg: env-resolved BYOK against a real stub engine ------
    chat = ThreadingHTTPServer(("127.0.0.1", 0), _CliChat)
    threading.Thread(target=chat.serve_forever, daemon=True).start()
    port = int(chat.server_address[1])
    try:
        res = _cli(
            ["harness", "eval", "tooluse", "--backend", "byok", "--seed", "3"],
            env={
                "FX1_BYOK_BASE_URL": f"http://127.0.0.1:{port}/v1",
                "FX1_BYOK_ALLOW_PRIVATE_NETWORKS": "1",
                "FX1_BYOK_API_KEY": "stub-key",
                "FX1_BYOK_MODEL": "stub-v0",
            },
        )
    finally:
        chat.shutdown()
        chat.server_close()
    docs = _json_docs(res.stdout) if res.exit_code == 0 else []
    out["cli_inproc_leg_same_shape"] = (
        res.exit_code == 0
        and len(docs) == 1
        and set(docs[0]) == set(wire_rec)
        and docs[0]["status"] == "succeeded"
        and docs[0]["backend"] == "byok"
        and _CliChat.calls > 0
    )

    # --- shared lifecycle facts ------------------------------------------------
    from fx1.serve.evals import EVAL_SAMPLING

    out["decode_pin_recorded"] = wire_rec["sampling"] == EVAL_SAMPLING.body_fields()
    sub_creds = _submit(client, byok=dict(_BYOK_CREDS))
    rec_creds = _wait(client, sub_creds.json()["eval_id"])
    out["record_never_carries_credentials"] = (
        rec_creds["status"] in _TERMINAL
        and "bk-audit-secret" not in json.dumps(rec_creds)
        and "callback_secret" not in rec_creds
        and "byok" not in rec_creds
    )
    return out


def _idempotency_probes() -> dict[str, Any]:
    """Same key replays the record; the model never runs twice."""
    out: dict[str, Any] = {}
    backend = _EvalBackend()
    client, _api = _client({"byok": lambda: backend})

    a = _submit(client, key="idem-a", seed=7)
    rec_a = _wait(client, a.json()["eval_id"])
    calls_after_run = backend.calls
    b = _submit(client, key="idem-a", seed=7)
    out["idem_replay_same_record"] = (
        b.status_code == 202
        and b.json()["replayed"] is True
        and b.json()["eval_id"] == a.json()["eval_id"]
        and rec_a["status"] == "succeeded"
    )
    out["idem_replay_never_reruns"] = backend.calls == calls_after_run

    conflict = _submit(client, key="idem-a", seed=8)
    out["idem_conflict_409"] = conflict.status_code == 409

    c = _submit(client, key="idem-b", seed=7)
    out["idem_distinct_keys_independent"] = (
        c.json()["eval_id"] != a.json()["eval_id"] and c.json()["replayed"] is False
    )
    _wait(client, c.json()["eval_id"])

    overlong = _submit(client, key="k" * 257)
    out["idem_key_overlong_400"] = overlong.status_code == 400

    listed = client.get(_EVALS_PATH).json()
    out["idem_list_no_duplicates"] = (
        listed["total"] == 2 and len({r["eval_id"] for r in listed["records"]}) == 2
    )
    return out


def _transition_probes() -> dict[str, Any]:
    """Status machine + cancel/capacity/drain contract."""
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _EvalBackend()}, max_inflight=4)

    # unknown ids 404 on every per-eval route
    out["get_unknown_404"] = _record(client, "nonexistent-eval").status_code == 404
    out["cancel_unknown_404"] = client.delete(f"{_EVALS_PATH}/nonexistent-eval").status_code == 404
    out["receipt_unknown_404"] = (
        client.get(f"{_EVALS_PATH}/nonexistent-eval/receipt").status_code == 404
    )

    # queued → running → succeeded observed; transitions only forward.
    # A fast stub can already be terminal when the 202 renders, so the
    # honest invariant is monotone progress, never a pinned first state.
    sub = _submit(client, seed=1)
    seen: list[str] = [sub.json()["status"]]
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        rec = _record(client, sub.json()["eval_id"]).json()
        if rec["status"] != seen[-1]:
            seen.append(rec["status"])
        if rec["status"] in _TERMINAL:
            break
        time.sleep(0.005)
    out["lifecycle_queued_to_terminal"] = (
        seen[-1] == "succeeded"
        and set(seen) <= {"queued", "running", "succeeded"}
        and len(seen) == len(set(seen))
    )

    # cancel on terminal fails closed 409; the record is untouched
    term = client.delete(f"{_EVALS_PATH}/{sub.json()['eval_id']}")
    out["cancel_terminal_409"] = term.status_code == 409
    out["cancel_terminal_keeps_record"] = (
        _record(client, sub.json()["eval_id"]).json()["status"] == "succeeded"
    )

    # cancel on running fails closed 409; the eval completes anyway
    gate = threading.Event()
    blocked = _EvalBackend(gate=gate)
    client2, _ = _client({"byok": lambda: blocked}, max_inflight=2)
    sub_run = _submit(client2)
    _wait_running(client2, sub_run.json()["eval_id"])
    run_cancel = client2.delete(f"{_EVALS_PATH}/{sub_run.json()['eval_id']}")
    out["cancel_running_409"] = run_cancel.status_code == 409
    gate.set()
    rec_run = _wait(client2, sub_run.json()["eval_id"])
    out["running_eval_completes_after_409"] = rec_run["status"] == "succeeded"

    # queued cancel: park both workers so the eval is still queued at DELETE
    client3, _ = _client({"byok": lambda: _EvalBackend()}, max_inflight=2)
    parked = threading.Event()
    release = threading.Event()

    def _park() -> None:
        parked.set()
        release.wait(15)

    _jobs_exec(client3).submit(_park)
    _jobs_exec(client3).submit(_park)
    assert parked.wait(10), "workers never parked"
    qsub = _submit(client3)
    qid = qsub.json()["eval_id"]
    # a queued eval's receipt refuses — the sealed doc exists only at terminal
    qreceipt = client3.get(f"{_EVALS_PATH}/{qid}/receipt")
    out["receipt_nonterminal_409"] = (
        qreceipt.status_code == 409 and qreceipt.json().get("code") == "eval_not_terminal"
    )
    qcancel = client3.delete(f"{_EVALS_PATH}/{qid}")
    out["cancel_queued_200_cancelled"] = (
        qcancel.status_code == 200 and qcancel.json()["status"] == "cancelled"
    )
    # re-cancel of a cancelled record is idempotent — 200, not a second
    # transition (the cancel outcome for a 'cancelled' record IS 'cancelled')
    again = client3.delete(f"{_EVALS_PATH}/{qid}")
    out["cancel_cancelled_idempotent_200"] = (
        again.status_code == 200 and again.json()["status"] == "cancelled"
    )
    release.set()
    time.sleep(0.4)
    final = _record(client3, qid).json()
    out["cancelled_never_runs"] = final["status"] == "cancelled" and final.get("report") is None
    # a cancelled record is terminal — its sealed doc still exports
    out["receipt_cancelled_terminal_200"] = (
        client3.get(f"{_EVALS_PATH}/{qid}/receipt").status_code == 200
    )
    return out


def _capacity_probes() -> dict[str, Any]:
    """Drain + over_capacity + the wait surface's error contract."""
    out: dict[str, Any] = {}

    # --- drain: keyed replay still answers; new work refuses ---------------
    drain_client, _ = _client({"byok": lambda: _EvalBackend()}, api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}
    sub = _submit(drain_client, root_h, key="drain-key", seed=3)
    drain_id = sub.json()["eval_id"]
    _wait(drain_client, drain_id, root_h)
    d = drain_client.post("/harness/drain", headers=root_h)
    out["drain_latches_200"] = d.status_code == 200 and d.json()["draining"] is True
    refused = _submit(drain_client, root_h, seed=4)
    out["drain_refuses_new_eval_503"] = (
        refused.status_code == 503 and refused.json().get("code") == "draining"
    )
    replay = _submit(drain_client, root_h, key="drain-key", seed=3)
    out["drain_idem_replay_still_answers"] = (
        replay.status_code == 202
        and replay.json()["replayed"] is True
        and replay.json()["eval_id"] == drain_id
    )

    # --- over_capacity: one slot, one blocked eval → second submit refuses --
    gate = threading.Event()
    blocked = _EvalBackend(gate=gate)
    cap_client, _ = _client({"byok": lambda: blocked}, max_inflight=1)
    first = _submit(cap_client)
    _wait_running(cap_client, first.json()["eval_id"])
    second = _submit(cap_client)
    out["over_capacity_503_retry_after"] = (
        second.status_code == 503
        and second.json().get("code") == "over_capacity"
        and second.headers.get("retry-after") == "1"
    )
    gate.set()
    _wait(cap_client, first.json()["eval_id"])

    # --- wait_eval error contract --------------------------------------------
    from fx1.serve.client import HarnessClient

    def _dead(name: str, *a: Any, **k: Any) -> Any:
        raise RuntimeError("no provider creds in this probe")

    fail_client, _ = _client({}, resolver=_dead)
    hc = HarnessClient("http://testserver", transport=_tc_transport(fail_client))
    fsub = hc.submit_eval("tooluse", backend="hosted_k3", seed=0)
    fid = fsub["eval_id"]
    frec = _wait(fail_client, fid)
    out["wait_failed_raises_job_error"] = (
        frec["status"] == "failed"
        and _raises(lambda: hc.wait_eval(fid, timeout_s=5.0)) == "HarnessJobError"
        and _raises(lambda: hc.eval_status("deadbeef")) == "KeyError"
    )

    # cancelled record → wait_eval raises HarnessJobError
    release = threading.Event()
    cx_client, _ = _client({"byok": lambda: _EvalBackend()}, max_inflight=2)
    parked = threading.Event()

    def _park() -> None:
        parked.set()
        release.wait(15)

    _jobs_exec(cx_client).submit(_park)
    _jobs_exec(cx_client).submit(_park)
    assert parked.wait(10), "workers never parked"
    cxsub = _submit(cx_client)
    hc2 = HarnessClient("http://testserver2", transport=_tc_transport(cx_client))
    cxid = cxsub.json()["eval_id"]
    cx_client.delete(f"{_EVALS_PATH}/{cxid}")
    out["wait_cancelled_raises"] = (
        _raises(lambda: hc2.wait_eval(cxid, timeout_s=5.0)) == "HarnessJobError"
    )
    release.set()

    # timeout keeps the eval running — wait_eval raises transport, not a cancel
    gate3 = threading.Event()
    blocked3 = _EvalBackend(gate=gate3)
    tw_client, _ = _client({"byok": lambda: blocked3}, max_inflight=2)
    tsub = _submit(tw_client)
    tid = tsub.json()["eval_id"]
    _wait_running(tw_client, tid)
    hc3 = HarnessClient("http://testserver3", transport=_tc_transport(tw_client))
    out["wait_timeout_raises_transport"] = (
        _raises(lambda: hc3.wait_eval(tid, poll_s=0.05, timeout_s=0.2)) == "HarnessTransportError"
    )
    out["wait_timeout_eval_keeps_running"] = _record(tw_client, tid).json()["status"] == "running"
    gate3.set()
    _wait(tw_client, tid)
    return out


def _store_probes() -> dict[str, Any]:
    """Unit-level EvalStore probes — the atomic claim pins the cancel race."""
    out: dict[str, Any] = {}
    from fx1.serve.evals import EvalRecord, EvalStore

    def _rec(status: str = "queued") -> EvalRecord:
        return EvalRecord.model_validate(
            {
                "eval_id": f"ev-{status}-{time.time_ns()}",
                "suite": "tooluse",
                "backend": "byok",
                "seed": 0,
                "status": status,
                "created_at": time.time(),
            }
        )

    store = EvalStore(max_entries=8)
    rec = _rec()
    store.put(rec, None, None)
    out["start_queued_wins"] = store.start(rec.eval_id) is rec and rec.status == "running"
    out["start_running_loses"] = store.start(rec.eval_id) is None

    rec2 = _rec()
    store.put(rec2, "key-1", "fp-1")
    cancelled, outcome = store.cancel(rec2.eval_id)
    rec2_after = store.get(rec2.eval_id)
    out["cancel_then_start_loses"] = (
        outcome == "cancelled"
        and cancelled is not None
        and cancelled.status == "cancelled"
        and store.start(rec2.eval_id) is None
        and rec2_after is not None
        and rec2_after.status == "cancelled"
    )

    out["start_unknown_loses"] = store.start("never-submitted") is None
    out["cancel_terminal_reports_status"] = store.cancel(rec.eval_id)[1] == "running"

    # idem mapping survives on the record and drops with eviction
    store2 = EvalStore(max_entries=1)
    r1, r2 = _rec(), _rec()
    store2.put(r1, "k1", "fp1")
    store2.put(r2, "k2", "fp2")
    out["eviction_drops_idem_mapping"] = (
        store2.get_key("k1") is None
        and store2.get(r1.eval_id) is None
        and store2.get_key("k2") is not None
    )

    # eviction over a keyed submit lets the same key run again — the mapping
    # dropped with the evicted record, so a resubmit is new work
    tiny, _ = _client({"byok": lambda: _EvalBackend()}, job_max=1)
    s1 = _submit(tiny, key="ev-1")
    _wait(tiny, s1.json()["eval_id"])
    s2 = _submit(tiny, key="ev-2")
    _wait(tiny, s2.json()["eval_id"])
    s3 = _submit(tiny, key="ev-1")
    out["idem_evicted_key_resubmits_new"] = (
        s3.status_code == 202
        and s3.json()["replayed"] is False
        and s3.json()["eval_id"] != s1.json()["eval_id"]
        and _record(tiny, s1.json()["eval_id"]).status_code == 404
    )
    _wait(tiny, s3.json()["eval_id"])
    return out


def _receipt_probes() -> dict[str, Any]:
    """Sealed fx1_eval_record.v1 export: verifies, tamper fails closed."""
    out: dict[str, Any] = {}
    from quant_fund.schemas.receipt import (  # noqa: PLC0415
        verify_receipt_file,
        verify_receipt_payload,
    )

    client, _api = _client({"byok": lambda: _EvalBackend()})
    sub = _submit(client, seed=5)
    eval_id = sub.json()["eval_id"]
    rec = _wait(client, eval_id)

    doc_r = client.get(f"{_EVALS_PATH}/{eval_id}/receipt")
    out["receipt_200_terminal"] = doc_r.status_code == 200
    doc = doc_r.json()
    out["receipt_schema_pinned"] = (
        doc["kind"] == "fx1_eval_record" and doc["schema"] == "fx1_eval_record.v1"
    )
    out["receipt_record_is_served_record"] = doc["record"] == rec
    out["receipt_verifies"] = verify_receipt_payload(doc)["valid"] is True

    tampered = json.loads(json.dumps(doc))
    tampered["record"]["seed"] = 999
    out["receipt_tamper_fails_closed"] = verify_receipt_payload(tampered)["valid"] is False

    doc2 = client.get(f"{_EVALS_PATH}/{eval_id}/receipt").json()
    out["receipt_seal_deterministic"] = doc2["receipt_sha256"] == doc["receipt_sha256"]

    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/receipt.json"
        with open(path, "w") as fh:
            fh.write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        out["receipt_file_verifies"] = verify_receipt_file(path)["valid"] is True

    # failed evals seal too — evidence of the failure, honestly
    def _dead(name: str, *a: Any, **k: Any) -> Any:
        raise RuntimeError("no provider creds in this probe")

    client_f, _ = _client({}, resolver=_dead)
    fsub = client_f.post(_EVALS_PATH, json={"suite": "tooluse", "backend": "hosted_k3", "seed": 0})
    frec = _wait(client_f, fsub.json()["eval_id"])
    fdoc_r = client_f.get(f"{_EVALS_PATH}/{fsub.json()['eval_id']}/receipt")
    out["receipt_failed_eval_verifies"] = (
        frec["status"] == "failed"
        and fdoc_r.status_code == 200
        and verify_receipt_payload(fdoc_r.json())["valid"] is True
    )
    return out


def _backend_probes() -> dict[str, Any]:
    """Backend resolution, fallback chain attribution, metering."""
    out: dict[str, Any] = {}
    from fx1.serve.backends import BackendNotConfiguredError  # noqa: PLC0415

    # Resolution-time fault on the primary → the chain advances: an
    # unconfigured link is a 503 availability fault, so the fallback link
    # serves and attribution lands on the record.
    def _hosted_dead(name: str, *a: Any, **k: Any) -> Any:
        if name == "hosted_k3":
            raise BackendNotConfiguredError("hosted_k3 creds absent")
        return _EvalBackend(
            reply="0.9", usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
        )

    client, _api = _client({}, resolver=_hosted_dead)

    sub = _submit(client, backend="hosted_k3", fallbacks=["byok"], seed=2)
    rec = _wait(client, sub.json()["eval_id"])
    out["fallback_attribution_serving_link"] = (
        rec["backend"] == "byok" and rec["status"] == "succeeded"
    )
    out["fallback_attempts_ordered"] = (
        isinstance(rec["attempts"], list)
        and len(rec["attempts"]) == 2
        and rec["attempts"][0]["backend"] == "hosted_k3"
        and rec["attempts"][0]["ok"] is False
        and rec["attempts"][0]["error_class"] == "backend_unavailable"
        and rec["attempts"][1]["backend"] == "byok"
        and rec["attempts"][1]["ok"] is True
    )

    # dead chain end-to-end: record fails closed and seals which links
    # were tried — evidence of the refusal, not a silent drop
    def _all_dead(name: str, *a: Any, **k: Any) -> Any:
        raise BackendNotConfiguredError(f"{name} creds absent")

    client_dead, _ = _client({}, resolver=_all_dead)
    dead = _submit(client_dead, backend="hosted_k3", fallbacks=["byok"], seed=0)
    drec = _wait(client_dead, dead.json()["eval_id"])
    out["dead_chain_fails_closed_record"] = (
        drec["status"] == "failed"
        and "503" in (drec.get("error") or "")
        and isinstance(drec.get("attempts"), list)
        and len(drec["attempts"]) == 2
        and all(a["ok"] is False for a in drec["attempts"])
    )

    # a serving link whose *call* faults lands in the report, not the
    # record status: the eval ran to completion — the model simply failed
    # every task (eval semantics, not transport semantics)
    client_rf, _ = _client({"hosted_k3": lambda: _FailBackend()})
    rf = _submit(client_rf, backend="hosted_k3", seed=0)
    rrec = _wait(client_rf, rf.json()["eval_id"])
    outcomes = (rrec.get("report") or {}).get("outcomes") or []
    out["call_fault_lands_in_report"] = (
        rrec["status"] == "succeeded"
        and rrec["backend"] == "hosted_k3"
        and rrec["report"].get("pass_rate") == 0.0
        and len(outcomes) > 0
        and all(o["completed"] is False for o in outcomes)
    )

    # metered under the eval key — never conflated with user completions
    client_m, _ = _client({"byok": lambda: _EvalBackend()})
    msub = _submit(client_m, seed=4)
    _wait(client_m, msub.json()["eval_id"])
    snap = client_m.get("/metrics").json()
    eval_key = "eval:tooluse:byok"
    out["metered_under_eval_key"] = (
        eval_key in snap["complete"]
        and snap["complete"][eval_key]["latency_count"] > 0
        and "byok" not in snap["complete"]
    )

    # judge suites: judge meters under its own key
    client_j, _ = _client({"byok": lambda: _EvalBackend(), "local_fx1": lambda: _EvalBackend()})
    jsub = client_j.post(
        _EVALS_PATH,
        json={
            "suite": "ext_bench",
            "backend": "byok",
            "seed": 0,
            "judge_backend": "byok",
        },
    )
    out["judge_suite_submit_accepted"] = jsub.status_code == 202
    jrec = _wait(client_j, jsub.json()["eval_id"], timeout_s=60.0)
    jsnap = client_j.get("/metrics").json()
    out["judge_metered_own_key"] = (
        jrec["status"] == "succeeded"
        and "eval:ext_bench:judge:byok" in jsnap["complete"]
        and "eval:ext_bench:byok" in jsnap["complete"]
    )

    # byok override binds provider creds for this request only — the
    # normalizer spreads them into resolver kwargs
    seen: dict[str, Any] = {}

    def _resolver(name: str, *a: Any, **k: Any) -> Any:
        seen["name"] = name
        seen["kwargs"] = k
        return _EvalBackend()

    client_o, _ = _client({}, resolver=_resolver)
    osub = client_o.post(
        _EVALS_PATH,
        json={"suite": "tooluse", "backend": "byok", "seed": 0, "byok": dict(_BYOK_CREDS)},
    )
    orec = _wait(client_o, osub.json()["eval_id"])
    out["byok_override_binds_resolver"] = (
        orec["status"] == "succeeded"
        and seen.get("name") == "byok"
        and (seen.get("kwargs") or {}).get("api_key") == _BYOK_CREDS["api_key"]
        and (seen.get("kwargs") or {}).get("model") == _BYOK_CREDS["model"]
    )
    out["byok_override_never_on_record"] = "api_key" not in json.dumps(orec)
    return out


def _concurrency_probes() -> dict[str, Any]:
    """N parallel evals never cross-contaminate records."""
    out: dict[str, Any] = {}
    n = 6
    client, _api = _client({"byok": lambda: _EvalBackend()}, max_inflight=8)

    def _one(i: int) -> tuple[str, int]:
        r = _submit(client, seed=i, key=f"par-{i}")
        assert r.status_code == 202, r.text
        _wait(client, r.json()["eval_id"])
        return str(r.json()["eval_id"]), i

    with ThreadPoolExecutor(max_workers=n) as pool:
        pairs = list(pool.map(_one, range(n)))
    ids = {eid for eid, _i in pairs}
    out["parallel_distinct_ids"] = len(ids) == n

    records = [_record(client, eid).json() for eid, _ in pairs]
    out["parallel_all_succeeded"] = all(r["status"] == "succeeded" for r in records)
    out["parallel_seed_attributed"] = all(
        r["seed"] == i for r, (_e, i) in zip(records, pairs, strict=True)
    )
    listed = client.get(_EVALS_PATH).json()
    out["parallel_list_consistent"] = (
        listed["total"] == n and len({r["eval_id"] for r in listed["records"]}) == n
    )
    return out


def _durability_probes() -> dict[str, Any]:
    """--state-dir journals: records, idem keys, and crash honesty."""
    out: dict[str, Any] = {}
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        client1, _ = _client({"byok": lambda: _EvalBackend()}, state_dir=td)
        sub = _submit(client1, key="durable-1", seed=3)
        eval_id = sub.json()["eval_id"]
        rec1 = _wait(client1, eval_id)

        # fresh process on the same dir answers the record verbatim
        client2, _ = _client({"byok": lambda: _EvalBackend()}, state_dir=td)
        rec2 = _record(client2, eval_id)
        out["state_dir_record_survives_restart"] = rec2.status_code == 200 and rec2.json() == rec1
        replay = _submit(client2, key="durable-1", seed=3)
        out["state_dir_idem_key_survives"] = (
            replay.status_code == 202
            and replay.json()["replayed"] is True
            and replay.json()["eval_id"] == eval_id
        )

    with tempfile.TemporaryDirectory() as td:
        gate = threading.Event()
        blocked = _EvalBackend(gate=gate)
        client3, _ = _client({"byok": lambda: blocked}, state_dir=td)
        csub = _submit(client3)
        cid = csub.json()["eval_id"]
        _wait_running(client3, cid)
        # simulate crash: fresh app on the same journal while it was running
        client4, _ = _client({"byok": lambda: _EvalBackend()}, state_dir=td)
        crashed = _record(client4, cid).json()
        out["state_dir_crash_restart_fails"] = crashed["status"] == "failed" and "restarted" in (
            crashed.get("error") or ""
        )
        resub = _submit(client4)
        out["post_crash_submits_fine"] = resub.status_code == 202
        _wait(client4, resub.json()["eval_id"])
        gate.set()

    # SDK twin binds the same journals
    with tempfile.TemporaryDirectory() as td:
        sdk1 = _sdk({"byok": lambda: _EvalBackend()}, state_dir=td)
        srec = sdk1.run_eval("tooluse", backend="byok", seed=9)
        sdk2 = _sdk({"byok": lambda: _EvalBackend()}, state_dir=td)
        back = sdk2.eval_record(srec.eval_id)
        out["sdk_state_dir_survives"] = (
            back is not None and back.status == "succeeded" and back.eval_id == srec.eval_id
        )
    return out


def _v1_evals_probes() -> dict[str, Any]:
    """The /v1/evals spec+run surface over the same store."""
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _EvalBackend()})

    spec_body = {
        "name": "regression-bank",
        "data_source_config": {
            "type": "custom",
            "item_schema": {"suite": "tooluse", "seed": 11, "backend": "byok"},
        },
        "metadata": {"lane": "145"},
    }
    spec_r = client.post("/v1/evals", json=spec_body)
    out["spec_create_201"] = spec_r.status_code == 201
    spec_id = spec_r.json()["id"]

    got = client.get(f"/v1/evals/{spec_id}")
    out["spec_get_roundtrip"] = got.status_code == 200 and got.json()["id"] == spec_id
    upd = client.post(f"/v1/evals/{spec_id}", json={"name": "regression-bank-v2"})
    out["spec_update_name"] = upd.status_code == 200 and upd.json()["name"] == "regression-bank-v2"
    frozen = client.post(
        f"/v1/evals/{spec_id}",
        json={"data_source_config": {"type": "custom", "item_schema": {}}},
    )
    out["spec_datasource_frozen_422"] = frozen.status_code == 422

    # runs bind spec+model; scoped idem replays per spec
    run = client.post(
        f"/v1/evals/{spec_id}/runs",
        json={"model": "byok"},
        headers={"Idempotency-Key": "run-key-1"},
    )
    out["run_create_201"] = run.status_code == 201
    run_id = run.json()["id"]
    bare = run_id.removeprefix("evalrun_")
    out["run_binds_spec_model"] = run.json()["eval_id"] == spec_id and run.json()["model"] == "byok"
    rec = _wait(client, bare)
    out["run_terminal_wire"] = rec["status"] == "succeeded"
    out["run_binding_on_record"] = rec["eval_spec"] == spec_id and rec["eval_model"] == "byok"

    replay = client.post(
        f"/v1/evals/{spec_id}/runs",
        json={"model": "byok"},
        headers={"Idempotency-Key": "run-key-1"},
    )
    out["run_scoped_idem_replays"] = replay.status_code == 201 and replay.json()["id"] == run_id

    # model mapping fails closed — v1 error bodies nest code under error
    bad_model = client.post(f"/v1/evals/{spec_id}/runs", json={"model": "gpt-9"})
    out["run_unknown_model_400"] = (
        bad_model.status_code == 400
        and bad_model.json().get("error", {}).get("code") == "invalid_request"
    )
    ft_model = client.post(f"/v1/evals/{spec_id}/runs", json={"model": "ft:ghost"})
    out["run_ft_unregistered_404"] = (
        ft_model.status_code == 404
        and ft_model.json().get("error", {}).get("code") == "model_not_found"
    )
    fx1_model = client.post(f"/v1/evals/{spec_id}/runs", json={"model": "fx1"})
    # 'fx1' maps to the local_fx1 link — never invalid_request. The link's
    # resolution is async: it lands on the record (no checkpoint configured
    # here → the local_fx1 gate 422 lands in the record's error, honestly).
    fx1_rec = (
        _wait(client, fx1_model.json()["id"].removeprefix("evalrun_"))
        if fx1_model.status_code == 201
        else {}
    )
    out["run_fx1_maps_local_link"] = (
        fx1_model.status_code == 201
        and fx1_rec.get("eval_model") == "fx1"
        and fx1_rec.get("status") == "failed"
        and "checkpoint" in (fx1_rec.get("error") or "")
    )

    # run cross-spec isolation + list cursors
    other = client.post("/v1/evals", json=spec_body).json()["id"]
    foreign = client.get(f"/v1/evals/{other}/runs/{run_id}")
    out["run_under_wrong_spec_404"] = foreign.status_code == 404
    listed = client.get(f"/v1/evals/{spec_id}/runs").json()
    out["run_list_scoped"] = (
        listed["has_more"] is False
        and len(listed["data"]) == 2
        and {r["id"] for r in listed["data"]} == {run_id, fx1_model.json()["id"]}
    )
    cursor_prefixed = client.get(f"/v1/evals/{spec_id}/runs", params={"after": run_id})
    cursor_bare = client.get(f"/v1/evals/{spec_id}/runs", params={"after": bare})
    out["run_list_cursor_roundtrips"] = (
        cursor_prefixed.status_code == 200
        and cursor_bare.status_code == 200
        and cursor_prefixed.json()["data"] == cursor_bare.json()["data"]
    )
    bad_cursor = client.get(f"/v1/evals/{spec_id}/runs", params={"after": "evalrun_nope"})
    out["run_list_bad_cursor_400"] = (
        bad_cursor.status_code == 400
        and bad_cursor.json().get("error", {}).get("code") == "invalid_cursor"
    )

    # output_items: per-task verdict rows, cursor binds the run
    items = client.get(f"/v1/evals/{spec_id}/runs/{run_id}/output_items")
    out["output_items_rows"] = (
        items.status_code == 200
        and len(items.json()["data"]) > 0
        and all(i["run_id"] == run_id for i in items.json()["data"])
        and all(i["id"].startswith(f"{run_id}-") for i in items.json()["data"])
    )
    page1 = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items", params={"limit": 2}
    ).json()
    nxt = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items",
        params={"limit": 2, "after": page1["last_id"]},
    ).json()
    out["output_items_cursor_paginates"] = (
        len(page1["data"]) == 2
        and page1["has_more"] is True
        and len(nxt["data"]) == 2
        and nxt["data"][0]["id"] != page1["data"][-1]["id"]
    )
    malformed = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items", params={"after": "zzz"}
    )
    foreign_item = client.get(
        f"/v1/evals/{spec_id}/runs/{run_id}/output_items",
        params={"after": "evalrun_deadbeef-3"},
    )
    out["output_items_bad_cursor_400"] = (
        malformed.status_code == 400
        and malformed.json().get("error", {}).get("code") == "invalid_cursor"
        and foreign_item.status_code == 400
        and foreign_item.json().get("error", {}).get("code") == "invalid_cursor"
    )

    # spec list cursor fail-closed
    spec_bad = client.get("/v1/evals", params={"after": "eval_neverexisted"})
    out["spec_list_bad_cursor_400"] = (
        spec_bad.status_code == 400
        and spec_bad.json().get("error", {}).get("code") == "invalid_cursor"
    )
    spec_page = client.get("/v1/evals", params={"after": spec_id})
    out["spec_list_cursor_known_200"] = spec_page.status_code == 200

    # delete run → tombstone both surfaces
    dele = client.delete(f"/v1/evals/{spec_id}/runs/{run_id}")
    out["run_delete_terminal_200"] = dele.status_code == 200
    out["run_delete_removes_record"] = _record(client, bare).status_code == 404

    # delete spec → tombstone
    sdel = client.delete(f"/v1/evals/{spec_id}")
    out["spec_delete_200_tombstone"] = (
        sdel.status_code == 200 and client.get(f"/v1/evals/{spec_id}").status_code == 404
    )

    # item_schema validated at create — unknown suite / garbage schema 422
    bad_spec = client.post(
        "/v1/evals",
        json={
            "name": "bad",
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "not-a-suite"},
            },
        },
    )
    out["spec_bad_item_schema_422"] = bad_spec.status_code == 422
    extra_field = client.post("/v1/evals", json={**spec_body, "compute_units": 5})
    out["spec_extra_field_forbidden"] = extra_field.status_code == 422
    return out


def _cli_probes() -> dict[str, Any]:
    """CLI legs — remote eval-wait re-attach + --receipt sealed doc."""
    out: dict[str, Any] = {}
    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    saved = {k: os.environ.get(k) for k in _SWEPT_ENVS}
    try:
        for k in _SWEPT_ENVS:
            os.environ.pop(k, None)
        os.environ[_API_KEY_ENV] = _ROOT
        app = api_mod.create_app(
            harness=Harness(runner=fake_runner),
            backend_resolver=lambda name, *a, **k: _EvalBackend(),
        )
        server, _thread, port = _serve_uvicorn(app)
        base = f"http://127.0.0.1:{port}"  # NOSONAR(S5332) — loopback probe URL
        try:
            res = _cli(
                [
                    "harness",
                    "eval",
                    "tooluse",
                    "--backend",
                    "byok",
                    "--seed",
                    "5",
                    "--remote",
                    base,
                    "--api-key",
                    _ROOT,
                    "--no-wait",
                ]
            )
            subs = _json_docs(res.stdout) if res.exit_code == 0 else []
            out["cli_remote_no_wait_submit"] = (
                res.exit_code == 0
                and len(subs) == 1
                and subs[0]["status"] in ("queued", "running", "succeeded")
                and subs[0]["replayed"] is False
            )
            eval_id = str(subs[0]["eval_id"]) if subs else "missing"

            wr = _cli(
                [
                    "harness",
                    "eval-wait",
                    eval_id,
                    "--remote",
                    base,
                    "--api-key",
                    _ROOT,
                    "--receipt",
                ]
            )
            wdocs = _json_docs(wr.stdout) if wr.exit_code == 0 else []
            out["cli_remote_eval_wait_terminal"] = (
                wr.exit_code == 0
                and len(wdocs) == 2
                and wdocs[0]["status"] == "succeeded"
                and wdocs[0]["eval_id"] == eval_id
            )
            out["cli_remote_receipt_sealed"] = (
                len(wdocs) == 2
                and wdocs[1]["kind"] == "fx1_eval_record"
                and wdocs[1]["schema"] == "fx1_eval_record.v1"
                and wdocs[1]["record"]["eval_id"] == eval_id
            )
            from quant_fund.schemas.receipt import (  # noqa: PLC0415
                verify_receipt_payload,
            )

            out["cli_remote_receipt_verifies"] = (
                len(wdocs) == 2 and verify_receipt_payload(wdocs[1])["valid"] is True
            )

            st = _cli(["harness", "eval-status", eval_id, "--remote", base, "--api-key", _ROOT])
            sdocs = _json_docs(st.stdout) if st.exit_code == 0 else []
            out["cli_eval_status_record"] = (
                st.exit_code == 0 and len(sdocs) == 1 and sdocs[0]["status"] == "succeeded"
            )
            st_rc = _cli(
                [
                    "harness",
                    "eval-status",
                    eval_id,
                    "--receipt",
                    "--remote",
                    base,
                    "--api-key",
                    _ROOT,
                ]
            )
            rdocs = _json_docs(st_rc.stdout) if st_rc.exit_code == 0 else []
            out["cli_eval_status_receipt"] = (
                st_rc.exit_code == 0
                and len(rdocs) == 1
                and rdocs[0]["record"]["eval_id"] == eval_id
            )

            ls = _cli(["harness", "evals", "--remote", base, "--api-key", _ROOT, "--limit", "10"])
            ldocs = _json_docs(ls.stdout) if ls.exit_code == 0 else []
            out["cli_evals_lists_run"] = (
                ls.exit_code == 0
                and len(ldocs) == 1
                and eval_id in {r["eval_id"] for r in ldocs[0]["records"]}
            )

            # terminal cancel → clean CLI failure (exit 2, one error line)
            cx = _cli(["harness", "eval-cancel", eval_id, "--remote", base, "--api-key", _ROOT])
            out["cli_cancel_terminal_exit2"] = cx.exit_code == 2

            ux = _cli(
                [
                    "harness",
                    "eval-status",
                    "eval-nonexistent",
                    "--remote",
                    base,
                    "--api-key",
                    _ROOT,
                ]
            )
            out["cli_unknown_id_exit2"] = ux.exit_code == 2
        finally:
            server.should_exit = True
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def _boundary_probes() -> dict[str, Any]:
    """Every knob violation answers its mapped wire code at submit."""
    out: dict[str, Any] = {}
    client, _api = _client({"byok": lambda: _EvalBackend()})

    cases = [
        ("unknown_suite_422", {"suite": "not-a-suite", "backend": "byok", "seed": 0}, 422),
        ("unknown_backend_422", {"suite": "tooluse", "backend": "nope", "seed": 0}, 422),
        ("malformed_body_422", {}, 422),
        ("seed_negative_422", {"suite": "tooluse", "backend": "byok", "seed": -1}, 422),
        (
            "timeout_over_cap_422",
            {"suite": "tooluse", "backend": "byok", "seed": 0, "timeout_s": 99999},
            422,
        ),
        (
            "fallbacks_over_two_422",
            {
                "suite": "tooluse",
                "backend": "byok",
                "seed": 0,
                "fallbacks": ["hosted_k3", "local_fx1", "byok"],
            },
            422,
        ),
        (
            "fallback_repeats_primary_422",
            {"suite": "tooluse", "backend": "byok", "seed": 0, "fallbacks": ["byok"]},
            422,
        ),
        (
            "judge_on_nonjudge_suite_422",
            {"suite": "tooluse", "backend": "byok", "seed": 0, "judge_backend": "byok"},
            422,
        ),
        (
            "judge_byok_without_byok_judge_422",
            {
                "suite": "capability",
                "backend": "byok",
                "seed": 0,
                "judge_backend": "local_fx1",
                "judge_byok": dict(_BYOK_CREDS),
            },
            422,
        ),
        (
            "callback_secret_needs_url_422",
            {"suite": "tooluse", "backend": "byok", "seed": 0, "callback_secret": "s"},
            422,
        ),
        (
            "callback_url_non_http_422",
            {"suite": "tooluse", "backend": "byok", "seed": 0, "callback_url": "ftp://x"},
            422,
        ),
        (
            "byok_override_wrong_link_422",
            {
                "suite": "tooluse",
                "backend": "hosted_k3",
                "seed": 0,
                "byok": dict(_BYOK_CREDS),
            },
            422,
        ),
        (
            "checkpoint_wrong_link_422",
            {
                "suite": "tooluse",
                "backend": "byok",
                "seed": 0,
                "checkpoint_dir": _MISSING_DIR,
            },
            422,
        ),
    ]
    for name, body, want in cases:
        r = client.post(_EVALS_PATH, json=body)
        out[name] = int(r.status_code) == want

    listed_bad_status = client.get(_EVALS_PATH, params={"status": "bogus"})
    out["list_status_filter_422"] = listed_bad_status.status_code == 422
    listed_over_limit = client.get(_EVALS_PATH, params={"limit": 999})
    out["list_limit_cap_422"] = listed_over_limit.status_code == 422

    # auth scopes: read can't submit/cancel; write can't mint; none 401s
    keyed, _ = _client({"byok": lambda: _EvalBackend()}, api_key=_ROOT)
    root_h = {"X-API-Key": _ROOT}
    read_raw, _read_id = _mint(keyed, root_h, scopes=["read"])
    write_raw, _write_id = _mint(keyed, root_h, scopes=["read", "write"])
    out["no_key_401"] = _submit(keyed).status_code == 401
    out["read_scope_submit_403"] = (
        _submit(keyed, headers={"X-API-Key": read_raw}).status_code == 403
    )
    out["read_scope_read_ok"] = (
        keyed.get(_EVALS_PATH, headers={"X-API-Key": read_raw}).status_code == 200
    )
    wsub = _submit(keyed, headers={"X-API-Key": write_raw})
    out["write_scope_submit_202"] = wsub.status_code == 202
    _wait(keyed, wsub.json()["eval_id"], {"X-API-Key": write_raw})
    out["write_scope_admin_403"] = (
        keyed.post("/harness/keys", json={}, headers={"X-API-Key": write_raw}).status_code == 403
    )
    out["read_scope_cancel_403"] = (
        keyed.delete(
            f"{_EVALS_PATH}/{wsub.json()['eval_id']}", headers={"X-API-Key": read_raw}
        ).status_code
        == 403
    )
    return out


def _callback_probes() -> dict[str, Any]:
    """Terminal webhook: payload is the record, secret signs but never echoes."""
    out: dict[str, Any] = {}
    sink = _Sink()
    # Loopback webhook sink — opt into private-network delivery for these
    # probes; production callbacks stay public-only unless opted in.
    hook_prev = os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS")
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
    try:
        client, _api = _client({"byok": lambda: _EvalBackend()})
        sub = _submit(
            client,
            callback_url=sink.url("/hook"),
            callback_secret="cb-secret-1",
        )
        eval_id = sub.json()["eval_id"]
        _wait(client, eval_id)
        deadline = time.monotonic() + 10.0
        while not sink.hits and time.monotonic() < deadline:
            time.sleep(0.05)
        hit = sink.hits[0] if sink.hits else None
        payload = json.loads(hit.body) if hit is not None else {}
        # the delivery verdict lands after the terminal status — re-read
        cb_deadline = time.monotonic() + 10.0
        rec = _record(client, eval_id).json()
        while rec.get("callback_status") is None and time.monotonic() < cb_deadline:
            time.sleep(0.05)
            rec = _record(client, eval_id).json()
        out["callback_fires_terminal"] = (
            hit is not None
            and hit.path == "/hook"
            and payload.get("eval_id") == eval_id
            and payload.get("status") == "succeeded"
            and rec["callback_status"] == "delivered"
            and rec["callback_attempts"] >= 1
        )
        out["callback_payload_is_full_record"] = hit is not None and payload.get(
            "report"
        ) == rec.get("report")
        out["callback_secret_signed_not_echoed"] = (
            hit is not None
            and "cb-secret-1" not in hit.body.decode(errors="replace")
            and any("signature" in k.lower() for k in hit.headers)
        )

        # dead endpoint → recorded failure, record still terminal. The
        # callback is delivered by the worker after the terminal status
        # lands — poll the record until the delivery verdict seals.
        sub2 = _submit(client, callback_url=sink.url("/fail"))
        sub2_id = sub2.json()["eval_id"]
        rec2 = _wait(client, sub2_id)
        cb_deadline = time.monotonic() + 15.0
        while rec2.get("callback_status") is None and time.monotonic() < cb_deadline:
            time.sleep(0.05)
            rec2 = _record(client, sub2_id).json()
        out["callback_dead_recorded_failed"] = (
            rec2["status"] == "succeeded"
            and rec2["callback_status"] == "failed"
            and isinstance(rec2.get("callback_error"), str)
        )

        # queued cancel fires the callback with the cancelled record
        client3, _ = _client({"byok": lambda: _EvalBackend()}, max_inflight=2)
        release = threading.Event()

        def _park() -> None:
            release.wait(15)

        _jobs_exec(client3).submit(_park)
        _jobs_exec(client3).submit(_park)
        time.sleep(0.1)
        qsub = _submit(client3, callback_url=sink.url("/cancelled"))
        client3.delete(f"{_EVALS_PATH}/{qsub.json()['eval_id']}")
        deadline = time.monotonic() + 10.0
        got = None
        while time.monotonic() < deadline:
            got = next((h for h in sink.hits if h.path == "/cancelled"), None)
            if got is not None:
                break
            time.sleep(0.05)
        release.set()
        out["cancel_fires_callback"] = (
            got is not None and json.loads(got.body).get("status") == "cancelled"
        )
    finally:
        try:
            sink.close()
        finally:
            # Never leak the private-network callback opt-in if sink cleanup
            # raises while unwinding the probe.
            if hook_prev is None:
                os.environ.pop("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", None)
            else:
                os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = hook_prev
    return out


def _diff_probes() -> dict[str, Any]:
    """Promotion-gate diff: comparable only on the same bank+seed."""
    out: dict[str, Any] = {}
    client, _api = _client(
        {"byok": lambda: _EvalBackend(reply="1.0"), "hosted_k3": lambda: _EvalBackend()}
    )

    a = client.post(_EVALS_PATH, json={"suite": "ts_reasoning", "backend": "byok", "seed": 0})
    b = client.post(_EVALS_PATH, json={"suite": "ts_reasoning", "backend": "hosted_k3", "seed": 0})
    _wait(client, a.json()["eval_id"])
    _wait(client, b.json()["eval_id"])
    d = client.get(f"{_EVALS_PATH}/{a.json()['eval_id']}/diff/{b.json()['eval_id']}")
    body = d.json() if d.status_code == 200 else {}
    out["diff_same_seed_comparable"] = (
        d.status_code == 200
        and body.get("same_suite") is True
        and body.get("same_seed") is True
        and body.get("comparable") is True
        and body.get("verdict") in ("improved", "regressed", "unchanged")
    )

    # non-terminal record fails the diff closed — both ids in one store
    gate = threading.Event()
    c2, _ = _client({"byok": lambda: _EvalBackend(gate=gate)}, max_inflight=2)
    hold_a = _submit(c2).json()["eval_id"]
    hold_b = _submit(c2).json()["eval_id"]
    _wait_running(c2, hold_a)
    d_hold = c2.get(f"{_EVALS_PATH}/{hold_a}/diff/{hold_b}")
    out["diff_nonterminal_409"] = d_hold.status_code == 409
    gate.set()
    _wait(c2, hold_a)
    _wait(c2, hold_b)

    c = client.post(_EVALS_PATH, json={"suite": "ts_reasoning", "backend": "byok", "seed": 9})
    _wait(client, c.json()["eval_id"])
    d2 = client.get(f"{_EVALS_PATH}/{a.json()['eval_id']}/diff/{c.json()['eval_id']}")
    body2 = d2.json() if d2.status_code == 200 else {}
    out["diff_cross_seed_unknown"] = (
        d2.status_code == 200
        and body2.get("same_seed") is False
        and body2.get("comparable") is False
        and body2.get("verdict") == "unknown"
    )
    out["diff_unknown_404"] = (
        client.get(f"{_EVALS_PATH}/{a.json()['eval_id']}/diff/nope").status_code == 404
    )
    return out


def eval_lifecycle_audit() -> dict[str, Any]:
    """Run the eval-lifecycle battery; returns literal bools."""
    out: dict[str, Any] = {}
    out.update(_legs_probes())
    out.update(_idempotency_probes())
    out.update(_transition_probes())
    out.update(_capacity_probes())
    out.update(_store_probes())
    out.update(_receipt_probes())
    out.update(_backend_probes())
    out.update(_concurrency_probes())
    out.update(_durability_probes())
    out.update(_v1_evals_probes())
    out.update(_cli_probes())
    out.update(_boundary_probes())
    out.update(_callback_probes())
    out.update(_diff_probes())
    return out


def eval_lifecycle_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under eval_lifecycle_audit.v1."""
    r = eval_lifecycle_audit()
    ok = all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "eval_lifecycle_audit",
        "schema": "eval_lifecycle_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "The eval surface is one durable state machine: every leg "
            "(HTTP, HarnessClient, Fx1Harness SDK, CLI in-process and "
            "--remote) produces the same record shape and walks "
            "queued→running→terminal. Idempotency replays the record "
            "without re-running the model, scopes per spec under /v1/evals, "
            "survives drain, and drops with eviction. Cancel is an atomic "
            "claim against the worker — a cancelled eval can never "
            "resurrect to succeeded; running/terminal cancels 409. Wait "
            "surfaces raise HarnessJobError on failed/cancelled and "
            "HarnessTransportError on timeout without killing the eval. "
            "Terminal records export sealed fx1_eval_record.v1 receipts "
            "that verify, fail closed under tamper, and seal "
            "deterministically; a non-terminal receipt 409s. Backend "
            "resolution attributes the serving link, orders chain "
            "attempts, and meters eval/judge calls under eval:* keys "
            "separate from user completions. Records, idem keys, and "
            "restart-failed honesty survive --state-dir restarts on both "
            "the wire and the SDK twin. The /v1/evals surface binds "
            "spec+model at construction, validates schemas and cursors "
            "fail-closed, and tombstones specs and terminal runs. Every "
            "boundary violation answers its mapped wire code before "
            "anything queues; read keys can't submit and write keys can't "
            "admin. Terminal callbacks POST the full record, sign without "
            "echoing the secret, and record delivery failures on the "
            "record itself. Diffs report comparable verdicts only on the "
            "same bank+seed and fail closed on non-terminal records."
            if ok
            else f"EVAL LIFECYCLE AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(eval_lifecycle_audit_bench(), indent=2, sort_keys=True))
