"""cancel_audit — ``cancel``/``pause``/``interrupt`` deep audit.

Probe battery over every cancel transition on the async surfaces:
``DELETE /harness/jobs/{id}`` + ``DELETE /harness/evals/{id}`` (harness
queue), ``POST /v1/evals/{eid}/runs/{rid}/cancel`` (OpenAI eval runs),
``POST /v1/responses/{id}/cancel`` (background responses),
``POST /v1/batches/{id}/cancel`` (OpenAI batch),
``POST /v1/messages/batches/{id}/cancel`` (Anthropic batch),
``POST /v1/uploads/{id}/cancel`` (upload intents),
``POST /v1/vector_stores/{vs}/file_batches/{bid}/cancel`` (vs batches —
terminal by construction), and
``POST /v1/fine_tuning/jobs/{id}/cancel`` + ``pause`` + ``resume``.

Pinned contracts, exercised end to end:

- *Transition ordering* — ``queued→cancelled`` direct (busy executor
  pins the pre-start window), ``running→cancel`` honoured or refused
  per surface contract (jobs/evals/eval-runs refuse mid-flight 409 —
  no kill handle; responses/batches/ft honour it cooperatively via a
  cancel flag/CAS the worker checks between units), terminal statuses
  always refuse (400/409 with an honest code), and re-cancels are
  either idempotent (jobs/evals/eval-runs/batches) or honestly refused
  (responses 409 ``cancel_terminal``).
- *Side effects* — a cancelled batch's ``output_file_id`` is minted only
  when lines completed (partial results, zero lines → no file); a
  cancelled bg response polls ``cancelled`` and the worker's late
  completion is dropped by ``put_unless_status`` (never flips back);
  a cancelled ft job publishes no checkpoint; a cancelled upload frees
  its parts; a cancelled eval run's ``output_items`` is empty and its
  wire shows ``error.code=eval_run_canceled``.
- *Terminal invariants* — cancel after terminal always refuses; cancel
  during processing is atomic (the CAS family from #2810 —
  ``transition_status``/``put_unless_status`` on responses, the batch
  ``_state_lock`` + ``_cancel`` event, the store-lock flips on
  jobs/evals/ft — a terminal verdict is never overwritten).
- *Cancel+webhook* — cancelled is terminal so the signed webhook fires
  once (``_callback_fired`` covers re-cancels, cancel+complete races,
  expiry re-projects); the HMAC verifies and the secret never echoes.
- *Cancel+conv* — a cancelled response never appends to its
  conversation (the commit-gate covers cancelled, not only failed).
- *Cancel+idem* — keyed cancel replays dedupe (uploads carry the
  ``X-Fx1-Idempotent-Replay`` header); a keyed submit's cancel verdict
  survives a delete+replay — never resurrects as ``queued`` or
  ``completed``; parallel cancels CAS — at most one wins the
  transition.
- *Cancel+storage* — the cancelled record persists under GET showing
  the cancelled verdict; a deleted cancelled record 404s (uploads
  carry no GET surface at all — pinned as the route's own 404).
- *Pause/resume (ft)* — queued pause parks pre-start, running pause
  parks at the runner's ``pause_gate`` boundary, resume restores the
  captured status, terminal pauses refuse 409, non-paused resumes
  refuse 409 ``job_not_paused``, and a second pause replays without
  double-counting its event.
- *Scope* — cancel needs write; a read-only key's cancel 403s
  ``insufficient_scope`` on every surface.
- *Drain* — drain refuses new submits (503 ``draining``) but cancels
  stay open — control-plane verbs are exempt.
- *Metering* — cancel bills ``uses`` on the caller key; a cancelled
  batch's token meter reflects only the lines that ran (partial usage
  is reported, never zeroed after the fact).
- *Envelope* — every refusal lands in the surface's error envelope
  ({detail, code} on harness routes, the OpenAI/Anthropic shapes on
  /v1).

Found while building this lane (fixed in the same commit):

* A cancelled background response could resurrect through the
  ``Idempotency-Key`` replay of its submit: ``DELETE`` dropped the live
  record and a keyed retry rehydrated the idem cache's stale snapshot
  — ``queued`` when the worker had not started (a zombie record that
  no worker was bound to, frozen non-terminal forever) and
  ``completed`` when the cancel landed mid-flight (the cancel verdict
  flipped back). The worker now syncs the cached envelope to the live
  verdict on every exit path (``_sync_idem_verdict`` — live record
  wins; a committed cancel stamps ``cancelled`` even when the record
  is already gone), and ``OpenAIEnvelopeStore.repin`` refuses to
  rehydrate a non-terminal response — a revived ``queued``/
  ``in_progress`` record can never advance, so the recorded answer is
  served without resurrecting the retrieval entry.

Sealed ``cancel_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

__all__ = ["cancel_audit", "cancel_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "c4nc3l-root"
_WAIT_S = 20.0
_IDEM = "Idempotency-Key"
_JOBS = "/harness/jobs"
_EVALS = "/harness/evals"
_DRAIN = "/harness/drain"
_ANTHROPIC_H = {"anthropic-version": "2023-06-01"}
_JOB_TERMINAL = {"succeeded", "failed", "cancelled"}
_RESP_TERMINAL = {"completed", "failed", "cancelled", "incomplete"}
_BATCH_TERMINAL = {"completed", "failed", "expired", "cancelled"}
_FT_TERMINAL = {"succeeded", "failed", "cancelled"}
_FT_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
_U = {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}


# ---------------------------------------------------------------------------
# Stub plumbing — runners, backends, webhook sink, app/client factories
# ---------------------------------------------------------------------------


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


class _Runner:
    """``Harness`` runner stub — records argv; an optional gate parks the
    subprocess call so a probe can land a cancel deterministically while
    the job is ``running``."""

    def __init__(
        self,
        gate: threading.Event | None = None,
        *,
        raise_exc: BaseException | None = None,
        exit_code: int = 0,
    ) -> None:
        self.calls: list[list[str]] = []
        self.gate = gate
        self.entered = threading.Event()
        self.raise_exc = raise_exc
        self.exit_code = exit_code

    def __call__(self, argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        del timeout_s
        self.calls.append(list(argv))
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(30)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.exit_code, "ok", ""


def _make_app(
    *,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    backend_map: dict[str, Callable[[], Any]] | None = None,
    ft_runner: Any = None,
    api_key: str | None = _ROOT,
    state_dir: Path | None = None,
    **create_kw: Any,
) -> FastAPI:
    """``create_app`` under the ambient env (``_audit_context`` already
    swept ``FX1_*``); backends resolve from ``backend_map[name]``
    zero-arg factories, defaulting to the completion stub."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415
    from fx1.serve.conv_audit import _StubBackend  # noqa: PLC0415

    backends = backend_map or {"byok": lambda: _StubBackend(), "hosted_k3": lambda: _StubBackend()}

    def fake_resolve(name: str, *a: Any, **k: Any) -> Any:
        del a, k
        return backends[name]()

    return api_mod.create_app(
        harness=Harness(runner=runner or _fast_runner),
        backend_resolver=fake_resolve,
        ft_runner=ft_runner,
        state_dir=state_dir,
        **create_kw,
    )


def _client(**kw: Any) -> tuple[TestClient, FastAPI]:
    """(TestClient, FastAPI) for a fresh app; the jobs executor pool is
    shut down on the shared resources stack so probes don't leak threads."""
    from typing import cast as _cast  # noqa: PLC0415

    from fastapi import FastAPI as _FA  # noqa: PLC0415
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from fx1.serve.conv_audit import _RESOURCES  # noqa: PLC0415

    api_key = kw.pop("api_key", _ROOT)
    if api_key is None:
        os.environ.pop(_API_KEY_ENV, None)
    else:
        os.environ[_API_KEY_ENV] = api_key
    app = _make_app(**kw)
    client = TestClient(app, raise_server_exceptions=False)
    stack = _RESOURCES.get(None)
    if stack is not None:
        app_typed = _cast(_FA, app)
        stack.callback(app_typed.state.jobs_executor.shutdown, False, cancel_futures=True)
        stack.callback(client.close)
    return client, app


def _h(auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    out = {"X-API-Key": auth} if auth else {}
    out.update(extra)
    return out


def _ih(key: str, auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    return _h(auth, **{_IDEM: key, **extra})


def _err_code(r: Any) -> str | None:
    """The refusal's machine code — harness shape uses ``code``, the
    OpenAI envelope nests it under ``error.code``, Anthropic under
    ``error.type``-adjacent ``error.code``-style fields."""
    if not r.headers.get("content-type", "").startswith("application/json"):
        return None
    body = r.json()
    if not isinstance(body, dict):
        return None
    code = body.get("code")
    if isinstance(code, str):
        return code
    err = body.get("error")
    if isinstance(err, dict):
        code = err.get("code")
        if isinstance(code, str):
            return code
    return None


def _enveloped(r: Any) -> bool:
    """Every refusal carries a structured error envelope — never a bare
    plaintext 500 (the honesty contract on refusal surfaces)."""
    if not r.headers.get("content-type", "").startswith("application/json"):
        return False
    body = r.json()
    return isinstance(body, dict) and ("detail" in body or "error" in body)


def _replay_hdr(r: Any) -> bool:
    return bool(r.headers.get("x-fx1-idempotent-replay") == "true")


def _wait_for(pred: Callable[[], bool], timeout: float = _WAIT_S) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


class _Hit:
    """One captured webhook delivery: path, headers, raw body bytes."""

    __slots__ = ("body", "headers", "path")

    def __init__(self, path: str, headers: dict[str, str], body: bytes) -> None:
        self.path = path
        self.headers = headers
        self.body = body


class _Sink:
    """A real loopback HTTP webhook sink — ``ThreadingHTTPServer`` on
    ``127.0.0.1:0``; each POST lands as a ``_Hit`` and 200s (``/fail``
    answers 500 so retry bookkeeping can be probed)."""

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
                self.send_response(500 if self.path == "/fail" else 200)
                self.end_headers()

            def log_message(self, *args: Any) -> None:
                del args

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}{path}"

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)


def _wait_hits(sink: _Sink, n: int, timeout: float = _WAIT_S) -> None:
    _wait_for(lambda: len(sink.hits) >= n, timeout)


def _busy_executor(app: FastAPI, sleep_s: float = 1.2) -> None:
    """Park every worker thread on a raw sleeper — outside the inflight
    accounting — so a submitted unit acquires its slot yet stays
    ``queued`` behind the future queue (the deterministic pre-start
    window every queued-cancel probe needs)."""
    executor = app.state.jobs_executor
    release = threading.Event()
    hold = release.wait
    for _ in range(int(getattr(executor, "_max_workers", 4))):  # noqa: SLF001
        executor.submit(lambda: hold(timeout=sleep_s))
    time.sleep(0.1)


def _mint(client: TestClient, **policy: Any) -> tuple[str, str]:
    """Mint a managed key under the root credential → (raw, key_id)."""
    r = client.post("/harness/keys", json=policy, headers=_h())
    assert r.status_code == 201, f"key mint refused: {r.status_code} {r.text}"
    body = r.json()
    return str(body["key"]), str(body["id"])


def _key_card(client: TestClient, key_id: str) -> dict[str, Any]:
    return dict(client.get(f"/harness/keys/{key_id}/usage", headers=_h()).json())


# ---------------------------------------------------------------------------
# Per-surface submit/poll helpers
# ---------------------------------------------------------------------------


def _job_submit(client: TestClient, auth: str | None = _ROOT, **fields: Any) -> Any:
    body: dict[str, Any] = {"command": "doctor", **fields}
    return client.post(_JOBS, json=body, headers=_h(auth))


def _job_record(client: TestClient, job_id: str) -> dict[str, Any]:
    return dict(client.get(f"{_JOBS}/{job_id}", headers=_h()).json())


def _job_wait(client: TestClient, job_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = _job_record(client, job_id)
        if st.get("status") in _JOB_TERMINAL:
            return st
        time.sleep(0.05)
    return st


def _eval_submit(client: TestClient, auth: str | None = _ROOT, **fields: Any) -> Any:
    body = {"suite": "tooluse", "backend": "byok", "seed": 11, **fields}
    return client.post(_EVALS, json=body, headers=_h(auth))


def _eval_record(client: TestClient, eval_id: str) -> dict[str, Any]:
    return dict(client.get(f"{_EVALS}/{eval_id}", headers=_h()).json())


def _eval_wait(client: TestClient, eval_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = _eval_record(client, eval_id)
        if st.get("status") in _JOB_TERMINAL:
            return st
        time.sleep(0.05)
    return st


def _eval_spec(client: TestClient, name: str) -> str:
    r = client.post(
        "/v1/evals",
        json={
            "name": name,
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 11, "backend": "byok"},
            },
            "metadata": {"probe": name},
        },
        headers=_h(),
    )
    assert r.status_code == 201, r.text
    return str(r.json()["id"])


def _eval_run_submit(
    client: TestClient, eval_id: str, auth: str | None = _ROOT, **fields: Any
) -> Any:
    body = {"model": "byok", **fields}
    return client.post(f"/v1/evals/{eval_id}/runs", json=body, headers=_h(auth))


def _eval_run_record(client: TestClient, eval_id: str, run_id: str) -> dict[str, Any]:
    return dict(client.get(f"/v1/evals/{eval_id}/runs/{run_id}", headers=_h()).json())


def _eval_run_wait(
    client: TestClient, eval_id: str, run_id: str, timeout: float = _WAIT_S
) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = _eval_run_record(client, eval_id, run_id)
        if st.get("status") in {"completed", "failed", "canceled"}:
            return st
        time.sleep(0.05)
    return st


def _resp_submit(
    client: TestClient, auth: str | None = _ROOT, *, key: str | None = None, **body: Any
) -> Any:
    headers = _ih(key, auth) if key is not None else _h(auth)
    payload = {"model": "byok", "input": "hi", "background": True, "store": True, **body}
    return client.post("/v1/responses", json=payload, headers=headers)


def _resp_get(client: TestClient, rid: str, auth: str | None = _ROOT) -> Any:
    return client.get(f"/v1/responses/{rid}", headers=_h(auth))


def _resp_wait(client: TestClient, rid: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        r = _resp_get(client, rid)
        if r.status_code == 200:
            st = r.json()
            if st.get("status") in _RESP_TERMINAL:
                return st
        time.sleep(0.05)
    return st


def _upload_lines(client: TestClient, lines: list[dict[str, Any]], auth: str | None = _ROOT) -> str:
    blob = "".join(json.dumps(ln) + "\n" for ln in lines).encode()
    r = client.post(
        "/v1/files",
        files={"file": ("in.jsonl", blob, "application/jsonl")},
        data={"purpose": "batch"},
        headers=_h(auth),
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _line(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


def _chat_body(content: str = "hi") -> dict[str, Any]:
    return {"model": "hosted_k3", "messages": [{"role": "user", "content": content}]}


def _batch_create(
    client: TestClient, lines: list[dict[str, Any]], auth: str | None = _ROOT, **extra: Any
) -> dict[str, Any]:
    fid = _upload_lines(client, lines, auth)
    body: dict[str, Any] = {
        "input_file_id": fid,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        **extra,
    }
    r = client.post("/v1/batches", json=body, headers=_h(auth))
    assert r.status_code == 200, f"batch submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _batch_record(client: TestClient, batch_id: str, auth: str | None = _ROOT) -> dict[str, Any]:
    return dict(client.get(f"/v1/batches/{batch_id}", headers=_h(auth)).json())


def _batch_wait(
    client: TestClient, batch_id: str, timeout: float = _WAIT_S, auth: str | None = _ROOT
) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = _batch_record(client, batch_id, auth)
        if st.get("status") in _BATCH_TERMINAL:
            return st
        time.sleep(0.05)
    return st


def _abatch_item(custom_id: str, content: str = "ping") -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "params": {
            "model": "hosted_k3",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": content}],
        },
    }


def _abatch_create(
    client: TestClient,
    items: list[dict[str, Any]],
    auth: str | None = _ROOT,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {"requests": items, **extra}
    r = client.post("/v1/messages/batches", json=body, headers={**_h(auth), **_ANTHROPIC_H})
    assert r.status_code == 200, f"abatch submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _abatch_record(client: TestClient, batch_id: str) -> dict[str, Any]:
    return dict(
        client.get(f"/v1/messages/batches/{batch_id}", headers={**_h(), **_ANTHROPIC_H}).json()
    )


def _abatch_wait(client: TestClient, batch_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = _abatch_record(client, batch_id)
        if st.get("status") == "ended":
            return st
        time.sleep(0.05)
    return st


def _output_rows(client: TestClient, file_id: str) -> list[dict[str, Any]]:
    r = client.get(f"/v1/files/{file_id}/content", headers=_h())
    assert r.status_code == 200, r.text
    return [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]


def _abatch_rows(client: TestClient, batch_id: str) -> list[dict[str, Any]]:
    r = client.get(f"/v1/messages/batches/{batch_id}/results", headers={**_h(), **_ANTHROPIC_H})
    assert r.status_code == 200, r.text
    return [json.loads(ln) for ln in r.text.splitlines() if ln.strip()]


def _upload_intent(client: TestClient, auth: str | None = _ROOT, **extra: Any) -> dict[str, Any]:
    body = {
        "purpose": "batch",
        "filename": "intent.jsonl",
        "bytes": 16,
        "mime_type": "application/jsonl",
        **extra,
    }
    r = client.post("/v1/uploads", json=body, headers=_h(auth))
    assert r.status_code == 200, r.text
    return dict(r.json())


def _upload_part(
    client: TestClient,
    upload_id: str,
    blob: bytes = b'{"a":1}\n{"b":2}\n',
    auth: str | None = _ROOT,
) -> Any:
    return client.post(
        f"/v1/uploads/{upload_id}/parts",
        files={"data": ("p0", blob, "application/jsonl")},
        headers=_h(auth),
    )


def _ft_file(client: TestClient, auth: str | None = _ROOT) -> str:
    r = client.post(
        "/v1/files",
        files={"file": ("ft.jsonl", _FT_CORPUS, "application/jsonl")},
        data={"purpose": "fine-tune"},
        headers=_h(auth),
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _ft_submit(client: TestClient, auth: str | None = _ROOT, **extra: Any) -> dict[str, Any]:
    body = {
        "model": "fx1",
        "training_file": _ft_file(client, auth),
        "suffix": "m",
        **extra,
    }
    r = client.post("/v1/fine_tuning/jobs", json=body, headers=_h(auth))
    assert r.status_code in (200, 201), f"ft submit refused: {r.status_code} {r.text}"
    return dict(r.json())


def _ft_record(client: TestClient, job_id: str, auth: str | None = _ROOT) -> dict[str, Any]:
    return dict(client.get(f"/v1/fine_tuning/jobs/{job_id}", headers=_h(auth)).json())


def _ft_wait(client: TestClient, job_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    end = time.monotonic() + timeout
    st: dict[str, Any] = {}
    while time.monotonic() < end:
        st = _ft_record(client, job_id)
        if st.get("status") in _FT_TERMINAL:
            return st
        time.sleep(0.05)
    return st


def _ft_events(client: TestClient, job_id: str) -> list[dict[str, Any]]:
    r = client.get(f"/v1/fine_tuning/jobs/{job_id}/events", headers=_h())
    assert r.status_code == 200, r.text
    return list(r.json()["data"])


class _LineGate:
    """Per-line parking for the batch worker: every ``complete`` waits on
    a semaphore, so a probe releases exactly the lines it wants through
    and cancels with the worker parked inside the next line — the
    deterministic mid-flight window ``_GateBackend``'s shared event
    can't give. ``last_usage`` is a fixed usage claim so partial
    billing checks are arithmetic, not estimation."""

    def __init__(self) -> None:
        self.sem = threading.Semaphore(0)
        self.entered = threading.Event()
        self.calls = 0
        self.last_usage = dict(_U)

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        self.calls += 1
        self.entered.set()
        self.sem.acquire(timeout=30)
        return f"parked line {messages[-1]['content']}"

    def close(self) -> None:
        pass


class _FTGate:
    """Pausable fine-tune runner stub: ``entered`` marks the job inside
    the pipeline, ``release`` lets it finish, ``pause_gate`` is invoked
    at one stage boundary so a mid-flight pause/cancel resolves
    deterministically."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stage2 = threading.Event()
        self.gate_calls = 0
        self.cancel_seen = False

    def __call__(
        self,
        spec: Any,
        *,
        emit: Callable[[str, str, dict[str, Any] | None], None],
        should_cancel: Callable[[], bool],
        pause_gate: Callable[[], bool],
    ) -> Any:
        from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

        emit("info", "stage one", None)
        self.entered.set()
        self.release.wait(30)
        self.gate_calls += 1
        if should_cancel() or pause_gate():
            self.cancel_seen = True
            return FTJobOutcome()
        self.stage2.wait(30)
        if should_cancel():
            self.cancel_seen = True
            return FTJobOutcome()
        ckpt = spec.work_dir / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            checkpoint=str(ckpt),
            trained_tokens=7,
        )


def _conv_create(client: TestClient) -> str:
    r = client.post("/v1/conversations", json={}, headers=_h())
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _conv_items(client: TestClient, cid: str) -> list[dict[str, Any]]:
    r = client.get(f"/v1/conversations/{cid}/items", headers=_h())
    assert r.status_code == 200, r.text
    return list(r.json()["data"])


# ---------------------------------------------------------------------------
# Probe sections
# ---------------------------------------------------------------------------


def _probe_jobs_cancel() -> dict[str, bool]:
    """``DELETE /harness/jobs/{id}`` — queued flips to cancelled and never
    runs; running/terminal refuse 409 honestly; re-cancel is idempotent;
    the cancelled terminal fires the signed webhook once."""
    from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

    out: dict[str, bool] = {}
    runner = _Runner()
    client, app = _client(runner=runner)
    _busy_executor(app)
    r = _job_submit(client)
    job_id = r.json()["job_id"]
    out["jobs.submit_202"] = r.status_code == 202

    c = client.delete(f"{_JOBS}/{job_id}", headers=_h())
    out["jobs.queued_cancel_200"] = c.status_code == 200
    body = c.json()
    out["jobs.queued_cancel_status"] = body["status"] == "cancelled"
    out["jobs.queued_cancel_finished_at"] = (
        isinstance(body["finished_at"], (int, float)) and body["finished_at"] >= body["created_at"]
    )
    # the queued-cancelled future never runs even after the executor frees
    time.sleep(1.5)
    _job_wait(client, job_id)
    out["jobs.queued_cancel_never_runs"] = runner.calls == []

    # running refuses — no mid-run kill handle
    gate = threading.Event()
    runner2 = _Runner(gate)
    client2, _app2 = _client(runner=runner2)
    r2 = _job_submit(client2)
    jid2 = r2.json()["job_id"]
    assert runner2.entered.wait(10), "runner never entered"
    c2 = client2.delete(f"{_JOBS}/{jid2}", headers=_h())
    out["jobs.running_cancel_409"] = c2.status_code == 409
    out["jobs.running_cancel_envelope"] = _enveloped(c2)
    out["jobs.running_still_running"] = _job_record(client2, jid2)["status"] == "running"
    gate.set()
    out["jobs.running_completes_after_cancel"] = _job_wait(client2, jid2)["status"] == "succeeded"

    c3 = client2.delete(f"{_JOBS}/{jid2}", headers=_h())
    out["jobs.terminal_cancel_409"] = c3.status_code == 409
    out["jobs.terminal_cancel_envelope"] = _enveloped(c3)
    out["jobs.terminal_message_names_state"] = "succeeded" in c3.json()["detail"]

    # failed refuses identically — a runner exception is the 'failed'
    # outcome (a nonzero exit code records it but still 'succeeded':
    # the run itself completed)
    runner3 = _Runner(raise_exc=RuntimeError("synthetic runner down"))
    client3, _a3 = _client(runner=runner3)
    jid3 = _job_submit(client3).json()["job_id"]
    _job_wait(client3, jid3)
    c4 = client3.delete(f"{_JOBS}/{jid3}", headers=_h())
    out["jobs.failed_cancel_409"] = c4.status_code == 409
    out["jobs.failed_message_names_state"] = "failed" in c4.json()["detail"]

    # cancelled re-cancel: outcome is "cancelled" → 200 idempotent, and the
    # terminal webhook still fired exactly once for the first cancel
    sink = _Sink()
    try:
        client4, app4 = _client()
        _busy_executor(app4)
        jid4 = _job_submit(client4, callback_url=sink.url("/hook"), callback_secret="cb-s").json()[
            "job_id"
        ]
        c5 = client4.delete(f"{_JOBS}/{jid4}", headers=_h())
        c6 = client4.delete(f"{_JOBS}/{jid4}", headers=_h())
        _wait_hits(sink, 1)
        hit = sink.hits[0] if sink.hits else None
        out["jobs.recancel_idempotent_200"] = (
            c5.status_code == 200 and c6.status_code == 200 and c6.json()["status"] == "cancelled"
        )
        payload = hit.body if hit else b""
        out["jobs.cancel_webhook_fires"] = hit is not None and b'"cancelled"' in payload
        out["jobs.cancel_webhook_signed"] = (
            hit is not None
            and verify_webhook(
                "cb-s",
                hit.headers.get("X-Fx1-Webhook-Timestamp"),
                hit.headers.get("X-Fx1-Webhook-Signature"),
                hit.body,
            )
            and b"cb-s" not in hit.body
        )
        time.sleep(0.4)
        out["jobs.cancel_webhook_fire_once"] = len(sink.hits) == 1
    finally:
        sink.close()

    missing = client4.delete(f"{_JOBS}/job_missing", headers=_h())
    out["jobs.cancel_missing_404"] = missing.status_code == 404
    out["jobs.cancel_missing_envelope"] = _enveloped(missing)
    return out


def _probe_evals_cancel() -> dict[str, bool]:
    """``DELETE /harness/evals/{id}`` — same store contract as jobs:
    queued→cancelled (worker never resolves a backend), running/terminal
    refuse 409, re-cancel is idempotent 200, webhook fires once."""
    from fx1.serve.conv_audit import _StubBackend  # noqa: PLC0415
    from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

    out: dict[str, bool] = {}
    backend = _StubBackend()
    client, app = _client(backend_map={"byok": lambda: backend})
    _busy_executor(app)
    eval_id = _eval_submit(client).json()["eval_id"]
    c = client.delete(f"{_EVALS}/{eval_id}", headers=_h())
    out["evals.queued_cancel_200"] = c.status_code == 200
    out["evals.queued_cancel_status"] = c.json()["status"] == "cancelled"
    time.sleep(1.5)
    rec = _eval_wait(client, eval_id)
    out["evals.queued_cancel_terminal"] = rec["status"] == "cancelled"
    out["evals.queued_cancel_never_runs"] = backend.calls == 0

    # terminal refuses
    sub2 = _eval_submit(client).json()["eval_id"]
    rec2 = _eval_wait(client, sub2)
    out["evals.succeeds_when_not_cancelled"] = rec2["status"] == "succeeded"
    c2 = client.delete(f"{_EVALS}/{sub2}", headers=_h())
    out["evals.terminal_cancel_409"] = c2.status_code == 409
    out["evals.terminal_cancel_envelope"] = _enveloped(c2)

    # running refuses mid-flight — gate backend pins the window
    from fx1.serve.conv_audit import _GateBackend  # noqa: PLC0415

    gate = _GateBackend()
    client2, _a = _client(backend_map={"byok": lambda: gate})
    eval3 = _eval_submit(client2).json()["eval_id"]
    deadline = time.monotonic() + 10.0
    while _eval_record(client2, eval3).get("status") != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    c3 = client2.delete(f"{_EVALS}/{eval3}", headers=_h())
    out["evals.running_cancel_409"] = c3.status_code == 409
    out["evals.running_cancel_envelope"] = _enveloped(c3)
    gate.gate.set()
    out["evals.running_completes"] = _eval_wait(client2, eval3)["status"] == "succeeded"

    # re-cancel idempotent + webhook once (signed)
    sink = _Sink()
    try:
        client3, app3 = _client(backend_map={"byok": lambda: _StubBackend()})
        _busy_executor(app3)
        eval4 = _eval_submit(
            client3, callback_url=sink.url("/hook"), callback_secret="ev-s"
        ).json()["eval_id"]
        c5 = client3.delete(f"{_EVALS}/{eval4}", headers=_h())
        c6 = client3.delete(f"{_EVALS}/{eval4}", headers=_h())
        _wait_hits(sink, 1)
        hit = sink.hits[0] if sink.hits else None
        out["evals.recancel_idempotent_200"] = c5.status_code == 200 and c6.status_code == 200
        out["evals.cancel_webhook_fires"] = hit is not None and b'"cancelled"' in hit.body
        out["evals.cancel_webhook_signed"] = hit is not None and verify_webhook(
            "ev-s",
            hit.headers.get("X-Fx1-Webhook-Timestamp"),
            hit.headers.get("X-Fx1-Webhook-Signature"),
            hit.body,
        )
        time.sleep(0.4)
        out["evals.cancel_webhook_fire_once"] = len(sink.hits) == 1
        # a cancelled eval's receipt still exports — the record is evidence
        r = client3.get(f"{_EVALS}/{eval4}/receipt", headers=_h())
        out["evals.cancelled_receipt_exports"] = r.status_code == 200
    finally:
        sink.close()

    missing = client3.delete(f"{_EVALS}/missing", headers=_h())
    out["evals.cancel_missing_404"] = missing.status_code == 404
    out["evals.cancel_missing_envelope"] = _enveloped(missing)
    return out


def _probe_eval_runs_cancel() -> dict[str, bool]:
    """``POST /v1/evals/{eid}/runs/{rid}/cancel`` — queued runs cancel
    (``canceled`` on the wire, output_items frozen empty,
    ``error.code=eval_run_canceled``); running/completed refuse 409;
    re-cancel is idempotent 200 and the run's webhook fires once."""
    from fx1.serve.conv_audit import _GateBackend, _StubBackend  # noqa: PLC0415

    out: dict[str, bool] = {}
    client, app = _client(backend_map={"byok": lambda: _StubBackend()})
    spec_id = _eval_spec(client, "queued-run")
    _busy_executor(app)
    sub = _eval_run_submit(client, spec_id)
    out["evalruns.submit_201"] = sub.status_code == 201
    run_id = sub.json()["id"]
    c = client.post(f"/v1/evals/{spec_id}/runs/{run_id}/cancel", headers=_h())
    out["evalruns.queued_cancel_200"] = c.status_code == 200
    out["evalruns.queued_cancel_wire_canceled"] = c.json()["status"] == "canceled"
    rec = _eval_run_wait(client, spec_id, run_id)
    out["evalruns.queued_cancel_terminal"] = rec["status"] == "canceled"
    out["evalruns.cancel_error_code"] = rec.get("error", {}).get("code") == "eval_run_canceled"
    items = client.get(f"/v1/evals/{spec_id}/runs/{run_id}/output_items", headers=_h())
    out["evalruns.cancelled_output_items_empty"] = (
        items.status_code == 200 and items.json()["data"] == []
    )

    # running run refuses — gate the backend so the run is mid-flight
    gate = _GateBackend()
    client2, _a = _client(backend_map={"byok": lambda: gate})
    spec2 = _eval_spec(client2, "running-run")
    run2 = _eval_run_submit(client2, spec2).json()["id"]
    deadline = time.monotonic() + 10.0
    while (
        _eval_run_record(client2, spec2, run2).get("status") != "in_progress"
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    c2 = client2.post(f"/v1/evals/{spec2}/runs/{run2}/cancel", headers=_h())
    out["evalruns.running_cancel_409"] = c2.status_code == 409
    out["evalruns.running_cancel_envelope"] = _enveloped(c2)
    gate.gate.set()
    _eval_run_wait(client2, spec2, run2)
    c3 = client2.post(f"/v1/evals/{spec2}/runs/{run2}/cancel", headers=_h())
    out["evalruns.completed_cancel_409"] = c3.status_code == 409
    out["evalruns.completed_cancel_envelope"] = _enveloped(c3)

    # re-cancel idempotent + run webhook fires on cancel
    sink = _Sink()
    try:
        client3, app3 = _client(backend_map={"byok": lambda: _StubBackend()})
        spec3 = _eval_spec(client3, "idem-run")
        _busy_executor(app3)
        run3 = _eval_run_submit(
            client3, spec3, callback_url=sink.url("/hook"), callback_secret="run-s"
        ).json()["id"]
        c5 = client3.post(f"/v1/evals/{spec3}/runs/{run3}/cancel", headers=_h())
        c6 = client3.post(f"/v1/evals/{spec3}/runs/{run3}/cancel", headers=_h())
        _wait_hits(sink, 1)
        out["evalruns.recancel_idempotent_200"] = c5.status_code == 200 and c6.status_code == 200
        out["evalruns.cancel_webhook_fires"] = (
            bool(sink.hits) and b'"cancelled"' in sink.hits[0].body
        )
        time.sleep(0.4)
        out["evalruns.cancel_webhook_fire_once"] = len(sink.hits) == 1
    finally:
        sink.close()

    missing = client3.post(f"/v1/evals/{spec3}/runs/evalrun_nope/cancel", headers=_h())
    out["evalruns.cancel_missing_404"] = missing.status_code == 404
    out["evalruns.cancel_missing_envelope"] = _enveloped(missing)
    return out


def _probe_batch_cancel() -> dict[str, bool]:
    """``POST /v1/batches/{id}/cancel`` — the two-phase cancel: ``queued``/
    ``validating``/``in_progress`` → ``cancelling`` → ``cancelled``;
    terminal refuses 409 ``batch_terminal``; re-cancel while
    ``cancelling`` replays 200; output file mints only for lines that
    ran; signed webhook fires once."""
    from fx1.serve.conv_audit import _GateBackend  # noqa: PLC0415
    from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

    out: dict[str, bool] = {}
    client, app = _client()
    _busy_executor(app)
    lines = [_line(f"c{i}", _chat_body(f"m{i}")) for i in range(3)]
    b = _batch_create(client, lines)
    bid = b["id"]
    c = client.post(f"/v1/batches/{bid}/cancel", headers=_h())
    out["batch.queued_cancel_200"] = c.status_code == 200
    out["batch.queued_cancel_cancelling"] = c.json()["status"] == "cancelling"
    out["batch.cancelling_at_stamped"] = c.json()["cancelling_at"] is not None
    rec = _batch_wait(client, bid)
    out["batch.queued_cancel_terminal"] = rec["status"] == "cancelled"
    out["batch.cancelled_at_stamped"] = rec["cancelled_at"] is not None
    out["batch.queued_cancel_no_output"] = (
        rec["output_file_id"] is None and rec["error_file_id"] is None
    )
    # nothing ran — counts report the declared input size with zero
    # completed/failed
    out["batch.queued_cancel_counts_zero"] = (
        rec["request_counts"]["completed"] == 0 and rec["request_counts"]["failed"] == 0
    )
    # mid-flight: gate the backend inside line 1, cancel lands mid-flight
    gate = _GateBackend()
    client2, _a = _client(backend_map={"hosted_k3": lambda: gate})
    b2 = _batch_create(client2, lines)
    bid2 = b2["id"]
    assert gate.entered.wait(10), "batch worker never entered the backend"
    c2 = client2.post(f"/v1/batches/{bid2}/cancel", headers=_h())
    out["batch.midflight_cancel_200"] = c2.status_code == 200
    out["batch.midflight_cancel_cancelling"] = c2.json()["status"] == "cancelling"
    # re-cancel while cancelling → idempotent 200 re-read
    c3 = client2.post(f"/v1/batches/{bid2}/cancel", headers=_h())
    out["batch.cancelling_recancel_200"] = c3.status_code == 200 and c3.json()["id"] == bid2
    gate.gate.set()
    rec2 = _batch_wait(client2, bid2)
    out["batch.midflight_cancel_terminal"] = rec2["status"] == "cancelled"
    out["batch.midflight_partial_output"] = rec2["output_file_id"] is not None
    if rec2["output_file_id"] is not None:
        rows = _output_rows(client2, rec2["output_file_id"])
        out["batch.midflight_output_rows_completed_only"] = 0 < len(rows) < len(lines)
        out["batch.midflight_counts_honest"] = rec2["request_counts"]["total"] == len(
            lines
        ) and rec2["request_counts"]["completed"] == len(rows)
    # cancelled record persists and refuses further cancels
    out["batch.cancelled_persists_get"] = _batch_record(client2, bid2)["status"] == "cancelled"
    c4 = client2.post(f"/v1/batches/{bid2}/cancel", headers=_h())
    out["batch.cancelled_recancel_409"] = c4.status_code == 409
    out["batch.cancelled_recancel_code"] = _err_code(c4) == "batch_terminal"

    # terminal refuses — completed and failed both
    b3 = _batch_create(client2, lines[:1])
    _batch_wait(client2, b3["id"])
    c5 = client2.post(f"/v1/batches/{b3['id']}/cancel", headers=_h())
    out["batch.completed_cancel_409"] = c5.status_code == 409
    out["batch.completed_cancel_code"] = _err_code(c5) == "batch_terminal"
    out["batch.completed_cancel_envelope"] = _enveloped(c5)

    # failed refuses — cap the file store so the output-file write
    # itself fails the batch (the only client-reachable 'failed' window:
    # per-line faults isolate into rows, they never fail the batch)
    client3, app3 = _client(file_bytes_max=256)
    blob_in = (json.dumps(lines[0]) + "\n").encode()
    rec_in = app3.state.file_store.put(filename="in.jsonl", purpose="batch", content=blob_in)
    r4 = client3.post(
        "/v1/batches",
        json={
            "input_file_id": rec_in.file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        headers=_h(),
    )
    bid4 = r4.json()["id"]
    rec4 = _batch_wait(client3, bid4)
    c6 = client3.post(f"/v1/batches/{bid4}/cancel", headers=_h())
    out["batch.failed_status_reached"] = rec4["status"] == "failed"
    out["batch.failed_cancel_409"] = c6.status_code == 409
    out["batch.failed_cancel_code"] = _err_code(c6) == "batch_terminal"

    # webhook: cancelled terminal fires the signed hook exactly once
    sink = _Sink()
    try:
        client4, app4 = _client()
        _busy_executor(app4)
        b5 = _batch_create(
            client4,
            lines[:1],
            callback_url=sink.url("/hook"),
            callback_secret="bat-s",
        )
        bid5 = b5["id"]
        client4.post(f"/v1/batches/{bid5}/cancel", headers=_h())
        _batch_wait(client4, bid5)
        _wait_hits(sink, 1)
        hit = sink.hits[0] if sink.hits else None
        out["batch.cancel_webhook_fires"] = hit is not None and b'"cancelled"' in hit.body
        out["batch.cancel_webhook_signed"] = hit is not None and verify_webhook(
            "bat-s",
            hit.headers.get("X-Fx1-Webhook-Timestamp"),
            hit.headers.get("X-Fx1-Webhook-Signature"),
            hit.body,
        )
        out["batch.cancel_webhook_delivered_bookkeep"] = (
            _batch_wait(client4, bid5).get("callback_status") == "delivered"
        )
        time.sleep(0.4)
        out["batch.cancel_webhook_fire_once"] = len(sink.hits) == 1
    finally:
        sink.close()

    missing = client4.post("/v1/batches/batch_nope/cancel", headers=_h())
    out["batch.cancel_missing_404"] = missing.status_code == 404
    out["batch.cancel_missing_code"] = _err_code(missing) == "batch_not_found"
    return out


def _probe_abatch_cancel() -> dict[str, bool]:
    """``POST /v1/messages/batches/{id}/cancel`` — Anthropic's two-phase
    cancel: ``in_progress → canceling → ended`` with ``canceled`` result
    rows; ``ended`` refuses 400 ``invalid_request``; re-cancel while
    ``canceling`` replays 200; delete stays refused until ``ended``."""
    from fx1.serve.conv_audit import _GateBackend, _StubBackend  # noqa: PLC0415

    out: dict[str, bool] = {}
    gate = _GateBackend()
    client, _a = _client(backend_map={"hosted_k3": lambda: gate})
    items = [_abatch_item(f"i{i}") for i in range(3)]
    b = _abatch_create(client, items)
    bid = b["id"]
    assert gate.entered.wait(10), "abatch worker never entered the backend"
    c = client.post(f"/v1/messages/batches/{bid}/cancel", headers={**_h(), **_ANTHROPIC_H})
    out["abatch.inflight_cancel_200"] = c.status_code == 200
    out["abatch.cancel_wire_canceling"] = c.json()["processing_status"] == "canceling"
    out["abatch.cancel_initiated_stamped"] = c.json()["cancel_initiated_at"] is not None
    # re-cancel while canceling → idempotent 200
    c2 = client.post(f"/v1/messages/batches/{bid}/cancel", headers={**_h(), **_ANTHROPIC_H})
    out["abatch.canceling_recancel_200"] = c2.status_code == 200 and c2.json()["id"] == bid
    # delete while still in-flight refuses honestly
    d0 = client.delete(f"/v1/messages/batches/{bid}", headers={**_h(), **_ANTHROPIC_H})
    out["abatch.delete_while_canceling_400"] = d0.status_code == 400
    gate.gate.set()
    rec = _abatch_wait(client, bid)
    out["abatch.cancel_terminal_ended"] = rec["processing_status"] == "ended"
    rows = _abatch_rows(client, bid)
    out["abatch.cancel_rows_all_present"] = len(rows) == len(items)
    out["abatch.cancel_tail_canceled_rows"] = any(
        row.get("result", {}).get("type") == "canceled" for row in rows
    )
    out["abatch.cancel_counts_tally"] = rec["request_counts"]["processing"] == 0 and (
        rec["request_counts"]["succeeded"]
        + rec["request_counts"]["canceled"]
        + rec["request_counts"]["errored"]
        == len(items)
    )
    # ended refuses — Anthropic's contract is 400 with the Anthropic
    # error envelope (``type`` grammar, not the OpenAI ``code``)
    c3 = client.post(f"/v1/messages/batches/{bid}/cancel", headers={**_h(), **_ANTHROPIC_H})
    out["abatch.ended_cancel_400"] = c3.status_code == 400
    out["abatch.ended_cancel_type"] = c3.json().get("error", {}).get("type") == (
        "invalid_request_error"
    )
    out["abatch.ended_cancel_envelope"] = _enveloped(c3)
    # delete works once ended
    d = client.delete(f"/v1/messages/batches/{bid}", headers={**_h(), **_ANTHROPIC_H})
    out["abatch.delete_after_ended_200"] = d.status_code == 200
    g = client.get(f"/v1/messages/batches/{bid}", headers={**_h(), **_ANTHROPIC_H})
    out["abatch.deleted_get_404"] = g.status_code == 404

    # a cancelled-never-started abatch (busy executor → cancel) ends with
    # every row canceled and zero completions
    client2, app2 = _client(backend_map={"hosted_k3": lambda: _StubBackend()})
    _busy_executor(app2)
    b2 = _abatch_create(client2, items)
    bid2 = b2["id"]
    c4 = client2.post(f"/v1/messages/batches/{bid2}/cancel", headers={**_h(), **_ANTHROPIC_H})
    rec2 = _abatch_wait(client2, bid2)
    rows2 = _abatch_rows(client2, bid2)
    out["abatch.queued_cancel_200"] = c4.status_code == 200
    out["abatch.queued_cancel_ended"] = rec2["processing_status"] == "ended"
    out["abatch.queued_cancel_all_canceled"] = all(
        row.get("result", {}).get("type") == "canceled" for row in rows2
    ) and len(rows2) == len(items)

    missing = client2.post("/v1/messages/batches/nope/cancel", headers={**_h(), **_ANTHROPIC_H})
    out["abatch.cancel_missing_404"] = missing.status_code == 404
    out["abatch.cancel_missing_envelope"] = _enveloped(missing)
    return out


def _probe_response_cancel() -> dict[str, bool]:
    """``POST /v1/responses/{id}/cancel`` — the CAS surface: queued and
    in_progress flip to ``cancelled``; every terminal status refuses 409
    ``cancel_terminal`` (cancelled included — honestly refused, not
    idempotent); the worker's late result is dropped by
    ``put_unless_status`` so a cancelled record never flips to
    completed; delete-after-cancel 404s; a cancelled response never
    appends to its conversation; and the keyed replay can never
    resurrect a false verdict (the lane's fix)."""
    from fx1.serve.conv_audit import _GateBackend, _StubBackend  # noqa: PLC0415

    out: dict[str, bool] = {}
    backend = _StubBackend()
    client, app = _client(backend_map={"byok": lambda: backend})
    _busy_executor(app)
    rid = _resp_submit(client).json()["id"]
    c = client.post(f"/v1/responses/{rid}/cancel", headers=_h())
    out["resp.queued_cancel_200"] = c.status_code == 200
    out["resp.queued_cancel_status"] = c.json()["status"] == "cancelled"
    time.sleep(1.5)
    rec = _resp_wait(client, rid)
    out["resp.queued_cancel_terminal"] = rec["status"] == "cancelled"
    out["resp.queued_cancel_never_runs"] = backend.calls == 0
    out["resp.cancelled_persists_get"] = _resp_get(client, rid).json()["status"] == "cancelled"

    # re-cancel refuses honestly — cancelled is terminal here (no
    # idempotent replay on this surface)
    c2 = client.post(f"/v1/responses/{rid}/cancel", headers=_h())
    out["resp.cancelled_recancel_409"] = c2.status_code == 409
    out["resp.cancelled_recancel_code"] = _err_code(c2) == "cancel_terminal"
    out["resp.cancelled_recancel_envelope"] = _enveloped(c2)

    # mid-flight: gate the backend, cancel while the worker is inside the
    # call — the late completion is dropped (CAS), GET stays cancelled
    gate = _GateBackend()
    client2, _a = _client(backend_map={"byok": lambda: gate})
    rid2 = _resp_submit(client2).json()["id"]
    assert gate.entered.wait(10), "response worker never entered the backend"
    c3 = client2.post(f"/v1/responses/{rid2}/cancel", headers=_h())
    out["resp.midflight_cancel_200"] = c3.status_code == 200
    out["resp.midflight_cancel_status"] = c3.json()["status"] == "cancelled"
    gate.gate.set()
    time.sleep(0.6)
    rec2 = _resp_wait(client2, rid2)
    out["resp.midflight_stays_cancelled"] = rec2["status"] == "cancelled"
    out["resp.midflight_no_succeeded_resurrect"] = (
        _resp_get(client2, rid2).json()["status"] == "cancelled"
    )

    # terminal refuses — completed (incomplete/failed share the terminal set)
    rid3 = _resp_submit(client2).json()["id"]
    _resp_wait(client2, rid3)
    c4 = client2.post(f"/v1/responses/{rid3}/cancel", headers=_h())
    out["resp.completed_cancel_409"] = c4.status_code == 409
    out["resp.completed_cancel_code"] = _err_code(c4) == "cancel_terminal"
    out["resp.completed_cancel_envelope"] = _enveloped(c4)

    class _FailBackend(_StubBackend):
        def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
            raise RuntimeError("synthetic backend down")

    client3, _a3 = _client(backend_map={"byok": lambda: _FailBackend()})
    rid4 = _resp_submit(client3).json()["id"]
    _resp_wait(client3, rid4)
    c5 = client3.post(f"/v1/responses/{rid4}/cancel", headers=_h())
    out["resp.failed_cancel_409"] = c5.status_code == 409
    out["resp.failed_cancel_envelope"] = _enveloped(c5)

    # cancel + storage: cancelled record deleted → 404 honestly
    d = client3.delete(f"/v1/responses/{rid4}", headers=_h())
    g = client3.get(f"/v1/responses/{rid4}", headers=_h())
    out["resp.cancelled_delete_200"] = d.status_code == 200
    out["resp.cancelled_deleted_get_404"] = g.status_code == 404

    # cancel + conv folding: a cancelled turn never appends to the conv
    gate2 = _GateBackend()
    client4, _a4 = _client(backend_map={"byok": lambda: gate2})
    cid = _conv_create(client4)
    rid5 = _resp_submit(client4, conversation=cid).json()["id"]
    assert gate2.entered.wait(10), "conv-bound response never entered the backend"
    client4.post(f"/v1/responses/{rid5}/cancel", headers=_h())
    gate2.gate.set()
    _resp_wait(client4, rid5)
    out["resp.cancel_conv_no_append"] = _conv_items(client4, cid) == []

    # cancel + idem: the keyed replay can never resurrect a false verdict.
    # queued-cancel + delete + worker sync + replay → the cancelled verdict
    # is pinned into the idem cache, so the replay rehydrates 'cancelled'
    # (never a zombie 'queued' — the lane's _sync_idem_verdict + repin fix)
    client5, app5 = _client(backend_map={"byok": lambda: _StubBackend()})
    _busy_executor(app5)
    body5 = {"model": "byok", "input": "hi", "background": True, "store": True}
    r5 = client5.post("/v1/responses", json=body5, headers=_ih("ca-k1"))
    rid6 = r5.json()["id"]
    client5.post(f"/v1/responses/{rid6}/cancel", headers=_h())
    client5.delete(f"/v1/responses/{rid6}", headers=_h())
    time.sleep(1.6)  # the sleeper queue drains; the worker pins the verdict
    rr = client5.post("/v1/responses", json=body5, headers=_ih("ca-k1"))
    g6 = client5.get(f"/v1/responses/{rid6}", headers=_h())
    out["resp.cancel_delete_replay_no_zombie"] = (
        rr.status_code == 200
        and _replay_hdr(rr)
        and rr.json()["status"] == "cancelled"
        and g6.status_code == 200
        and g6.json()["status"] == "cancelled"
    )

    # mid-flight cancel + delete + keyed replay → the cancelled verdict is
    # preserved, never the rejected 'completed' the cache would have held
    gate3 = _GateBackend()
    client6, _a6 = _client(backend_map={"byok": lambda: gate3})
    body6 = {"model": "byok", "input": "hi", "background": True, "store": True}
    r6 = client6.post("/v1/responses", json=body6, headers=_ih("ca-k2"))
    rid7 = r6.json()["id"]
    assert gate3.entered.wait(10), "keyed response never entered the backend"
    client6.post(f"/v1/responses/{rid7}/cancel", headers=_h())
    gate3.gate.set()
    _resp_wait(client6, rid7)
    client6.delete(f"/v1/responses/{rid7}", headers=_h())
    rr2 = client6.post("/v1/responses", json=body6, headers=_ih("ca-k2"))
    g7 = client6.get(f"/v1/responses/{rid7}", headers=_h())
    out["resp.cancel_idem_replay_keeps_verdict"] = (
        rr2.status_code == 200
        and _replay_hdr(rr2)
        and rr2.json()["status"] == "cancelled"
        and g7.status_code == 200
        and g7.json()["status"] == "cancelled"
    )

    missing = client6.post("/v1/responses/resp_nope/cancel", headers=_h())
    out["resp.cancel_missing_404"] = missing.status_code == 404
    out["resp.cancel_missing_envelope"] = _enveloped(missing)
    return out


def _probe_upload_cancel() -> dict[str, bool]:
    """``POST /v1/uploads/{id}/cancel`` — pending intents cancel and drop
    their parts; completed refuses 409 ``upload_terminal``; already-
    cancelled replays 200 honestly; a keyed cancel dedupes under
    ``X-Fx1-Idempotent-Replay``; a cancelled intent refuses parts and
    completions forever."""
    out: dict[str, bool] = {}
    client, _a = _client()
    u = _upload_intent(client)
    uid = u["id"]
    out["upload.intent_200_pending"] = u["status"] == "pending"
    p = _upload_part(client, uid)
    out["upload.part_accepted"] = p.status_code == 200
    c = client.post(f"/v1/uploads/{uid}/cancel", headers=_h())
    out["upload.pending_cancel_200"] = c.status_code == 200
    out["upload.cancel_status"] = c.json()["status"] == "cancelled"
    # there is no GET /v1/uploads surface — the cancelled record lives
    # only in the store (retrievable via the keyed cancel replay below)
    g = client.get(f"/v1/uploads/{uid}", headers=_h())
    out["upload.get_absent_404"] = g.status_code == 404
    # parts dropped — re-adding to a cancelled intent refuses
    p2 = _upload_part(client, uid)
    out["upload.cancelled_refuses_parts"] = p2.status_code in (400, 409)
    out["upload.cancelled_refuses_parts_envelope"] = _enveloped(p2)
    # re-cancel: unkeyed replay of the recorded verdict → 200
    c2 = client.post(f"/v1/uploads/{uid}/cancel", headers=_h())
    out["upload.recancel_replay_200"] = c2.status_code == 200 and c2.json()["status"] == "cancelled"

    # keyed cancel dedupes — second keyed call replays the recorded answer
    u2 = _upload_intent(client)
    uid2 = u2["id"]
    c3 = client.post(f"/v1/uploads/{uid2}/cancel", headers=_ih("up-cancel"))
    c4 = client.post(f"/v1/uploads/{uid2}/cancel", headers=_ih("up-cancel"))
    out["upload.keyed_cancel_200"] = c3.status_code == 200
    out["upload.keyed_cancel_replay_hdr"] = _replay_hdr(c4)
    out["upload.keyed_cancel_replay_same"] = (
        c4.status_code == 200 and c4.json()["id"] == uid2 and c4.json() == c3.json()
    )

    # completed refuses; the minted file stays live
    u3 = _upload_intent(client)
    uid3 = u3["id"]
    part = _upload_part(client, uid3).json()
    done = client.post(
        f"/v1/uploads/{uid3}/complete",
        json={"part_ids": [part["id"]]},
        headers=_h(),
    )
    out["upload.complete_200"] = done.status_code == 200
    fid = (done.json().get("file") or {}).get("id")
    c5 = client.post(f"/v1/uploads/{uid3}/cancel", headers=_h())
    out["upload.completed_cancel_409"] = c5.status_code == 409
    out["upload.completed_cancel_code"] = _err_code(c5) == "upload_terminal"
    out["upload.completed_cancel_envelope"] = _enveloped(c5)
    gf = client.get(f"/v1/files/{fid}", headers=_h()) if isinstance(fid, str) else None
    out["upload.completed_file_survives"] = gf is not None and gf.status_code == 200

    # a cancelled intent can never complete — no way to mint the file late
    # (send the real part id so the refusal is the terminal 409, not a
    # body-shape 422)
    c6 = client.post(
        f"/v1/uploads/{uid}/complete",
        json={"part_ids": [p.json()["id"]]},
        headers=_h(),
    )
    out["upload.cancelled_never_completes"] = c6.status_code == 409

    missing = client.post("/v1/uploads/upload_nope/cancel", headers=_h())
    out["upload.cancel_missing_404"] = missing.status_code == 404
    out["upload.cancel_missing_code"] = _err_code(missing) == "upload_not_found"
    out["upload.cancel_missing_envelope"] = _enveloped(missing)
    return out


def _probe_vs_file_batch_cancel() -> dict[str, bool]:
    """``POST /v1/vector_stores/{vs}/file_batches/{bid}/cancel`` — members
    attach synchronously at create so the batch is terminal by
    construction; cancel is a permanent honest 409."""
    out: dict[str, bool] = {}
    client, _a = _client()
    blob = _FT_CORPUS
    r = client.post(
        "/v1/files",
        files={"file": ("vs.jsonl", blob, "application/jsonl")},
        data={"purpose": "batch"},
        headers=_h(),
    )
    assert r.status_code == 200, r.text
    fid = r.json()["id"]
    vs = client.post("/v1/vector_stores", json={"name": "kb"}, headers=_h()).json()
    fb = client.post(
        f"/v1/vector_stores/{vs['id']}/file_batches",
        json={"file_ids": [fid]},
        headers=_h(),
    ).json()
    c = client.post(f"/v1/vector_stores/{vs['id']}/file_batches/{fb['id']}/cancel", headers=_h())
    out["vs.file_batch_cancel_409"] = c.status_code == 409
    out["vs.file_batch_cancel_code"] = _err_code(c) == "file_batch_terminal"
    out["vs.file_batch_cancel_envelope"] = _enveloped(c)
    g = client.get(f"/v1/vector_stores/{vs['id']}/file_batches/{fb['id']}", headers=_h())
    out["vs.file_batch_still_completed"] = g.json()["status"] == "completed"
    missing = client.post(
        f"/v1/vector_stores/{vs['id']}/file_batches/batch_nope/cancel", headers=_h()
    )
    out["vs.file_batch_cancel_missing_404"] = missing.status_code == 404
    return out


def _probe_ft_cancel_pause() -> dict[str, bool]:
    """``POST /v1/fine_tuning/jobs/{id}/cancel|pause|resume`` — the full
    cooperative surface: queued cancel/pause lands the terminal/parked
    state outright (the worker's pre-start gate owns the dequeue race);
    running sets the cooperative flag — the runner's ``pause_gate``
    resolves it at a stage boundary; terminal refuses 409
    ``job_terminal``; resume on a non-paused job 409s
    ``job_not_paused``; pause is idempotent without double events; a
    cancelled job publishes no checkpoint."""
    from fx1.serve.finetune import FTJobOutcome  # noqa: PLC0415

    def _quick_runner(spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any) -> Any:
        del emit, should_cancel, pause_gate
        ckpt = spec.work_dir / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name, checkpoint=str(ckpt), trained_tokens=7
        )

    out: dict[str, bool] = {}
    client, app = _client(ft_runner=_quick_runner)
    _busy_executor(app)
    job = _ft_submit(client)
    jid = job["id"]
    out["ft.submit_queued"] = job["status"] == "queued"
    c = client.post(f"/v1/fine_tuning/jobs/{jid}/cancel", headers=_h())
    out["ft.queued_cancel_200"] = c.status_code == 200
    out["ft.queued_cancel_status"] = c.json()["status"] == "cancelled"
    out["ft.queued_cancel_finished_at"] = c.json()["finished_at"] is not None
    time.sleep(1.5)
    rec = _ft_wait(client, jid)
    out["ft.queued_cancel_terminal"] = rec["status"] == "cancelled"
    ev = _ft_events(client, jid)
    out["ft.queued_cancel_event_logged"] = any("cancel" in e["message"].lower() for e in ev)
    ckpts = client.get(f"/v1/fine_tuning/jobs/{jid}/checkpoints", headers=_h())
    out["ft.cancelled_no_checkpoint"] = ckpts.status_code == 200 and ckpts.json()["data"] == []
    c2 = client.post(f"/v1/fine_tuning/jobs/{jid}/cancel", headers=_h())
    out["ft.cancelled_recancel_409"] = c2.status_code == 409
    out["ft.cancelled_recancel_code"] = _err_code(c2) == "job_terminal"

    # running cancel: flag lands at the next stage boundary
    gate = _FTGate()
    client2, _a2 = _client(ft_runner=gate)
    job2 = _ft_submit(client2)
    jid2 = job2["id"]
    assert gate.entered.wait(10), "ft worker never entered the runner"
    c3 = client2.post(f"/v1/fine_tuning/jobs/{jid2}/cancel", headers=_h())
    out["ft.running_cancel_200"] = c3.status_code == 200
    out["ft.running_cancel_reports_running"] = c3.json()["status"] == "running"
    gate.release.set()
    rec2 = _ft_wait(client2, jid2)
    out["ft.running_cancel_lands_cancelled"] = rec2["status"] == "cancelled"
    ev2 = _ft_events(client2, jid2)
    out["ft.running_cancel_boundary_event"] = any(
        "cancellation requested" in e["message"] for e in ev2
    )

    # terminal cancels refuse — succeeded then failed
    job3 = _ft_submit(client)
    jid3 = job3["id"]
    _ft_wait(client, jid3)
    c4 = client.post(f"/v1/fine_tuning/jobs/{jid3}/cancel", headers=_h())
    out["ft.completed_cancel_409"] = c4.status_code == 409
    out["ft.completed_cancel_code"] = _err_code(c4) == "job_terminal"
    out["ft.completed_cancel_envelope"] = _enveloped(c4)

    def _boom(spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any) -> Any:
        del spec, emit, should_cancel, pause_gate
        raise RuntimeError("synthetic trainer down")

    client3, _a3 = _client(ft_runner=_boom)
    job4 = _ft_submit(client3)
    jid4 = job4["id"]
    _ft_wait(client3, jid4)
    c5 = client3.post(f"/v1/fine_tuning/jobs/{jid4}/cancel", headers=_h())
    out["ft.failed_cancel_409"] = c5.status_code == 409

    # pause on a queued job parks pre-start; resume re-queues honestly.
    # The restore is pinned while the executor is still busy so the
    # response (a live record read) deterministically shows 'queued' —
    # the worker can only run after the sleepers drain.
    client4, app4 = _client(ft_runner=_quick_runner)
    _busy_executor(app4, sleep_s=4.0)
    job5 = _ft_submit(client4)
    jid5 = job5["id"]
    p = client4.post(f"/v1/fine_tuning/jobs/{jid5}/pause", headers=_h())
    out["ft.queued_pause_200"] = p.status_code == 200
    out["ft.queued_pause_status"] = p.json()["status"] == "paused"
    time.sleep(0.3)
    out["ft.queued_pause_holds"] = _ft_record(client4, jid5)["status"] == "paused"
    p2 = client4.post(f"/v1/fine_tuning/jobs/{jid5}/pause", headers=_h())
    out["ft.pause_idempotent_200"] = p2.status_code == 200 and p2.json()["status"] == "paused"
    ev5 = _ft_events(client4, jid5)
    out["ft.pause_event_single"] = sum(1 for e in ev5 if "paused" in e["message"].lower()) == 1
    rs = client4.post(f"/v1/fine_tuning/jobs/{jid5}/resume", headers=_h())
    out["ft.queued_resume_200"] = rs.status_code == 200
    out["ft.queued_resume_restores_queued"] = rs.json()["status"] == "queued"
    out["ft.resume_held_queued_record"] = _ft_record(client4, jid5)["status"] == "queued"
    out["ft.resumed_completes"] = _ft_wait(client4, jid5)["status"] == "succeeded"

    # running pause parks at the runner's stage gate; cancel while paused
    # resolves through the cancel check
    gate2 = _FTGate()
    client5, _a5 = _client(ft_runner=gate2)
    job6 = _ft_submit(client5)
    jid6 = job6["id"]
    assert gate2.entered.wait(10), "ft worker never entered the runner"
    p3 = client5.post(f"/v1/fine_tuning/jobs/{jid6}/pause", headers=_h())
    out["ft.running_pause_200"] = p3.status_code == 200
    out["ft.running_pause_status"] = p3.json()["status"] == "paused"
    gate2.release.set()
    time.sleep(0.3)
    out["ft.running_pause_parks"] = _ft_record(client5, jid6)["status"] == "paused"
    rs2 = client5.post(f"/v1/fine_tuning/jobs/{jid6}/resume", headers=_h())
    out["ft.running_resume_200"] = rs2.status_code == 200
    # the worker is parked at the runner's stage-2 gate, so 'running' is
    # observable (the live record can no longer race to 'succeeded')
    out["ft.running_resume_restores"] = rs2.json()["status"] == "running"
    out["ft.running_resume_record"] = _ft_record(client5, jid6)["status"] == "running"
    gate2.stage2.set()
    out["ft.running_resumed_completes"] = _ft_wait(client5, jid6)["status"] == "succeeded"

    # cancel while paused → cancelled; resume on cancelled refuses
    gate3 = _FTGate()
    client6, _a6 = _client(ft_runner=gate3)
    job7 = _ft_submit(client6)
    jid7 = job7["id"]
    assert gate3.entered.wait(10), "ft worker never entered the runner"
    client6.post(f"/v1/fine_tuning/jobs/{jid7}/pause", headers=_h())
    gate3.release.set()
    time.sleep(0.3)
    c7 = client6.post(f"/v1/fine_tuning/jobs/{jid7}/cancel", headers=_h())
    out["ft.paused_cancel_200"] = c7.status_code == 200
    out["ft.paused_cancel_status"] = c7.json()["status"] == "cancelled"
    rs3 = client6.post(f"/v1/fine_tuning/jobs/{jid7}/resume", headers=_h())
    out["ft.cancelled_resume_409"] = rs3.status_code == 409
    out["ft.cancelled_resume_code"] = _err_code(rs3) == "job_terminal"

    # pause terminal / resume not-paused refuse — the not-paused pin
    # uses a job parked (unpaused) at the runner's stage-2 gate so the
    # refusal is 'job_not_paused', not 'job_terminal'
    p4 = client5.post(f"/v1/fine_tuning/jobs/{jid6}/pause", headers=_h())
    out["ft.terminal_pause_409"] = p4.status_code == 409
    out["ft.terminal_pause_code"] = _err_code(p4) == "job_terminal"
    job8 = _ft_submit(client6)
    jid8 = job8["id"]
    deadline = time.monotonic() + 10.0
    while _ft_record(client6, jid8)["status"] != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    rs4 = client6.post(f"/v1/fine_tuning/jobs/{jid8}/resume", headers=_h())
    out["ft.not_paused_resume_409"] = rs4.status_code == 409
    out["ft.not_paused_resume_code"] = _err_code(rs4) == "job_not_paused"
    gate3.stage2.set()

    missing = client6.post("/v1/fine_tuning/jobs/ftjob-nope/cancel", headers=_h())
    out["ft.cancel_missing_404"] = missing.status_code == 404
    out["ft.cancel_missing_envelope"] = _enveloped(missing)
    pm = client6.post("/v1/fine_tuning/jobs/ftjob-nope/pause", headers=_h())
    out["ft.pause_missing_404"] = pm.status_code == 404
    return out


def _probe_scope() -> dict[str, bool]:
    """Cancel is a mutation — every surface needs ``write``; a read-only
    key's cancel 403s ``insufficient_scope``; a write-scoped key
    cancels."""
    out: dict[str, bool] = {}
    client, app = _client(ft_runner=lambda *a, **k: None)
    ro_key, _ro_id = _mint(client, scopes=["read"])
    rw_key, _rw_id = _mint(client, scopes=["read", "write"])
    _busy_executor(app)

    # read-only refusals on every cancel surface
    job_id = _job_submit(client).json()["job_id"]
    c = client.delete(f"{_JOBS}/{job_id}", headers=_h(ro_key))
    out["scope.job_cancel_read_403"] = c.status_code == 403
    out["scope.job_cancel_read_code"] = _err_code(c) == "insufficient_scope"
    out["scope.job_cancel_read_envelope"] = _enveloped(c)

    eval_id = _eval_submit(client).json()["eval_id"]
    c2 = client.delete(f"{_EVALS}/{eval_id}", headers=_h(ro_key))
    out["scope.eval_cancel_read_403"] = c2.status_code == 403
    out["scope.eval_cancel_read_code"] = _err_code(c2) == "insufficient_scope"

    rid = _resp_submit(client).json()["id"]
    c3 = client.post(f"/v1/responses/{rid}/cancel", headers=_h(ro_key))
    out["scope.resp_cancel_read_403"] = c3.status_code == 403
    out["scope.resp_cancel_read_code"] = _err_code(c3) == "insufficient_scope"
    out["scope.resp_cancel_read_envelope"] = _enveloped(c3)

    b = _batch_create(client, [_line("c0", _chat_body("m"))])
    c4 = client.post(f"/v1/batches/{b['id']}/cancel", headers=_h(ro_key))
    out["scope.batch_cancel_read_403"] = c4.status_code == 403
    out["scope.batch_cancel_read_code"] = _err_code(c4) == "insufficient_scope"

    ab = _abatch_create(client, [_abatch_item("i0")])
    c5 = client.post(
        f"/v1/messages/batches/{ab['id']}/cancel",
        headers={**_h(ro_key), **_ANTHROPIC_H},
    )
    out["scope.abatch_cancel_read_403"] = c5.status_code == 403

    u = _upload_intent(client)
    c6 = client.post(f"/v1/uploads/{u['id']}/cancel", headers=_h(ro_key))
    out["scope.upload_cancel_read_403"] = c6.status_code == 403

    spec_id = _eval_spec(client, "scope-run")
    run_id = _eval_run_submit(client, spec_id).json()["id"]
    c7 = client.post(f"/v1/evals/{spec_id}/runs/{run_id}/cancel", headers=_h(ro_key))
    out["scope.evalrun_cancel_read_403"] = c7.status_code == 403

    ft = _ft_submit(client)
    c8 = client.post(f"/v1/fine_tuning/jobs/{ft['id']}/cancel", headers=_h(ro_key))
    p8 = client.post(f"/v1/fine_tuning/jobs/{ft['id']}/pause", headers=_h(ro_key))
    r8 = client.post(f"/v1/fine_tuning/jobs/{ft['id']}/resume", headers=_h(ro_key))
    out["scope.ft_cancel_read_403"] = c8.status_code == 403
    out["scope.ft_pause_read_403"] = p8.status_code == 403
    out["scope.ft_resume_read_403"] = r8.status_code == 403

    # write scope cancels — control still answers honestly
    job_id2 = _job_submit(client).json()["job_id"]
    c9 = client.delete(f"{_JOBS}/{job_id2}", headers=_h(rw_key))
    out["scope.job_cancel_write_200"] = c9.status_code == 200
    return out


def _probe_drain() -> dict[str, bool]:
    """Drain refuses new submits (503 ``draining``) but cancels stay open
    — control-plane verbs are exempt from the drain latch."""
    out: dict[str, bool] = {}
    client, app = _client()
    _busy_executor(app)
    job_id = _job_submit(client).json()["job_id"]
    rid = _resp_submit(client).json()["id"]
    b = _batch_create(client, [_line("c0", _chat_body("m"))])
    eval_id = _eval_submit(client).json()["eval_id"]

    d = client.post(_DRAIN, headers=_h())
    out["drain.latch_200"] = d.status_code == 200
    sub_after = _job_submit(client)
    out["drain.submit_refused_503"] = sub_after.status_code == 503
    out["drain.submit_refused_code"] = _err_code(sub_after) == "draining"

    c1 = client.delete(f"{_JOBS}/{job_id}", headers=_h())
    out["drain.job_cancel_open"] = c1.status_code == 200
    c2 = client.post(f"/v1/responses/{rid}/cancel", headers=_h())
    out["drain.resp_cancel_open"] = c2.status_code == 200
    c3 = client.post(f"/v1/batches/{b['id']}/cancel", headers=_h())
    out["drain.batch_cancel_open"] = c3.status_code == 200
    c4 = client.delete(f"{_EVALS}/{eval_id}", headers=_h())
    out["drain.eval_cancel_open"] = c4.status_code == 200
    return out


def _probe_metering() -> dict[str, bool]:
    """Cancel bills ``uses`` on the caller key like any authorized call;
    cancelled work reports only the usage that actually ran — a queued-
    cancelled batch bills zero tokens, a mid-flight-cancelled one keeps
    the partial lines' charges."""
    out: dict[str, bool] = {}
    client, app = _client()
    rw_key, rw_id = _mint(client, scopes=["read", "write"])
    card0 = _key_card(client, rw_id)
    job_id = _job_submit(client).json()["job_id"]
    client.delete(f"{_JOBS}/{job_id}", headers=_h(rw_key))
    card1 = _key_card(client, rw_id)
    out["meter.cancel_bills_uses"] = card1["uses"] == card0["uses"] + 1

    # a refused cancel still bills the authorized call
    job2 = _job_submit(client).json()["job_id"]
    _job_wait(client, job2)
    client.delete(f"{_JOBS}/{job2}", headers=_h(rw_key))
    card2 = _key_card(client, rw_id)
    out["meter.refused_cancel_bills_uses"] = card2["uses"] == card1["uses"] + 1

    # queued-cancelled batch: zero lines ran → zero tokens
    _busy_executor(app)
    fid = _upload_lines(client, [_line(f"t{i}", _chat_body(f"m{i}")) for i in range(3)])
    r = client.post(
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        headers=_h(rw_key),
    )
    bid = r.json()["id"]
    card3 = _key_card(client, rw_id)
    client.post(f"/v1/batches/{bid}/cancel", headers=_h(rw_key))
    _batch_wait(client, bid, auth=rw_key)
    card4 = _key_card(client, rw_id)
    out["meter.queued_cancelled_batch_zero_tokens"] = card4.get("tokens_used", 0) == card3.get(
        "tokens_used", 0
    )
    out["meter.cancel_uses_counted"] = card4["uses"] > card3["uses"]

    # mid-flight cancelled batch: cancel while the worker is parked
    # inside a line — that line's charge lands, the tail never bills;
    # partial usage is reported honestly, not zeroed
    gate = _LineGate()
    client2, _a = _client(backend_map={"hosted_k3": lambda: gate})
    rw2, rw2_id = _mint(client2, scopes=["read", "write"])
    lines = [_line(f"p{i}", _chat_body(f"m{i}")) for i in range(4)]
    fid2 = _upload_lines(client2, lines)
    r2 = client2.post(
        "/v1/batches",
        json={
            "input_file_id": fid2,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        headers=_h(rw2),
    )
    bid2 = r2.json()["id"]
    card5 = _key_card(client2, rw2_id)
    assert gate.entered.wait(10), "batch worker never entered the backend"
    # line 1 lands; the worker immediately parks inside line 2's acquire
    gate.sem.release()
    deadline = time.monotonic() + 10.0
    while gate.calls < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    out["meter.midflight_worker_inside_line"] = gate.calls == 2
    c2 = client2.post(f"/v1/batches/{bid2}/cancel", headers=_h(rw2))
    out["meter.midflight_cancel_200"] = c2.status_code == 200
    # release the in-flight line; the worker stops at the next boundary
    gate.sem.release()
    final = _batch_wait(client2, bid2, auth=rw2)
    card6 = _key_card(client2, rw2_id)
    completed = final["request_counts"]["completed"]
    out["meter.midflight_cancelled_terminal"] = final["status"] == "cancelled"
    out["meter.midflight_partial_tokens_honest"] = (
        completed >= 1
        and completed < 4
        and card6["tokens_used"] == card5["tokens_used"] + completed * _U["total_tokens"]
    )
    out["meter.midflight_tail_unbilled"] = (
        card6["tokens_used"] < card5["tokens_used"] + len(lines) * _U["total_tokens"]
    )
    return out


def _probe_concurrent() -> dict[str, bool]:
    """Parallel cancels CAS — exactly one wins the transition, and a
    cancel racing a terminal completion never produces a torn record:
    either the cancel lands first (record cancelled, late result
    dropped) or the completion lands first (cancel 409s)."""
    out: dict[str, bool] = {}
    client, app = _client()
    _busy_executor(app)
    job_id = _job_submit(client).json()["job_id"]

    codes: list[int] = []
    lock = threading.Lock()

    def _cancel() -> None:
        r = client.delete(f"{_JOBS}/{job_id}", headers=_h())
        with lock:
            codes.append(r.status_code)

    threads = [threading.Thread(target=_cancel) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out["concurrent.job_cancels_never_torn"] = all(c == 200 for c in codes)
    rec = _job_record(client, job_id)
    out["concurrent.job_single_terminal_verdict"] = rec["status"] == "cancelled"

    # parallel cancels on a queued response — the terminal check + CAS
    # means at most one request performs the flip; losers see 409 or the
    # idempotent 200 depending on read timing — never a torn record
    rid = _resp_submit(client).json()["id"]
    codes2: list[int] = []
    lock2 = threading.Lock()

    def _cancel2() -> None:
        r = client.post(f"/v1/responses/{rid}/cancel", headers=_h())
        with lock2:
            codes2.append(r.status_code)

    threads2 = [threading.Thread(target=_cancel2) for _ in range(4)]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join()
    out["concurrent.resp_cancels_cas_winner"] = 200 in codes2
    out["concurrent.resp_cancels_all_enveloped"] = all(c in (200, 409) for c in codes2)
    rec2 = _resp_wait(client, rid)
    out["concurrent.resp_single_terminal_verdict"] = rec2["status"] == "cancelled"

    # racing cancel vs terminal completion — deterministic variant: the
    # gate backend parks the worker; cancel wins while in-flight; the
    # late completion must never overwrite the verdict (CAS both ways)
    from fx1.serve.conv_audit import _GateBackend  # noqa: PLC0415

    gate = _GateBackend()
    client2, _a = _client(backend_map={"byok": lambda: gate})
    rid2 = _resp_submit(client2).json()["id"]
    assert gate.entered.wait(10), "worker never entered the backend"
    done: list[int] = []
    ev = threading.Event()

    def _late_cancel() -> None:
        ev.wait(5)
        r = client2.post(f"/v1/responses/{rid2}/cancel", headers=_h())
        done.append(r.status_code)

    th = threading.Thread(target=_late_cancel)
    th.start()
    c = client2.post(f"/v1/responses/{rid2}/cancel", headers=_h())
    gate.gate.set()
    ev.set()
    th.join()
    time.sleep(0.4)
    rec3 = _resp_wait(client2, rid2)
    out["concurrent.cancel_vs_complete_no_resurrect"] = (
        c.status_code == 200 and done == [409] and rec3["status"] == "cancelled"
    )
    return out


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def cancel_audit() -> dict[str, Any]:
    """Run the cancel/pause battery; returns literal bools."""
    from fx1.serve.conv_audit import _audit_context  # noqa: PLC0415

    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_probe_jobs_cancel())
        out.update(_probe_evals_cancel())
        out.update(_probe_eval_runs_cancel())
        out.update(_probe_batch_cancel())
        out.update(_probe_abatch_cancel())
        out.update(_probe_response_cancel())
        out.update(_probe_upload_cancel())
        out.update(_probe_vs_file_batch_cancel())
        out.update(_probe_ft_cancel_pause())
        out.update(_probe_scope())
        out.update(_probe_drain())
        out.update(_probe_metering())
        out.update(_probe_concurrent())
        return out


def cancel_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under cancel_audit.v1."""
    r = cancel_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "cancel_audit",
        "schema": "cancel_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok, "defects": defects},
        "coverage": {
            "transport": [
                "harness.jobs.cancel",
                "harness.evals.cancel",
                "v1.evals.runs.cancel",
                "v1.responses.cancel",
                "v1.batches.cancel",
                "v1.messages.batches.cancel",
                "v1.uploads.cancel",
                "v1.vector_stores.file_batches.cancel",
                "v1.fine_tuning.cancel",
                "v1.fine_tuning.pause",
                "v1.fine_tuning.resume",
                "harness.drain",
                "harness.keys.mint",
            ],
            "not_verified": [
                "real subprocess kill of a running job (there is no kill handle by design)",
                "webhook delivery under real network partitions",
                "cancel racing a store eviction mid-flight",
                "durable restart with a queued-cancelled record (store journal covered by sibling lanes)",
            ],
            "not_executed": [
                "responses cancel on a streaming (non-background) call — background-only surface",
            ],
        },
        "interpretation": (
            "Every probe True means: on this checkout every cancel/pause/"
            "resume surface honours its pinned transition contract — "
            "queued→cancelled direct paths never run, running refuses "
            "where no kill handle exists and lands at a stage boundary "
            "where a cooperative flag does, terminal statuses always "
            "refuse with the honest code, re-cancels are idempotent only "
            "where the surface declares it, side effects (partial output "
            "files, empty eval output_items, dropped parts, unpublished "
            "checkpoints, untouched conversations) hold, webhooks fire "
            "once and signed, read-only keys 403 insufficient_scope, "
            "drain leaves cancels open while new submits 503, cancel "
            "bills uses like any authorized call and cancelled work "
            "charges only the usage that ran, parallel cancels CAS "
            "without tearing, and a keyed Idempotency-Key replay can "
            "never resurrect a cancelled or deleted response under a "
            "false status. SYNTHETIC stubs only — no research claim."
            if ok
            else f"CANCEL AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(cancel_audit_bench(), indent=1))
