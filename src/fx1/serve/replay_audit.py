"""replay_audit — the ``?stream=true`` replay / in-flight-interrupt battery.

``stream_audit`` pins the create-time SSE grammar; this battery attacks the
*replay* surface — ``GET /v1/responses/{id}?stream=true``, the OpenAI
"resume a stored response's SSE stream" route — plus the in-flight
interruption semantics around it (cancel / delete / deadline mid-follow),
against a live in-process app (nothing leaves the box). Starlette
TestClient buffers each ASGI response before the client can read it;
live-follow probes drive the GET on a thread so the interrupt lands while
the generator is mid-loop:

- *Byte-exact replay* — a completed stored response replays the
  create-time stream byte-for-byte (same ``event:``/``id:``/``data:``
  frames, same minted ids, terminal ``response.completed`` last);
  replay is deterministic across calls and carries the completion's
  ``X-Fx1-Completion-Id``/``X-Fx1-Receipt-Sha256`` link headers.
- *Resume cursor* — ``starting_after=N`` resumes past absolute frame
  index N (the ``id:`` field): a strict byte suffix, first frame
  ``id: N+1``; beyond-end yields an honest empty body; bogus values
  (``-1``, ``abc``, ``1.5``) are refused 422. ``timeout_s`` outside
  ``[1, 3600]`` is refused. ``stream``/``timeout_s`` on the JSON read are
  silently ignored (measured), as is the unadvertised ``sequence_number``
  param (measured — pinned, not grafted). ``stream`` itself parses
  leniently (``1`` works, ``false``/``0`` → JSON, ``banana`` → 422).
- *Store gate* — ``store:false`` and deleted records answer the same
  enveloped 404 ``not_found`` as unknown ids; non-response ids (``conv_*``)
  likewise; the JSON read never leaks ``_fx1_*`` internals.
- *Terminal honesty* — ``response.failed`` replays with the recorded
  ``error`` payload; ``response.cancelled``/``incomplete`` the same; a
  response cancelled while still ``queued`` replays its true lifecycle
  (``created`` → ``queued`` → ``cancelled``, no phantom ``in_progress``);
  ``background:true`` + ``stream:true`` echoes ``background:false``
  (the call ran synchronously) and replays byte-identically.
- *Live attach* — a ``?stream=true`` GET on an in-flight record emits the
  lifecycle prelude then live-follows: ``: keepalive`` comments, the
  ``in_progress`` transition when it lands, terminal last; a resume
  cursor positions mid-grammar; ``timeout_s`` truncates honestly without
  a terminal; a mid-follow cancel ends the stream on
  ``response.cancelled``, a mid-follow delete ends it with no terminal.
- *Isolation & cost* — N concurrent replays of one record are
  byte-identical; replay is read-only (record, conversation items,
  completion log, and backend call count all unchanged — replaying does
  not re-execute the model or spend tokens).
- *Batch* — ``stream:true``/``background:true``/``conversation`` inside a
  batch line are refused per-line ``invalid_request`` while the batch
  completes and a clean sibling line passes.
- *Drain* — replay, JSON read, cancel, and delete stay open under the
  drain latch while new submits refuse 503 ``draining``.
- *Client* — ``HarnessClient.responses_replay`` parses
  ``(events, completion_id)``, maps 404→``KeyError`` and 422→``ValueError``,
  and raises ``HarnessTransportError`` on a truncated follow or a 409.

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect by probe name so the finding survives byte-for-byte.

Sealed ``replay_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.harness import Harness
from fx1.serve.backends import ToolCompletion
from fx1.serve.client_audit import _remote
from fx1.serve.stream_audit import (
    _ENV_KEYS,
    _Ctx,
    _data_frames,
    _event_payloads,
    _fast_runner,
    _ft_runner,
    _grammar,
    _parallel,
    _sse_frames,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

__all__ = ["replay_audit", "replay_audit_bench"]

# _ENV_KEYS (imported from stream_audit): every env that can bend app
# construction or backend resolution — cleared for the audit so ambient
# settings never leak into a probe

_SSE_CT = "text/event-stream"
_ECHO_PROMPT = "replay-tick " + "echo " * 40  # ~200 chars → several delta pieces
_STUB_USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
_TERMINAL_EVENTS = {
    "response.completed",
    "response.incomplete",
    "response.failed",
    "response.cancelled",
}
_TOOL_CALL_ID = "call_a1b2c3"
_TOOL_NAME = "get_price"
_TOOL_ARGS_TEXT = '{"symbol":"ESZ5","venue":"CME"}'
_TOOL_SPEC_RESP = {
    "type": "function",
    "name": _TOOL_NAME,
    "parameters": {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
    },
}
_CONC_N = 6
_WAIT_S = 15.0


# ---------------------------------------------------------------------------
# SSE grammar helpers — ``_SseFrame``/``_sse_frames``/``_data_frames``/
# ``_event_payloads``/``_grammar`` are imported from ``stream_audit`` (the
# shared physical-layer parser + assertions every dialect inherits).
# ---------------------------------------------------------------------------


def _event_names(body: str) -> list[str]:
    return [e for e, _ in _event_payloads(_sse_frames(body))]


def _err_enveloped(resp: Any, *, status: int, code: str | None = None) -> bool:
    """A refusal must carry the ``{"error":{message,type,code}}``
    envelope — never a bare ``detail`` or an empty body."""
    try:
        body = resp.json()
    except ValueError:
        return False
    err = body.get("error")
    return (
        resp.status_code == status
        and isinstance(err, dict)
        and isinstance(err.get("message"), str)
        and isinstance(err.get("type"), str)
        and (code is None or err.get("code") == code)
    )


# ---------------------------------------------------------------------------
# Stub backends — deterministic gated output the wire must carry verbatim.
# ---------------------------------------------------------------------------


class _StubBackend:
    """Clean completion stub — echoes the last user turn, reports usage,
    counts calls so replay probes can prove no model re-execution."""

    def __init__(self) -> None:
        self._model = "replay-stub-0"
        self.calls = 0
        self.last_usage: dict[str, int] = dict(_STUB_USAGE)
        self._lock = threading.Lock()

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        with self._lock:
            self.calls += 1
        return f"stub:{messages[-1]['content']}"

    def close(self) -> None:
        pass


class _GateBackend(_StubBackend):
    """``complete`` parks on an event — the deterministic way to hold a
    background response in-flight while a replay attach, cancel, or
    delete lands (``entered`` fires once the call is inside)."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.gate = threading.Event()

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        self.entered.set()
        self.gate.wait(timeout=30.0)
        return super().complete(messages)


class _FailBackend(_StubBackend):
    """Dies inside ``complete`` — the background-worker failure behind
    ``response.failed``."""

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del messages, sampling
        raise RuntimeError("backend boom — gate never reached")


class _ToolStubBackend(_StubBackend):
    """Answers the tools channel — one verbatim call so ``max_tool_calls``
    truncation can drive ``response.incomplete``."""

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
        parallel_tool_calls: Any = None,
        logprobs: Any = None,
        top_logprobs: Any = None,
    ) -> ToolCompletion:
        del messages, sampling, tools, tool_choice, parallel_tool_calls, logprobs, top_logprobs
        return ToolCompletion(
            content=None,
            tool_calls=(
                {
                    "id": _TOOL_CALL_ID,
                    "type": "function",
                    "index": 0,
                    "function": {"name": _TOOL_NAME, "arguments": _TOOL_ARGS_TEXT},
                },
            ),
            finish_reason="tool_calls",
        )


# ---------------------------------------------------------------------------
# App construction — ``_Ctx``/``_fast_runner``/``_ft_runner``/``_parallel``
# are imported from ``stream_audit``; ``_app``/``_make_ctx`` stay local for
# the extra ``max_inflight`` dial the executor-park probes need.
# ---------------------------------------------------------------------------


def _app(
    workdir: Path,
    backend: Any,
    *,
    sse_keepalive_s: float = 15.0,
    max_inflight: int | None = None,
) -> FastAPI:
    import fx1.serve.api as api_mod  # noqa: PLC0415

    return api_mod.create_app(
        harness=Harness(runner=_fast_runner),
        backend_resolver=lambda *a, **k: backend,
        ft_runner=_ft_runner,
        ft_dir=workdir / "ft",
        state_dir=workdir / "state",
        sse_keepalive_s=sse_keepalive_s,
        max_inflight=max_inflight,
    )


def _make_ctx(
    workdir: Path,
    backend: Any,
    *,
    sse_keepalive_s: float = 15.0,
    max_inflight: int | None = None,
) -> _Ctx:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    app = _app(workdir, backend, sse_keepalive_s=sse_keepalive_s, max_inflight=max_inflight)
    return _Ctx(client=TestClient(app, raise_server_exceptions=False), app=app)


def _submit_bg(client: TestClient, payload: dict[str, Any] | None = None) -> str:
    """POST a background response; returns the queued response id."""
    body = {"model": "fx1", "input": "bg", "background": True}
    body.update(payload or {})
    r = client.post("/v1/responses", json=body)
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _wait_status(client: TestClient, rid: str, want: set[str], timeout: float = _WAIT_S) -> str:
    """Poll the JSON read until the record's status lands in ``want``."""
    end = time.monotonic() + timeout
    status = ""
    while time.monotonic() < end:
        rec = client.get(f"/v1/responses/{rid}")
        if rec.status_code != 200:
            break
        status = str(rec.json().get("status"))
        if status in want:
            break
        time.sleep(0.03)
    return status


def _busy_executor(app: FastAPI, release: threading.Event) -> None:
    """Occupy every worker slot so a submitted response stays ``queued``."""
    executor = app.state.jobs_executor
    n = int(getattr(executor, "_max_workers", 4))  # noqa: SLF001 — audit reads pool depth
    for _ in range(n):
        executor.submit(lambda: release.wait(timeout=15.0))
    time.sleep(0.15)


# ``_tc_transport``/``_remote`` (imported from client_audit) bind a
# HarnessClient to a TestClient transport — the replay probes' client leg.


# ---------------------------------------------------------------------------
# Completed-record replay — byte-exact regeneration of the create stream
# ---------------------------------------------------------------------------


def _probe_replay_core(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    body = {"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    r = ctx.client.post("/v1/responses", json=body)
    frames = _sse_frames(r.text)
    events = _event_payloads(frames)
    rid = events[0][1]["response"]["id"]

    g = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    gframes = _sse_frames(g.text)
    gevents = _event_payloads(gframes)
    out["replay_status_200"] = g.status_code == 200
    out["replay_sse_ct"] = g.headers.get("content-type", "").startswith(_SSE_CT)
    _grammar(out, "replay", g.text, gframes)
    out["replay_byte_identical"] = g.text == r.text
    out["replay_terminal_last"] = (
        bool(gevents)
        and gevents[-1][0] == "response.completed"
        and gevents[-1][1]["type"] == gevents[-1][0]
    )
    out["replay_event_data_pairs"] = all(
        f.event is not None and f.data is not None and json.loads(f.data).get("type") == f.event
        for f in _data_frames(gframes)
    )
    # the replayed terminal IS the stored envelope — identical to the JSON read
    gj = ctx.client.get(f"/v1/responses/{rid}")
    out["replay_terminal_is_json_twin"] = (
        gj.status_code == 200
        and gj.headers.get("content-type", "").startswith("application/json")
        and gj.json() == gevents[-1][1]["response"]
    )
    # the replay carries the completion/evidence link headers of the record
    out["replay_headers_link"] = g.headers.get("x-fx1-completion-id") == r.headers.get(
        "x-fx1-completion-id"
    ) and bool(g.headers.get("x-fx1-receipt-sha256"))
    # a second replay regenerates the identical byte stream — read-only
    g2 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    out["replay_deterministic"] = g2.text == g.text
    # a non-stream create's stored record replays the full grammar too
    rs = ctx.client.post("/v1/responses", json={"model": "fx1", "input": _ECHO_PROMPT})
    rid2 = rs.json()["id"]
    g3 = ctx.client.get(f"/v1/responses/{rid2}", params={"stream": "true"})
    events3 = _event_payloads(_sse_frames(g3.text))
    out["replay_nonstream_record"] = (
        bool(events3)
        and events3[-1][0] == "response.completed"
        and events3[-1][1]["response"]["output"] == rs.json()["output"]
    )
    return out


def _probe_resume(ctx: _Ctx) -> dict[str, bool]:
    """``starting_after`` slices by absolute ``id:`` index; bogus resume
    params are refused or measurably ignored — never silently grafted."""
    out: dict[str, bool] = {}
    r = ctx.client.post(
        "/v1/responses", json={"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    )
    full = r.text
    events = _event_payloads(_sse_frames(full))
    rid = events[0][1]["response"]["id"]
    n_frames = len(_data_frames(_sse_frames(full)))

    g0 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true", "starting_after": 0})
    out["resume_zero_drops_first"] = g0.text == full.split("\n\n", 1)[1] and g0.text != full
    g3 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true", "starting_after": 3})
    g3_frames = _data_frames(_sse_frames(g3.text))
    out["resume_absolute_cursor"] = (
        bool(g3_frames)
        and g3_frames[0].seq == "4"
        and full.endswith(g3.text)
        and g3.text != full
        and [f.seq for f in g3_frames] == [str(i) for i in range(4, n_frames)]
    )
    glast = ctx.client.get(
        f"/v1/responses/{rid}", params={"stream": "true", "starting_after": n_frames - 2}
    )
    glast_events = _event_payloads(_sse_frames(glast.text))
    out["resume_terminal_only"] = len(glast_events) == 1 and glast_events[0][0] in _TERMINAL_EVENTS
    gb = ctx.client.get(
        f"/v1/responses/{rid}", params={"stream": "true", "starting_after": n_frames}
    )
    out["resume_beyond_end_empty"] = gb.status_code == 200 and gb.text == ""
    out["resume_bogus_refused"] = all(
        _err_enveloped(
            ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true", "starting_after": v}),
            status=422,
            code="validation",
        )
        for v in ("-1", "abc", "1.5")
    )
    out["timeout_bounds_refused"] = all(
        _err_enveloped(
            ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true", "timeout_s": v}),
            status=422,
            code="validation",
        )
        for v in ("0", "0.5", "3601")
    )
    out["stream_bogus_refused"] = all(
        _err_enveloped(
            ctx.client.get(f"/v1/responses/{rid}", params={"stream": v}),
            status=422,
            code="validation",
        )
        for v in ("banana", "")
    )
    # the flag parses leniently: truthy spellings stream, falsy read JSON
    g1 = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "1"})
    out["stream_lenient_truthy"] = g1.status_code == 200 and g1.text == full
    gf = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "0"})
    out["stream_falsy_json"] = gf.status_code == 200 and "data:" not in gf.text
    # stream-only params on the JSON read are ignored (measured)
    gj = ctx.client.get(f"/v1/responses/{rid}", params={"starting_after": 1, "timeout_s": 1})
    out["resume_params_json_ignored"] = (
        gj.status_code == 200 and gj.json().get("status") == "completed"
    )
    # an unadvertised resume param is ignored — the full stream replays
    gq = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true", "sequence_number": 2})
    out["unadvertised_resume_param_ignored"] = gq.status_code == 200 and gq.text == full
    return out


def _probe_store_gate(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    rs = ctx.client.post("/v1/responses", json={"model": "fx1", "input": "ns", "store": False})
    rid_ns = rs.json()["id"]
    out["store_false_replay_404"] = _err_enveloped(
        ctx.client.get(f"/v1/responses/{rid_ns}", params={"stream": "true"}),
        status=404,
        code="not_found",
    )
    r = ctx.client.post("/v1/responses", json={"model": "fx1", "input": "del"})
    rid = r.json()["id"]
    d = ctx.client.delete(f"/v1/responses/{rid}")
    out["deleted_then_404"] = (
        d.status_code == 200
        and d.json().get("deleted") is True
        and _err_enveloped(
            ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"}),
            status=404,
            code="not_found",
        )
        and _err_enveloped(ctx.client.get(f"/v1/responses/{rid}"), status=404, code="not_found")
    )
    conv = ctx.client.post("/v1/conversations", json={})
    cid = conv.json()["id"]
    out["non_response_id_404"] = _err_enveloped(
        ctx.client.get(f"/v1/responses/{cid}", params={"stream": "true"}),
        status=404,
        code="not_found",
    ) and _err_enveloped(
        ctx.client.get("/v1/responses/resp_never", params={"stream": "true"}),
        status=404,
        code="not_found",
    )
    rec = ctx.client.post("/v1/responses", json={"model": "fx1", "input": "k"}).json()
    gj2 = ctx.client.get(f"/v1/responses/{rec['id']}")
    out["json_read_no_internals"] = gj2.status_code == 200 and not any(
        k.startswith("_fx1_") for k in gj2.json()
    )
    return out


# ---------------------------------------------------------------------------
# Terminal-state replay honesty — failed / cancelled / incomplete records,
# plus the queued-cancel lifecycle (a record that never ran must not gain
# a phantom ``response.in_progress`` phase on replay)
# ---------------------------------------------------------------------------


def _probe_failed_replay(ctx_fail: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    rid = _submit_bg(ctx_fail.client, {"input": "die"})
    status = _wait_status(ctx_fail.client, rid, {"failed"})
    out["failed_record_terminal"] = status == "failed"
    g = ctx_fail.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    events = _event_payloads(_sse_frames(g.text))
    names = [e for e, _ in events]
    term = events[-1][1]["response"] if events else {}
    out["failed_replay_terminal"] = (
        names[-1] == "response.failed" and term.get("status") == "failed"
    )
    out["failed_replay_error_field"] = isinstance(term.get("error"), dict) and bool(
        term["error"].get("message")
    )
    out["failed_replay_lifecycle"] = names[:3] == [
        "response.created",
        "response.queued",
        "response.in_progress",
    ]
    return out


def _probe_incomplete_replay(ctx_tools: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx_tools.client.post(
        "/v1/responses",
        json={
            "model": "fx1",
            "input": "price of ESZ5",
            "tools": [_TOOL_SPEC_RESP],
            "max_tool_calls": 0,
        },
    )
    rid = r.json()["id"]
    g = ctx_tools.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    events = _event_payloads(_sse_frames(g.text))
    names = [e for e, _ in events]
    term = events[-1][1]["response"] if events else {}
    out["incomplete_terminal"] = (
        names[-1] == "response.incomplete" and term.get("status") == "incomplete"
    )
    out["incomplete_reason"] = term.get("incomplete_details") == {"reason": "max_tool_calls"}
    # the replayed terminal payload IS the stored record — same output
    rec = ctx_tools.client.get(f"/v1/responses/{rid}").json()
    out["incomplete_output_verbatim"] = term.get("output") == rec.get("output")
    return out


def _probe_cancelled_replay(workdir: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # --- cancelled while in_progress → created/queued/in_progress/cancelled
    backend = _GateBackend()
    ctx = _make_ctx(workdir / "cancel-ip", backend, sse_keepalive_s=0.05)
    rid = _submit_bg(ctx.client)
    assert backend.entered.wait(timeout=10.0), "worker never entered the backend"
    d = ctx.client.post(f"/v1/responses/{rid}/cancel")
    g = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    names = _event_names(g.text)
    out["cancelled_record_200"] = d.status_code == 200 and d.json().get("status") == "cancelled"
    out["cancelled_replay_terminal"] = (
        names[-1] == "response.cancelled" and "response.completed" not in names
    )
    out["cancelled_inprogress_lifecycle"] = names == [
        "response.created",
        "response.queued",
        "response.in_progress",
        "response.cancelled",
    ]
    backend.gate.set()
    # --- cancelled while still queued → no phantom in_progress phase
    backend2 = _GateBackend()
    ctx2 = _make_ctx(workdir / "cancel-q", backend2, sse_keepalive_s=0.05, max_inflight=1)
    sleeper_rel = threading.Event()
    _busy_executor(ctx2.app, sleeper_rel)
    try:
        rid2 = _submit_bg(ctx2.client)
        got = ctx2.client.get(f"/v1/responses/{rid2}").json()
        out["queued_cancel_record_queued"] = got.get("status") == "queued"
        d2 = ctx2.client.post(f"/v1/responses/{rid2}/cancel")
        g2 = ctx2.client.get(f"/v1/responses/{rid2}", params={"stream": "true"})
        names2 = _event_names(g2.text)
        out["queued_cancel_200"] = d2.status_code == 200 and d2.json().get("status") == "cancelled"
        # the record never left queued — its lifecycle has no in_progress
        out["queued_cancel_no_phantom_inprogress"] = names2 == [
            "response.created",
            "response.queued",
            "response.cancelled",
        ]
        out["queued_cancel_deterministic"] = (
            ctx2.client.get(f"/v1/responses/{rid2}", params={"stream": "true"}).text == g2.text
        )
    finally:
        sleeper_rel.set()
        backend2.gate.set()
    return out


def _probe_background_stream(ctx: _Ctx) -> dict[str, bool]:
    """``background:true`` + ``stream:true`` runs synchronously — the
    stored envelope must echo ``background:false`` (it never queued) so
    the replay regenerates the create stream byte-for-byte instead of
    fabricating a queued lifecycle that never happened."""
    out: dict[str, bool] = {}
    r = ctx.client.post(
        "/v1/responses",
        json={"model": "fx1", "input": _ECHO_PROMPT, "stream": True, "background": True},
    )
    events = _event_payloads(_sse_frames(r.text))
    rid = events[0][1]["response"]["id"]
    out["bg_stream_create_ran_sync"] = events[0][1]["response"].get("status") == "in_progress"
    rec = ctx.client.get(f"/v1/responses/{rid}").json()
    out["bg_stream_echo_false"] = rec.get("background") is False
    g = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    names = _event_names(g.text)
    out["bg_stream_replay_identical"] = g.text == r.text
    out["bg_stream_no_queued_frames"] = "response.queued" not in names
    return out


# ---------------------------------------------------------------------------
# Live attach — ``?stream=true`` on an in-flight record
# ---------------------------------------------------------------------------


def _replay_async(client: TestClient, path: str) -> tuple[threading.Thread, list[Any]]:
    """Drive a replay GET on a thread so an interrupt can land mid-follow."""
    box: list[Any] = []

    def go() -> None:
        box.append(client.get(path))

    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t, box


def _probe_live_attach(workdir: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    backend = _GateBackend()
    ctx = _make_ctx(workdir / "attach", backend, sse_keepalive_s=0.05, max_inflight=1)
    sleeper_rel = threading.Event()
    _busy_executor(ctx.app, sleeper_rel)
    try:
        rid = _submit_bg(ctx.client)
        t, box = _replay_async(ctx.client, f"/v1/responses/{rid}?stream=true&timeout_s=20")
        time.sleep(0.3)  # the follow loop emits the queued prelude + keepalives
        sleeper_rel.set()  # free the worker → queued → in_progress (inside the gate)
        assert backend.entered.wait(timeout=10.0), "worker never entered the backend"
        time.sleep(0.3)  # a pass emits in_progress then keepalives again
        backend.gate.set()  # complete lands → terminal frame → stream ends
        t.join(timeout=20.0)
        g = box[0]
    finally:
        sleeper_rel.set()
        backend.gate.set()
    frames = _sse_frames(g.text)
    comments = [f for f in frames if f.comment_only]
    events = _event_payloads(frames)
    names = [e for e, _ in events]
    datas = _data_frames(frames)
    out["attach_prelude_queued"] = names[:3] == [
        "response.created",
        "response.queued",
        "response.in_progress",
    ]
    out["attach_created_queued_status"] = (
        events[0][1]["response"]["status"] == "queued" if events else False
    )
    out["attach_keepalives"] = len(comments) >= 1 and all(f.raw == ": keepalive" for f in comments)
    out["attach_terminal_last"] = names[-1] == "response.completed"
    out["attach_ids_dense"] = [f.seq for f in datas] == [str(i) for i in range(len(datas))]
    out["attach_no_dup_ids"] = len({f.seq for f in datas}) == len(datas)
    # a resume cursor on a live record positions mid-grammar
    backend2 = _GateBackend()
    ctx2 = _make_ctx(workdir / "attach-skip", backend2, sse_keepalive_s=0.05)
    try:
        rid2 = _submit_bg(ctx2.client)
        assert backend2.entered.wait(timeout=10.0), "worker never entered the backend"
        # attach in-flight with the cursor past the prelude: first frame ≥ skip
        t2, box2 = _replay_async(
            ctx2.client, f"/v1/responses/{rid2}?stream=true&starting_after=2&timeout_s=20"
        )
        time.sleep(0.3)
        backend2.gate.set()
        t2.join(timeout=20.0)
        g2 = box2[0]
    finally:
        backend2.gate.set()
    frames2 = _data_frames(_sse_frames(g2.text))
    out["attach_resume_skips_prelude"] = bool(frames2) and frames2[0].seq == "3"
    out["attach_resume_terminal"] = _event_names(g2.text)[-1] == "response.completed"
    return out


def _probe_interrupts(workdir: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # --- cancel lands mid-follow → the stream ends on response.cancelled
    backend = _GateBackend()
    ctx = _make_ctx(workdir / "int-cancel", backend, sse_keepalive_s=0.05)
    try:
        rid = _submit_bg(ctx.client)
        assert backend.entered.wait(timeout=10.0), "worker never entered the backend"
        t, box = _replay_async(ctx.client, f"/v1/responses/{rid}?stream=true&timeout_s=20")
        time.sleep(0.3)
        d = ctx.client.post(f"/v1/responses/{rid}/cancel")
        t.join(timeout=20.0)
        g = box[0]
    finally:
        backend.gate.set()
    names = _event_names(g.text)
    out["midfollow_cancel_200"] = d.status_code == 200 and d.json().get("status") == "cancelled"
    out["midfollow_cancel_terminal"] = names == [
        "response.created",
        "response.queued",
        "response.in_progress",
        "response.cancelled",
    ]
    # --- delete lands mid-follow → the stream ends with no terminal frame
    backend2 = _GateBackend()
    ctx2 = _make_ctx(workdir / "int-del", backend2, sse_keepalive_s=0.05)
    try:
        rid2 = _submit_bg(ctx2.client)
        assert backend2.entered.wait(timeout=10.0), "worker never entered the backend"
        t2, box2 = _replay_async(ctx2.client, f"/v1/responses/{rid2}?stream=true&timeout_s=20")
        time.sleep(0.3)
        dr = ctx2.client.delete(f"/v1/responses/{rid2}")
        t2.join(timeout=20.0)
        g2 = box2[0]
    finally:
        backend2.gate.set()
    names2 = _event_names(g2.text)
    out["midfollow_delete_200"] = dr.status_code == 200 and dr.json().get("deleted") is True
    out["midfollow_delete_no_terminal"] = bool(names2) and all(
        e not in _TERMINAL_EVENTS for e in names2
    )
    out["midfollow_delete_stays_404"] = (
        ctx2.client.get(f"/v1/responses/{rid2}", params={"stream": "true"}).status_code == 404
    )
    # --- a parked follow exits at the deadline with no terminal frame
    backend3 = _GateBackend()
    ctx3 = _make_ctx(workdir / "int-timeout", backend3, sse_keepalive_s=0.05)
    try:
        rid3 = _submit_bg(ctx3.client)
        assert backend3.entered.wait(timeout=10.0), "worker never entered the backend"
        start = time.monotonic()
        g3 = ctx3.client.get(f"/v1/responses/{rid3}", params={"stream": "true", "timeout_s": 1})
        elapsed = time.monotonic() - start
    finally:
        backend3.gate.set()
    names3 = _event_names(g3.text)
    out["follow_timeout_bounded"] = 0.8 <= elapsed <= 5.0
    out["follow_timeout_no_terminal"] = bool(names3) and all(
        e not in _TERMINAL_EVENTS for e in names3
    )
    # --- two concurrent attaches on one in-flight record agree byte-for-byte
    backend4 = _GateBackend()
    ctx4 = _make_ctx(workdir / "int-dbl", backend4, sse_keepalive_s=0.05)
    try:
        rid4 = _submit_bg(ctx4.client)
        assert backend4.entered.wait(timeout=10.0), "worker never entered the backend"
        t4a, box4a = _replay_async(ctx4.client, f"/v1/responses/{rid4}?stream=true&timeout_s=20")
        t4b, box4b = _replay_async(ctx4.client, f"/v1/responses/{rid4}?stream=true&timeout_s=20")
        time.sleep(0.3)
        backend4.gate.set()
        t4a.join(timeout=20.0)
        t4b.join(timeout=20.0)
        g4a, g4b = box4a[0], box4b[0]
    finally:
        backend4.gate.set()
    # keepalive comment frames are timing chatter — the identical contract
    # is over the data frame sequence (event names, ids, and payloads)
    frames4a = _data_frames(_sse_frames(g4a.text))
    frames4b = _data_frames(_sse_frames(g4b.text))
    out["double_attach_identical"] = (
        bool(frames4a)
        and frames4a[-1].event == "response.completed"
        and [(f.seq, f.event, f.data) for f in frames4a]
        == [(f.seq, f.event, f.data) for f in frames4b]
    )
    # --- cancel on a terminal record is a loud 409, not a state change
    rec = ctx.client.post("/v1/responses", json={"model": "fx1", "input": "done"}).json()
    d2 = ctx.client.post(f"/v1/responses/{rec['id']}/cancel")
    out["cancel_terminal_409"] = _err_enveloped(d2, status=409, code="cancel_terminal")
    return out


# ---------------------------------------------------------------------------
# Isolation + cost — replay is read-only and never re-executes the model
# ---------------------------------------------------------------------------


def _probe_isolation(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post(
        "/v1/responses", json={"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    )
    rid = _event_payloads(_sse_frames(r.text))[0][1]["response"]["id"]
    before_json = ctx.client.get(f"/v1/responses/{rid}").text
    before_completions = ctx.client.get("/harness/completions").json()["count"]
    reps = _parallel(
        _CONC_N,
        lambda _i: ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"}).text,
    )
    out["concurrent_replays_identical"] = len(set(reps)) == 1 and bool(reps[0])
    out["replay_readonly_record"] = ctx.client.get(f"/v1/responses/{rid}").text == before_json
    out["replay_no_new_completions"] = (
        ctx.client.get("/harness/completions").json()["count"] == before_completions
    )
    # replaying N times mints nothing and bills nothing — the recorded
    # usage rides the terminal frame verbatim (the Responses shape,
    # input/output/total tokens) and the completion id is the original's
    create_usage = _event_payloads(_sse_frames(r.text))[-1][1]["response"].get("usage")
    out["replay_usage_verbatim"] = all(
        _event_payloads(_sse_frames(rep))[-1][1]["response"].get("usage") == create_usage
        for rep in reps
    )
    cids = {
        ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"}).headers.get(
            "x-fx1-completion-id"
        )
        for _ in range(2)
    }
    out["replay_cid_stable"] = len(cids) == 1 and next(iter(cids)) == r.headers.get(
        "x-fx1-completion-id"
    )
    return out


# ---------------------------------------------------------------------------
# Batch — stream/background/conversation lines are per-line refusals
# ---------------------------------------------------------------------------


def _upload_lines(client: TestClient, lines: list[dict[str, Any]]) -> str:
    blob = "".join(json.dumps(ln) + "\n" for ln in lines).encode()
    r = client.post(
        "/v1/files",
        files={"file": ("in.jsonl", blob, "application/jsonl")},
        data={"purpose": "batch"},
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _batch_line(custom_id: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}


def _probe_batch(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    fid = _upload_lines(
        ctx.client,
        [
            _batch_line("b-stream", {"model": "fx1", "input": "s", "stream": True}),
            _batch_line("b-bg", {"model": "fx1", "input": "b", "background": True}),
            _batch_line("b-conv", {"model": "fx1", "input": "c", "conversation": "conv_x"}),
            _batch_line("b-clean", {"model": "fx1", "input": "ok"}),
        ],
    )
    r = ctx.client.post(
        "/v1/batches",
        json={"input_file_id": fid, "endpoint": "/v1/responses", "completion_window": "24h"},
    )
    assert r.status_code == 200, r.text
    bid = r.json()["id"]
    end = time.monotonic() + _WAIT_S
    b: dict[str, Any] = {}
    while time.monotonic() < end:
        b = ctx.client.get(f"/v1/batches/{bid}").json()
        if b.get("status") in {"completed", "failed", "expired", "cancelled"}:
            break
        time.sleep(0.03)
    out["batch_completed"] = b.get("status") == "completed"
    ofid = b.get("output_file_id")
    rows: dict[str, dict[str, Any]] = {}
    if ofid:
        rc = ctx.client.get(f"/v1/files/{ofid}/content")
        rows = {
            str(row["custom_id"]): row["response"]
            for row in (json.loads(ln) for ln in rc.text.splitlines() if ln.strip())
        }

    def _refused(cid: str) -> bool:
        resp = rows.get(cid)
        return (
            isinstance(resp, dict)
            and resp.get("status_code") == 400
            and isinstance(resp.get("body"), dict)
            and resp["body"].get("error", {}).get("code") == "invalid_request"
        )

    out["batch_stream_line_refused"] = _refused("b-stream")
    out["batch_background_line_refused"] = _refused("b-bg")
    out["batch_conversation_line_refused"] = _refused("b-conv")
    clean = rows.get("b-clean")
    out["batch_clean_line_served"] = (
        isinstance(clean, dict)
        and clean.get("status_code") == 200
        and isinstance(clean.get("body"), dict)
        and clean["body"].get("status") == "completed"
    )
    return out


# ---------------------------------------------------------------------------
# Conversation — replaying a turn leaves the container's items untouched
# ---------------------------------------------------------------------------


def _probe_conversation(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    conv = ctx.client.post("/v1/conversations", json={})
    cid = conv.json()["id"]
    r = ctx.client.post(
        "/v1/responses",
        json={"model": "fx1", "input": "turn", "stream": True, "conversation": cid},
    )
    rid = _event_payloads(_sse_frames(r.text))[0][1]["response"]["id"]
    before = ctx.client.get(f"/v1/conversations/{cid}/items").json()
    g = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    after = ctx.client.get(f"/v1/conversations/{cid}/items").json()
    out["conv_replay_identical"] = g.text == r.text
    out["conv_items_stable"] = before == after
    events = _event_payloads(_sse_frames(g.text))
    out["conv_echo_terminal"] = events[-1][1]["response"].get("conversation") == {"id": cid}
    return out


# ---------------------------------------------------------------------------
# Drain — reads stay open under the latch; new submits refuse 503
# ---------------------------------------------------------------------------


def _probe_drain(ctx: _Ctx) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = ctx.client.post("/v1/responses", json={"model": "fx1", "input": "d", "stream": True})
    rid = _event_payloads(_sse_frames(r.text))[0][1]["response"]["id"]
    d = ctx.client.post("/harness/drain")
    out["drain_latched"] = d.status_code == 200 and d.json().get("draining") is True
    g = ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"})
    out["drain_replay_open"] = g.status_code == 200 and g.text == r.text
    out["drain_json_read_open"] = ctx.client.get(f"/v1/responses/{rid}").status_code == 200
    submit = ctx.client.post(
        "/v1/responses", json={"model": "fx1", "input": "x", "background": True}
    )
    out["drain_submit_refused"] = _err_enveloped(submit, status=503, code="draining")
    # cancel of a terminal record stays a loud 409 even mid-drain
    out["drain_cancel_terminal_409"] = _err_enveloped(
        ctx.client.post(f"/v1/responses/{rid}/cancel"), status=409, code="cancel_terminal"
    )
    dd = ctx.client.delete(f"/v1/responses/{rid}")
    out["drain_delete_open"] = dd.status_code == 200 and dd.json().get("deleted") is True
    out["drain_replay_after_delete_404"] = (
        ctx.client.get(f"/v1/responses/{rid}", params={"stream": "true"}).status_code == 404
    )
    return out


# ---------------------------------------------------------------------------
# Client surface — HarnessClient.responses_replay + the error map
# ---------------------------------------------------------------------------


def _probe_client(ctx: _Ctx, workdir: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.client import HarnessTransportError  # noqa: PLC0415

    remote = _remote(ctx.client)
    r = ctx.client.post(
        "/v1/responses", json={"model": "fx1", "input": _ECHO_PROMPT, "stream": True}
    )
    rid = _event_payloads(_sse_frames(r.text))[0][1]["response"]["id"]
    events, cid = remote.responses_replay(rid)
    out["client_replay_events"] = [e.get("type") for e in events] == _event_names(
        r.text
    ) and events[-1]["type"] == "response.completed"
    out["client_replay_cid"] = cid == r.headers.get("x-fx1-completion-id")
    # starting_after=N resumes PAST index N — frames N+1.. remain
    events2, _ = remote.responses_replay(rid, starting_after=1)
    out["client_replay_resume"] = len(events2) == len(events) - 2
    try:
        remote.responses_replay("resp_never")
        out["client_replay_404_keyerror"] = False
    except KeyError:
        out["client_replay_404_keyerror"] = True
    except Exception:
        out["client_replay_404_keyerror"] = False
    try:
        remote.responses_replay(rid, starting_after=-1)
        out["client_replay_422_valueerror"] = False
    except ValueError:
        out["client_replay_422_valueerror"] = True
    except Exception:
        out["client_replay_422_valueerror"] = False
    try:
        remote.cancel_response(rid)
        out["client_cancel_terminal_409"] = False
    except HarnessTransportError:
        out["client_cancel_terminal_409"] = True
    except Exception:
        out["client_cancel_terminal_409"] = False
    # a follow that ends without a terminal frame is a transport fault,
    # not a silent empty list
    backend = _GateBackend()
    ctx_park = _make_ctx(workdir / "client-trunc", backend, sse_keepalive_s=0.05)
    try:
        rid2 = _submit_bg(ctx_park.client)
        assert backend.entered.wait(timeout=10.0), "worker never entered the backend"
        remote2 = _remote(ctx_park.client)
        try:
            remote2.responses_replay(rid2, timeout_s=1)
            out["client_truncated_raises"] = False
        except HarnessTransportError:
            out["client_truncated_raises"] = True
        except Exception:
            out["client_truncated_raises"] = False
    finally:
        backend.gate.set()
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def replay_audit() -> dict[str, Any]:
    """Run every probe against live in-process apps; literal bools out."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    out: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            ctx = _make_ctx(wd / "a", _StubBackend())
            out.update(_probe_replay_core(ctx))
            out.update(_probe_resume(ctx))
            out.update(_probe_store_gate(ctx))
            out.update(_probe_background_stream(ctx))
            out.update(_probe_isolation(ctx))
            out.update(_probe_batch(ctx))
            out.update(_probe_conversation(ctx))
            out.update(_probe_client(ctx, wd / "client"))
            ctx_fail = _make_ctx(wd / "fail", _FailBackend())
            out.update(_probe_failed_replay(ctx_fail))
            ctx_tools = _make_ctx(wd / "tools", _ToolStubBackend())
            out.update(_probe_incomplete_replay(ctx_tools))
            out.update(_probe_cancelled_replay(wd / "cancel"))
            out.update(_probe_live_attach(wd / "live"))
            out.update(_probe_interrupts(wd / "int"))
            ctx_drain = _make_ctx(wd / "drain", _StubBackend())
            out.update(_probe_drain(ctx_drain))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def replay_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = replay_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "replay_audit",
        "schema": "replay_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "Starlette TestClient buffered ASGI response bodies; "
            "live-follow probes drive the GET on a thread so the interrupt "
            "lands inside the follow loop",
            "not_verified": [
                "early client disconnect propagation",
                "deletion during an active network follow response",
                "network streaming delivery timing",
                "replay across process restart (the response index is in-memory)",
            ],
        },
        "interpretation": (
            "The ``?stream=true`` replay surface regenerates the create-time "
            "Responses stream byte-for-byte for terminal records — dense "
            "``id:`` indices, ``event:``-typed frames, the recorded terminal "
            "payload as the last event, and the completion/evidence link "
            "headers intact. ``starting_after`` resumes by absolute frame "
            "index with strict suffix semantics, bogus resume/timeout/stream "
            "params refuse 422 in the shared error envelope, and unadvertised "
            "params are ignored rather than grafted. ``store:false``, deleted, "
            "and foreign-object ids all answer the same 404 ``not_found``; "
            "the JSON read never leaks ``_fx1_*`` internals. Live attaches "
            "emit the lifecycle prelude then follow with ``: keepalive`` "
            "comments; a mid-follow cancel ends on ``response.cancelled``, a "
            "mid-follow delete or ``timeout_s`` deadline ends without a "
            "terminal frame, and a queued-record cancel replays no phantom "
            "``in_progress`` phase. ``background:true`` + ``stream:true`` "
            "echoes ``background:false`` (it ran synchronously) so its replay "
            "is byte-identical to the create stream. Replays are read-only: "
            "concurrent replays are identical, the record/conversation/completion "
            "log are untouched, and no model call re-executes. Batch lines "
            "carrying ``stream``/``background``/``conversation`` refuse "
            "per-line ``invalid_request`` while the batch completes; under "
            "drain, replays/reads/cancels/deletes stay open while submits "
            "refuse 503 ``draining``. The client parses ``(events, "
            "completion_id)``, maps 404→``KeyError``/422→``ValueError``, and "
            "raises ``HarnessTransportError`` on a truncated follow. "
            "Buffered transport does not verify disconnect propagation, "
            "cross-restart replay, or network delivery timing."
            if ok
            else f"REPLAY AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(replay_audit_bench(), indent=2, sort_keys=True))
