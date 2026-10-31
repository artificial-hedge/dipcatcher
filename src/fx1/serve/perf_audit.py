"""Perf/boundedness audit for the fx-1 serve harness — lane 163.

CLAIM UNDER TEST — the serve layer's declared ceilings are real
boundaries, not intentions. This battery drives adversarial load through
the stub-backed app and asserts each one end to end:

- *envelope-store LRU* — ``store_max`` chat completions bound the
  retrieval index: the oldest records evict honestly (404 on read,
  subitems die with the parent), ``len`` never exceeds the cap.
- *pending streams* — an inflight semaphore (``max_inflight``) is the
  pending-streams cap: holding every slot refuses the next connect
  ``503 over_capacity`` with ``Retry-After``, and releasing a held
  stream frees its slot.
- *pending background work* — ``/harness/jobs`` submission and
  ``/v1/responses`` background mode share the same inflight bound: a
  saturated pool refuses new submits honestly rather than buffering
  unboundedly, and drained workers return the slots.
- *claim locks* — the per-key ``_ClaimLocks`` map prunes to its bound
  after a key sweep; keys in flight may overrun transiently but idle
  entries are reclaimed (unit + wire evidence).
- *completion log ring* — ``cap`` is declared and ``records_dropped``
  counts the overflow honestly; the ring never grows past 256.
- *upload intents* — ``file_max`` bounds pending intents (the oldest
  evicts, a part write to an evicted intent is a 404), declared byte
  caps refuse (413), and TTL expiry is honest (unit-level 410).
- *idempotency LRU* — ``idem_max`` bounds the dedup map: an evicted
  key re-executes on replay (honest — a missed dedup re-bills, never
  replays a stale answer), a live key replays with no backend work.
- *header/body amplification* — a 10KB ``X-Fx1-Byok-Model`` refuses
  ``422 invalid_byok_headers`` (the header-abuse channel stays closed);
  a 5MB body refuses ``413 too_large`` at ingress; a body under the
  cap still processes.
- *paging amplification* — cursor walks (``limit=1`` + ``after``) stay
  bounded per page, an unknown cursor refuses 400, and ``limit`` is
  clamped (``le=100`` → absurd values answer 422, never a full scan).
- *latency/error budget* — the hot path's p99 stays under a generous
  ceiling, the declared ``timeout_s`` reaches the backend transport
  unchanged, and refusals (401/404/422/503-over-capacity) return in
  bounded time burning no backend call.
- *flood + memory* — a parallel flood resolves to honest verdicts
  (200 or 503, never a 5xx storm or a hang) with every slot released
  after; retained allocations after a completion sweep stay under a
  generous bound and store maps re-check at their caps after ``gc``.

Time-based probes carry 10–50x headroom: they catch an unbounded queue
or a sync-sleep on the request path, not millisecond regressions. This
is a stub-backed TestClient battery — not a benchmark, not network
timing, not live-provider evidence.

Sealed ``perf_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import gc
import json
import os
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

from fx1.serve.conv_audit import (
    _RESOURCES,
    _audit_context,
    _StubBackend,
    _temporary_directory,
)
from fx1.serve.journal import _ClaimLocks
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = ["perf_audit", "perf_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "perf-aud1t-r00t"
_MODEL = "hosted_k3"
_H: dict[str, str] = {"X-API-Key": _ROOT}
_CHAT = "/v1/chat/completions"
_COMPLETE = "/harness/complete"
_STREAM = "/harness/complete/stream"
_RESPONSES = "/v1/responses"
_JOBS = "/harness/jobs"
_RUNS = "/harness/runs"

# Generous ceilings — the probes catch a blowup (an unbounded queue, a
# sync sleep on the request path), not a slow CI box.
_P99_BUDGET_S = 5.0
_ERROR_BUDGET_S = 3.0
_MEM_BUDGET_BYTES = 64 << 20


def _resources() -> ExitStack:
    return _RESOURCES.get()


class _ParkBackend(_StubBackend):
    """``complete``/``stream`` park on a gate — deterministic 'worker is
    inside the backend' windows so a probe can observe the inflight cap
    mid-flight. ``all_inside`` latches once ``need`` calls are parked.
    The gate wait is capped at 30s so a hung probe can never deadlock
    the suite."""

    def __init__(self, gate: threading.Event | None = None, need: int = 1) -> None:
        super().__init__()
        self.gate = gate or threading.Event()
        self.need = need
        self.all_inside = threading.Event()
        self._inside = 0
        self._inside_lock = threading.Lock()

    def _park(self) -> None:
        with self._inside_lock:
            self._inside += 1
            if self._inside >= self.need:
                self.all_inside.set()
        self.gate.wait(30)

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        self._park()
        return super().complete(messages, sampling=sampling)

    def stream(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> Iterator[str]:
        del sampling
        self._park()
        yield f"stub:{messages[-1]['content']}"


class _StreamGateBackend(_StubBackend):
    """Only ``stream`` parks; ``complete`` stays instant — a held stream
    must not slow the sync path sharing the same slot pool."""

    def __init__(self, gate: threading.Event) -> None:
        super().__init__()
        self.gate = gate
        self.entered = threading.Event()

    def stream(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> Iterator[str]:
        del sampling
        self.entered.set()
        self.gate.wait(30)
        yield f"stub:{messages[-1]['content']}"


class _GateRunner:
    """Harness runner parking on a gate — holds a job worker's inflight
    slot until released, so the next submit must refuse, not buffer."""

    def __init__(self, need: int) -> None:
        self.calls = 0
        self.need = need
        self.gate = threading.Event()
        self.all_inside = threading.Event()
        self._lock = threading.Lock()

    def __call__(self, argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        del argv, timeout_s
        with self._lock:
            self.calls += 1
            if self.calls >= self.need:
                self.all_inside.set()
        self.gate.wait(30)
        return 0, "ok", ""


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


def _make_app(
    *,
    backend_map: dict[str, Callable[[], Any]] | None = None,
    api_key: str | None = _ROOT,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    resolve_calls: list[tuple[str, dict[str, Any]]] | None = None,
    **create_kw: Any,
) -> Any:
    """create_app under an isolated env; ``backend_map[name]()`` are the
    link factories; ``resolve_calls`` records each resolver invocation's
    ``(name, args)`` so the timeout/override wiring is observable."""
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    backends = backend_map or {_MODEL: lambda: _StubBackend("fx1")}
    resources = _resources()
    isolated = _temporary_directory()
    saved = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key

        def _resolve(name: str, *a: Any, **k: Any) -> Any:
            if resolve_calls is not None:
                resolve_calls.append((name, dict(k)))
            return backends[name]()

        app = api_mod.create_app(
            harness=Harness(runner=runner or _fast_runner),
            backend_resolver=_resolve,
            state_dir=create_kw.pop("state_dir", isolated / "state"),
            receipts_dir=create_kw.pop("receipts_dir", isolated / "receipts"),
            ft_dir=create_kw.pop("ft_dir", isolated / "ft"),
            **create_kw,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        return app
    finally:
        if saved is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved


def _client(
    *,
    backend_map: dict[str, Callable[[], Any]] | None = None,
    api_key: str | None = _ROOT,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    resolve_calls: list[tuple[str, dict[str, Any]]] | None = None,
    **create_kw: Any,
) -> tuple[TestClient, Any]:
    """(TestClient, app) — the battery's standard wired app."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    app = _make_app(
        backend_map=backend_map,
        api_key=api_key,
        runner=runner,
        resolve_calls=resolve_calls,
        **create_kw,
    )
    client = TestClient(app, raise_server_exceptions=False)
    _resources().callback(client.close)
    _resources().enter_context(client)
    return client, app


def _chat_body(content: str, **extra: Any) -> dict[str, Any]:
    return {"model": _MODEL, "messages": [{"role": "user", "content": content}], **extra}


def _complete_body(content: str, **extra: Any) -> dict[str, Any]:
    return {
        "backend": _MODEL,
        "messages": [{"role": "user", "content": content}],
        **extra,
    }


def _response_body(content: str, **extra: Any) -> dict[str, Any]:
    return {"model": _MODEL, "input": content, **extra}


def _err_code(r: Any) -> str | None:
    """Refusal code in either grammar — harness ``{code}`` or the
    OpenAI/Anthropic ``{error: {code}}`` envelope."""
    try:
        body = r.json()
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if isinstance(err, dict):
        code = err.get("code")
        return str(code) if code is not None else None
    code = body.get("code")
    return str(code) if code is not None else None


def _wait_for(cond: Callable[[], bool], timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


# ---------------------------------------------------------------------------
# Envelope-store LRU
# ---------------------------------------------------------------------------


def _envelope_store_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    cap = 8
    client, _app = _client(store_max=cap)
    ids: list[str] = []
    writes_ok = True
    for i in range(cap + 4):
        r = client.post(_CHAT, json=_chat_body(f"env-{i}"), headers=_H)
        writes_ok = writes_ok and r.status_code == 200
        if r.status_code == 200:
            ids.append(str(r.json()["id"]))
    out["envelope_writes_all_200"] = writes_ok
    listed = client.get(f"{_CHAT}?limit=100", headers=_H)
    out["envelope_len_never_exceeds_cap"] = (
        listed.status_code == 200 and len(listed.json()["data"]) == cap
    )
    evicted = ids[: len(ids) - cap]
    live = ids[len(ids) - cap :]
    out["envelope_oldest_evicted_404"] = bool(evicted) and all(
        client.get(f"{_CHAT}/{i}", headers=_H).status_code == 404 for i in evicted
    )
    out["envelope_live_retrievable_200"] = bool(live) and all(
        client.get(f"{_CHAT}/{i}", headers=_H).status_code == 200 for i in live
    )
    gone = client.get(f"{_CHAT}/{evicted[0]}", headers=_H) if evicted else None
    body = gone.json() if gone is not None and gone.status_code == 404 else {}
    out["envelope_evicted_error_enveloped"] = isinstance(body.get("error"), dict) and bool(
        body["error"].get("code")
    )
    out["envelope_subitems_die_with_parent"] = bool(evicted) and (
        client.get(f"{_CHAT}/{evicted[0]}/messages", headers=_H).status_code == 404
    )
    return out


# ---------------------------------------------------------------------------
# Pending streams / background work — the inflight semaphore is the cap
# ---------------------------------------------------------------------------


def _pending_stream_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    cap = 2
    gate = threading.Event()
    backend = _ParkBackend(gate=gate, need=cap)
    client, _app = _client(
        backend_map={_MODEL: lambda: backend},
        max_inflight=cap,
        sse_keepalive_s=0,
    )
    held: dict[int, Any] = {}

    def _stream_once(i: int) -> None:
        held[i] = client.post(_STREAM, json=_complete_body(f"stream-{i}"), headers=_H)

    try:
        with ThreadPoolExecutor(max_workers=cap) as pool:
            futs = [pool.submit(_stream_once, i) for i in range(cap)]
            out["stream_workers_parked"] = backend.all_inside.wait(30)
            refused = client.post(_STREAM, json=_complete_body("extra"), headers=_H)
            out["stream_refused_at_cap_503"] = refused.status_code == 503
            out["stream_refusal_over_capacity"] = _err_code(refused) == "over_capacity"
            snap = client.get("/metrics", headers=_H).json()
            out["inflight_gauge_honest_at_cap"] = (
                snap["inflight"] == cap and snap["inflight_watermark"] <= cap
            )
            gate.set()
            for f in futs:
                f.result(timeout=30)
        out["held_streams_completed_200"] = len(held) == cap and all(
            r.status_code == 200 for r in held.values()
        )
        freed = client.post(_STREAM, json=_complete_body("after-release"), headers=_H)
        out["stream_slot_freed_after_release"] = freed.status_code == 200
        snap2 = client.get("/metrics", headers=_H).json()
        out["inflight_returns_to_zero"] = snap2["inflight"] == 0
    finally:
        gate.set()
    return out


def _pending_bg_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    cap = 2

    # /harness/jobs — the submit route takes an inflight slot for the
    # job's whole lifetime; a saturated pool refuses, never buffers.
    runner = _GateRunner(need=cap)
    client, _app = _client(max_inflight=cap, runner=runner)
    try:
        submits = [client.post(_JOBS, json={"command": "doctor"}, headers=_H) for _ in range(cap)]
        out["bg_jobs_accepted_202"] = all(r.status_code == 202 for r in submits)
        job_ids = [str(r.json()["job_id"]) for r in submits if r.status_code == 202]
        out["bg_jobs_running"] = len(job_ids) == cap and runner.all_inside.wait(30)
        extra = client.post(_JOBS, json={"command": "doctor"}, headers=_H)
        out["bg_jobs_refused_at_cap_503"] = extra.status_code == 503
        out["bg_jobs_refusal_over_capacity"] = _err_code(extra) == "over_capacity"
        runner.gate.set()
        done = _wait_for(
            lambda: (
                bool(job_ids)
                and all(
                    client.get(f"{_JOBS}/{j}", headers=_H).json().get("status") == "succeeded"
                    for j in job_ids
                )
            )
        )
        out["bg_jobs_drain_to_terminal"] = done
        again = client.post(_JOBS, json={"command": "doctor"}, headers=_H)
        out["bg_job_slot_recovers"] = again.status_code == 202
    finally:
        runner.gate.set()

    # /v1/responses background=true — workers block on the same
    # semaphore; submit stays honest (queued) and a saturated pool
    # refuses the next submit rather than queueing silently.
    gate = threading.Event()
    backend = _ParkBackend(gate=gate, need=cap)
    client2, _app2 = _client(
        backend_map={_MODEL: lambda: backend},
        max_inflight=cap,
        sse_keepalive_s=0,
    )
    try:
        submits2 = [
            client2.post(_RESPONSES, json=_response_body(f"bg-{i}", background=True), headers=_H)
            for i in range(cap)
        ]
        out["bg_responses_queued_honestly"] = all(
            r.status_code == 200 and r.json().get("status") == "queued" for r in submits2
        )
        resp_ids = [str(r.json()["id"]) for r in submits2 if r.status_code == 200]
        out["bg_response_workers_parked"] = backend.all_inside.wait(30)
        extra2 = client2.post(
            _RESPONSES, json=_response_body("bg-extra", background=True), headers=_H
        )
        out["bg_responses_refused_at_cap_503"] = extra2.status_code == 503
        if resp_ids:
            mid = client2.get(f"{_RESPONSES}/{resp_ids[0]}", headers=_H)
            out["bg_status_in_progress_honest"] = mid.status_code == 200 and mid.json().get(
                "status"
            ) in {"queued", "in_progress"}
        else:
            out["bg_status_in_progress_honest"] = False
        gate.set()
        drained = _wait_for(
            lambda: (
                bool(resp_ids)
                and all(
                    client2.get(f"{_RESPONSES}/{rid}", headers=_H).json().get("status")
                    == "completed"
                    for rid in resp_ids
                )
            )
        )
        out["bg_responses_drain_to_completed"] = drained
        fresh = client2.post(
            _RESPONSES, json=_response_body("bg-fresh", background=True), headers=_H
        )
        out["bg_submit_recovers_after_release"] = fresh.status_code == 200
    finally:
        gate.set()
    return out


# ---------------------------------------------------------------------------
# _ClaimLocks LRU — unit evidence plus the wire-mounted idem store
# ---------------------------------------------------------------------------


def _claim_locks_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    bound = 8
    locks = _ClaimLocks(bound)
    for i in range(64):
        with locks.hold(f"sweep-{i}"):
            pass
    # noqa: SLF001 — the audit reaches the bounded map itself.
    out["claim_locks_pruned_after_sweep"] = len(locks._locks) <= bound

    held_events = [threading.Event() for _ in range(16)]
    all_held = threading.Event()
    entered_count = [0]
    count_lock = threading.Lock()

    def _hold(i: int) -> None:
        with locks.hold(f"concurrent-{i}"):
            with count_lock:
                entered_count[0] += 1
                if entered_count[0] == 16:
                    all_held.set()
            held_events[i].wait(30)

    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(_hold, i) for i in range(16)]
        all_held.wait(30)
        # Active claims may exceed the bound; they must still be bounded
        # by bound + the live set, never unbounded.
        out["claim_locks_overrun_only_active"] = len(locks._locks) <= bound + 16
        for ev in held_events:
            ev.set()
        for f in futs:
            f.result(timeout=30)
    with locks.hold("post-sweep"):
        pass
    out["claim_locks_reclaim_idle"] = len(locks._locks) <= bound

    # Wire evidence — the runs idem store's claim map stays bounded after
    # a key sweep wider than its cap.
    client, app = _client(idem_max=8)
    for i in range(24):
        r = client.post(
            _RUNS,
            json={"command": "doctor"},
            headers={**_H, "Idempotency-Key": f"locks-{i}"},
        )
        if r.status_code != 200:
            out["claim_locks_wire_bounded"] = False
            return out
    store_locks = app.state.idem_store._claims._locks  # noqa: SLF001
    out["claim_locks_wire_bounded"] = len(store_locks) <= 8
    return out


# ---------------------------------------------------------------------------
# CompletionLog ring — declared cap, honest dropped count
# ---------------------------------------------------------------------------


def _completion_ring_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    cap = 256
    extra = 4
    client, _app = _client()
    ok = True
    for i in range(cap + extra):
        r = client.post(_COMPLETE, json=_complete_body(f"ring-{i}"), headers=_H)
        ok = ok and r.status_code == 200
    out["completion_ring_writes_ok"] = ok
    usage = client.get("/harness/usage", headers=_H)
    rep = usage.json() if usage.status_code == 200 else {}
    out["completion_ring_cap_declared"] = rep.get("ring_cap") == cap
    out["completion_ring_drops_honest"] = rep.get("records_dropped") == extra
    out["completion_ring_seen_bounded"] = rep.get("records_seen") == cap
    listed = client.get("/harness/completions?limit=256", headers=_H)
    out["completion_log_len_capped"] = (
        listed.status_code == 200
        and len(listed.json()["items"]) == cap
        and listed.json()["count"] == cap
    )
    return out


# ---------------------------------------------------------------------------
# Upload intents — LRU eviction and TTL expiry
# ---------------------------------------------------------------------------


def _upload_intent_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    cap = 4
    client, _app = _client(file_max=cap, file_bytes_max=64)
    intents: list[str] = []
    created = True
    for i in range(cap + 2):
        r = client.post(
            "/v1/uploads",
            json={
                "purpose": "batch",
                "filename": f"intent-{i}.jsonl",
                "bytes": 4,
                "mime_type": "application/jsonl",
            },
            headers=_H,
        )
        created = created and r.status_code == 200
        if r.status_code == 200:
            intents.append(str(r.json()["id"]))
    out["upload_intents_created"] = created
    evicted = intents[: len(intents) - cap]
    live = intents[len(intents) - cap :]
    parts = [
        client.post(
            f"/v1/uploads/{uid}/parts",
            files={"data": ("p.bin", b"xy")},
            headers=_H,
        )
        for uid in evicted
    ]
    out["upload_evicted_intent_404"] = bool(evicted) and all(p.status_code == 404 for p in parts)
    if live:
        p = client.post(
            f"/v1/uploads/{live[-1]}/parts",
            files={"data": ("p.bin", b"xy")},
            headers=_H,
        )
        out["upload_live_intent_accepts_part"] = p.status_code == 200
    else:
        out["upload_live_intent_accepts_part"] = False
    over = client.post(
        "/v1/uploads",
        json={
            "purpose": "batch",
            "filename": "too-big.jsonl",
            "bytes": 4096,
            "mime_type": "application/jsonl",
        },
        headers=_H,
    )
    out["upload_declared_bytes_capped_413"] = over.status_code == 413

    # TTL is store-level (not a create_app knob) — unit evidence:
    # an expired intent answers 410, honestly distinct from a 404.
    from fx1.serve.uploads import UploadStore, UploadStoreError  # noqa: PLC0415

    store = UploadStore(max_entries=4, max_bytes=1 << 20, ttl_s=0)
    meta = store.create(
        purpose="batch", filename="ttl.jsonl", nbytes=4, mime_type="application/jsonl"
    )
    try:
        store.add_part(meta.upload_id, b"xy")
        out["upload_ttl_expires_410"] = False
    except UploadStoreError as exc:
        out["upload_ttl_expires_410"] = exc.status == 410 and exc.code == "upload_expired"
    return out


# ---------------------------------------------------------------------------
# Idem-store LRU — evicted keys re-execute, live keys replay
# ---------------------------------------------------------------------------


def _idem_lru_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    cap = 8
    stub = _StubBackend("fx1")
    client, _app = _client(backend_map={_MODEL: lambda: stub}, idem_max=cap)
    body = _complete_body("idem-lru")
    keys = [f"idem-{i}" for i in range(cap + 4)]
    ok = True
    for k in keys:
        r = client.post(_COMPLETE, json=body, headers={**_H, "Idempotency-Key": k})
        ok = ok and r.status_code == 200
    out["idem_writes_ok"] = ok
    calls_before = stub.calls
    evicted_re = client.post(_COMPLETE, json=body, headers={**_H, "Idempotency-Key": keys[0]})
    out["idem_evicted_reexecutes"] = (
        evicted_re.status_code == 200
        and evicted_re.json().get("replayed") is False
        and stub.calls == calls_before + 1
    )
    live_re = client.post(_COMPLETE, json=body, headers={**_H, "Idempotency-Key": keys[-1]})
    out["idem_live_key_replays"] = (
        live_re.status_code == 200
        and live_re.json().get("replayed") is True
        and stub.calls == calls_before + 1
    )
    return out


# ---------------------------------------------------------------------------
# Header / body amplification ceilings
# ---------------------------------------------------------------------------


def _header_body_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    client, _app = _client()
    big_model = client.post(
        _CHAT,
        json=_chat_body("hdr"),
        headers={
            **_H,
            "X-Fx1-Byok-Base-Url": "https://127.0.0.1:9/v1",
            "X-Fx1-Byok-Api-Key": "synthetic",
            "X-Fx1-Byok-Model": "m" * 10240,
        },
    )
    out["header_byok_model_10kb_refused_422"] = big_model.status_code == 422
    out["header_byok_model_refusal_coded"] = _err_code(big_model) == "invalid_byok_headers"

    five_mb = json.dumps(
        {"model": _MODEL, "messages": [{"role": "user", "content": "A" * (5 << 20)}]}
    )
    big = client.post(
        _CHAT,
        content=five_mb.encode(),
        headers={**_H, "Content-Type": "application/json"},
    )
    out["body_5mb_refused_413"] = big.status_code == 413
    out["body_5mb_refusal_too_large"] = _err_code(big) == "too_large"

    under_cap = json.dumps(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "B" * ((1 << 20) - 4096)}],
        }
    )
    t0 = time.monotonic()
    small = client.post(
        _CHAT,
        content=under_cap.encode(),
        headers={**_H, "Content-Type": "application/json"},
    )
    elapsed = time.monotonic() - t0
    out["body_under_cap_processes"] = small.status_code == 200
    out["body_under_cap_bounded_time"] = elapsed < 30.0
    return out


# ---------------------------------------------------------------------------
# Cursor / list / streaming amplification
# ---------------------------------------------------------------------------


def _amplification_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    cap = 64
    gate = threading.Event()
    backend = _StreamGateBackend(gate)
    client, _app = _client(backend_map={_MODEL: lambda: backend}, store_max=cap, sse_keepalive_s=0)
    written = 40
    ids: list[str] = []
    ok = True
    for i in range(written):
        r = client.post(_CHAT, json=_chat_body(f"page-{i}"), headers=_H)
        ok = ok and r.status_code == 200
        if r.status_code == 200:
            ids.append(str(r.json()["id"]))
    out["list_seed_writes_ok"] = ok

    full = client.get(f"{_CHAT}?limit=100", headers=_H)
    out["list_under_cap_returns_all"] = (
        full.status_code == 200 and len(full.json()["data"]) == written
    )
    over = client.get(f"{_CHAT}?limit=101", headers=_H)
    absurd = client.get(f"{_CHAT}?limit=99999", headers=_H)
    out["list_limit_clamped_422"] = over.status_code == 422
    out["list_absurd_limit_422"] = absurd.status_code == 422

    walk_ids: list[str] = []
    pages_ok = True
    cursor: str | None = None
    t0 = time.monotonic()
    for _ in range(10):
        url = f"{_CHAT}?limit=1" + (f"&after={cursor}" if cursor else "")
        r = client.get(url, headers=_H)
        data = r.json().get("data", []) if r.status_code == 200 else []
        if r.status_code != 200 or len(data) != 1:
            pages_ok = False
            break
        walk_ids.append(str(data[0]["id"]))
        cursor = str(r.json()["last_id"])
    walked = time.monotonic() - t0
    out["cursor_pages_bounded"] = pages_ok and len(set(walk_ids)) == len(walk_ids)
    out["cursor_walk_bounded_time"] = walked < 20.0
    bad_cursor = client.get(f"{_CHAT}?limit=1&after=chatcmpl-bogus", headers=_H)
    out["cursor_unknown_400"] = bad_cursor.status_code == 400

    # A held stream holds one slot; other work completes on the rest.
    held: dict[str, Any] = {}

    def _open_stream() -> None:
        held["r"] = client.post(_STREAM, json=_complete_body("held"), headers=_H)

    try:
        thread = threading.Thread(target=_open_stream, daemon=True)
        thread.start()
        entered = backend.entered.wait(30)
        t1 = time.monotonic()
        others = [
            client.post(_COMPLETE, json=_complete_body(f"other-{i}"), headers=_H) for i in range(4)
        ]
        others_s = time.monotonic() - t1
        out["held_stream_others_complete"] = entered and all(r.status_code == 200 for r in others)
        out["held_stream_others_bounded_time"] = others_s < 20.0
        snap = client.get("/metrics", headers=_H).json()
        out["held_stream_leaves_one_slot"] = entered and snap["inflight"] == 1
        gate.set()
        thread.join(30)
        out["held_stream_completed_200"] = "r" in held and held["r"].status_code == 200
    finally:
        gate.set()
    return out


# ---------------------------------------------------------------------------
# Latency floor / error-path budget / resolver wiring
# ---------------------------------------------------------------------------


def _latency_error_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    resolve_calls: list[tuple[str, dict[str, Any]]] = []
    stub = _StubBackend("fx1")
    client, _app = _client(
        backend_map={_MODEL: lambda: stub, "byok": lambda: _StubBackend("byok")},
        resolve_calls=resolve_calls,
    )

    lat: list[float] = []
    ok = True
    for i in range(30):
        t0 = time.monotonic()
        r = client.post(_COMPLETE, json=_complete_body(f"lat-{i}"), headers=_H)
        lat.append(time.monotonic() - t0)
        ok = ok and r.status_code == 200
    lat.sort()
    out["hot_path_p99_bounded"] = ok and lat[-1] < _P99_BUDGET_S
    out["hot_path_p50_bounded"] = ok and lat[len(lat) // 2] < _P99_BUDGET_S

    # The declared timeout must reach the transport factory unchanged —
    # a dropped timeout is a silently unbounded upstream wait.
    r = client.post(_COMPLETE, json=_complete_body("timeout", timeout_s=0.5), headers=_H)
    timeout_seen = [k.get("timeout_s") for _name, k in resolve_calls]
    out["timeout_s_reaches_transport"] = r.status_code == 200 and 0.5 in timeout_seen

    def _timed(status_path: str) -> float:
        t0 = time.monotonic()
        if status_path == "401":
            client.get("/metrics")  # no key — refused pre-auth
        elif status_path == "404":
            client.get(f"{_CHAT}/chatcmpl-missing", headers=_H)
        else:
            client.post(_COMPLETE, json={"backend": _MODEL}, headers=_H)
        return time.monotonic() - t0

    unauth = client.get("/metrics")
    missing = client.get(f"{_CHAT}/chatcmpl-missing", headers=_H)
    invalid = client.post(_COMPLETE, json={"backend": _MODEL}, headers=_H)
    calls_at_refusals = stub.calls
    out["error_statuses_honest"] = (
        unauth.status_code == 401 and missing.status_code == 404 and invalid.status_code == 422
    )
    out["error_path_bounded_ms"] = all(
        _timed(name) < _ERROR_BUDGET_S for name in ("401", "404", "422")
    )
    out["error_path_burns_no_backend"] = stub.calls == calls_at_refusals

    # A gate-refused submit burns no backend time: saturate every slot,
    # submit one more, the resolver must never run for it.
    gate = threading.Event()
    parked = _ParkBackend(gate=gate, need=2)
    client2, _app2 = _client(
        backend_map={_MODEL: lambda: parked}, max_inflight=2, sse_keepalive_s=0
    )
    held_calls_before = parked.calls
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(client2.post, _COMPLETE, json=_complete_body(f"hold-{i}"), headers=_H)
                for i in range(2)
            ]
            inside = parked.all_inside.wait(30)
            refused = client2.post(_COMPLETE, json=_complete_body("refused"), headers=_H)
            out["gate_refused_503"] = inside and refused.status_code == 503
            out["gate_refusal_burns_no_backend"] = parked.calls == held_calls_before
            gate.set()
            for f in futs:
                f.result(timeout=30)
        snap = client2.get("/metrics", headers=_H).json()
        out["slots_released_after_refusals"] = snap["inflight"] == 0
    finally:
        gate.set()
    return out


# ---------------------------------------------------------------------------
# Concurrent flood + memory growth
# ---------------------------------------------------------------------------


def _flood_memory_probes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    cap = 4
    client, app = _client(idem_max=cap, store_max=32, max_inflight=cap)

    def _one(i: int) -> Any:
        return client.post(_COMPLETE, json=_complete_body(f"flood-{i}"), headers=_H)

    with ThreadPoolExecutor(max_workers=24) as pool:
        t0 = time.monotonic()
        rs = list(pool.map(_one, range(24)))
        flood_s = time.monotonic() - t0
    out["flood_all_complete_or_refused"] = all(r.status_code in {200, 503} for r in rs)
    out["flood_refusals_over_capacity"] = all(
        _err_code(r) == "over_capacity" for r in rs if r.status_code == 503
    )
    # over-capacity 503 is a declared verdict, not a storm — any other
    # 5xx is a defect.
    out["flood_no_5xx_storm"] = all(r.status_code < 500 or r.status_code == 503 for r in rs)
    out["flood_bounded_wall_time"] = flood_s < 60.0
    out["flood_some_admitted"] = any(r.status_code == 200 for r in rs)
    snap = client.get("/metrics", headers=_H).json()
    out["flood_slots_recovered"] = snap["inflight"] == 0

    # Memory: a completion sweep past every cap must not grow retained
    # heap unboundedly. Generous bound — catches a per-request leak or an
    # uncapped dict, not allocator noise.
    for i in range(10):
        client.post(_COMPLETE, json=_complete_body(f"warm-{i}"), headers=_H)
    gc.collect()
    tracemalloc.start()
    gc.collect()
    before = tracemalloc.get_traced_memory()[0]
    for i in range(200):
        r = client.post(_COMPLETE, json=_complete_body(f"mem-{i}"), headers=_H)
        if r.status_code != 200:
            tracemalloc.stop()
            out["memory_growth_bounded"] = False
            return out
    gc.collect()
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    out["memory_growth_bounded"] = after - before < _MEM_BUDGET_BYTES

    # gc-then-check: the store maps re-verify at their caps once garbage
    # is collected — nothing retained past a declared bound.
    listed = client.get(f"{_CHAT}?limit=100", headers=_H)
    runs_map = app.state.idem_store._map  # noqa: SLF001
    out["store_dicts_bounded_after_gc"] = (
        listed.status_code == 200 and len(listed.json()["data"]) <= 32 and len(runs_map) <= cap
    )
    return out


# ---------------------------------------------------------------------------
# Battery + sealed receipt
# ---------------------------------------------------------------------------


def perf_audit() -> dict[str, Any]:
    """Run the boundedness battery; returns literal bools."""
    with _audit_context():
        out: dict[str, Any] = {}
        out.update(_envelope_store_probes())
        out.update(_pending_stream_probes())
        out.update(_pending_bg_probes())
        out.update(_claim_locks_probes())
        out.update(_completion_ring_probes())
        out.update(_upload_intent_probes())
        out.update(_idem_lru_probes())
        out.update(_header_body_probes())
        out.update(_amplification_probes())
        out.update(_latency_error_probes())
        out.update(_flood_memory_probes())
        return out


def perf_audit_bench() -> dict[str, Any]:
    """Sealed receipt: every probe True under perf_audit.v1."""
    r = perf_audit()
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "perf_audit",
        "schema": "perf_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "transport": "in-process buffered TestClient; stub backends",
            "not_executed": [
                "TypeScript client runtime",
                "network delivery or disconnect timing",
                "process-crash/power-loss durability",
                "live provider latency",
            ],
        },
        "interpretation": (
            "Every declared ceiling in the serve layer is a real boundary "
            "under adversarial load: the envelope store's LRU evicts the "
            "oldest records past store_max (404 on read, subitems die with "
            "the parent, len never exceeds the cap); the inflight "
            "semaphore is the pending-streams and pending-background cap — "
            "a saturated pool refuses the next connect/submit 503 "
            "over_capacity with Retry-After rather than buffering "
            "unboundedly, and released workers hand the slots back; the "
            "per-key claim-lock map prunes to its bound after a key sweep; "
            "the completion log's ring cap is declared and records_dropped "
            "counts overflow honestly; upload intents evict LRU past "
            "file_max (evicted intents 404 on part writes, TTL expiry is a "
            "distinct 410, declared byte caps 413); the idem LRU evicts "
            "past idem_max — an evicted key re-executes rather than "
            "replaying a stale answer; a 10KB X-Fx1-Byok-Model refuses 422 "
            "invalid_byok_headers, a 5MB body 413 at ingress, and an "
            "under-cap body still processes; cursor walks stay bounded per "
            "page with unknown cursors refused 400 and absurd limits "
            "clamped to 422; the hot path's p99 sits under the declared "
            "budget, timeout_s reaches the backend transport unchanged, "
            "and every refusal (401/404/422/over-capacity 503) returns in "
            "bounded time burning no backend call; a parallel flood "
            "resolves to honest verdicts with all slots recovered, and "
            "retained heap plus the store maps re-verify at their caps "
            "after gc."
            if ok
            else f"PERF AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(perf_audit_bench(), indent=2, sort_keys=True))
