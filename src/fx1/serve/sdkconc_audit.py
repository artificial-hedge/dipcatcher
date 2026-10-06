"""sdkconc_audit — SDK-surface concurrency and robustness probes.

The two caller-facing SDK surfaces are the audit targets here: the
in-process ``Fx1Harness`` twin (thread-safety across every method family
over the shared stores — completion log, credential registry, eval,
conversation, file and upload stores, background-response registry —
and ContextVar
hygiene at the auth boundary) and the ``HarnessClient`` wire twin
(per-call retry budgets on a shared client, the shared circuit breaker,
timeout values reaching the transport verbatim, isolated SSE readers,
idempotency-key handling, and typed error envelopes). Wire-side store
contention on the app itself is the sibling ``concurrency_audit`` lane;
this lane covers the SDK half.

Durable ``state_dir`` journals are single-writer: ``Fx1Harness`` claims
each bound dir in a per-process weakref registry and refuses a second
live bind at construction, released by ``close()`` or GC. Two live
writers used to interleave the hash-chained journals — the next boot
quarantined every recovered record. That was a real corruption path,
not a passing semantic; it is now enforced, and the cross-process
interleave case (a second *process*, which the registry cannot see) is
the crash-torn class the journals already detect and quarantine —
covered by ``journal_audit``'s recovery pins.

Every probe is a measured statement — the battery exercises real
end-to-end behavior (barrier-released thread storms, scripted
transports, a real silent loopback socket) and reports booleans. The
battery is deterministic: bounded waits, no timing-dependent verdicts.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fx1.sdk import Fx1Harness
from fx1.serve.backends import EmbeddingResult, InferenceBackend
from fx1.serve.client import HarnessClient, HarnessTransportError
from fx1.serve.client_audit import (
    _closed_port,
    _exc,
    _mk,
    _remote,
    _vclock,
)
from fx1.serve.conv_audit import (
    _RESOURCES,
    _ROOT,
    _audit_context,
    _h,
    _mint_key,
    _temporary_directory,
)
from fx1.serve.conv_audit import (
    _client as _app_client,
)
from fx1.serve.openai_compat import OpenAICompatError
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

_N = 8  # storm width
_MODEL = "byok"  # the link every stub registers under
_MSGS = [{"role": "user", "content": "hi"}]
_IDEM = "Idempotency-Key"
_RA = "Retry-After"
_HEADER_KEYS = {
    "x-request-id",
    "x-fx1-api-version",
    "openai-processing-ms",
    "x-fx1-completion-id",
}


# The exact battery contract: an audit that drops or renames a probe
# fails loudly here, and ``bench`` refuses to seal an incomplete or
# partial result set as a pass.
_EXPECTED_PROBES: frozenset[str] = frozenset(
    {
        "api_key_header_sent",
        "bg_all_terminal",
        "bg_cancel_never_completed",
        "bg_cancel_typed",
        "bg_reader_no_traceback",
        "bg_started",
        "bg_submit_all_ok",
        "bg_terminal_cancel_409",
        "bg_untouched_completed",
        "byok_client_typed",
        "byok_kwargs_reach_resolver",
        "byok_payload_verbatim",
        "byok_wire_200s",
        "byok_wire_all_answered",
        "byok_wire_overcap_releases",
        "byok_wire_overcap_typed",
        "byok_wire_refusals_422",
        "cb_halfopen_recovers",
        "cb_open_fails_fast",
        "cb_storm_all_typed",
        "cl_404_mapped_per_lane",
        "cl_429_no_ra_terminal",
        "cl_500_terminal_no_retry",
        "cl_retry_attempts_per_call",
        "cl_retry_lanes_succeed",
        "cl_retry_sleeps_recorded",
        "ctx_billing_per_key",
        "ctx_denied_no_billing",
        "ctx_inflight_isolated",
        "ctx_parked_served",
        "ctx_post_storm_clean",
        "ctx_statuses_honest",
        "ctx_still_clean",
        "ctx_storm_all_answered",
        "headers_track_call",
        "idem_auto_keys_distinct",
        "idem_explicit_verbatim",
        "idem_lanes_all_ok",
        "idem_retry_same_key",
        "identity_dirs_isolated",
        "identity_post_release_writes",
        "identity_replay_convs",
        "identity_replay_keys",
        "identity_second_live_bind_refused",
        "identity_sep_logs",
        "identity_shared_registry",
        "interleave_backend_total",
        "interleave_no_exceptions",
        "interleave_provisioned_serves",
        "interleave_sdk_isolated",
        "interleave_wire_served",
        "journal_gc_releases_claim",
        "journal_post_release_mints",
        "journal_replay_after_close",
        "journal_second_writer_refused",
        "mixed_all_succeed",
        "mixed_anthropic",
        "mixed_backend_total",
        "mixed_batch_pairs",
        "mixed_complete_marker",
        "mixed_conv_ids",
        "mixed_embeddings",
        "mixed_eval_run_lands",
        "mixed_eval_runs_listed",
        "mixed_key_ids",
        "mixed_openai_chat",
        "mixed_self_usage_stable",
        "mixed_stream_chunks",
        "refuse_byok_shape",
        "refuse_byok_url",
        "refuse_missing_key",
        "refuse_missing_spec",
        "refuse_revoke_unknown",
        "refuse_storm_all_typed",
        "refuse_unknown_backend",
        "socket_refused_fast",
        "socket_refused_typed",
        "socket_silent_bounded",
        "socket_silent_typed",
        "store_all_succeed",
        "store_commands_listed",
        "store_conv_items_land",
        "store_file_roundtrip",
        "store_gate_readers_honest",
        "store_ids_all_fetchable",
        "store_upload_assembles",
        "storm_all_succeed",
        "storm_backend_calls_exact",
        "storm_close_per_call",
        "storm_content_isolated",
        "storm_ids_unique",
        "storm_log_complete",
        "storm_snapshots_coherent",
        "storm_usage_exact",
        "stream_error_frame_mapped",
        "stream_lanes_isolated",
        "stream_missing_done_refused",
        "timeout_all_ok",
        "timeout_kwarg_rides_payload",
        "timeout_per_client_isolated",
        "url_path_prefix_kept",
        "url_scheme_preserved",
        "url_trailing_slash_joined",
        "url_v1_prefix_kept",
    }
)

_STATE_DIR_ENV = "FX1_SDK_STATE_DIR"


@contextlib.contextmanager
def _sdkconc_context() -> Iterator[None]:
    """Battery scope: serialize probes and pin ``FX1_SDK_STATE_DIR``
    absent — an ambient export would bind every ``state_dir=None``
    harness to one dir and the single-writer refusal would fail the
    whole battery. The caller's value is restored verbatim."""
    with _audit_context():
        prev = os.environ.pop(_STATE_DIR_ENV, None)
        try:
            yield
        finally:
            if prev is not None:
                os.environ[_STATE_DIR_ENV] = prev


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _CountingBackend(InferenceBackend):
    """Shared resolver backend for SDK storms — one instance serving all
    lanes, with counters proving every call lands whole and nothing
    cross-contaminates. ``close`` is a counted no-op: the SDK closes
    whatever the resolver hands out after each call, so a shared instance
    must tolerate repeated closes."""

    def __init__(self) -> None:
        self.calls = 0
        self.stream_calls = 0
        self.embed_calls = 0
        self.close_calls = 0
        self._lock = threading.Lock()
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        with self._lock:
            self.calls += 1
        return f"ok:{messages[-1]['content']}"

    def stream(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> Any:
        del sampling
        with self._lock:
            self.stream_calls += 1
        yield f"chunk:{messages[-1]['content']}"
        yield "."

    def embeddings(
        self,
        input: Any,  # noqa: A002 — the wire field's own name
        *,
        model: str,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResult:
        del encoding_format, dimensions, user
        with self._lock:
            self.embed_calls += 1
        return EmbeddingResult(
            data=({"object": "embedding", "index": 0, "embedding": [0.5, 0.25]},),
            model=model,
            usage=dict(self.last_usage),
        )

    def close(self) -> None:
        with self._lock:
            self.close_calls += 1


class _ParkingBackend(_CountingBackend):
    """Counts in-flight completions and parks each one on ``gate`` —
    deterministic mid-flight windows under a storm."""

    def __init__(self) -> None:
        super().__init__()
        self.inflight = 0
        self.gate = threading.Event()

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        with self._lock:
            self.calls += 1
            self.inflight += 1
        try:
            self.gate.wait(30)
        finally:
            with self._lock:
                self.inflight -= 1
        del sampling
        return f"ok:{messages[-1]['content']}"


def _sdk(backend: Any | None = None, state_dir: Path | str | None = None) -> Fx1Harness:
    """Fx1Harness whose every link resolves to the same stub instance —
    the worst-case shared-backend arrangement (``complete_many`` does
    this for real)."""
    shared = backend if backend is not None else _CountingBackend()
    return Fx1Harness(
        backend_resolver=lambda *a, **k: shared,
        state_dir=state_dir,
    )


def _parallel(fn: Callable[[int], Any], n: int = _N) -> list[tuple[Any, BaseException | None]]:
    """Run ``fn(0..n-1)`` released together behind a barrier; returns
    ``(result, error)`` per lane so assertions never lose a lane."""
    barrier = threading.Barrier(n)
    out: list[tuple[Any, BaseException | None]] = [(None, None)] * n

    def _w(i: int) -> None:
        try:
            barrier.wait(10)
            out[i] = (fn(i), None)
        except BaseException as exc:  # noqa: BLE001 — collected, asserted
            out[i] = (None, exc)

    ths = [threading.Thread(target=_w, args=(i,), daemon=True) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(120)
    # Fail closed: a lane that did not finish inside the join budget is
    # an error, not a silent (None, None) — a wedged worker can never
    # pass as "no error, no result".
    for i, t in enumerate(ths):
        if t.is_alive():
            out[i] = (None, TimeoutError(f"lane {i} unfinished after join budget"))
    return out


def _is(val: Any, exact: float) -> bool:
    """Exact scripted-value comparison for floats — the transport
    echoes the literal we sent, so equality is the assertion."""
    return isinstance(val, int | float) and float(val) == exact  # NOSONAR(S1244)


def _wait_for(pred: Callable[[], bool], *, timeout_s: float = 15.0) -> bool:
    """Poll ``pred`` until it holds or the deadline lapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def _path_scripted(
    scripts: dict[str, list[tuple[int, dict[str, str], bytes] | BaseException]],
) -> tuple[Any, list[tuple[str, str]]]:
    """Per-path canned transport safe under concurrent callers — each
    path replays its own step list under a lock; past the end the last
    triple repeats. Steps may be ``(status, headers, body)`` triples or
    exception instances to raise."""
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()
    iters = {p: iter(list(steps)) for p, steps in scripts.items()}
    last: dict[str, tuple[int, dict[str, str], bytes]] = {}

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,  # NOSONAR(S1172) — contract signature
        headers: dict[str, str],  # NOSONAR(S1172)
        timeout_s: float,  # NOSONAR(S1172)
    ) -> tuple[int, Any, bytes]:
        path = urllib.parse.urlparse(url).path
        with lock:
            calls.append((method, path))
            it = iters.get(path)
            step = next(it, None) if it is not None else None
            if step is None:
                step = last.get(path, (500, {}, b'{"detail":"script exhausted"}'))
            elif not isinstance(step, BaseException):
                last[path] = step
        if isinstance(step, BaseException):
            raise step
        return step

    return send, calls


def _payloaded(
    responder: Callable[[str, Any], tuple[int, dict[str, str], bytes]],
) -> tuple[Any, list[tuple[str, str, Any, dict[str, str], float]]]:
    """Recording transport: captures (method, path, payload, headers,
    timeout) per call under a lock, then calls ``responder``."""
    calls: list[tuple[str, str, Any, dict[str, str], float]] = []
    lock = threading.Lock()

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Any, bytes]:
        path = urllib.parse.urlparse(url).path
        with lock:
            calls.append((method, path, payload, dict(headers), timeout_s))
        return responder(path, payload)

    return send, calls


def _sse(frames: list[str], *, done: bool = True) -> bytes:
    """An SSE wire body: ``data:``-framed payloads plus a terminal DONE."""
    lines = [f"data: {f}" for f in frames]
    if done:
        lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode()


def _sse_token(text: str) -> str:
    return json.dumps({"type": "token", "content": text})


def _http_ok(body: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    return 200, {"Content-Type": "application/json"}, json.dumps(body).encode()


# ---------------------------------------------------------------------------
# Fx1Harness completion storm — same-method N-way calls on one SDK
# ---------------------------------------------------------------------------


def _probe_sdk_complete_storm() -> dict[str, bool]:
    out: dict[str, bool] = {}
    backend = _CountingBackend()
    sdk = _sdk(backend=backend)
    lanes = _parallel(
        lambda i: sdk.complete([{"role": "user", "content": f"m{i}"}], backend=_MODEL)
    )
    oks = [v for v, e in lanes if e is None]
    out["storm_all_succeed"] = all(e is None for _, e in lanes)
    out["storm_content_isolated"] = {v.content for v in oks} == {f"ok:m{i}" for i in range(_N)}
    out["storm_ids_unique"] = len({v.completion_id for v in oks}) == _N
    out["storm_backend_calls_exact"] = backend.calls == _N
    out["storm_close_per_call"] = backend.close_calls == _N
    recs = sdk.completions(limit=_N + 8)
    out["storm_log_complete"] = {r.completion_id for r in recs} >= {v.completion_id for v in oks}
    usage = sdk.usage()
    out["storm_usage_exact"] = (
        usage.totals.requests == _N
        and usage.totals.ok == _N
        and usage.totals.errors == 0
        and usage.totals.total_tokens == _N * 5
    )
    # Readers racing a write storm always see coherent snapshots — never
    # torn, never raising.
    problems: list[str] = []
    stop = threading.Event()

    def _writer() -> None:
        for j in range(40):
            if stop.is_set():
                return
            sdk.complete([{"role": "user", "content": f"w{j}"}], backend=_MODEL)

    def _reader() -> None:
        while not stop.is_set():
            try:
                h = sdk.last_response_headers
                if h and set(h) != _HEADER_KEYS:
                    problems.append(f"torn:{sorted(h)}")
                sdk.completions(limit=8)
                sdk.usage()
            except BaseException as exc:  # noqa: BLE001 — recorded
                problems.append(f"raised:{type(exc).__name__}:{exc}")

    wt = threading.Thread(target=_writer)
    rt = threading.Thread(target=_reader)
    rt.start()
    wt.start()
    wt.join(120)
    stop.set()
    rt.join(30)
    out["storm_snapshots_coherent"] = not problems and backend.calls == _N + 40
    # Sequential calls track their own call exactly — completion id in
    # the tracing surface names the call that just finished.
    r = sdk.complete([{"role": "user", "content": "seq"}], backend=_MODEL)
    out["headers_track_call"] = (
        sdk.last_response_headers.get("x-fx1-completion-id") == r.completion_id
    )
    return out


# ---------------------------------------------------------------------------
# Fx1Harness mixed-method storm — every method family at once
# ---------------------------------------------------------------------------


def _probe_sdk_mixed_storm() -> dict[str, bool]:
    out: dict[str, bool] = {}
    backend = _CountingBackend()
    sdk = _sdk(backend=backend)
    spec = sdk.eval_spec_create("mixed", suite="tooluse")
    spec_id = str(spec["id"])

    def _lane(i: int) -> Any:
        kind = i % 8
        if kind == 0:
            return sdk.complete([{"role": "user", "content": f"c{i}"}], backend=_MODEL).content
        if kind == 1:
            batch = sdk.complete_many(
                [
                    [{"role": "user", "content": f"b{i}a"}],
                    [{"role": "user", "content": f"b{i}b"}],
                ],
                backend=_MODEL,
                max_workers=2,
            )
            return {r.content for r in batch}
        if kind == 2:
            return tuple(
                sdk.stream_complete([{"role": "user", "content": f"s{i}"}], backend=_MODEL)
            )
        if kind == 3:
            env, _ = sdk.openai_chat(
                {"model": _MODEL, "messages": [{"role": "user", "content": f"o{i}"}]}
            )
            return env.choices[0].message["content"]
        if kind == 4:
            anth, _ = sdk.anthropic_message(
                {
                    "model": _MODEL,
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": f"a{i}"}],
                }
            )
            return anth.content[0]["text"]
        if kind == 5:
            emb, _ = sdk.openai_embeddings({"model": _MODEL, "input": f"e{i}"})
            return emb["data"][0]["embedding"]
        if kind == 6:
            conv = sdk.openai_conversation_create(metadata={"lane": str(i)})
            return conv["id"]
        return sdk.key_create(f"mix-{i}")["id"]

    lanes = _parallel(_lane, 2 * _N)
    out["mixed_all_succeed"] = all(e is None for _, e in lanes)
    values = [v for v, _ in lanes]
    out["mixed_complete_marker"] = "ok:c0" in values and "ok:c8" in values
    out["mixed_batch_pairs"] = {"ok:b1a", "ok:b1b"} in values and {
        "ok:b9a",
        "ok:b9b",
    } in values
    out["mixed_stream_chunks"] = ("chunk:s2", ".") in values
    out["mixed_openai_chat"] = "ok:o3" in values and "ok:o11" in values
    out["mixed_anthropic"] = "ok:a4" in values
    out["mixed_embeddings"] = [0.5, 0.25] in values
    out["mixed_conv_ids"] = (
        sum(1 for v in values if isinstance(v, str) and v.startswith("conv_")) == 2
    )
    out["mixed_key_ids"] = len(sdk.keys()) == 2
    run = sdk.eval_run_create(spec_id, model=_MODEL, model_fn=lambda msgs: "answer")
    out["mixed_eval_run_lands"] = bool(run.get("id"))
    out["mixed_eval_runs_listed"] = any(
        r.get("id") == run.get("id") for r in sdk.eval_runs(spec_id)
    )
    out["mixed_backend_total"] = (
        backend.calls == 2 + 4 + 2 + 2  # complete + batch items + chat + anthropic
        and backend.stream_calls == 2
        and backend.embed_calls == 2
    )
    out["mixed_self_usage_stable"] = (
        sdk.self_usage().get("credential") == "env" and sdk.self_usage().get("metered") is False
    )
    return out


# ---------------------------------------------------------------------------
# Fx1Harness store storm — conversations/files/uploads/gate readers racing
# ---------------------------------------------------------------------------


def _probe_sdk_store_storm() -> dict[str, bool]:
    out: dict[str, bool] = {}
    sdk = _sdk()

    def _lane(i: int) -> Any:
        kind = i % 5
        if kind == 0:
            conv = sdk.openai_conversation_create()
            added = sdk.openai_conversation_items_add(
                conv["id"],
                [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"item{i}"}],
                    }
                ],
            )
            return (conv["id"], [it["id"] for it in added["data"]])
        if kind == 1:
            fobj = sdk.openai_file_create(
                f'{{"i":{i}}}\n'.encode(), purpose="batch", filename=f"f{i}.jsonl"
            )
            return (fobj["id"], sdk.file_content(fobj["id"]))
        if kind == 2:
            blob = f"part-body-{i}".encode()
            up = sdk.upload_create(purpose="batch", filename=f"u{i}.jsonl", bytes=len(blob) * 2)
            p1 = sdk.upload_part(up["id"], blob)
            p2 = sdk.upload_part(up["id"], blob)
            fin = sdk.upload_complete(up["id"], [p1["id"], p2["id"]])
            return (fin["file"]["id"], sdk.file_content(fin["file"]["id"]))
        if kind == 3:
            return (
                sdk.check_text(f"text{i}").ok,
                sdk.score(f"text{i}")[0]["total"],
                sdk.moderate(f"text{i}")["results"][0]["flagged"],
            )
        return sdk.commands()

    lanes = _parallel(_lane, 5 * _N)
    values = [v for v, _ in lanes]
    out["store_all_succeed"] = all(e is None for _, e in lanes)
    out["store_conv_items_land"] = all(
        isinstance(v, tuple) and len(v) == 2 and str(v[0]).startswith("conv_")
        for i, v in enumerate(values)
        if i % 5 == 0
    )
    out["store_file_roundtrip"] = all(
        isinstance(v, tuple) and str(v[0]).startswith("file-") and v[1] == f'{{"i":{i}}}\n'.encode()
        for i, v in enumerate(values)
        if i % 5 == 1
    )
    out["store_upload_assembles"] = all(
        isinstance(v, tuple) and v[1] == f"part-body-{i}".encode() * 2
        for i, v in enumerate(values)
        if i % 5 == 2
    )
    out["store_gate_readers_honest"] = all(
        v == (True, 4.0, False) for i, v in enumerate(values) if i % 5 == 3
    )
    out["store_commands_listed"] = all(
        isinstance(v, list) and "doctor" in v for i, v in enumerate(values) if i % 5 == 4
    )
    # Store objects stay consistent under the storm: every minted id is
    # fetchable afterwards — the store is the shared surface.
    conv_ids = [v[0] for i, v in enumerate(values) if i % 5 == 0]
    file_ids = [v[0] for i, v in enumerate(values) if i % 5 in (1, 2)]

    def _conv_ok(conv_id: str) -> bool:
        return _exc(lambda: sdk.openai_conversation_get(conv_id)) is None

    def _file_ok(file_id: str) -> bool:
        return _exc(lambda: sdk.file_content(file_id)) is None

    out["store_ids_all_fetchable"] = all(_conv_ok(str(c)) for c in conv_ids) and all(
        _file_ok(str(f)) for f in file_ids
    )
    return out


# ---------------------------------------------------------------------------
# Fx1Harness background-response storm — submit/cancel/read racing
# ---------------------------------------------------------------------------


def _probe_sdk_background_storm() -> dict[str, bool]:
    out: dict[str, bool] = {}
    backend = _ParkingBackend()
    sdk = _sdk(backend=backend)

    def _submit(i: int) -> str:
        env, _ = sdk.openai_response(
            {"model": _MODEL, "input": f"bg{i}", "background": True, "store": True}
        )
        return str(env["id"])

    lanes = _parallel(_submit)
    out["bg_submit_all_ok"] = all(e is None for _, e in lanes)
    ids = [v for v, e in lanes if e is None]
    # Every worker parked inside the backend before cancels land.
    started = _wait_for(lambda: backend.inflight >= _N)
    cancel_lanes = _parallel(lambda i: sdk.openai_response_cancel(ids[i]), _N // 2)
    cancelled = set(ids[: _N // 2])
    out["bg_cancel_typed"] = all(
        e is None and v.get("status") == "cancelled" for v, e in cancel_lanes
    )
    backend.gate.set()
    seen: list[str] = []
    done = threading.Event()

    def _reader() -> None:
        while not done.is_set():
            for rid in ids:
                try:
                    seen.append(str(sdk.openai_response_get(rid).get("status")))
                except (KeyError, OpenAICompatError) as exc:
                    seen.append(f"typed:{type(exc).__name__}")

    rt = threading.Thread(target=_reader)
    rt.start()
    final_ok = _wait_for(
        lambda: all(
            sdk.openai_response_get(rid).get("status") in {"completed", "cancelled"} for rid in ids
        )
    )
    done.set()
    rt.join(30)
    statuses = {rid: str(sdk.openai_response_get(rid)["status"]) for rid in ids}
    out["bg_started"] = started
    out["bg_all_terminal"] = final_ok and set(statuses.values()) <= {
        "completed",
        "cancelled",
    }
    out["bg_cancel_never_completed"] = all(statuses[rid] == "cancelled" for rid in cancelled)
    out["bg_untouched_completed"] = all(
        statuses[rid] == "completed" for rid in set(ids) - cancelled
    )
    out["bg_reader_no_traceback"] = all(
        s in {"queued", "in_progress", "completed", "cancelled"} or s.startswith("typed:")
        for s in seen
    )
    done_ids = [rid for rid in ids if statuses[rid] == "completed"]
    late = _exc(lambda: sdk.openai_response_cancel(done_ids[0])) if done_ids else None
    out["bg_terminal_cancel_409"] = isinstance(late, OpenAICompatError) and late.status == 409
    return out


# ---------------------------------------------------------------------------
# Fx1Harness identity — separate instances, shared state_dir semantics
# ---------------------------------------------------------------------------


def _probe_sdk_identity() -> dict[str, bool]:
    out: dict[str, bool] = {}
    iso = _temporary_directory()
    sa = _sdk(state_dir=iso / "a")
    sb = _sdk(state_dir=iso / "b")
    ka = sa.key_create("a")
    out["identity_dirs_isolated"] = isinstance(_exc(lambda: sb.key_get(ka["id"])), KeyError)
    sa.complete([{"role": "user", "content": "x"}], backend=_MODEL)
    out["identity_sep_logs"] = sb.completions(limit=8) == []
    # Boot replay: a fresh instance on a populated dir sees the journals.
    shared = iso / "shared"
    s1 = _sdk(state_dir=shared)
    k1 = s1.key_create("k1")
    s1_conv = s1.openai_conversation_create(metadata={"owner": "s1"})
    # Single-writer: a second live bind on the same dir is refused at
    # construction — journals never see interleaved live writers.
    refused = _exc(lambda: _sdk(state_dir=shared))
    out["identity_second_live_bind_refused"] = isinstance(
        refused, RuntimeError
    ) and "single-writer" in str(refused)
    s1.close()  # release — models process restart
    s2 = _sdk(state_dir=shared)
    out["identity_replay_keys"] = s2.key_get(k1["id"])["name"] == "k1"
    out["identity_replay_convs"] = (
        s2.openai_conversation_get(s1_conv["id"]).get("id") == s1_conv["id"]
    )
    # The released claim is the only shared-dir writer — post-release
    # writes bind cleanly and read back in the sole live view.
    k2 = s2.key_create("k2")
    out["identity_post_release_writes"] = s2.key_get(k2["id"])["name"] == "k2"
    s2.close()
    # Shared command registry: two SDKs over one Harness object answer
    # the same command names — ``harness=`` is the identity contract.
    from fx1.harness import Harness  # noqa: PLC0415

    harness = Harness()
    sh1 = Fx1Harness(backend_resolver=lambda *a, **k: _CountingBackend(), harness=harness)
    sh2 = Fx1Harness(backend_resolver=lambda *a, **k: _CountingBackend(), harness=harness)
    out["identity_shared_registry"] = sorted(sh1.commands()) == sorted(sh2.commands())
    return out


# ---------------------------------------------------------------------------
# Fx1Harness journal — durable state is single-writer, enforced
# ---------------------------------------------------------------------------


def _probe_sdk_journal_contention() -> dict[str, bool]:
    """Writer discipline on ``state_dir`` journals.

    Two live harnesses on one ``state_dir`` interleave the hash-chained
    journals; the next boot then quarantines every recovered record —
    measured data loss that this probe used to record as *passing*
    booleans. The second live bind is now REFUSED at construction (a
    per-process weakref claim in ``Fx1Harness``); the only shared-dir
    pattern left is sequential write -> close -> replay.
    """
    out: dict[str, bool] = {}
    iso = _temporary_directory()
    shared = iso / "jrnl"
    s1 = _sdk(state_dir=shared)
    pre1 = s1.key_create("pre-1")
    # The second live writer is refused at construction — a typed
    # RuntimeError, never a corrupt journal.
    refused = _exc(lambda: _sdk(state_dir=shared))
    out["journal_second_writer_refused"] = isinstance(
        refused, RuntimeError
    ) and "single-writer" in str(refused)
    # ``close()`` models process shutdown: a fresh boot replays cleanly
    # and keeps minting.
    s1.close()
    s3 = _sdk(state_dir=shared)
    out["journal_replay_after_close"] = s3.key_get(pre1["id"])["name"] == "pre-1"
    out["journal_post_release_mints"] = _exc(lambda: s3.key_create("post-release")) is None
    s3.close()
    # A crashed/GC'd writer releases its claim — the registry is a
    # weakref map, not a lock file, so nothing wedges the dir.
    gc_dir = iso / "gc"
    ep = _sdk(state_dir=gc_dir)
    ep_id = ep.key_create("ep")["id"]
    del ep
    gc.collect()
    s4 = _sdk(state_dir=gc_dir)
    out["journal_gc_releases_claim"] = s4.key_get(ep_id)["name"] == "ep"
    s4.close()
    return out


# ---------------------------------------------------------------------------
# ContextVar hygiene — _REQUEST_KEY_ID cannot leak across calls/threads
# ---------------------------------------------------------------------------


def _probe_contextvar_isolation() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.api import _REQUEST_KEY_ID  # noqa: PLC0415

    backend = _CountingBackend()
    client, _api = _app_client({_MODEL: lambda: backend}, api_key=_ROOT)
    key_a, id_a = _mint_key(client, scopes=["read", "write"])
    key_b, id_b = _mint_key(client, scopes=["read"])
    body = {"backend": _MODEL, "messages": _MSGS}

    def _lane(i: int) -> int:
        if i % 2 == 0:  # 16 lanes: A writes, 200s
            return client.post("/harness/complete", json=body, headers=_h(key_a)).status_code
        if i % 4 == 1:  # 8 lanes: B writes — scope-refused
            return client.post("/harness/complete", json=body, headers=_h(key_b)).status_code
        # 8 lanes: B reads — authorized
        return client.get("/v1/models", headers=_h(key_b)).status_code

    lanes = _parallel(_lane, 4 * _N)
    statuses = [v for v, _ in lanes if isinstance(v, int)]
    out["ctx_storm_all_answered"] = all(e is None for _, e in lanes)
    a_writes = sum(1 for i in range(4 * _N) if i % 2 == 0)
    b_denied = sum(1 for i in range(4 * _N) if i % 4 == 1)
    b_reads = sum(1 for i in range(4 * _N) if i % 4 == 3)
    out["ctx_statuses_honest"] = (
        statuses.count(200) == a_writes + b_reads and statuses.count(403) == b_denied
    )
    a_card = client.get(f"/harness/keys/{id_a}/usage", headers=_h(_ROOT)).json()
    b_card = client.get(f"/harness/keys/{id_b}/usage", headers=_h(_ROOT)).json()
    out["ctx_billing_per_key"] = (
        a_card["uses"] == a_writes
        and a_card["tokens_used"] == a_writes * 5
        and a_card["served"]["calls"] == a_writes
    )
    out["ctx_denied_no_billing"] = (
        b_card["uses"] == b_reads and b_card["tokens_used"] == 0 and b_card["served"]["calls"] == 0
    )
    out["ctx_post_storm_clean"] = _REQUEST_KEY_ID.get() is None
    # Mid-flight: while a request sits inside the backend, the driver
    # thread's contextvar is still unset — thread isolation is real.
    gate = _ParkingBackend()
    client2, _api2 = _app_client({_MODEL: lambda: gate}, api_key=_ROOT)
    resp: list[Any] = []

    def _parked() -> None:
        resp.append(client2.post("/harness/complete", json=body, headers=_h(_ROOT)))

    pt = threading.Thread(target=_parked)
    pt.start()
    in_flight = _wait_for(lambda: gate.inflight >= 1)
    out["ctx_inflight_isolated"] = in_flight and _REQUEST_KEY_ID.get() is None
    gate.gate.set()
    pt.join(30)
    out["ctx_parked_served"] = bool(resp) and resp[0].status_code == 200
    out["ctx_still_clean"] = _REQUEST_KEY_ID.get() is None
    return out


# ---------------------------------------------------------------------------
# HarnessClient — shared-client concurrency: retries, breaker, idem
# ---------------------------------------------------------------------------


def _probe_client_retry_concurrent() -> dict[str, bool]:
    """Shared client under a storm: retries keyed by a per-lane marker in
    the run payload, so every lane's attempt count is deterministic
    however the lanes interleave on the wire."""
    out: dict[str, bool] = {}
    clk, slp, sleeps = _vclock()
    seen: dict[str, int] = {}
    paths: list[str] = []
    lock = threading.Lock()

    def _ok_for(cmd: str) -> bytes:
        return json.dumps({"command": cmd, "exit_code": 0, "stdout": "", "stderr": ""}).encode()

    def send(
        method: str,
        url: str,
        payload: Any,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Any, bytes]:
        del method, headers, timeout_s
        path = urllib.parse.urlparse(url).path
        cmd = str(payload.get("command", "")) if isinstance(payload, dict) else ""
        with lock:
            paths.append(path)
            n = seen.get(cmd, 0)
            seen[cmd] = n + 1
        if cmd.startswith("ra-") or path == "/v1/models":
            if n == 0:
                return 429, {_RA: "0"}, b'{"detail":"slow"}'
            return 200, {}, _ok_for(cmd or "list")
        if cmd.startswith("s5-"):
            if n == 0:
                return 503, {_RA: "0"}, b'{"detail":"unavail"}'
            return 200, {}, _ok_for(cmd)
        if cmd.startswith("t5-"):
            return 500, {}, b'{"detail":"boom"}'
        if cmd.startswith("n4-"):
            return 429, {}, b'{"detail":"no hint"}'
        if path == "/v1/models/x":
            return 404, {}, b'{"detail":"missing"}'
        return 200, {}, _ok_for(cmd)

    client = _mk(send, clock=clk, sleep=slp, max_retries=4)
    tags = {k: [f"{k}{i}" for i in range(_N)] for k in ("ra-", "s5-", "t5-", "n4-")}

    def _lane(i: int) -> Any:
        kind = i % 4
        tag = tags[["ra-", "s5-", "t5-", "n4-"][kind]][i // 4]
        if kind < 2:
            return client.run(tag, [])
        return ("exc", type(_exc(lambda: client.run(tag, []))).__name__)

    lanes = _parallel(_lane, 4 * _N)
    # One GET lane each for the retryable and mapped-error legs (single
    # lanes keep the per-path script deterministic).
    models_res = _exc(client.list_models)
    missing_res = _exc(lambda: client.retrieve_model("x"))
    ok_lanes = lanes[0::4] + lanes[1::4]
    out["cl_retry_lanes_succeed"] = (
        all(e is None for _, e in ok_lanes)
        and all(
            v.command == t for (v, _), t in zip(ok_lanes, tags["ra-"] + tags["s5-"], strict=True)
        )
        and models_res is None
    )
    out["cl_retry_attempts_per_call"] = (
        all(seen[t] == 2 for t in tags["ra-"] + tags["s5-"])
        and all(seen[t] == 1 for t in tags["t5-"] + tags["n4-"])
        and paths.count("/v1/models") == 2
        and paths.count("/v1/models/x") == 1
    )
    out["cl_retry_sleeps_recorded"] = len(sleeps) == 2 * _N + 1
    out["cl_500_terminal_no_retry"] = all(
        v == ("exc", "HarnessTransportError") for v, _ in lanes[2::4]
    )
    out["cl_429_no_ra_terminal"] = all(
        v == ("exc", "HarnessTransportError") for v, _ in lanes[3::4]
    )
    out["cl_404_mapped_per_lane"] = isinstance(missing_res, KeyError)
    return out


def _probe_client_circuit_concurrent() -> dict[str, bool]:
    out: dict[str, bool] = {}
    clk, tick, _sleeps = _vclock()
    fault = HarnessTransportError("conn refused")
    send, calls = _path_scripted({"/v1/models": [fault]})
    client = _mk(
        send,
        clock=clk,
        sleep=lambda s: None,
        max_retries=2,
        circuit_breaker_threshold=2,
        circuit_reset_s=30.0,
    )
    # _exc captures into the result slot: v is the raised exception.
    lanes = _parallel(lambda i: _exc(client.list_models), _N)
    out["cb_storm_all_typed"] = all(isinstance(v, HarnessTransportError) for v, _ in lanes)
    # Breaker open: concurrent follow-ups fail fast, transport untouched —
    # on ANY call, including ones that would map to a different error.
    before = len(calls)
    follow = _parallel(lambda i: _exc(client.list_models), _N)
    mapped = _exc(lambda: client.retrieve_model("x"))
    out["cb_open_fails_fast"] = (
        all(isinstance(v, HarnessTransportError) for v, _ in follow)
        and isinstance(mapped, HarnessTransportError)
        and len(calls) == before
    )
    # Advance past reset: the half-open probe succeeds and closes it.
    tick(31.0)
    ok_send, _ok_calls = _path_scripted({"/v1/models": [_http_ok({"object": "list", "data": []})]})
    client._transport = ok_send  # noqa: SLF001 — swap under test to answer 200
    probe = _exc(client.list_models)
    out["cb_halfopen_recovers"] = (
        probe is None and client._cb_failures == 0  # noqa: SLF001
    )
    return out


def _probe_client_idem_concurrent() -> dict[str, bool]:
    out: dict[str, bool] = {}

    def responder(path: str, payload: Any) -> tuple[int, dict[str, str], bytes]:
        del path, payload
        return 200, {}, b'{"command":"c","exit_code":0,"stdout":"","stderr":""}'

    send, calls = _payloaded(responder)
    client = _mk(send, max_retries=2)
    lanes = _parallel(lambda i: client.run(f"cmd-{i}", []), _N)
    out["idem_lanes_all_ok"] = all(e is None for _, e in lanes)
    keys = [h.get(_IDEM) for _, p, _, h, _ in calls if p == "/harness/runs"]
    out["idem_auto_keys_distinct"] = len(set(keys)) == _N and all(keys)
    # Explicit key rides through verbatim even under contention.
    send2, calls2 = _payloaded(responder)
    client2 = _mk(send2, max_retries=2)
    fixed = _parallel(lambda i: client2.run(f"k{i}", [], idempotency_key=f"fixed-{i % 2}"), _N)
    seen_keys = {h.get(_IDEM) for _, p, _, h, _ in calls2 if p == "/harness/runs"}
    out["idem_explicit_verbatim"] = all(e is None for _, e in fixed) and seen_keys == {
        "fixed-0",
        "fixed-1",
    }
    # A retried write replays under the SAME key: the first attempt per
    # key earns a retryable 429, its retry answers 200 — so every key
    # appears exactly twice on the wire.
    first_seen: set[str] = set()
    seen_attempt_keys: list[str] = []
    lock3 = threading.Lock()

    def send3(
        method: str,
        url: str,
        payload: Any,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Any, bytes]:
        del method, url, payload, timeout_s
        key = str(headers.get(_IDEM))
        with lock3:
            seen_attempt_keys.append(key)
            fresh = key not in first_seen
            first_seen.add(key)
        if fresh:
            return 429, {_RA: "0"}, b'{"detail":"slow"}'
        return 200, {}, b'{"command":"c","exit_code":0,"stdout":"","stderr":""}'

    client3 = _mk(send3, max_retries=2, retry_writes=True)
    retry_lanes = _parallel(lambda i: client3.run(f"r{i}", [], idempotency_key=f"rep-{i}"), 3)
    out["idem_retry_same_key"] = all(e is None for _, e in retry_lanes) and sorted(
        seen_attempt_keys
    ) == ["rep-0", "rep-0", "rep-1", "rep-1", "rep-2", "rep-2"]
    return out


def _probe_client_timeout_concurrent() -> dict[str, bool]:
    out: dict[str, bool] = {}
    seen: list[tuple[str, float]] = []
    lock = threading.Lock()

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Any, bytes]:
        del method, payload, headers
        with lock:
            seen.append((urllib.parse.urlparse(url).path, timeout_s))
        return _http_ok({"object": "list", "data": []})

    c_fast = _mk(send, timeout_s=0.25)
    c_slow = _mk(send, timeout_s=9.0)
    lanes = _parallel(lambda i: (c_fast if i % 2 == 0 else c_slow).list_models(), _N)
    out["timeout_all_ok"] = all(e is None for _, e in lanes)
    out["timeout_per_client_isolated"] = {t for _, t in seen} == {0.25, 9.0}
    # ``complete(timeout_s=...)`` is a REQUEST-side budget: it rides the
    # payload verbatim while the transport still sees the client's own
    # timeout — two different knobs, pinned apart.
    seen_t: list[float] = []
    bodies: list[Any] = []

    def send2(
        method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float
    ) -> tuple[int, Any, bytes]:
        del method, url, headers
        seen_t.append(timeout_s)
        bodies.append(payload)
        return (
            200,
            {},
            b'{"backend":"b","model":"m","content":"ok","receipt_hashes":[],"completion_id":"c1"}',
        )

    c3 = _mk(send2, timeout_s=5.0)
    c3.complete(_MSGS, backend="byok", timeout_s=1.5)
    out["timeout_kwarg_rides_payload"] = (
        len(seen_t) == 1
        and _is(seen_t[0], 5.0)
        and isinstance(bodies[0], dict)
        and _is(bodies[0].get("timeout_s"), 1.5)
    )
    return out


def _probe_client_timeout_socket() -> dict[str, bool]:
    """Real-socket pins: connect refusal fails fast; a silent-accept
    server enforces the read timeout inside the budget."""
    import socket  # noqa: PLC0415

    out: dict[str, bool] = {}
    refused_port = _closed_port()
    t0 = time.monotonic()
    exc = _exc(
        lambda: HarnessClient(f"http://127.0.0.1:{refused_port}", timeout_s=2.0).list_models()
    )
    refused_s = time.monotonic() - t0
    out["socket_refused_typed"] = isinstance(exc, HarnessTransportError)
    out["socket_refused_fast"] = refused_s < 10.0
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def _accept() -> None:
        try:
            conn, _ = srv.accept()
            time.sleep(5)
            conn.close()
        except OSError:
            pass

    threading.Thread(target=_accept, daemon=True).start()
    t1 = time.monotonic()
    exc2 = _exc(lambda: HarnessClient(f"http://127.0.0.1:{port}", timeout_s=1.0).list_models())
    hung_s = time.monotonic() - t1
    srv.close()
    out["socket_silent_typed"] = isinstance(exc2, HarnessTransportError)
    out["socket_silent_bounded"] = 0.2 < hung_s < 12.0
    return out


def _probe_client_stream_concurrent() -> dict[str, bool]:
    out: dict[str, bool] = {}
    err_frame = _sse(
        [
            _sse_token("x"),
            json.dumps(
                {
                    "type": "error",
                    "status": 503,
                    "detail": "no backend",
                    "code": "backend_unavailable",
                }
            ),
        ],
        done=False,
    )
    nodone = _sse([_sse_token("y")], done=False)

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Any, bytes]:
        del method, url, headers, timeout_s
        marker = ""
        if isinstance(payload, dict):
            msgs = payload.get("messages") or []
            marker = str(msgs[-1].get("content", "")) if msgs else ""
        if marker == "err":
            return 200, {}, err_frame
        if marker == "nodone":
            return 200, {}, nodone
        return 200, {}, _sse([_sse_token(f"tok:{marker}"), _sse_token("fin")])

    client = _mk(send)

    def _lane(i: int) -> Any:
        marker = {0: "err", 1: "nodone"}.get(i % 3, f"m{i}")
        try:
            return client.stream_complete([{"role": "user", "content": marker}], backend="byok")
        except BaseException as exc:  # noqa: BLE001 — collected
            return exc

    lanes = _parallel(_lane, 3 * _N)
    out["stream_lanes_isolated"] = all(
        v == [f"tok:m{i}", "fin"] for i, (v, e) in enumerate(lanes) if i % 3 == 2
    )
    out["stream_error_frame_mapped"] = all(
        type(v).__name__ == "BackendNotConfiguredError"
        for i, (v, e) in enumerate(lanes)
        if i % 3 == 0
    )
    out["stream_missing_done_refused"] = all(
        isinstance(v, HarnessTransportError) for i, (v, e) in enumerate(lanes) if i % 3 == 1
    )
    return out


def _probe_client_url_headers() -> dict[str, bool]:
    out: dict[str, bool] = {}
    seen: list[str] = []
    hdrs: list[dict[str, str]] = []

    def send(
        method: str, url: str, payload: Any, headers: dict[str, str], timeout_s: float
    ) -> tuple[int, Any, bytes]:
        del method, payload, timeout_s
        seen.append(url)
        hdrs.append(dict(headers))
        return _http_ok({"object": "list", "data": []})

    HarnessClient("http://h.test/", transport=send).list_models()
    HarnessClient("http://h.test/v1", transport=send).list_models()
    HarnessClient("http://h.test/base/path/", transport=send).list_models()
    HarnessClient("https://h.test", transport=send).list_models()
    out["url_trailing_slash_joined"] = seen[0] == "http://h.test/v1/models"
    out["url_v1_prefix_kept"] = seen[1] == "http://h.test/v1/v1/models"
    out["url_path_prefix_kept"] = seen[2] == "http://h.test/base/path/v1/models"
    out["url_scheme_preserved"] = seen[3].startswith("https://")
    HarnessClient("http://h.test", api_key="sdkk", transport=send).list_models()
    out["api_key_header_sent"] = hdrs[-1].get("X-API-Key") == "sdkk"
    # byok override lands in the payload verbatim.
    payloads: list[Any] = []

    def send2(m: str, u: str, p: Any, h: dict[str, str], t: float) -> tuple[int, Any, bytes]:
        del m, u, h, t
        payloads.append(p)
        return (
            200,
            {},
            b'{"backend":"b","model":"m","content":"ok","receipt_hashes":[],"completion_id":"c1"}',
        )

    _mk(send2).complete(
        _MSGS,
        backend="byok",
        byok={"base_url": "http://up.test", "api_key": "k", "model": "m"},
    )
    body = payloads[-1]
    out["byok_payload_verbatim"] = isinstance(body, dict) and body.get("byok") == {
        "base_url": "http://up.test",
        "api_key": "k",
        "model": "m",
    }
    return out


def _probe_sdk_refusal_storm() -> dict[str, bool]:
    """Every refusal surface answers its typed exception under a storm —
    never a bare traceback."""
    out: dict[str, bool] = {}
    # The real resolver (no stub): unknown names refuse through the
    # registry — what a caller with no configured backends hits.
    sdk = Fx1Harness(state_dir=_temporary_directory())

    def _lane(i: int) -> str:
        kind = i % 6
        if kind == 0:
            return type(_exc(lambda: sdk.complete(_MSGS, backend="bogus"))).__name__
        if kind == 1:
            return type(_exc(lambda: sdk.key_get("none"))).__name__
        if kind == 2:
            return type(_exc(lambda: sdk.eval_run_create("eval_none", model=_MODEL))).__name__
        if kind == 3:
            return type(_exc(lambda: sdk.key_revoke("none"))).__name__
        if kind == 4:
            return type(
                _exc(
                    lambda: sdk.complete(
                        _MSGS, backend="byok", byok={"base_url": "x", "api_key": "k"}
                    )
                )
            ).__name__
        return type(
            _exc(
                lambda: sdk.complete(
                    _MSGS,
                    backend="byok",
                    byok={
                        "base_url": "http://u:p@x.test",
                        "api_key": "k",
                        "model": "m",
                    },
                )
            )
        ).__name__

    lanes = _parallel(_lane, 2 * _N)
    kinds = [v for v, e in lanes if e is None]
    out["refuse_storm_all_typed"] = all(e is None for _, e in lanes)
    out["refuse_unknown_backend"] = set(kinds[0::6]) == {"KeyError"}
    out["refuse_missing_key"] = set(kinds[1::6]) == {"KeyError"}
    out["refuse_missing_spec"] = set(kinds[2::6]) == {"KeyError"}
    out["refuse_revoke_unknown"] = set(kinds[3::6]) == {"KeyError"}
    out["refuse_byok_shape"] = set(kinds[4::6]) == {"ValueError"}
    out["refuse_byok_url"] = set(kinds[5::6]) == {"ValueError"}
    return out


def _probe_sdk_wire_interleave() -> dict[str, bool]:
    """SDK calls and HTTP calls against the same deployment interleave
    honestly: an app booted on a state_dir sees keys the SDK minted, and
    concurrent loads don't corrupt either side."""
    out: dict[str, bool] = {}
    iso = _temporary_directory()
    state = iso / "state"
    backend = _CountingBackend()
    sdk = _sdk(backend=backend, state_dir=state)
    provisioned = sdk.key_create("wire-key")
    client, _api = _app_client({_MODEL: lambda: backend}, api_key=_ROOT, state_dir=state)
    r = client.post(
        "/harness/complete",
        json={"backend": _MODEL, "messages": _MSGS},
        headers=_h(provisioned["key"]),
    )
    out["interleave_provisioned_serves"] = r.status_code == 200
    body = {"backend": _MODEL, "messages": _MSGS}

    def _lane(i: int) -> Any:
        if i % 2:
            return sdk.complete([{"role": "user", "content": f"s{i}"}], backend=_MODEL).content
        return client.post("/harness/complete", json=body, headers=_h(_ROOT)).status_code

    lanes = _parallel(_lane, 2 * _N)
    out["interleave_no_exceptions"] = all(e is None for _, e in lanes)
    out["interleave_sdk_isolated"] = all(
        isinstance(v, str) and v.startswith("ok:") for i, (v, e) in enumerate(lanes) if i % 2 == 1
    )
    out["interleave_wire_served"] = all(v == 200 for i, (v, e) in enumerate(lanes) if i % 2 == 0)
    out["interleave_backend_total"] = backend.calls == 1 + 2 * _N
    return out


def _probe_sdk_byok_wire() -> dict[str, bool]:
    """BYOK override end-to-end on the wire: the request body's ``byok``
    reaches backend resolution verbatim; malformed overrides refuse
    honestly under concurrent load."""
    out: dict[str, bool] = {}
    seen: list[tuple[str, dict[str, Any]]] = []
    lock = threading.Lock()

    def resolver(name: str, **k: Any) -> InferenceBackend:
        with lock:
            seen.append((name, dict(k)))
        return _CountingBackend()

    from fastapi.testclient import TestClient  # noqa: PLC0415

    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    isolated = _temporary_directory()
    (isolated / "rc").mkdir()
    app = api_mod.create_app(
        harness=Harness(runner=lambda argv, timeout_s: (0, "ok", "")),
        backend_resolver=resolver,
        state_dir=isolated / "st",
        receipts_dir=isolated / "rc",
        ft_dir=isolated / "ft",
        byok_override=True,
    )
    _RESOURCES.get().callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
    client = TestClient(app, raise_server_exceptions=False)
    _RESOURCES.get().callback(client.close)
    good = {
        "backend": "byok",
        "messages": _MSGS,
        "byok": {"base_url": "http://up.test", "api_key": "k", "model": "m"},
    }
    bad_missing = {
        "backend": "byok",
        "messages": _MSGS,
        "byok": {"base_url": "http://up.test"},
    }
    bad_extra = {
        "backend": "byok",
        "messages": _MSGS,
        "byok": {
            "base_url": "http://up.test",
            "api_key": "k",
            "model": "m",
            "key": "z",
        },
    }

    def _lane(i: int) -> int:
        if i % 3 == 0:
            return client.post("/harness/complete", json=good).status_code
        if i % 3 == 1:
            return client.post("/harness/complete", json=bad_missing).status_code
        return client.post("/harness/complete", json=bad_extra).status_code

    # Under the inflight cap (16 by default) the verdicts are strict.
    lanes = _parallel(_lane, 12)
    codes = [v for v, _ in lanes if isinstance(v, int)]
    out["byok_wire_all_answered"] = all(e is None for _, e in lanes)
    out["byok_wire_200s"] = codes.count(200) == 4
    out["byok_wire_refusals_422"] = codes.count(422) == 8
    byok_kwargs = [k for n, k in seen if n == "byok"]
    out["byok_kwargs_reach_resolver"] = len(byok_kwargs) == 4 and all(
        k.get("base_url") == "http://up.test" and k.get("api_key") == "k" and k.get("model") == "m"
        for k in byok_kwargs
    )
    # Over the cap the inflight gate claims the slot BEFORE body
    # validation: hold every slot with parked good calls, then a
    # malformed burst MUST surface the capacity refusal — it can never
    # reach validation while the semaphore is full.
    park = _ParkingBackend()
    iso2 = _temporary_directory()
    (iso2 / "rc").mkdir()
    app2 = api_mod.create_app(
        harness=Harness(runner=lambda argv, timeout_s: (0, "ok", "")),
        backend_resolver=lambda name, **k: park,
        state_dir=iso2 / "st",
        receipts_dir=iso2 / "rc",
        ft_dir=iso2 / "ft",
        byok_override=True,
    )
    _RESOURCES.get().callback(app2.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
    client2 = TestClient(app2, raise_server_exceptions=False)
    _RESOURCES.get().callback(client2.close)
    holders: list[Any] = []

    def _hold() -> None:
        holders.append(client2.post("/harness/complete", json=good).status_code)

    hold_threads = [threading.Thread(target=_hold, daemon=True) for _ in range(16)]
    for t in hold_threads:
        t.start()
    parked = _wait_for(lambda: park.inflight >= 16)
    mal = _parallel(lambda i: client2.post("/harness/complete", json=bad_missing), _N)
    mal_bodies = [v.json().get("code") for v, _ in mal if v is not None and v.status_code == 503]
    out["byok_wire_overcap_typed"] = (
        parked
        and all(e is None for _, e in mal)
        and all(v.status_code == 503 for v, _ in mal)
        and mal_bodies == ["over_capacity"] * _N
    )
    park.gate.set()
    for t in hold_threads:
        t.join(60)
    out["byok_wire_overcap_releases"] = holders == [200] * 16
    # Client-side: malformed byok surfaces as a typed error, not raw.
    remote = _remote(client)
    exc = _exc(lambda: remote.complete(_MSGS, backend="byok", byok={"base_url": "x"}))
    out["byok_client_typed"] = isinstance(exc, ValueError)
    return out


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def sdkconc_audit() -> dict[str, bool]:
    """Run the whole SDK-concurrency battery."""
    results: dict[str, bool] = {}
    with _sdkconc_context():
        for probe in (
            _probe_sdk_complete_storm,
            _probe_sdk_mixed_storm,
            _probe_sdk_store_storm,
            _probe_sdk_background_storm,
            _probe_sdk_identity,
            _probe_sdk_journal_contention,
            _probe_contextvar_isolation,
            _probe_client_retry_concurrent,
            _probe_client_circuit_concurrent,
            _probe_client_idem_concurrent,
            _probe_client_timeout_concurrent,
            _probe_client_timeout_socket,
            _probe_client_stream_concurrent,
            _probe_client_url_headers,
            _probe_sdk_refusal_storm,
            _probe_sdk_wire_interleave,
            _probe_sdk_byok_wire,
        ):
            results.update(probe())
    missing = _EXPECTED_PROBES - results.keys()
    extra = results.keys() - _EXPECTED_PROBES
    if missing or extra:
        raise AssertionError(
            f"battery drifted from the pinned probe set — missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )
    return results


def sdkconc_audit_bench(results: dict[str, bool] | None = None) -> dict[str, Any]:
    """Seal the battery as an ``fx1_sdkconc_audit.v1`` receipt."""
    measured = dict(results) if results is not None else sdkconc_audit()
    out: dict[str, Any] = {
        "kind": "fx1_sdkconc_audit",
        "schema": "fx1_sdkconc_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "results": measured,
            "ok": set(measured) == _EXPECTED_PROBES and all(v is True for v in measured.values()),
        },
        "coverage": {
            "n_probes": len(measured),
            "not_verified": [
                "cross-process two-writer contention on one state_dir — "
                "the writer claim is per-process by design; OS-level "
                "interleave between processes remains the crash-torn "
                "case the journals detect and quarantine on next boot "
                "(journal_audit covers that recovery path)",
            ],
        },
        "interpretation": (
            "Every probe is a measured statement about SDK concurrency "
            "behavior — Fx1Harness thread-safety, ContextVar hygiene, "
            "HarnessClient retry/timeout/circuit-breaker correctness — "
            "exercised end to end against real transports and stores."
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(sdkconc_audit_bench(), indent=1))
