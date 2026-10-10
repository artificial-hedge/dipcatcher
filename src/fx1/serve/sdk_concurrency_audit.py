"""sdk_concurrency_audit — adversarial thread-safety probes on ``Fx1Harness``.

The in-process SDK is the second consumption leg of the fx-1 serve surface:
a single ``Fx1Harness`` is expected to be shared across worker threads
(an HTTP gateway, a batch runner, a notebook fanning out completions). This
battery measures what its shared mutable state does under real concurrency
— not what it is *supposed* to do. Every probe is a literal ``bool``; a
``False`` is a defect pinned as ``False`` until fixed, never retried into
passing.

Seams under test:

- ``_log`` — the cap-256 newest-first ``_CompletionLog`` ring every gated
  call appends to: parallel appends must all land, ``dropped`` must account
  exactly for evictions, snapshots must never show duplicate ids, and
  readers (``completions``/``completion``/``latest``/``usage``) must stay
  coherent while writers flood.
- ``_last_response_headers`` — the latest-call header dict. ``_record_call``
  first stores ``{"x-request-id": ..., ...}`` and *then* mutates the stored
  attribute in place to add ``x-fx1-completion-id`` — a store-then-mutate
  window a second writer can tear (the surviving dict holds one call's
  request-id but the other call's completion-id). The battery schedules a
  writer exactly into that window via ``sys.settrace`` and checks pair
  consistency arithmetically under a per-thread-tagged ``uuid4`` (same-call
  mints are consecutive integers under one tag).
- ``_bg_cancel`` — the ``dict[str, threading.Event]`` mapping in-flight
  background ``/v1/responses`` rids to their cancel events: concurrent
  enqueues must mint unique rids, every worker must reach a terminal
  state, cancel races must resolve coherently, and the map must drain to
  empty after the storm.
- ``_files`` — the OrderedDict of upload-minted file records under
  ``_files_lock``: parallel uploads mint unique ids, content round-trips
  byte-exact, and the 256-cap eviction stays correct under flood.
- Mixed storm — completions + uploads + background responses + reader
  threads all at once: no crashes, no leaked cancel entries, log still
  coherent after the storm settles.

Scheduling determinism comes from ``threading.Barrier`` fan-outs plus the
targeted settrace gate — the torn-pair interleaving is reproduced on
demand, not left to a hopeful scheduler. The battery is self-contained:
``Fx1Harness`` instances against stub backends and temp state dirs, no
server, no network, no API key, nothing written outside the OS temp dir.
All numbers are synthetic measurements.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from types import FrameType
from typing import Any
from unittest import mock

__all__ = ["sdk_concurrency_audit", "sdk_concurrency_audit_bench"]

_AUDIT_LOCK = threading.Lock()
_TAG_MASK = (1 << 64) - 1  # per-thread counter lives in the low 64 bits


def _clean(parts: list[str]) -> str:
    return "ok " + " ".join(parts)[:64]


class _StubBackend:
    """Deterministic completion stub — echoes the last user turn, exposes
    a usage channel so ``usage()`` aggregation has real numbers."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s
        self.last_usage = {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
        del sampling
        self.calls += 1
        if self._delay_s:
            time.sleep(self._delay_s)
        return _clean([m["content"] for m in messages if isinstance(m.get("content"), str)])

    def close(self) -> None:
        pass


def _stub_backend() -> Any:
    return _StubBackend()


def _slow_backend() -> Any:
    """Stub whose complete() sleeps ~50ms — keeps a turn in flight so
    cancels and reader races land inside the window."""
    return _StubBackend(delay_s=0.05)


def _make_sdk(backend: Any = None, state_dir: Path | None = None) -> Any:
    from fx1.harness import Harness
    from fx1.sdk import Fx1Harness

    def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    be = backend if backend is not None else _stub_backend()
    return Fx1Harness(
        harness=Harness(runner=_runner),
        backend_resolver=lambda name, *a, **k: be,
        state_dir=state_dir or Path(tempfile.mkdtemp(prefix="fx1-sdkconc-")),
    )


def _msg(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _parallel(workers: list[Any]) -> None:
    """Run worker callables on threads released by one barrier."""
    barrier = threading.Barrier(len(workers))

    def _wrap(fn: Any) -> None:
        barrier.wait()
        fn()

    ths = [threading.Thread(target=_wrap, args=(fn,), daemon=True) for fn in workers]
    for t in ths:
        t.start()
    for t in ths:
        t.join(60)


def _drain_bg(sdk: Any, timeout_s: float = 30.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not sdk._bg_cancel:
            return True
        time.sleep(0.02)
    return not sdk._bg_cancel


def _wait_terminal(sdk: Any, rid: str, timeout_s: float = 30.0) -> dict[str, Any] | None:
    from fx1.serve.openai_compat import OPENAI_RESPONSE_TERMINAL

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        env: dict[str, Any] = sdk.openai_response_get(rid)
        if env.get("status") in OPENAI_RESPONSE_TERMINAL:
            return env
        time.sleep(0.01)
    env = sdk.openai_response_get(rid)
    return env if isinstance(env, dict) else None


@contextmanager
def _audit_context() -> Iterator[None]:
    """Serializes the battery so probes measure their own threads only."""
    _AUDIT_LOCK.acquire()
    try:
        yield
    finally:
        _AUDIT_LOCK.release()


@contextmanager
def _tagged_uuid4() -> Iterator[None]:
    """Replace ``uuid.uuid4`` with per-thread-tagged counters.

    Each minted UUID is ``UUID(int=(tag << 64) | n)`` — ``tag`` a small
    per-thread ordinal, ``n`` a per-thread counter — so the two mints of
    one ``_record_call`` (completion id, then request id) are consecutive
    low-64 values under one tag, and a torn pair shows either mismatched
    tags or a skipped counter. Restores on exit.
    """
    tls = threading.local()
    tag_lock = threading.Lock()
    tag_of: dict[int, int] = {}

    def _uuid4() -> uuid.UUID:
        tid = threading.get_ident()
        with tag_lock:
            tag = tag_of.setdefault(tid, len(tag_of) + 1)
        n = getattr(tls, "n", 0) + 1
        tls.n = n
        return uuid.UUID(int=(tag << 64) | n)

    with mock.patch.object(uuid, "uuid4", new=_uuid4):
        yield


def _pair_consistent(headers: dict[str, str]) -> bool:
    """Tagged-uuid arithmetic: the request-id must be the mint immediately
    after the completion-id's, under the same thread tag."""
    cid = headers.get("x-fx1-completion-id")
    rid = headers.get("x-request-id")
    if cid is None or rid is None:
        return False
    try:
        cid_i, rid_i = int(cid, 16), int(rid, 16)
    except ValueError:
        return False
    return (cid_i >> 64) == (rid_i >> 64) and (cid_i & _TAG_MASK) + 1 == (rid_i & _TAG_MASK)


# ---------------------------------------------------------------------------
# completion-log integrity under parallel writers
# ---------------------------------------------------------------------------


def _probe_log_writes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    sdk = _make_sdk()

    _parallel(
        [
            lambda t=t: [sdk.complete([_msg(f"w{t}-{i}")], backend="byok") for i in range(8)]
            for t in range(8)
        ]
    )
    snap = sdk.completions(limit=200)
    out["log_all_calls_recorded"] = len(snap) == 64
    out["log_ids_unique"] = len({r.completion_id for r in snap}) == len(snap)
    out["log_dropped_zero_under_cap"] = sdk._log.dropped == 0

    _parallel(
        [
            lambda t=t: [sdk.complete([_msg(f"f{t}-{i}")], backend="byok") for i in range(60)]
            for t in range(5)
        ]
    )
    snap2 = sdk.completions(limit=300)
    out["log_cap_holds_at_256"] = len(snap2) == 256
    out["log_dropped_accounts_exactly"] = sdk._log.dropped == 364 - 256
    out["log_ids_unique_after_flood"] = len({r.completion_id for r in snap2}) == 256
    newest = sdk.completions(limit=1)
    out["log_latest_is_first"] = (
        len(newest) == 1 and newest[0].completion_id == snap2[0].completion_id
    )
    return out


def _probe_log_readers() -> dict[str, bool]:
    """Readers racing a writer flood: every snapshot stays coherent."""
    out: dict[str, bool] = {}
    sdk = _make_sdk()
    stop = threading.Event()
    bad_snapshot = threading.Event()
    bad_get = threading.Event()
    bad_usage = threading.Event()

    def _writer(t: int) -> None:
        for i in range(40):
            sdk.complete([_msg(f"r{t}-{i}")], backend="byok")

    def _reader_snap() -> None:
        from fx1.sdk import CompletionRecord

        while not stop.is_set():
            snap = sdk.completions(limit=300)
            ids = [r.completion_id for r in snap]
            if len(ids) != len(set(ids)) or len(snap) > 256:
                bad_snapshot.set()
            if not all(isinstance(r, CompletionRecord) for r in snap):
                bad_snapshot.set()

    def _reader_get() -> None:
        while not stop.is_set():
            snap = sdk.completions(limit=50)
            if not snap:
                continue
            cid = snap[0].completion_id
            try:
                rec = sdk.completion(cid)
            except KeyError:
                continue  # evicted between snapshot and get is legal
            if rec.completion_id != cid:
                bad_get.set()

    def _reader_usage() -> None:
        while not stop.is_set():
            try:
                sdk.usage()
            except Exception:
                bad_usage.set()

    workers: list[Any] = [lambda t=t: _writer(t) for t in range(6)] + [
        _reader_snap,
        _reader_snap,
        _reader_get,
        _reader_get,
        _reader_usage,
    ]
    _parallel(workers)
    stop.set()
    time.sleep(0.05)

    out["read_snapshot_unique_ids"] = not bad_snapshot.is_set()
    out["read_get_returns_matching_record"] = not bad_get.is_set()
    out["read_usage_never_raises"] = not bad_usage.is_set()

    snap = sdk.completions(limit=300)
    ats = [r.at for r in snap]
    out["read_final_snapshot_newest_first"] = ats == sorted(ats, reverse=True)
    out["read_final_accounts_all"] = len(snap) == 240 and sdk._log.dropped == 0
    return out


def _probe_record_fields() -> dict[str, bool]:
    """Every retained record carries intact fields after a flood."""
    out: dict[str, bool] = {}
    sdk = _make_sdk()
    _parallel(
        [
            lambda t=t: [sdk.complete([_msg(f"v{t}-{i}")], backend="byok") for i in range(20)]
            for t in range(4)
        ]
    )
    snap = sdk.completions(limit=300)
    out["fields_error_absent_on_ok"] = all(r.error is None for r in snap if r.ok)
    out["fields_ok_bool"] = all(isinstance(r.ok, bool) for r in snap)
    out["fields_prompt_hash_64"] = all(len(r.prompt_sha256) == 64 for r in snap)
    out["fields_backend_named"] = all(bool(r.backend) for r in snap)
    out["fields_at_positive"] = all(r.at > 0 for r in snap)
    out["fields_all_succeeded"] = all(r.ok for r in snap)
    out["fields_latency_nonneg"] = all(r.latency_ms >= 0 for r in snap)
    return out


# ---------------------------------------------------------------------------
# _last_response_headers pairing under interleaving
# ---------------------------------------------------------------------------


def _mutation_lineno() -> int:
    """Absolute line of the completion-id attribute mutation in
    ``Fx1Harness._record_call`` — the window a second writer can tear."""
    import inspect

    import fx1.sdk

    try:
        src, start = inspect.getsourcelines(fx1.sdk.Fx1Harness._record_call)
    except (OSError, TypeError):
        return -1
    for i, line in enumerate(src):
        if '"x-fx1-completion-id"] = cid' in line:
            return start + i
    return -1


def _probe_headers_basic() -> dict[str, bool]:
    out: dict[str, bool] = {}
    sdk = _make_sdk()
    res = sdk.complete([_msg("hdr-a")], backend="byok")
    hdrs = sdk.last_response_headers
    out["hdr_ok_shape"] = (
        "x-request-id" in hdrs and hdrs.get("x-fx1-completion-id") == res.completion_id
    )
    out["hdr_ok_has_version_and_ms"] = (
        "x-fx1-api-version" in hdrs and "openai-processing-ms" in hdrs
    )
    h1 = sdk.last_response_headers
    sdk.complete([_msg("hdr-b")], backend="byok")
    h2 = sdk.last_response_headers
    out["hdr_fresh_dict_per_read"] = h1 is not h2
    h1["x-request-id"] = "poisoned"
    out["hdr_caller_mutation_isolated"] = (
        sdk.last_response_headers.get("x-request-id") != "poisoned"
    )
    # serial pairing sanity under tagged mints — the baseline must hold
    ok = True
    with _tagged_uuid4():
        for _ in range(20):
            sdk.complete([_msg("hdr-s")], backend="byok")
            if not _pair_consistent(sdk.last_response_headers):
                ok = False
    out["hdr_serial_pairing_consistent"] = ok
    return out


def _probe_headers_torn_pair() -> dict[str, bool]:
    """Deterministic torn-pair schedule: park writer A between the header
    dict store and the completion-id attribute mutation; run writer B to
    completion; release A. If the surviving dict holds B's request-id with
    A's completion-id, the pairing tore — a defect pinned False."""
    out: dict[str, bool] = {}
    target = _mutation_lineno()
    out["hdr_located_mutation_site"] = target > 0
    if target <= 0:
        out["hdr_pair_consistent_under_interleave"] = False
        return out

    sdk = _make_sdk()
    parked = threading.Event()
    release = threading.Event()
    results: dict[str, Any] = {}
    faults: list[str] = []

    def _tracer(frame: FrameType, event: str, arg: Any) -> Any:
        if event == "line" and frame.f_code.co_name == "_record_call" and frame.f_lineno == target:
            parked.set()
            release.wait(60)
        return _tracer

    def _writer(tag: str, traced: bool) -> None:
        if traced:
            sys.settrace(_tracer)
        try:
            results[tag] = sdk.complete([_msg(f"race-{tag}")], backend="byok")
        except Exception as exc:
            faults.append(f"{tag}:{type(exc).__name__}")
        finally:
            if traced:
                sys.settrace(None)

    with _tagged_uuid4():
        t_a = threading.Thread(target=_writer, args=("a", True), daemon=True)
        t_a.start()
        if not parked.wait(60):
            out["hdr_writer_parked_in_window"] = False
            out["hdr_pair_consistent_under_interleave"] = False
            release.set()
            t_a.join(60)
            return out
        out["hdr_writer_parked_in_window"] = True
        t_b = threading.Thread(target=_writer, args=("b", False), daemon=True)
        t_b.start()
        t_b.join(60)
        release.set()
        t_a.join(60)
        hdrs = sdk.last_response_headers

    out["hdr_no_writer_faults"] = not faults
    res_a = results.get("a")
    res_b = results.get("b")
    a_cid = res_a.completion_id if res_a is not None else None
    b_cid = res_b.completion_id if res_b is not None else None
    out["hdr_both_calls_completed"] = a_cid is not None and b_cid is not None
    out["hdr_pair_consistent_under_interleave"] = _pair_consistent(hdrs)
    if a_cid is not None and b_cid is not None:
        try:
            out["hdr_both_records_logged"] = (
                sdk.completion(a_cid).completion_id == a_cid
                and sdk.completion(b_cid).completion_id == b_cid
            )
        except KeyError:
            out["hdr_both_records_logged"] = False
    else:
        out["hdr_both_records_logged"] = False
    return out


def _probe_headers_under_flood() -> dict[str, bool]:
    """Post-flood coherence: the surviving headers must be one call's own
    pair — never a phantom rid/cid mix."""
    out: dict[str, bool] = {}
    sdk = _make_sdk()
    with _tagged_uuid4():
        _parallel(
            [
                lambda t=t: [sdk.complete([_msg(f"hf{t}-{i}")], backend="byok") for i in range(15)]
                for t in range(6)
            ]
        )
        hdrs = sdk.last_response_headers
    out["hdr_flood_present"] = "x-request-id" in hdrs
    out["hdr_flood_pair_consistent"] = _pair_consistent(hdrs)
    cid = hdrs.get("x-fx1-completion-id")
    if cid is not None:
        try:
            out["hdr_flood_cid_resolves"] = sdk.completion(cid).completion_id == cid
        except KeyError:
            out["hdr_flood_cid_resolves"] = False
    else:
        out["hdr_flood_cid_resolves"] = False
    return out


# ---------------------------------------------------------------------------
# _bg_cancel map under background-response races
# ---------------------------------------------------------------------------


def _probe_bg_enqueue() -> dict[str, bool]:
    out: dict[str, bool] = {}
    sdk = _make_sdk()
    envs: list[dict[str, Any]] = []
    elock = threading.Lock()

    def _enq(t: int) -> None:
        env, _ = sdk.openai_response(
            {"model": "byok", "input": f"bg-{t}", "background": True, "store": True}
        )
        with elock:
            envs.append(env)

    _parallel([lambda t=t: _enq(t) for t in range(16)])
    rids = [str(e["id"]) for e in envs]
    out["bg_all_enqueued"] = len(envs) == 16
    out["bg_unique_rids"] = len(set(rids)) == 16
    out["bg_all_start_queued"] = all(e["status"] == "queued" for e in envs)
    terms = [_wait_terminal(sdk, r) for r in rids]
    out["bg_all_reach_terminal"] = all(
        t is not None and t.get("status") in ("completed", "failed", "cancelled") for t in terms
    )
    out["bg_cancel_map_drains"] = _drain_bg(sdk)
    return out


def _probe_bg_cancel_races() -> dict[str, bool]:
    out: dict[str, bool] = {}
    sdk = _make_sdk(_slow_backend())

    env, _ = sdk.openai_response(
        {"model": "byok", "input": "cancel-me", "background": True, "store": True}
    )
    rid = str(env["id"])
    cancelled = sdk.openai_response_cancel(rid)
    final = _wait_terminal(sdk, rid)
    out["bg_cancel_returns_cancelled"] = cancelled.get("status") == "cancelled"
    out["bg_cancel_persists_terminal"] = final is not None and final.get("status") == "cancelled"
    out["bg_map_drains_after_immediate"] = _drain_bg(sdk)

    env2, _ = sdk.openai_response(
        {"model": "byok", "input": "double-cancel", "background": True, "store": True}
    )
    rid2 = str(env2["id"])
    seen: list[str] = []
    slock = threading.Lock()
    barrier = threading.Barrier(2)

    def _cancel() -> None:
        barrier.wait()
        try:
            r = sdk.openai_response_cancel(rid2)
            with slock:
                seen.append(f"ok:{r.get('status')}")
        except Exception as exc:
            with slock:
                seen.append(f"err:{type(exc).__name__}")

    t1, t2 = threading.Thread(target=_cancel), threading.Thread(target=_cancel)
    t1.start()
    t2.start()
    t1.join(60)
    t2.join(60)
    out["bg_double_cancel_coherent"] = len(seen) == 2 and all(
        s in ("ok:cancelled", "err:OpenAICompatError", "err:KeyError") for s in seen
    )
    out["bg_double_cancel_one_winner"] = "ok:cancelled" in seen
    _wait_terminal(sdk, rid2)
    out["bg_map_drains_after_double"] = _drain_bg(sdk)

    env3, _ = sdk.openai_response(
        {"model": "byok", "input": "finish-first", "background": True, "store": True}
    )
    rid3 = str(env3["id"])
    done3 = _wait_terminal(sdk, rid3)
    out["bg_completed_is_terminal"] = done3 is not None and done3.get("status") in (
        "completed",
        "cancelled",
    )
    try:
        sdk.openai_response_cancel(rid3)
        out["bg_cancel_terminal_refused"] = False
    except Exception as exc:
        out["bg_cancel_terminal_refused"] = "409" in str(exc) or "cancel" in str(exc)
    try:
        sdk.openai_response_cancel("resp_deadbeef")
        out["bg_cancel_unknown_keyerror"] = False
    except KeyError:
        out["bg_cancel_unknown_keyerror"] = True
    except Exception:
        out["bg_cancel_unknown_keyerror"] = False
    out["bg_final_map_empty"] = _drain_bg(sdk)
    return out


def _probe_bg_cancel_sweep() -> dict[str, bool]:
    """Cancel sweep at enqueue / ~mid-flight / post-terminal positions —
    every rid ends terminal and the map drains regardless."""
    out: dict[str, bool] = {}
    sdk = _make_sdk(_slow_backend())
    rids: list[str] = []
    for i in range(9):
        env, _ = sdk.openai_response(
            {"model": "byok", "input": f"sweep-{i}", "background": True, "store": True}
        )
        rid = str(env["id"])
        rids.append(rid)
        if i % 3 == 0:
            sdk.openai_response_cancel(rid)
        elif i % 3 == 1:
            time.sleep(0.02)
            with suppress(Exception):
                sdk.openai_response_cancel(rid)  # terminal is a legal race outcome
    finals = [_wait_terminal(sdk, r) for r in rids]
    out["bg_sweep_all_terminal"] = all(
        f is not None and f.get("status") in ("completed", "failed", "cancelled") for f in finals
    )
    out["bg_sweep_map_drains"] = _drain_bg(sdk)
    out["bg_sweep_immediate_cancels_hold"] = all(
        f["status"] == "cancelled" for i, f in enumerate(finals) if i % 3 == 0 and f is not None
    )
    return out


# ---------------------------------------------------------------------------
# _files store under parallel uploads
# ---------------------------------------------------------------------------


def _probe_files() -> dict[str, bool]:
    out: dict[str, bool] = {}
    sdk = _make_sdk()
    recs: list[tuple[int, dict[str, Any]]] = []
    rlock = threading.Lock()

    def _upload(t: int) -> None:
        rec = sdk.openai_file_create(
            f'{{"x":{t}}}\n'.encode(), purpose="batch", filename=f"f{t}.jsonl"
        )
        with rlock:
            recs.append((t, rec))

    _parallel([lambda t=t: _upload(t) for t in range(32)])
    ids = [r["id"] for _, r in recs]
    out["files_all_uploaded"] = len(recs) == 32
    out["files_unique_ids"] = len(set(ids)) == 32
    out["files_all_retrievable"] = all(sdk.file_card(r["id"])["id"] == r["id"] for _, r in recs)
    out["files_card_omits_content"] = all("_content" not in sdk.file_card(r["id"]) for _, r in recs)
    out["files_content_roundtrip"] = all(
        sdk.file_content(r["id"]) == f'{{"x":{t}}}\n'.encode() for t, r in recs
    )
    try:
        sdk.file_card("file_nonexistent")
        out["files_unknown_card_keyerror"] = False
    except KeyError:
        out["files_unknown_card_keyerror"] = True
    try:
        sdk.file_content("file_nonexistent")
        out["files_unknown_content_keyerror"] = False
    except KeyError:
        out["files_unknown_content_keyerror"] = True
    return out


def _probe_files_cap() -> dict[str, bool]:
    out: dict[str, bool] = {}
    sdk = _make_sdk()
    first_id = last_id = ""
    for i in range(270):
        rec = sdk.openai_file_create(b'{"a":1}\n', purpose="batch", filename=f"c{i}.jsonl")
        if i == 0:
            first_id = rec["id"]
        last_id = rec["id"]
    out["files_cap_holds_256"] = len(sdk._files) == 256
    try:
        sdk.file_card(first_id)
        out["files_oldest_evicted"] = False
    except KeyError:
        out["files_oldest_evicted"] = True
    out["files_newest_retained"] = sdk.file_card(last_id)["id"] == last_id

    def _burst(t: int) -> None:
        sdk.openai_file_create(b'{"b":2}\n', purpose="batch", filename=f"b{t}.jsonl")

    _parallel([lambda t=t: _burst(t) for t in range(24)])
    out["files_cap_holds_under_parallel"] = len(sdk._files) == 256
    out["files_ids_unique_at_cap"] = len(set(sdk._files)) == 256
    return out


# ---------------------------------------------------------------------------
# the everything-at-once storm
# ---------------------------------------------------------------------------


def _probe_storm() -> dict[str, bool]:
    out: dict[str, bool] = {}
    sdk = _make_sdk(_slow_backend())
    stop = threading.Event()
    faults: list[str] = []
    flock = threading.Lock()

    def _fault(tag: str, exc: Exception) -> None:
        with flock:
            faults.append(f"{tag}:{type(exc).__name__}")

    def _completer(t: int) -> None:
        try:
            for i in range(20):
                sdk.complete([_msg(f"storm-{t}-{i}")], backend="byok")
        except Exception as exc:
            _fault("complete", exc)

    def _uploader(t: int) -> None:
        try:
            for i in range(10):
                sdk.openai_file_create(b'{"s":1}\n', purpose="batch", filename=f"s{t}-{i}.jsonl")
        except Exception as exc:
            _fault("upload", exc)

    def _bger(t: int) -> None:
        try:
            for i in range(6):
                sdk.openai_response(
                    {
                        "model": "byok",
                        "input": f"sbg-{t}-{i}",
                        "background": True,
                        "store": True,
                    }
                )
        except Exception as exc:
            _fault("bg", exc)

    def _reader() -> None:
        while not stop.is_set():
            try:
                snap = sdk.completions(limit=5)
                if snap:
                    sdk.completion(snap[0].completion_id)
            except KeyError:
                pass
            except Exception as exc:
                _fault("read", exc)

    workers: list[Any] = (
        [lambda t=t: _completer(t) for t in range(5)]
        + [lambda t=t: _uploader(t) for t in range(3)]
        + [lambda t=t: _bger(t) for t in range(3)]
        + [_reader, _reader]
    )
    _parallel(workers)
    stop.set()
    time.sleep(0.1)
    drained = _drain_bg(sdk, 60)

    out["storm_no_faults"] = not faults
    snap = sdk.completions(limit=300)
    out["storm_log_unique_ids"] = len({r.completion_id for r in snap}) == len(snap)
    out["storm_log_bounded"] = len(snap) <= 256
    out["storm_bg_map_drains"] = drained and len(sdk._bg_cancel) == 0
    out["storm_files_bounded"] = len(sdk._files) <= 256
    out["storm_headers_present"] = "x-request-id" in sdk.last_response_headers
    return out


# ---------------------------------------------------------------------------
# battery + bench
# ---------------------------------------------------------------------------


def sdk_concurrency_audit() -> dict[str, bool]:
    """Every probe, measured against fresh ``Fx1Harness`` instances."""
    out: dict[str, bool] = {}
    with _audit_context():
        out.update(_probe_log_writes())
        out.update(_probe_log_readers())
        out.update(_probe_record_fields())
        out.update(_probe_headers_basic())
        out.update(_probe_headers_torn_pair())
        out.update(_probe_headers_under_flood())
        out.update(_probe_bg_enqueue())
        out.update(_probe_bg_cancel_races())
        out.update(_probe_bg_cancel_sweep())
        out.update(_probe_files())
        out.update(_probe_files_cap())
        out.update(_probe_storm())
    return out


def _git_rev() -> str:
    import subprocess

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def sdk_concurrency_audit_bench() -> dict[str, Any]:
    """Sealed receipt: results are literal measured bools."""
    from fx1 import __version__ as _ver
    from quant_fund.research.receipt_v2 import canonical_json_bytes, hash_bytes

    results = sdk_concurrency_audit()
    ok = bool(results) and all(v is True for v in results.values())
    out: dict[str, Any] = {
        "kind": "fx1_serve_audit",
        "schema": "sdk_concurrency_audit.v1",
        "fx1_version": _ver,
        "git_revision": _git_rev(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {
            "statement": (
                "Fx1Harness shared state (completion log, response headers, "
                "background-cancel map, file store) stays coherent under "
                "parallel callers; every probe is a measured bool."
            ),
            "results": results,
            "ok": ok,
        },
        "coverage": {
            "transport": "in-process SDK (no server, no network)",
            "not_verified": [
                "cross-process safety (the SDK is single-process by design)",
                "wire-level concurrency (the route audits cover ASGI clients)",
                "state_dir journal replay races (durability lane)",
            ],
        },
        "interpretation": (
            "all probes passed — shared-state seams hold under the measured schedules"
            if ok
            else "defects pinned — see claim.results for the False keys"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(sdk_concurrency_audit_bench(), indent=2, sort_keys=True))
