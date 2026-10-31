"""uploads_audit — /v1/uploads chunked-upload lifecycle probe battery.

The uploads surface is a three-phase intent protocol: ``POST /v1/uploads``
opens a byte-bounded intent, ``POST /v1/uploads/{id}/parts`` lands one
multipart ``data`` chunk each, ``POST /v1/uploads/{id}/complete``
assembles the caller-declared parts into a ``file-`` record behind an
md5 gate, and ``POST /v1/uploads/{id}/cancel`` ends a pending intent.
This battery pins the whole lifecycle end to end:

- *Happy path* — create/parts/complete envelopes; the minted ``file-``
  record is retrievable over ``/v1/files`` and byte-exact, in the
  caller's declared part order (repeats included).
- *md5 contract* — a declared checksum gates the mint: mismatch refuses
  fail-closed (``400 checksum_mismatch``), mints no orphan file, and
  stays retryable; absent md5 mints.
- *Part bounds* — the 65th part, an oversized part, an oversized total,
  an empty part, and cumulative-over-declared all refuse; an
  under-declared complete is a retryable 400, not a terminal burn.
- *Ordering/dedup/concurrency* — caller-declared part order governs the
  concat (repeats duplicate bytes); concurrent part adds all land;
  racing completes yield exactly one 200 and exactly one minted file.
- *Lifecycle races* — terminal intents (completed/cancelled/expired)
  refuse parts and completion; cancel replays 200 on a cancelled
  record and 409s on a completed one.
- *TTL/expiry* — an expired intent refuses add/assemble/complete at 410
  and reports ``expired`` honestly; expiry never resurrects.
- *LRU/tombstone* — an evicted intent is 404 everywhere live; under a
  state dir the eviction journals an ``expired`` tombstone that replays
  as 410 after restart — never a resurrected record.
- *Tenancy* — uploads are workspace-level: any authenticated key (env
  or managed, write-scoped) may drive any upload; read-scoped keys get
  403; no key gets 401.
- *Durability* — parts land blob-first + journaled under ``--state-dir``;
  a restart resumes a pending upload under the same md5 gate; cancelled
  and completed intents stay terminal across restarts; completed intents
  drop their part blobs.
- *Downstream* — a minted ``file-`` flows into ``/v1/batches``
  ``input_file_id`` (runs to an output file), ``/v1/fine_tuning/jobs``
  ``training_file``, and ``/v1/vector_stores/{id}/files`` attach; wrong
  purposes and bogus ids refuse in-envelope.
- *Error paths* — malformed multipart, missing boundary, wrong
  content-type, bad JSON bodies, unknown ids: the uniform
  ``{error: {message, type, param, code}}`` envelope, never a bare
  ``{"detail": ...}`` or a 5xx.
- *Client map* — 401/403 → HarnessAuthError, 404 → KeyError,
  422 → ValueError, 501 → NotImplementedError, 503 →
  BackendNotConfiguredError, anything else → HarnessTransportError.

Existing behavior pinned by this battery:

- *Error envelope* — the existing Starlette base-exception handler
  covers body-parse failures as well as FastAPI exceptions. This battery
  extends coverage of that already-merged behavior.

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect by probe name so the finding survives byte-for-byte.

These are SYNTHETIC, stub-backed in-process checks. Restart probes close
all clients and workers before reconstructing apps over the same directory;
they do not start a fresh external process or establish crash atomicity. The
fine-tuning fixture tests file plumbing, not corpus or training quality.

Sealed ``uploads_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fx1.serve.uploads import UPLOAD_MAX_PARTS, UPLOAD_TTL_S, UploadStore, UploadStoreError

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = ["uploads_audit", "uploads_audit_bench"]

_AUDIT_LOCK = threading.Lock()
_RESOURCES: ContextVar[ExitStack] = ContextVar("uploads_audit_resources")


@contextmanager
def _audit_context() -> Iterator[None]:
    """Own temporary state and restore all ambient FX1 configuration.

    Run this diagnostic in a dedicated process: environment overrides are
    process-wide. The lock serializes calls made through this module.
    """
    with _AUDIT_LOCK:
        saved = {
            name: value
            for name, value in os.environ.items()
            if name.startswith("FX1_") or name == "MOONSHOT_API_KEY"
        }
        for name in saved:
            os.environ.pop(name, None)
        try:
            with ExitStack() as resources:
                token = _RESOURCES.set(resources)
                try:
                    yield
                finally:
                    _RESOURCES.reset(token)
        finally:
            for name in list(os.environ):
                if name.startswith("FX1_") or name == "MOONSHOT_API_KEY":
                    os.environ.pop(name, None)
            os.environ.update(saved)


def _temporary_directory() -> Path:
    return Path(
        _RESOURCES.get().enter_context(tempfile.TemporaryDirectory(prefix="uploads_audit_"))
    )


@contextmanager
def _client_session(**kw: Any) -> Iterator[TestClient]:
    """Finish one application's lifespan before a persistence replay."""
    with ExitStack() as resources:
        token = _RESOURCES.set(resources)
        try:
            yield _client(**kw)
        finally:
            _RESOURCES.reset(token)


_API_KEY_ENV = "FX1_API_KEY"
_ROOT_KEY = "uploads-audit-root"

# Two batch-request lines — the upload's declared bytes and concat truth.
_BATCH_LINE = {
    "custom_id": "r{n}",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {"model": "fx1", "messages": [{"role": "user", "content": "hi"}]},
}
_BODY = (
    json.dumps({**_BATCH_LINE, "custom_id": "r1"})
    + "\n"
    + json.dumps({**_BATCH_LINE, "custom_id": "r2"})
    + "\n"
).encode()
_FT_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'

_BATCH_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, "ok", ""


class _B:
    """Backend stub — ``complete`` returns a fixed token string."""

    def complete(self, *a: Any, **k: Any) -> Any:
        return "ok"


def _ft_runner(spec: Any, *, emit: Any, should_cancel: Any) -> Any:
    """Trivial fine-tune runner — a receipt artifact + the ft model name."""
    from fx1.serve.finetune import FTJobOutcome

    art = spec.work_dir / "receipt.json"
    art.write_text("{}")
    return FTJobOutcome(fine_tuned_model=spec.ft_model_name, artifacts={"receipt.json": art})


def _client(**kw: Any) -> TestClient:
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    resources = _RESOURCES.get()
    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()
    kw.setdefault("state_dir", isolated / "state")
    kw.setdefault("ft_dir", isolated / "fine_tuning")
    kw.setdefault("receipts_dir", receipts)
    app = api_mod.create_app(
        harness=Harness(runner=_runner),
        backend_resolver=lambda *a, **k: _B(),
        **kw,
    )
    resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
    client = TestClient(app, raise_server_exceptions=False)
    resources.callback(client.close)
    resources.enter_context(client)
    return client


@contextmanager
def _env_key() -> Iterator[None]:
    prev = os.environ.get(_API_KEY_ENV)
    os.environ[_API_KEY_ENV] = _ROOT_KEY
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = prev


def _create(
    c: TestClient,
    nbytes: int = len(_BODY),
    *,
    purpose: str = "batch",
    filename: str = "in.jsonl",
    headers: dict[str, str] | None = None,
) -> Any:
    return c.post(
        "/v1/uploads",
        json={
            "purpose": purpose,
            "filename": filename,
            "bytes": nbytes,
            "mime_type": "application/jsonl",
        },
        headers=headers or {},
    )


def _part(c: TestClient, uid: str, data: bytes, headers: dict[str, str] | None = None) -> Any:
    return c.post(f"/v1/uploads/{uid}/parts", files={"data": ("p", data)}, headers=headers or {})


def _complete_upload(
    c: TestClient,
    uid: str,
    part_ids: list[str],
    *,
    md5: str | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    body: dict[str, Any] = {"part_ids": part_ids}
    if md5 is not None:
        body["md5"] = md5
    return c.post(f"/v1/uploads/{uid}/complete", json=body, headers=headers or {})


def _code(resp: Any) -> Any:
    """``error.code`` out of the envelope — None if the envelope leaked."""
    body = resp.json()
    err = body.get("error") if isinstance(body, dict) else None
    return err.get("code") if isinstance(err, dict) else None


def _is_envelope(resp: Any) -> bool:
    body = resp.json()
    err = body.get("error") if isinstance(body, dict) else None
    return isinstance(err, dict) and isinstance(err.get("code"), str)


def _minted(
    c: TestClient,
    body: bytes = _BODY,
    *,
    purpose: str = "batch",
    filename: str = "in.jsonl",
) -> str:
    """Drive a full lifecycle → minted ``file-`` id."""
    r = _create(c, len(body), purpose=purpose, filename=filename)
    assert r.status_code == 200, r.text
    uid = str(r.json()["id"])
    mid = len(body) // 2
    p1 = _part(c, uid, body[:mid]).json()["id"]
    p2 = _part(c, uid, body[mid:]).json()["id"]
    done = _complete_upload(c, uid, [str(p1), str(p2)])
    assert done.status_code == 200, done.text
    return str(done.json()["file"]["id"])


def _run_threads(fn: Callable[[int], None], n: int = 8) -> None:
    """Release workers together and propagate exceptions to the caller."""
    barrier = threading.Barrier(n, timeout=10)

    def _w(i: int) -> None:
        barrier.wait()
        fn(i)

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_w, i) for i in range(n)]
        for future in futures:
            future.result(timeout=30)


def _tc_transport(client: TestClient) -> Any:
    """Adapt HarnessClient's transport contract to a TestClient."""

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, Mapping[str, str], bytes]:
        p = urllib.parse.urlparse(url)
        path = p.path + (f"?{p.query}" if p.query else "")
        if method == "GET":
            resp = client.get(path, headers=headers)
        elif method == "DELETE":
            resp = client.delete(path, headers=headers)
        elif method == "PATCH":
            resp = client.patch(path, json=payload, headers=headers)
        elif isinstance(payload, bytes):
            resp = client.post(path, content=payload, headers=headers)
        else:
            resp = client.post(path, json=payload, headers=headers)
        return resp.status_code, dict(resp.headers), resp.content

    return send


def _probe_happy_path(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = _create(c)
    meta = r.json()
    uid = str(meta["id"])
    out["create_returns_upload_envelope"] = (
        r.status_code == 200
        and meta["object"] == "upload"
        and uid.startswith("upload_")
        and meta["status"] == "pending"
        and meta["file"] is None
        and meta["purpose"] == "batch"
        and meta["filename"] == "in.jsonl"
        and meta["bytes"] == len(_BODY)
    )
    out["create_expires_after_ttl"] = (
        isinstance(meta["expires_at"], int)
        and meta["expires_at"] - meta["created_at"] == UPLOAD_TTL_S
    )
    # parts land one at a time; the part envelope carries the upload id
    mid = len(_BODY) // 2
    p1 = _part(c, uid, _BODY[:mid])
    p2 = _part(c, uid, _BODY[mid:])
    out["part_envelope_shape"] = (
        p1.status_code == 200
        and p1.json()["object"] == "upload.part"
        and str(p1.json()["id"]).startswith("part_")
        and p1.json()["upload_id"] == uid
    )
    # caller-declared order governs the concat — add [1, 2], declare [2, 1]
    done = _complete_upload(c, uid, [str(p2.json()["id"]), str(p1.json()["id"])])
    body = done.json()
    out["complete_mints_file_record"] = (
        done.status_code == 200
        and body["status"] == "completed"
        and body["object"] == "upload"
        and str(body["file"]["id"]).startswith("file-")
        and body["file"]["object"] == "file"
        and body["file"]["bytes"] == len(_BODY)
        and body["file"]["status"] == "processed"
        and body["file"]["filename"] == "in.jsonl"
        and body["file"]["purpose"] == "batch"
    )
    fid = str(body["file"]["id"])
    content = c.get(f"/v1/files/{fid}/content")
    out["concat_order_is_caller_declared"] = content.content == _BODY[mid:] + _BODY[:mid]
    got = c.get(f"/v1/files/{fid}")
    out["minted_file_retrievable"] = (
        got.status_code == 200 and got.json()["id"] == fid and got.json()["bytes"] == len(_BODY)
    )
    listed = c.get("/v1/files").json()["data"]
    out["minted_file_listed"] = any(f["id"] == fid for f in listed)
    # the surface exposes no upload-read route — the catch-all owns GETs
    got_up = c.get(f"/v1/uploads/{uid}")
    out["upload_read_is_catch_all_404"] = (
        got_up.status_code == 404 and _code(got_up) == "not_found" and _is_envelope(got_up)
    )
    return out


def _probe_md5(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    uid = str(_create(c, 2).json()["id"])
    pid = str(_part(c, uid, b"ab").json()["id"])
    files_before = len(c.get("/v1/files").json()["data"])
    bad = _complete_upload(c, uid, [pid], md5="0" * 32)
    files_after = len(c.get("/v1/files").json()["data"])
    out["md5_mismatch_400_checksum_mismatch"] = (
        bad.status_code == 400 and _code(bad) == "checksum_mismatch"
    )
    out["md5_mismatch_mints_no_file"] = (
        files_after == files_before and bad.json().get("file") is None
    )
    # mismatch leaves the intent pending — a correct md5 completes it
    ok = _complete_upload(c, uid, [pid], md5=hashlib.md5(b"ab", usedforsecurity=False).hexdigest())
    out["md5_retry_after_mismatch_mints"] = ok.status_code == 200
    fid = str(ok.json()["file"]["id"])
    raw = c.get(f"/v1/files/{fid}/content")
    out["md5_file_bytes_exact"] = (
        raw.content == b"ab"
        and hashlib.md5(raw.content, usedforsecurity=False).hexdigest()
        == hashlib.md5(b"ab", usedforsecurity=False).hexdigest()
    )
    uid2 = str(_create(c, 2).json()["id"])
    p2 = str(_part(c, uid2, b"cd").json()["id"])
    out["md5_absent_mints"] = _complete_upload(c, uid2, [p2]).status_code == 200
    uid3 = str(_create(c, 1).json()["id"])
    p3 = str(_part(c, uid3, b"z").json()["id"])
    malformed = _complete_upload(c, uid3, [p3], md5="tooshort")
    out["md5_malformed_422_envelope"] = malformed.status_code == 422 and _is_envelope(malformed)
    return out


def _probe_part_bounds(c: TestClient, capped: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    big = _create(capped, 65)  # file_bytes_max=64
    out["create_over_cap_413"] = big.status_code == 413 and _code(big) == "file_too_large"
    zero = _create(capped, 0)
    out["create_zero_bytes_422"] = zero.status_code == 422 and _is_envelope(zero)
    uid = str(_create(capped, 64).json()["id"])
    fat = _part(capped, uid, b"x" * 65)
    out["part_over_cap_413"] = fat.status_code == 413 and _code(fat) == "file_too_large"
    empty = _part(capped, uid, b"")
    out["empty_part_400"] = empty.status_code == 400 and _code(empty) == "invalid_request"
    declared = str(_create(capped, 4).json()["id"])
    assert _part(capped, declared, b"ab").status_code == 200
    over = _part(capped, declared, b"cde")
    out["cumulative_over_declared_400"] = (
        over.status_code == 400 and _code(over) == "part_exceeds_declared_bytes"
    )
    # the 64-part bound is exact — the 65th refuses before the byte check
    wide = _create(c, UPLOAD_MAX_PARTS + 1).json()
    wuid = str(wide["id"])
    for _ in range(UPLOAD_MAX_PARTS):
        rr = _part(c, wuid, b"x")
        assert rr.status_code == 200, rr.text
    extra = _part(c, wuid, b"x")
    out["sixty_fifth_part_400"] = extra.status_code == 400 and _code(extra) == "too_many_parts"
    # under-declared complete is a 400, not a burn — the intent stays usable
    u2 = str(_create(c, 2).json()["id"])
    pa = str(_part(c, u2, b"a").json()["id"])
    under = _complete_upload(c, u2, [pa])
    out["complete_under_declared_400"] = (
        under.status_code == 400 and _code(under) == "upload_incomplete"
    )
    pb = str(_part(c, u2, b"b").json()["id"])
    repaired = _complete_upload(c, u2, [pa, pb])
    out["under_declared_is_retryable"] = repaired.status_code == 200
    u3 = str(_create(c, 4).json()["id"])
    _part(c, u3, b"ab")
    ghost = _complete_upload(c, u3, ["part_ghost"])
    out["unknown_part_400_part_not_found"] = (
        ghost.status_code == 400 and _code(ghost) == "part_not_found"
    )
    too_many = _complete_upload(c, u2, ["part_x"] * (UPLOAD_MAX_PARTS + 1))
    out["part_ids_over_cap_422"] = too_many.status_code == 422 and _is_envelope(too_many)
    return out


def _probe_ordering(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # repeats in the declared order duplicate bytes — caller intent governs
    uid = str(_create(c, 4).json()["id"])
    pid = str(_part(c, uid, b"ab").json()["id"])
    done = _complete_upload(c, uid, [pid, pid])
    fid = str(done.json()["file"]["id"]) if done.status_code == 200 else ""
    raw = c.get(f"/v1/files/{fid}/content") if fid else None
    out["duplicate_part_id_repeats_bytes"] = (
        done.status_code == 200 and raw is not None and raw.content == b"abab"
    )
    # concurrent part adds all land — distinct bytes survive the lock
    uid2 = str(_create(c, 8).json()["id"])
    added: list[tuple[int, str]] = []

    def _add(i: int) -> None:
        r = _part(c, uid2, bytes([65 + i]))
        added.append((r.status_code, str(r.json()["id"]) if r.status_code == 200 else ""))

    _run_threads(_add, 8)
    out["concurrent_parts_all_land"] = len(added) == 8 and all(s == 200 for s, _ in added)
    ids = [pid for _, pid in added]
    done2 = _complete_upload(c, uid2, ids)
    fid2 = str(done2.json()["file"]["id"]) if done2.status_code == 200 else ""
    raw2 = c.get(f"/v1/files/{fid2}/content") if fid2 else None
    out["concurrent_parts_complete_exact"] = (
        done2.status_code == 200
        and raw2 is not None
        and sorted(raw2.content) == sorted(bytes(range(65, 73)))
    )
    # racing completes — exactly one winner, exactly one file minted
    uid3 = str(_create(c, 2).json()["id"])
    pid3 = str(_part(c, uid3, b"ab").json()["id"])
    before = {f["id"] for f in c.get("/v1/files").json()["data"]}
    codes: list[int] = []

    def _race(i: int) -> None:
        codes.append(_complete_upload(c, uid3, [pid3]).status_code)

    _run_threads(_race, 8)
    after = {f["id"] for f in c.get("/v1/files").json()["data"]}
    out["racing_complete_single_winner"] = codes.count(200) == 1 and codes.count(409) == 7
    out["racing_complete_mints_one_file"] = len(after - before) == 1
    # sequential second complete is a terminal refusal, not an idempotent replay
    again = _complete_upload(c, uid3, [pid3])
    out["complete_replay_409_terminal"] = again.status_code == 409 and (
        _code(again) == "upload_terminal"
    )
    return out


def _probe_lifecycle(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # completed → everything refuses at 409
    uid = str(_create(c, 2).json()["id"])
    pid = str(_part(c, uid, b"ab").json()["id"])
    assert _complete_upload(c, uid, [pid]).status_code == 200
    r1 = _part(c, uid, b"zz")
    r2 = _complete_upload(c, uid, [pid])
    r3 = c.post(f"/v1/uploads/{uid}/cancel")
    out["part_after_complete_409"] = r1.status_code == 409 and _code(r1) == "upload_terminal"
    out["complete_after_complete_409"] = r2.status_code == 409 and _code(r2) == "upload_terminal"
    out["cancel_after_complete_409"] = r3.status_code == 409 and (_code(r3) == "upload_terminal")
    # cancelled → 409s too, but cancel itself replays 200
    uid2 = str(_create(c, 2).json()["id"])
    first = c.post(f"/v1/uploads/{uid2}/cancel")
    r4 = _part(c, uid2, b"ab")
    r5 = _complete_upload(c, uid2, ["part_x"])
    replay = c.post(f"/v1/uploads/{uid2}/cancel")
    out["cancel_terminal_200"] = first.status_code == 200 and first.json()["status"] == "cancelled"
    out["part_after_cancel_409"] = r4.status_code == 409 and _code(r4) == "upload_terminal"
    out["complete_after_cancel_409"] = r5.status_code == 409 and (_code(r5) == "upload_terminal")
    out["cancel_replay_idempotent_200"] = (
        replay.status_code == 200 and replay.json()["status"] == "cancelled"
    )
    # unknown ids 404 in-envelope on every verb
    u = "upload_nope"
    out["unknown_upload_404_all_verbs"] = (
        _part(c, u, b"x").status_code == 404
        and _code(_part(c, u, b"x")) == "upload_not_found"
        and _complete_upload(c, u, ["part_x"]).status_code == 404
        and _code(_complete_upload(c, u, ["part_x"])) == "upload_not_found"
        and c.post(f"/v1/uploads/{u}/cancel").status_code == 404
        and _code(c.post(f"/v1/uploads/{u}/cancel")) == "upload_not_found"
    )
    return out


def _probe_ttl() -> dict[str, bool]:
    out: dict[str, bool] = {}
    # ttl_s=-1 expires the intent the moment it's touched — no clock games
    store = UploadStore(max_entries=8, max_bytes=64, ttl_s=-1)
    meta = store.create(purpose="batch", filename="x.jsonl", nbytes=2, mime_type="t")

    def _status(fn: Any) -> tuple[int, str] | None:
        try:
            fn()
        except UploadStoreError as exc:
            return exc.status, exc.code
        return None

    out["expired_add_part_410"] = _status(lambda: store.add_part(meta.upload_id, b"ab")) == (
        410,
        "upload_expired",
    )
    out["expired_assemble_410"] = _status(lambda: store.assemble(meta.upload_id, ["p"])) == (
        410,
        "upload_expired",
    )
    out["expired_complete_410"] = _status(
        lambda: store.complete(meta.upload_id, ["p"], content=b"ab", file_id="file-x")
    ) == (410, "upload_expired")
    got = store.get(meta.upload_id)
    out["expired_reports_expired"] = got is not None and got.status == "expired"
    cancelled = store.cancel(meta.upload_id)
    out["expired_cancel_reports_expired"] = cancelled.status == "expired"
    # expiry is terminal — a second touch stays 410, and the store keeps serving
    out["expiry_never_resurrects"] = (
        _status(lambda: store.add_part(meta.upload_id, b"ab")) == (410, "upload_expired")
        and store.create(purpose="batch", filename="y.jsonl", nbytes=1, mime_type="t").status
        == "pending"
    )
    return out


def _probe_eviction(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # wire-level LRU: file_max bounds the upload store too
    c = _client(file_max=1)
    uid_a = str(_create(c, 2).json()["id"])
    uid_b = str(_create(c, 2).json()["id"])
    out["evicted_upload_404_everywhere"] = (
        _part(c, uid_a, b"ab").status_code == 404
        and _complete_upload(c, uid_a, ["p"]).status_code == 404
        and c.post(f"/v1/uploads/{uid_a}/cancel").status_code == 404
    )
    out["evicting_upload_stays_healthy"] = _part(c, uid_b, b"ab").status_code == 200
    # in-memory store: same tombstone, no record survives
    store = UploadStore(max_entries=1, max_bytes=64)
    m1 = store.create(purpose="batch", filename="a.jsonl", nbytes=2, mime_type="t")
    store.create(purpose="batch", filename="b.jsonl", nbytes=2, mime_type="t")

    def _code_of(fn: Any) -> tuple[int, str] | None:
        try:
            fn()
        except UploadStoreError as exc:
            return exc.status, exc.code
        return None

    out["store_eviction_is_tombstone_404"] = _code_of(
        lambda: store.add_part(m1.upload_id, b"ab")
    ) == (404, "upload_not_found")
    # durable nuance: an evicted *pending* intent journals an `expired`
    # tombstone, so a restart replays 410 (honest expiry), not 404 — and
    # never a resurrected pending record.
    state = tmp / "evict"
    store1 = UploadStore(max_entries=1, max_bytes=64, state_dir=state)
    d1 = store1.create(purpose="batch", filename="a.jsonl", nbytes=2, mime_type="t")
    store1.add_part(d1.upload_id, b"ab")
    store1.create(purpose="batch", filename="b.jsonl", nbytes=2, mime_type="t")
    store2 = UploadStore(max_entries=1, max_bytes=64, state_dir=state)
    got = store2.get(d1.upload_id)
    out["durable_eviction_replays_expired_410"] = (
        got is not None
        and got.status == "expired"
        and _code_of(lambda: store2.add_part(d1.upload_id, b"cd")) == (410, "upload_expired")
        and _code_of(lambda: store2.cancel(d1.upload_id)) is None
    )
    return out


def _probe_tenancy() -> dict[str, bool]:
    out: dict[str, bool] = {}
    with _env_key():
        c = _client()
        env_h = {"X-API-Key": _ROOT_KEY}
        r = _create(c, 2, headers=None)
        out["unauthenticated_401"] = r.status_code == 401 and _code(r) == "unauthorized"
        r_bad = _create(c, 2, headers={"X-API-Key": "wrong"})
        out["wrong_key_401"] = r_bad.status_code == 401 and _code(r_bad) == "unauthorized"
        r_bearer = _create(c, 2, headers={"Authorization": f"Bearer {_ROOT_KEY}"})
        out["bearer_auth_accepted_on_v1"] = r_bearer.status_code == 200
        # mint a write-scoped and a read-scoped managed key off the env root
        m1 = c.post("/harness/keys", json={}, headers=env_h)
        m2 = c.post("/harness/keys", json={"scopes": ["read"]}, headers=env_h)
        out["managed_keys_mint_under_env_key"] = m1.status_code == 201 and m2.status_code == 201
        k_write = str(m1.json()["key"])
        k_read = str(m2.json()["key"])
        h_write = {"X-API-Key": k_write}
        h_read = {"X-API-Key": k_read}
        # the pinned rule: uploads are NOT key-tenanted — a second
        # credential drives an upload the env key opened
        uid = str(_create(c, 2, headers=env_h).json()["id"])
        p = _part(c, uid, b"ab", headers=h_write)
        done = _complete_upload(c, uid, [str(p.json()["id"])], headers=h_write)
        out["uploads_are_workspace_level_not_key_scoped"] = (
            p.status_code == 200 and done.status_code == 200
        )
        ro_create = _create(c, 1, headers=h_read)
        ro_part = _part(c, uid, b"x", headers=h_read)
        out["read_scoped_key_cannot_write_403"] = (
            ro_create.status_code == 403
            and _code(ro_create) == "insufficient_scope"
            and ro_part.status_code == 403
        )
        ro_list = c.get("/v1/files", headers=h_read)
        out["read_scoped_key_reads_200"] = ro_list.status_code == 200
        return out


def _probe_durability(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    state = tmp / "state"
    with _client_session(state_dir=state) as c1:
        uid = str(_create(c1, 6).json()["id"])
        p1 = str(_part(c1, uid, b"ab").json()["id"])
        p2 = str(_part(c1, uid, b"cd").json()["id"])
        journal = state / "uploads.jsonl"
        blobs = state / "uploads" / uid
        out["parts_journal_and_blob_on_disk"] = (
            journal.is_file()
            and blobs.is_dir()
            and (blobs / f"{p1}.bin").read_bytes() == b"ab"
            and (blobs / f"{p2}.bin").read_bytes() == b"cd"
        )
        # restart mid-upload: the intent + parts replay; the md5 gate still binds
    with _client_session(state_dir=state) as c2:
        p3 = str(_part(c2, uid, b"ef").json()["id"])
        bad = _complete_upload(c2, uid, [p1, p2, p3], md5="0" * 32)
        out["md5_gate_holds_after_restart"] = (
            bad.status_code == 400 and _code(bad) == "checksum_mismatch"
        )
        done = _complete_upload(
            c2, uid, [p1, p2, p3], md5=hashlib.md5(b"abcdef", usedforsecurity=False).hexdigest()
        )
        fid = str(done.json()["file"]["id"]) if done.status_code == 200 else ""
        raw = c2.get(f"/v1/files/{fid}/content") if fid else None
        out["restart_resumes_and_completes_exact"] = (
            done.status_code == 200 and raw is not None and raw.content == b"abcdef"
        )
        out["parts_dir_dropped_after_complete"] = not blobs.exists()
        # cancelled and completed intents stay terminal across a restart
    with _client_session(state_dir=state) as c3:
        out["completed_stays_terminal_after_restart"] = (
            _part(c3, uid, b"zz").status_code == 409
            and c3.post(f"/v1/uploads/{uid}/cancel").status_code == 409
        )
        uid4 = str(_create(c3, 2).json()["id"])
        c3.post(f"/v1/uploads/{uid4}/cancel")
    with _client_session(state_dir=state) as c4:
        out["cancelled_stays_terminal_after_restart"] = (
            _part(c4, uid4, b"ab").status_code == 409
            and _complete_upload(c4, uid4, ["part_x"]).status_code == 409
            and c4.post(f"/v1/uploads/{uid4}/cancel").json()["status"] == "cancelled"
        )
    return out


def _probe_downstream(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # /v1/batches consumes the minted batch-purpose file end to end
    fid = _minted(c, _BODY, purpose="batch")
    r = c.post(
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    batch = r.json()
    out["batch_accepts_minted_input"] = (
        r.status_code == 200
        and batch["object"] == "batch"
        and batch["input_file_id"] == fid
        and batch["request_counts"]["total"] == 2
    )
    bid = str(batch["id"])
    deadline = time.monotonic() + 30.0
    while True:
        rec = c.get(f"/v1/batches/{bid}").json()
        if rec["status"] in _BATCH_TERMINAL or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    out_file = rec.get("output_file_id")
    out_raw = c.get(f"/v1/files/{out_file}/content") if out_file else None
    out["batch_runs_minted_input_to_output_file"] = (
        rec["status"] == "completed"
        and rec["request_counts"]["completed"] == 2
        and out_file is not None
        and out_raw is not None
        and b"r1" in out_raw.content
        and b"r2" in out_raw.content
    )
    ft_fid = _minted(c, _FT_CORPUS, purpose="fine-tune", filename="t.jsonl")
    wrong_purpose = c.post(
        "/v1/batches",
        json={
            "input_file_id": ft_fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    out["batch_refuses_wrong_purpose_400"] = (
        wrong_purpose.status_code == 400 and _code(wrong_purpose) == "invalid_request"
    )
    ghost = c.post(
        "/v1/batches",
        json={
            "input_file_id": "file-ghost",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    out["batch_unknown_input_404"] = ghost.status_code == 404 and _code(ghost) == "file_not_found"
    # /v1/fine_tuning/jobs validates the minted file's chat shape at submit
    job = c.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": ft_fid},
    )
    out["finetune_accepts_minted_training_file"] = (
        job.status_code == 200
        and job.json()["object"] == "fine_tuning.job"
        and job.json()["training_file"] == ft_fid
    )
    bad_job = c.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": fid},
    )
    out["finetune_refuses_wrong_purpose_400"] = (
        bad_job.status_code == 400 and _code(bad_job) == "invalid_training_file"
    )
    # /v1/vector_stores/{id}/files attaches the minted record to the index
    vs = c.post("/v1/vector_stores", json={"name": "audit-vs"})
    vsid = str(vs.json()["id"]) if vs.status_code == 200 else ""
    att = c.post(f"/v1/vector_stores/{vsid}/files", json={"file_id": fid})
    out["vector_store_attach_minted_file"] = (
        att.status_code == 200
        and att.json()["id"] == fid
        and att.json()["object"] == "vector_store.file"
        and att.json()["status"] in ("in_progress", "completed")
    )
    ghost_att = c.post(f"/v1/vector_stores/{vsid}/files", json={"file_id": "file-ghost"})
    out["vector_store_attach_unknown_404"] = ghost_att.status_code == 404
    return out


def _probe_error_paths(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    uid = str(_create(c, 2).json()["id"])
    parts = f"/v1/uploads/{uid}/parts"
    # multipart declared with no boundary — starlette's parse failure
    r1 = c.post(parts, content=b"--x\r\n", headers={"content-type": "multipart/form-data"})
    out["missing_boundary_400_envelope"] = (
        r1.status_code == 400 and _is_envelope(r1) and _code(r1) == "bad_request"
    )
    # boundary declared but the body is garbage
    r2 = c.post(
        parts,
        content=b"garbage",
        headers={"content-type": "multipart/form-data; boundary=x"},
    )
    out["malformed_multipart_400_envelope"] = (
        r2.status_code == 400 and _is_envelope(r2) and _code(r2) == "bad_request"
    )
    # a valid multipart body whose field isn't named `data`
    r3 = c.post(parts, files={"file": ("p", b"ab")})
    out["wrong_multipart_field_400_envelope"] = (
        r3.status_code == 400 and _code(r3) == "invalid_request"
    )
    # a JSON body on the multipart surface
    r4 = c.post(parts, json={"data": "ab"})
    out["json_body_to_parts_400_envelope"] = (
        r4.status_code == 400 and _code(r4) == "invalid_request"
    )
    # text/plain — no multipart parse path at all
    r5 = c.post(parts, content=b"ab", headers={"content-type": "text/plain"})
    out["text_body_to_parts_400_envelope"] = (
        r5.status_code == 400 and _code(r5) == "invalid_request"
    )
    # unparsable JSON to a JSON route — the routing-layer 400 envelope
    r6 = c.post(
        f"/v1/uploads/{uid}/complete",
        content=b"{bad json",
        headers={"content-type": "application/json"},
    )
    out["bad_json_body_envelope"] = r6.status_code in (400, 422) and _is_envelope(r6)
    # create-time rejections — model bounds and the intent validator
    extra_bad = c.post(
        "/v1/uploads",
        json={
            "purpose": "batch",
            "filename": "x.jsonl",
            "bytes": 1,
            "mime_type": "t",
            "bogus": 1,
        },
    )
    out["create_extra_field_422"] = extra_bad.status_code == 422 and _is_envelope(extra_bad)
    purpose = c.post(
        "/v1/uploads",
        json={"purpose": "user_data", "filename": "x.jsonl", "bytes": 1, "mime_type": "t"},
    )
    out["create_bad_purpose_400"] = purpose.status_code == 400 and (
        _code(purpose) == "invalid_request"
    )
    ext = c.post(
        "/v1/uploads",
        json={"purpose": "batch", "filename": "x.txt", "bytes": 1, "mime_type": "t"},
    )
    out["create_non_jsonl_400"] = ext.status_code == 400 and _code(ext) == "invalid_request"
    no_body = c.post("/v1/uploads")
    out["create_missing_body_422_envelope"] = no_body.status_code == 422 and _is_envelope(no_body)
    return out


def _probe_client_map(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.client import (
        BackendNotConfiguredError,
        HarnessAuthError,
        HarnessClient,
        HarnessTransportError,
    )

    client = HarnessClient(base_url="http://uploads-audit", transport=_tc_transport(c))
    # live statuses through the real wire
    try:
        client.upload_cancel("upload_nope")
        r404 = False
    except KeyError:
        r404 = True
    out["client_404_maps_keyerror"] = r404
    try:
        client.upload_create(bytes=0)
        r422 = False
    except ValueError:
        r422 = True
    out["client_422_maps_valueerror"] = r422
    uid = str(client.upload_create(bytes=2)["id"])
    client.upload_cancel(uid)
    try:
        client.upload_part(uid, b"ab")
        r409 = False
    except HarnessTransportError:
        r409 = True
    out["client_409_maps_transport_error"] = r409
    # the client roundtrip mints the same file the wire does
    up = client.upload_create(bytes=len(_BODY), filename="rt.jsonl")
    pa = client.upload_part(str(up["id"]), _BODY[: len(_BODY) // 2])
    pb = client.upload_part(str(up["id"]), _BODY[len(_BODY) // 2 :])
    done = client.upload_complete(str(up["id"]), [str(pa["id"]), str(pb["id"])])
    out["client_upload_roundtrip_mints_file"] = done["status"] == "completed" and str(
        done["file"]["id"]
    ).startswith("file-")

    # the rest of the map — canned statuses through the same code path
    def _mapped(status: int) -> Any:
        def send(
            method: str,
            url: str,
            payload: dict[str, Any] | bytes | None,
            headers: dict[str, str],
            timeout_s: float,
        ) -> tuple[int, Mapping[str, str], bytes]:
            return status, {}, b'{"error": {"message": "m", "code": "x"}}'

        c2 = HarnessClient(base_url="http://uploads-audit", transport=send)
        try:
            c2.upload_cancel("upload_x")
        except Exception as exc:  # noqa: BLE001 — the map IS the contract
            return exc
        return None

    out["client_401_maps_autherror"] = isinstance(_mapped(401), HarnessAuthError)
    out["client_403_maps_autherror"] = isinstance(_mapped(403), HarnessAuthError)
    out["client_501_maps_notimplemented"] = isinstance(_mapped(501), NotImplementedError)
    out["client_503_maps_backenderror"] = isinstance(_mapped(503), BackendNotConfiguredError)
    out["client_500_maps_transport_error"] = isinstance(_mapped(500), HarnessTransportError)
    # keyed app: a client without the key gets the auth error
    with _env_key():
        keyed = _client()
        kc = HarnessClient(base_url="http://uploads-audit", transport=_tc_transport(keyed))
        try:
            kc.upload_create(bytes=2)
            r401 = False
        except HarnessAuthError:
            r401 = True
        kc2 = HarnessClient(
            base_url="http://uploads-audit",
            transport=_tc_transport(keyed),
            api_key=_ROOT_KEY,
        )
        out["client_no_key_401"] = r401
        out["client_api_key_roundtrips"] = kc2.upload_create(bytes=2)["object"] == "upload"
    return out


def _probe_sdk_twin() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.sdk import Fx1Harness

    sdk = Fx1Harness(backend_resolver=lambda *a, **k: _B())
    up = sdk.upload_create(bytes=len(_BODY))
    p1 = sdk.upload_part(str(up["id"]), _BODY[: len(_BODY) // 2])
    p2 = sdk.upload_part(str(up["id"]), _BODY[len(_BODY) // 2 :])
    done = sdk.upload_complete(str(up["id"]), [str(p1["id"]), str(p2["id"])])
    out["sdk_twin_lifecycle_mints_file"] = (
        up["object"] == "upload"
        and done["status"] == "completed"
        and str(done["file"]["id"]).startswith("file-")
        and sdk.file_content(str(done["file"]["id"])) == _BODY
    )
    up2 = sdk.upload_create(bytes=2)
    q1 = sdk.upload_part(str(up2["id"]), b"ab")
    try:
        sdk.upload_complete(str(up2["id"]), [str(q1["id"])], md5="0" * 32)
        gated = False
    except UploadStoreError as exc:
        gated = exc.code == "checksum_mismatch"
    out["sdk_twin_md5_fail_closed"] = gated
    try:
        sdk.upload_cancel(str(up2["id"]))
        sdk.upload_part(str(up2["id"]), b"zz")
        refused = False
    except UploadStoreError:
        refused = True
    out["sdk_twin_terminal_refusals"] = refused
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def uploads_audit() -> dict[str, Any]:
    """Run every probe against live in-process apps; literal bools out."""
    out: dict[str, Any] = {}
    with _audit_context():
        wd = _temporary_directory()
        c = _client()
        capped = _client(file_bytes_max=64)
        ft_client = _client(ft_runner=_ft_runner, ft_dir=wd / "ft")
        out.update(_probe_happy_path(c))
        out.update(_probe_md5(c))
        out.update(_probe_part_bounds(c, capped))
        out.update(_probe_ordering(c))
        out.update(_probe_lifecycle(c))
        out.update(_probe_ttl())
        out.update(_probe_eviction(wd))
        out.update(_probe_tenancy())
        out.update(_probe_durability(wd))
        out.update(_probe_downstream(ft_client))
        out.update(_probe_error_paths(c))
        out.update(_probe_client_map(c))
        out.update(_probe_sdk_twin())
    return out


def uploads_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = uploads_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "uploads_audit",
        "schema": "uploads_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok, "defects": defects},
        "interpretation": (
            "Selected SYNTHETIC in-process /v1/uploads checks passed: intents carry a "
            "byte bound and a TTL, parts land bounded by the declared total "
            "and the 64-part cap, caller-declared part order (repeats "
            "included) assembles byte-exact behind an md5 gate that refuses "
            "fail-closed and stays retryable, terminal intents refuse parts "
            "and completion at 409/410; cancelled cancel replays and expired "
            "cancel reports expired, "
            "expiry never resurrects, evicted intents are real tombstones "
            "(404 live, honest 410-expired after a durable restart), "
            "credentials are workspace-level with scope-gated writes, parts "
            "survive restarts under the same checksum gate, minted files "
            "flow into batches/fine-tuning/vector stores, every refusal "
            "arrives in the OpenAI error envelope — including body-parse "
            "failures — and the client maps statuses to its exception "
            "taxonomy. The SDK twin covers the tested lifecycle. Reconstruction "
            "uses closed in-process apps, not external-process or crash tests; "
            "stub fine-tuning checks plumbing, not training or corpus quality."
            if ok
            else f"UPLOADS AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(uploads_audit_bench(), indent=2, sort_keys=True))
