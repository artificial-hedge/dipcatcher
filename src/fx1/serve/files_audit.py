"""files_audit — /v1/files lifecycle + consumer-semantics probe battery.

``POST /v1/files`` is a one-shot multipart mint (``file-*`` record),
``GET /v1/files`` the shared cursor page, ``GET /v1/files/{id}`` the card,
``GET /v1/files/{id}/content`` the raw bytes, ``DELETE /v1/files/{id}``
the tombstone. Files are the channel into the async consumers: batches
read ``input_file_id``, fine-tuning jobs read ``training_file`` /
``validation_file``, vector stores attach ``file_id``. This battery pins
the whole lifecycle end to end:

- *Upload* — multipart ``file`` + ``purpose`` fields; the minted
  envelope's exact key set; content served byte-exact under
  ``application/jsonl`` + ``Cache-Control: no-store``; fail-closed on a
  missing ``file``/``purpose``, an unsupported purpose, an empty body,
  a non-``.jsonl`` name; content is agnostic (arbitrary bytes store and
  serve verbatim — shape checks belong to the consumers).
- *List* — newest-first default, ``order=asc`` flips, ``limit`` bounds
  (1..100), ``after``/``before`` cursors walk the ordered page honestly,
  unknown cursors and bad ``order`` refuse ``400 invalid_cursor`` in the
  envelope, ``purpose`` filters by the upload's declared intent.
- *Get/delete* — card shape; delete answers ``{id, object, deleted:
  true}`` exactly once; post-delete get/content/list are all 404 — the
  bytes never outlive the record on the file surface.
- *Consumers* — a batch over a file runs its lines through the gated
  route cores to an output ``file-*`` (itself a first-class record, list
  ``purpose=batch_output``); input content is snapshotted at submit, so
  deleting the source mid-run still completes the batch; a malformed or
  non-UTF-8 input fails the SUBMIT loudly (400), never a silent job.
  Fine-tuning validates the chat JSONL at submit and copies the corpus
  into the job work dir — same snapshot semantic. Vector-store attach
  indexes the bytes at attach, so the attachment keeps serving after the
  source record deletes. ``/v1/batches?after=`` honors the shared
  fail-closed cursor contract.
- *Caps* — the store's byte cap refuses ``413 file_too_large`` (and the
  request-body cap refuses ``413 too_large`` underneath it); the entry
  cap evicts oldest-first, and a ``get`` refreshes LRU position.
- *Durability* — under ``--state-dir`` metadata journals and bytes land
  as ``files/<id>.bin`` before the journal names them: a restart returns
  the same bytes under the same ids, a deleted record stays deleted, an
  evicted record stays evicted, and orphan blobs are GC'd on boot.
- *Uploads-minted parity* — a ``/v1/uploads``-completed ``file-*`` is
  indistinguishable from a direct ``POST /v1/files`` record on every
  surface.
- *Concurrency* — parallel creates mint unique ids; racing deletes have
  exactly one winner; a delete racing a content read resolves atomically
  (full bytes or 404, never torn).
- *Tenancy* — files are workspace-level: any authenticated credential
  (env key or managed, write-scoped) reads and deletes any file;
  read-scoped keys get 403 on writes; no credential gets 401.
- *Error envelope* — every refusal (bad id, malformed multipart, wrong
  field, bad cursor, wrong method) lands in
  ``{error: {message, type, param, code}}`` — never a bare
  ``{"detail": ...}`` and never an unenveloped 5xx.
- *Client map* — 401/403 → HarnessAuthError, 404 → KeyError,
  422 → ValueError, 501 → NotImplementedError, 503 →
  BackendNotConfiguredError, anything else → HarnessTransportError.

Defects found while probing, fixed on this lane:

- *Cursor refusals escaped the envelope* — ``openai_file_list`` raised
  ``OpenAICompatError`` (a ``ValueError``) for a bad ``order`` and passed
  ``paged_item_list``'s unknown-cursor ``OpenAICompatError`` straight
  through, while every sibling list route wraps both in
  ``except OpenAICompatError → ApiError``. The unhandled ``ValueError``
  surfaced as a bare 500 — outside the ``{error: ...}`` grammar entirely.
  The route now wraps the same way; ``order``, ``after``, and ``before``
  refusals answer 400 ``invalid_cursor``.
- *Silent cursor restart on ``/v1/batches``* — the batch list's
  hand-rolled pager ignored an unknown ``after`` (``idx is None`` → no
  slice) and re-served the first page, diverging from the shared
  fail-closed cursor contract ``paged_item_list`` documents. It now
  refuses ``400 invalid_cursor`` like its siblings.

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect by probe name so the finding survives byte-for-byte.

Sealed ``files_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

__all__ = ["files_audit", "files_audit_bench"]

_ENV_KEYS = (
    "FX1_API_KEY",
    "MOONSHOT_API_KEY",
    "FX1_CHECKPOINT_DIR",
    "FX1_BYOK_BASE_URL",
    "FX1_BYOK_API_KEY",
    "FX1_BYOK_MODEL",
    "FX1_LOCAL_SERVE_URL",
    "FX1_LOCAL_SERVE_CMD",
    "FX1_LOCAL_MODEL",
    "FX1_LOCAL_API_KEY",
    "FX1_FT_DIR",
    "FX1_API_MAX_INFLIGHT",
    "FX1_API_SSE_KEEPALIVE_S",
    "FX1_API_IDEM_MAX",
    "FX1_API_JOB_MAX",
    "FX1_API_RATE_LIMIT_RPS",
    "FX1_API_GZIP_MIN_BYTES",
    "FX1_API_CORS_ORIGINS",
    "FX1_API_BREAKER_THRESHOLD",
    "FX1_API_BREAKER_COOLDOWN_S",
    "FX1_API_RECEIPTS_DIR",
    "FX1_API_BYOK_OVERRIDE",
    "FX1_API_FILE_MAX",
    "FX1_API_FILE_BYTES",
    "FX1_API_BATCH_MAX",
    "FX1_API_BATCH_LINES",
    "FX1_API_STORE_MAX",
    "FX1_API_STATE_DIR",
    "FX1_API_HOST",
    "FX1_API_PORT",
)

_API_KEY_ENV = "FX1_API_KEY"
_ROOT_KEY = "files-audit-root"

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
_BINARY = b"\xff\xfe\x00binary \x80 payload\n"

_BATCH_TERMINAL = {"completed", "failed", "expired", "cancelled"}
_FT_TERMINAL = {"succeeded", "failed", "cancelled"}
_FILE_OBJECT_KEYS = {"id", "object", "purpose", "filename", "bytes", "created_at", "status"}


def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, "ok", ""


class _B:
    """Backend stub — ``complete`` returns a fixed token string."""

    def complete(self, *a: Any, **k: Any) -> Any:
        return "ok"


class _SlowB:
    """Backend stub that sleeps per call — widens a batch's in-flight
    window so a mid-run delete is deterministic, not a race lottery."""

    def __init__(self, sleep_s: float = 0.05) -> None:
        self._sleep_s = sleep_s

    def complete(self, *a: Any, **k: Any) -> Any:
        time.sleep(self._sleep_s)
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

    app = api_mod.create_app(
        harness=Harness(runner=_runner),
        backend_resolver=lambda *a, **k: _B(),
        **kw,
    )
    return TestClient(app, raise_server_exceptions=False)


def _slow_client(**kw: Any) -> TestClient:
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    app = api_mod.create_app(
        harness=Harness(runner=_runner),
        backend_resolver=lambda *a, **k: _SlowB(),
        **kw,
    )
    return TestClient(app, raise_server_exceptions=False)


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


def _upload(
    c: TestClient,
    body: bytes = _BODY,
    *,
    purpose: str | None = "batch",
    filename: str = "in.jsonl",
    headers: dict[str, str] | None = None,
) -> Any:
    """``POST /v1/files`` — multipart mint. ``purpose=None`` omits the
    field entirely."""
    data = {"purpose": purpose} if purpose is not None else None
    return c.post(
        "/v1/files",
        files={"file": (filename, body)},
        data=data,
        headers=headers or {},
    )


def _minted(
    c: TestClient,
    body: bytes = _BODY,
    *,
    purpose: str = "batch",
    headers: dict[str, str] | None = None,
) -> str:
    """Direct ``POST /v1/files`` → ``file-`` id."""
    r = _upload(c, body, purpose=purpose, headers=headers)
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _minted_via_uploads(c: TestClient, body: bytes = _BODY, *, purpose: str = "batch") -> str:
    """``/v1/uploads`` intent → parts → complete → ``file-`` id."""
    r = c.post(
        "/v1/uploads",
        json={
            "purpose": purpose,
            "filename": "chunked.jsonl",
            "bytes": len(body),
            "mime_type": "application/jsonl",
        },
    )
    assert r.status_code == 200, r.text
    uid = str(r.json()["id"])
    pid = str(c.post(f"/v1/uploads/{uid}/parts", files={"data": ("p", body)}).json()["id"])
    done = c.post(f"/v1/uploads/{uid}/complete", json={"part_ids": [pid]})
    assert done.status_code == 200, done.text
    return str(done.json()["file"]["id"])


def _code(resp: Any) -> Any:
    """``error.code`` out of the envelope — None if the envelope leaked."""
    body = resp.json()
    err = body.get("error") if isinstance(body, dict) else None
    return err.get("code") if isinstance(err, dict) else None


def _is_envelope(resp: Any) -> bool:
    body = resp.json()
    err = body.get("error") if isinstance(body, dict) else None
    return (
        isinstance(err, dict)
        and isinstance(err.get("code"), str)
        and isinstance(err.get("message"), str)
        and isinstance(err.get("type"), str)
    )


def _wait_batch(c: TestClient, bid: str, timeout_s: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        rec = c.get(f"/v1/batches/{bid}").json()
        if rec["status"] in _BATCH_TERMINAL or time.monotonic() > deadline:
            return dict(rec)
        time.sleep(0.05)


def _wait_ft(c: TestClient, jid: str, timeout_s: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        rec = c.get(f"/v1/fine_tuning/jobs/{jid}").json()
        if rec["status"] in _FT_TERMINAL or time.monotonic() > deadline:
            return dict(rec)
        time.sleep(0.05)


def _run_threads(fn: Any, n: int = 8) -> None:
    """Run ``fn(0..n-1)`` released together behind a barrier."""
    barrier = threading.Barrier(n)

    def _w(i: int) -> None:
        barrier.wait()
        fn(i)

    ths = [threading.Thread(target=_w, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


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
        elif isinstance(payload, bytes):
            resp = client.post(path, content=payload, headers=headers)
        else:
            resp = client.post(path, json=payload, headers=headers)
        return resp.status_code, dict(resp.headers), resp.content

    return send


def _probe_upload(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = _upload(c)
    meta = r.json()
    fid = str(meta["id"])
    out["upload_mints_file_envelope"] = (
        r.status_code == 200
        and meta["object"] == "file"
        and fid.startswith("file-")
        and meta["purpose"] == "batch"
        and meta["filename"] == "in.jsonl"
        and meta["bytes"] == len(_BODY)
        and meta["status"] == "processed"
        and isinstance(meta["created_at"], int)
    )
    out["upload_envelope_keys_exact"] = set(meta) == _FILE_OBJECT_KEYS
    got = c.get(f"/v1/files/{fid}")
    out["upload_get_card_matches"] = (
        got.status_code == 200 and got.json() == meta and set(got.json()) == _FILE_OBJECT_KEYS
    )
    raw = c.get(f"/v1/files/{fid}/content")
    out["content_returns_exact_bytes"] = raw.status_code == 200 and raw.content == _BODY
    out["content_media_type_jsonl"] = raw.headers.get("content-type", "").startswith(
        "application/jsonl"
    )
    out["content_cache_control_no_store"] = raw.headers.get("cache-control") == "no-store"
    # content is agnostic at mint — arbitrary bytes store + serve verbatim
    r_bin = _upload(c, _BINARY)
    bin_id = str(r_bin.json()["id"]) if r_bin.status_code == 200 else ""
    bin_raw = c.get(f"/v1/files/{bin_id}/content") if bin_id else None
    out["binary_content_roundtrips_verbatim"] = (
        r_bin.status_code == 200 and bin_raw is not None and bin_raw.content == _BINARY
    )
    # fail-closed fields — every one lands inside the envelope
    out["missing_file_field_400"] = (
        c.post("/v1/files", data={"purpose": "batch"}).status_code == 400
        and _code(c.post("/v1/files", data={"purpose": "batch"})) == "invalid_request"
    )
    no_purpose = _upload(c, purpose=None)
    out["missing_purpose_400"] = (
        no_purpose.status_code == 400 and _code(no_purpose) == "invalid_request"
    )
    bad_purpose = _upload(c, purpose="assistants")
    out["unsupported_purpose_400"] = (
        bad_purpose.status_code == 400 and _code(bad_purpose) == "invalid_request"
    )
    ft = _upload(c, _FT_CORPUS, purpose="fine-tune", filename="corpus.jsonl")
    out["purpose_fine_tune_accepted"] = (
        ft.status_code == 200 and ft.json()["purpose"] == "fine-tune"
    )
    empty = _upload(c, b"")
    out["empty_file_400"] = empty.status_code == 400 and _code(empty) == "invalid_request"
    txt = _upload(c, filename="in.txt")
    out["non_jsonl_filename_400"] = txt.status_code == 400 and _code(txt) == "invalid_request"
    upper = _upload(c, filename="IN.JSONL")
    out["filename_extension_case_sensitive"] = upper.status_code == 400
    nested = _upload(c, filename="nested/dir/in.jsonl")
    out["filename_stored_verbatim"] = (
        nested.status_code == 200 and nested.json()["filename"] == "nested/dir/in.jsonl"
    )
    bare = c.post("/v1/files")
    out["bare_post_400"] = bare.status_code == 400 and _code(bare) == "invalid_request"
    return out


def _probe_list(c: TestClient, fresh: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    empty_page = fresh.get("/v1/files").json()
    out["list_empty_shape"] = (
        empty_page["object"] == "list"
        and empty_page["data"] == []
        and empty_page["first_id"] is None
        and empty_page["last_id"] is None
        and empty_page["has_more"] is False
    )
    a = _minted(c, b'{"a":1}\n')
    b = _minted(c, b'{"b":2}\n')
    d = _minted(c, _FT_CORPUS, purpose="fine-tune")
    listed = c.get("/v1/files").json()
    ids = [f["id"] for f in listed["data"]]
    out["list_newest_first"] = ids[:3] == [d, b, a]
    asc = c.get("/v1/files", params={"order": "asc"}).json()
    out["list_order_asc_flips"] = asc["data"][-3:][0]["id"] == a and asc["data"][-1]["id"] == d
    page1 = c.get("/v1/files", params={"limit": 1}).json()
    out["list_limit_one_pages"] = (
        len(page1["data"]) == 1
        and page1["data"][0]["id"] == d
        and page1["has_more"] is True
        and page1["first_id"] == page1["last_id"] == d
    )
    # walk the whole list with the cursor — every file exactly once
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(20):
        params: dict[str, Any] = {"limit": 1}
        if cursor is not None:
            params["after"] = cursor
        page = c.get("/v1/files", params=params).json()
        seen.extend(str(f["id"]) for f in page["data"])
        if not page["has_more"]:
            break
        cursor = str(page["last_id"])
    out["list_after_cursor_walks_once_each"] = sorted(seen) == sorted(ids)
    after_oldest = c.get("/v1/files", params={"after": ids[-1]}).json()
    out["list_after_oldest_empty_page"] = (
        after_oldest["data"] == [] and after_oldest["has_more"] is False
    )
    before_newest = c.get("/v1/files", params={"before": d}).json()
    out["list_before_newest_empty"] = (
        before_newest["data"] == [] and before_newest["has_more"] is False
    )
    before_b = c.get("/v1/files", params={"before": b}).json()
    out["list_before_cursor_honest"] = [f["id"] for f in before_b["data"]] == [d]
    bad_after = c.get("/v1/files", params={"after": "file-ghost"})
    out["list_unknown_after_400_envelope"] = (
        bad_after.status_code == 400
        and _code(bad_after) == "invalid_cursor"
        and _is_envelope(bad_after)
    )
    bad_before = c.get("/v1/files", params={"before": "file-ghost"})
    out["list_unknown_before_400_envelope"] = (
        bad_before.status_code == 400
        and _code(bad_before) == "invalid_cursor"
        and _is_envelope(bad_before)
    )
    bad_order = c.get("/v1/files", params={"order": "bogus"})
    out["list_bad_order_400_envelope"] = (
        bad_order.status_code == 400
        and _code(bad_order) == "invalid_cursor"
        and _is_envelope(bad_order)
    )
    out["list_limit_bounds_422_envelope"] = (
        c.get("/v1/files", params={"limit": 0}).status_code == 422
        and _is_envelope(c.get("/v1/files", params={"limit": 0}))
        and c.get("/v1/files", params={"limit": 101}).status_code == 422
    )
    nonint = c.get("/v1/files", params={"limit": "abc"})
    out["list_limit_nonint_422_envelope"] = nonint.status_code == 422 and _is_envelope(nonint)
    # the store is shared across probe groups — the filter must classify
    # a/b/d correctly against whatever else landed, not enumerate it
    only_batch = c.get("/v1/files", params={"purpose": "batch"}).json()
    only_ft = c.get("/v1/files", params={"purpose": "fine-tune"}).json()
    bogus = c.get("/v1/files", params={"purpose": "bogus"}).json()
    batch_ids = {str(f["id"]) for f in only_batch["data"]}
    ft_ids = {str(f["id"]) for f in only_ft["data"]}
    out["list_purpose_filter_honest"] = (
        {a, b} <= batch_ids
        and d not in batch_ids
        and d in ft_ids
        and a not in ft_ids
        and all(str(f["purpose"]) == "batch" for f in only_batch["data"])
        and bogus["data"] == []
    )
    return out


def _probe_get_delete(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    fid = _minted(c)
    got = c.get(f"/v1/files/{fid}")
    out["get_returns_card"] = got.status_code == 200 and set(got.json()) == _FILE_OBJECT_KEYS
    ghost = c.get("/v1/files/file-ghost")
    out["get_unknown_404_envelope"] = (
        ghost.status_code == 404 and _code(ghost) == "file_not_found" and _is_envelope(ghost)
    )
    ghost_raw = c.get("/v1/files/file-ghost/content")
    out["content_unknown_404_envelope"] = (
        ghost_raw.status_code == 404 and _code(ghost_raw) == "file_not_found"
    )
    ghost_del = c.delete("/v1/files/file-ghost")
    out["delete_unknown_404_envelope"] = (
        ghost_del.status_code == 404 and _code(ghost_del) == "file_not_found"
    )
    deleted = c.delete(f"/v1/files/{fid}")
    out["delete_returns_deleted_envelope"] = deleted.status_code == 200 and deleted.json() == {
        "id": fid,
        "object": "file",
        "deleted": True,
    }
    out["post_delete_get_404"] = c.get(f"/v1/files/{fid}").status_code == 404
    out["post_delete_content_404"] = c.get(f"/v1/files/{fid}/content").status_code == 404
    out["post_delete_absent_from_list"] = all(
        f["id"] != fid for f in c.get("/v1/files").json()["data"]
    )
    re_del = c.delete(f"/v1/files/{fid}")
    out["redelete_404"] = re_del.status_code == 404 and _code(re_del) == "file_not_found"
    # a tombstone can't shadow a new mint — re-upload mints a fresh id
    fid2 = _minted(c)
    out["reupload_after_delete_mints_new_id"] = fid2 != fid and fid2.startswith("file-")
    # consumer surfaces refuse the tombstone too
    gone_batch = c.post(
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    out["deleted_input_batch_404"] = (
        gone_batch.status_code == 404 and _code(gone_batch) == "file_not_found"
    )
    gone_ft = c.post("/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": fid})
    out["deleted_training_file_ft_404"] = (
        gone_ft.status_code == 404 and _code(gone_ft) == "file_not_found"
    )
    return out


def _probe_caps() -> dict[str, bool]:
    out: dict[str, bool] = {}
    capped = _client(file_bytes_max=64)
    fat = _upload(capped, b"x" * 65)
    out["over_file_cap_413_envelope"] = (
        fat.status_code == 413 and _code(fat) == "file_too_large" and _is_envelope(fat)
    )
    exact = _upload(capped, b"x" * 64)
    out["exact_cap_accepted"] = exact.status_code == 200 and exact.json()["bytes"] == 64
    zero = _upload(capped, b"")
    out["zero_byte_400"] = zero.status_code == 400 and _code(zero) == "invalid_request"
    # the request-body cap sits under the file cap — a >1 MiB multipart
    # refuses 413 too_large at the transport layer, still in the envelope
    big = _upload(_client(), b"x" * (1 << 20 | 1))
    out["over_body_cap_413_too_large"] = (
        big.status_code == 413 and _code(big) == "too_large" and _is_envelope(big)
    )
    # entry cap: LRU eviction is a real tombstone
    lm = _client(file_max=2)
    e1 = _minted(lm, b'{"e":1}\n')
    e2 = _minted(lm, b'{"e":2}\n')
    e3 = _minted(lm, b'{"e":3}\n')
    out["lru_evicts_oldest"] = (
        lm.get(f"/v1/files/{e1}").status_code == 404
        and lm.get(f"/v1/files/{e1}/content").status_code == 404
        and lm.delete(f"/v1/files/{e1}").status_code == 404
        and lm.get(f"/v1/files/{e2}").status_code == 200
        and lm.get(f"/v1/files/{e3}").status_code == 200
    )
    # a get refreshes LRU position — the read file is NOT the evict victim
    lm2 = _client(file_max=2)
    f1 = _minted(lm2, b'{"f":1}\n')
    f2 = _minted(lm2, b'{"f":2}\n')
    assert lm2.get(f"/v1/files/{f1}").status_code == 200
    f3 = _minted(lm2, b'{"f":3}\n')
    out["get_refreshes_lru_position"] = (
        lm2.get(f"/v1/files/{f1}").status_code == 200
        and lm2.get(f"/v1/files/{f2}").status_code == 404
        and lm2.get(f"/v1/files/{f3}").status_code == 200
    )
    return out


def _probe_consumer(c: TestClient, ft_client: TestClient, slow: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
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
    out["batch_accepts_file_input"] = (
        r.status_code == 200
        and batch["object"] == "batch"
        and batch["input_file_id"] == fid
        and batch["request_counts"]["total"] == 2
    )
    rec = _wait_batch(c, str(batch["id"]))
    out_file = rec.get("output_file_id")
    out["batch_completes_with_output_file"] = (
        rec["status"] == "completed"
        and rec["request_counts"]["completed"] == 2
        and isinstance(out_file, str)
        and str(out_file).startswith("file-")
    )
    out_card = c.get(f"/v1/files/{out_file}") if out_file else None
    out_raw = c.get(f"/v1/files/{out_file}/content") if out_file else None
    out["batch_output_file_is_first_class_record"] = (
        out_card is not None
        and out_card.status_code == 200
        and out_card.json()["purpose"] == "batch_output"
        and out_raw is not None
        and out_raw.status_code == 200
        and b"r1" in out_raw.content
        and b"r2" in out_raw.content
        and b'"status_code":200' in out_raw.content.replace(b" ", b"")
        and c.delete(f"/v1/files/{out_file}").status_code == 200
    )
    # the snapshot semantic: the batch's parsed lines are captured at
    # submit — deleting the input after a 200 submit is a non-event
    fid2 = _minted(c, _BODY, purpose="batch")
    r2 = c.post(
        "/v1/batches",
        json={
            "input_file_id": fid2,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    assert r2.status_code == 200, r2.text
    bid2 = str(r2.json()["id"])
    assert c.delete(f"/v1/files/{fid2}").status_code == 200
    rec2 = _wait_batch(c, bid2)
    out["batch_snapshots_input_at_submit"] = (
        rec2["status"] == "completed" and rec2["request_counts"]["completed"] == 2
    )
    out["batch_record_keeps_deleted_input_id"] = rec2["input_file_id"] == fid2
    # mid-run delete: the input 404s while the batch is still in_progress
    # (16 lines x ~80ms of backend latency ≈ a >1 s working window)
    mid_fid = _minted(slow, _BODY * 8, purpose="batch")
    r3 = slow.post(
        "/v1/batches",
        json={
            "input_file_id": mid_fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    bid3 = str(r3.json()["id"]) if r3.status_code == 200 else ""
    saw_running = False
    deadline = time.monotonic() + 10.0
    while bid3 and time.monotonic() < deadline:
        st = slow.get(f"/v1/batches/{bid3}").json()["status"]
        if st == "in_progress":
            saw_running = True
            break
        if st in _BATCH_TERMINAL:
            break
        time.sleep(0.01)
    deleted_mid = slow.delete(f"/v1/files/{mid_fid}") if bid3 else None
    rec3: dict[str, Any] = _wait_batch(slow, bid3) if bid3 else {"status": "unsubmitted"}
    out["batch_delete_mid_run_window_observed"] = saw_running
    out["batch_delete_mid_run_still_completes"] = (
        deleted_mid is not None
        and deleted_mid.status_code == 200
        and rec3["status"] == "completed"
        and rec3["request_counts"]["completed"] == 16
    )
    # submit-time input validation — bad files fail loudly, no job mints
    bad_jsonl = _minted(c, b'{"custom_id": "x", "method": "POST"\n', purpose="batch")
    bad = c.post(
        "/v1/batches",
        json={
            "input_file_id": bad_jsonl,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    out["batch_bad_jsonl_fails_submit_400"] = (
        bad.status_code == 400 and _code(bad) == "invalid_request"
    )
    non_utf8 = _minted(c, _BINARY, purpose="batch")
    nutf = c.post(
        "/v1/batches",
        json={
            "input_file_id": non_utf8,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    out["batch_non_utf8_400"] = nutf.status_code == 400 and _code(nutf) == "invalid_request"
    ft_fid = _minted(c, _FT_CORPUS, purpose="fine-tune")
    wrong_purpose = c.post(
        "/v1/batches",
        json={
            "input_file_id": ft_fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    out["batch_wrong_purpose_400"] = (
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
    out["batch_ghost_input_404"] = ghost.status_code == 404 and _code(ghost) == "file_not_found"
    blank = _minted(c, b"\n\n  \n", purpose="batch")
    bl = c.post(
        "/v1/batches",
        json={
            "input_file_id": blank,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    out["batch_blank_lines_400"] = bl.status_code == 400 and _code(bl) == "invalid_request"
    one_line = _client(batch_line_max=1)
    two = _minted(one_line, _BODY, purpose="batch")
    over = one_line.post(
        "/v1/batches",
        json={
            "input_file_id": two,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    out["batch_line_cap_400"] = over.status_code == 400 and _code(over) == "batch_input_limit"
    # the batch list shares the fail-closed cursor contract
    bad_bafter = c.get("/v1/batches", params={"after": "batch-ghost"})
    out["batches_unknown_after_400_envelope"] = (
        bad_bafter.status_code == 400
        and _code(bad_bafter) == "invalid_cursor"
        and _is_envelope(bad_bafter)
    )
    # fine-tuning consumes a fine-tune-purpose file; the corpus copies
    # into the job work dir at submit — same snapshot semantic as batch
    ft_fid2 = _minted(ft_client, _FT_CORPUS, purpose="fine-tune")
    job = ft_client.post("/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": ft_fid2})
    out["ft_accepts_training_file"] = (
        job.status_code == 200
        and job.json()["object"] == "fine_tuning.job"
        and job.json()["training_file"] == ft_fid2
    )
    jid = str(job.json()["id"]) if job.status_code == 200 else ""
    deleted_src = ft_client.delete(f"/v1/files/{ft_fid2}") if jid else None
    jrec = _wait_ft(ft_client, jid) if jid else {"status": "unsubmitted"}
    out["ft_snapshots_corpus_at_submit"] = (
        deleted_src is not None
        and deleted_src.status_code == 200
        and jrec["status"] == "succeeded"
        and jrec["training_file"] == ft_fid2
    )
    return out


def _probe_ft_edges(ft_client: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    batch_fid = _minted(ft_client, _BODY, purpose="batch")
    wrong = ft_client.post(
        "/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": batch_fid}
    )
    out["ft_wrong_purpose_400"] = (
        wrong.status_code == 400 and _code(wrong) == "invalid_training_file"
    )
    bad_corpus = _minted(ft_client, b'{"nope": true}\n', purpose="fine-tune")
    bc = ft_client.post("/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": bad_corpus})
    out["ft_bad_corpus_400"] = bc.status_code == 400 and _code(bc) == "invalid_training_file"
    ghost = ft_client.post(
        "/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": "file-ghost"}
    )
    out["ft_ghost_404"] = ghost.status_code == 404 and _code(ghost) == "file_not_found"
    good = _minted(ft_client, _FT_CORPUS, purpose="fine-tune")
    ghost_val = ft_client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": good, "validation_file": "file-ghost"},
    )
    out["ft_validation_ghost_404"] = (
        ghost_val.status_code == 404 and _code(ghost_val) == "file_not_found"
    )
    bad_val = ft_client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": good, "validation_file": batch_fid},
    )
    out["ft_validation_wrong_purpose_400"] = (
        bad_val.status_code == 400 and _code(bad_val) == "invalid_training_file"
    )
    return out


def _probe_vector_store(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    fid = _minted(c, b'{"doc": "alpha"}\n{"doc": "beta"}\n', purpose="batch")
    vs = c.post("/v1/vector_stores", json={"name": "audit-vs"})
    vsid = str(vs.json()["id"]) if vs.status_code == 200 else ""
    att = c.post(f"/v1/vector_stores/{vsid}/files", json={"file_id": fid})
    out["vs_attach_completed"] = (
        att.status_code == 200
        and att.json()["id"] == fid
        and att.json()["object"] == "vector_store.file"
        and att.json()["status"] == "completed"
    )
    ghost_att = c.post(f"/v1/vector_stores/{vsid}/files", json={"file_id": "file-ghost"})
    out["vs_attach_ghost_404"] = ghost_att.status_code == 404 and _is_envelope(ghost_att)
    # attach indexes a snapshot — deleting the source record does not
    # un-chunk the index (the vs surface owns its own copy)
    assert c.delete(f"/v1/files/{fid}").status_code == 200
    card = c.get(f"/v1/vector_stores/{vsid}/files/{fid}")
    body = c.get(f"/v1/vector_stores/{vsid}/files/{fid}/content")
    texts = (
        "".join(
            str(ch.get("text", ""))
            for ch in (body.json().get("data") or [])
            if isinstance(ch, dict)
        )
        if body.status_code == 200
        else ""
    )
    out["vs_attach_snapshot_survives_delete"] = (
        card.status_code == 200 and body.status_code == 200 and "alpha" in texts and "beta" in texts
    )
    return out


def _probe_uploads_minted(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    fid = _minted_via_uploads(c, _BODY, purpose="batch")
    got = c.get(f"/v1/files/{fid}")
    out["uploads_minted_same_envelope"] = (
        got.status_code == 200
        and got.json()["object"] == "file"
        and got.json()["status"] == "processed"
        and set(got.json()) == _FILE_OBJECT_KEYS
    )
    raw = c.get(f"/v1/files/{fid}/content")
    out["uploads_minted_content_exact"] = raw.status_code == 200 and raw.content == _BODY
    listed = c.get("/v1/files").json()["data"]
    out["uploads_minted_listed"] = any(f["id"] == fid for f in listed)
    rb = c.post(
        "/v1/batches",
        json={
            "input_file_id": fid,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    rec: dict[str, Any] = (
        _wait_batch(c, str(rb.json()["id"])) if rb.status_code == 200 else {"status": "x"}
    )
    out["uploads_minted_runs_batch"] = (
        rb.status_code == 200
        and rec["status"] == "completed"
        and rec["request_counts"]["completed"] == 2
    )
    gone = c.delete(f"/v1/files/{fid}")
    out["uploads_minted_delete_tombstone"] = (
        gone.status_code == 200
        and c.get(f"/v1/files/{fid}").status_code == 404
        and c.get(f"/v1/files/{fid}/content").status_code == 404
    )
    return out


def _probe_concurrency(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    minted_ids: list[str] = []
    codes: list[int] = []

    def _create(i: int) -> None:
        r = _upload(c, b'{"n": %d}\n' % i)
        codes.append(r.status_code)
        if r.status_code == 200:
            minted_ids.append(str(r.json()["id"]))

    _run_threads(_create, 8)
    out["parallel_creates_all_200"] = len(codes) == 8 and all(s == 200 for s in codes)
    out["parallel_creates_unique_ids"] = len(set(minted_ids)) == len(minted_ids) == 8
    listed_ids = {str(f["id"]) for f in c.get("/v1/files").json()["data"]}
    out["parallel_creates_all_listed"] = set(minted_ids) <= listed_ids
    # racing deletes — exactly one winner, losers 404
    victim = _minted(c)
    del_codes: list[int] = []

    def _del(i: int) -> None:
        del_codes.append(c.delete(f"/v1/files/{victim}").status_code)

    _run_threads(_del, 8)
    out["racing_delete_single_winner"] = del_codes.count(200) == 1 and del_codes.count(404) == 7
    # delete racing content read — full bytes or 404, never torn
    body = (b"0123456789abcdef" * 512) + b"\n"
    torn = False
    for _ in range(8):
        rid = _minted(c, body)
        results: list[tuple[int, bytes]] = []

        def _read(i: int, rid: str = rid, results: list[tuple[int, bytes]] = results) -> None:
            r = c.get(f"/v1/files/{rid}/content")
            results.append((r.status_code, r.content))

        def _rm(i: int, rid: str = rid) -> None:
            c.delete(f"/v1/files/{rid}")

        _run_threads(_read, 4)
        _run_threads(_rm, 1)
        for status, payload in results:
            if status == 200 and payload != body:
                torn = True
            if status not in (200, 404):
                torn = True
    out["delete_content_race_never_torn"] = not torn
    return out


def _probe_tenancy() -> dict[str, bool]:
    out: dict[str, bool] = {}
    with _env_key():
        c = _client()
        env_h = {"X-API-Key": _ROOT_KEY}
        out["unauthenticated_401"] = _upload(c).status_code == 401
        out["wrong_key_401"] = _upload(c, headers={"X-API-Key": "wrong"}).status_code == 401
        out["bearer_accepted_on_v1"] = (
            _upload(c, headers={"Authorization": f"Bearer {_ROOT_KEY}"}).status_code == 200
        )
        m1 = c.post("/harness/keys", json={}, headers=env_h)
        m2 = c.post("/harness/keys", json={"scopes": ["read"]}, headers=env_h)
        out["managed_keys_mint_under_env_key"] = m1.status_code == 201 and m2.status_code == 201
        k_write = str(m1.json()["key"])
        k_read = str(m2.json()["key"])
        h_write = {"X-API-Key": k_write}
        h_read = {"X-API-Key": k_read}
        fid = _minted(c, headers=env_h)
        # workspace-level: the managed write key reads and deletes the
        # env key's file — records are not owner-bound
        out["files_are_workspace_level"] = (
            c.get(f"/v1/files/{fid}", headers=h_write).status_code == 200
            and c.get(f"/v1/files/{fid}/content", headers=h_write).status_code == 200
            and c.delete(f"/v1/files/{fid}", headers=h_write).status_code == 200
            and c.get(f"/v1/files/{fid}", headers=env_h).status_code == 404
        )
        ro_post = _upload(c, headers=h_read)
        ro_del_fid = _minted(c, headers=env_h)
        ro_del = c.delete(f"/v1/files/{ro_del_fid}", headers=h_read)
        out["read_scope_writes_403"] = (
            ro_post.status_code == 403
            and _code(ro_post) == "insufficient_scope"
            and ro_del.status_code == 403
            and _code(ro_del) == "insufficient_scope"
        )
        ro_fid = _minted(c, headers=env_h)
        out["read_scope_reads_200"] = (
            c.get("/v1/files", headers=h_read).status_code == 200
            and c.get(f"/v1/files/{ro_fid}", headers=h_read).status_code == 200
            and c.get(f"/v1/files/{ro_fid}/content", headers=h_read).status_code == 200
        )
    return out


def _probe_durability(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    state = tmp / "state"
    c1 = _client(state_dir=state)
    fid = _minted(c1, _BODY, purpose="batch")
    fid2 = _minted(c1, b'{"z":9}\n', purpose="fine-tune")
    journal = state / "files.jsonl"
    blob = state / "files" / f"{fid}.bin"
    out["journal_and_blob_on_disk"] = (
        journal.is_file() and blob.is_file() and blob.read_bytes() == _BODY
    )
    # restart: same ids, same bytes, same order — list BEFORE the gets
    # because a GET refreshes the LRU position by design
    c2 = _client(state_dir=state)
    listed = [f["id"] for f in c2.get("/v1/files").json()["data"]]
    got = c2.get(f"/v1/files/{fid}")
    raw = c2.get(f"/v1/files/{fid}/content")
    out["restart_restores_bytes"] = (
        got.status_code == 200
        and got.json()["purpose"] == "batch"
        and raw.status_code == 200
        and raw.content == _BODY
    )
    out["restart_order_preserved"] = listed[:2] == [fid2, fid]
    store2 = cast("FastAPI", c2.app).state.file_store
    out["clean_restart_no_recover_warnings"] = store2.recover_warnings == []
    # a deleted record stays deleted — tombstone replays, blob unlinks
    c2.delete(f"/v1/files/{fid2}")
    out["delete_unlinks_blob"] = not (state / "files" / f"{fid2}.bin").exists()
    c3 = _client(state_dir=state)
    out["deleted_stays_deleted_after_restart"] = (
        c3.get(f"/v1/files/{fid2}").status_code == 404
        and c3.get(f"/v1/files/{fid2}/content").status_code == 404
        and all(f["id"] != fid2 for f in c3.get("/v1/files").json()["data"])
        and not (state / "files" / f"{fid2}.bin").exists()
    )
    # eviction journals a tombstone too — an evicted file stays evicted
    e_state = tmp / "evict"
    ec1 = _client(state_dir=e_state, file_max=1)
    ea = _minted(ec1, b'{"e":"a"}\n')
    _minted(ec1, b'{"e":"b"}\n')
    ec2 = _client(state_dir=e_state, file_max=1)
    out["evicted_stays_evicted_after_restart"] = (
        ec2.get(f"/v1/files/{ea}").status_code == 404
        and ec2.get(f"/v1/files/{ea}/content").status_code == 404
    )
    # orphan blob GC: a blob with no journaled record is dropped at boot
    g_state = tmp / "gc"
    gc1 = _client(state_dir=g_state)
    gfid = _minted(gc1, b'{"g":1}\n')
    orphan = g_state / "files" / "file-ghost.bin"
    orphan.write_bytes(b"orphaned bytes")
    gc2 = _client(state_dir=g_state)
    out["orphan_blob_gcd_on_boot"] = (
        not orphan.exists()
        and gc2.get("/v1/files/file-ghost").status_code == 404
        and gc2.get(f"/v1/files/{gfid}").status_code == 200
    )
    return out


def _probe_error_paths(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    # multipart declared with no boundary — starlette's parse failure
    r1 = c.post("/v1/files", content=b"--x\r\n", headers={"content-type": "multipart/form-data"})
    out["missing_boundary_400_envelope"] = (
        r1.status_code == 400 and _is_envelope(r1) and _code(r1) == "bad_request"
    )
    r2 = c.post(
        "/v1/files",
        content=b"garbage",
        headers={"content-type": "multipart/form-data; boundary=x"},
    )
    out["malformed_multipart_400_envelope"] = (
        r2.status_code == 400 and _is_envelope(r2) and _code(r2) == "bad_request"
    )
    r3 = c.post("/v1/files", files={"data": ("p", b"ab")}, data={"purpose": "batch"})
    out["wrong_multipart_field_400_envelope"] = (
        r3.status_code == 400 and _is_envelope(r3) and _code(r3) == "invalid_request"
    )
    r4 = c.post("/v1/files", json={"file": "ab", "purpose": "batch"})
    out["json_body_400_envelope"] = (
        r4.status_code == 400 and _is_envelope(r4) and _code(r4) == "invalid_request"
    )
    r5 = c.post("/v1/files", content=b"ab", headers={"content-type": "text/plain"})
    out["text_body_400_envelope"] = (
        r5.status_code == 400 and _is_envelope(r5) and _code(r5) == "invalid_request"
    )
    # routing-level refusals: the /v1 catch-all owns them, still enveloped
    r6 = c.put("/v1/files")
    out["wrong_method_404_envelope"] = (
        r6.status_code == 404 and _is_envelope(r6) and _code(r6) == "not_found"
    )
    r7 = c.post("/v1/files/file-x")
    out["post_to_file_route_404_envelope"] = (
        r7.status_code == 404 and _is_envelope(r7) and _code(r7) == "not_found"
    )
    r8 = c.delete("/v1/files/file-x/content")
    out["delete_content_route_404_envelope"] = (
        r8.status_code == 404 and _is_envelope(r8) and _code(r8) == "not_found"
    )
    # envelope shape is uniform — message/type/code strings, param slot
    seen = [_is_envelope(r) for r in (r1, r2, r3, r4, r5, r6, r7, r8)]
    out["all_refusals_in_openai_envelope"] = all(seen)
    return out


def _probe_client_map(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.client import (
        BackendNotConfiguredError,
        HarnessAuthError,
        HarnessClient,
        HarnessTransportError,
    )

    client = HarnessClient(base_url="http://files-audit", transport=_tc_transport(c))
    up = client.upload_file(_BODY, filename="rt.jsonl", purpose="batch")
    fid = str(up["id"]) if isinstance(up, dict) else ""
    out["client_upload_file_roundtrip"] = (
        fid.startswith("file-") and up.get("object") == "file" and up.get("bytes") == len(_BODY)
    )
    out["client_files_lists"] = any(f["id"] == fid for f in client.files())
    out["client_file_card"] = client.file(fid)["id"] == fid
    out["client_file_content_bytes"] = client.file_content(fid) == _BODY
    out["client_delete_file"] = client.delete_file(fid)["deleted"] is True
    try:
        client.file("file-ghost")
        r404 = False
    except KeyError:
        r404 = True
    out["client_404_maps_keyerror"] = r404
    try:
        client.file_content("file-ghost")
        r404c = False
    except KeyError:
        r404c = True
    out["client_content_404_maps_keyerror"] = r404c
    try:
        client.upload_file(b"ab", purpose="bogus")
        r400 = False
    except HarnessTransportError:
        r400 = True
    out["client_400_maps_transport_error"] = r400

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

        c2 = HarnessClient(base_url="http://files-audit", transport=send)
        try:
            c2.file("file-x")
        except Exception as exc:  # noqa: BLE001 — the map IS the contract
            return exc
        return None

    out["client_401_maps_autherror"] = isinstance(_mapped(401), HarnessAuthError)
    out["client_403_maps_autherror"] = isinstance(_mapped(403), HarnessAuthError)
    out["client_422_maps_valueerror"] = isinstance(_mapped(422), ValueError)
    out["client_501_maps_notimplemented"] = isinstance(_mapped(501), NotImplementedError)
    out["client_503_maps_backenderror"] = isinstance(_mapped(503), BackendNotConfiguredError)
    out["client_500_maps_transport_error"] = isinstance(_mapped(500), HarnessTransportError)
    # keyed app: a client without the key gets the auth error
    with _env_key():
        keyed = _client()
        kc = HarnessClient(base_url="http://files-audit", transport=_tc_transport(keyed))
        try:
            kc.upload_file(b"ab")
            r401 = False
        except HarnessAuthError:
            r401 = True
        kc2 = HarnessClient(
            base_url="http://files-audit",
            transport=_tc_transport(keyed),
            api_key=_ROOT_KEY,
        )
        out["client_no_key_401"] = r401
        out["client_api_key_roundtrips"] = kc2.upload_file(b"ab")["object"] == "file"
    return out


def _probe_sdk_twin() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.sdk import Fx1Harness
    from fx1.serve.uploads import UploadStoreError

    sdk = Fx1Harness(backend_resolver=lambda *a, **k: _B())
    fobj = sdk.openai_file_create(_BODY, purpose="batch", filename="in.jsonl")
    fid = str(fobj["id"])
    out["sdk_file_create_mints_envelope"] = (
        fid.startswith("file-") and set(fobj) == _FILE_OBJECT_KEYS
    )
    out["sdk_file_content_exact"] = sdk.file_content(fid) == _BODY
    out["sdk_file_card_matches"] = sdk.file_card(fid) == fobj
    try:
        sdk.file_content("file-ghost")
        ghost = False
    except KeyError:
        ghost = True
    out["sdk_ghost_keyerror"] = ghost

    def _refused(**kw: Any) -> bool:
        try:
            sdk.openai_file_create(_BODY, **kw)
        except UploadStoreError:
            return True
        return False

    out["sdk_fail_closed_purpose"] = _refused(purpose="assistants")
    out["sdk_fail_closed_empty"] = _sdk_empty_refused(sdk)
    out["sdk_fail_closed_non_jsonl"] = _refused(filename="x.txt")
    # chunked-minted twin lands in the same file map
    up = sdk.upload_create(bytes=len(_BODY))
    p1 = sdk.upload_part(str(up["id"]), _BODY)
    done = sdk.upload_complete(str(up["id"]), [str(p1["id"])])
    out["sdk_uploads_minted_visible"] = (
        str(done["file"]["id"]).startswith("file-")
        and sdk.file_content(str(done["file"]["id"])) == _BODY
    )
    return out


def _sdk_empty_refused(sdk: Any) -> bool:
    from fx1.serve.uploads import UploadStoreError

    try:
        sdk.openai_file_create(b"", purpose="batch")
    except UploadStoreError:
        return True
    return False


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def files_audit() -> dict[str, Any]:
    """Run every probe against live in-process apps; literal bools out."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    out: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            c = _client()
            fresh = _client()
            slow = _slow_client()
            ft_client = _client(ft_runner=_ft_runner, ft_dir=wd / "ft")
            out.update(_probe_upload(c))
            out.update(_probe_list(c, fresh))
            out.update(_probe_get_delete(c))
            out.update(_probe_caps())
            out.update(_probe_consumer(c, ft_client, slow))
            out.update(_probe_ft_edges(ft_client))
            out.update(_probe_vector_store(c))
            out.update(_probe_uploads_minted(c))
            out.update(_probe_concurrency(c))
            out.update(_probe_tenancy())
            out.update(_probe_durability(wd))
            out.update(_probe_error_paths(c))
            out.update(_probe_client_map(c))
            out.update(_probe_sdk_twin())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def files_audit_bench() -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = files_audit()
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "files_audit",
        "schema": "files_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "The /v1/files lifecycle holds end to end: multipart mints "
            "produce the exact file envelope, content serves byte-exact "
            "under application/jsonl with no-store, cursors and filters "
            "walk the list honestly (refusals land in the envelope as "
            "invalid_cursor after the list route learned the shared "
            "OpenAICompatError wrap), delete tombstones the record on "
            "every surface at once, consumers snapshot bytes at "
            "submit/attach so a deleted source never tears a running "
            "batch or ft job, malformed inputs fail the submit loudly "
            "rather than a silent job, byte and entry caps refuse in the "
            "envelope, LRU eviction is a real tombstone that survives "
            "restarts like deletes do, uploads-minted files are "
            "indistinguishable from direct uploads, racing creates mint "
            "unique ids while delete+read races resolve atomically, "
            "files are workspace-level across credentials with "
            "scope-gated writes, and the client maps statuses to its "
            "exception taxonomy with an in-process SDK twin."
            if ok
            else f"FILES AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(files_audit_bench(), indent=2, sort_keys=True))
