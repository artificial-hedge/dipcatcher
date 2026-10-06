"""vs_audit — /v1/vector_stores lifecycle + retrieval probe battery.

``POST /v1/vector_stores`` mints a journaled lexical retrieval corpus
(``vs_*``); ``POST /{id}/files`` attaches a ``file-*`` record by
decoding, chunking, and indexing its bytes in-place; ``file_batches``
attach many in one call; ``POST /{id}/search`` runs the same hashed
bag-of-words cosine + per-store idf scorer ``file_search`` uses inside
``/v1/responses``. This battery pins the whole surface end to end:

- *Create* — empty body mints a store honestly (``name: null``); name/
  metadata echo; pydantic bounds (name ≤512, file_ids ≤64, dict-typed
  fields) answer 422; ``expires_after`` accepts only OpenAI's
  ``{"anchor": "last_active_at", "days": 1..365}`` shape and lands a
  real ``expires_at``; ``file_ids`` attach at create and a bogus,
  duplicate, or over-capacity member fails the WHOLE create — no
  partial store is left on the list.
- *Update* — ``name``/``metadata`` replace wholesale when present (a
  metadata update drops unset keys); an empty body is an honest no-op;
  ``expires_after`` re-anchors from ``last_active_at``.
- *Attach* — wire shape is the OpenAI ``vector_store.file`` object plus
  honest ``indexed_chunks``/``truncated``/``last_error`` extensions;
  the sync attach means only ``completed``/``failed`` ever persist;
  a file that decodes to no words lands ``failed`` with
  ``empty_file`` instead of silently indexing nothing; duplicates
  refuse ``409 file_already_attached``; ghost stores/files, deleted
  stores, and deleted file records all 404 with their own codes;
  attribute and chunking_strategy validation lands in the envelope.
- *Detach* — ``{id, object: vector_store.file.deleted, deleted: true}``
  exactly; the card and list drop the membership, the chunks leave the
  index (search stops hitting), the underlying ``file-*`` record
  survives, and usage_bytes shrinks honestly.
- *File batches* — members attach synchronously; per-file refusals
  count ``failed`` with ``last_error``, never abort; status is terminal
  at return; ``file_counts`` are honest; ``/files`` lists the frozen
  per-file verdicts which a later detach does not rewrite; cancel
  always answers ``409 file_batch_terminal`` because there is no
  mid-flight window to cancel — an honest refusal, not a fake.
- *Search* — deterministic lexical scoring: a unique term ranks the
  right file first; score desc order holds; ``query`` accepts a string
  or joined list; empty/over-long queries, out-of-range
  ``max_num_results``, ``rewrite_query``, and non-``auto`` rankers all
  refuse in the envelope; ``filters`` run the OpenAI comparison grammar
  against file attributes (``ne``/``nin`` treat absent as not-equal);
  ``score_threshold`` drops weak hits; empty/no-match stores answer
  honest empty pages; scores are byte-stable across identical calls.
- *Chunking* — auto is a 200-word window with 40-word overlap; static
  token bounds map ~0.75× to words; the echoed strategy is verbatim;
  overlap shares real words across adjacent chunks; past
  ``VS_MAX_CHUNKS`` the record flags ``truncated: true``.
- *Expiry* — ``expires_after`` sets ``expires_at``; expiry is lazy and
  computed per read (no sweeper): an expired store still GETs with
  ``status: "expired"`` while attach/batch/search refuse
  ``410 vector_store_expired``; ``update`` with a fresh policy revives
  it — an update without one leaves it expired.
- *Delete/list/caps* — delete tombstones the store on every subroute at
  once while member ``file-*`` records survive; lists honor
  cursor/limit/order exactly like the other surfaces (unknown cursors
  and bad order answer ``400 invalid_cursor``); the store-count cap
  LRU-evicts the oldest and a GET refreshes position.
- *Scope/auth* — ``FX1_API_KEY`` turns auth on: no credential 401s,
  wrong credential 401s, bearer is accepted on /v1; minted keys need
  ``write`` scope to mutate and ``read`` scope to read — they are
  disjoint scopes, not a ladder; every credential sees the same shared
  workspace (any key may read/attach/detach/delete any store).
- *Drain* — the latch refuses vs mutations ``503 draining`` (create,
  update, attach, batch create); GETs, search, deletes, detaches, and
  the already-terminal batch cancel stay open — the same
  submit-gated/lifecycle-open contract the job surface pins.
- *Idempotency* — ``Idempotency-Key`` on create/attach/batch-create
  dedupes in the bounded vector-store replay ledger with per-route and
  per-store namespaces: mutation and replay response share one journal
  record, so a keyed retry replays the recorded envelope
  byte-identical with ``X-Fx1-Idempotent-Replay: true``; a key reused
  with a different body 409s; refused requests never poison a key; a
  stored replay stays readable under drain.
- *Durability* — under ``--state-dir`` ``vector_stores.jsonl`` records
  identity events and replay re-reads file bytes to rebuild the index:
  stores, updates, attachments, batches, delete/detach tombstones,
  expiry policy, and idem records survive a restart; a file whose
  bytes vanished re-indexes as ``failed`` with
  ``file_missing_at_replay`` — never phantom-searchable; LRU order
  survives.
- *Concurrency* — parallel attaches to one store serialize on the
  membership lane and mint every file once; racing attaches of the
  same file have exactly one winner; creates mint distinct ids; a
  delete racing an attach resolves atomically (attach commits then the
  store drops, or attach 404s — never torn); the file cap gives
  exactly one winner at the boundary.
- *Metering* — minted keys bill ``uses`` on every authenticated vs call
  including refusals past the auth gate, and never ``tokens_used``
  (no model work); scope refusals bill nothing.
- *Envelope* — every refusal lands in
  ``{error: {message, type, param, code}}`` — routing misses, pydantic
  422s, store errors, and the drain latch all share the grammar.
- *Client map* — the HarnessClient ``vector_store_*`` surface
  round-trips every route; statuses map 401/403 → HarnessAuthError,
  404 → KeyError, 422 → ValueError, 501 → NotImplementedError,
  503 → BackendNotConfiguredError, else HarnessTransportError; the
  in-process SDK twin answers the same objects.

Defects found while probing, fixed on this lane:

- *Create-with-``file_ids`` leaked a store on member refusal* —
  ``VectorStoreStore.create`` journaled ``{vs}`` and published the meta
  before attaching the initial files; a bogus or duplicate member 404/
  409ed while the store stayed listed (empty, ``completed``) — the
  route's own "no partial store" contract violated. The create now
  prepares every member off-state and publishes the eviction, store,
  members, touch, and optional replay record through one journal line;
  a refusal therefore needs no compensating tombstone.
- *No drain gate on vector-store mutations* — create/update/attach/
  file-batch ran under the drain latch, contradicting the
  submit-gated contract every sibling mutation surface pins
  post-#2848. All four now answer ``503 draining`` after the
  idempotency replay check (stored replays and lifecycle removals —
  delete/detach/cancel — stay open like job cancel does).
- *No ``Idempotency-Key`` on the vector-store mints* — create, attach,
  and batch create ignored the header, so a retried request minted a
  second store or 409ed on its own attachment. All three now claim,
  replay and conflict under a bounded vector-store ledger with
  ``vs``/``vsfile:{id}``/``vsbatch:{id}`` namespaces. Each successful
  mutation and its replay response now share one durability boundary;
  no destructive cross-journal rollback is required.
- *``order`` was pydantic-locked to ``Literal``* — the three vs list
  routes declared ``Literal["asc", "desc"]`` so a bad order answered a
  bare ``422 validation`` while the shared ``_page`` contract (and the
  files surface) answers ``400 invalid_cursor``. ``order`` is now a
  plain str so the shared pager's fail-closed refusal applies.

Probes are literal bools: ``True`` pins a contract that holds;
``False`` pins a measured divergence — the sealed receipt names every
defect by probe name so the finding survives byte-for-byte.

Sealed ``vs_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

__all__ = ["vs_audit", "vs_audit_bench"]

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
_ROOT_KEY = "vs-audit-root"

_VS_KEYS = {
    "id",
    "object",
    "created_at",
    "name",
    "status",
    "usage_bytes",
    "file_counts",
    "last_active_at",
    "expires_after",
    "expires_at",
    "metadata",
}
_VS_FILE_KEYS = {
    "id",
    "object",
    "vector_store_id",
    "created_at",
    "status",
    "usage_bytes",
    "last_error",
    "attributes",
    "chunking_strategy",
    "indexed_chunks",
    "truncated",
}
_VS_BATCH_KEYS = {"id", "object", "created_at", "vector_store_id", "status", "file_counts"}
_VS_COUNT_KEYS = {"in_progress", "completed", "cancelled", "failed", "total"}
_SEARCH_PAGE_KEYS = {"object", "search_query", "data", "has_more", "next_page"}
_SEARCH_HIT_KEYS = {"file_id", "filename", "score", "attributes", "content"}
_PAGE_KEYS = {"object", "data", "first_id", "last_id", "has_more"}


def _runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, "ok", ""


class _B:
    """Backend stub — ``complete`` returns a fixed token string."""

    def complete(self, *a: Any, **k: Any) -> Any:
        return "ok"


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


def _h(auth: str | None) -> dict[str, str]:
    return {"X-API-Key": auth} if auth else {}


def _bearer(auth: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth}"}


def _mint(c: TestClient, **policy: Any) -> tuple[str, str]:
    """Mint a managed key under the env root → (raw material, key_id)."""
    r = c.post("/harness/keys", json=policy, headers=_h(_ROOT_KEY))
    assert r.status_code in (200, 201), r.text
    return str(r.json()["key"]), str(r.json()["id"])


def _card(c: TestClient, key_id: str) -> dict[str, Any]:
    r = c.get(f"/harness/keys/{key_id}/usage", headers=_h(_ROOT_KEY))
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


def _uses(c: TestClient, key_id: str) -> int:
    return int(_card(c, key_id)["uses"])


def _tokens(c: TestClient, key_id: str) -> int:
    return int(_card(c, key_id)["tokens_used"])


def _upload_text(
    c: TestClient,
    text: str,
    *,
    name: str = "doc.jsonl",
    headers: dict[str, str] | None = None,
) -> str:
    """``POST /v1/files`` — mint a ``file-`` record carrying ``text``."""
    r = c.post(
        "/v1/files",
        files={"file": (name, text.encode())},
        data={"purpose": "batch"},
        headers=headers or {},
    )
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _mk_vs(c: TestClient, headers: dict[str, str] | None = None, **body: Any) -> str:
    r = c.post("/v1/vector_stores", json=body, headers=headers or {})
    assert r.status_code == 200, r.text
    return str(r.json()["id"])


def _attach(
    c: TestClient,
    vid: str,
    file_id: str,
    headers: dict[str, str] | None = None,
    **body: Any,
) -> dict[str, Any]:
    r = c.post(
        f"/v1/vector_stores/{vid}/files",
        json={"file_id": file_id, **body},
        headers=headers or {},
    )
    assert r.status_code == 200, r.text
    out: dict[str, Any] = r.json()
    return out


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


def _run_calls(calls: list[Any]) -> None:
    """Run heterogeneous calls concurrently and propagate worker failures."""
    barrier = threading.Barrier(len(calls))
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def _w(call: Any) -> None:
        try:
            barrier.wait()
            call()
        except BaseException as exc:  # noqa: BLE001 — worker errors must reach the audit
            with errors_lock:
                errors.append(exc)

    ths = [threading.Thread(target=_w, args=(call,)) for call in calls]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout=30.0)
    if any(t.is_alive() for t in ths):
        raise RuntimeError("vs audit worker did not terminate")
    if errors:
        raise RuntimeError(f"vs audit worker failed: {errors[0]}") from errors[0]


def _run_threads(fn: Any, n: int = 8) -> None:
    """Run ``fn(0..n-1)`` released together behind a barrier."""
    _run_calls([lambda i=i: fn(i) for i in range(n)])


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


# ---------------------------------------------------------------------------
# Probe batteries
# ---------------------------------------------------------------------------


def _probe_create(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    r = c.post("/v1/vector_stores", json={})
    vs = r.json()
    vid = str(vs["id"])
    out["create_empty_body_mints_store"] = (
        r.status_code == 200
        and vs["object"] == "vector_store"
        and vid.startswith("vs_")
        and vs["name"] is None
        and vs["status"] == "completed"
    )
    out["create_envelope_keys_exact"] = set(vs) == _VS_KEYS
    out["create_file_counts_zero_shape"] = set(vs["file_counts"]) == _VS_COUNT_KEYS and all(
        v == 0 for v in vs["file_counts"].values()
    )
    r2 = c.post("/v1/vector_stores", json={"name": "docs", "metadata": {"team": "q"}})
    out["create_name_metadata_echo"] = (
        r2.status_code == 200
        and r2.json()["name"] == "docs"
        and r2.json()["metadata"] == {"team": "q"}
    )
    out["create_distinct_ids"] = vid != r2.json()["id"]
    out["create_name_too_long_422"] = (
        c.post("/v1/vector_stores", json={"name": "x" * 513}).status_code == 422
    )
    too_many_meta = c.post(
        "/v1/vector_stores", json={"metadata": {f"k{i}": "v" for i in range(17)}}
    )
    out["create_metadata_cap_400"] = (
        too_many_meta.status_code == 400 and _code(too_many_meta) == "invalid_request"
    )
    long_val = c.post("/v1/vector_stores", json={"metadata": {"k": "v" * 513}})
    out["create_metadata_value_cap_400"] = (
        long_val.status_code == 400 and _code(long_val) == "invalid_request"
    )
    out["create_metadata_non_object_422"] = (
        c.post("/v1/vector_stores", json={"metadata": "flat"}).status_code == 422
    )
    ok_exp = c.post(
        "/v1/vector_stores",
        json={"expires_after": {"anchor": "last_active_at", "days": 7}},
    )
    exp_vs = ok_exp.json()
    out["create_expires_after_lands_expires_at"] = (
        ok_exp.status_code == 200
        and exp_vs["expires_after"] == {"anchor": "last_active_at", "days": 7}
        and exp_vs["expires_at"] - exp_vs["created_at"] == 7 * 86400
    )
    for label, ea in (
        ("bad_anchor", {"anchor": "created_at", "days": 1}),
        ("days_zero", {"anchor": "last_active_at", "days": 0}),
        ("days_over_cap", {"anchor": "last_active_at", "days": 366}),
        ("extra_key", {"anchor": "last_active_at", "days": 1, "x": 1}),
        ("days_nonint", {"anchor": "last_active_at", "days": 1.5}),
    ):
        resp = c.post("/v1/vector_stores", json={"expires_after": ea})
        out[f"create_expires_after_{label}_400"] = (
            resp.status_code == 400 and _code(resp) == "invalid_expires_after"
        )
    out["create_expires_after_non_object_422"] = (
        c.post("/v1/vector_stores", json={"expires_after": "soon"}).status_code == 422
    )
    # ``file_ids`` attach existing file-* records at create — all-or-nothing
    f1 = _upload_text(c, "alpha bravo charlie delta echo")
    r3 = c.post("/v1/vector_stores", json={"file_ids": [f1]})
    out["create_file_ids_attach_at_create"] = (
        r3.status_code == 200
        and r3.json()["file_counts"]["completed"] == 1
        and r3.json()["file_counts"]["total"] == 1
        and r3.json()["usage_bytes"] > 0
    )
    out["create_file_ids_over_cap_422"] = (
        c.post("/v1/vector_stores", json={"file_ids": ["file-x"] * 65}).status_code == 422
    )
    out["create_file_ids_nonstring_422"] = (
        c.post("/v1/vector_stores", json={"file_ids": [1]}).status_code == 422
    )

    def _n_stores() -> set[str]:
        page = c.get("/v1/vector_stores", params={"limit": 100}).json()
        return {str(v["id"]) for v in page["data"]}

    before = _n_stores()
    bogus = c.post("/v1/vector_stores", json={"file_ids": ["file-bogus"]})
    out["create_file_ids_bogus_404"] = bogus.status_code == 404 and _code(bogus) == "file_not_found"
    out["create_file_ids_bogus_no_partial_store"] = _n_stores() == before
    f2 = _upload_text(c, "zeta eta theta")
    mixed = c.post("/v1/vector_stores", json={"file_ids": [f2, "file-bogus"]})
    out["create_file_ids_mixed_404_no_partial_store"] = (
        mixed.status_code == 404 and _n_stores() == before
    )
    f3 = _upload_text(c, "iota kappa lambda")
    dup = c.post("/v1/vector_stores", json={"file_ids": [f3, f3]})
    out["create_file_ids_dup_refused"] = dup.status_code == 409 and _code(dup) in (
        "file_already_attached",
        "invalid_request",
    )
    out["create_file_ids_dup_no_partial_store"] = _n_stores() == before
    # a rolled-back create's members stay free — the same file attaches fresh
    reattach = c.post("/v1/vector_stores", json={"file_ids": [f2]})
    out["create_rollback_frees_member_files"] = (
        reattach.status_code == 200 and reattach.json()["file_counts"]["completed"] == 1
    )
    out["create_extra_fields_tolerated"] = (
        c.post("/v1/vector_stores", json={"name": "e", "unseen": 1}).status_code == 200
    )
    return out


def _probe_update(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c, name="v1", metadata={"a": "1"})
    before = c.get(f"/v1/vector_stores/{vid}").json()
    r = c.post(f"/v1/vector_stores/{vid}", json={"name": "v2", "metadata": {"b": "2"}})
    out["update_replaces_name_metadata"] = (
        r.status_code == 200
        and r.json()["name"] == "v2"
        and r.json()["metadata"] == {"b": "2"}
        and r.json()["id"] == vid
    )
    out["update_bumps_last_active"] = r.json()["last_active_at"] >= before["last_active_at"]
    r2 = c.post(f"/v1/vector_stores/{vid}", json={"metadata": {"c": "3"}})
    out["update_metadata_replaces_wholesale"] = (
        r2.json()["metadata"] == {"c": "3"} and r2.json()["name"] == "v2"
    )
    r3 = c.post(f"/v1/vector_stores/{vid}", json={})
    out["update_empty_body_noop"] = (
        r3.status_code == 200 and r3.json()["name"] == "v2" and r3.json()["metadata"] == {"c": "3"}
    )
    ghost = c.post("/v1/vector_stores/vs_ghost", json={"name": "x"})
    out["update_ghost_404"] = ghost.status_code == 404 and _code(ghost) == "vector_store_not_found"
    bad_exp = c.post(f"/v1/vector_stores/{vid}", json={"expires_after": {"anchor": "x", "days": 1}})
    out["update_bad_expires_400"] = (
        bad_exp.status_code == 400 and _code(bad_exp) == "invalid_expires_after"
    )
    bad_meta = c.post(
        f"/v1/vector_stores/{vid}", json={"metadata": {f"k{i}": "v" for i in range(17)}}
    )
    out["update_metadata_cap_400"] = bad_meta.status_code == 400
    return out


def _probe_attach(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c, name="attach")
    f1 = _upload_text(c, "alpha quasar nebula market photon")
    r = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f1})
    rec = r.json()
    out["attach_wire_shape"] = (
        r.status_code == 200
        and rec["id"] == f1
        and rec["object"] == "vector_store.file"
        and rec["vector_store_id"] == vid
        and rec["status"] == "completed"
        and rec["last_error"] is None
        and rec["chunking_strategy"] == {"type": "auto"}
        and rec["indexed_chunks"] >= 1
        and rec["truncated"] is False
        and rec["usage_bytes"] > 0
    )
    out["attach_envelope_keys_exact"] = set(rec) == _VS_FILE_KEYS
    got = c.get(f"/v1/vector_stores/{vid}/files/{f1}")
    out["attach_get_card_matches"] = got.status_code == 200 and got.json() == rec
    vs_now = c.get(f"/v1/vector_stores/{vid}").json()
    out["attach_bumps_file_counts_and_usage"] = (
        vs_now["file_counts"]["completed"] == 1
        and vs_now["usage_bytes"] == rec["usage_bytes"]
        and vs_now["file_counts"]["total"] == 1
    )
    f2 = _upload_text(c, "beta ledger margin arbitrage market")
    r2 = c.post(
        f"/v1/vector_stores/{vid}/files",
        json={
            "file_id": f2,
            "attributes": {"cat": "finance", "n": 7, "b": True},
            "chunking_strategy": {
                "type": "static",
                "static": {"max_chunk_size_tokens": 400, "chunk_overlap_tokens": 0},
            },
        },
    )
    rec2 = r2.json()
    out["attach_attributes_echo_verbatim"] = r2.status_code == 200 and rec2["attributes"] == {
        "cat": "finance",
        "n": 7,
        "b": True,
    }
    out["attach_static_strategy_echo_verbatim"] = rec2["chunking_strategy"] == {
        "type": "static",
        "static": {"max_chunk_size_tokens": 400, "chunk_overlap_tokens": 0},
    }
    cap_attr = c.post(
        f"/v1/vector_stores/{vid}/files",
        json={"file_id": f1, "attributes": {f"k{i}": "v" for i in range(17)}},
    )
    out["attach_attributes_cap_400"] = (
        cap_attr.status_code == 400 and _code(cap_attr) == "invalid_request"
    )
    nested_attr = c.post(
        f"/v1/vector_stores/{vid}/files",
        json={"file_id": f1, "attributes": {"k": {"nested": 1}}},
    )
    out["attach_attributes_nonscalar_400"] = (
        nested_attr.status_code == 400 and _code(nested_attr) == "invalid_request"
    )
    out["attach_attributes_list_422"] = (
        c.post(
            f"/v1/vector_stores/{vid}/files",
            json={"file_id": f1, "attributes": ["k"]},
        ).status_code
        == 422
    )
    out["attach_file_id_bounds_422"] = (
        c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": "x" * 129}).status_code == 422
        and c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": ""}).status_code == 422
    )
    for label, strat in (
        ("bad_type", {"type": "bogus"}),
        ("size_low", {"type": "static", "static": {"max_chunk_size_tokens": 99}}),
        ("size_high", {"type": "static", "static": {"max_chunk_size_tokens": 4097}}),
        (
            "overlap_over_half",
            {
                "type": "static",
                "static": {"max_chunk_size_tokens": 200, "chunk_overlap_tokens": 101},
            },
        ),
        (
            "overlap_negative",
            {
                "type": "static",
                "static": {"max_chunk_size_tokens": 200, "chunk_overlap_tokens": -1},
            },
        ),
        ("missing_static", {"type": "static"}),
        ("non_object_static", {"type": "static", "static": 5}),
    ):
        resp = c.post(
            f"/v1/vector_stores/{vid}/files",
            json={"file_id": f1, "chunking_strategy": strat},
        )
        out[f"attach_chunking_{label}_400"] = (
            resp.status_code == 400 and _code(resp) == "invalid_request"
        )
    dup = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f1})
    out["attach_duplicate_409"] = dup.status_code == 409 and _code(dup) == "file_already_attached"
    ghost_store = c.post("/v1/vector_stores/vs_ghost/files", json={"file_id": f1})
    out["attach_ghost_store_404"] = (
        ghost_store.status_code == 404 and _code(ghost_store) == "vector_store_not_found"
    )
    ghost_file = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": "file-ghost"})
    out["attach_ghost_file_404"] = (
        ghost_file.status_code == 404 and _code(ghost_file) == "file_not_found"
    )
    f_del = _upload_text(c, "doomed content")
    assert c.delete(f"/v1/files/{f_del}").status_code == 200
    del_attach = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f_del})
    out["attach_deleted_file_record_404"] = (
        del_attach.status_code == 404 and _code(del_attach) == "file_not_found"
    )
    vid_del = _mk_vs(c)
    assert c.delete(f"/v1/vector_stores/{vid_del}").status_code == 200
    ds_attach = c.post(f"/v1/vector_stores/{vid_del}/files", json={"file_id": f2})
    out["attach_deleted_store_404"] = (
        ds_attach.status_code == 404 and _code(ds_attach) == "vector_store_not_found"
    )
    # a file whose bytes decode to no indexable words lands failed honestly
    f_ws = _upload_text(c, "   \n\t \r\n  \u3000 ")
    r_ws = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f_ws})
    ws_rec = r_ws.json()
    out["attach_empty_text_failed_honestly"] = (
        r_ws.status_code == 200
        and ws_rec["status"] == "failed"
        and ws_rec["last_error"]["code"] == "empty_file"
        and ws_rec["indexed_chunks"] == 0
        and ws_rec["truncated"] is False
    )
    vs2 = c.get(f"/v1/vector_stores/{vid}").json()
    out["attach_failed_counts_honestly"] = (
        vs2["file_counts"]["failed"] == 1 and vs2["file_counts"]["completed"] == 2
    )
    # per-store file cap: 32 attached, the 33rd refuses — file text is tiny
    cap_vs = _mk_vs(c)
    ok = 0
    overflow: Any = None
    for _ in range(40):
        fid = _upload_text(c, f"filler {ok} word")
        r_cap = c.post(f"/v1/vector_stores/{cap_vs}/files", json={"file_id": fid})
        if r_cap.status_code == 200:
            ok += 1
        else:
            overflow = r_cap
            break
    out["attach_store_full_409"] = (
        ok == 32
        and overflow is not None
        and overflow.status_code == 409
        and _code(overflow) == "vector_store_full"
    )
    # decoded text over the 4 MiB indexable cap refuses 413; the HTTP
    # request-body cap (1 MiB) can't carry a payload that big, so pin the
    # store layer the route delegates to
    from fx1.serve.vectorstores import VS_MAX_TEXT_BYTES, VectorStoreError, VectorStoreStore

    store = VectorStoreStore(
        4,
        file_reader=lambda _fid: (b"x " * ((VS_MAX_TEXT_BYTES // 2) + 2), "fat.bin"),
    )
    big_meta = store.create(name="big")
    try:
        store.attach(str(big_meta["id"]), "file-fat")
        oversized_refused = False
    except VectorStoreError as exc:
        oversized_refused = exc.status == 413 and exc.code == "file_too_large"
    out["attach_oversized_413"] = oversized_refused
    return out


def _probe_detach(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c)
    f1 = _upload_text(c, "detachable zebra yarrow xylem")
    _attach(c, vid, f1)
    f2 = _upload_text(c, "second member walnut")
    _attach(c, vid, f2)
    usage_before = c.get(f"/v1/vector_stores/{vid}").json()["usage_bytes"]
    r = c.delete(f"/v1/vector_stores/{vid}/files/{f1}")
    out["detach_wire_shape"] = r.status_code == 200 and r.json() == {
        "id": f1,
        "object": "vector_store.file.deleted",
        "deleted": True,
    }
    out["detach_removes_card_404"] = c.get(f"/v1/vector_stores/{vid}/files/{f1}").status_code == 404
    page = c.get(f"/v1/vector_stores/{vid}/files").json()
    out["detach_removes_from_list"] = [f["id"] for f in page["data"]] == [f2]
    again = c.delete(f"/v1/vector_stores/{vid}/files/{f1}")
    out["detach_twice_404"] = again.status_code == 404 and _code(again) == "file_not_found"
    unattached = c.delete(f"/v1/vector_stores/{vid}/files/file-never")
    out["detach_unattached_404"] = (
        unattached.status_code == 404 and _code(unattached) == "file_not_found"
    )
    ghost = c.delete("/v1/vector_stores/vs_ghost/files/file-x")
    out["detach_ghost_store_404"] = (
        ghost.status_code == 404 and _code(ghost) == "vector_store_not_found"
    )
    out["detach_file_record_survives"] = c.get(f"/v1/files/{f1}").status_code == 200
    out["detach_usage_bytes_shrinks"] = (
        c.get(f"/v1/vector_stores/{vid}").json()["usage_bytes"] < usage_before
    )
    # chunks leave the index — a previously-hitting term stops hitting
    vid2 = _mk_vs(c)
    fu = _upload_text(c, "uniqueterm zebra striped")
    _attach(c, vid2, fu)
    hit_before = c.post(f"/v1/vector_stores/{vid2}/search", json={"query": "uniqueterm"})
    c.delete(f"/v1/vector_stores/{vid2}/files/{fu}")
    hit_after = c.post(f"/v1/vector_stores/{vid2}/search", json={"query": "uniqueterm"})
    out["detach_evicts_chunks_from_search"] = (
        len(hit_before.json()["data"]) == 1 and hit_after.json()["data"] == []
    )
    return out


def _probe_files_list(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c)
    ids = [_upload_text(c, f"member {i} corpus {i}") for i in range(3)]
    for fid in ids:
        _attach(c, vid, fid)
    page = c.get(f"/v1/vector_stores/{vid}/files").json()
    out["files_list_asc_default"] = [f["id"] for f in page["data"]] == ids
    out["files_list_page_shape"] = (
        page["object"] == "list" and set(page) == _PAGE_KEYS and page["has_more"] is False
    )
    desc = c.get(f"/v1/vector_stores/{vid}/files", params={"order": "desc"}).json()
    out["files_list_desc_flips"] = [f["id"] for f in desc["data"]] == ids[::-1]
    p1 = c.get(f"/v1/vector_stores/{vid}/files", params={"limit": 1}).json()
    out["files_list_limit_one_pages"] = (
        len(p1["data"]) == 1 and p1["data"][0]["id"] == ids[0] and p1["has_more"] is True
    )
    p2 = c.get(f"/v1/vector_stores/{vid}/files", params={"limit": 1, "after": ids[0]}).json()
    out["files_list_after_walks"] = p2["data"][0]["id"] == ids[1]
    pb = c.get(f"/v1/vector_stores/{vid}/files", params={"before": ids[2]}).json()
    out["files_list_before_walks"] = [f["id"] for f in pb["data"]] == ids[:2]
    bad_cur = c.get(f"/v1/vector_stores/{vid}/files", params={"after": "file-nonexistent"})
    out["files_list_unknown_cursor_400"] = (
        bad_cur.status_code == 400 and _code(bad_cur) == "invalid_cursor"
    )
    bad_order = c.get(f"/v1/vector_stores/{vid}/files", params={"order": "bogus"})
    out["files_list_bad_order_400"] = (
        bad_order.status_code == 400 and _code(bad_order) == "invalid_cursor"
    )
    filt = c.get(f"/v1/vector_stores/{vid}/files", params={"filter": "completed"}).json()
    out["files_list_filter_completed"] = [f["id"] for f in filt["data"]] == ids
    filt_f = c.get(f"/v1/vector_stores/{vid}/files", params={"filter": "failed"}).json()
    out["files_list_filter_failed_empty"] = filt_f["data"] == []
    bad_filt = c.get(f"/v1/vector_stores/{vid}/files", params={"filter": "bogus"})
    out["files_list_bogus_filter_400"] = (
        bad_filt.status_code == 400 and _code(bad_filt) == "invalid_filters"
    )
    ghost = c.get("/v1/vector_stores/vs_ghost/files")
    out["files_list_ghost_store_404"] = (
        ghost.status_code == 404 and _code(ghost) == "vector_store_not_found"
    )
    empty_vs = _mk_vs(c)
    empty = c.get(f"/v1/vector_stores/{empty_vs}/files").json()
    out["files_list_empty_shape"] = (
        empty["data"] == []
        and empty["first_id"] is None
        and empty["last_id"] is None
        and empty["has_more"] is False
    )
    return out


def _probe_file_batches(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c)
    f1 = _upload_text(c, "batch member one quasar")
    f2 = _upload_text(c, "batch member two ledger")
    pre = _upload_text(c, "pre attached member")
    _attach(c, vid, pre)
    r = c.post(
        f"/v1/vector_stores/{vid}/file_batches",
        json={"file_ids": [f1, f2, "file-ghost", pre]},
    )
    batch = r.json()
    bid = str(batch["id"])
    out["batch_wire_shape"] = (
        r.status_code == 200
        and bid.startswith("vsfb_")
        and batch["object"] == "vector_store.files_batch"
        and batch["vector_store_id"] == vid
        and set(batch) == _VS_BATCH_KEYS
    )
    out["batch_file_counts_honest"] = batch["file_counts"] == {
        "in_progress": 0,
        "completed": 2,
        "cancelled": 0,
        "failed": 2,
        "total": 4,
    }
    out["batch_status_terminal_at_return"] = batch["status"] == "completed"
    got = c.get(f"/v1/vector_stores/{vid}/file_batches/{bid}")
    out["batch_get_roundtrip"] = got.status_code == 200 and got.json() == batch
    all_bad = c.post(
        f"/v1/vector_stores/{vid}/file_batches",
        json={"file_ids": ["file-x", "file-y"]},
    )
    out["batch_all_failed_status_failed"] = (
        all_bad.status_code == 200
        and all_bad.json()["status"] == "failed"
        and all_bad.json()["file_counts"]["completed"] == 0
        and all_bad.json()["file_counts"]["failed"] == 2
    )
    f3 = _upload_text(c, "dup member thrice")
    dup_b = c.post(f"/v1/vector_stores/{vid}/file_batches", json={"file_ids": [f3, f3]})
    out["batch_dup_member_one_wins"] = (
        dup_b.status_code == 200
        and dup_b.json()["file_counts"]["completed"] == 1
        and dup_b.json()["file_counts"]["failed"] == 1
    )
    cancel = c.post(f"/v1/vector_stores/{vid}/file_batches/{bid}/cancel")
    out["batch_cancel_409_terminal"] = (
        cancel.status_code == 409 and _code(cancel) == "file_batch_terminal"
    )
    ghost_batch = c.post(f"/v1/vector_stores/{vid}/file_batches/vsfb_ghost/cancel")
    out["batch_cancel_ghost_404"] = (
        ghost_batch.status_code == 404 and _code(ghost_batch) == "file_batch_not_found"
    )
    ghost_get = c.get(f"/v1/vector_stores/{vid}/file_batches/vsfb_ghost")
    out["batch_get_ghost_404"] = (
        ghost_get.status_code == 404 and _code(ghost_get) == "file_batch_not_found"
    )
    ghost_vs = c.post("/v1/vector_stores/vs_ghost/file_batches", json={"file_ids": [f1]})
    out["batch_ghost_store_404"] = (
        ghost_vs.status_code == 404 and _code(ghost_vs) == "vector_store_not_found"
    )
    rows = c.get(f"/v1/vector_stores/{vid}/file_batches/{bid}/files").json()
    out["batch_files_frozen_verdicts"] = (
        [r_["id"] for r_ in rows["data"]] == [f1, f2, "file-ghost", pre]
        and [r_["status"] for r_ in rows["data"]] == ["completed", "completed", "failed", "failed"]
        and rows["data"][2]["last_error"]["code"] == "file_not_found"
        and rows["data"][3]["last_error"]["code"] == "file_already_attached"
    )
    # frozen at processing time: detaching a member does not rewrite history
    c.delete(f"/v1/vector_stores/{vid}/files/{f1}")
    rows2 = c.get(f"/v1/vector_stores/{vid}/file_batches/{bid}/files").json()
    out["batch_verdicts_frozen_after_detach"] = rows2["data"][0]["status"] == "completed"
    p1 = c.get(f"/v1/vector_stores/{vid}/file_batches/{bid}/files", params={"limit": 2}).json()
    p2 = c.get(
        f"/v1/vector_stores/{vid}/file_batches/{bid}/files",
        params={"limit": 2, "after": f2},
    ).json()
    out["batch_files_page_walk"] = (
        len(p1["data"]) == 2
        and p1["has_more"] is True
        and [r_["id"] for r_ in p2["data"]] == ["file-ghost", pre]
    )
    bad_cur = c.get(
        f"/v1/vector_stores/{vid}/file_batches/{bid}/files",
        params={"after": "file-nonexistent"},
    )
    out["batch_files_bad_cursor_400"] = (
        bad_cur.status_code == 400 and _code(bad_cur) == "invalid_cursor"
    )
    bad_order = c.get(
        f"/v1/vector_stores/{vid}/file_batches/{bid}/files", params={"order": "bogus"}
    )
    out["batch_files_bad_order_400"] = (
        bad_order.status_code == 400 and _code(bad_order) == "invalid_cursor"
    )
    bad_filt = c.get(
        f"/v1/vector_stores/{vid}/file_batches/{bid}/files", params={"filter": "bogus"}
    )
    out["batch_files_bogus_filter_400"] = (
        bad_filt.status_code == 400 and _code(bad_filt) == "invalid_filters"
    )
    filt_f = c.get(
        f"/v1/vector_stores/{vid}/file_batches/{bid}/files", params={"filter": "failed"}
    ).json()
    out["batch_files_filter_failed"] = [r_["id"] for r_ in filt_f["data"]] == [
        "file-ghost",
        pre,
    ]
    out["batch_empty_ids_422"] = (
        c.post(f"/v1/vector_stores/{vid}/file_batches", json={"file_ids": []}).status_code == 422
    )
    out["batch_over_cap_422"] = (
        c.post(
            f"/v1/vector_stores/{vid}/file_batches",
            json={"file_ids": ["file-x"] * 501},
        ).status_code
        == 422
    )
    out["batch_member_too_long_422"] = (
        c.post(
            f"/v1/vector_stores/{vid}/file_batches",
            json={"file_ids": ["x" * 129]},
        ).status_code
        == 422
    )
    # batch on an empty store works — and carries member-level attrs/strategy
    empty_vs = _mk_vs(c)
    f4 = _upload_text(c, "empty store batch member")
    eb = c.post(
        f"/v1/vector_stores/{empty_vs}/file_batches",
        json={
            "file_ids": [f4],
            "attributes": {"src": "batch"},
            "chunking_strategy": {"type": "auto"},
        },
    )
    erow = c.get(f"/v1/vector_stores/{empty_vs}/file_batches/{eb.json()['id']}/files").json()[
        "data"
    ][0]
    out["batch_on_empty_store_works"] = (
        eb.status_code == 200
        and eb.json()["file_counts"]
        == {
            "in_progress": 0,
            "completed": 1,
            "cancelled": 0,
            "failed": 0,
            "total": 1,
        }
        and erow["attributes"] == {"src": "batch"}
    )
    return out


def _probe_search(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c)
    f_alpha = _upload_text(c, "alpha quasar nebula photon market")
    f_beta = _upload_text(c, "beta ledger margin arbitrage market")
    _attach(c, vid, f_alpha, attributes={"cat": "astronomy"})
    _attach(c, vid, f_beta, attributes={"cat": "finance"})
    s = c.post(f"/v1/vector_stores/{vid}/search", json={"query": "quasar"})
    page = s.json()
    out["search_page_envelope"] = (
        s.status_code == 200
        and set(page) == _SEARCH_PAGE_KEYS
        and page["object"] == "vector_store.search_results.page"
        and page["search_query"] == "quasar"
        and page["has_more"] is False
        and page["next_page"] is None
    )
    hit = page["data"][0]
    out["search_hit_shape"] = (
        set(hit) == _SEARCH_HIT_KEYS
        and hit["file_id"] == f_alpha
        and hit["filename"].endswith(".jsonl")
        and float(hit["score"]) > 0.0
        and hit["attributes"] == {"cat": "astronomy"}
        and hit["content"][0]["type"] == "text"
        and "quasar" in hit["content"][0]["text"]
    )
    out["search_top_hit_is_query_file"] = len(page["data"]) == 1
    s2 = c.post(f"/v1/vector_stores/{vid}/search", json={"query": "market"})
    scores = [h["score"] for h in s2.json()["data"]]
    out["search_scores_desc_sorted"] = len(scores) == 2 and scores == sorted(scores, reverse=True)
    s3 = c.post(f"/v1/vector_stores/{vid}/search", json={"query": ["quasar", "ledger"]})
    out["search_query_list_joins"] = s3.json()["search_query"] == "quasar ledger" and {
        h["file_id"] for h in s3.json()["data"]
    } == {f_alpha, f_beta}
    empty_q = c.post(f"/v1/vector_stores/{vid}/search", json={"query": ""})
    out["search_empty_query_400"] = (
        empty_q.status_code == 400 and _code(empty_q) == "invalid_request"
    )
    ws_q = c.post(f"/v1/vector_stores/{vid}/search", json={"query": "   "})
    out["search_whitespace_query_400"] = (
        ws_q.status_code == 400 and _code(ws_q) == "invalid_request"
    )
    out["search_query_empty_list_422"] = (
        c.post(f"/v1/vector_stores/{vid}/search", json={"query": []}).status_code == 422
    )
    out["search_query_nonstr_member_422"] = (
        c.post(f"/v1/vector_stores/{vid}/search", json={"query": ["a", 1]}).status_code == 422
    )
    out["search_query_int_422"] = (
        c.post(f"/v1/vector_stores/{vid}/search", json={"query": 5}).status_code == 422
    )
    long_q = c.post(f"/v1/vector_stores/{vid}/search", json={"query": "x" * 8193})
    out["search_query_too_long_400"] = (
        long_q.status_code == 400 and _code(long_q) == "invalid_request"
    )
    capped = c.post(
        f"/v1/vector_stores/{vid}/search",
        json={"query": "market", "max_num_results": 1},
    )
    out["search_max_results_caps_hits"] = len(capped.json()["data"]) == 1
    out["search_max_results_bounds_422"] = (
        c.post(
            f"/v1/vector_stores/{vid}/search",
            json={"query": "market", "max_num_results": 0},
        ).status_code
        == 422
        and c.post(
            f"/v1/vector_stores/{vid}/search",
            json={"query": "market", "max_num_results": 51},
        ).status_code
        == 422
    )
    no_match = c.post(f"/v1/vector_stores/{vid}/search", json={"query": "zzznothere"})
    out["search_no_match_empty_data"] = (
        no_match.status_code == 200 and no_match.json()["data"] == []
    )
    empty_vs = _mk_vs(c)
    empty_s = c.post(f"/v1/vector_stores/{empty_vs}/search", json={"query": "quasar"})
    out["search_empty_store_empty_hits"] = (
        empty_s.status_code == 200 and empty_s.json()["data"] == []
    )
    ghost = c.post("/v1/vector_stores/vs_ghost/search", json={"query": "q"})
    out["search_ghost_store_404"] = (
        ghost.status_code == 404 and _code(ghost) == "vector_store_not_found"
    )
    eq = c.post(
        f"/v1/vector_stores/{vid}/search",
        json={
            "query": "market",
            "filters": {"type": "eq", "key": "cat", "value": "finance"},
        },
    )
    out["search_filters_eq_narrows"] = [h["file_id"] for h in eq.json()["data"]] == [f_beta]
    f_noattr = _upload_text(c, "gamma market tangent")
    _attach(c, vid, f_noattr)
    ne = c.post(
        f"/v1/vector_stores/{vid}/search",
        json={
            "query": "market",
            "filters": {"type": "ne", "key": "cat", "value": "finance"},
        },
    )
    out["search_filters_ne_absent_passes"] = {h["file_id"] for h in ne.json()["data"]} == {
        f_alpha,
        f_noattr,
    }
    compound = c.post(
        f"/v1/vector_stores/{vid}/search",
        json={
            "query": "market",
            "filters": {
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "cat", "value": "finance"},
                    {"type": "ne", "key": "missing", "value": "x"},
                ],
            },
        },
    )
    out["search_filters_compound_and"] = [h["file_id"] for h in compound.json()["data"]] == [f_beta]
    in_f = c.post(
        f"/v1/vector_stores/{vid}/search",
        json={
            "query": "market",
            "filters": {"type": "in", "key": "cat", "value": ["finance"]},
        },
    )
    out["search_filters_in_list"] = [h["file_id"] for h in in_f.json()["data"]] == [f_beta]
    bad_f = c.post(
        f"/v1/vector_stores/{vid}/search",
        json={"query": "market", "filters": {"type": "regex", "key": "cat", "value": "x"}},
    )
    out["search_filters_bogus_type_400"] = (
        bad_f.status_code == 400 and _code(bad_f) == "invalid_filters"
    )
    deep: dict[str, Any] = {"type": "eq", "key": "cat", "value": "finance"}
    for _ in range(6):
        deep = {"type": "and", "filters": [deep]}
    deep_f = c.post(
        f"/v1/vector_stores/{vid}/search",
        json={"query": "market", "filters": deep},
    )
    out["search_filters_too_deep_400"] = (
        deep_f.status_code == 400 and _code(deep_f) == "invalid_filters"
    )
    strong = c.post(
        f"/v1/vector_stores/{vid}/search",
        json={"query": "market", "ranking_options": {"score_threshold": 0.9999}},
    )
    out["search_score_threshold_drops_weak"] = strong.status_code == 200 and len(
        strong.json()["data"]
    ) < len(c.post(f"/v1/vector_stores/{vid}/search", json={"query": "market"}).json()["data"])
    bad_st = c.post(
        f"/v1/vector_stores/{vid}/search",
        json={"query": "market", "ranking_options": {"score_threshold": 2.0}},
    )
    out["search_score_threshold_range_400"] = (
        bad_st.status_code == 400 and _code(bad_st) == "invalid_filters"
    )
    out["search_ranker_bogus_422"] = (
        c.post(
            f"/v1/vector_stores/{vid}/search",
            json={"query": "market", "ranking_options": {"ranker": "bm25"}},
        ).status_code
        == 422
    )
    out["search_ranking_options_unknown_key_422"] = (
        c.post(
            f"/v1/vector_stores/{vid}/search",
            json={"query": "market", "ranking_options": {"hyper": 1}},
        ).status_code
        == 422
    )
    out["search_rewrite_query_422"] = (
        c.post(
            f"/v1/vector_stores/{vid}/search",
            json={"query": "market", "rewrite_query": True},
        ).status_code
        == 422
    )
    vid_del = _mk_vs(c)
    fd = _upload_text(c, "doomed search doc")
    _attach(c, vid_del, fd)
    c.delete(f"/v1/vector_stores/{vid_del}")
    del_s = c.post(f"/v1/vector_stores/{vid_del}/search", json={"query": "doomed"})
    out["search_deleted_store_404"] = (
        del_s.status_code == 404 and _code(del_s) == "vector_store_not_found"
    )
    det1 = c.post(f"/v1/vector_stores/{vid}/search", json={"query": "market"}).json()
    det2 = c.post(f"/v1/vector_stores/{vid}/search", json={"query": "market"}).json()
    out["search_deterministic_scores"] = [h["score"] for h in det1["data"]] == [
        h["score"] for h in det2["data"]
    ]
    return out


def _probe_chunking(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c)
    # 600 distinct words → auto window 200 / step 160 → chunks at 0,160,320,480
    auto_text = " ".join(f"w{i}" for i in range(600))
    fa = _upload_text(c, auto_text)
    rec_a = _attach(c, vid, fa)
    out["chunking_auto_window_counts"] = (
        rec_a["indexed_chunks"] == 4 and rec_a["truncated"] is False
    )
    fs = _upload_text(c, auto_text)
    rec_s = _attach(
        c,
        vid,
        fs,
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 400, "chunk_overlap_tokens": 0},
        },
    )
    # 400 tokens → 300-word window, step 300 → 2 chunks over 600 words
    out["chunking_static_window_counts"] = rec_s["indexed_chunks"] == 2
    fo = _upload_text(c, auto_text)
    rec_o = _attach(
        c,
        vid,
        fo,
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 400, "chunk_overlap_tokens": 80},
        },
    )
    # 300-word window, step 240 → chunks at 0,240,480 → 3 chunks
    out["chunking_static_overlap_windows"] = rec_o["indexed_chunks"] == 3
    content = c.get(f"/v1/vector_stores/{vid}/files/{fo}/content").json()
    texts = [p["text"] for p in content["data"]]
    first_tail = texts[0].split()[-60:]
    second_head = texts[1].split()[:60]
    out["chunking_overlap_shares_words"] = first_tail == second_head
    out["file_content_page_shape"] = (
        content["object"] == "vector_store.file_content.page"
        and all(p["type"] == "text" for p in content["data"])
        and content["has_more"] is False
        and content["next_page"] is None
    )
    got = c.get(f"/v1/vector_stores/{vid}/files/{fs}").json()
    out["chunking_strategy_echo_on_get"] = got["chunking_strategy"] == {
        "type": "static",
        "static": {"max_chunk_size_tokens": 400, "chunk_overlap_tokens": 0},
    }
    out["chunking_default_auto_echoed"] = rec_a["chunking_strategy"] == {"type": "auto"}
    # past VS_MAX_CHUNKS (512) the record flags truncated honestly:
    # static 100-token window → 75 words/step → 40k words = 534 windows
    big_text = " ".join(f"t{i}" for i in range(40_000))
    fb = _upload_text(c, big_text)
    rec_b = _attach(
        c,
        vid,
        fb,
        chunking_strategy={
            "type": "static",
            "static": {"max_chunk_size_tokens": 100, "chunk_overlap_tokens": 0},
        },
    )
    out["chunking_truncated_past_cap"] = (
        rec_b["indexed_chunks"] == 512 and rec_b["truncated"] is True
    )
    return out


def _probe_expiry(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c, expires_after={"anchor": "last_active_at", "days": 1})
    created = c.get(f"/v1/vector_stores/{vid}").json()
    out["expiry_create_sets_fields"] = (
        created["expires_after"] == {"anchor": "last_active_at", "days": 1}
        and isinstance(created["expires_at"], int)
        and created["status"] == "completed"
    )
    # expiry is lazy (computed per read, no sweeper) — force it by
    # aging the record past its horizon
    store = cast("FastAPI", c.app).state.vs_store
    meta = store._stores[vid]
    meta.expires_at = int(time.time()) - 1
    expired_read = c.get(f"/v1/vector_stores/{vid}")
    out["expired_read_200_status_expired"] = (
        expired_read.status_code == 200 and expired_read.json()["status"] == "expired"
    )
    fx = _upload_text(c, "expired member")
    att = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": fx})
    out["expired_attach_410"] = att.status_code == 410 and _code(att) == "vector_store_expired"
    bt = c.post(f"/v1/vector_stores/{vid}/file_batches", json={"file_ids": [fx]})
    out["expired_batch_410"] = bt.status_code == 410 and _code(bt) == "vector_store_expired"
    se = c.post(f"/v1/vector_stores/{vid}/search", json={"query": "expired"})
    out["expired_search_410"] = se.status_code == 410 and _code(se) == "vector_store_expired"
    # reads stay open: file list + store list both serve an expired store
    out["expired_files_list_still_200"] = c.get(
        f"/v1/vector_stores/{vid}/files"
    ).status_code == 200 and any(v["id"] == vid for v in c.get("/v1/vector_stores").json()["data"])
    # detach carries no expiry check — membership removal stays open
    vid_x = _mk_vs(c)
    fxe = _upload_text(c, "detachable on expired")
    _attach(c, vid_x, fxe)
    store._stores[vid_x].expires_at = int(time.time()) - 1
    det = c.delete(f"/v1/vector_stores/{vid_x}/files/{fxe}")
    out["expired_detach_still_works"] = det.status_code == 200
    # a name-only update on an expired store is allowed — stays expired
    nu = c.post(f"/v1/vector_stores/{vid}", json={"name": "still-expired"})
    out["expired_update_name_only_stays_expired"] = (
        nu.status_code == 200 and nu.json()["status"] == "expired"
    )
    rev = c.post(
        f"/v1/vector_stores/{vid}",
        json={"expires_after": {"anchor": "last_active_at", "days": 2}},
    )
    revived = rev.json()
    out["expired_update_revives"] = (
        rev.status_code == 200
        and revived["status"] == "completed"
        and revived["expires_after"]["days"] == 2
        and int(revived["expires_at"]) > int(time.time())
    )
    att2 = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": fx})
    out["revived_store_accepts_writes"] = att2.status_code == 200
    return out


def _probe_delete(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c)
    f1 = _upload_text(c, "member survives store delete")
    _attach(c, vid, f1)
    f2 = _upload_text(c, "batch member delete check")
    bid = c.post(f"/v1/vector_stores/{vid}/file_batches", json={"file_ids": [f2]})
    batch_id = str(bid.json()["id"])
    r = c.delete(f"/v1/vector_stores/{vid}")
    out["delete_wire_shape"] = r.status_code == 200 and r.json() == {
        "id": vid,
        "object": "vector_store.deleted",
        "deleted": True,
    }
    subroutes = {
        "get": c.get(f"/v1/vector_stores/{vid}").status_code,
        "files_list": c.get(f"/v1/vector_stores/{vid}/files").status_code,
        "file_card": c.get(f"/v1/vector_stores/{vid}/files/{f1}").status_code,
        "file_content": c.get(f"/v1/vector_stores/{vid}/files/{f1}/content").status_code,
        "attach": c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f1}).status_code,
        "search": c.post(f"/v1/vector_stores/{vid}/search", json={"query": "x"}).status_code,
        "batch_create": c.post(
            f"/v1/vector_stores/{vid}/file_batches", json={"file_ids": [f1]}
        ).status_code,
        "batch_get": c.get(f"/v1/vector_stores/{vid}/file_batches/{batch_id}").status_code,
        "update": c.post(f"/v1/vector_stores/{vid}", json={"name": "x"}).status_code,
        "detach": c.delete(f"/v1/vector_stores/{vid}/files/{f1}").status_code,
        "batch_files": c.get(f"/v1/vector_stores/{vid}/file_batches/{batch_id}/files").status_code,
        "batch_cancel": c.post(
            f"/v1/vector_stores/{vid}/file_batches/{batch_id}/cancel"
        ).status_code,
    }
    out["delete_404s_every_subroute"] = all(v == 404 for v in subroutes.values())
    out["delete_removes_from_list"] = all(
        v["id"] != vid for v in c.get("/v1/vector_stores").json()["data"]
    )
    again = c.delete(f"/v1/vector_stores/{vid}")
    out["delete_twice_404"] = again.status_code == 404 and _code(again) == "vector_store_not_found"
    ghost = c.delete("/v1/vector_stores/vs_ghost")
    out["delete_ghost_404"] = ghost.status_code == 404 and _code(ghost) == "vector_store_not_found"
    out["delete_member_file_records_survive"] = (
        c.get(f"/v1/files/{f1}").status_code == 200
        and c.get(f"/v1/files/{f1}/content").status_code == 200
    )
    return out


def _probe_list(c: TestClient, fresh: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    empty = fresh.get("/v1/vector_stores").json()
    out["list_empty_shape"] = (
        empty["object"] == "list"
        and empty["data"] == []
        and empty["first_id"] is None
        and empty["last_id"] is None
        and empty["has_more"] is False
        and set(empty) == _PAGE_KEYS
    )
    a = _mk_vs(c, name="a")
    b = _mk_vs(c, name="b")
    d = _mk_vs(c, name="d")
    lst = c.get("/v1/vector_stores").json()
    ids = [v["id"] for v in lst["data"]]
    out["list_newest_first_desc_default"] = ids[:3] == [d, b, a]
    # default limit truncates the page — read the full asc listing to
    # pin the tail (a/b/d are the newest members of a populated store)
    asc = c.get("/v1/vector_stores", params={"order": "asc", "limit": 100}).json()
    out["list_order_asc_flips"] = asc["data"][-3:][0]["id"] == a and asc["data"][-1]["id"] == d
    p1 = c.get("/v1/vector_stores", params={"limit": 1}).json()
    out["list_limit_one_pages"] = (
        len(p1["data"]) == 1 and p1["data"][0]["id"] == d and p1["has_more"] is True
    )
    p2 = c.get("/v1/vector_stores", params={"limit": 1, "after": d}).json()
    out["list_after_cursor_walks"] = p2["data"][0]["id"] == b
    pb = c.get("/v1/vector_stores", params={"before": b, "order": "asc", "limit": 100}).json()
    out["list_before_cursor_walks"] = pb["data"][-1]["id"] == a
    bad_cur = c.get("/v1/vector_stores", params={"after": "vs_nonexistent"})
    out["list_unknown_cursor_400"] = (
        bad_cur.status_code == 400 and _code(bad_cur) == "invalid_cursor"
    )
    bad_order = c.get("/v1/vector_stores", params={"order": "bogus"})
    out["list_bad_order_400"] = (
        bad_order.status_code == 400 and _code(bad_order) == "invalid_cursor"
    )
    out["list_limit_bounds_422"] = (
        c.get("/v1/vector_stores", params={"limit": 0}).status_code == 422
        and c.get("/v1/vector_stores", params={"limit": 101}).status_code == 422
    )
    return out


def _probe_store_cap(stack: ExitStack) -> dict[str, bool]:
    out: dict[str, bool] = {}
    c2 = stack.enter_context(_client(store_max=2))
    v1 = _mk_vs(c2, name="first")
    v2 = _mk_vs(c2, name="second")
    v3 = _mk_vs(c2, name="third")
    out["cap_evicts_oldest"] = (
        c2.get(f"/v1/vector_stores/{v1}").status_code == 404
        and c2.get(f"/v1/vector_stores/{v2}").status_code == 200
        and c2.get(f"/v1/vector_stores/{v3}").status_code == 200
    )
    # a GET promotes the store past the LRU victim slot
    c2.get(f"/v1/vector_stores/{v2}")
    v4 = _mk_vs(c2, name="fourth")
    out["cap_get_refreshes_lru"] = (
        c2.get(f"/v1/vector_stores/{v2}").status_code == 200
        and c2.get(f"/v1/vector_stores/{v3}").status_code == 404
        and c2.get(f"/v1/vector_stores/{v4}").status_code == 200
    )
    # eviction drops members with the store; the file-* record survives.
    # (attach itself touches the LRU, so v4 is refreshed past v2 here to
    # make the member-bearing v2 the evictee)
    f1 = _upload_text(c2, "cap member")
    _attach(c2, v2, f1)
    c2.get(f"/v1/vector_stores/{v4}")
    v5 = _mk_vs(c2, name="fifth")
    out["cap_evicted_members_gone"] = (
        c2.get(f"/v1/vector_stores/{v2}/files").status_code == 404
        and c2.get(f"/v1/files/{f1}").status_code == 200
        and c2.get(f"/v1/vector_stores/{v4}").status_code == 200
        and c2.get(f"/v1/vector_stores/{v5}").status_code == 200
    )
    return out


def _probe_scope_auth(stack: ExitStack) -> dict[str, bool]:
    out: dict[str, bool] = {}
    with _env_key():
        c = stack.enter_context(_client())
        env_h = _h(_ROOT_KEY)
        out["auth_required_401"] = c.post("/v1/vector_stores", json={}).status_code == 401
        out["auth_wrong_key_401"] = (
            c.post("/v1/vector_stores", json={}, headers=_h("wrong")).status_code == 401
        )
        out["auth_bearer_accepted_on_v1"] = (
            c.post("/v1/vector_stores", json={}, headers=_bearer(_ROOT_KEY)).status_code == 200
        )
        k_rw, _id_rw = _mint(c, name="rw", scopes=["read", "write"])
        k_r, _id_r = _mint(c, name="r", scopes=["read"])
        k_w, _id_w = _mint(c, name="w", scopes=["write"])
        out["managed_keys_mint_under_env_key"] = all(
            isinstance(k, str) and k for k in (k_rw, k_r, k_w)
        )
        vid = _mk_vs(c, headers=env_h)
        out["scope_write_creates_200"] = (
            c.post("/v1/vector_stores", json={}, headers=_h(k_w)).status_code == 200
        )
        w_get = c.get(f"/v1/vector_stores/{vid}", headers=_h(k_w))
        out["scope_write_cannot_read_403"] = (
            w_get.status_code == 403 and _code(w_get) == "insufficient_scope"
        )
        r_create = c.post("/v1/vector_stores", json={}, headers=_h(k_r))
        out["scope_read_cannot_write_403"] = (
            r_create.status_code == 403 and _code(r_create) == "insufficient_scope"
        )
        out["scope_read_reads_200"] = (
            c.get("/v1/vector_stores", headers=_h(k_r)).status_code == 200
            and c.get(f"/v1/vector_stores/{vid}", headers=_h(k_r)).status_code == 200
        )
        out["scope_rw_both_verbs_200"] = (
            c.post("/v1/vector_stores", json={}, headers=_h(k_rw)).status_code == 200
            and c.get("/v1/vector_stores", headers=_h(k_rw)).status_code == 200
        )
        # shared workspace: key B reads/attaches/detaches/deletes A's store
        k_b, _id_b = _mint(c, name="b", scopes=["read", "write"])
        f1 = _upload_text(c, "shared doc", headers=env_h)
        vid_a = _mk_vs(c, headers=env_h)
        att_b = c.post(
            f"/v1/vector_stores/{vid_a}/files",
            json={"file_id": f1},
            headers=_h(k_b),
        )
        det_b = c.delete(f"/v1/vector_stores/{vid_a}/files/{f1}", headers=_h(k_b))
        del_b = c.delete(f"/v1/vector_stores/{vid_a}", headers=_h(k_b))
        out["cross_key_shared_workspace"] = (
            att_b.status_code == 200 and det_b.status_code == 200 and del_b.status_code == 200
        )
        # control plane needs admin — a data-plane key cannot mint/drain;
        # /harness refusals carry the {detail, code} envelope
        adm = c.post("/harness/keys", json={}, headers=_h(k_rw))
        out["admin_scope_gates_control_plane_403"] = (
            adm.status_code == 403 and adm.json()["code"] == "insufficient_scope"
        )
    return out


def _probe_drain(stack: ExitStack) -> dict[str, bool]:
    out: dict[str, bool] = {}
    with _env_key():
        c = stack.enter_context(_client())
        env_h = _h(_ROOT_KEY)
        vid = _mk_vs(c, headers=env_h)
        f1 = _upload_text(c, "drain member", headers=env_h)
        _attach(c, vid, f1, headers=env_h)
        bid = c.post(
            f"/v1/vector_stores/{vid}/file_batches",
            json={"file_ids": [_upload_text(c, "batch member", headers=env_h)]},
            headers=env_h,
        ).json()["id"]
        dr = c.post("/harness/drain", headers=env_h)
        out["drain_latches"] = dr.status_code == 200 and dr.json()["draining"] is True
        mutations = {
            "create": c.post("/v1/vector_stores", json={}, headers=env_h),
            "update": c.post(f"/v1/vector_stores/{vid}", json={"name": "x"}, headers=env_h),
            "attach": c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f1}, headers=env_h),
            "batch": c.post(
                f"/v1/vector_stores/{vid}/file_batches",
                json={"file_ids": [f1]},
                headers=env_h,
            ),
        }
        for name, resp in mutations.items():
            out[f"drain_{name}_503"] = resp.status_code == 503 and _code(resp) == "draining"
        reads = {
            "get": c.get(f"/v1/vector_stores/{vid}", headers=env_h),
            "list": c.get("/v1/vector_stores", headers=env_h),
            "files": c.get(f"/v1/vector_stores/{vid}/files", headers=env_h),
            "file_card": c.get(f"/v1/vector_stores/{vid}/files/{f1}", headers=env_h),
            "file_content": c.get(f"/v1/vector_stores/{vid}/files/{f1}/content", headers=env_h),
            "batch_get": c.get(f"/v1/vector_stores/{vid}/file_batches/{bid}", headers=env_h),
            "batch_files": c.get(
                f"/v1/vector_stores/{vid}/file_batches/{bid}/files", headers=env_h
            ),
            "search": c.post(
                f"/v1/vector_stores/{vid}/search", json={"query": "drain"}, headers=env_h
            ),
        }
        out["drain_reads_stay_open"] = all(r.status_code == 200 for r in reads.values())
        # deletes/detaches release capacity — they stay open under drain,
        # like job cancel on the submit surface
        out["drain_delete_open"] = (
            c.delete(f"/v1/vector_stores/{vid}", headers=env_h).status_code == 200
        )
        return out


def _probe_drain_lifecycle(stack: ExitStack) -> dict[str, bool]:
    out: dict[str, bool] = {}
    with _env_key():
        c = stack.enter_context(_client())
        env_h = _h(_ROOT_KEY)
        vid = _mk_vs(c, headers=env_h)
        f1 = _upload_text(c, "lifecycle member", headers=env_h)
        _attach(c, vid, f1, headers=env_h)
        bid = c.post(
            f"/v1/vector_stores/{vid}/file_batches",
            json={"file_ids": [_upload_text(c, "b2", headers=env_h)]},
            headers=env_h,
        ).json()["id"]
        c.post("/harness/drain", headers=env_h)
        det = c.delete(f"/v1/vector_stores/{vid}/files/{f1}", headers=env_h)
        out["drain_detach_open"] = det.status_code == 200
        cancel = c.post(f"/v1/vector_stores/{vid}/file_batches/{bid}/cancel", headers=env_h)
        out["drain_batch_cancel_terminal_409"] = (
            cancel.status_code == 409 and _code(cancel) == "file_batch_terminal"
        )
        # a stored idem replay stays readable under drain — the work ran
        c2 = stack.enter_context(_client())
        vid2 = _mk_vs(c2, headers=env_h)
        kh = {**env_h, "Idempotency-Key": "drain-idem"}
        fx = _upload_text(c2, "drain replay member", headers=env_h)
        c2.post(f"/v1/vector_stores/{vid2}/files", json={"file_id": fx}, headers=kh)
        c2.post("/harness/drain", headers=env_h)
        replay = c2.post(f"/v1/vector_stores/{vid2}/files", json={"file_id": fx}, headers=kh)
        out["drain_idem_replay_bypasses_gate"] = (
            replay.status_code == 200 and replay.headers.get("x-fx1-idempotent-replay") == "true"
        )
    return out


def _probe_idem(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    kh = {"Idempotency-Key": "vs-idem-a"}
    r1 = c.post("/v1/vector_stores", json={"name": "i1"}, headers=kh)
    r2 = c.post("/v1/vector_stores", json={"name": "i1"}, headers=kh)
    out["idem_create_replays_same_id"] = (
        r1.status_code == 200 and r2.status_code == 200 and r1.json()["id"] == r2.json()["id"]
    )
    out["idem_create_replay_header"] = (
        r2.headers.get("x-fx1-idempotent-replay") == "true"
        and r1.headers.get("x-fx1-idempotent-replay") is None
    )
    out["idem_create_replay_byte_identical"] = r1.content == r2.content
    conflict = c.post("/v1/vector_stores", json={"name": "DIFF"}, headers=kh)
    out["idem_create_conflict_409"] = (
        conflict.status_code == 409 and _code(conflict) == "idempotency_conflict"
    )
    n1 = c.post("/v1/vector_stores", json={"name": "n"})
    n2 = c.post("/v1/vector_stores", json={"name": "n"})
    out["idem_no_key_reruns"] = n1.json()["id"] != n2.json()["id"]
    # refused requests never poison the key: a failed create under K
    # leaves K free for the next attempt
    bad_key = {"Idempotency-Key": "vs-idem-bad"}
    bad = c.post("/v1/vector_stores", json={"file_ids": ["file-x"]}, headers=bad_key)
    good = c.post("/v1/vector_stores", json={"name": "ok"}, headers=bad_key)
    out["idem_refusal_not_recorded"] = bad.status_code == 404 and good.status_code == 200
    vid = _mk_vs(c)
    f1 = _upload_text(c, "idem attach member")
    ah = {"Idempotency-Key": "vs-idem-attach"}
    a1 = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f1}, headers=ah)
    a2 = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f1}, headers=ah)
    out["idem_attach_replays_not_409"] = (
        a1.status_code == 200
        and a2.status_code == 200
        and a2.headers.get("x-fx1-idempotent-replay") == "true"
        and a1.json() == a2.json()
    )
    f2 = _upload_text(c, "idem attach member two")
    a_conf = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f2}, headers=ah)
    out["idem_attach_conflict_409"] = (
        a_conf.status_code == 409 and _code(a_conf) == "idempotency_conflict"
    )
    f3 = _upload_text(c, "idem batch member")
    bh = {"Idempotency-Key": "vs-idem-batch"}
    b1 = c.post(f"/v1/vector_stores/{vid}/file_batches", json={"file_ids": [f3]}, headers=bh)
    b2 = c.post(f"/v1/vector_stores/{vid}/file_batches", json={"file_ids": [f3]}, headers=bh)
    out["idem_batch_replays_same_batch"] = (
        b1.json()["id"] == b2.json()["id"] and b2.headers.get("x-fx1-idempotent-replay") == "true"
    )
    f4 = _upload_text(c, "idem batch conflict")
    b_conf = c.post(f"/v1/vector_stores/{vid}/file_batches", json={"file_ids": [f4]}, headers=bh)
    out["idem_batch_conflict_409"] = (
        b_conf.status_code == 409 and _code(b_conf) == "idempotency_conflict"
    )
    # namespacing: one key may pin a create AND an attach AND the same
    # attach on a different store — the record is per-route/per-object
    shared = {"Idempotency-Key": "vs-idem-shared"}
    c_same = c.post("/v1/vector_stores", json={"name": "s"}, headers=shared)
    f5 = _upload_text(c, "shared key member")
    vid2 = _mk_vs(c)
    at1 = c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f5}, headers=shared)
    at2 = c.post(f"/v1/vector_stores/{vid2}/files", json={"file_id": f5}, headers=shared)
    out["idem_key_namespaced_by_route_and_store"] = (
        c_same.status_code == 200 and at1.status_code == 200 and at2.status_code == 200
    )
    # an over-long key refuses 400 before any lookup
    too_long = c.post("/v1/vector_stores", json={}, headers={"Idempotency-Key": "k" * 257})
    out["idem_key_overlong_400"] = too_long.status_code == 400
    return out


def _probe_durability(tmp: Path) -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.vectorstores import VectorStoreStore

    state = tmp / "vs-state"
    with _client(state_dir=state) as c1:
        fa = _upload_text(c1, "alpha quasar nebula")
        fb = _upload_text(c1, "beta ledger margin")
        vid = _mk_vs(
            c1,
            name="durable",
            metadata={"k": "v"},
            expires_after={"anchor": "last_active_at", "days": 5},
        )
        _attach(c1, vid, fa)
        bid = c1.post(f"/v1/vector_stores/{vid}/file_batches", json={"file_ids": [fb]}).json()["id"]
        # a doomed store — deleted before restart — must stay deleted
        doomed = _mk_vs(c1, name="doomed")
        c1.delete(f"/v1/vector_stores/{doomed}")
        # a detached member's tombstone must survive too
        vid2 = _mk_vs(c1, name="detach-check")
        fc = _upload_text(c1, "detached member")
        _attach(c1, vid2, fc)
        c1.delete(f"/v1/vector_stores/{vid2}/files/{fc}")
        # update is journaled: rename must survive the restart
        c1.post(f"/v1/vector_stores/{vid}", json={"name": "renamed"})
        # idem record journals under the shared store
        idem_h = {"Idempotency-Key": "vs-idem-durable"}
        created_idem = c1.post("/v1/vector_stores", json={"name": "idem"}, headers=idem_h).json()[
            "id"
        ]
        # LRU order: touching vid2 (created after vid) makes vid older —
        # list order must survive restart
        c1.get(f"/v1/vector_stores/{vid2}")
        c1.get(f"/v1/vector_stores/{vid}")
        usage_at_rest = c1.get(f"/v1/vector_stores/{vid}").json()["usage_bytes"]
        out["journal_file_on_disk"] = (state / "vector_stores.jsonl").is_file()
        # a file whose bytes vanish before restart re-indexes as failed
        vid3 = _mk_vs(c1, name="missing-bytes")
        fgone = _upload_text(c1, "bytes will vanish")
        _attach(c1, vid3, fgone)
        c1.delete(f"/v1/files/{fgone}")
    with _client(state_dir=state) as c2:
        got = c2.get(f"/v1/vector_stores/{vid}")
        out["store_survives_restart"] = (
            got.status_code == 200
            and got.json()["name"] == "renamed"
            and got.json()["metadata"] == {"k": "v"}
        )
        out["update_survives_restart"] = got.json()["name"] == "renamed"
        fobj = c2.get(f"/v1/vector_stores/{vid}/files/{fa}")
        out["files_survive_restart"] = (
            fobj.status_code == 200 and fobj.json()["status"] == "completed"
        )
        out["files_survive_both_members"] = (
            c2.get(f"/v1/vector_stores/{vid}/files/{fb}").status_code == 200
        )
        bobj = c2.get(f"/v1/vector_stores/{vid}/file_batches/{bid}")
        out["batch_survives_restart"] = (
            bobj.status_code == 200
            and bobj.json()["status"] == "completed"
            and bobj.json()["file_counts"]["completed"] == 1
        )
        page = c2.get(f"/v1/vector_stores/{vid}/file_batches/{bid}/files").json()
        out["batch_verdicts_survive_restart"] = [r_["status"] for r_ in page["data"]] == [
            "completed"
        ]
        srch = c2.post(f"/v1/vector_stores/{vid}/search", json={"query": "quasar"})
        out["search_reindexed_survives_restart"] = srch.status_code == 200 and any(
            h["file_id"] == fa for h in srch.json()["data"]
        )
        out["deleted_store_stays_deleted"] = (
            c2.get(f"/v1/vector_stores/{doomed}").status_code == 404
        )
        det_list = c2.get(f"/v1/vector_stores/{vid2}/files").json()
        out["detached_file_stays_detached"] = det_list["data"] == []
        got2 = c2.get(f"/v1/vector_stores/{vid}")
        out["expiry_policy_survives_restart"] = got2.json()["expires_after"] == {
            "anchor": "last_active_at",
            "days": 5,
        } and isinstance(got2.json()["expires_at"], int)
        out["usage_bytes_survives_restart"] = got2.json()["usage_bytes"] == usage_at_rest
        replayed = c2.post("/v1/vector_stores", json={"name": "idem"}, headers=idem_h)
        out["idem_record_survives_restart"] = (
            replayed.status_code == 200
            and replayed.json()["id"] == created_idem
            and replayed.headers.get("x-fx1-idempotent-replay") == "true"
        )
        # replay re-reads bytes through file_reader — missing bytes land
        # failed honestly instead of phantom-searchable
        gone_rec = c2.get(f"/v1/vector_stores/{vid3}/files/{fgone}")
        out["missing_bytes_replay_fails_closed"] = (
            gone_rec.status_code == 200
            and gone_rec.json()["status"] == "failed"
            and gone_rec.json()["last_error"]["code"] == "file_missing_at_replay"
        )
        gone_search = c2.post(f"/v1/vector_stores/{vid3}/search", json={"query": "vanish"})
        out["missing_bytes_never_searchable"] = gone_search.json()["data"] == []
        # LRU touch order is journaled: vid2 (touched earlier) is older
        # than vid in recency order → desc list puts vid first
        order = [v["id"] for v in c2.get("/v1/vector_stores").json()["data"]]
        out["lru_order_survives_restart"] = order.index(vid) < order.index(vid2)
        store2 = cast("FastAPI", c2.app).state.vs_store
        out["clean_restart_no_recover_warnings"] = store2.recover_warnings == []
    # eviction is journaled — an evicted store stays evicted
    estate = tmp / "vs-evict"
    with _client(state_dir=estate, store_max=1) as e1:
        ev1 = _mk_vs(e1, name="old")
        _mk_vs(e1, name="new")
    with _client(state_dir=estate, store_max=1) as e2:
        out["evicted_store_stays_evicted"] = e2.get(f"/v1/vector_stores/{ev1}").status_code == 404
    # journal-before-publish: a failed append publishes nothing
    jstate = tmp / "vs-journal"
    vstore = VectorStoreStore(4, state_dir=jstate)
    vstore.create(name="a")
    journal = cast("Any", vstore._journal)
    append = journal.append

    def fail_append(payload: dict[str, Any]) -> None:
        raise OSError("synthetic journal failure")

    journal.append = fail_append
    try:
        vstore.create(name="b")
        create_failed = False
    except OSError:
        create_failed = True
    out["journal_failed_create_not_published"] = create_failed and [
        v["name"] for v in vstore.list_stores()["data"]
    ] == ["a"]
    try:
        vstore.delete([v["id"] for v in vstore.list_stores()["data"]][0])
        delete_failed = False
    except OSError:
        delete_failed = True
    out["journal_failed_delete_not_published"] = (
        delete_failed and len(vstore.list_stores()["data"]) == 1
    )
    journal.append = append
    return out


def _probe_concurrency(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c)
    fids = [_upload_text(c, f"concurrent member {i} token{i}") for i in range(8)]
    results: list[int] = []
    _run_threads(
        lambda i: results.append(
            c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": fids[i]}).status_code
        )
    )
    out["concurrent_attach_distinct_files_all_commit"] = sorted(results) == [200] * 8
    page = c.get(f"/v1/vector_stores/{vid}/files", params={"limit": 100}).json()
    out["concurrent_attach_all_present"] = {f["id"] for f in page["data"]} == set(fids)

    same_results: list[int] = []
    fs = _upload_text(c, "race me")
    vid_s = _mk_vs(c)
    _run_threads(
        lambda i: same_results.append(
            c.post(f"/v1/vector_stores/{vid_s}/files", json={"file_id": fs}).status_code
        )
    )
    out["concurrent_same_file_single_winner"] = sorted(same_results) == [200] + [409] * 7

    created: list[str] = []
    _run_threads(lambda i: created.append(c.post("/v1/vector_stores", json={}).json()["id"]))
    out["concurrent_create_distinct_ids"] = len(set(created)) == 8

    # delete racing attach on the same store: attach commits then the
    # store drops, or attach 404s — either serial order is consistent
    race_vid = _mk_vs(c)
    fr = _upload_text(c, "race file")
    race: list[int] = []
    _run_calls(
        [
            lambda: race.append(
                c.post(
                    f"/v1/vector_stores/{race_vid}/files",
                    json={"file_id": fr},
                ).status_code
            ),
            lambda: race.append(c.delete(f"/v1/vector_stores/{race_vid}").status_code),
        ]
    )
    out["concurrent_delete_attach_atomic"] = (
        sorted(race) in ([200, 200], [200, 404])
        and c.get(f"/v1/vector_stores/{race_vid}").status_code == 404
    )

    # capacity boundary: one slot left, four racers attach — exactly one wins
    cap_vid = _mk_vs(c)
    for _ in range(31):
        _attach(c, cap_vid, _upload_text(c, "cap fill"))
    edge: list[int] = []
    edge_fids = [_upload_text(c, f"edge {i}") for i in range(4)]
    _run_threads(
        lambda i: edge.append(
            c.post(
                f"/v1/vector_stores/{cap_vid}/files",
                json={"file_id": edge_fids[i]},
            ).status_code
        ),
        n=4,
    )
    out["concurrent_cap_single_winner"] = sorted(edge) == [200] + [409] * 3

    # two batches on one store serialize on the membership lane
    b_vid = _mk_vs(c)
    b_fids = [_upload_text(c, f"batchconc {i}") for i in range(4)]
    bids: list[dict[str, Any]] = []
    _run_calls(
        [
            lambda: bids.append(
                c.post(
                    f"/v1/vector_stores/{b_vid}/file_batches",
                    json={"file_ids": b_fids[:2]},
                ).json()
            ),
            lambda: bids.append(
                c.post(
                    f"/v1/vector_stores/{b_vid}/file_batches",
                    json={"file_ids": b_fids[2:]},
                ).json()
            ),
        ]
    )
    out["concurrent_batches_both_terminal"] = all(
        b["status"] == "completed" and b["file_counts"]["completed"] == 2 for b in bids
    )

    # detach racing attach of a different file on the same store: both
    # resolve in some serial order, never torn
    d_vid = _mk_vs(c)
    f_a = _upload_text(c, "detach race a")
    f_b = _upload_text(c, "detach race b")
    _attach(c, d_vid, f_a)
    dres: list[int] = []
    _run_calls(
        [
            lambda: dres.append(c.delete(f"/v1/vector_stores/{d_vid}/files/{f_a}").status_code),
            lambda: dres.append(
                c.post(
                    f"/v1/vector_stores/{d_vid}/files",
                    json={"file_id": f_b},
                ).status_code
            ),
        ]
    )
    members = c.get(f"/v1/vector_stores/{d_vid}/files").json()["data"]
    out["concurrent_detach_attach_atomic"] = sorted(dres) == [200, 200] and [
        m["id"] for m in members
    ] == [f_b]
    return out


def _probe_metering(stack: ExitStack) -> dict[str, bool]:
    out: dict[str, bool] = {}
    with _env_key():
        c = stack.enter_context(_client())
        env_h = _h(_ROOT_KEY)
        k_raw, k_id = _mint(c, name="metered", scopes=["read", "write"])
        h = _h(k_raw)
        before = _uses(c, k_id)
        vid = c.post("/v1/vector_stores", json={}, headers=h).json()["id"]
        f1 = c.post(
            "/v1/files",
            files={"file": ("m.jsonl", b"metered words")},
            data={"purpose": "batch"},
            headers=h,
        ).json()["id"]
        c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": f1}, headers=h)
        c.get(f"/v1/vector_stores/{vid}", headers=h)
        c.get("/v1/vector_stores", headers=h)
        c.post(f"/v1/vector_stores/{vid}/search", json={"query": "metered"}, headers=h)
        c.post(
            f"/v1/vector_stores/{vid}/file_batches",
            json={"file_ids": [f1]},
            headers=h,
        )
        c.get(f"/v1/vector_stores/{vid}/files", headers=h)
        c.delete(f"/v1/vector_stores/{vid}/files/{f1}", headers=h)
        out["metered_vs_ops_bill_exact_uses"] = _uses(c, k_id) - before == 9
        out["metered_no_token_spend"] = _tokens(c, k_id) == 0
        # refusals past the auth gate still bill — a 404 is metered work
        c.get("/v1/vector_stores/vs_ghost", headers=h)
        out["metered_404_still_bills"] = _uses(c, k_id) - before == 10
        # a scope refusal exits before the metering hook — no bill
        k_ro, k_ro_id = _mint(c, name="ro", scopes=["read"])
        ro_before = _uses(c, k_ro_id)
        c.post("/v1/vector_stores", json={}, headers=_h(k_ro))
        out["metered_scope_refusal_unbilled"] = _uses(c, k_ro_id) == ro_before
        # unauthenticated calls are refused before any key — nothing billed
        c.post("/v1/vector_stores", json={})
        out["metered_401_bills_nothing"] = _uses(c, k_id) - before == 10
        # drain refusals bill uses — the credential authenticated and the
        # gate refused it (same accounting as the job surface)
        vid2 = c.post("/v1/vector_stores", json={}, headers=env_h).json()["id"]
        c.post("/harness/drain", headers=env_h)
        drain_before = _uses(c, k_id)
        refused = c.post(
            f"/v1/vector_stores/{vid2}/files",
            json={"file_id": f1},
            headers=h,
        )
        out["drain_refusal_still_bills_uses"] = (
            refused.status_code == 503 and _uses(c, k_id) - drain_before == 1
        )
    return out


def _probe_envelope(c: TestClient) -> dict[str, bool]:
    out: dict[str, bool] = {}
    vid = _mk_vs(c)
    f1 = _upload_text(c, "envelope member")
    refusals = [
        c.get("/v1/vector_stores/vs_ghost"),
        c.post("/v1/vector_stores/vs_ghost", json={}),
        c.delete("/v1/vector_stores/vs_ghost"),
        c.get("/v1/vector_stores/vs_ghost/files"),
        c.post("/v1/vector_stores/vs_ghost/files", json={"file_id": f1}),
        c.post(f"/v1/vector_stores/{vid}/files", json={"file_id": "file-ghost"}),
        c.delete("/v1/vector_stores/vs_ghost/files/file-x"),
        c.get(f"/v1/vector_stores/{vid}/files/file-ghost"),
        c.get("/v1/vector_stores/vs_ghost/files/file-x/content"),
        c.post("/v1/vector_stores/vs_ghost/search", json={"query": "x"}),
        c.post("/v1/vector_stores/vs_ghost/file_batches", json={"file_ids": ["f"]}),
        c.get("/v1/vector_stores/vs_ghost/file_batches/vsfb_x"),
        c.post("/v1/vector_stores/vs_ghost/file_batches/vsfb_x/cancel"),
        c.get("/v1/vector_stores/vs_ghost/file_batches/vsfb_x/files"),
        c.get("/v1/vector_stores", params={"after": "vs_nope"}),
        c.get("/v1/vector_stores", params={"order": "bogus"}),
        c.post("/v1/vector_stores", json={"name": "x" * 513}),
        c.post("/v1/vector_stores", json={"expires_after": {"days": 0}}),
        c.post(
            f"/v1/vector_stores/{vid}/files",
            json={"file_id": f1, "chunking_strategy": {"type": "x"}},
        ),
        c.post(f"/v1/vector_stores/{vid}/search", json={"query": ""}),
        c.post(
            f"/v1/vector_stores/{vid}/search", json={"query": "x", "filters": {"type": "bogus"}}
        ),
        c.post(f"/v1/vector_stores/{vid}/search", json={"query": "x", "rewrite_query": True}),
        c.put("/v1/vector_stores"),
        c.post("/v1/vector_stores/notjson", content=b"{"),
    ]
    out["envelope_every_refusal_shaped"] = all(_is_envelope(r) for r in refusals)
    codes = {_code(r) for r in refusals}
    out["envelope_codes_from_grammar"] = codes <= {
        "vector_store_not_found",
        "file_not_found",
        "file_batch_not_found",
        "invalid_cursor",
        "invalid_request",
        "invalid_expires_after",
        "invalid_filters",
        "validation",
        "not_found",
        "bad_request",
    }
    put_r = c.put("/v1/vector_stores")
    out["wrong_method_404_envelope"] = put_r.status_code == 404 and _code(put_r) == "not_found"
    return out


def _probe_client_map(c: TestClient, stack: ExitStack) -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.client import (
        BackendNotConfiguredError,
        HarnessAuthError,
        HarnessClient,
        HarnessTransportError,
    )

    client = HarnessClient(base_url="http://vs-audit", transport=_tc_transport(c))
    up = client.upload_file(b"client doc words", filename="c.jsonl", purpose="batch")
    fid = str(up["id"])
    vs = client.vector_store_create(name="client", file_ids=[fid])
    vid = str(vs["id"])
    out["client_create_roundtrip"] = vid.startswith("vs_") and vs["object"] == "vector_store"
    out["client_get_roundtrip"] = client.vector_store_get(vid)["id"] == vid
    out["client_list_contains"] = any(v["id"] == vid for v in client.vector_store_list()["data"])
    out["client_file_list"] = [f["id"] for f in client.vector_store_file_list(vid)["data"]] == [fid]
    out["client_file_get"] = client.vector_store_file_get(vid, fid)["status"] == "completed"
    content = client.vector_store_file_content(vid, fid)
    out["client_file_content_chunks"] = (
        content["object"] == "vector_store.file_content.page"
        and "words" in content["data"][0]["text"]
    )
    srch = client.vector_store_search(vid, "words")
    out["client_search_hits"] = srch["data"][0]["file_id"] == fid
    b = client.vector_store_file_batch_create(vid, [_upload_text(c, "batch cli")])
    out["client_batch_create"] = b["object"] == "vector_store.files_batch"
    out["client_batch_get"] = client.vector_store_file_batch_get(vid, b["id"]) == b
    out["client_batch_files"] = len(client.vector_store_file_batch_files(vid, b["id"])["data"]) == 1
    try:
        client.vector_store_file_batch_cancel(vid, b["id"])
        cancel_409 = False
    except HarnessTransportError:
        cancel_409 = True
    out["client_batch_cancel_409_maps"] = cancel_409
    out["client_update"] = client.vector_store_update(vid, name="renamed")["name"] == "renamed"
    out["client_detach"] = client.vector_store_file_delete(vid, fid)["deleted"] is True
    try:
        client.vector_store_file_create(vid, "file-ghost")
        map404 = False
    except KeyError:
        map404 = True
    out["client_404_maps_keyerror"] = map404
    f_dup = _upload_text(c, "client dup member")
    client.vector_store_file_create(vid, f_dup)
    try:
        client.vector_store_file_create(vid, f_dup)
        dup_409 = False
    except HarnessTransportError:
        dup_409 = True
    out["client_409_maps_transport_error"] = dup_409
    try:
        client.vector_store_search(vid, "x", max_num_results=0)
        map422 = False
    except ValueError:
        map422 = True
    out["client_422_maps_valueerror"] = map422

    def _mapped(status: int) -> Any:
        def send(
            method: str,
            url: str,
            payload: dict[str, Any] | bytes | None,
            headers: dict[str, str],
            timeout_s: float,
        ) -> tuple[int, Mapping[str, str], bytes]:
            return status, {}, b'{"error": {"message": "m", "code": "x"}}'

        c2 = HarnessClient(base_url="http://vs-audit", transport=send)
        try:
            c2.vector_store_get("vs-x")
        except Exception as exc:  # noqa: BLE001 — the map IS the contract
            return exc
        return None

    out["client_401_maps_autherror"] = isinstance(_mapped(401), HarnessAuthError)
    out["client_403_maps_autherror"] = isinstance(_mapped(403), HarnessAuthError)
    out["client_501_maps_notimplemented"] = isinstance(_mapped(501), NotImplementedError)
    out["client_503_maps_backenderror"] = isinstance(_mapped(503), BackendNotConfiguredError)
    out["client_500_maps_transport_error"] = isinstance(_mapped(500), HarnessTransportError)
    out["client_delete_roundtrip"] = client.vector_store_delete(vid)["deleted"] is True
    with _env_key():
        keyed = stack.enter_context(_client())
        kc = HarnessClient(base_url="http://vs-audit", transport=_tc_transport(keyed))
        try:
            kc.vector_store_create()
            no_key_401 = False
        except HarnessAuthError:
            no_key_401 = True
        kc2 = HarnessClient(
            base_url="http://vs-audit",
            transport=_tc_transport(keyed),
            api_key=_ROOT_KEY,
        )
        out["client_no_key_401"] = no_key_401
        out["client_api_key_roundtrips"] = (
            kc2.vector_store_create(name="k")["object"] == "vector_store"
        )
    return out


def _probe_sdk_twin() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.sdk import Fx1Harness
    from fx1.serve.openai_compat import OpenAICompatError
    from fx1.serve.vectorstores import VectorStoreError

    sdk = Fx1Harness(backend_resolver=lambda *a, **k: _B())
    fid = sdk.openai_file_create(b"sdk doc quasar", purpose="batch", filename="d.jsonl")["id"]
    vs = sdk.vector_store_create(name="sdk", file_ids=[fid])
    vid = str(vs["id"])
    out["sdk_create_mints_envelope"] = (
        vid.startswith("vs_") and vs["object"] == "vector_store" and set(vs) == _VS_KEYS
    )
    out["sdk_get_matches"] = sdk.vector_store_get(vid)["id"] == vid
    out["sdk_update_replaces"] = sdk.vector_store_update(vid, name="sdk2")["name"] == "sdk2"
    out["sdk_file_list"] = [f["id"] for f in sdk.vector_store_file_list(vid)["data"]] == [fid]
    frec = sdk.vector_store_file_get(vid, fid)
    out["sdk_file_get_shape"] = frec["status"] == "completed" and set(frec) == _VS_FILE_KEYS
    fcontent = sdk.vector_store_file_content(vid, fid)
    out["sdk_file_content_page"] = (
        fcontent["object"] == "vector_store.file_content.page"
        and "quasar" in fcontent["data"][0]["text"]
    )
    # the SDK twin answers the same search_results page the route does
    hits = sdk.vector_store_search(vid, "quasar")
    out["sdk_search_hits"] = (
        hits["object"] == "vector_store.search_results.page"
        and len(hits["data"]) == 1
        and hits["data"][0]["file_id"] == fid
        and hits["data"][0]["score"] > 0
    )
    fid2 = sdk.openai_file_create(b"sdk second", purpose="batch", filename="e.jsonl")["id"]
    batch = sdk.vector_store_file_batch_create(vid, [fid2])
    out["sdk_batch_completed"] = (
        batch["object"] == "vector_store.files_batch" and batch["file_counts"]["completed"] == 1
    )
    out["sdk_batch_get"] = sdk.vector_store_file_batch_get(vid, batch["id"])["id"] == batch["id"]
    out["sdk_batch_files_frozen"] = (
        len(sdk.vector_store_file_batch_files(vid, batch["id"])["data"]) == 1
    )
    # SDK error mapping is measured, not assumed: some wrappers
    # translate VectorStoreError → OpenAICompatError, others let it
    # propagate — both carry .status/.code
    try:
        sdk.vector_store_file_batch_cancel(vid, batch["id"])
        cancel_409 = False
    except (OpenAICompatError, VectorStoreError) as exc:
        cancel_409 = exc.status == 409 and exc.code == "file_batch_terminal"
    out["sdk_batch_cancel_409"] = cancel_409
    try:
        sdk.vector_store_get("vs_ghost")
        ghost_404 = False
    except (OpenAICompatError, VectorStoreError) as exc:
        ghost_404 = exc.status == 404 and exc.code == "vector_store_not_found"
    out["sdk_ghost_raises_vectorstore_error"] = ghost_404
    try:
        sdk.vector_store_file_create(vid, fid)
        dup_409 = False
    except (OpenAICompatError, VectorStoreError) as exc:
        dup_409 = exc.status == 409
    out["sdk_dup_attach_409"] = dup_409
    out["sdk_detach"] = sdk.vector_store_file_delete(vid, fid)["deleted"] is True
    out["sdk_delete"] = sdk.vector_store_delete(vid)["deleted"] is True
    out["sdk_list_after_delete_empty"] = all(
        v["id"] != vid for v in sdk.vector_store_list()["data"]
    )
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def vs_audit() -> dict[str, Any]:
    """Run every probe against live in-process apps; literal bools out."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    out: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory() as td, ExitStack() as stack:
            wd = Path(td)
            c = stack.enter_context(_client())
            fresh = stack.enter_context(_client())
            out.update(_probe_create(c))
            out.update(_probe_update(c))
            out.update(_probe_attach(c))
            out.update(_probe_detach(c))
            out.update(_probe_files_list(c))
            out.update(_probe_file_batches(c))
            out.update(_probe_search(c))
            out.update(_probe_chunking(c))
            out.update(_probe_expiry(c))
            out.update(_probe_delete(c))
            out.update(_probe_list(c, fresh))
            out.update(_probe_store_cap(stack))
            out.update(_probe_scope_auth(stack))
            out.update(_probe_drain(stack))
            out.update(_probe_drain_lifecycle(stack))
            out.update(_probe_idem(c))
            out.update(_probe_durability(wd))
            out.update(_probe_concurrency(c))
            out.update(_probe_metering(stack))
            out.update(_probe_envelope(c))
            out.update(_probe_client_map(c, stack))
            out.update(_probe_sdk_twin())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def vs_audit_bench(
    measured: Mapping[str, bool] | None = None,
    *,
    revision: str | None = None,
) -> dict[str, Any]:
    """Sealed receipt: contract probes True, divergences named."""
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
    from quant_fund.utils.reproducibility import git_revision

    r = dict(measured) if measured is not None else vs_audit()
    expected_probes = 277
    ok = len(r) == expected_probes and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    if len(r) != expected_probes:
        defects.append(f"probe_count:{len(r)}!={expected_probes}")
    out: dict[str, Any] = {
        "kind": "vs_audit",
        "schema": "vs_audit.v1",
        "git_revision": revision or git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "interpretation": (
            "The /v1/vector_stores surface holds end to end: create "
            "validates and attaches initial file_ids all-or-nothing with "
            "no partial store behind a refusal, update replaces "
            "name/metadata wholesale, attach indexes synchronously and "
            "lands only completed/failed records with honest "
            "indexed_chunks/truncated/last_error, detach drops the "
            "membership and its chunks while the file-* record survives, "
            "file_batches count per-file verdicts honestly and freeze "
            "them against later detaches, cancel refuses with the "
            "terminal status instead of faking a mid-flight window, "
            "search runs the deterministic lexical scorer with real "
            "filters/ranking bounds, chunking honors auto and static "
            "window contracts, expiry is lazy per-read with writes and "
            "search refusing 410 while update can revive, deletes "
            "tombstone every subroute at once, lists honor the shared "
            "cursor/order contract through invalid_cursor, the store "
            "cap LRU-evicts honestly, scope and workspace rules hold "
            "across credentials, drain refuses mutations but keeps "
            "reads/search/lifecycle-removals open, Idempotency-Key "
            "dedupes the three mints under per-route and per-store "
            "namespaces, journaled identity events replay with the index "
            "rebuilt from bytes (missing bytes fail closed), concurrent "
            "membership writers serialize without torn state, metering "
            "bills every authenticated call, and every refusal lands in "
            "the OpenAI error envelope."
            if ok
            else f"VECTOR STORE AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(vs_audit_bench(), indent=2, sort_keys=True))
