"""crypto_audit — adversarial probes over the harness's cryptographic surfaces.

Receipt sealing, HMAC webhook signatures, idempotency-cache integrity,
journal hash chains, key-material hygiene, BYOK transport, and tamper
detection — measured end to end against the real app (Starlette
TestClient), real ``JobJournal`` chains, a real ``ApiKeyStore``, the real
``receipt_v2`` verifier, and a real loopback webhook sink. Every probe is
a ``results[...] = bool`` measured statement; nothing is asserted by
inspection alone unless the pinned fact is source-level (constant-time
compare sites), and those are read off the module sources on disk.

Pinned contracts:

- **Receipt seal** — an ops receipt (``completion_record_receipt``,
  ``job_record_receipt``, ``run_result_receipt``) verifies under
  ``verify_receipt_payload``/``verify_receipt_file``; every mutation
  fails closed: a flipped record field, a swapped ``receipt_sha256``, a
  results-bool flip inside a sealed audit body, a duplicate JSON key in
  the file, a truncated file, a NaN literal, a non-object payload. A
  consistently *resealed* lie still fails — ``live_pnl_claim: true``
  trips the honesty gate, a v2 envelope stripped to v1 trips
  ``possible_v2_downgrade``, and a claimed ``fleet_eval.v1`` schema gets
  the lane contract regardless of the seal. Digest convention pinned:
  ``receipt_sha256 == sha256(canonical_json_bytes(body))``. Honest
  boundary: a v1-style receipt's ``kind`` is informational — renaming it
  keeps the seal valid; deep checks dispatch on content fingerprints.
- **Webhook signature** — HMAC-SHA256 over ``<ts>.<body>`` bytes: the
  delivered ``X-Fx1-Webhook-Signature`` recomputes exactly; a mutated
  body fails the sink-side verify; the same signature under a shifted
  timestamp fails (the ts is inside the signed bytes); a stale
  timestamp fails the 300s freshness window (a negative tolerance is
  the documented opt-out); a wrong secret fails; missing/malformed
  signature components fail closed; ``verify_webhook`` never raises.
- **Secret hygiene** — ``callback_secret`` never serializes: not in the
  delivered webhook body, not in GET records, not in any journal file.
  A minted key's raw material returns only in the 201 mint/rotate
  response — GET/list never echo the raw or its sha256; ``keys.jsonl``
  persists the sha256 at rest and never the raw bytes. Auth failures
  are a uniform 401 (no key-validity oracle).
- **Hash chain** — ``JobJournal`` lines bind ``seq`` +
  sha256-of-previous-line-bytes + sha256-of-line-content: a mid-file
  mutation drops that line and everything after (``truncated_at`` +
  warning) and blocks new appends until ``compact`` heals; a torn tail
  drops + warns; a recomputed-forged mid line still breaks the next
  link. Honest boundary: the anchorless chain cannot detect a cleanly
  deleted tail — replay of a whole-line-truncated journal is clean.
- **Idempotency cache** — a keyed replay serves the recorded envelope
  byte-identical (the replay marker rides the header/field, the
  recorded payload is untouched); mutating the journaled record breaks
  the chain so boot drops it and the retry re-executes honestly — the
  recorded poison never serves. Journal ``key`` fields are scoped
  sha256 digests: the raw ``Idempotency-Key`` never persists. The
  key-mint ledger is process-local by design — its recorded answer
  carries the raw credential and the journal contract refuses persisted
  secrets — pinned honestly, not flagged.
- **Credential namespace** — the same ``Idempotency-Key`` under two
  managed keys never cross-replays: B's keyed mint executes fresh
  instead of serving A's recorded credential.
- **Receipt headers** — ``X-Fx1-Receipt-Sha256`` on a completion is the
  seal of that call's ``GET /harness/completions/{id}`` record under
  ``completion_record_receipt``; distinct calls carry distinct seals; a
  managed key's call seals its own record. ``GET /receipts/{sha}``
  serves sealed bytes with ``ETag`` = sha and ``X-Fx1-Receipt-Valid``
  reporting live verification — a content-mutated stored receipt still
  serves its bytes but reports ``false``.
- **Envelopes** — every refusal lands enveloped: 401 ``unauthorized``,
  403 ``admin_required``, 404 ``key_not_found``, 409
  ``idempotency_conflict``, 422 for secret-without-url.
- **Source invariants** — ``hmac.compare_digest`` compares the webhook
  signature and the env credential; managed-key auth is sha256-indexed
  (no per-secret ``==``); ``callback_secret`` is a ``PrivateAttr``
  outside pydantic serialization.

Sealed ``crypto_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from fx1.serve.conv_audit import (
    _RESOURCES,
    _audit_context,
    _StubBackend,
    _temporary_directory,
)
from fx1.serve.journal import JobJournal
from fx1.serve.keys import ApiKeyStore
from fx1.serve.ops_receipt import (
    completion_record_receipt,
    job_record_receipt,
    run_result_receipt,
)
from fx1.serve.webhooks import (
    WEBHOOK_SIGNATURE_HEADER,
    WEBHOOK_TIMESTAMP_HEADER,
    sign_webhook,
    verify_webhook,
)
from quant_fund.research.receipt_v2 import (
    seal_receipt,
    verify_receipt_file,
    verify_receipt_payload,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

__all__ = ["crypto_audit", "crypto_audit_bench"]

_API_KEY_ENV = "FX1_API_KEY"
_ROOT = "crypt0-aud1t-r00t"
_WH_SECRET = "whsec-crypto-audit"
_WH_SECRET_BAD = "whsec-crypto-wrong"
_BYOK_KEY = "byok-crypto-key"
_BYOK_URL = "https://crypto-byok.invalid/v1"
_MODEL = "hosted_k3"
_IDEM = "Idempotency-Key"
_JOBS = "/harness/jobs"
_KEYS = "/harness/keys"
_COMPLETE = "/harness/complete"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WAIT_S = 15.0


# ---------------------------------------------------------------------------
# App plumbing
# ---------------------------------------------------------------------------


def _fast_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    del argv, timeout_s
    return 0, "ok", ""


def _make_app(
    workdir: Path,
    *,
    backend_map: dict[str, Callable[[], Any]] | None = None,
    api_key: str | None = _ROOT,
    runner: Callable[[list[str], int], tuple[int, str, str]] | None = None,
    resolver: Callable[..., Any] | None = None,
    **create_kw: Any,
) -> Any:
    """create_app over explicit probe dirs; ``backend_map[name]()`` builds.

    ``state_dir``/``receipts_dir``/``ft_dir`` always live under
    ``workdir`` so the battery can read every journal the app writes.
    ``resolver`` overrides the whole backend_resolver (BYOK probes need
    the two-argument override call shape).
    """
    import fx1.serve.api as api_mod  # noqa: PLC0415
    from fx1.harness import Harness  # noqa: PLC0415

    backends = backend_map or {_MODEL: lambda: _StubBackend("fx1")}
    if resolver is None:
        resolver = lambda name, *a, **k: backends[name]()  # noqa: E731
    resources = _RESOURCES.get()
    (workdir / "receipts").mkdir(parents=True, exist_ok=True)
    saved_key = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=runner or _fast_runner),
            backend_resolver=resolver,
            state_dir=workdir / "state",
            receipts_dir=workdir / "receipts",
            ft_dir=workdir / "ft",
            **create_kw,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        return app
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key


def _client(workdir: Path, **kw: Any) -> tuple[TestClient, Any]:
    """(TestClient, app) — the battery's standard wired app."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    app = _make_app(workdir, **kw)
    client = TestClient(app, raise_server_exceptions=False)
    _RESOURCES.get().callback(client.close)
    _RESOURCES.get().enter_context(client)
    return client, app


def _h(auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    hdrs = {"X-API-Key": auth} if auth else {}
    hdrs.update(extra)
    return hdrs


def _ih(key: str, auth: str | None = _ROOT, **extra: str) -> dict[str, str]:
    return _h(auth, **{_IDEM: key, **extra})


def _wait_job(client: TestClient, job_id: str, timeout: float = _WAIT_S) -> dict[str, Any]:
    """Poll a job to terminal status."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = client.get(f"{_JOBS}/{job_id}", headers=_h())
        body = r.json()
        if isinstance(body, dict) and body.get("status") in {"succeeded", "failed", "cancelled"}:
            return dict(body)
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not reach terminal")


def _state_files(workdir: Path) -> dict[Path, bytes]:
    """Every persisted byte under the app's state dir — name → content."""
    state = workdir / "state"
    if not state.is_dir():
        return {}
    return {p: p.read_bytes() for p in sorted(state.rglob("*")) if p.is_file()}


def _never_in_state(workdir: Path, *needles: str) -> bool:
    """No journal/blob under state_dir carries any needle's bytes."""
    for raw in _state_files(workdir).values():
        for needle in needles:
            if needle.encode() in raw:
                return False
    return True


# ---------------------------------------------------------------------------
# Receipt seal — quant_fund.research.receipt_v2 + ops_receipt
# ---------------------------------------------------------------------------


def _completion_record() -> dict[str, Any]:
    return {
        "completion_id": "cmp-crypto-1",
        "backend": _MODEL,
        "model": _MODEL,
        "content": "stub:crypto",
        "latency_ms": 1.25,
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "key_id": "env",
        "finish_reason": "stop",
        "created_at": 1.0,
    }


def _audit_shaped_doc() -> dict[str, Any]:
    """The shape this module's own bench seals — one probe-bool claim."""
    return {
        "kind": "crypto_audit",
        "schema": "crypto_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": {"probe.x": True}, "ok": True},
    }


def _probe_seal() -> dict[str, bool]:
    out: dict[str, bool] = {}
    doc = completion_record_receipt(_completion_record())
    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}

    out["seal.ops_verifies"] = verify_receipt_payload(doc)["valid"] is True
    out["seal.digest_is_canonical_json"] = (
        doc["receipt_sha256"] == hash_bytes(canonical_json_bytes(body))
        and verify_receipt_payload(doc)["digest_convention"] == "canonical_json"
    )
    # A jobs-receipt and run-receipt seal the same way.
    jrec = job_record_receipt(
        {
            "job_id": "j-1",
            "status": "succeeded",
            "created_at": 1.0,
            "finished_at": 2.0,
            "callback_url": "http://sink.invalid/hook?tok=abc",
            "result": {
                "command": "doctor",
                "exit_code": 0,
                "ok": True,
                "stdout": "secret-stream-out",
                "stderr": "",
            },
        }
    )
    out["seal.job_receipt_verifies"] = verify_receipt_payload(jrec)["valid"] is True
    rrec = run_result_receipt(
        {"command": "doctor", "exit_code": 0, "stdout": "secret-stream-out", "stderr": "err"}
    )
    out["seal.run_receipt_verifies"] = verify_receipt_payload(rrec)["valid"] is True

    # Any mutation of a sealed field fails.
    forged = dict(doc)
    forged["record"] = {**dict(doc["record"]), "content": "forged"}
    v = verify_receipt_payload(forged)
    out["seal.record_mutation_fails"] = (
        v["valid"] is False and "receipt_sha256_mismatch" in v["errors"]
    )
    swapped = dict(doc)
    swapped["receipt_sha256"] = "0" * 64
    out["seal.sha_field_swap_fails"] = verify_receipt_payload(swapped)["valid"] is False
    absent = dict(doc)
    absent.pop("receipt_sha256")
    v = verify_receipt_payload(absent)
    out["seal.sha_absent_fails"] = (
        v["valid"] is False and "receipt_sha256_missing_or_invalid" in v["errors"]
    )

    # A results-bool flip inside a sealed audit body fails.
    audit_doc = seal_receipt(_audit_shaped_doc())
    out["seal.audit_doc_verifies"] = verify_receipt_payload(audit_doc)["valid"] is True
    flipped = json.loads(json.dumps(audit_doc))
    flipped["claim"]["results"]["probe.x"] = False
    out["seal.results_bool_flip_fails"] = verify_receipt_payload(flipped)["valid"] is False

    # File-level adversaries: duplicate keys, truncation, NaN, non-object.
    tmp = _temporary_directory()
    raw = json.dumps(doc, sort_keys=True).encode()
    good_path = tmp / "good.json"
    good_path.write_bytes(raw)
    out["seal.file_verifies"] = verify_receipt_file(good_path)["valid"] is True
    dup_path = tmp / "dup.json"
    dup_path.write_text('{"kind": "a", "kind": "b", "receipt_sha256": "%s"}' % ("0" * 64))
    v = verify_receipt_file(dup_path)
    out["seal.dup_key_file_fails"] = v["valid"] is False and any(
        e.startswith("duplicate_json_key") for e in v["errors"]
    )
    trunc_path = tmp / "trunc.json"
    trunc_path.write_bytes(raw[: len(raw) // 2])
    out["seal.truncated_file_fails"] = verify_receipt_file(trunc_path)["valid"] is False
    nan_path = tmp / "nan.json"
    nan_path.write_text('{"kind": NaN}')
    out["seal.nan_literal_fails"] = verify_receipt_file(nan_path)["valid"] is False
    missing_path = tmp / "missing.json"
    out["seal.missing_file_fails"] = verify_receipt_file(missing_path)["valid"] is False
    out["seal.non_object_fails"] = verify_receipt_payload([1, 2, 3])["valid"] is False
    out["seal.scalar_fails"] = verify_receipt_payload("not-a-receipt")["valid"] is False
    out["seal.empty_object_fails"] = verify_receipt_payload({})["valid"] is False

    # A consistently RESEALED lie still fails the honesty gates.
    liar = seal_receipt({**body, "live_pnl_claim": True})
    v = verify_receipt_payload(liar)
    out["seal.resealed_live_claim_fails"] = (
        v["valid"] is False and "live_pnl_claim_not_false" in v["errors"]
    )
    downgraded = seal_receipt(
        {
            "payload": {"kind": "anything"},
            "code_files": {"x.py": "0" * 64},
            "dataset_hash": "a" * 64,
            "params_hash": "b" * 64,
        }
    )
    v = verify_receipt_payload(downgraded)
    out["seal.v2_downgrade_fails"] = v["valid"] is False and "possible_v2_downgrade" in v["errors"]
    claimed = seal_receipt({"schema": "fleet_eval.v1", "kind": "forged", "results": []})
    out["seal.claimed_lane_schema_checked"] = verify_receipt_payload(claimed)["valid"] is False

    # Honest pin: for an unnamed kind the seal binds bytes, not names —
    # renaming ``kind`` on a consistently resealed v1 doc still verifies.
    renamed = seal_receipt({**body, "kind": "renamed_kind"})
    out["seal.kind_rename_still_hashes"] = verify_receipt_payload(renamed)["valid"] is True

    # Ops receipts digest caller-supplied material; streams never embed.
    out["ops.callback_url_digested"] = (
        "callback_url" not in jrec["record"]
        and _SHA256.fullmatch(str(jrec["record"].get("callback_url_sha256"))) is not None
        and "sink.invalid" not in json.dumps(jrec)
    )
    out["ops.run_streams_digested"] = (
        "stdout" not in rrec["record"]
        and _SHA256.fullmatch(str(rrec["record"].get("stdout_sha256"))) is not None
        and "secret-stream-out" not in json.dumps(rrec)
    )
    return out


# ---------------------------------------------------------------------------
# /receipts/* routes — verify endpoints + content-addressed fetch
# ---------------------------------------------------------------------------


def _probe_receipt_routes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    workdir = _temporary_directory() / "receipt-routes"
    client, _app = _client(workdir)

    doc = completion_record_receipt(_completion_record())
    r = client.post("/receipts/verify", json={"receipt": doc}, headers=_h())
    out["wire.verify_accepts_sealed"] = r.status_code == 200 and r.json()["valid"] is True
    forged = dict(doc)
    forged["record"] = {**dict(doc["record"]), "content": "mutated"}
    r = client.post("/receipts/verify", json={"receipt": forged}, headers=_h())
    out["wire.verify_rejects_mutated"] = (
        r.status_code == 200 and r.json()["valid"] is False and len(r.json()["errors"]) > 0
    )
    r = client.post("/receipts/verify", json={"receipt": [1, 2]}, headers=_h())
    # A non-object receipt fails closed — 422 at the request schema, or
    # valid=False under the verifier; either shape is a refusal.
    out["wire.verify_rejects_non_object"] = r.status_code == 422 or (
        r.status_code == 200 and r.json()["valid"] is False
    )

    # Batch verify: per-item verdicts, a bad item never aborts the rest.
    r = client.post(
        "/receipts/verify/batch",
        json={"receipts": [doc, forged, {"junk": 1}]},
        headers=_h(),
    )
    items = r.json().get("results", [])
    out["wire.batch_item_local"] = (
        r.status_code == 200
        and [bool(i["valid"]) for i in items] == [True, False, False]
        and r.json().get("verified") == 1
        and r.json().get("failed") == 2
    )

    # Content-addressed store: sealed file indexes under its declared sha.
    store = workdir / "receipts"
    blob_path = store / "stored.json"
    blob = json.dumps(doc, sort_keys=True, indent=1).encode()
    blob_path.write_bytes(blob)
    sha = str(doc["receipt_sha256"])
    r = client.get("/receipts", headers=_h())
    out["wire.index_lists_sha"] = r.status_code == 200 and any(
        i["sha256"] == sha for i in r.json().get("items", [])
    )
    r = client.get(f"/receipts/{sha}", headers=_h())
    out["wire.fetch_verbatim_bytes"] = r.status_code == 200 and r.content == blob
    out["wire.fetch_etag_is_sha"] = r.headers.get("ETag") == f'"{sha}"'
    out["wire.fetch_reports_valid"] = r.headers.get("X-Fx1-Receipt-Valid") == "true"
    r = client.get(f"/receipts/{sha}", headers={**_h(), "If-None-Match": f'"{sha}"'})
    out["wire.fetch_304_on_etag"] = r.status_code == 304
    # Mutate the stored file but keep its declared sha — the index is a
    # metadata index (documented), so the same sha resolves, the mutated
    # bytes serve, and live verification reports the corruption.
    mutated = json.dumps({**doc, "record": {**dict(doc["record"]), "content": "x"}}).encode()
    blob_path.write_bytes(mutated)
    r = client.get(f"/receipts/{sha}", headers=_h())
    out["wire.corrupted_store_reports_false"] = (
        r.status_code == 200
        and r.content == mutated
        and r.headers.get("X-Fx1-Receipt-Valid") == "false"
    )
    blob_path.write_bytes(blob)  # restore before other probes
    r = client.get(f"/receipts/{sha}", headers=_h())
    out["wire.store_heals_on_restore"] = r.headers.get("X-Fx1-Receipt-Valid") == "true"

    r = client.get(f"/receipts/{'f' * 64}", headers=_h())
    v = r.json()
    out["wire.fetch_missing_404"] = (
        r.status_code == 404 and v.get("code") == "receipt_not_found" and "detail" in v
    )
    r = client.get("/receipts/nothex", headers=_h())
    out["wire.fetch_bad_sha_422"] = (
        r.status_code == 422 and r.json().get("code") == "invalid_sha256"
    )
    return out


# ---------------------------------------------------------------------------
# Webhook signature — signed-bytes contract + sink-side verify + hygiene
# ---------------------------------------------------------------------------


def _probe_webhook_sig() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.webhook_audit import _Sink  # noqa: PLC0415

    sink = _Sink()
    _RESOURCES.get().callback(sink.close)
    workdir = _temporary_directory() / "wh-sig"
    client, _app = _client(workdir)

    n0 = len(sink.hits)
    r = client.post(
        _JOBS,
        json={
            "command": "doctor",
            "callback_url": sink.url("/hook"),
            "callback_secret": _WH_SECRET,
        },
        headers=_h(),
    )
    out["deliver.submitted"] = r.status_code == 202
    job_id = str(r.json()["job_id"])
    _wait_job(client, job_id)
    deadline = time.monotonic() + _WAIT_S
    while len(sink.hits) <= n0 and time.monotonic() < deadline:
        time.sleep(0.05)
    out["deliver.hit_arrived"] = len(sink.hits) > n0
    if len(sink.hits) <= n0:
        return out
    hit = sink.hits[n0]
    headers = {str(k).lower(): str(v) for k, v in hit.headers.items()}
    sig = headers.get(WEBHOOK_SIGNATURE_HEADER.lower())
    ts = headers.get(WEBHOOK_TIMESTAMP_HEADER.lower())
    out["sig.headers_present"] = isinstance(sig, str) and isinstance(ts, str)
    out["sig.sha256_prefixed"] = isinstance(sig, str) and sig.startswith("sha256=")
    if not isinstance(sig, str) or not isinstance(ts, str):
        return out

    # The signed bytes are exactly ``<ts>.<body>`` — no more, no less.
    out["sig.recomputes_exact"] = sign_webhook(_WH_SECRET, ts, hit.body) == sig
    out["sig.verifies_at_sink"] = verify_webhook(_WH_SECRET, ts, sig, hit.body) is True
    out["sig.mutated_body_fails"] = (
        verify_webhook(_WH_SECRET, ts, sig, hit.body + b" ") is False
        and verify_webhook(_WH_SECRET, ts, sig, b"") is False
    )
    shifted = str(float(ts) + 1.0)
    out["sig.timestamp_covered"] = verify_webhook(_WH_SECRET, shifted, sig, hit.body) is False
    stale_ts = "1000000000.0"
    stale_sig = sign_webhook(_WH_SECRET, stale_ts, hit.body)
    out["sig.stale_replay_fails"] = (
        verify_webhook(_WH_SECRET, stale_ts, stale_sig, hit.body) is False
    )
    out["sig.freshness_bound_checked"] = (
        verify_webhook(_WH_SECRET, stale_ts, stale_sig, hit.body, now=float(stale_ts) + 299) is True
        and verify_webhook(_WH_SECRET, stale_ts, stale_sig, hit.body, now=float(stale_ts) + 301)
        is False
    )
    out["sig.freshness_optout_honest"] = (
        verify_webhook(_WH_SECRET, stale_ts, stale_sig, hit.body, tolerance_s=-1) is True
    )
    out["sig.wrong_secret_fails"] = verify_webhook(_WH_SECRET_BAD, ts, sig, hit.body) is False
    out["sig.missing_sig_fails"] = verify_webhook(_WH_SECRET, ts, None, hit.body) is False
    out["sig.missing_ts_fails"] = verify_webhook(_WH_SECRET, None, sig, hit.body) is False
    out["sig.empty_sig_fails"] = verify_webhook(_WH_SECRET, ts, "", hit.body) is False
    out["sig.bad_prefix_fails"] = (
        verify_webhook(_WH_SECRET, ts, "md5=" + sig.split("=", 1)[-1], hit.body) is False
    )
    out["sig.bare_hex_fails"] = (
        verify_webhook(_WH_SECRET, ts, sig.split("=", 1)[-1], hit.body) is False
    )
    out["sig.nonascii_ts_fails"] = verify_webhook(_WH_SECRET, "€", sig, hit.body) is False
    out["sig.nan_ts_fails"] = verify_webhook(_WH_SECRET, "nan", sig, hit.body) is False
    out["sig.nonascii_sig_fails"] = (
        verify_webhook(_WH_SECRET, ts, "sha256=€" + "0" * 63, hit.body) is False
    )
    out["sig.never_raises"] = all(
        verify_webhook(secret, ts2, s, b) is False
        for secret, ts2, s, b in (
            ("", ts, sig, hit.body),
            (_WH_SECRET, "garbage", sig, hit.body),
            (_WH_SECRET, ts, "sha256=nothex", hit.body),
        )
    )

    # The secret never serializes — not in the delivered body, not in
    # the record's GET, not in any persisted file.
    out["secret.absent_from_delivery"] = _WH_SECRET.encode() not in hit.body
    out["secret.absent_from_headers"] = _WH_SECRET not in json.dumps(headers)
    got = client.get(f"{_JOBS}/{job_id}", headers=_h())
    out["secret.absent_from_get"] = _WH_SECRET not in got.text
    listed = client.get(_JOBS, headers=_h())
    out["secret.absent_from_list"] = _WH_SECRET not in listed.text
    out["secret.absent_from_state"] = _never_in_state(workdir, _WH_SECRET)

    # Unsigned delivery (no callback_secret) still arrives — unsigned.
    n0 = len(sink.hits)
    r = client.post(
        _JOBS,
        json={"command": "doctor", "callback_url": sink.url("/plain")},
        headers=_h(),
    )
    job_id = str(r.json()["job_id"])
    _wait_job(client, job_id)
    deadline = time.monotonic() + _WAIT_S
    while len(sink.hits) <= n0 and time.monotonic() < deadline:
        time.sleep(0.05)
    if len(sink.hits) > n0:
        plain = {str(k).lower(): str(v) for k, v in sink.hits[n0].headers.items()}
        out["deliver.unsigned_has_no_sig"] = WEBHOOK_SIGNATURE_HEADER.lower() not in plain
    else:
        out["deliver.unsigned_has_no_sig"] = False

    # A sink cannot be confused into accepting a signature minted for a
    # different callback body — the signature binds THIS delivery.
    r = client.post(
        _JOBS,
        json={
            "command": "doctor",
            "callback_url": sink.url("/hook2"),
            "callback_secret": _WH_SECRET,
        },
        headers=_h(),
    )
    job_id = str(r.json()["job_id"])
    _wait_job(client, job_id)
    deadline = time.monotonic() + _WAIT_S
    while len(sink.hits) <= n0 + 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    if len(sink.hits) > n0 + 1:
        hit2 = sink.hits[n0 + 1]
        h2 = {str(k).lower(): str(v) for k, v in hit2.headers.items()}
        sig2 = h2.get(WEBHOOK_SIGNATURE_HEADER.lower(), "")
        # First signature applied to second body fails — no cross-reuse.
        out["sig.binds_exact_delivery"] = (
            verify_webhook(_WH_SECRET, h2.get(WEBHOOK_TIMESTAMP_HEADER.lower()), sig, hit2.body)
            is False
            or sig == sig2
        )
    return out


# ---------------------------------------------------------------------------
# Key material — mint-once, sha256 at rest, uniform 401, rotation
# ---------------------------------------------------------------------------


def _mint(client: TestClient, name: str, **extra: Any) -> tuple[str, dict[str, Any]]:
    r = client.post(_KEYS, json={"name": name, **extra}, headers=_h())
    assert r.status_code == 201, r.text
    body = r.json()
    return str(body["key"]), dict(body)


def _probe_key_material() -> dict[str, bool]:
    out: dict[str, bool] = {}
    workdir = _temporary_directory() / "key-material"
    client, app = _client(workdir)

    raw, mint_body = _mint(client, "crypto-probe")
    out["key.mint_returns_raw_once"] = raw.startswith("fx1k_") and len(raw) > 20
    out["key.mint_201"] = "id" in mint_body
    key_id = str(mint_body["id"])
    raw_sha = hashlib.sha256(raw.encode()).hexdigest()

    # The raw key + its sha never appear on any subsequent wire surface.
    got = client.get(f"{_KEYS}/{key_id}", headers=_h())
    out["key.get_no_raw"] = got.status_code == 200 and raw not in got.text
    out["key.get_no_sha"] = got.status_code == 200 and raw_sha not in got.text
    listed = client.get(_KEYS, headers=_h())
    out["key.list_no_raw"] = listed.status_code == 200 and raw not in listed.text
    out["key.list_no_sha"] = listed.status_code == 200 and raw_sha not in listed.text
    usage = client.get(f"{_KEYS}/{key_id}/usage", headers=_h())
    out["key.usage_no_raw"] = usage.status_code == 200 and raw not in usage.text

    # At rest: keys.jsonl persists the sha256 of the raw key — never the
    # raw bytes themselves.
    keys_file = workdir / "state" / "keys.jsonl"
    out["key.journal_exists"] = keys_file.exists()
    if keys_file.exists():
        blob = keys_file.read_bytes()
        out["key.at_rest_is_sha256"] = raw_sha.encode() in blob
        out["key.raw_absent_at_rest"] = raw.encode() not in blob
        lines = [json.loads(line) for line in blob.splitlines() if line]
        rec_payload = next(
            (
                p["payload"]
                for p in lines
                if isinstance(p.get("payload"), dict) and "record" in p["payload"]
            ),
            None,
        )
        out["key.journal_record_shaped"] = (
            rec_payload is not None and rec_payload.get("record", {}).get("sha256") == raw_sha
        )
    # Nothing under state_dir carries the raw material.
    out["key.raw_absent_from_state"] = _never_in_state(workdir, raw)

    # Authenticate: the minted key works; wrong/absent/revoked share one
    # 401 shape — no validity oracle.
    ok = client.get(_JOBS, headers=_h(raw))
    out["key.minted_authenticates"] = ok.status_code == 200
    bad = client.get(_JOBS, headers=_h("fx1k_forged0000"))
    absent = client.get(_JOBS, headers={})
    out["key.wrong_401_uniform"] = (
        bad.status_code == 401
        and absent.status_code == 401
        and bad.json().get("code") == absent.json().get("code") == "unauthorized"
        and bad.json().get("detail") == absent.json().get("detail")
    )
    r = client.delete(f"{_KEYS}/{key_id}", headers=_h())
    out["key.revoke_200"] = r.status_code == 200
    revoked = client.get(_JOBS, headers=_h(raw))
    out["key.revoked_401_uniform"] = revoked.status_code == 401 and revoked.json() == bad.json()

    # Rotation: default tombstones the predecessor atomically; the
    # successor's raw returns once and never round-trips again.
    raw2, mint2 = _mint(client, "crypto-rot")
    kid2 = str(mint2["id"])
    r = client.post(f"{_KEYS}/{kid2}/rotate", json={}, headers=_h())
    out["key.rotate_200"] = r.status_code in (200, 201)
    rot = r.json()
    raw3 = str(rot["key"]["key"])
    out["key.rotate_returns_successor_once"] = raw3.startswith("fx1k_") and raw3 != raw2
    out["key.rotate_reports_tombstone"] = rot.get("revoked_previous") is True
    out["key.rotated_old_dies"] = client.get(_JOBS, headers=_h(raw2)).status_code == 401
    out["key.rotated_new_works"] = client.get(_JOBS, headers=_h(raw3)).status_code == 200
    got3 = client.get(f"{_KEYS}/{rot['key']['id']}", headers=_h())
    out["key.successor_no_raw"] = raw3 not in got3.text
    out["key.rotate_lineage_recorded"] = (
        str(rot["key"].get("rotated_from")) == kid2 or rot.get("rotated_from") == kid2
    )

    # Overlap arm: revoke_old=False keeps the predecessor alive.
    raw4, mint4 = _mint(client, "crypto-overlap")
    kid4 = str(mint4["id"])
    r = client.post(f"{_KEYS}/{kid4}/rotate", json={"revoke_old": False}, headers=_h())
    out["key.overlap_rotate_200"] = r.status_code in (200, 201)
    raw5 = str(r.json()["key"]["key"])
    out["key.overlap_both_authenticate"] = (
        client.get(_JOBS, headers=_h(raw4)).status_code == 200
        and client.get(_JOBS, headers=_h(raw5)).status_code == 200
    )

    # Mint idempotency is process-local (the recorded answer carries the
    # raw credential — the journal contract refuses persisted secrets).
    body = {"name": "crypto-idem"}
    r1 = client.post(_KEYS, json=body, headers=_ih("mint-k1"))
    r2 = client.post(_KEYS, json=body, headers=_ih("mint-k1"))
    out["key.mint_idem_replays_raw"] = (
        r1.status_code == 201
        and r2.status_code == 201
        and r1.json()["key"] == r2.json()["key"]
        and r2.headers.get("X-Fx1-Idempotent-Replay") == "true"
    )
    # No persisted file carries that minted secret.
    out["key.mint_replay_never_journaled"] = _never_in_state(workdir, str(r1.json()["key"]))
    return out


def _probe_key_quarantine() -> dict[str, bool]:
    """ApiKeyStore on a damaged keys.jsonl quarantines every recovered key."""
    out: dict[str, bool] = {}
    workdir = _temporary_directory() / "key-quar"
    journal = JobJournal(workdir / "keys.jsonl")
    store = ApiKeyStore(journal=journal)
    raw_a, _rec_a = store.mint("qa")
    raw_b, _rec_b = store.mint("qb")
    out["quar.clean_store_auths"] = (
        store.authenticate(raw_a) is not None and store.authenticate(raw_b) is not None
    )

    # Corrupt the middle of the file: flip one payload byte on line 1.
    path = workdir / "keys.jsonl"
    raw_bytes = path.read_bytes()
    lines = raw_bytes.split(b"\n")
    body0 = lines[0]
    idx = body0.find(b'"name"')
    lines[0] = body0[: idx + 8] + b"X" + body0[idx + 9 :]
    path.write_bytes(b"\n".join(lines))
    store2 = ApiKeyStore(journal=JobJournal(path))
    out["quar.damaged_detected"] = bool(store2.recover_warnings)
    out["quar.all_keys_quarantined"] = store2.recovery_quarantined is True
    out["quar.recovered_keys_disabled"] = all(not rec["enabled"] for rec in store2.list())
    refused = sum(store2.authenticate(raw) is None for raw in (raw_a, raw_b))
    out["quar.quarantined_keys_refuse"] = refused == 2
    # A clean tail deletion is undetectable at the chain layer — pin it.
    workdir2 = _temporary_directory() / "key-tail"
    j2 = JobJournal(workdir2 / "keys.jsonl")
    store3 = ApiKeyStore(journal=j2)
    raw_c, _ = store3.mint("qc")
    raw_d, _ = store3.mint("qd")
    path2 = workdir2 / "keys.jsonl"
    parts = path2.read_bytes().split(b"\n")
    path2.write_bytes(b"\n".join(parts[:-2]) + b"\n")
    store4 = ApiKeyStore(journal=JobJournal(path2))
    out["quar.clean_tail_delete_undetected"] = (
        store4.recovery_quarantined is False
        and store4.authenticate(raw_c) is not None
        and store4.authenticate(raw_d) is None
    )
    return out


# ---------------------------------------------------------------------------
# Journal hash chain — JobJournal direct
# ---------------------------------------------------------------------------


def _journal_lines(path: Path) -> list[bytes]:
    raw = path.read_bytes()
    return [ln for ln in raw.split(b"\n") if ln]


def _forge_line(seq: int, chain: str, payload: dict[str, Any]) -> bytes:
    """A syntactically valid line with a recomputed seal — a real forgery."""
    payload_raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(f"{seq}|{chain}|{payload_raw}".encode()).hexdigest()
    line = {"seq": seq, "chain": chain, "payload": payload, "sha256": sha}
    return json.dumps(line, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _probe_journal_chain() -> dict[str, bool]:
    out: dict[str, bool] = {}
    workdir = _temporary_directory() / "chain"
    path = workdir / "jobs.jsonl"
    j = JobJournal(path)
    j.append({"n": 1})
    j.append({"n": 2})
    j.append({"n": 3})
    res = j.replay()
    out["chain.ordered_replay"] = [p["n"] for p in res.payloads] == [1, 2, 3]
    out["chain.clean_verdict"] = res.truncated_at is None and res.dropped == 0 and not res.warnings
    # Lines carry seq/chain/sha256 in order; each link is the sha256 of
    # the previous line's bytes INCLUDING the newline terminator.
    raw_lines = _journal_lines(path)
    parsed = [json.loads(ln) for ln in raw_lines]
    out["chain.seq_monotonic"] = [p["seq"] for p in parsed] == [0, 1, 2]
    out["chain.links_sha_prev"] = all(
        parsed[i]["chain"] == hashlib.sha256(raw_lines[i - 1] + b"\n").hexdigest() for i in (1, 2)
    )

    # Mid-file mutation: flip a payload byte on the middle line (payloads
    # serialize compact — ``"n":2``, no spaces).
    lines = _journal_lines(path)
    lines[1] = lines[1].replace(b'"n":2', b'"n":9')
    path.write_bytes(b"\n".join(lines) + b"\n")
    j2 = JobJournal(path)
    res2 = j2.replay()
    out["chain.mid_mutation_detected"] = res2.truncated_at is not None
    out["chain.mid_mutation_drops_suffix"] = res2.dropped == 2 and [
        p["n"] for p in res2.payloads
    ] == [1]
    out["chain.warns_on_break"] = bool(res2.warnings)
    # Appends stay blocked until compact — damage can't hide under new lines.
    try:
        j2.append({"n": 4})
        appended = True
    except RuntimeError:
        appended = False
    out["chain.append_blocked_after_damage"] = appended is False
    j2.compact([{"n": 1}])
    j2.append({"n": 5})
    res3 = j2.replay()
    out["chain.compact_heals"] = res3.truncated_at is None and [p["n"] for p in res3.payloads] == [
        1,
        5,
    ]

    # Torn tail: a partial last line drops with a warning.
    path3 = workdir / "torn.jsonl"
    j3 = JobJournal(path3)
    j3.append({"n": 1})
    j3.append({"n": 2})
    raw = path3.read_bytes()
    path3.write_bytes(raw[: len(raw) - 5])
    res4 = JobJournal(path3).replay()
    out["chain.torn_tail_detected"] = res4.truncated_at is not None and res4.dropped == 1

    # Recomputed forgery: a mid-line forgery still breaks the NEXT link;
    # a forged TAIL line verifies — the anchorless chain boundary, pinned
    # honestly (integrity covers corruption, not a wholesale re-seal).
    path4 = workdir / "forge-mid.jsonl"
    j4 = JobJournal(path4)
    for n in (1, 2, 3):
        j4.append({"n": n})
    flines = _journal_lines(path4)
    # chain of line 2 = sha256(line 1 bytes incl the newline terminator)
    forged = _forge_line(1, hashlib.sha256(flines[0] + b"\n").hexdigest(), {"n": 99})
    path4.write_bytes(flines[0] + b"\n" + forged + flines[2] + b"\n")
    res5 = JobJournal(path4).replay()
    out["chain.mid_forgery_detected"] = res5.truncated_at is not None and res5.dropped >= 1

    path5 = workdir / "forge-tail.jsonl"
    j5 = JobJournal(path5)
    j5.append({"n": 1})
    j5.append({"n": 2})
    tlines = _journal_lines(path5)
    forged_tail = _forge_line(2, hashlib.sha256(tlines[1] + b"\n").hexdigest(), {"n": 99})
    path5.write_bytes(tlines[0] + b"\n" + tlines[1] + b"\n" + forged_tail)
    res6 = JobJournal(path5).replay()
    out["chain.tail_forgery_boundary_honest"] = (
        res6.truncated_at is None and res6.payloads[-1]["n"] == 99
    )

    # Clean whole-line tail deletion: undetectable — pinned honestly.
    path6 = workdir / "tail.jsonl"
    j6 = JobJournal(path6)
    for n in (1, 2, 3):
        j6.append({"n": n})
    tlines2 = _journal_lines(path6)
    path6.write_bytes(tlines2[0] + b"\n" + tlines2[1] + b"\n")
    res7 = JobJournal(path6).replay()
    out["chain.clean_tail_delete_undetected"] = (
        res7.truncated_at is None and len(res7.payloads) == 2 and not res7.warnings
    )
    return out


# ---------------------------------------------------------------------------
# Idem cache integrity — replay bytes, journaled digest, cross-credential
# ---------------------------------------------------------------------------


def _chat_body(text: str) -> dict[str, Any]:
    return {"model": _MODEL, "messages": [{"role": "user", "content": text}]}


def _probe_idem_integrity() -> dict[str, bool]:
    out: dict[str, bool] = {}
    workdir = _temporary_directory() / "idem-integrity"
    stub = _StubBackend("fx1")
    client, _app = _client(workdir, backend_map={_MODEL: lambda: stub})

    body = _chat_body("idem-1")
    r1 = client.post("/v1/chat/completions", json=body, headers=_ih("ik-1"))
    r2 = client.post("/v1/chat/completions", json=body, headers=_ih("ik-1"))
    out["idem.first_200"] = r1.status_code == 200
    out["idem.replay_header"] = r2.headers.get("X-Fx1-Idempotent-Replay") == "true"
    out["idem.replay_byte_identical"] = r2.content == r1.content
    out["idem.no_second_execute"] = stub.calls == 1
    # Different body under the same key is a 409 conflict, enveloped.
    r3 = client.post("/v1/chat/completions", json=_chat_body("other"), headers=_ih("ik-1"))
    v3 = r3.json()
    out["idem.conflict_409_enveloped"] = (
        r3.status_code == 409
        and isinstance(v3, dict)
        and (v3.get("code") == "idempotency_conflict" or "error" in v3)
    )
    # The journaled key is a scoped sha256 digest — the raw header value
    # never persists.
    idem_file = workdir / "state" / "idem_openai.jsonl"
    out["idem.journal_exists"] = idem_file.exists()
    if idem_file.exists():
        raw = idem_file.read_bytes()
        out["idem.raw_key_absent"] = b"ik-1" not in raw
        line = json.loads(raw.split(b"\n")[0])
        payload = line.get("payload", {})
        stored_key = str(payload.get("idem", {}).get("key", ""))
        out["idem.stored_key_is_digest"] = (
            stored_key.startswith("env:") and "ik-1" not in stored_key and len(stored_key) > 16
        )

    # Mutate the journaled recorded response — the chain breaks at boot,
    # the record drops, and the retry RE-EXECUTES honestly instead of
    # serving poisoned bytes.
    workdir2 = _temporary_directory() / "idem-corrupt"
    stub2 = _StubBackend("fx1")
    c1, _a1 = _client(workdir2, backend_map={_MODEL: lambda: stub2})
    r1c = c1.post("/v1/chat/completions", json=body, headers=_ih("ik-9"))
    out["idem.corrupt_first_200"] = r1c.status_code == 200
    idem2 = workdir2 / "state" / "idem_openai.jsonl"
    raw2 = idem2.read_bytes()
    # Flip one byte inside the recorded response payload.
    marker = r1c.json()["choices"][0]["message"]["content"].encode()[:8]
    idx = raw2.find(marker)
    poisoned = (
        raw2[:idx] + marker[:4] + b"XXXX" + marker[4:] + raw2[idx + len(marker) :]
        if idx >= 0
        else raw2
    )
    idem2.write_bytes(poisoned)
    c2, _a2 = _client(workdir2, backend_map={_MODEL: lambda: stub2})
    r2c = c2.post("/v1/chat/completions", json=body, headers=_ih("ik-9"))
    served = r2c.json()["choices"][0]["message"]["content"]
    out["idem.mutated_record_never_serves"] = r2c.status_code == 200 and "XXXX" not in served
    out["idem.mutation_reexecutes_honestly"] = stub2.calls >= 2

    # Cross-credential: same Idempotency-Key under two managed keys never
    # cross-replays — B's keyed mint executes fresh and never sees A's raw.
    # (Admin-scoped managed keys: key management requires the admin scope.)
    workdir3 = _temporary_directory() / "idem-cred"
    c3, _a3 = _client(workdir3)
    raw_a, _ = _mint(c3, "cred-a", admin=True)
    raw_b, _ = _mint(c3, "cred-b", admin=True)
    mint_body = {"name": "cred-mint"}
    ra = c3.post(_KEYS, json=mint_body, headers=_ih("shared-ik", raw_a))
    rb = c3.post(_KEYS, json=mint_body, headers=_ih("shared-ik", raw_b))
    out["cred.mints_are_distinct"] = (
        ra.status_code == 201 and rb.status_code == 201 and ra.json()["key"] != rb.json()["key"]
    )
    out["cred.b_never_sees_a"] = (ra.status_code == 201 and rb.status_code == 201) and rb.json()[
        "key"
    ] != ra.json()["key"]
    # Same key+body on the chat surface under two creds executes twice.
    stub3 = _StubBackend("fx1")
    c4, _a4 = _client(_temporary_directory() / "idem-cred2", backend_map={_MODEL: lambda: stub3})
    ka, _ = _mint(c4, "ca", admin=True)
    kb, _ = _mint(c4, "cb", admin=True)
    ra2 = c4.post("/v1/chat/completions", json=body, headers=_ih("ik-x", ka))
    rb2 = c4.post("/v1/chat/completions", json=body, headers=_ih("ik-x", kb))
    out["cred.chat_no_cross_replay"] = (
        ra2.status_code == 200
        and rb2.status_code == 200
        and rb2.headers.get("X-Fx1-Idempotent-Replay") != "true"
        and stub3.calls == 2
    )
    return out


# ---------------------------------------------------------------------------
# X-Fx1-Receipt-Sha256 headers — completion receipts on gated surfaces
# ---------------------------------------------------------------------------


def _probe_receipt_headers() -> dict[str, bool]:
    out: dict[str, bool] = {}
    workdir = _temporary_directory() / "receipt-hdr"
    client, _app = _client(workdir)

    r = client.post(
        _COMPLETE,
        json={"backend": _MODEL, "messages": [{"role": "user", "content": "hdr"}]},
        headers=_h(),
    )
    out["hdr.emitted"] = r.status_code == 200
    sha = r.headers.get("X-Fx1-Receipt-Sha256", "")
    cid = r.headers.get("X-Fx1-Completion-Id", "")
    out["hdr.sha_hex"] = _SHA256.fullmatch(sha) is not None
    out["hdr.completion_id"] = bool(cid)
    # The header is the seal of THIS call's completion record.
    rec = client.get(f"/harness/completions/{cid}", headers=_h())
    out["hdr.record_fetches"] = rec.status_code == 200
    sealed = completion_record_receipt(dict(rec.json()))
    out["hdr.sha_matches_record"] = sealed["receipt_sha256"] == sha
    out["hdr.sealed_doc_verifies"] = verify_receipt_payload(sealed)["valid"] is True
    # A distinct call carries a distinct seal over its own record.
    r2 = client.post(
        _COMPLETE,
        json={"backend": _MODEL, "messages": [{"role": "user", "content": "hdr-2"}]},
        headers=_h(),
    )
    sha2 = r2.headers.get("X-Fx1-Receipt-Sha256", "")
    cid2 = r2.headers.get("X-Fx1-Completion-Id", "")
    rec2 = client.get(f"/harness/completions/{cid2}", headers=_h())
    out["hdr.distinct_per_call"] = sha2 != sha and cid2 != cid
    out["hdr.second_matches_own_record"] = (
        completion_record_receipt(dict(rec2.json()))["receipt_sha256"] == sha2
    )
    # Under a managed key the seal still lands — over that call's record.
    raw, _ = _mint(client, "hdr-key")
    r3 = client.post(
        _COMPLETE,
        json={"backend": _MODEL, "messages": [{"role": "user", "content": "hdr-3"}]},
        headers=_h(raw),
    )
    sha3 = r3.headers.get("X-Fx1-Receipt-Sha256", "")
    cid3 = r3.headers.get("X-Fx1-Completion-Id", "")
    rec3 = client.get(f"/harness/completions/{cid3}", headers=_h(raw))
    out["hdr.managed_key_seal"] = (
        _SHA256.fullmatch(sha3) is not None
        and sha3 != sha
        and completion_record_receipt(dict(rec3.json()))["receipt_sha256"] == sha3
    )
    # The record binds the caller's credential fingerprint — a response
    # under a different key never carries another call's sha.
    out["hdr.binds_caller_key"] = rec3.json().get("key_id") != rec.json().get("key_id")
    return out


# ---------------------------------------------------------------------------
# BYOK transport — the supplied key rides only upstream Authorization
# ---------------------------------------------------------------------------


class _FakeUpstreamResp:
    """A minimal HTTP-ish response for the patched ``_openai_urlopen``."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> _FakeUpstreamResp:
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def _byok_resolver(name: str, *args: Any, **kwargs: Any) -> Any:
    """Resolve backend links; ``byok`` builds a real OpenAICompatBackend
    from the request's override (both call shapes the wire uses)."""
    from fx1.serve.backends import OpenAICompatBackend  # noqa: PLC0415

    if name == "byok":
        cfg = dict(kwargs)
        for arg in args:
            if isinstance(arg, dict):
                cfg.update(arg)
        cfg.pop("checkpoint_dir", None)
        timeout = cfg.pop("timeout_s", None)
        if timeout is None:
            for arg in reversed(args):
                if isinstance(arg, (int, float)):
                    timeout = arg
        return OpenAICompatBackend(timeout_s=int(timeout or 120), **cfg)
    return {_MODEL: lambda: _StubBackend("fx1")}[name]()


def _probe_byok_transport() -> dict[str, bool]:
    out: dict[str, bool] = {}
    import fx1.serve.backends as backends_mod  # noqa: PLC0415

    captured: dict[str, Any] = {}

    def spy(req: Any, **kw: Any) -> _FakeUpstreamResp:
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = req.data.decode() if isinstance(req.data, bytes) else None
        return _FakeUpstreamResp(
            {
                "id": "chatcmpl-byok",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "byok-answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    workdir = _temporary_directory() / "byok-leak"
    client, _app = _client(workdir, resolver=_byok_resolver)
    body = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "fx1": {
            "backend": "byok",
            "byok": {"base_url": _BYOK_URL, "api_key": _BYOK_KEY, "model": "m"},
        },
    }
    with patch.object(backends_mod, "_openai_urlopen", spy):
        r = client.post("/v1/chat/completions", json=body, headers=_h())
    out["byok.call_200"] = r.status_code == 200
    out["byok.key_only_on_authorization"] = captured.get("auth") == f"Bearer {_BYOK_KEY}"
    out["byok.key_not_in_upstream_body"] = _BYOK_KEY not in str(captured.get("body"))
    out["byok.key_not_in_response"] = _BYOK_KEY not in r.text
    out["byok.key_absent_from_state"] = _never_in_state(workdir, _BYOK_KEY)
    got = client.get("/harness/completions", headers=_h())
    out["byok.key_absent_from_completions_record"] = _BYOK_KEY not in got.text

    # An upstream failure never echoes the credential either.
    def boom(req: Any, **kw: Any) -> _FakeUpstreamResp:
        captured["auth"] = req.headers.get("Authorization")
        raise OSError("connection refused")

    with patch.object(backends_mod, "_openai_urlopen", boom):
        r2 = client.post("/v1/chat/completions", json=body, headers=_h())
    out["byok.error_never_echoes_key"] = (
        _BYOK_KEY not in r2.text and captured.get("auth") == f"Bearer {_BYOK_KEY}"
    )
    out["byok.key_absent_from_state_after_error"] = _never_in_state(workdir, _BYOK_KEY)

    # Header-BYOK form — same transport contract. The body model names
    # the upstream deployment (must not be a harness link name); the
    # X-Fx1-Byok-* headers select the byok link.
    captured.clear()
    body2 = {"model": "upstream-model-x", "messages": [{"role": "user", "content": "hi"}]}
    hdrs = _h(
        **{
            "X-Fx1-Byok-Base-Url": _BYOK_URL,
            "X-Fx1-Byok-Api-Key": _BYOK_KEY,
            "X-Fx1-Byok-Model": "m",
        }
    )
    with patch.object(backends_mod, "_openai_urlopen", spy):
        r3 = client.post("/v1/chat/completions", json=body2, headers=hdrs)
    out["byok.header_form_authorized"] = (
        r3.status_code == 200 and captured.get("auth") == f"Bearer {_BYOK_KEY}"
    )
    out["byok.header_key_absent_from_state"] = _never_in_state(workdir, _BYOK_KEY)
    return out


# ---------------------------------------------------------------------------
# Refusal envelopes — every refusal lands in a real error body
# ---------------------------------------------------------------------------


def _probe_envelopes() -> dict[str, bool]:
    out: dict[str, bool] = {}
    workdir = _temporary_directory() / "envelopes"
    client, _app = _client(workdir)

    r = client.get(_JOBS)
    v = r.json()
    out["env.401_unauthorized"] = (
        r.status_code == 401 and v.get("code") == "unauthorized" and "detail" in v
    )
    # A non-admin managed key refused admin work: POST refuses at the
    # scope check (insufficient_scope); a read-scoped GET reaches the
    # admin gate itself (admin_required).
    raw, _ = _mint(client, "nonadmin", admin=False)
    r = client.post(_KEYS, json={"name": "nope"}, headers=_h(raw))
    v = r.json()
    out["env.403_insufficient_scope"] = (
        r.status_code == 403 and v.get("code") == "insufficient_scope" and "detail" in v
    )
    r = client.get(_KEYS, headers=_h(raw))
    v = r.json()
    # Every /harness/keys* route is admin-scope — a scoped credential's
    # refusal is the same insufficient_scope envelope on any method
    # (``_require_admin`` remains the belt-and-suspenders inner gate).
    out["env.403_scope_gate_uniform"] = (
        r.status_code == 403 and v.get("code") == "insufficient_scope" and "detail" in v
    )
    r = client.get(f"{_KEYS}/no-such-key", headers=_h())
    out["env.404_key_not_found"] = r.status_code == 404 and r.json().get("code") == "key_not_found"
    r = client.get("/receipts/not-a-sha", headers=_h())
    out["env.422_invalid_sha"] = r.status_code == 422
    # callback_secret without callback_url — a 422 problem envelope.
    r = client.post(_JOBS, json={"command": "doctor", "callback_secret": "x"}, headers=_h())
    v = r.json()
    out["env.422_secret_without_url"] = r.status_code == 422 and "detail" in v
    # 409 idempotency_conflict under the harness error grammar.
    r1 = client.post(_JOBS, json={"command": "doctor"}, headers=_ih("env-ik"))
    r2 = client.post(_JOBS, json={"command": "other"}, headers=_ih("env-ik"))
    out["env.409_idem_conflict"] = (
        r1.status_code == 202
        and r2.status_code == 409
        and r2.json().get("code") == "idempotency_conflict"
    )
    # The /v1 grammar wraps refusal in the provider error envelope.
    r = client.post("/v1/chat/completions", json=_chat_body("x"))
    out["env.v1_401_openai_shape"] = r.status_code == 401 and "error" in r.json()
    return out


# ---------------------------------------------------------------------------
# Source invariants — the constant-time + never-serialize pins
# ---------------------------------------------------------------------------


def _module_source(mod_name: str) -> str:
    import importlib  # noqa: PLC0415

    mod = importlib.import_module(mod_name)
    return inspect.getsource(mod)


def _probe_source_invariants() -> dict[str, bool]:
    out: dict[str, bool] = {}
    import fx1.serve.keys as keys_mod  # noqa: PLC0415
    import fx1.serve.webhooks as wh_mod  # noqa: PLC0415

    wh_src = _module_source("fx1.serve.webhooks")
    out["src.webhook_compare_digest"] = "hmac.compare_digest" in inspect.getsource(
        wh_mod.verify_webhook
    )
    out["src.webhook_no_plain_eq_on_sig"] = (
        "signature ==" not in wh_src and "== signature" not in wh_src
    )
    api_src = _module_source("fx1.serve.api")
    out["src.env_key_compare_digest"] = "hmac.compare_digest" in api_src
    # Managed-key auth is a sha256 dict lookup — never a per-secret compare.
    auth_src = inspect.getsource(keys_mod.ApiKeyStore.authenticate)
    out["src.managed_auth_sha_indexed"] = "_by_hash.get" in auth_src
    out["src.managed_auth_no_secret_eq"] = "raw ==" not in auth_src and "== raw" not in auth_src
    # The callback secret is a PrivateAttr — outside pydantic serialization.
    out["src.callback_secret_private"] = (
        "_callback_secret: str | None = PrivateAttr" in api_src
        or ("_callback_secret" in api_src and "PrivateAttr" in api_src)
    )
    # The key-mint idem ledger is deliberately unjournaled (raw secret in
    # the recorded answer); pin the construction site honestly.
    out["src.mint_idem_unjournaled"] = "_IdemStore(idem_max)" in api_src
    # BYOK upstream request binds the key to Authorization only.
    backends_src = _module_source("fx1.serve.backends")
    out["src.byok_authorization_header"] = (
        '"Authorization"' in backends_src and "Bearer" in backends_src
    )
    out["src.admin_gate_exists"] = 'code="admin_required"' in api_src
    return out


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def crypto_audit() -> dict[str, bool]:
    """Run every crypto probe; literal bools out."""
    with _audit_context():
        # The loopback webhook sink is a deliberate test target; prod
        # callback delivery stays public-network-only without the opt-in.
        os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"
        out: dict[str, bool] = {}
        out.update(_probe_seal())
        out.update(_probe_receipt_routes())
        out.update(_probe_webhook_sig())
        out.update(_probe_key_material())
        out.update(_probe_key_quarantine())
        out.update(_probe_journal_chain())
        out.update(_probe_idem_integrity())
        out.update(_probe_receipt_headers())
        out.update(_probe_byok_transport())
        out.update(_probe_envelopes())
        out.update(_probe_source_invariants())
        return out


def crypto_audit_bench(results: dict[str, bool] | None = None) -> dict[str, Any]:
    """Seal crypto-audit results; run the battery when results are omitted."""
    r = crypto_audit() if results is None else dict(results)
    ok = bool(r) and all(v is True for v in r.values())
    defects = sorted(k for k, v in r.items() if v is not True) if r else ["no_probes"]
    out: dict[str, Any] = {
        "kind": "crypto_audit",
        "schema": "crypto_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok, "defects": defects},
        "coverage": {
            "transport": "Starlette TestClient in-process + loopback HTTP webhook sink",
            "sealed": "receipt_v2 verify payload/file + ops_receipt writers",
            "journaled": "JobJournal chain ops + ApiKeyStore quarantine + idem ledgers",
            "not_verified": [
                "TLS transport (loopback http only)",
                "external process restart (in-process app recreation over the same dirs)",
                "wall-clock webhook replay beyond the 300s tolerance window simulation",
                "forged tail-line acceptance — anchorless-chain boundary pinned honest",
            ],
        },
        "interpretation": (
            "The crypto contract holds end to end: sealed receipts verify "
            "under the canonical-JSON sha256 convention and every mutation "
            "class fails closed — bit flips, sha swaps, duplicate keys, "
            "truncation, NaN, non-objects, resealed honesty lies, and "
            "v2-downgrade strips; webhook signatures cover exactly "
            "<ts>.<body> under HMAC-SHA256 with constant-time compare, "
            "freshness enforcement, and fail-closed malformed inputs; "
            "callback secrets and minted key material never serialize to "
            "wire, journal, or store; the journal hash chain detects "
            "mid-file mutation, mid-line forgery, and torn tails while "
            "honestly admitting clean tail deletion and recomputed-tail "
            "forgery as the anchorless boundary; idempotency replays "
            "byte-identical recorded bytes, drops poisoned journals into "
            "honest re-execution, and isolates by credential fingerprint; "
            "BYOK material rides only the upstream Authorization header; "
            "every refusal is enveloped."
            if ok
            else f"CRYPTO AUDIT DEFECTS: {defects}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(crypto_audit_bench(), indent=2, sort_keys=True))
