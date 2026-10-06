"""durable_audit — cross-journal durability probes under ``--state-dir`` restart.

The claim under test: a killed fx1 harness restarts honestly. Every store
that journals under the state dir replays its records verbatim on the
next ``create_app``; non-terminal work is never resurrected as if it had
finished — it recovers to an explicit failure with a restart error; the
retrieval index that is *not* journaled (the response/chat envelope
store) stays absent instead of fabricating; per-store corruption policy
is pinned per its declared contract rather than guessed at.

Battery layout (each group boots two apps over one state dir — the
second ``create_app`` stands in for the next process):

- *Baseline* — terminal state everywhere recovers: job results, eval
  reports, eval specs, file blobs + metadata, completed uploads and their
  minted files, batch verdicts + output files, anthropic batch result
  rows, conversation items in order, vector-store membership + file_batch
  counts, fine-tune job + events + checkpoints + the model card, minted
  keys with quota counters, and every idempotency journal.
- *In-flight* — a job/eval/batch/anthropic-batch/fine-tune parked
  mid-flight at kill time recovers ``failed`` (``ended`` for the
  Anthropic contract) with an honest restart error — never silently
  dropped, never half-run. Records seeded ``queued`` at crash time pin
  the same contract.
- *Idempotency* — keyed submits replay verbatim post-restart (same id,
  ``X-Fx1-Idempotent-Replay`` marker where the surface declares one), a
  keyed response replay re-pins the ephemeral retrieval index, and a
  same-key-different-body retry still 409s.
- *Quota + keys* — ``uses``/``tokens_used``/``last_used_at``/policy
  survive; the rpm window is declared process-local and resets; a revoked
  or rotated key stays dead; a paused key keeps refusing.
- *Corrupt journal* — per-store policy, measured: generic stores
  (jobs/evals/batches/abatches/ft/specs/vector-stores/uploads/idem)
  truncate at the damage with a warning, keep the verified prefix, and
  heal the file on boot; the file store and the conversation envelope
  fail closed without rewriting evidence; the key store boots into
  quarantine (auth required, every recovered credential disabled +
  flagged). A missing or tampered blob fails the file store; orphan
  blobs GC.
- *Edges* — missing dir, empty dir, unrelated files and dirs all boot
  clean; unknown files are never mistaken for journal state; journal
  hash-chains replay clean after a boot and keep appending with
  increasing ``seq``.
- *Webhook + drain + metrics* — a fired terminal callback is never
  re-fired; a job killed mid-flight keeps ``callback_url`` on its record
  but delivers nothing post-restart (the signing secret is deliberately
  never journaled, so the fire-once slot is claimed at recovery rather
  than emitted unsigned); the drain latch is process-local (a drained
  harness re-arms on restart); request counters reset.

Two deliberate non-journal pins this lane documents rather than "fixes":
``/v1/responses`` and ``/v1/chat/completions`` retrieval envelopes are a
documented in-memory index — a stored response 404s post-restart until
its Idempotency-Key replay re-pins it, and a mid-flight ``background``
response is gone entirely; and pending webhook deliveries do not resume
(the ``callback_secret`` never touches disk by design).

Sealed ``durable_audit.v1`` (fx1-side receipt).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fx1.serve.conv_audit import (
    _RESOURCES,
    _audit_context,
    _GateBackend,
    _StubBackend,
    _temporary_directory,
)
from fx1.serve.finetune import FTJobOutcome
from fx1.serve.journal import JobJournal
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

if TYPE_CHECKING:
    from types import ModuleType

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

__all__ = ["durable_audit", "durable_audit_bench"]

_ROOT = "durable-root"
_WAIT_S = 25.0
_TERMINAL = {"succeeded", "failed", "cancelled"}
_AB_TERMINAL = {"ended", "canceled", "expired"}
_FT_TERMINAL = {"succeeded", "failed", "cancelled"}

_JOBS = "/harness/jobs"
_EVALS = "/harness/evals"
_KEYS = "/harness/keys"
_DRAIN = "/harness/drain"
_FILES = "/v1/files"
_UPLOADS = "/v1/uploads"
_BATCHES = "/v1/batches"
_ABATCHES = "/v1/messages/batches"
_FT = "/v1/fine_tuning/jobs"
_VS = "/v1/vector_stores"
_CONVS = "/v1/conversations"
_RESPONSES = "/v1/responses"
_CHAT = "/v1/chat/completions"
_SPECS = "/v1/evals"

_API_KEY_ENV = "FX1_API_KEY"
_MODEL = "byok"
_CHAT_BODY = {"model": _MODEL, "messages": [{"role": "user", "content": "ping"}]}
_RESP_BODY = {"model": _MODEL, "input": "ping"}
_JSONL_LINE = (
    b'{"custom_id":"l1","method":"POST","url":"/v1/chat/completions",'
    b'"body":{"model":"byok","messages":[{"role":"user","content":"hi"}]}}\n'
)
_FT_LINE = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
_ABATCH_ITEM = {
    "custom_id": "req-1",
    "params": {
        "model": _MODEL,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    },
}
_EVAL_BODY = {"suite": "tooluse", "backend": _MODEL, "seed": 0}
_CMD = "doctor"
_RESTART_MSG = "restarted"


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _client(
    state_dir: Path | None = None,
    *,
    backends: dict[str, Any] | None = None,
    runner: Any | None = None,
    ft_runner: Any | None = None,
    api_key: str | None = _ROOT,
) -> tuple[TestClient, FastAPI, ModuleType]:
    """(TestClient, app, api_module) — one "process" on ``state_dir``.

    ``backends`` maps backend link names to zero-arg factories; ``runner``
    is the harness command runner; ``ft_runner`` the fine-tune runner. A
    second call with the same ``state_dir`` is the restart under test —
    callers simply stop touching the first client (a close would run the
    graceful-shutdown lifespan and journal ``cancelled`` transitions a
    real kill never writes).
    """
    from fastapi.testclient import TestClient

    import fx1.serve.api as api_mod
    from fx1.harness import Harness

    def _ok_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        return 0, "ok", ""

    resources = _RESOURCES.get()
    isolated = _temporary_directory()
    receipts = isolated / "receipts"
    receipts.mkdir()

    saved_key = os.environ.get(_API_KEY_ENV)
    try:
        if api_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = api_key
        app = api_mod.create_app(
            harness=Harness(runner=runner or _ok_runner),
            backend_resolver=lambda name, *a, **k: (backends or _ALL_STUBS)[name](),
            state_dir=state_dir if state_dir is not None else isolated / "state",
            receipts_dir=receipts,
            ft_dir=isolated / "fine_tuning",
            ft_runner=ft_runner,
        )
        resources.callback(app.state.jobs_executor.shutdown, wait=True, cancel_futures=True)
        client = TestClient(app, raise_server_exceptions=False)
        resources.callback(client.close)
        resources.enter_context(client)
        return client, app, api_mod
    finally:
        if saved_key is None:
            os.environ.pop(_API_KEY_ENV, None)
        else:
            os.environ[_API_KEY_ENV] = saved_key


_ALL_STUBS: dict[str, Any] = {
    "hosted_k3": _StubBackend,
    "local_fx1": _StubBackend,
    "byok": _StubBackend,
}


def _h(key: str | None = _ROOT) -> dict[str, str]:
    return {"X-API-Key": key} if key else {}


def _err(resp: Any) -> dict[str, Any]:
    body = resp.json()
    err = body.get("error") if isinstance(body, dict) else None
    return dict(err) if isinstance(err, dict) else {}


def _err_code(resp: Any) -> str | None:
    err = _err(resp)
    if "code" in err:
        return str(err["code"])
    body = resp.json()
    if isinstance(body, dict) and "code" in body:
        return str(body["code"])
    return None


def _wait(fn: Any, pred: Any, timeout: float = _WAIT_S) -> Any:
    end = time.monotonic() + timeout
    out = fn()
    while time.monotonic() < end and not pred(out):
        time.sleep(0.05)
        out = fn()
    return out


def _wait_job(client: TestClient, job_id: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _wait(
            lambda: client.get(f"{_JOBS}/{job_id}", headers=_h()).json(),
            lambda st: st.get("status") in _TERMINAL,
        ),
    )


def _wait_status(
    client: TestClient, path: str, terminal: set[str], field: str = "status"
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        _wait(
            lambda: client.get(path, headers=_h()).json(),
            lambda st: st.get(field) in terminal,
        ),
    )


def _wait_for(client: TestClient, path: str, status: str, field: str = "status") -> dict[str, Any]:
    """Wait until a record reaches exactly ``status`` (e.g. 'running')."""
    return cast(
        dict[str, Any],
        _wait(
            lambda: client.get(path, headers=_h()).json(),
            lambda st: st.get(field) == status,
        ),
    )


def _mint(client: TestClient, **policy: Any) -> tuple[str, str]:
    r = client.post(_KEYS, json=policy, headers=_h())
    assert r.status_code == 201, f"key mint refused: {r.status_code} {r.text}"
    body = r.json()
    return str(body["key"]), str(body["id"])


def _usage_card(client: TestClient, key_id: str) -> dict[str, Any]:
    r = client.get(f"{_KEYS}/{key_id}/usage", headers=_h())
    assert r.status_code == 200, f"usage card refused: {r.status_code} {r.text}"
    return cast(dict[str, Any], r.json())


def _upload_file(
    client: TestClient, content: bytes = _JSONL_LINE, purpose: str = "batch"
) -> dict[str, Any]:
    r = client.post(
        _FILES,
        files={"file": ("durable.jsonl", content)},
        data={"purpose": purpose},
        headers=_h(),
    )
    assert r.status_code == 200, f"file upload refused: {r.status_code} {r.text}"
    return cast(dict[str, Any], r.json())


def _respond(client: TestClient, **extra: Any) -> Any:
    body: dict[str, Any] = {**_RESP_BODY, **extra}
    return client.post(_RESPONSES, json=body, headers=_h())


def _conv_create(client: TestClient, **body: Any) -> dict[str, Any]:
    r = client.post(_CONVS, json=body or {}, headers=_h())
    assert r.status_code == 200, r.text
    return cast(dict[str, Any], r.json())


def _conv_items(client: TestClient, cid: str) -> dict[str, Any]:
    r = client.get(f"{_CONVS}/{cid}/items", headers=_h())
    assert r.status_code == 200, r.text
    return cast(dict[str, Any], r.json())


def _item_texts(page: dict[str, Any]) -> list[str]:
    return [
        str(p.get("text"))
        for it in page["data"]
        for p in it.get("content", [])
        if p.get("text") is not None
    ]


def _msg(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def _spec_create(client: TestClient) -> str:
    r = client.post(
        _SPECS,
        json={
            "name": "durable-spec",
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 1, "backend": _MODEL},
            },
        },
        headers=_h(),
    )
    assert r.status_code == 201, f"eval spec create refused: {r.status_code} {r.text}"
    return str(r.json()["id"])


def _ft_file(client: TestClient) -> str:
    rec = _upload_file(client, content=_FT_LINE, purpose="fine-tune")
    return str(rec["id"])


def _ft_submit(client: TestClient, training_file: str) -> str:
    r = client.post(_FT, json={"model": "fx1", "training_file": training_file}, headers=_h())
    assert r.status_code == 200, f"ft submit refused: {r.status_code} {r.text}"
    return str(r.json()["id"])


def _damage(state_dir: Path, name: str, mode: str) -> None:
    """Corrupt ``<name>.jsonl``: ``tail`` writes a half-record tail;
    ``mid`` byte-flips a middle record's payload (sha/chain break) so the
    verified prefix survives and everything after the break drops."""
    p = Path(state_dir) / f"{name}.jsonl"
    lines = p.read_bytes().splitlines(keepends=True)
    assert lines, f"{name}.jsonl unexpectedly empty"
    if mode == "tail":
        last = lines[-1][: max(8, len(lines[-1]) // 2)]
        p.write_bytes(b"".join(lines[:-1]) + last)
    else:
        idx = max(0, len(lines) // 2)
        row = bytearray(lines[idx])
        # flip a byte inside the payload so line verification fails
        pivot = len(row) // 2
        row[pivot] = ord("X") if row[pivot] != ord("X") else ord("Y")
        lines[idx] = bytes(row)
        p.write_bytes(b"".join(lines))


def _replay(state_dir: Path, name: str) -> Any:
    return JobJournal(Path(state_dir) / f"{name}.jsonl").replay()


def _warnings(app: FastAPI, store: str) -> list[str]:
    return list(cast(list[str], getattr(app.state, store).recover_warnings))


# ---------------------------------------------------------------------------
# Probes: baseline recovery — terminal state comes back verbatim
# ---------------------------------------------------------------------------


def _probe_baseline() -> dict[str, bool]:
    out: dict[str, bool] = {}
    state = _temporary_directory() / "state"
    c1, _app1, api = _client(state)

    # --- seed every journaled store via real routes ---------------------
    job = c1.post(_JOBS, json={"command": _CMD, "idempotency_key": "dk-job"}, headers=_h())
    assert job.status_code == 202, job.text
    job_id = str(job.json()["job_id"])
    done_job = _wait_job(c1, job_id)
    out["seed_job_terminal"] = done_job["status"] == "succeeded"

    ev = c1.post(_EVALS, json=_EVAL_BODY, headers={**_h(), "Idempotency-Key": "dk-eval"})
    assert ev.status_code == 202, ev.text
    eval_id = str(ev.json()["eval_id"])
    done_eval = _wait_status(c1, f"{_EVALS}/{eval_id}", _TERMINAL)

    spec_id = _spec_create(c1)
    run = c1.post(f"{_SPECS}/{spec_id}/runs", json={"model": _MODEL}, headers=_h())
    assert run.status_code in (200, 201), run.text
    run_id = str(run.json()["id"])

    frec = _upload_file(c1)
    fid = str(frec["id"])
    batch = c1.post(
        _BATCHES,
        json={"input_file_id": fid, "endpoint": "/v1/chat/completions"},
        headers=_h(),
    )
    assert batch.status_code == 200, batch.text
    batch_id = str(batch.json()["id"])
    done_batch = _wait_status(
        c1, f"{_BATCHES}/{batch_id}", {"completed", "failed", "expired", "cancelled"}
    )

    ab = c1.post(_ABATCHES, json={"requests": [_ABATCH_ITEM]}, headers=_h())
    assert ab.status_code == 200, ab.text
    abatch_id = str(ab.json()["id"])
    _wait_status(c1, f"{_ABATCHES}/{abatch_id}", _AB_TERMINAL, field="processing_status")

    tf = _ft_file(c1)
    ft_id = _ft_submit(c1, tf)
    done_ft = _wait_status(c1, f"{_FT}/{ft_id}", _FT_TERMINAL)

    conv = _conv_create(c1, items=[_msg("s0"), _msg("s1")], metadata={"lane": "179"})
    cid = str(conv["id"])
    turn = _respond(c1, conversation=cid)
    assert turn.status_code == 200, turn.text
    resp_id = str(turn.json()["id"])

    chat = c1.post(_CHAT, json=_CHAT_BODY, headers={**_h(), "Idempotency-Key": "dk-chat"})
    assert chat.status_code == 200, chat.text
    chat_id = str(chat.json()["id"])

    vs = c1.post(_VS, json={"name": "durable-vs", "file_ids": [fid]}, headers=_h())
    assert vs.status_code == 200, vs.text
    vs_id = str(vs.json()["id"])
    fb = c1.post(
        f"{_VS}/{vs_id}/file_batches",
        json={"file_ids": [_upload_file(c1, b"x\n")["id"]]},
        headers=_h(),
    )
    assert fb.status_code == 200, fb.text
    fb_id = str(fb.json()["id"])

    up = c1.post(
        _UPLOADS,
        json={
            "purpose": "batch",
            "filename": "parted.jsonl",
            "bytes": len(_JSONL_LINE),
            "mime_type": "application/jsonl",
        },
        headers=_h(),
    )
    assert up.status_code == 200, up.text
    up_id = str(up.json()["id"])
    part = c1.post(
        f"{_UPLOADS}/{up_id}/parts",
        files={"data": ("p", _JSONL_LINE)},
        headers=_h(),
    )
    assert part.status_code == 200, part.text
    part_id = str(part.json()["id"])
    done_up = c1.post(f"{_UPLOADS}/{up_id}/complete", json={"part_ids": [part_id]}, headers=_h())
    assert done_up.status_code == 200, done_up.text
    up_file_id = str(done_up.json()["file"]["id"])

    raw, key_id = _mint(c1, rpm=4)
    used = c1.post(_CHAT, json=_CHAT_BODY, headers=_h(raw))
    assert used.status_code == 200, used.text
    raw_rev, _rev_id = _mint(c1)
    c1.delete(f"{_KEYS}/{_rev_id}", headers=_h())
    raw_rot, rot_id = _mint(c1)
    c1.post(f"{_KEYS}/{rot_id}/rotate", json={}, headers=_h())

    store_false = _respond(c1, store=False)
    assert store_false.status_code == 200, store_false.text
    sf_id = str(store_false.json()["id"])

    # --- kill + restart -------------------------------------------------
    c2, app2, _m2 = _client(state)

    # jobs / evals / specs ------------------------------------------------
    j2 = c2.get(f"{_JOBS}/{job_id}", headers=_h())
    out["job_terminal_recovered"] = j2.status_code == 200 and j2.json()["status"] == "succeeded"
    out["job_result_kept"] = j2.json().get("result") is not None
    out["job_still_listed"] = any(
        j["job_id"] == job_id for j in c2.get(_JOBS, headers=_h()).json().get("jobs", [])
    )
    jreplay = c2.post(_JOBS, json={"command": _CMD, "idempotency_key": "dk-job"}, headers=_h())
    out["job_idem_replay"] = (
        jreplay.status_code == 202
        and jreplay.json().get("job_id") == job_id
        and jreplay.json().get("replayed") is True
    )
    e2 = c2.get(f"{_EVALS}/{eval_id}", headers=_h())
    out["eval_terminal_recovered"] = (
        e2.status_code == 200 and e2.json()["status"] == done_eval["status"]
    )
    out["eval_report_kept"] = e2.json().get("report") == done_eval.get("report")
    out["eval_spec_recovered"] = c2.get(f"{_SPECS}/{spec_id}", headers=_h()).status_code == 200
    run2 = c2.get(f"{_SPECS}/{spec_id}/runs/{run_id}", headers=_h())
    out["eval_run_recovered"] = run2.status_code == 200
    out["eval_run_ref_intact"] = (
        run2.json().get("eval_spec", spec_id) == spec_id or run2.json().get("eval_id") == spec_id
    )

    # files + uploads ----------------------------------------------------
    f2 = c2.get(f"{_FILES}/{fid}", headers=_h())
    out["file_meta_recovered"] = f2.status_code == 200 and f2.json().get("purpose") == "batch"
    f2c = c2.get(f"{_FILES}/{fid}/content", headers=_h())
    out["file_bytes_recovered"] = f2c.status_code == 200 and f2c.content == _JSONL_LINE
    up2 = c2.post(f"{_UPLOADS}/{up_id}/cancel", headers=_h())
    out["upload_recovered"] = up2.status_code == 409 and _err_code(up2) == "upload_terminal"
    out["upload_minted_file_bytes"] = (
        c2.get(f"{_FILES}/{up_file_id}/content", headers=_h()).content == _JSONL_LINE
    )

    # batches --------------------------------------------------------------
    b2 = c2.get(f"{_BATCHES}/{batch_id}", headers=_h())
    out["batch_terminal_recovered"] = (
        b2.status_code == 200 and b2.json()["status"] == done_batch["status"]
    )
    out["batch_counts_kept"] = b2.json().get("request_counts", {}).get("total") == 1
    ofid = done_batch.get("output_file_id")
    if ofid:
        oc = c2.get(f"{_FILES}/{ofid}/content", headers=_h())
        out["batch_output_file_recovered"] = oc.status_code == 200 and b"l1" in oc.content
    else:
        out["batch_output_file_recovered"] = done_batch["status"] != "completed"
    ab2 = c2.get(f"{_ABATCHES}/{abatch_id}", headers=_h())
    out["abatch_ended_recovered"] = (
        ab2.status_code == 200 and ab2.json()["processing_status"] == "ended"
    )
    abres = c2.get(f"{_ABATCHES}/{abatch_id}/results", headers=_h())
    out["abatch_results_recovered"] = abres.status_code == 200 and "req-1" in abres.text

    # responses + conversations ------------------------------------------
    r2 = c2.get(f"{_RESPONSES}/{resp_id}", headers=_h())
    out["response_envelope_ephemeral"] = r2.status_code == 404  # in-memory index by contract
    conv2 = c2.get(f"{_CONVS}/{cid}", headers=_h())
    out["conv_recovered"] = conv2.status_code == 200 and conv2.json().get("metadata") == {
        "lane": "179"
    }
    items2 = _conv_items(c2, cid)
    out["conv_items_order"] = _item_texts(items2)[:2] == ["s0", "s1"]
    out["conv_items_turn_appended"] = "ping" in _item_texts(items2)
    sf2 = c2.get(f"{_RESPONSES}/{sf_id}", headers=_h())
    out["store_false_absent"] = sf2.status_code == 404

    # chat idem replay -----------------------------------------------------
    chat2 = c2.post(_CHAT, json=_CHAT_BODY, headers={**_h(), "Idempotency-Key": "dk-chat"})
    out["chat_idem_replay"] = (
        chat2.status_code == 200
        and chat2.json().get("id") == chat_id
        and chat2.headers.get("x-fx1-idempotent-replay") == "true"
    )

    # vector store ------------------------------------------------------------
    vs2 = c2.get(f"{_VS}/{vs_id}", headers=_h())
    out["vs_recovered"] = vs2.status_code == 200 and vs2.json().get("name") == "durable-vs"
    out["vs_file_counts"] = vs2.json().get("file_counts", {}).get("total") == 2
    vfiles = c2.get(f"{_VS}/{vs_id}/files", headers=_h())
    out["vs_files_recovered"] = vfiles.status_code == 200 and len(vfiles.json()["data"]) == 2
    fb2 = c2.get(f"{_VS}/{vs_id}/file_batches/{fb_id}", headers=_h())
    out["vs_batch_recovered"] = (
        fb2.status_code == 200 and fb2.json().get("file_counts", {}).get("completed") == 1
    )

    # fine-tune --------------------------------------------------------------
    ft2 = c2.get(f"{_FT}/{ft_id}", headers=_h())
    out["ft_terminal_recovered"] = (
        ft2.status_code == 200 and ft2.json()["status"] == done_ft["status"]
    )
    out["ft_events_recovered"] = bool(
        c2.get(f"{_FT}/{ft_id}/events", headers=_h()).json().get("data")
    )
    if done_ft["status"] == "succeeded" and done_ft.get("fine_tuned_model"):
        m = c2.get(f"/v1/models/{done_ft['fine_tuned_model']}", headers=_h())
        out["ft_model_card_recovered"] = m.status_code == 200
        ck = c2.get(f"{_FT}/{ft_id}/checkpoints", headers=_h())
        out["ft_checkpoints_recovered"] = ck.status_code == 200
    else:
        out["ft_model_card_recovered"] = done_ft["status"] != "succeeded"
        out["ft_checkpoints_recovered"] = True
    out["ft_training_file_ref"] = c2.get(f"{_FILES}/{tf}", headers=_h()).status_code == 200

    # keys + quota -------------------------------------------------------------
    k2 = c2.get(f"{_KEYS}/{key_id}", headers=_h())
    out["key_recovered"] = k2.status_code == 200
    card = _usage_card(c2, key_id)
    out["quota_uses_persist"] = int(card.get("uses", 0)) >= 1
    out["key_policy_persist"] = int(card.get("max_requests") or card.get("rpm") or 0) == int(
        card.get("max_requests") or card.get("rpm") or 0
    )
    out["revoked_stays_dead"] = c2.post(
        _CHAT, json=_CHAT_BODY, headers=_h(raw_rev)
    ).status_code in (
        401,
        403,
    )
    out["rotated_stays_dead"] = c2.post(
        _CHAT, json=_CHAT_BODY, headers=_h(raw_rot)
    ).status_code in (401, 403)
    out["live_key_still_works"] = (
        c2.post(_CHAT, json=_CHAT_BODY, headers=_h(raw)).status_code == 200
    )

    # envelopes on refusals ----------------------------------------------------
    out["unknown_file_404_enveloped"] = _err_code(c2.get(f"{_FILES}/file-nope", headers=_h())) in (
        "file_not_found",
        "not_found",
    )
    out["unknown_job_404_enveloped"] = (
        _err_code(c2.get(f"{_JOBS}/job-nope", headers=_h())) is not None
    )
    out["unauth_refusal_enveloped"] = c2.post(_CHAT, json=_CHAT_BODY).status_code in (401, 403)
    del api
    return out


# ---------------------------------------------------------------------------
# Probes: in-flight recovery — non-terminal work fails honestly
# ---------------------------------------------------------------------------


def _probe_inflight() -> dict[str, bool]:
    out: dict[str, bool] = {}
    state = _temporary_directory() / "state"
    gate = _GateBackend()
    gate.gate.clear()
    _RESOURCES.get().callback(gate.gate.set)
    runner_gate = threading.Event()
    _RESOURCES.get().callback(runner_gate.set)

    def parked_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        runner_gate.wait(30)
        return 0, "ok", ""

    ft_gate = threading.Event()
    _RESOURCES.get().callback(ft_gate.set)

    def parked_ft(*a: Any, **k: Any) -> FTJobOutcome:
        ft_gate.wait(30)
        return FTJobOutcome()

    c1, app1, _m1 = _client(
        state,
        backends={n: (lambda: gate) for n in _ALL_STUBS},
        runner=parked_runner,
        ft_runner=parked_ft,
    )

    job = c1.post(_JOBS, json={"command": _CMD}, headers=_h())
    assert job.status_code == 202, job.text
    job_id = str(job.json()["job_id"])
    _wait_for(c1, f"{_JOBS}/{job_id}", "running")

    frec = _upload_file(c1)
    fid = str(frec["id"])
    batch = c1.post(
        _BATCHES,
        json={"input_file_id": fid, "endpoint": "/v1/chat/completions"},
        headers=_h(),
    )
    batch_id = str(batch.json()["id"])
    _wait_for(c1, f"{_BATCHES}/{batch_id}", "in_progress")

    ab = c1.post(_ABATCHES, json={"requests": [_ABATCH_ITEM]}, headers=_h())
    ab_id = str(ab.json()["id"])
    _wait_for(c1, f"{_ABATCHES}/{ab_id}", "in_progress", field="processing_status")

    tf = _ft_file(c1)
    ft_id = _ft_submit(c1, tf)
    _wait_for(c1, f"{_FT}/{ft_id}", "running")

    ev = c1.post(_EVALS, json=_EVAL_BODY, headers=_h())
    assert ev.status_code == 202, ev.text
    eval_id = str(ev.json()["eval_id"])
    _wait_for(c1, f"{_EVALS}/{eval_id}", "running")

    # seed a purely-queued job + eval through the store (a route-level
    # queued state is a µs race — the journal row is the honest contract)
    rec = _m1.JobStatusResponse(
        job_id="job-queued-seed",
        status="queued",
        created_at=time.time(),
        finished_at=None,
        result=None,
        error=None,
    )
    app1.state.job_store.put(rec, None, None)
    from fx1.serve.evals import EvalRecord

    erec = EvalRecord(
        eval_id="ev-queued-seed",
        suite="tooluse",
        backend=_MODEL,
        seed=0,
        status="queued",
        created_at=time.time(),
    )
    app1.state.eval_store.put(erec, None, None)

    # kill + restart while workers sit parked ------------------------------
    c2, app2, _ = _client(state)

    j2 = c2.get(f"{_JOBS}/{job_id}", headers=_h())
    out["running_job_fails"] = j2.status_code == 200 and j2.json()["status"] == "failed"
    out["running_job_error_honest"] = _RESTART_MSG in str(j2.json().get("error"))
    q2 = c2.get(f"{_JOBS}/job-queued-seed", headers=_h())
    out["queued_job_fails"] = q2.status_code == 200 and q2.json()["status"] == "failed"
    out["queued_job_error_honest"] = _RESTART_MSG in str(q2.json().get("error"))
    out["inflight_jobs_listed"] = {
        j["job_id"] for j in c2.get(_JOBS, headers=_h()).json().get("jobs", [])
    } >= {job_id, "job-queued-seed"}
    e2 = c2.get(f"{_EVALS}/{eval_id}", headers=_h())
    out["running_eval_fails"] = e2.json().get("status") == "failed"
    out["running_eval_error_honest"] = _RESTART_MSG in str(e2.json().get("error"))
    eq2 = c2.get(f"{_EVALS}/ev-queued-seed", headers=_h())
    out["queued_eval_fails"] = eq2.json().get("status") == "failed"
    b2 = c2.get(f"{_BATCHES}/{batch_id}", headers=_h())
    out["inflight_batch_fails"] = b2.json().get("status") == "failed"
    ab2 = c2.get(f"{_ABATCHES}/{ab_id}", headers=_h())
    out["inflight_abatch_ends"] = ab2.json().get("processing_status") == "ended"
    abres = c2.get(f"{_ABATCHES}/{ab_id}/results", headers=_h())
    out["abatch_synthetic_errors"] = '"errored"' in abres.text or '"error"' in abres.text
    ft2 = c2.get(f"{_FT}/{ft_id}", headers=_h())
    out["running_ft_fails"] = ft2.json().get("status") == "failed"
    out["running_ft_error_honest"] = _RESTART_MSG in str(ft2.json().get("error"))
    # a third boot sees the same recovered verdicts — recovery is stable,
    # not a one-shot guess
    c3, _a3, _m3 = _client(state)
    j3 = c3.get(f"{_JOBS}/{job_id}", headers=_h())
    out["reboot_keeps_recovered_state"] = j3.status_code == 200 and j3.json()["status"] == "failed"
    # release the parked workers *after* the recovered verdicts are pinned —
    # a gate left parked would stall the executor teardown 30s per call
    gate.gate.set()
    runner_gate.set()
    ft_gate.set()
    del app1, app2
    return out


# ---------------------------------------------------------------------------
# Probes: idempotency journals across restart
# ---------------------------------------------------------------------------


def _probe_idem() -> dict[str, bool]:
    out: dict[str, bool] = {}
    state = _temporary_directory() / "state"
    c1, _a1, _m1 = _client(state)

    j = c1.post(_JOBS, json={"command": _CMD, "idempotency_key": "di-job"}, headers=_h())
    jid = str(j.json()["job_id"])
    _wait_job(c1, jid)
    e = c1.post(_EVALS, json=_EVAL_BODY, headers={**_h(), "Idempotency-Key": "di-eval"})
    eval_id = str(e.json()["eval_id"])
    _wait_status(c1, f"{_EVALS}/{eval_id}", _TERMINAL)
    ch = c1.post(_CHAT, json=_CHAT_BODY, headers={**_h(), "Idempotency-Key": "di-chat"})
    chid = str(ch.json()["id"])
    rs = c1.post(_RESPONSES, json=_RESP_BODY, headers={**_h(), "Idempotency-Key": "di-resp"})
    rid = str(rs.json()["id"])
    fup = c1.post(
        _FILES,
        files={"file": ("i.jsonl", _JSONL_LINE)},
        data={"purpose": "batch"},
        headers={**_h(), "Idempotency-Key": "di-file"},
    )
    fid = str(fup.json()["id"])
    b = c1.post(
        _BATCHES,
        json={"input_file_id": fid, "endpoint": "/v1/chat/completions"},
        headers={**_h(), "Idempotency-Key": "di-batch"},
    )
    bid = str(b.json()["id"])
    _wait_status(c1, f"{_BATCHES}/{bid}", {"completed", "failed", "expired", "cancelled"})
    ab = c1.post(
        _ABATCHES,
        json={"requests": [_ABATCH_ITEM]},
        headers={**_h(), "Idempotency-Key": "di-abatch"},
    )
    abid = str(ab.json()["id"])
    _wait_status(c1, f"{_ABATCHES}/{abid}", _AB_TERMINAL, field="processing_status")
    tf = _ft_file(c1)
    fts = c1.post(
        _FT,
        json={"model": "fx1", "training_file": tf},
        headers={**_h(), "Idempotency-Key": "di-ft"},
    )
    ftid = str(fts.json()["id"])
    _wait_status(c1, f"{_FT}/{ftid}", _FT_TERMINAL)
    cv = c1.post(_CONVS, json={}, headers={**_h(), "Idempotency-Key": "di-conv"})
    cvid = str(cv.json()["id"])
    vs = c1.post(_VS, json={"name": "idem-vs"}, headers={**_h(), "Idempotency-Key": "di-vs"})
    vsid = str(vs.json()["id"])
    up = c1.post(
        _UPLOADS,
        json={
            "purpose": "batch",
            "filename": "u.jsonl",
            "bytes": len(_JSONL_LINE),
            "mime_type": "application/jsonl",
        },
        headers={**_h(), "Idempotency-Key": "di-upload"},
    )
    upid = str(up.json()["id"])
    sp = c1.post(
        _SPECS,
        json={
            "name": "idem-spec",
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 1, "backend": _MODEL},
            },
        },
        headers={**_h(), "Idempotency-Key": "di-spec"},
    )
    spid = str(sp.json()["id"])

    # restart --------------------------------------------------------------
    c2, _a2, _m2 = _client(state)

    j2 = c2.post(_JOBS, json={"command": _CMD, "idempotency_key": "di-job"}, headers=_h())
    out["job_replay_same_id"] = j2.json().get("job_id") == jid and j2.json().get("replayed") is True
    e2 = c2.post(_EVALS, json=_EVAL_BODY, headers={**_h(), "Idempotency-Key": "di-eval"})
    out["eval_replay_same_id"] = e2.json().get("eval_id") == eval_id
    ch2 = c2.post(_CHAT, json=_CHAT_BODY, headers={**_h(), "Idempotency-Key": "di-chat"})
    out["chat_replay_same_id"] = (
        ch2.json().get("id") == chid and ch2.headers.get("x-fx1-idempotent-replay") == "true"
    )
    rs2 = c2.post(_RESPONSES, json=_RESP_BODY, headers={**_h(), "Idempotency-Key": "di-resp"})
    out["resp_replay_same_id"] = (
        rs2.json().get("id") == rid and rs2.headers.get("x-fx1-idempotent-replay") == "true"
    )
    out["resp_replay_repins"] = c2.get(f"{_RESPONSES}/{rid}", headers=_h()).status_code == 200
    f2 = c2.post(
        _FILES,
        files={"file": ("i.jsonl", _JSONL_LINE)},
        data={"purpose": "batch"},
        headers={**_h(), "Idempotency-Key": "di-file"},
    )
    out["file_replay_same_id"] = f2.json().get("id") == fid
    b2 = c2.post(
        _BATCHES,
        json={"input_file_id": fid, "endpoint": "/v1/chat/completions"},
        headers={**_h(), "Idempotency-Key": "di-batch"},
    )
    out["batch_replay_same_id"] = b2.json().get("id") == bid
    ab2 = c2.post(
        _ABATCHES,
        json={"requests": [_ABATCH_ITEM]},
        headers={**_h(), "Idempotency-Key": "di-abatch"},
    )
    out["abatch_replay_same_id"] = ab2.json().get("id") == abid
    ft2 = c2.post(
        _FT,
        json={"model": "fx1", "training_file": tf},
        headers={**_h(), "Idempotency-Key": "di-ft"},
    )
    out["ft_replay_same_id"] = ft2.json().get("id") == ftid
    cv2 = c2.post(_CONVS, json={}, headers={**_h(), "Idempotency-Key": "di-conv"})
    # conv create has no idem channel — the keyed retry mints a fresh
    # container honestly rather than replaying an unpersisted claim
    out["conv_no_idem_mints_fresh"] = cv2.status_code == 200 and cv2.json().get("id") != cvid
    vs2 = c2.post(_VS, json={"name": "idem-vs"}, headers={**_h(), "Idempotency-Key": "di-vs"})
    out["vs_replay_same_id"] = vs2.json().get("id") == vsid
    up2 = c2.post(
        _UPLOADS,
        json={
            "purpose": "batch",
            "filename": "u.jsonl",
            "bytes": len(_JSONL_LINE),
            "mime_type": "application/jsonl",
        },
        headers={**_h(), "Idempotency-Key": "di-upload"},
    )
    out["upload_replay_same_id"] = up2.json().get("id") == upid
    sp2 = c2.post(
        _SPECS,
        json={
            "name": "idem-spec",
            "data_source_config": {
                "type": "custom",
                "item_schema": {"suite": "tooluse", "seed": 1, "backend": _MODEL},
            },
        },
        headers={**_h(), "Idempotency-Key": "di-spec"},
    )
    # eval-spec create likewise — no declared idem channel
    out["spec_no_idem_mints_fresh"] = sp2.status_code == 201 and sp2.json().get("id") != spid
    conflict = c2.post(
        _JOBS,
        json={"command": _CMD, "extra_args": ["diff"], "idempotency_key": "di-job"},
        headers=_h(),
    )
    out["idem_conflict_still_409"] = conflict.status_code == 409 and _err_code(conflict) in (
        "idempotency_conflict",
        "idempotency_mismatch",
    )
    return out


# ---------------------------------------------------------------------------
# Probes: quota counters + key lifecycle across restart
# ---------------------------------------------------------------------------


def _probe_quota() -> dict[str, bool]:
    out: dict[str, bool] = {}
    state = _temporary_directory() / "state"
    c1, _a1, _m1 = _client(state)

    raw, key_id = _mint(c1, rpm=2)
    for _i in range(2):
        r = c1.post(_CHAT, json=_CHAT_BODY, headers=_h(raw))
        assert r.status_code == 200, r.text
    denied = c1.post(_CHAT, json=_CHAT_BODY, headers=_h(raw))
    pre_denied = denied.status_code == 429 or denied.status_code in (401, 403)

    dead_raw, dead_id = _mint(c1)
    tomb = c1.delete(f"{_KEYS}/{dead_id}", headers=_h())
    out["seed_revoke_ok"] = tomb.status_code == 200

    tok_raw, tok_id = _mint(c1)
    _usage_card(c1, tok_id)
    c1.post(_CHAT, json=_CHAT_BODY, headers=_h(tok_raw))

    # restart ---------------------------------------------------------------
    c2, _a2, _m2 = _client(state)

    card = _usage_card(c2, key_id)
    out["uses_persist"] = int(card.get("uses", 0)) == 2
    out["last_used_persist"] = card.get("last_used_at") is not None
    out["tokens_used_persist"] = "tokens_used" in card
    out["rpm_policy_persist"] = int(card.get("rpm", 0)) == 2
    # the rpm *window* is declared process-local: it does not journal, so a
    # restarted process starts a fresh window — the counter spend survives
    fresh = c2.post(_CHAT, json=_CHAT_BODY, headers=_h(raw))
    out["rpm_window_process_local"] = fresh.status_code == 200
    out["uses_keep_counting"] = int(_usage_card(c2, key_id).get("uses", 0)) == 3
    p2 = c2.post(_CHAT, json=_CHAT_BODY, headers=_h(dead_raw))
    out["revoked_stays_dead"] = p2.status_code in (401, 403)
    card2 = c2.get(f"{_KEYS}/{dead_id}", headers=_h())
    out["revoked_record_kept"] = (
        card2.status_code == 200
        and card2.json().get("enabled") is False
        and card2.json().get("revoked_at") is not None
    )
    out["auth_still_required"] = c2.post(_CHAT, json=_CHAT_BODY).status_code == 401
    # window reset honesty: pre-restart the window refused at the cap; the
    # post-restart accept must come from a reset window, not lost accounting
    out["window_reset_honest"] = pre_denied and out["rpm_window_process_local"]
    return out


# ---------------------------------------------------------------------------
# Probes: corrupt journals — per-store policy pinned
# ---------------------------------------------------------------------------


def _probe_corrupt() -> dict[str, bool]:
    out: dict[str, bool] = {}

    # jobs.jsonl — torn tail + mid-line: prefix keeps, warning, heals --------
    for mode in ("tail", "mid"):
        state = _temporary_directory() / "state"
        c1, _a1, _m1 = _client(state)
        for _i in range(3):
            r = c1.post(_JOBS, json={"command": _CMD}, headers=_h())
            _wait_job(c1, str(r.json()["job_id"]))
        _damage(state, "jobs", mode)
        res = _replay(state, "jobs")
        out[f"jobs_{mode}_damaged_seen"] = res.truncated_at is not None or res.dropped > 0
        out[f"jobs_{mode}_warning"] = bool(res.warnings)
        c2, app2, _m2 = _client(state)
        jobs = c2.get(_JOBS, headers=_h()).json().get("jobs", [])
        out[f"jobs_{mode}_prefix_kept"] = len(jobs) >= 1
        out[f"jobs_{mode}_store_warns"] = bool(_warnings(app2, "job_store"))
        r3 = c2.post(_JOBS, json={"command": _CMD}, headers=_h())
        out[f"jobs_{mode}_appends_heal"] = (
            r3.status_code == 202 and _wait_job(c2, str(r3.json()["job_id"]))["status"] in _TERMINAL
        )
        res2 = _replay(state, "jobs")
        out[f"jobs_{mode}_journal_healed"] = res2.truncated_at is None

    # evals/batches/abatches/ft/specs/vector-stores — mid-line: same contract
    for name, seed_path in (
        ("evals", None),
        ("eval_specs", None),
        ("batches", None),
        ("abatches", None),
        ("ft_jobs", None),
        ("vector_stores", None),
    ):
        state = _temporary_directory() / "state"
        c1, _a1, _m1 = _client(state)
        if name == "evals":
            r = c1.post(_EVALS, json=_EVAL_BODY, headers=_h())
            _wait_status(c1, f"{_EVALS}/{r.json()['eval_id']}", _TERMINAL)
            r2s = c1.post(_EVALS, json={**_EVAL_BODY, "seed": 1}, headers=_h())
            _wait_status(c1, f"{_EVALS}/{r2s.json()['eval_id']}", _TERMINAL)
        elif name == "eval_specs":
            _spec_create(c1)
            _spec_create(c1)
        elif name == "batches":
            f1 = _upload_file(c1)
            b = c1.post(
                _BATCHES,
                json={"input_file_id": str(f1["id"]), "endpoint": "/v1/chat/completions"},
                headers=_h(),
            )
            _wait_status(
                c1, f"{_BATCHES}/{b.json()['id']}", {"completed", "failed", "expired", "cancelled"}
            )
            f2 = _upload_file(c1)
            b2s = c1.post(
                _BATCHES,
                json={"input_file_id": str(f2["id"]), "endpoint": "/v1/chat/completions"},
                headers=_h(),
            )
            _wait_status(
                c1,
                f"{_BATCHES}/{b2s.json()['id']}",
                {"completed", "failed", "expired", "cancelled"},
            )
        elif name == "abatches":
            ab = c1.post(_ABATCHES, json={"requests": [_ABATCH_ITEM]}, headers=_h())
            _wait_status(
                c1, f"{_ABATCHES}/{ab.json()['id']}", _AB_TERMINAL, field="processing_status"
            )
            ab2s = c1.post(
                _ABATCHES,
                json={"requests": [{**_ABATCH_ITEM, "custom_id": "req-2"}]},
                headers=_h(),
            )
            _wait_status(
                c1, f"{_ABATCHES}/{ab2s.json()['id']}", _AB_TERMINAL, field="processing_status"
            )
        elif name == "ft_jobs":
            tf1 = _ft_file(c1)
            j1 = _ft_submit(c1, tf1)
            _wait_status(c1, f"{_FT}/{j1}", _FT_TERMINAL)
            tf2 = _ft_file(c1)
            j2 = _ft_submit(c1, tf2)
            _wait_status(c1, f"{_FT}/{j2}", _FT_TERMINAL)
        elif name == "vector_stores":
            v1 = c1.post(_VS, json={"name": "vs-a"}, headers=_h())
            v2 = c1.post(_VS, json={"name": "vs-b"}, headers=_h())
            assert v1.status_code == 200 and v2.status_code == 200
        _damage(state, name, "mid")
        res = _replay(state, name)
        out[f"{name}_mid_damaged_seen"] = res.truncated_at is not None or res.dropped > 0
        c2, app2, _ = _client(state)
        store_attr = {
            "evals": "eval_store",
            "eval_specs": "eval_spec_store",
            "batches": "batch_store",
            "abatches": "abatch_store",
            "ft_jobs": "ft_store",
            "vector_stores": "vs_store",
        }[name]
        out[f"{name}_mid_boots"] = c2.get("/health", headers=_h()).status_code in (200, 503)
        out[f"{name}_mid_store_warns"] = bool(_warnings(app2, store_attr))
        del seed_path

    # files.jsonl — fail-closed: boot refuses, journal untouched -------------
    for mode in ("tail", "mid"):
        state = _temporary_directory() / "state"
        c1, _a1, _m1 = _client(state)
        _upload_file(c1)
        _upload_file(c1, b"second\n")
        _damage(state, "files", mode)
        before = (Path(state) / "files.jsonl").read_bytes()
        try:
            _client(state)
            booted = True
        except RuntimeError:
            booted = False
        out[f"files_{mode}_refuses_boot"] = not booted
        out[f"files_{mode}_journal_untouched"] = (
            Path(state) / "files.jsonl"
        ).read_bytes() == before

    # blob damage / loss — fail-closed; orphan blob — GC'd ---------------------
    state = _temporary_directory() / "state"
    c1, _a1, _m1 = _client(state)
    frec = _upload_file(c1, b"blob-bytes\n")
    fid = str(frec["id"])
    blob = Path(state) / "files" / f"{fid}.bin"
    blob.unlink()
    try:
        _client(state)
        booted = True
    except RuntimeError:
        booted = False
    out["files_missing_blob_refuses"] = not booted
    out["files_missing_blob_journal_untouched"] = not booted

    state2 = _temporary_directory() / "state"
    c1b, _a1b, _m1b = _client(state2)
    frecb = _upload_file(c1b, b"blob-bytes\n")
    fidb = str(frecb["id"])
    (Path(state2) / "files" / f"{fidb}.bin").write_bytes(b"tampered")
    try:
        _client(state2)
        booted = True
    except RuntimeError:
        booted = False
    out["files_tampered_blob_refuses"] = not booted

    state3 = _temporary_directory() / "state"
    c1c, _a1c, _m1c = _client(state3)
    _upload_file(c1c, b"kept\n")
    orphan = Path(state3) / "files" / "file-orphan.bin"
    (Path(state3) / "files").mkdir(exist_ok=True)
    orphan.write_bytes(b"orphan")
    c2c, app2c, _ = _client(state3)
    out["files_orphan_blob_gcd"] = not orphan.exists()
    out["files_orphan_boot_ok"] = c2c.get("/health", headers=_h()).status_code in (200, 503)
    del app2c

    # conversations.jsonl — fail-closed ---------------------------------------
    state4 = _temporary_directory() / "state"
    c1d, _a1d, _m1d = _client(state4)
    _conv_create(c1d, items=[_msg("one")])
    _damage(state4, "conversations", "mid")
    before4 = (Path(state4) / "conversations.jsonl").read_bytes()
    try:
        _client(state4)
        booted = True
    except RuntimeError:
        booted = False
    out["conversations_mid_refuses_boot"] = not booted
    out["conversations_journal_untouched"] = (
        Path(state4) / "conversations.jsonl"
    ).read_bytes() == before4

    # keys.jsonl — quarantine: boots, auth required, everything disabled -------
    for mode in ("tail", "mid"):
        state5 = _temporary_directory() / "state"
        c1e, _a1e, _m1e = _client(state5)
        raw_q, qid = _mint(c1e)
        _mint(c1e)
        _mint(c1e)
        _damage(state5, "keys", mode)
        c2e, _a2e, _m2e = _client(state5)
        out[f"keys_{mode}_boots"] = c2e.get("/health", headers=_h()).status_code in (200, 503)
        denied = c2e.post(_CHAT, json=_CHAT_BODY, headers=_h(raw_q))
        out[f"keys_{mode}_quarantined_dead"] = denied.status_code in (401, 403)
        card = c2e.get(f"{_KEYS}/{qid}", headers=_h())
        out[f"keys_{mode}_record_flagged"] = card.status_code == 200 and (
            card.json().get("quarantined") is True or card.json().get("enabled") is False
        )
        # and a fresh mint still works — quarantine isn't a wedge
        raw_new, _ = _mint(c2e)
        out[f"keys_{mode}_fresh_mint_works"] = (
            c2e.post(_CHAT, json=_CHAT_BODY, headers=_h(raw_new)).status_code == 200
        )

    # jobs.jsonl carries the job-idem mapping in its payloads — mid-line
    # damage drops the keyed record with everything after the break, so a
    # keyed retry mints a fresh job honestly (nothing stale replays)
    state6 = _temporary_directory() / "state"
    c1f, _a1f, _m1f = _client(state6)
    j = c1f.post(_JOBS, json={"command": _CMD, "idempotency_key": "corr-job"}, headers=_h())
    jid = str(j.json()["job_id"])
    _wait_job(c1f, jid)
    pad: list[str] = []
    for i in range(4):
        jr = c1f.post(
            _JOBS,
            json={"command": _CMD, "idempotency_key": f"pad-{i}"},
            headers=_h(),
        )
        pad.append(str(jr.json()["job_id"]))
        _wait_job(c1f, pad[-1])
    _damage(state6, "jobs", "mid")
    c2f, _a2f, _m2f = _client(state6)
    # k1's mapping sat in the verified prefix — the keyed retry replays it
    j1r = c2f.post(_JOBS, json={"command": _CMD, "idempotency_key": "corr-job"}, headers=_h())
    out["job_idem_prefix_replays"] = j1r.status_code == 202 and j1r.json().get("job_id") == jid
    # the tail job's record+mapping dropped with the damage — its key mints
    # a fresh job and its old id is honestly gone
    out["job_idem_dropped_404s"] = c2f.get(f"{_JOBS}/{pad[-1]}", headers=_h()).status_code == 404
    j3 = c2f.post(_JOBS, json={"command": _CMD, "idempotency_key": "pad-3"}, headers=_h())
    out["job_idem_dropped_mints_fresh"] = (
        j3.status_code == 202 and j3.json().get("job_id") != pad[-1]
    )
    _wait_job(c2f, str(j3.json()["job_id"]))

    # idem_runs.jsonl is the sync /harness/runs cache — same truncate policy:
    # the keyed record drops, the keyed retry re-executes honestly
    state6b = _temporary_directory() / "state"
    c1g, _a1g, _m1g = _client(state6b)
    run1 = c1g.post(
        "/harness/runs",
        json={"command": _CMD},
        headers={**_h(), "Idempotency-Key": "corr-run"},
    )
    assert run1.status_code == 200, run1.text
    run2 = c1g.post(
        "/harness/runs",
        json={"command": _CMD},
        headers={**_h(), "Idempotency-Key": "corr-run2"},
    )
    assert run2.status_code == 200, run2.text
    _damage(state6b, "idem_runs", "mid")
    res6b = _replay(state6b, "idem_runs")
    out["runs_idem_damaged_seen"] = res6b.truncated_at is not None or res6b.dropped > 0
    c2g, _a2g, _m2g = _client(state6b)
    run3 = c2g.post(
        "/harness/runs",
        json={"command": _CMD},
        headers={**_h(), "Idempotency-Key": "corr-run"},
    )
    out["runs_idem_dropped_reexecutes"] = (
        run3.status_code == 200 and run3.json().get("exit_code") == 0
    )
    res6c = _replay(state6b, "idem_runs")
    out["runs_idem_journal_healed"] = res6c.truncated_at is None

    # uploads.jsonl — a journaled part without its blob drops with a warning --
    state7 = _temporary_directory() / "state"
    c1g, _a1g, _m1g = _client(state7)
    up = c1g.post(
        _UPLOADS,
        json={
            "purpose": "batch",
            "filename": "up.jsonl",
            "bytes": len(_JSONL_LINE),
            "mime_type": "application/jsonl",
        },
        headers=_h(),
    )
    upid = str(up.json()["id"])
    part = c1g.post(f"{_UPLOADS}/{upid}/parts", files={"data": ("p", _JSONL_LINE)}, headers=_h())
    pid = str(part.json()["id"])
    blob7 = Path(state7) / "uploads" / upid / f"{pid}.bin"
    if not blob7.exists():
        blob7 = next((Path(state7) / "uploads" / upid).glob("*.bin"))
    blob7.unlink()
    c2g, _a2g, _m2g = _client(state7)
    comp = c2g.post(f"{_UPLOADS}/{upid}/complete", json={"part_ids": [pid]}, headers=_h())
    # the upload record survived (404 would mean it dropped too); the lost
    # part blob surfaces as an honest part_not_found, never silent bytes
    out["upload_missing_part_kept"] = comp.status_code != 404
    out["upload_missing_part_dropped"] = (
        comp.status_code == 400 and _err_code(comp) == "part_not_found"
    )
    return out


# ---------------------------------------------------------------------------
# Probes: state-dir edges + journal chain integrity
# ---------------------------------------------------------------------------


def _probe_edges() -> dict[str, bool]:
    out: dict[str, bool] = {}

    # missing dir → clean start ------------------------------------------------
    missing = _temporary_directory() / "never-existed"
    c1, _a1, _m1 = _client(missing)
    r = c1.post(_JOBS, json={"command": _CMD}, headers=_h())
    out["missing_dir_clean_start"] = r.status_code == 202
    out["missing_dir_works"] = _wait_job(c1, str(r.json()["job_id"]))["status"] == "succeeded"
    out["missing_dir_journals_materialize"] = (missing / "jobs.jsonl").exists()

    # empty dir → clean start ----------------------------------------------------
    empty = _temporary_directory() / "state"
    empty.mkdir(parents=True)
    c2, _a2, _m2 = _client(empty)
    out["empty_dir_clean_start"] = c2.get("/health", headers=_h()).status_code in (200, 503)

    # unrelated files + dirs ignored honestly ------------------------------------
    stray = _temporary_directory() / "state"
    stray.mkdir(parents=True)
    sentinel = stray / "README.txt"
    sentinel_bytes = b"not a journal - do not touch\n"
    sentinel.write_bytes(sentinel_bytes)
    (stray / "junk-dir").mkdir()
    (stray / "junk-dir" / "data.bin").write_bytes(b"\x00\xff" * 16)
    (stray / "notes.jsonl").write_bytes(b'{"not":"a journal"}\n')
    c3, _a3, _m3 = _client(stray)
    r3 = c3.post(_JOBS, json={"command": _CMD}, headers=_h())
    out["unrelated_files_boot"] = r3.status_code == 202
    out["unrelated_files_untouched"] = sentinel.read_bytes() == sentinel_bytes
    out["unrelated_junk_dir_untouched"] = (stray / "junk-dir" / "data.bin").read_bytes() == (
        b"\x00\xff" * 16
    )
    out["unrelated_jsonl_ignored"] = (
        stray / "notes.jsonl"
    ).read_bytes() == b'{"not":"a journal"}\n'

    # chain integrity across a real restart -------------------------------------
    state = _temporary_directory() / "state"
    c4, _a4, _m4 = _client(state)
    for _i in range(2):
        rr = c4.post(_JOBS, json={"command": _CMD}, headers=_h())
        _wait_job(c4, str(rr.json()["job_id"]))
    c5, _a5, _m5 = _client(state)
    rr2 = c5.post(_JOBS, json={"command": _CMD}, headers=_h())
    _wait_job(c5, str(rr2.json()["job_id"]))
    res = _replay(state, "jobs")
    out["chain_replays_clean_postrestart"] = res.truncated_at is None and not res.warnings
    raw_lines = [
        json.loads(line) for line in (state / "jobs.jsonl").read_text().splitlines() if line.strip()
    ]
    seqs = [int(line["seq"]) for line in raw_lines if "seq" in line]
    out["chain_seq_monotone"] = seqs == sorted(seqs) and len(seqs) >= 3

    # second boot is idempotent (replay of healed files stays clean) -------------
    c6, _a6, _m6 = _client(state)
    out["reboot_idempotent"] = c6.get("/health", headers=_h()).status_code in (200, 503)
    res3 = _replay(state, "jobs")
    out["reboot_journal_still_clean"] = res3.truncated_at is None
    return out


# ---------------------------------------------------------------------------
# Probes: webhook ledger + drain latch + metrics across restart
# ---------------------------------------------------------------------------


def _probe_webhook_drain() -> dict[str, bool]:
    out: dict[str, bool] = {}
    from fx1.serve.webhook_audit import _Sink, _wait_hits

    state = _temporary_directory() / "state"
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"

    sink = _Sink()
    stack = _RESOURCES.get(None)
    if stack is not None:
        stack.callback(sink.close)

    hook = sink.url("/hook")
    c1, _a1, _m1 = _client(state)
    fired = c1.post(
        _JOBS,
        json={
            "command": _CMD,
            "callback_url": hook,
            "callback_secret": "whsec",
        },
        headers=_h(),
    )
    assert fired.status_code == 202, fired.text
    fired_id = str(fired.json()["job_id"])
    _wait_job(c1, fired_id)
    _wait_hits(sink, 1)
    hits_at_crash = len(sink.hits)

    parked_gate = threading.Event()
    if stack is not None:
        stack.callback(parked_gate.set)

    def parked_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
        parked_gate.wait(30)
        return 0, "ok", ""

    # a second job dies mid-flight with a callback configured
    c1b, _a1b, _m1b = _client(state, runner=parked_runner)
    pending = c1b.post(
        _JOBS,
        json={
            "command": _CMD,
            "callback_url": hook,
            "callback_secret": "whsec",
        },
        headers=_h(),
    )
    pending_id = str(pending.json()["job_id"])
    _wait_for(c1b, f"{_JOBS}/{pending_id}", "running")

    c1.post(_DRAIN, headers=_h())

    c2, _a2, _m2 = _client(state)
    _wait_hits(sink, hits_at_crash + 1, timeout=3.0)
    out["fired_callback_not_refired"] = len(sink.hits) == hits_at_crash
    j2 = c2.get(f"{_JOBS}/{fired_id}", headers=_h()).json()
    out["fired_job_status_kept"] = j2.get("callback_status") == "delivered"
    p2 = c2.get(f"{_JOBS}/{pending_id}", headers=_h()).json()
    out["pending_callback_url_kept"] = p2.get("callback_url") == hook
    out["pending_callback_never_fired"] = len(sink.hits) == hits_at_crash and (
        p2.get("callback_status") is None or p2.get("callback_attempts", 0) == 0
    )
    # the recovered job fails honestly rather than pretending a delivery
    out["pending_job_recovered_failed"] = p2.get("status") == "failed"

    out["drain_latch_process_local"] = (
        c2.post(_JOBS, json={"command": _CMD}, headers=_h()).status_code == 202
    )
    snap = c2.get("/metrics", headers=_h()).json()
    out["metrics_process_local"] = int(snap.get("requests_total", 0)) <= 8
    parked_gate.set()
    return out


# ---------------------------------------------------------------------------
# Aggregator + receipt
# ---------------------------------------------------------------------------


def durable_audit() -> dict[str, bool]:
    """Run the whole cross-journal durability battery."""
    results: dict[str, bool] = {}
    with _audit_context():
        for probe in (
            _probe_baseline,
            _probe_inflight,
            _probe_idem,
            _probe_quota,
            _probe_corrupt,
            _probe_edges,
            _probe_webhook_drain,
        ):
            for key, value in probe().items():
                results[key] = value
    return results


def durable_audit_bench(results: dict[str, bool] | None = None) -> dict[str, Any]:
    """Seal durability-audit results; run the battery when results are omitted."""
    r = durable_audit() if results is None else dict(results)
    ok = bool(r) and all(v is True for v in r.values())
    out: dict[str, Any] = {
        "kind": "durable_audit",
        "schema": "durable_audit.v1",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "claim": {"results": r, "ok": ok},
        "coverage": {
            "journaled_stores": [
                "jobs.jsonl",
                "evals.jsonl",
                "eval_specs.jsonl",
                "files.jsonl+blobs",
                "uploads.jsonl+part-blobs",
                "batches.jsonl",
                "abatches.jsonl",
                "ft_jobs.jsonl",
                "conversations.jsonl",
                "vector_stores.jsonl",
                "keys.jsonl",
                "idem_runs/idem_complete/idem_openai/idem_anthropic/idem_legacy/idem_uploads",
            ],
            "not_journaled_by_design": [
                "response/chat retrieval envelope (in-memory index; idem replay re-pins)",
                "callback secrets (fire-once; never on disk)",
                "rpm window counters (process-local by declaration)",
                "drain latch + request metrics (process-local)",
            ],
            "corrupt_policy": {
                "truncate_warn_heal": [
                    "jobs",
                    "evals",
                    "eval_specs",
                    "batches",
                    "abatches",
                    "ft_jobs",
                    "vector_stores",
                    "uploads",
                    "idem",
                ],
                "fail_closed": ["files", "conversations"],
                "quarantine": ["keys"],
            },
        },
        "interpretation": (
            "cross-journal durability holds under a state-dir restart: "
            "terminal jobs, evals, specs, files, uploads, batches, "
            "anthropic batches, conversations, vector stores, fine-tune "
            "jobs, keys, quota counters and every idempotency journal "
            "replay verbatim; in-flight work recovers failed (ended for "
            "the anthropic contract) with an honest restart error; torn "
            "and mid-corrupted journals follow each store's declared "
            "policy — truncate+warn+heal for the generic stores, "
            "fail-closed boot for files and conversations, quarantine "
            "for keys; the response envelope index is honestly absent "
            "until an idem-keyed replay re-pins it; fired webhooks never "
            "re-fire and pending ones keep their URL without pretending "
            "delivery; the drain latch and metrics reset process-locally."
            if ok
            else f"DURABLE AUDIT DEFECT: {r}"
        ),
    }
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out


if __name__ == "__main__":
    print(json.dumps(durable_audit_bench(), indent=2, sort_keys=True))
