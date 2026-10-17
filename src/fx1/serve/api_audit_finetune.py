"""Hermetic OpenAI fine-tuning lifecycle probes for api_audit.v1."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

_PATH_FT_JOBS = "/v1/fine_tuning/jobs"
_CORPUS_FILE = "c.jsonl"


def probe_finetune(api_mod: Any, out: dict[str, Any]) -> None:
    """/v1/fine_tuning/jobs — the OpenAI fine-tune lifecycle over the
    staged pipeline. The stub runner is hermetic; probes pin submit-time
    validation, the event feed, cooperative cancel, idempotent replay,
    and artifact registration into the files store."""
    from fastapi.testclient import TestClient as _TC3

    from fx1.serve.finetune import FTJobOutcome

    _CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'

    def _runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        emit("info", "stub train step", {"epoch": 1})
        art = spec.work_dir / "train_receipt.json"
        art.write_text("{}")
        return FTJobOutcome(
            fine_tuned_model=spec.ft_model_name,
            artifacts={"train_receipt": art},
            trained_tokens=None,
            checkpoint=str(spec.work_dir / "ckpt"),
        )

    ft = _TC3(api_mod.create_app(public_release=False, ft_runner=_runner))

    def _upload(content: bytes, purpose: str = "fine-tune") -> Any:
        return ft.post(
            "/v1/files",
            files={"file": ("corpus.jsonl", content)},
            data={"purpose": purpose},
        )

    def _wait_ft(job_id: str) -> dict[str, Any]:
        for _i in range(400):
            j = ft.get(f"/v1/fine_tuning/jobs/{job_id}").json()
            if j["status"] in ("succeeded", "failed", "cancelled"):
                return dict(j)
            time.sleep(0.02)
        return dict(ft.get(f"/v1/fine_tuning/jobs/{job_id}").json())

    # Happy path: upload purpose=fine-tune → submit → poll → succeeded,
    # artifacts registered into the files store.
    up = _upload(_CORPUS)
    out["ft_upload_finetune_purpose"] = (
        up.status_code == 200 and up.json()["purpose"] == "fine-tune"
    )
    fid = up.json()["id"]
    sub = ft.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": fid, "suffix": "audit"})
    out["ft_create_200"] = (
        sub.status_code == 200
        and sub.json()["id"].startswith("ftjob-")
        and sub.json()["object"] == "fine_tuning.job"
        and sub.json()["training_file"] == fid
    )
    jid = sub.json()["id"]
    fin = _wait_ft(jid)
    out["ft_succeeds"] = (
        fin["status"] == "succeeded"
        and fin["fine_tuned_model"] == f"ft:fx1:audit:{jid.split('-', 1)[1][:12]}"
        and fin["finished_at"] is not None
        and len(fin["result_files"]) == 1
        and fin["trained_tokens"] is None  # no tokenizer — never fabricated
    )
    out["ft_artifact_downloadable"] = (
        bool(fin["result_files"])
        and ft.get(f"/v1/files/{fin['result_files'][0]}/content").status_code == 200
        and ft.get(f"/v1/files/{fin['result_files'][0]}").json()["purpose"] == "fine-tune-result"
    )
    evs = ft.get(f"/v1/fine_tuning/jobs/{jid}/events")
    ev_msgs = [e["message"] for e in evs.json()["data"]] if evs.status_code == 200 else []
    out["ft_events_feed"] = (
        evs.status_code == 200
        and evs.json()["object"] == "list"
        and any("validated" in m for m in ev_msgs)
        and any("job started" in m for m in ev_msgs)
        and any("stub train step" in m for m in ev_msgs)
        and ev_msgs[-1].startswith("job succeeded")
    )
    out["ft_list_shape"] = (
        ft.get(_PATH_FT_JOBS).status_code == 200
        and ft.get(_PATH_FT_JOBS).json()["object"] == "list"
        and ft.get(_PATH_FT_JOBS).json()["has_more"] is False
        and ft.get(_PATH_FT_JOBS).json()["data"][0]["id"] == jid
    )

    # Fail-closed submit surface.
    out["ft_model_not_trainable_400"] = (
        ft.post(_PATH_FT_JOBS, json={"model": "byok", "training_file": fid})
        .json()
        .get("error", {})
        .get("code")
        == "model_not_trainable"
    )
    out["ft_unknown_file_404"] = (
        ft.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": "file-nope"})
        .json()
        .get("error", {})
        .get("code")
        == "file_not_found"
    )
    bfid = _upload(_CORPUS, purpose="batch").json()["id"]
    out["ft_wrong_purpose_400"] = (
        ft.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": bfid})
        .json()
        .get("error", {})
        .get("code")
        == "invalid_training_file"
    )
    mfid = _upload(b"not jsonl\n").json()["id"]
    out["ft_malformed_corpus_400"] = (
        ft.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": mfid})
        .json()
        .get("error", {})
        .get("code")
        == "invalid_training_file"
    )
    out["ft_bad_purpose_upload_400"] = (
        ft.post(
            "/v1/files",
            files={"file": (_CORPUS_FILE, _CORPUS)},
            data={"purpose": "user_data"},
        ).status_code
        == 400
    )
    out["ft_missing_routes_404"] = (
        ft.get("/v1/fine_tuning/jobs/ftjob-nope").status_code == 404
        and ft.post("/v1/fine_tuning/jobs/ftjob-nope/cancel").status_code == 404
        and ft.get("/v1/fine_tuning/jobs/ftjob-nope/events").status_code == 404
    )
    out["ft_cancel_terminal_409"] = (
        ft.post(f"/v1/fine_tuning/jobs/{jid}/cancel").status_code == 409
        and ft.post(f"/v1/fine_tuning/jobs/{jid}/cancel").json().get("error", {}).get("code")
        == "job_terminal"
    )

    # Idempotent replay: same key+body returns the same job; a different
    # body under the same key is a 409 conflict.
    ik = "ft-audit-key-1"
    r1 = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": fid},
        headers={"Idempotency-Key": ik},
    )
    r2 = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": fid},
        headers={"Idempotency-Key": ik},
    )
    r3 = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": fid, "suffix": "other"},
        headers={"Idempotency-Key": ik},
    )
    out["ft_idem_replay"] = (
        r1.status_code == 200
        and r2.status_code == 200
        and r2.json()["id"] == r1.json()["id"]
        and r3.status_code == 409
        and r3.json().get("error", {}).get("code") == "idempotency_conflict"
    )

    # Runner failure → failed job with the typed error + error event;
    # never a 5xx on the submit itself.
    def _boom(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        raise RuntimeError("no gpu")

    ftf = _TC3(api_mod.create_app(public_release=False, ft_runner=_boom))
    bf = ftf.post(
        _PATH_FT_JOBS,
        json={
            "model": "fx1",
            "training_file": ftf.post(
                "/v1/files",
                files={"file": (_CORPUS_FILE, _CORPUS)},
                data={"purpose": "fine-tune"},
            ).json()["id"],
        },
    )
    out["ft_runner_failure_submit_200"] = bf.status_code == 200
    for _i in range(400):
        jj = ftf.get(f"/v1/fine_tuning/jobs/{bf.json()['id']}").json()
        if jj["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    out["ft_runner_failure_failed"] = (
        jj["status"] == "failed"
        and jj["error"]["code"] == "job_failed"
        and "no gpu" in jj["error"]["message"]
    )

    # Cooperative cancel: a running job honors the flag at the runner's
    # boundary and lands 'cancelled' (the runner returns early — never
    # killed mid-write).
    def _gate_runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        for _i in range(500):
            if should_cancel():
                return FTJobOutcome()
            time.sleep(0.02)
        return FTJobOutcome(fine_tuned_model=spec.ft_model_name, artifacts={})

    ftc = _TC3(api_mod.create_app(public_release=False, ft_runner=_gate_runner))
    cf = ftc.post(
        "/v1/files", files={"file": (_CORPUS_FILE, _CORPUS)}, data={"purpose": "fine-tune"}
    )
    cj = ftc.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": cf.json()["id"]})
    cjid = cj.json()["id"]
    for _i in range(400):
        if ftc.get(f"/v1/fine_tuning/jobs/{cjid}").json()["status"] == "running":
            break
        time.sleep(0.02)
    cc = ftc.post(f"/v1/fine_tuning/jobs/{cjid}/cancel")
    for _i in range(500):
        cjj = ftc.get(f"/v1/fine_tuning/jobs/{cjid}").json()
        if cjj["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    out["ft_cancel_running_cooperative"] = cc.status_code == 200 and cjj["status"] == "cancelled"

    # Cooperative pause/resume: a gate-aware runner parks inside
    # pause_gate once paused — status reads 'paused' while parked and
    # resume releases it to completion. 'paused' is non-terminal: a
    # parked job still honors cancel, exactly once; pause on a paused
    # job replays 200; pause or resume on a terminal job is 409; resume
    # on a non-paused job is 409.
    #
    # The queued branch is probed through a store-fabricated entry:
    # inflight == worker count, so 'queued' only exists in the scheduler
    # gap between admission and worker dispatch — unreachable
    # deterministically over the wire. The endpoint + store contract for
    # it is identical (the parked worker waits on resume.wait).
    p_hold = threading.Event()
    p_gate = threading.Event()
    p_hold2 = threading.Event()
    p_gate2 = threading.Event()

    def _pause_runner(spec: Any, *, emit: Any, should_cancel: Any, pause_gate: Any) -> FTJobOutcome:
        emit("info", "stage A")
        p_hold.wait(timeout=20)
        p_gate.set()
        if pause_gate():
            return FTJobOutcome()
        emit("info", "stage B")
        # hold the worker in-flight across the resume read — the
        # response body is the record at read time, so the probe can
        # only observe the restored 'running' while the runner is live
        p_hold2.wait(timeout=20)
        p_gate2.set()
        return FTJobOutcome(fine_tuned_model=spec.ft_model_name, artifacts={})

    ftp_app = api_mod.create_app(public_release=False, ft_runner=_pause_runner, max_inflight=1)
    ftp = _TC3(ftp_app)
    pfid = ftp.post(
        "/v1/files", files={"file": (_CORPUS_FILE, _CORPUS)}, data={"purpose": "fine-tune"}
    ).json()["id"]
    pa = ftp.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": pfid, "suffix": "pa"},
    ).json()
    for _i in range(400):  # pa holds the single slot, running
        if ftp.get(f"/v1/fine_tuning/jobs/{pa['id']}").json()["status"] == "running":
            break
        time.sleep(0.02)
    pqa = ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/pause")
    pqa2 = ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/pause")

    from fx1.serve.finetune import FTJob as _FTJob

    qentry = ftp_app.state.ft_store.put(
        _FTJob(
            id="ftjob-qprobe",
            model="fx1",
            created_at=int(time.time()),
            status="queued",
            training_file=pfid,
        ),
        None,
        "qprobe",
    )
    pqj = ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/pause")
    rqj = ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/resume")
    out["ft_pause_queued_running_200"] = (
        pqa.status_code == 200
        and pqa.json()["status"] == "paused"
        and pqa2.status_code == 200
        and pqa2.json()["status"] == "paused"  # idempotent replay
        and pqj.status_code == 200
        and pqj.json()["status"] == "paused"
        and rqj.status_code == 200
        and rqj.json()["status"] == "queued"  # resume restores paused_from
        and qentry.paused_from is None
    )
    # queued resume still leaves the fabricated job queued (no worker
    # ever ran) — pause it again and cancel from paused.
    ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/pause")
    cq = ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/cancel")
    out["ft_queued_cancel_terminal"] = (
        cq.status_code == 200
        and cq.json()["status"] == "cancelled"
        and ftp.post("/v1/fine_tuning/jobs/ftjob-qprobe/resume").status_code == 409
    )

    p_hold.set()  # pa's runner reaches the gate and parks
    p_gate.wait(timeout=15)
    time.sleep(0.05)
    out["ft_paused_holds_at_gate"] = ftp.get(f"/v1/fine_tuning/jobs/{pa['id']}").json()[
        "status"
    ] == "paused" and not any(
        "job succeeded" in e["message"]
        for e in ftp.get(f"/v1/fine_tuning/jobs/{pa['id']}/events").json()["data"]
    )
    rpa = ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/resume")
    p_hold2.set()  # release the resumed runner to terminal
    for _i in range(400):
        paf = ftp.get(f"/v1/fine_tuning/jobs/{pa['id']}").json()
        if paf["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    out["ft_resume_releases"] = (
        rpa.status_code == 200
        and rpa.json()["status"] == "running"  # restores paused_from
        and paf["status"] == "succeeded"
    )

    # cancel against a parked worker: one 'job cancelled' event total —
    # the store owns the terminal write, the parked worker dedupes.
    p_hold.clear()
    p_gate.clear()
    p_hold2.clear()
    p_gate2.clear()
    pc = ftp.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": pfid, "suffix": "pc"},
    ).json()
    for _i in range(400):
        if ftp.get(f"/v1/fine_tuning/jobs/{pc['id']}").json()["status"] == "running":
            break
        time.sleep(0.02)
    ftp.post(f"/v1/fine_tuning/jobs/{pc['id']}/pause")
    p_hold.set()
    p_gate.wait(timeout=15)
    ccancel = ftp.post(f"/v1/fine_tuning/jobs/{pc['id']}/cancel")
    for _i in range(400):
        pcj = ftp.get(f"/v1/fine_tuning/jobs/{pc['id']}").json()
        if pcj["status"] == "cancelled":
            break
        time.sleep(0.02)
    pc_events = ftp.get(f"/v1/fine_tuning/jobs/{pc['id']}/events").json()["data"]
    out["ft_cancel_paused_terminal"] = (
        ccancel.status_code == 200
        and ccancel.json()["status"] == "cancelled"
        and pcj["status"] == "cancelled"
        and sum("job cancelled" in e["message"] for e in pc_events) == 1
        and ftp.post(f"/v1/fine_tuning/jobs/{pc['id']}/resume").status_code == 409
    )
    out["ft_pause_guards"] = (
        ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/pause").json().get("error", {}).get("code")
        == "job_terminal"
        and ftp.post(f"/v1/fine_tuning/jobs/{pa['id']}/resume").json().get("error", {}).get("code")
        == "job_terminal"
        and ftp.post("/v1/fine_tuning/jobs/ftjob-nope/pause").status_code == 404
        and ftp.post("/v1/fine_tuning/jobs/ftjob-nope/resume").status_code == 404
    )

    # Gate-free runners (no pause_gate kwarg) accept the pause verb but
    # run to completion — the hook is opt-in, never a crash.
    def _nogate_runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        deadline = time.time() + 1.5
        while time.time() < deadline:
            if should_cancel():
                return FTJobOutcome()
            time.sleep(0.02)
        return FTJobOutcome(fine_tuned_model=spec.ft_model_name, artifacts={})

    ftn = _TC3(api_mod.create_app(public_release=False, ft_runner=_nogate_runner))
    nfid = ftn.post(
        "/v1/files", files={"file": (_CORPUS_FILE, _CORPUS)}, data={"purpose": "fine-tune"}
    ).json()["id"]
    nj = ftn.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": nfid}).json()
    for _i in range(400):
        if ftn.get(f"/v1/fine_tuning/jobs/{nj['id']}").json()["status"] == "running":
            break
        time.sleep(0.02)
    npa = ftn.post(f"/v1/fine_tuning/jobs/{nj['id']}/pause")
    for _i in range(400):
        njf = ftn.get(f"/v1/fine_tuning/jobs/{nj['id']}").json()
        if njf["status"] in ("succeeded", "failed", "cancelled", "paused"):
            break
        time.sleep(0.02)
    if njf["status"] == "paused":  # worker parked pre-start — release it
        ftn.post(f"/v1/fine_tuning/jobs/{nj['id']}/resume")
        for _i in range(400):
            njf = ftn.get(f"/v1/fine_tuning/jobs/{nj['id']}").json()
            if njf["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.02)
    out["ft_pause_gatefree_completes"] = npa.status_code == 200 and njf["status"] == "succeeded"

    # Model registry: a succeeded job with a checkpoint registers its
    # ft: name into the model inventory, and a request naming it resolves
    # to the local_fx1 lane pinned at the job's checkpoint — never the
    # default link. Unregistered ft: names fail closed 404.
    mname = fin["fine_tuned_model"]
    models = ft.get("/v1/models").json()
    out["ft_model_listed"] = mname in {m["id"] for m in models["data"]}
    card = ft.get(f"/v1/models/{mname}")
    out["ft_model_card_200"] = card.status_code == 200 and card.json()["id"] == mname
    out["ft_model_ghost_404"] = ft.get("/v1/models/ft:fx1:ghost:000000000000").status_code == 404

    resolved: list[tuple[str, Any]] = []

    def _spy(name: str, *a: Any, **k: Any) -> Any:
        resolved.append((name, k.get("checkpoint_dir") or (a[0] if a else None)))
        raise RuntimeError("no engine — resolution reached")

    ft2 = _TC3(api_mod.create_app(public_release=False, backend_resolver=_spy, ft_runner=_runner))
    fid2 = ft2.post(
        "/v1/files", files={"file": (_CORPUS_FILE, _CORPUS)}, data={"purpose": "fine-tune"}
    ).json()["id"]
    j2 = ft2.post(_PATH_FT_JOBS, json={"model": "fx1", "training_file": fid2, "suffix": "r"}).json()
    fin2 = j2
    for _i in range(400):
        fin2 = ft2.get(f"/v1/fine_tuning/jobs/{j2['id']}").json()
        if fin2["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    ftname = fin2["fine_tuned_model"]
    chat = ft2.post(
        "/v1/chat/completions",
        json={"model": ftname, "messages": [{"role": "user", "content": "hi"}]},
    )
    out["ft_model_routes_local_fx1"] = (
        chat.status_code == 503
        and resolved
        and resolved[-1][0] == "local_fx1"
        and str(resolved[-1][1]).endswith("ckpt")
    )
    ghost = ft2.post(
        "/v1/chat/completions",
        json={
            "model": "ft:fx1:ghost:000000000000",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    out["ft_model_unknown_404"] = (
        ghost.status_code == 404 and ghost.json().get("error", {}).get("code") == "model_not_found"
    )
    # An explicit backend + checkpoint header still wins over an ft:
    # model name — the registry never overrides a caller's stated link.
    resolved.clear()
    ft2.post(
        "/v1/chat/completions",
        json={"model": ftname, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Fx1-Backend": "local_fx1", "X-Fx1-Checkpoint-Dir": "/srv/fx1/explicit-ckpt"},
    )
    out["ft_model_explicit_backend_wins"] = bool(
        resolved and resolved[-1] == ("local_fx1", "/srv/fx1/explicit-ckpt")
    )

    # GET /v1/fine_tuning/jobs/{id}/checkpoints — OpenAI's
    # list_checkpoints: the registered model artifacts a job produced,
    # oldest-first; unknown jobs fail closed 404.
    ck = ft2.get(f"/v1/fine_tuning/jobs/{j2['id']}/checkpoints")
    out["ft_checkpoints_list"] = (
        ck.status_code == 200
        and ck.json()["object"] == "list"
        and ck.json()["has_more"] is False
        and ck.json()["first_id"] == ck.json()["last_id"]
        and [c["fine_tuned_model_checkpoint"] for c in ck.json()["data"]] == [ftname]
        and ck.json()["data"][0]["id"].startswith("ftckpt-")
        and ck.json()["data"][0]["object"] == "fine_tuning.job.checkpoint"
    )
    out["ft_checkpoints_404"] = (
        ft2.get("/v1/fine_tuning/jobs/ftjob-nope/checkpoints").status_code == 404
    )

    # DELETE /v1/models/{id} — OpenAI's models.delete for ft: names: the
    # tombstone is real (list/retrieve/chat all go 404 after), a built-in
    # link id refuses 400, and a ghost name fails closed 404 — a delete
    # verdict is never fabricated.
    dele = ft2.delete(f"/v1/models/{ftname}")
    out["ft_model_delete_200"] = (
        dele.status_code == 200
        and dele.json()["id"] == ftname
        and dele.json()["object"] == "model"
        and dele.json()["deleted"] is True
    )
    out["ft_model_delete_gone"] = (
        ft2.get(f"/v1/models/{ftname}").status_code == 404
        and ftname not in {m["id"] for m in ft2.get("/v1/models").json()["data"]}
        and ft2.post(
            "/v1/chat/completions",
            json={"model": ftname, "messages": [{"role": "user", "content": "hi"}]},
        ).status_code
        == 404
    )
    out["ft_model_delete_ghost_404"] = (
        ft2.delete("/v1/models/ft:fx1:ghost:000000000000").status_code == 404
    )
    out["ft_model_delete_builtin_400"] = ft2.delete("/v1/models/fx1").status_code == 400
    # a deleted ft: name drops off the job's checkpoint listing — the
    # tombstone is real, no fabricated history.
    out["ft_checkpoints_delete_drops"] = (
        ft2.get(f"/v1/fine_tuning/jobs/{j2['id']}/checkpoints").json()["data"] == []
    )

    # Terminal webhooks on the /v1 surface — the fx1 extension mirrors
    # the /harness/jobs contract: fire once at the terminal transition,
    # HMAC-signed X-Fx1-Webhook-* headers when callback_secret is set,
    # and the delivery verdict (status/attempts/error) rides the record.
    import json as _json4  # noqa: PLC0415
    import threading as _threading  # noqa: PLC0415
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: PLC0415

    _ft_hits: list[tuple[dict[str, str], bytes]] = []

    class _FTHook(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — http.server handler name
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            _ft_hits.append((dict(self.headers.items()), raw))
            self.send_response(404 if self.path == "/reject" else 200)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    _ft_srv = ThreadingHTTPServer(("127.0.0.1", 0), _FTHook)
    _threading.Thread(target=_ft_srv.serve_forever, daemon=True).start()
    _ft_cb = f"http://127.0.0.1:{_ft_srv.server_address[1]}/ft"
    # loopback callback_url is refused by the SSRF guard without the
    # private-networks opt-in, read per request (saved/restored verbatim).
    saved_wh_env = os.environ.get("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS")
    os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = "1"

    def _wait_cb(job_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + 10.0
        j: dict[str, Any] = {}
        while time.monotonic() < deadline:
            j = _wait_ft(job_id)
            if j.get("callback_status") is not None:
                return j
            time.sleep(0.02)
        return j

    cb_fid = _upload(_CORPUS).json()["id"]
    cb_job = ft.post(
        _PATH_FT_JOBS,
        json={
            "model": "fx1",
            "training_file": cb_fid,
            "callback_url": _ft_cb,
            "callback_secret": "whsec-audit",
        },
    ).json()
    cb_fin = _wait_cb(cb_job["id"])
    signed_ok = False
    if _ft_hits:
        from fx1.serve.webhooks import verify_webhook  # noqa: PLC0415

        _h, _b = _ft_hits[0]
        signed_ok = verify_webhook(
            "whsec-audit",
            _h.get("X-Fx1-Webhook-Timestamp"),
            _h.get("X-Fx1-Webhook-Signature"),
            _b,
        )
    out["ft_webhook_fires_signed"] = (
        len(_ft_hits) == 1
        and cb_fin.get("callback_status") == "delivered"
        and cb_fin.get("callback_attempts") == 1
        and _json4.loads(_ft_hits[0][1])["status"] == "succeeded"
        and signed_ok
    )
    out["ft_webhook_secret_never_serializes"] = (
        "callback_secret" not in cb_fin and "callback_secret" not in _json4.loads(_ft_hits[0][1])
    )
    # 4xx is definitive — one attempt, no retry storm.
    rj_fid = _upload(_CORPUS).json()["id"]
    rj_job = ft.post(
        _PATH_FT_JOBS,
        json={
            "model": "fx1",
            "training_file": rj_fid,
            "callback_url": _ft_cb.replace("/ft", "/reject"),
        },
    ).json()
    rj_fin = _wait_cb(rj_job["id"])
    out["ft_webhook_4xx_never_retried"] = (
        rj_fin.get("callback_status") == "failed"
        and rj_fin.get("callback_attempts") == 1
        and len(_ft_hits) == 2
        and "404" in (rj_fin.get("callback_error") or "")
    )
    # Submit-time guards: secret requires url; url must be http(s) with a
    # host — both as the /v1 envelope's 422, never a queued zombie.
    sec_only = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": cb_fid, "callback_secret": "x"},
    )
    bad_url = ft.post(
        _PATH_FT_JOBS,
        json={"model": "fx1", "training_file": cb_fid, "callback_url": "ftp://x"},
    )
    out["ft_webhook_guards_422"] = (
        sec_only.status_code == 422
        and bad_url.status_code == 422
        and sec_only.json().get("error", {}).get("code") == "validation"
    )
    _ft_srv.shutdown()
    _ft_srv.server_close()
    if saved_wh_env is None:
        os.environ.pop("FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS", None)
    else:
        os.environ["FX1_WEBHOOK_ALLOW_PRIVATE_NETWORKS"] = saved_wh_env
