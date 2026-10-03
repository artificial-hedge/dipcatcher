"""KATs for the /v1/fine_tuning surface — store, validation, API, SDK twin."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fx1.sdk import Fx1Harness
from fx1.serve.api import create_app
from fx1.serve.finetune import (
    FTJobOutcome,
    FTJobStore,
    validate_chat_jsonl,
)

_CORPUS = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'


def _runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
    emit("info", "runner working")
    art = spec.work_dir / "receipt.json"
    art.write_text("{}")
    return FTJobOutcome(fine_tuned_model=spec.ft_model_name, artifacts={"receipt.json": art})


def _client(tmp_path: Path, runner: Any = _runner) -> TestClient:
    return TestClient(
        create_app(
            backend_resolver=lambda *a, **k: _B(),
            ft_runner=runner,
            ft_dir=tmp_path / "ft",
        ),
        raise_server_exceptions=False,
    )


class _B:
    def complete(self, *a: Any, **k: Any) -> Any:
        return "ok"


def _upload(client: TestClient, content: bytes = _CORPUS, purpose: str = "fine-tune") -> str:
    resp = client.post(
        "/v1/files",
        files={"file": ("c.jsonl", content, "application/jsonl")},
        data={"purpose": purpose},
    )
    assert resp.status_code == 200, resp.text
    return str(resp.json()["id"])


def test_validate_chat_jsonl():
    assert validate_chat_jsonl(_CORPUS, file_id="f") == 1
    with pytest.raises(ValueError):
        validate_chat_jsonl(b"not json\n", file_id="f")
    with pytest.raises(ValueError):
        validate_chat_jsonl(b'{"nope": true}\n', file_id="f")
    with pytest.raises(ValueError):
        validate_chat_jsonl(b'{"messages": "x"}\n', file_id="f")


def test_store_lifecycle(tmp_path: Path):
    sdk = Fx1Harness(ft_runner=_runner, ft_dir=tmp_path)
    job = sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS, suffix="s")
    assert job.status == "succeeded"
    assert job.fine_tuned_model == f"ft:fx1:s:{job.id.split('-', 1)[1][:12]}"
    assert job.finished_at is not None
    assert job.training_file.startswith("inline:")
    got = sdk.finetune_job(job.id)
    assert got.id == job.id
    events = sdk.finetune_job_events(job.id)
    assert len(events) >= 2 and events[-1]["message"] == "job succeeded"
    assert job.id in {j.id for j in sdk.finetune_jobs()}
    with pytest.raises(KeyError):
        sdk.finetune_job("ftjob-nope")


def test_sdk_guards(tmp_path: Path):
    sdk = Fx1Harness(ft_runner=_runner, ft_dir=tmp_path)
    with pytest.raises(ValueError):
        sdk.create_finetune_job(model="byok", training_jsonl=_CORPUS)
    with pytest.raises(ValueError):
        sdk.create_finetune_job(model="fx1", training_jsonl=b"not jsonl\n")


def test_sdk_runner_failure_is_failed_not_raise(tmp_path: Path):
    def boom(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        raise RuntimeError("no trainer")

    sdk = Fx1Harness(ft_runner=boom, ft_dir=tmp_path)
    job = sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS)
    assert job.status == "failed"
    assert job.error is not None and job.error.code == "job_failed"


def test_api_happy_path(tmp_path: Path):
    c = _client(tmp_path)
    fid = _upload(c)
    job = c.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": fid, "suffix": "pp"},
    ).json()
    assert job["object"] == "fine_tuning.job"
    job = _wait_terminal(c, job["id"])
    assert job["status"] == "succeeded"
    assert len(job["result_files"]) == 1
    events = c.get(f"/v1/fine_tuning/jobs/{job['id']}/events").json()
    assert events["object"] == "list" and len(events["data"]) >= 2
    listed = c.get("/v1/fine_tuning/jobs").json()
    assert any(j["id"] == job["id"] for j in listed["data"])
    art = c.get(f"/v1/files/{job['result_files'][0]}/content")
    assert art.status_code == 200
    card = c.get(f"/v1/files/{job['result_files'][0]}").json()
    assert card["purpose"] == "fine-tune-result"


def test_api_guards(tmp_path: Path):
    c = _client(tmp_path)
    fid = _upload(c)
    r = c.post("/v1/fine_tuning/jobs", json={"model": "byok", "training_file": fid})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_trainable"
    r = c.post("/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": "file-nope"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "file_not_found"
    bfid = _upload(c, purpose="batch")
    r = c.post("/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": bfid})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_training_file"
    mfid = _upload(c, content=b"not jsonl\n")
    r = c.post("/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": mfid})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_training_file"
    assert c.get("/v1/fine_tuning/jobs/ftjob-nope").status_code == 404
    assert c.post("/v1/fine_tuning/jobs/ftjob-nope/cancel").status_code == 404


def test_api_idempotency_and_terminal_cancel(tmp_path: Path):
    c = _client(tmp_path)
    fid = _upload(c)
    body = {"model": "fx1", "training_file": fid}
    r1 = c.post("/v1/fine_tuning/jobs", json=body, headers={"Idempotency-Key": "k1"})
    r2 = c.post("/v1/fine_tuning/jobs", json=body, headers={"Idempotency-Key": "k1"})
    assert r1.json()["id"] == r2.json()["id"]
    r3 = c.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": fid, "suffix": "other"},
        headers={"Idempotency-Key": "k1"},
    )
    assert r3.status_code == 409 and r3.json()["error"]["code"] == "idempotency_conflict"
    _wait_terminal(c, r1.json()["id"])
    r4 = c.post(f"/v1/fine_tuning/jobs/{r1.json()['id']}/cancel")
    assert r4.status_code == 409 and r4.json()["error"]["code"] == "job_terminal"


def test_api_cooperative_cancel(tmp_path: Path):
    def gate(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
        for _ in range(200):
            if should_cancel():
                return FTJobOutcome(fine_tuned_model=None, artifacts={})
            time.sleep(0.02)
        raise AssertionError("should_cancel never fired")

    c = _client(tmp_path, runner=gate)
    fid = _upload(c)
    job = c.post("/v1/fine_tuning/jobs", json={"model": "fx1", "training_file": fid}).json()
    c.post(f"/v1/fine_tuning/jobs/{job['id']}/cancel")
    deadline = time.time() + 30
    while True:
        cur = c.get(f"/v1/fine_tuning/jobs/{job['id']}").json()
        if cur["status"] in {"succeeded", "failed", "cancelled"}:
            break
        assert time.time() < deadline
        time.sleep(0.05)
    assert cur["status"] == "cancelled"


def test_wire_client_error_map(tmp_path: Path):
    import urllib.parse  # noqa: PLC0415

    from fx1.serve.client import HarnessClient, HarnessTransportError  # noqa: PLC0415

    c = _client(tmp_path)

    def send(
        method: str,
        url: str,
        payload: dict[str, Any] | bytes | None,
        headers: dict[str, str],
        timeout_s: float,
    ) -> tuple[int, dict[str, str], bytes]:
        p = urllib.parse.urlparse(url)
        path = p.path + (f"?{p.query}" if p.query else "")
        if method == "GET":
            resp = c.get(path, headers=headers)
        else:
            resp = c.post(path, json=payload, headers=headers)
        return resp.status_code, dict(resp.headers), resp.content

    remote = HarnessClient("http://harness.test", transport=send)
    with pytest.raises(KeyError):
        remote.finetune_job("ftjob-nope")
    with pytest.raises(HarnessTransportError):
        remote.create_finetune_job(model="byok", training_file="file-x")


def test_list_pagination(tmp_path: Path):
    c = _client(tmp_path)
    fid = _upload(c)
    ids = [
        c.post(
            "/v1/fine_tuning/jobs",
            json={"model": "fx1", "training_file": fid, "suffix": f"s{i}"},
        ).json()["id"]
        for i in range(3)
    ]
    page1 = c.get("/v1/fine_tuning/jobs?limit=2").json()
    assert len(page1["data"]) == 2 and page1["has_more"] is True
    page2 = c.get(f"/v1/fine_tuning/jobs?limit=2&after={page1['data'][-1]['id']}").json()
    assert page2["data"] and page2["data"][-1]["id"] == ids[0]
    ev = c.get(f"/v1/fine_tuning/jobs/{ids[0]}/events?limit=1").json()
    assert len(ev["data"]) == 1 and ev["has_more"] is True
    ev2 = c.get(f"/v1/fine_tuning/jobs/{ids[0]}/events?limit=5&after={ev['data'][0]['id']}").json()
    assert ev2["data"] and ev2["data"][0]["id"] != ev["data"][0]["id"]


def test_store_bound(tmp_path: Path):
    store = FTJobStore(2)
    sdk = Fx1Harness(ft_runner=_runner, ft_dir=tmp_path)
    # swap in the small store to exercise eviction
    sdk._ft_store = store
    j1 = sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS)
    sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS)
    sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS)
    with pytest.raises(KeyError):
        sdk.finetune_job(j1.id)


def _ckpt_runner(spec: Any, *, emit: Any, should_cancel: Any) -> FTJobOutcome:
    art = spec.work_dir / "receipt.json"
    art.write_text("{}")
    return FTJobOutcome(
        fine_tuned_model=spec.ft_model_name,
        artifacts={"receipt.json": art},
        checkpoint=str(spec.work_dir / "ckpt"),
    )


def _wait_terminal(client: TestClient, job_id: str) -> dict[str, Any]:
    for _i in range(400):
        j = client.get(f"/v1/fine_tuning/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return dict(j)
        time.sleep(0.02)
    return dict(client.get(f"/v1/fine_tuning/jobs/{job_id}").json())


def test_model_registry_api(tmp_path: Path):
    client = _client(tmp_path, runner=_ckpt_runner)
    fid = _upload(client)
    job = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": fid, "suffix": "reg"},
    ).json()
    name = _wait_terminal(client, job["id"])["fine_tuned_model"]
    assert name in {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert client.get(f"/v1/models/{name}").json()["id"] == name
    assert client.get("/v1/models/ft:fx1:ghost:000").status_code == 404


def test_model_registry_routes_local_fx1(tmp_path: Path):
    calls: list[tuple[str, Any]] = []

    def spy(name: str, *a: Any, **k: Any) -> Any:
        calls.append((name, k.get("checkpoint_dir") or (a[0] if a else None)))
        return _B()

    client = TestClient(
        create_app(
            backend_resolver=spy,
            ft_runner=_ckpt_runner,
            ft_dir=tmp_path / "ft",
        ),
        raise_server_exceptions=False,
    )
    fid = _upload(client)
    job = client.post(
        "/v1/fine_tuning/jobs",
        json={"model": "fx1", "training_file": fid},
    ).json()
    name = _wait_terminal(client, job["id"])["fine_tuned_model"]
    ckpt = calls  # resolver spy
    resp = client.post(
        "/v1/chat/completions",
        json={"model": name, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["system_fingerprint"] == "local_fx1"
    assert ckpt[-1][0] == "local_fx1" and str(ckpt[-1][1]).endswith("ckpt")
    # unknown ft: name fails closed
    ghost = client.post(
        "/v1/chat/completions",
        json={
            "model": "ft:fx1:ghost:000000000000",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert ghost.status_code == 404
    assert ghost.json()["error"]["code"] == "model_not_found"
    # explicit backend header overrides the registry
    calls.clear()
    resp2 = client.post(
        "/v1/chat/completions",
        json={"model": name, "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Fx1-Backend": "local_fx1", "X-Fx1-Checkpoint-Dir": "/tmp/other"},
    )
    assert resp2.status_code == 200
    assert calls[-1] == ("local_fx1", "/tmp/other")


def test_model_registry_sdk(tmp_path: Path):
    sdk = Fx1Harness(
        ft_runner=_ckpt_runner,
        ft_dir=tmp_path,
        backend_resolver=lambda name, **kw: _B(),
    )
    job = sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS)
    assert job.fine_tuned_model is not None
    name = job.fine_tuned_model
    assert name in {m.id for m in sdk.openai_models().data}
    assert sdk.openai_model(name).id == name
    with pytest.raises(ValueError):
        sdk.openai_model("ft:fx1:ghost:000")
    resp, _cid = sdk.openai_chat({"model": name, "messages": [{"role": "user", "content": "hi"}]})
    assert resp.system_fingerprint == "local_fx1"
    with pytest.raises(ValueError):
        sdk.openai_chat(
            {
                "model": "ft:fx1:ghost:000000000000",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )


def test_model_registry_evicts_with_job(tmp_path: Path):
    sdk = Fx1Harness(
        ft_runner=_ckpt_runner,
        ft_dir=tmp_path,
        backend_resolver=lambda name, **kw: _B(),
    )
    sdk._ft_store = FTJobStore(1)
    j1 = sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS)
    assert j1.fine_tuned_model is not None
    sdk.create_finetune_job(model="fx1", training_jsonl=_CORPUS)
    with pytest.raises(KeyError):
        sdk.finetune_job(j1.id)
    with pytest.raises(ValueError):
        sdk.openai_model(j1.fine_tuned_model)
