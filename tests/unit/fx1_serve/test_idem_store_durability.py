"""Regression checks for the journal-backed replay-store commit boundary."""

from __future__ import annotations

from fastapi.testclient import TestClient
from pydantic import BaseModel

from fx1.serve.api import _idem_scope, _IdemStore, create_app
from fx1.serve.journal import JobJournal


class _Reply(BaseModel):
    value: str


class _Backend:
    model_name = "synthetic"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, sampling=None) -> str:  # noqa: ANN001
        del messages, sampling
        self.calls += 1
        return "synthetic answer"

    def close(self) -> None:
        pass


def test_scoped_key_does_not_retain_raw_header_value() -> None:
    raw = "customer-email@example.test:opaque-retry-secret"

    scoped = _idem_scope(raw, namespace="route:object")

    assert scoped is not None
    assert raw not in scoped
    assert scoped == _idem_scope(raw, namespace="route:object")
    assert scoped != _idem_scope(raw + "-different", namespace="route:object")


def test_failed_journal_append_does_not_publish_or_evict(tmp_path, monkeypatch) -> None:
    path = tmp_path / "idem.jsonl"
    journal = JobJournal(path)
    store = _IdemStore(1, journal=journal, model=_Reply)
    store.put("old-key", "old-fingerprint", _Reply(value="old"))

    def fail_append(_payload: dict[str, object]) -> None:
        raise OSError("forced append failure")

    monkeypatch.setattr(journal, "append", fail_append)

    try:
        store.put("new-key", "new-fingerprint", _Reply(value="new"))
    except OSError as exc:
        assert str(exc) == "forced append failure"
    else:  # pragma: no cover - makes a swallowed durability fault explicit
        raise AssertionError("journal failure did not propagate")

    old = store.get("old-key")
    assert old is not None
    assert old[0] == "old-fingerprint"
    assert old[1].value == "old"
    assert store.get("new-key") is None

    restarted = _IdemStore(1, journal=JobJournal(path), model=_Reply)
    recovered = restarted.get("old-key")
    assert recovered is not None
    assert recovered[0] == "old-fingerprint"
    assert recovered[1].value == "old"
    assert restarted.get("new-key") is None


def test_put_publishes_an_immutable_snapshot(tmp_path) -> None:
    path = tmp_path / "idem.jsonl"
    store = _IdemStore(1, journal=JobJournal(path), model=_Reply)
    response = _Reply(value="original")

    store.put("key", "fingerprint", response)
    response.value = "mutated-after-put"

    live = store.get("key")
    assert live is not None
    assert live[1].value == "original"

    restarted = _IdemStore(1, journal=JobJournal(path), model=_Reply)
    recovered = restarted.get("key")
    assert recovered is not None
    assert recovered[1].value == "original"


def test_file_upload_rolls_back_when_idem_journal_append_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FX1_API_KEY", raising=False)
    original_append = JobJournal.append
    failed = False

    def fail_idem_once(journal: JobJournal, payload: dict[str, object]) -> None:
        nonlocal failed
        if journal.path.name == "idem_uploads.jsonl" and not failed:
            failed = True
            raise OSError("forced idem append failure")
        original_append(journal, payload)

    monkeypatch.setattr(JobJournal, "append", fail_idem_once)
    app = create_app(state_dir=tmp_path, max_inflight=1)
    files = {"file": ("batch.jsonl", b'{"custom_id":"1"}\n', "application/jsonl")}
    headers = {"Idempotency-Key": "retry-file"}

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/v1/files", files=files, data={"purpose": "batch"}, headers=headers)
        after_failure = client.get("/v1/files").json()
        retry = client.post("/v1/files", files=files, data={"purpose": "batch"}, headers=headers)
        after_retry = client.get("/v1/files").json()

    assert first.status_code == 500
    assert after_failure["data"] == []
    assert retry.status_code == 200
    assert retry.headers.get("X-Fx1-Idempotent-Replay") is None
    assert [item["id"] for item in after_retry["data"]] == [retry.json()["id"]]


def test_replay_bypasses_capacity_and_drain(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FX1_API_KEY", raising=False)
    backend = _Backend()
    app = create_app(
        backend_resolver=lambda *args, **kwargs: backend,
        state_dir=tmp_path,
        max_inflight=1,
    )
    body = {
        "backend": "hosted_k3",
        "messages": [{"role": "user", "content": "question"}],
    }
    headers = {"Idempotency-Key": "available-replay"}

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post("/harness/complete", json=body, headers=headers)
        assert first.status_code == 200
        assert "available-replay" not in (tmp_path / "idem_complete.jsonl").read_text()

        app.state.inflight_slots.acquire()
        try:
            saturated = client.post("/harness/complete", json=body, headers=headers)
        finally:
            app.state.inflight_slots.release()

        app.state.metrics.draining.set()
        draining = client.post("/harness/complete", json=body, headers=headers)

    assert saturated.status_code == 200
    assert saturated.json()["replayed"] is True
    assert draining.status_code == 200
    assert draining.json()["replayed"] is True
    assert backend.calls == 1
