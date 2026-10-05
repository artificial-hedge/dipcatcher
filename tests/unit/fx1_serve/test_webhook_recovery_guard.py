"""Recovered records cannot send unsigned duplicate webhooks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fx1.serve import api, webhook_audit
from fx1.serve.evals import EvalRecord, EvalStore
from fx1.serve.finetune import FTJob, FTJobStore
from fx1.serve.journal import JobJournal

_SURFACES = ("job", "eval", "fine_tune", "batch", "anthropic_batch")


def _case(surface: str, *, terminal: bool) -> tuple[Any, Any, str, str]:
    callback = {"callback_url": "http://127.0.0.1:9999/callback", "created_at": 100}
    status = "cancelled" if terminal else "queued"
    if surface == "job":
        return (
            api.JobStatusResponse(
                job_id="job_test",
                status=status,
                finished_at=None,
                result=None,
                error=None,
                **callback,
            ),
            api._JobStore,
            "job",
            "job_test",
        )
    if surface == "eval":
        return (
            EvalRecord(
                eval_id="eval_test",
                suite="calibration",
                backend="stub",
                seed=0,
                status=status,
                **callback,
            ),
            EvalStore,
            "record",
            "eval_test",
        )
    if surface == "fine_tune":
        return (
            FTJob(id="ft_test", model="fx1", training_file="file_test", status=status, **callback),
            FTJobStore,
            "ft_job",
            "ft_test",
        )
    if surface == "batch":
        return (
            api._BatchRecord(
                batch_id="batch_test",
                input_file_id="file_test",
                endpoint="/v1/chat/completions",
                completion_window="24h",
                expires_at=200,
                status="cancelled" if terminal else "in_progress",
                **callback,
            ),
            api._BatchStore,
            "batch",
            "batch_test",
        )
    return (
        api._AnthropicBatchRecord(
            batch_id="msgbatch_test",
            expires_at=200,
            status="ended" if terminal else "in_progress",
            **callback,
        ),
        api._AnthropicBatchStore,
        "batch",
        "msgbatch_test",
    )


def _restore(store_type: Any, path: Path, identity: str) -> Any:
    store = store_type(8, journal=JobJournal(path))
    recovered = store.get(identity)
    assert recovered is not None
    return recovered.job if isinstance(store, FTJobStore) else recovered


@pytest.mark.parametrize("surface", _SURFACES)
@pytest.mark.parametrize("secret", [None, "synthetic-callback-secret"])
def test_callback_fires_once_and_never_repeats_after_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, secret: str | None
) -> None:
    record, store_type, payload_key, identity = _case(surface, terminal=True)
    record._callback_secret = secret
    path = tmp_path / "records.jsonl"
    # Persist the transition before delivery, matching a process that exits
    # before its post-delivery bookkeeping reaches the journal.
    JobJournal(path).append({payload_key: record.model_dump(mode="json")})
    calls: list[str | None] = []

    def deliver(_url: str, key: str | None, _body: bytes) -> tuple[bool, None, int]:
        calls.append(key)
        return True, None, 1

    monkeypatch.setattr(api, "deliver_signed", deliver)
    api._deliver_callback(record)
    api._deliver_callback(record)
    assert calls == [secret]

    recovered = _restore(store_type, path, identity)
    assert recovered._callback_secret is None
    api._deliver_callback(recovered)
    api._deliver_callback(recovered)
    assert calls == [secret]
    assert recovered.callback_status is None
    assert recovered.callback_attempts == 0
    assert "synthetic-callback-secret" not in path.read_text()


@pytest.mark.parametrize("surface", _SURFACES)
def test_unfinished_recovered_record_never_attempts_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    record, store_type, payload_key, identity = _case(surface, terminal=False)
    path = tmp_path / "records.jsonl"
    JobJournal(path).append({payload_key: record.model_dump(mode="json")})
    calls: list[str] = []

    def deliver(url: str, _key: str | None, _body: bytes) -> tuple[bool, None, int]:
        calls.append(url)
        return True, None, 1

    monkeypatch.setattr(api, "deliver_signed", deliver)
    recovered = _restore(store_type, path, identity)
    assert recovered.status in {"failed", "ended"}
    api._deliver_callback(recovered)
    assert calls == []
    assert recovered.callback_status is None
    assert recovered.callback_attempts == 0


def test_unsigned_probe_waits_for_the_submitted_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits: list[Any] = []
    sink = SimpleNamespace(hits=hits, url=lambda path: "http://127.0.0.1" + path)
    ctx = SimpleNamespace(sink=sink, client=object())
    waits: list[int] = []

    def submit(_client: Any, _url: str) -> dict[str, str]:
        hits.append(SimpleNamespace(headers={}, path="/unsigned"))
        return {"job_id": "job_test"}

    monkeypatch.setattr(webhook_audit, "_submit_job", submit)
    monkeypatch.setattr(webhook_audit, "_wait_job", lambda *_args: {})
    monkeypatch.setattr(webhook_audit, "_wait_hits", lambda _sink, count: waits.append(count))

    assert all(webhook_audit._probe_unsigned(ctx).values())
    assert waits == [1]
