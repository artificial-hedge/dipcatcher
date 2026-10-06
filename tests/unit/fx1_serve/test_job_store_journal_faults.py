"""SYNTHETIC fault injection for job-store durability and admission ownership."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from fx1.harness import Harness
from fx1.serve import api
from fx1.serve.journal import JobJournal


def _job(job_id: str) -> api.JobStatusResponse:
    return api.JobStatusResponse(
        job_id=job_id,
        status="queued",
        created_at=time.time(),
        finished_at=None,
        result=None,
        error=None,
    )


def _assert_capacity_released(metrics: api._Metrics, slots: threading.BoundedSemaphore) -> None:
    assert metrics.snapshot().inflight == 0
    assert slots.acquire(blocking=False)
    slots.release()


def test_failed_put_does_not_publish_or_evict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal = JobJournal(tmp_path / "jobs.jsonl")
    store = api._JobStore(1, journal)
    old = _job("old")
    store.put(old, "old-key", "old-fp")

    def fail(payload: dict[str, Any]) -> None:
        raise OSError("SYNTHETIC journal failure")

    monkeypatch.setattr(journal, "append", fail)
    with pytest.raises(OSError, match="SYNTHETIC journal failure"):
        store.put(_job("new"), "new-key", "new-fp")

    assert [record.job_id for record in store.list()] == ["old"]
    assert store.get_key("old-key") == ("old-fp", "old")
    assert store.get_key("new-key") is None


def test_failed_start_leaves_job_queued(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    journal = JobJournal(tmp_path / "jobs.jsonl")
    store = api._JobStore(2, journal)
    store.put(_job("queued"), None, None)

    def fail(payload: dict[str, Any]) -> None:
        raise OSError("SYNTHETIC journal failure")

    monkeypatch.setattr(journal, "append", fail)
    with pytest.raises(OSError, match="SYNTHETIC journal failure"):
        store.start("queued")
    assert store.get("queued").status == "queued"


@pytest.mark.parametrize("fail_at", [1, 2])
def test_submission_journal_failure_never_leaks_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail_at: int
) -> None:
    journal = JobJournal(tmp_path / "jobs.jsonl")
    append = journal.append
    calls = 0

    def fail_once(payload: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("SYNTHETIC journal failure")
        append(payload)

    monkeypatch.setattr(journal, "append", fail_once)
    store = api._JobStore(2, journal)
    metrics = api._Metrics(1)
    slots = threading.BoundedSemaphore(1)
    pending: list[Callable[[], None]] = []

    class _Executor:
        def submit(self, fn: Callable[[], None]) -> Future[None]:
            pending.append(fn)
            return Future()

    def submit() -> api.JobSubmitResponse:
        return api._submit_job(
            api.HarnessRunRequest(command="doctor"),
            None,
            Harness(runner=lambda argv, timeout_s: (0, "ok", "")),
            store,
            metrics,
            slots,
            _Executor(),  # type: ignore[arg-type]
        )

    if fail_at == 1:
        with pytest.raises(OSError, match="SYNTHETIC journal failure"):
            submit()
        assert not pending
    else:
        submitted = submit()
        with pytest.raises(OSError, match="SYNTHETIC journal failure"):
            pending.pop()()
        assert store.get(submitted.job_id).status == "queued"
    _assert_capacity_released(metrics, slots)


def test_executor_refusal_releases_capacity_when_tombstone_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    journal = JobJournal(tmp_path / "jobs.jsonl")
    append = journal.append

    def fail_tombstone(payload: dict[str, Any]) -> None:
        if "deleted" in payload:
            raise OSError("SYNTHETIC tombstone failure")
        append(payload)

    class _ClosedExecutor:
        def submit(self, fn: Callable[[], None]) -> Future[None]:
            raise RuntimeError("executor closed")

    monkeypatch.setattr(journal, "append", fail_tombstone)
    store = api._JobStore(2, journal)
    metrics = api._Metrics(1)
    slots = threading.BoundedSemaphore(1)
    with pytest.raises(OSError, match="SYNTHETIC tombstone failure"):
        api._submit_job(
            api.HarnessRunRequest(command="doctor"),
            None,
            Harness(runner=lambda argv, timeout_s: (0, "ok", "")),
            store,
            metrics,
            slots,
            _ClosedExecutor(),  # type: ignore[arg-type]
        )
    _assert_capacity_released(metrics, slots)
