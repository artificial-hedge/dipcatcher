"""SYNTHETIC adversarial probes for the fine-tuning job store — the
atomic queued→running claim (cancel/pause can never be overwritten back
to running), eviction releasing parked workers, and fail-closed cursors
on every paged listing."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from fx1.serve.finetune import FTJob, FTJobStore, validate_chat_jsonl
from fx1.serve.journal import JobJournal


def _store(path: Path | None = None, max_entries: int = 4) -> FTJobStore:
    journal = JobJournal(path) if path is not None else None
    return FTJobStore(max_entries=max_entries, journal=journal)


def _job(idx: str = "ftjob-a") -> FTJob:
    return FTJob(id=idx, model="fx1", created_at=1, training_file="file-t")


def test_start_loses_to_cancel_deterministically(tmp_path: Path) -> None:
    """The worker claim is atomic: a cancel that lands before the claim
    is never overwritten back to running — the resurrection class
    fixed in #2822."""
    store = _store(tmp_path / "ft.jsonl")
    entry = store.put(_job(), None, "")
    assert store.request_cancel(entry.job.id) == "queued"
    assert entry.job.status == "cancelled"
    assert store.start(entry) is False
    assert entry.job.status == "cancelled"

    revived = _store(tmp_path / "ft.jsonl")
    assert revived.get(entry.job.id).job.status == "cancelled"


def test_start_loses_to_pause_and_parks(tmp_path: Path) -> None:
    """A pause that lands before the claim leaves the job ``paused`` —
    start refuses, the worker would park on the resume gate."""
    store = _store()
    entry = store.put(_job(), None, "")
    assert store.request_pause(entry.job.id) == "queued"
    assert entry.job.status == "paused"
    assert store.start(entry) is False
    assert entry.job.status == "paused"
    # resume restores the queued claim — start then succeeds once
    assert store.request_resume(entry.job.id) == "queued"
    assert store.start(entry) is True
    assert entry.job.status == "running"
    # a second start refuses — running is not queued
    assert store.start(entry) is False


def test_start_loses_to_eviction_and_parked_workers_release(tmp_path: Path) -> None:
    """Eviction drops the entry AND opens the resume gate — a worker
    parked pre-claim observes the verdict and exits instead of pinning
    an inflight slot forever."""
    store = _store(max_entries=1)
    entry = store.put(_job("ftjob-parked"), None, "")
    assert store.request_pause(entry.job.id) == "queued"
    waiters: list[bool] = []

    def _parked() -> None:
        # the worker's loop: start refuses paused -> wait on resume
        entry.resume.wait()
        waiters.append(True)

    worker = threading.Thread(target=_parked, daemon=True)
    worker.start()
    entry2 = store.put(_job("ftjob-churn"), None, "")  # evicts the parked entry
    assert entry2.job.id == "ftjob-churn"
    worker.join(timeout=2)
    assert waiters == [True]  # the gate opened — no leaked worker
    assert store.start(entry) is False  # evicted: the verdict stands


def test_running_cancel_is_flag_only(tmp_path: Path) -> None:
    """A running job's cancel is cooperative: status stays ``running``
    and the worker owns the terminal write."""
    store = _store(tmp_path / "ft.jsonl")
    entry = store.put(_job(), None, "")
    assert store.start(entry) is True
    assert store.request_cancel(entry.job.id) == "running"
    assert entry.job.status == "running"
    assert entry.cancel.is_set()


def test_queued_cancel_durable_across_restart(tmp_path: Path) -> None:
    store = _store(tmp_path / "ft.jsonl")
    entry = store.put(_job(), None, "")
    store.request_cancel(entry.job.id)
    revived = _store(tmp_path / "ft.jsonl")
    assert revived.get(entry.job.id).job.status == "cancelled"


@pytest.mark.parametrize(
    "method", ["list_jobs", "list_events", "checkpoints_for"], ids=["jobs", "events", "ckpts"]
)
def test_unknown_cursor_fails_closed(tmp_path: Path, method: str) -> None:
    """A mistyped cursor must never masquerade as end-of-list — the same
    contract eval specs and vector stores already hold."""
    store = _store(tmp_path / "ft.jsonl")
    entry = store.put(_job(), None, "")
    with pytest.raises(ValueError, match="cursor"):
        if method == "list_jobs":
            store.list_jobs(limit=10, after="ftjob-bogus")
        elif method == "list_events":
            store.list_events(entry.job.id, limit=10, after="ftev-bogus")
        else:
            store.checkpoints_for(entry.job.id, limit=10, after="ftckpt-bogus")


def test_validate_chat_jsonl_fails_closed() -> None:
    good = b'{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n'
    assert validate_chat_jsonl(good, file_id="file-t") == 1
    for bad in (
        b"not json\n",
        b"[1,2]\n",
        b'{"messages":[]}\n',
        b'{"messages":[{"role":"hacker","content":"x"}]}\n',
        b'{"messages":[{"role":"user","content":""}]}\n',
        b'{"messages":[{"role":"user","content":5}]}\n',
        b"",
    ):
        with pytest.raises(ValueError):
            validate_chat_jsonl(bad, file_id="file-t")


def test_queued_jobs_recover_failed_not_running(tmp_path: Path) -> None:
    """A job non-terminal at crash never resurrects as queued/running —
    it fails with a restart-explaining error."""
    store = _store(tmp_path / "ft.jsonl")
    entry = store.put(_job(), None, "")
    revived = _store(tmp_path / "ft.jsonl")
    job = revived.get(entry.job.id).job
    assert job.status == "failed"
    assert job.error is not None and "restart" in job.error.message
