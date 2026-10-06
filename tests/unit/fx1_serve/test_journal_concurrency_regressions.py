"""Regression checks for journal and idempotency synchronization."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import journal as journal_mod
from fx1.serve.journal import JobJournal


def test_parallel_appends_preserve_every_payload(tmp_path: Path) -> None:
    journal = JobJournal(tmp_path / "jobs.jsonl")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: journal.append({"id": i}), range(64)))
    replay = journal.replay()
    assert replay.truncated_at is None
    assert len(replay.payloads) == 64
    assert {p["id"] for p in replay.payloads} == set(range(64))


def test_replay_cannot_rewind_concurrent_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old replay snapshot must not reset counters after a new append."""
    journal = JobJournal(tmp_path / "jobs.jsonl")
    journal.append({"id": "first"})
    snapshot_read = threading.Event()
    release_snapshot = threading.Event()
    append_finished = threading.Event()
    real_open = Path.open

    def paused_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        fh = real_open(path, *args, **kwargs)
        if path == journal.path and not snapshot_read.is_set():
            snapshot_read.set()
            assert release_snapshot.wait(2)
        return fh

    monkeypatch.setattr(Path, "open", paused_open)

    def append() -> None:
        journal.append({"id": "second"})
        append_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        replay_future = pool.submit(journal.replay)
        assert snapshot_read.wait(2)
        append_future = pool.submit(append)
        append_finished.wait(0.2)
        release_snapshot.set()
        replay_future.result(timeout=2)
        append_future.result(timeout=2)

    journal.append({"id": "third"})
    replay = journal.replay()
    assert replay.truncated_at is None
    assert replay.payloads == [{"id": "first"}, {"id": "second"}, {"id": "third"}]


def test_parallel_append_sequence_is_read_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pause serialization at the old pre-lock race window."""
    journal = JobJournal(tmp_path / "jobs.jsonl")
    rendezvous = threading.Barrier(2)
    line_bytes = journal_mod._line_bytes

    def paused_line(seq: int, chain: str, payload: dict[str, object]) -> bytes:
        with suppress(threading.BrokenBarrierError):
            rendezvous.wait(timeout=0.2)
        return line_bytes(seq, chain, payload)

    monkeypatch.setattr(journal_mod, "_line_bytes", paused_line)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(journal.append, {"id": i}) for i in range(2)]
        for future in futures:
            future.result(timeout=2)
    replay = journal.replay()
    assert replay.truncated_at is None
    assert {p["id"] for p in replay.payloads} == {0, 1}
