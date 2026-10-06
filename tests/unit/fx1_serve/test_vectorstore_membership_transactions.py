"""Failure-atomicity regressions for vector-store file membership."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.journal import JobJournal
from fx1.serve.vectorstores import VectorStoreError, VectorStoreStore


def _reader(file_id: str) -> tuple[bytes, str] | None:
    if file_id == "missing":
        return None
    return f"searchable text for {file_id}".encode(), f"{file_id}.txt"


def _fail_append(_payload: dict[str, Any]) -> None:
    raise OSError("synthetic journal failure")


def _attached_ids(store: VectorStoreStore, vs_id: str) -> set[str]:
    return set(store._stores[vs_id].files)


def test_failed_attach_publishes_no_file_or_derived_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vector-stores.jsonl"
    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal, file_reader=_reader)
    vs_id = str(store.create(name="files")["id"])
    before = store._stores[vs_id].model_dump(mode="json")
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.attach(vs_id, "file-a")

    assert store._stores[vs_id].model_dump(mode="json") == before
    assert store._chunks.get(vs_id, {}) == {}
    restored = VectorStoreStore(2, journal=JobJournal(path), file_reader=_reader)
    assert _attached_ids(restored, vs_id) == set()


def test_failed_detach_preserves_file_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vector-stores.jsonl"
    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal, file_reader=_reader)
    vs_id = str(store.create(name="files")["id"])
    store.attach(vs_id, "file-a")
    before_chunks = list(store._chunks[vs_id]["file-a"])
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.detach(vs_id, "file-a")

    assert _attached_ids(store, vs_id) == {"file-a"}
    assert store._chunks[vs_id]["file-a"] == before_chunks
    restored = VectorStoreStore(2, journal=JobJournal(path), file_reader=_reader)
    assert _attached_ids(restored, vs_id) == {"file-a"}


def test_failed_touch_preserves_activity_and_lru_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vector-stores.jsonl"
    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal, file_reader=_reader)
    first = str(
        store.create(
            name="first",
            expires_after={"anchor": "last_active_at", "days": 1},
        )["id"]
    )
    second = str(store.create(name="second")["id"])
    meta = store._stores[first]
    before_activity = (meta.last_active_at, meta.expires_at)
    assert list(store._stores) == [first, second]
    monkeypatch.setattr("fx1.serve.vectorstores.time.time", lambda: before_activity[0] + 1)
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.search([first], "searchable")

    assert (meta.last_active_at, meta.expires_at) == before_activity
    assert list(store._stores) == [first, second]
    restored = VectorStoreStore(2, journal=JobJournal(path), file_reader=_reader)
    restored_meta = restored._stores[first]
    assert (restored_meta.last_active_at, restored_meta.expires_at) == before_activity


def test_failed_batch_record_publishes_neither_members_nor_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vector-stores.jsonl"
    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal, file_reader=_reader)
    vs_id = str(store.create(name="batch")["id"])
    original_append = journal.append

    def fail_batch(payload: dict[str, Any]) -> None:
        if "vs_batch" in payload:
            raise OSError("synthetic batch journal failure")
        original_append(payload)

    monkeypatch.setattr(journal, "append", fail_batch)
    with pytest.raises(OSError, match="synthetic batch journal failure"):
        store.file_batch_create(vs_id, ["file-a", "file-b"])

    assert _attached_ids(store, vs_id) == set()
    assert store._batches.get(vs_id, {}) == {}
    restored = VectorStoreStore(2, journal=JobJournal(path), file_reader=_reader)
    assert _attached_ids(restored, vs_id) == set()
    assert restored._batches.get(vs_id, {}) == {}


def test_batch_commits_members_batch_and_touch_in_one_record(tmp_path: Path) -> None:
    path = tmp_path / "vector-stores.jsonl"
    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal, file_reader=_reader)
    vs_id = str(store.create(name="batch")["id"])
    before_records = len(journal.replay().payloads)

    batch = store.file_batch_create(vs_id, ["file-a", "missing", "file-b"])

    payloads = journal.replay().payloads
    assert len(payloads) == before_records + 1
    assert set(payloads[-1]) >= {"vs_files", "vs_batch", "vs_touch"}
    assert batch["file_counts"] == {
        "in_progress": 0,
        "completed": 2,
        "failed": 1,
        "cancelled": 0,
        "total": 3,
    }
    restored = VectorStoreStore(2, journal=JobJournal(path), file_reader=_reader)
    assert _attached_ids(restored, vs_id) == {"file-a", "file-b"}
    assert restored.file_batch_get(vs_id, str(batch["id"])) == batch


def test_batch_serializes_competing_membership_writers(tmp_path: Path) -> None:
    path = tmp_path / "vector-stores.jsonl"
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def blocking_reader(file_id: str) -> tuple[bytes, str]:
        calls.append(file_id)
        if len(calls) == 1:
            started.set()
            if not release.wait(timeout=5):
                raise AssertionError("test did not release batch reader")
        return b"searchable content", f"{file_id}.txt"

    store = VectorStoreStore(2, journal=JobJournal(path), file_reader=blocking_reader)
    vs_id = str(store.create(name="batch")["id"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        batch_future = pool.submit(store.file_batch_create, vs_id, ["file-a"])
        assert started.wait(timeout=5)
        attach_future = pool.submit(store.attach, vs_id, "file-a")
        time.sleep(0.05)
        assert not attach_future.done()
        assert calls == ["file-a"]
        release.set()
        batch = batch_future.result(timeout=5)
        with pytest.raises(VectorStoreError, match="already attached"):
            attach_future.result(timeout=5)

    assert batch["file_counts"]["completed"] == 1
    assert _attached_ids(store, vs_id) == {"file-a"}
    restored = VectorStoreStore(2, journal=JobJournal(path), file_reader=_reader)
    assert _attached_ids(restored, vs_id) == {"file-a"}
