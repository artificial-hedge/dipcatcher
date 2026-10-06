"""Failure-atomicity regressions for vector-store lifecycle journaling."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from fx1.serve.journal import JobJournal
from fx1.serve.vectorstores import VectorStoreError, VectorStoreStore


class _BlockingReader:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, file_id: str) -> tuple[bytes, str]:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise AssertionError("test did not release the file reader")
        return b"alpha beta gamma", f"{file_id}.txt"


def _fail_append(_payload: dict[str, Any]) -> None:
    raise OSError("synthetic journal failure")


def _ids(store: VectorStoreStore) -> set[str]:
    return {str(row["id"]) for row in store.list_stores()["data"]}


def _lru_ids(store: VectorStoreStore) -> list[str]:
    return [str(row["id"]) for row in store.list_stores(order="asc")["data"]]


def _wait_for_waiters(store: VectorStoreStore, count: int) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with store._condition:
            if len(vars(store._condition)["_waiters"]) >= count:
                return
        time.sleep(0.005)
    raise AssertionError(f"expected {count} condition waiter(s)")


def test_failed_create_publishes_no_phantom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vector_stores.jsonl"
    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal)
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.create(name="phantom")

    assert _ids(store) == set()
    assert _ids(VectorStoreStore(2, journal=JobJournal(path))) == set()


def test_failed_capacity_create_preserves_victim_and_derived_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vector_stores.jsonl"

    def reader(file_id: str) -> tuple[bytes, str]:
        return b"alpha beta gamma", f"{file_id}.txt"

    journal = JobJournal(path)
    store = VectorStoreStore(1, journal=journal, file_reader=reader)
    victim_id = str(store.create(name="victim")["id"])
    store.attach(victim_id, "file-a")
    assert set(store._chunks) == {victim_id}
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.create(name="phantom replacement")

    assert _ids(store) == {victim_id}
    assert store.get(victim_id)["file_counts"]["completed"] == 1
    assert set(store._chunks) == {victim_id}
    restored = VectorStoreStore(1, journal=JobJournal(path), file_reader=reader)
    assert _ids(restored) == {victim_id}
    assert restored.get(victim_id)["file_counts"]["completed"] == 1


def test_rejected_capacity_create_with_member_preserves_victim(tmp_path: Path) -> None:
    path = tmp_path / "vector_stores.jsonl"

    def reader(file_id: str) -> tuple[bytes, str] | None:
        if file_id == "missing":
            return None
        return b"alpha beta gamma", f"{file_id}.txt"

    journal = JobJournal(path)
    store = VectorStoreStore(1, journal=journal, file_reader=reader)
    victim_id = str(store.create(name="victim")["id"])
    before = journal.replay().payloads

    with pytest.raises(VectorStoreError, match="not found"):
        store.create(name="rejected", file_ids=["missing"])

    assert _ids(store) == {victim_id}
    assert journal.replay().payloads == before
    assert _ids(VectorStoreStore(1, journal=JobJournal(path), file_reader=reader)) == {victim_id}


def test_create_members_and_replay_record_commit_in_one_journal_line(tmp_path: Path) -> None:
    path = tmp_path / "vector_stores.jsonl"

    def reader(file_id: str) -> tuple[bytes, str]:
        return b"alpha beta gamma", f"{file_id}.txt"

    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal, file_reader=reader)
    response = store.create(
        name="atomic",
        file_ids=["file-a", "file-b"],
        idempotency_key="scoped-key",
        body_fingerprint="body-fp",
    )

    payloads = journal.replay().payloads
    assert len(payloads) == 1
    assert set(payloads[0]) >= {"vs", "vs_files", "vs_touch", "vs_idem"}
    assert [row["file_id"] for row in payloads[0]["vs_files"]] == ["file-a", "file-b"]
    assert store.idempotency_get("scoped-key") == ("body-fp", response)

    restored = VectorStoreStore(2, journal=JobJournal(path), file_reader=reader)
    assert restored.get(str(response["id"])) == response
    assert restored.idempotency_get("scoped-key") == ("body-fp", response)


def test_failed_update_is_not_visible_and_does_not_revert_at_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vector_stores.jsonl"
    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal)
    vs_id = str(store.create(name="before", metadata={"version": "before"})["id"])
    other_id = str(store.create(name="other")["id"])
    original_lru = [vs_id, other_id]
    assert _lru_ids(store) == original_lru
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.update(vs_id, name="after", metadata={"version": "after"})

    assert _lru_ids(store) == original_lru
    assert store.get(vs_id)["name"] == "before"
    assert store.get(vs_id)["metadata"] == {"version": "before"}
    restored = VectorStoreStore(2, journal=JobJournal(path))
    assert restored.get(vs_id)["name"] == "before"
    assert restored.get(vs_id)["metadata"] == {"version": "before"}


def test_failed_delete_does_not_resurrect_at_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vector_stores.jsonl"
    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal)
    vs_id = str(store.create(name="survivor")["id"])
    other_id = str(store.create(name="other")["id"])
    original_lru = [vs_id, other_id]
    assert _lru_ids(store) == original_lru
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.delete(vs_id)

    assert _lru_ids(store) == original_lru
    assert _ids(store) == {vs_id, other_id}
    assert _ids(VectorStoreStore(2, journal=JobJournal(path))) == {vs_id, other_id}


def test_failed_delete_does_not_wake_capacity_waiter_or_exceed_replay_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vector_stores.jsonl"
    reader = _BlockingReader()
    journal = JobJournal(path)
    store = VectorStoreStore(2, journal=journal, file_reader=reader)

    with ThreadPoolExecutor(max_workers=3) as pool:
        pinned_id = str(store.create(name="pinned")["id"])
        attaching: Future[dict[str, Any]] = pool.submit(store.attach, pinned_id, "file-a")
        assert reader.started.wait(timeout=2)
        other_id = str(store.create(name="other")["id"])
        creator: Future[dict[str, Any]] = pool.submit(store.create, name="replacement")
        _wait_for_waiters(store, 1)

        notifications = 0
        real_notify_all = store._condition.notify_all

        def count_notify_all() -> None:
            nonlocal notifications
            notifications += 1
            real_notify_all()

        monkeypatch.setattr(store._condition, "notify_all", count_notify_all)
        real_append = journal.append

        def fail_delete(payload: dict[str, Any]) -> None:
            if "vs_delete" in payload:
                raise OSError("synthetic journal failure")
            real_append(payload)

        monkeypatch.setattr(journal, "append", fail_delete)
        deleted = pool.submit(store.delete, other_id)
        with pytest.raises(OSError, match="synthetic journal failure"):
            deleted.result(timeout=2)

        # A rejected delete has not freed real or apparent capacity.
        assert notifications == 0
        assert _ids(store) == {pinned_id, other_id}
        assert not creator.done()

        reader.release.set()
        attaching.result(timeout=3)
        replacement_id = str(creator.result(timeout=3)["id"])

    # Completing the attachment refreshes the pinned store's LRU position,
    # so the subsequently-created replacement legitimately evicts ``other``.
    assert _ids(store) == {pinned_id, replacement_id}
    restored = VectorStoreStore(2, journal=JobJournal(path), file_reader=reader)
    assert _ids(restored) == {pinned_id, replacement_id}
    assert len(restored._stores) == restored.max_stores
