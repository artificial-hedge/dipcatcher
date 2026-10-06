"""Deterministic lifecycle-race regressions for the vector-store state machine."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fx1.serve.vectorstores import VectorStoreStore


class _BlockingReader:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, file_id: str) -> tuple[bytes, str]:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise AssertionError("test did not release the file reader")
        return b"alpha beta gamma", f"{file_id}.txt"


def test_zero_store_capacity_fails_fast() -> None:
    """A zero-capacity store must not return phantom, immediately-evicted IDs."""
    with pytest.raises(ValueError, match="max_stores"):
        VectorStoreStore(0)


def test_delete_cannot_split_file_batch_lifecycle() -> None:
    """A delete must not interleave between batch lookup and publication."""
    reader = _BlockingReader()
    store = VectorStoreStore(8, file_reader=reader)
    vs_id = store.create(name="race")["id"]
    delete_done = threading.Event()

    def delete() -> dict[str, object]:
        result = store.delete(vs_id)
        delete_done.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        batch_future = pool.submit(store.file_batch_create, vs_id, ["file-a"])
        assert reader.started.wait(timeout=2)
        # The lifecycle pin is per store: unrelated work must not wait on a
        # slow reader/index build.
        unrelated = store.create(name="unrelated")
        delete_future = pool.submit(delete)
        delete_finished_during_batch = delete_done.wait(timeout=0.25)
        reader.release.set()
        batch = batch_future.result(timeout=3)
        deleted = delete_future.result(timeout=3)

    assert not delete_finished_during_batch
    assert batch["status"] == "completed"
    assert unrelated["name"] == "unrelated"
    assert deleted["deleted"] is True
    assert vs_id not in store._batches


def test_capacity_eviction_cannot_split_create_with_files() -> None:
    """create(file_ids=...) is one lifecycle operation at the capacity boundary."""
    reader = _BlockingReader()
    store = VectorStoreStore(1, file_reader=reader)
    second_done = threading.Event()

    def create_second() -> dict[str, object]:
        result = store.create(name="second")
        second_done.set()
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(store.create, name="first", file_ids=["file-a"])
        assert reader.started.wait(timeout=2)
        second_future = pool.submit(create_second)
        second_finished_during_first = second_done.wait(timeout=0.25)
        reader.release.set()
        first = first_future.result(timeout=3)
        second = second_future.result(timeout=3)

    assert not second_finished_during_first
    assert first["file_counts"]["completed"] == 1
    assert second["name"] == "second"
    assert store.list_stores()["data"][0]["id"] == second["id"]
