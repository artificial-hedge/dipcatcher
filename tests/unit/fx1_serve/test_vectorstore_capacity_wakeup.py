"""Store-only regressions for capacity released during a pinned file read."""

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fx1.serve.vectorstores import VectorStoreStore


class _BlockingReader:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, file_id: str) -> tuple[bytes, str]:
        self.started.set()
        if not self.release.wait(timeout=30):
            raise AssertionError("test did not release the file reader")
        return b"alpha beta gamma", f"{file_id}.txt"


class TestVectorStoreCapacityWakeup(unittest.TestCase):
    def _wait_for_waiters(self, store: VectorStoreStore, count: int) -> None:
        # Observe the real CPython Condition without replacing its wait or
        # notification behavior. Merely starting a thread is not evidence
        # that it reached the capacity wait before deletion.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with store._condition:
                if len(vars(store._condition)["_waiters"]) >= count:
                    return
            time.sleep(0.005)
        self.fail(f"expected {count} condition waiter(s)")

    def _created_while_reader_blocked(
        self, future: Future[dict[str, Any]], reader: _BlockingReader
    ) -> dict[str, Any]:
        done, _ = wait([future], timeout=2)
        self.assertIn(future, done, "creation stayed blocked after deletion freed capacity")
        self.assertFalse(reader.release.is_set())
        return future.result()

    def _check_delete_during_read(self, *, batch: bool) -> None:
        reader = _BlockingReader()
        with TemporaryDirectory() as directory:
            path = Path(directory)
            store = VectorStoreStore(2, state_dir=path, file_reader=reader)
            with ThreadPoolExecutor(max_workers=2) as pool:
                try:
                    if batch:
                        pinned_id = store.create(name="pinned")["id"]
                        pinned = pool.submit(store.file_batch_create, pinned_id, ["file-a"])
                    else:
                        pinned = pool.submit(store.create, name="pinned", file_ids=["file-a"])
                    self.assertTrue(reader.started.wait(timeout=3))
                    pinned_id = store.list_stores()["data"][0]["id"]
                    # Created after the pin starts, so the pinned store is
                    # the oldest even though batch lookup refreshes its LRU.
                    other = store.create(name="other")
                    creator = pool.submit(store.create, name="replacement")
                    self._wait_for_waiters(store, 1)

                    self.assertTrue(store.delete(other["id"])["deleted"])
                    replacement = self._created_while_reader_blocked(creator, reader)
                    self.assertFalse(pinned.done())
                    expected = {pinned_id, replacement["id"]}
                    self.assertEqual({row["id"] for row in store.list_stores()["data"]}, expected)
                finally:
                    # This also wakes the baseline's stranded creator before
                    # the executor joins, so a failed regression cannot hang.
                    reader.release.set()
                pinned.result(timeout=3)
                creator.result(timeout=3)

            restored = VectorStoreStore(2, state_dir=path, file_reader=reader)
            self.assertEqual({row["id"] for row in restored.list_stores()["data"]}, expected)
            self.assertEqual(restored.get(pinned_id)["file_counts"]["completed"], 1)
            self.assertEqual(restored.recover_warnings, [])

    def test_delete_wakes_create_during_create_with_files(self) -> None:
        self._check_delete_during_read(batch=False)

    def test_delete_wakes_create_during_file_batch(self) -> None:
        self._check_delete_during_read(batch=True)

    def test_delete_wakes_creator_behind_a_delete_waiter(self) -> None:
        reader = _BlockingReader()
        store = VectorStoreStore(2, file_reader=reader)
        with ThreadPoolExecutor(max_workers=3) as pool:
            try:
                pinned = pool.submit(store.create, name="pinned", file_ids=["file-a"])
                self.assertTrue(reader.started.wait(timeout=3))
                pinned_id = store.list_stores()["data"][0]["id"]
                deleter = pool.submit(store.delete, pinned_id)
                self._wait_for_waiters(store, 1)
                other = store.create(name="other")
                creator = pool.submit(store.create, name="replacement")
                self._wait_for_waiters(store, 2)

                store.delete(other["id"])
                replacement = self._created_while_reader_blocked(creator, reader)
                self.assertFalse(pinned.done())
                self.assertFalse(deleter.done())
                self.assertEqual(
                    {row["id"] for row in store.list_stores()["data"]},
                    {pinned_id, replacement["id"]},
                )
            finally:
                reader.release.set()
            pinned.result(timeout=3)
            self.assertTrue(deleter.result(timeout=3)["deleted"])
            creator.result(timeout=3)
        self.assertEqual([row["id"] for row in store.list_stores()["data"]], [replacement["id"]])

    def test_replay_handles_deletions_and_capacity_evictions(self) -> None:
        # Replay calls _drop without owning the condition. Notifications
        # belong in the live deletion path, not that shared cleanup helper.
        with TemporaryDirectory() as directory:
            path = Path(directory)
            store = VectorStoreStore(2, state_dir=path)
            deleted = store.create(name="deleted")
            store.create(name="evicted")
            store.delete(deleted["id"])
            first = store.create(name="first")
            second = store.create(name="second")

            restored = VectorStoreStore(2, state_dir=path)
            self.assertEqual(
                {row["id"] for row in restored.list_stores()["data"]},
                {first["id"], second["id"]},
            )
            self.assertEqual(restored.recover_warnings, [])
