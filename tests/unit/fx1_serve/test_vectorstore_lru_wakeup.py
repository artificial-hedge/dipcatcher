"""SYNTHETIC store-only regressions for capacity after LRU promotion."""

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from fx1.serve.journal import JobJournal
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


class _FailingUpdateJournal(JobJournal):
    fail_update = False

    def append(self, payload: dict[str, Any]) -> None:
        if self.fail_update and "vs_update" in payload:
            raise OSError("synthetic update append failure")
        super().append(payload)


class TestVectorStoreLRUWakeup(unittest.TestCase):
    def _wait_for_waiters(self, store: VectorStoreStore, count: int) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with store._condition:
                if len(vars(store._condition)["_waiters"]) >= count:
                    return
            time.sleep(0.005)
        self.fail(f"expected {count} condition waiter(s)")

    def _check_promotion(self, *, update: bool, batch: bool, delete_waiter: bool = False) -> None:
        reader = _BlockingReader()
        with TemporaryDirectory() as directory:
            path = Path(directory)
            store = VectorStoreStore(2, state_dir=path, file_reader=reader)
            with ThreadPoolExecutor(max_workers=3) as pool:
                try:
                    if batch:
                        pinned_id = store.create(name="pinned")["id"]
                        pinned = pool.submit(store.file_batch_create, pinned_id, ["file-a"])
                    else:
                        pinned = pool.submit(store.create, name="pinned", file_ids=["file-a"])
                    self.assertTrue(reader.started.wait(timeout=3))
                    pinned_id = store.list_stores()["data"][0]["id"]
                    if delete_waiter:
                        deleter = pool.submit(store.delete, pinned_id)
                        self._wait_for_waiters(store, 1)
                    idle_id = store.create(name="idle")["id"]
                    creator = pool.submit(store.create, name="replacement")
                    self._wait_for_waiters(store, 2 if delete_waiter else 1)

                    if update:
                        store.update(pinned_id, name="updated")
                    else:
                        store.get(pinned_id)
                    done, _ = wait([creator], timeout=2)
                    self.assertIn(creator, done, "LRU promotion stranded the creator")
                    self.assertFalse(reader.release.is_set())
                    self.assertFalse(pinned.done())
                    replacement_id = creator.result()["id"]
                    expected = {pinned_id, replacement_id}
                    self.assertNotIn(idle_id, store._stores)
                    self.assertEqual(set(store._stores), expected)
                    self.assertEqual(store._inflight[pinned_id], 1)
                    if delete_waiter:
                        self.assertFalse(deleter.done())
                finally:
                    # Also releases the baseline's stranded creator before joining.
                    reader.release.set()
                pinned.result(timeout=3)
                creator.result(timeout=3)
                if delete_waiter:
                    self.assertTrue(deleter.result(timeout=3)["deleted"])
                    expected.remove(pinned_id)

            self.assertEqual(store._inflight, {})
            restored = VectorStoreStore(2, state_dir=path, file_reader=reader)
            self.assertEqual(set(restored._stores), expected)
            if not delete_waiter:
                self.assertEqual(restored.get(pinned_id)["file_counts"]["completed"], 1)
                self.assertEqual(restored.get(pinned_id)["name"], "updated" if update else "pinned")
            self.assertEqual(restored.recover_warnings, [])

    def test_get_wakes_creator_during_create_with_files(self) -> None:
        self._check_promotion(update=False, batch=False)

    def test_get_wakes_creator_during_file_batch(self) -> None:
        self._check_promotion(update=False, batch=True)

    def test_update_wakes_creator_during_create_with_files(self) -> None:
        self._check_promotion(update=True, batch=False)

    def test_update_wakes_creator_during_file_batch(self) -> None:
        self._check_promotion(update=True, batch=True)

    def test_get_wakes_creator_behind_delete_waiter(self) -> None:
        self._check_promotion(update=False, batch=False, delete_waiter=True)

    def test_update_wakes_creator_behind_delete_waiter(self) -> None:
        self._check_promotion(update=True, batch=False, delete_waiter=True)

    def test_failed_update_preserves_order_and_replay(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vector_stores.jsonl"
            journal = _FailingUpdateJournal(path)
            store = VectorStoreStore(2, journal=journal)
            first = store.create(name="first")["id"]
            second = store.create(name="second")["id"]
            original_bytes = path.read_bytes()
            journal.fail_update = True

            with self.assertRaisesRegex(OSError, "synthetic update append failure"):
                store.update(first, name="failed")

            self.assertEqual(list(store._stores), [first, second])
            self.assertEqual(store._stores[first].name, "first")
            self.assertEqual(path.read_bytes(), original_bytes)
            restored = VectorStoreStore(2, journal=JobJournal(path))
            self.assertEqual(list(restored._stores), [first, second])
            self.assertEqual(restored.get(first)["name"], "first")

    def test_mru_access_preserves_order(self) -> None:
        store = VectorStoreStore(2)
        first = store.create(name="first")["id"]
        second = store.create(name="second")["id"]
        store.get(second)
        self.assertEqual(list(store._stores), [first, second])
        store.update(second, name="updated")
        self.assertEqual(list(store._stores), [first, second])


if __name__ == "__main__":
    unittest.main()
