"""SYNTHETIC store-only regressions for capacity after LRU promotion."""

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

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
    fail_touch = False

    def append(self, payload: dict[str, Any]) -> None:
        if self.fail_update and "vs_update" in payload:
            raise OSError("synthetic update append failure")
        if self.fail_touch and ("vs_touch" in payload or "vs_touches" in payload):
            raise OSError("synthetic touch append failure")
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

    def _check_promotion(
        self,
        *,
        promotion: Literal["get", "update", "search"],
        membership: Literal["create", "batch", "attach"],
        delete_waiter: bool = False,
    ) -> None:
        reader = _BlockingReader()
        with TemporaryDirectory() as directory:
            path = Path(directory)
            store = VectorStoreStore(2, state_dir=path, file_reader=reader)
            with ThreadPoolExecutor(max_workers=3) as pool:
                try:
                    policy = {"anchor": "last_active_at", "days": 1}
                    if membership != "create":
                        pinned_id = store.create(name="pinned", expires_after=policy)["id"]
                        if membership == "batch":
                            pinned = pool.submit(store.file_batch_create, pinned_id, ["file-a"])
                        else:
                            pinned = pool.submit(store.attach, pinned_id, "file-a")
                    else:
                        pinned = pool.submit(
                            store.create,
                            name="pinned",
                            file_ids=["file-a"],
                            expires_after=policy,
                        )
                    self.assertTrue(reader.started.wait(timeout=3))
                    pinned_id = store.list_stores()["data"][0]["id"]
                    if delete_waiter:
                        deleter = pool.submit(store.delete, pinned_id)
                        self._wait_for_waiters(store, 1)
                    idle_id = store.create(name="idle")["id"]
                    creator = pool.submit(store.create, name="replacement")
                    self._wait_for_waiters(store, 2 if delete_waiter else 1)

                    if promotion == "update":
                        store.update(pinned_id, name="updated")
                    elif promotion == "search":
                        before_search = int(time.time())
                        self.assertEqual(store.search([pinned_id], "alpha"), [])
                        meta = store._stores[pinned_id]
                        self.assertGreaterEqual(meta.last_active_at, before_search)
                        self.assertEqual(meta.expires_at, meta.last_active_at + 86400)
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
                self.assertEqual(
                    restored.get(pinned_id)["name"],
                    "updated" if promotion == "update" else "pinned",
                )
                live_meta = store._stores[pinned_id]
                replay_meta = restored._stores[pinned_id]
                self.assertEqual(replay_meta.last_active_at, live_meta.last_active_at)
                self.assertEqual(replay_meta.expires_at, live_meta.expires_at)
                self.assertEqual(replay_meta.expires_at, replay_meta.last_active_at + 86400)
            self.assertEqual(restored.recover_warnings, [])

    def test_get_wakes_creator_during_create_with_files(self) -> None:
        self._check_promotion(promotion="get", membership="create")

    def test_get_wakes_creator_during_file_batch(self) -> None:
        self._check_promotion(promotion="get", membership="batch")

    def test_update_wakes_creator_during_create_with_files(self) -> None:
        self._check_promotion(promotion="update", membership="create")

    def test_update_wakes_creator_during_file_batch(self) -> None:
        self._check_promotion(promotion="update", membership="batch")

    def test_get_wakes_creator_behind_delete_waiter(self) -> None:
        self._check_promotion(promotion="get", membership="create", delete_waiter=True)

    def test_update_wakes_creator_behind_delete_waiter(self) -> None:
        self._check_promotion(promotion="update", membership="create", delete_waiter=True)

    def test_search_wakes_creator_during_create_with_files(self) -> None:
        self._check_promotion(promotion="search", membership="create")

    def test_search_wakes_creator_during_file_batch(self) -> None:
        self._check_promotion(promotion="search", membership="batch")

    def test_search_wakes_creator_during_attach(self) -> None:
        self._check_promotion(promotion="search", membership="attach")

    def test_search_wakes_creator_behind_delete_waiter(self) -> None:
        self._check_promotion(promotion="search", membership="batch", delete_waiter=True)

    def _check_failed_touch(self, operation: Literal["search", "attach", "batch"]) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vector_stores.jsonl"
            journal = _FailingUpdateJournal(path)

            def reader(_fid: str) -> tuple[bytes, str]:
                return b"alpha beta gamma", "test.txt"

            store = VectorStoreStore(2, journal=journal, file_reader=reader)
            first = store.create(
                name="first", expires_after={"anchor": "last_active_at", "days": 1}
            )["id"]
            second = store.create(name="second")["id"]
            before = store._stores[first].model_dump()
            original_bytes = path.read_bytes()
            journal.fail_touch = True

            with self.assertRaisesRegex(OSError, "synthetic touch append failure"):
                if operation == "search":
                    store.search([first], "alpha")
                elif operation == "attach":
                    store.attach(first, "file-a")
                else:
                    store.file_batch_create(first, ["file-a"])

            self.assertEqual(list(store._stores), [first, second])
            self.assertEqual(store._stores[first].model_dump(), before)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(store._inflight, {})
            self.assertEqual(store._membership_busy, set())
            restored = VectorStoreStore(2, journal=JobJournal(path), file_reader=reader)
            self.assertEqual(list(restored._stores), [first, second])
            self.assertEqual(restored._stores[first].model_dump(), before)

    def test_failed_search_touch_preserves_state_and_replay(self) -> None:
        self._check_failed_touch("search")

    def test_failed_attach_touch_preserves_state_and_replay(self) -> None:
        self._check_failed_touch("attach")

    def test_failed_batch_touch_preserves_state_and_replay(self) -> None:
        self._check_failed_touch("batch")

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
