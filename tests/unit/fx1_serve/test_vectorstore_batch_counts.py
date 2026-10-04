"""Store-only regressions for terminal vector-store batch accounting."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fx1.serve.vectorstores import VectorStoreError, VectorStoreStore


def _store(docs: dict[str, bytes], state_dir: Path | None = None) -> VectorStoreStore:
    def read(file_id: str) -> tuple[bytes, str] | None:
        if file_id not in docs:
            return None
        return docs[file_id], f"{file_id}.txt"

    return VectorStoreStore(8, state_dir=state_dir, file_reader=read)


def _counts(completed: int, failed: int) -> dict[str, int]:
    return {
        "in_progress": 0,
        "completed": completed,
        "failed": failed,
        "cancelled": 0,
        "total": completed + failed,
    }


class TestVectorStoreBatchCounts(unittest.TestCase):
    def test_empty_and_whitespace_files_count_as_failed(self) -> None:
        for content in (b"", b" \t\r\n"):
            with self.subTest(content=content):
                store = _store({"file-empty": content})
                vs_id = store.create()["id"]
                batch = store.file_batch_create(vs_id, ["file-empty"])

                self.assertEqual(batch["file_counts"], _counts(0, 1))
                self.assertEqual(batch["status"], "failed")
                self.assertEqual(store.get(vs_id)["file_counts"], _counts(0, 1))
                rows = store.file_batch_files(vs_id, batch["id"], filter="failed")["data"]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0], store.get_file(vs_id, "file-empty"))
                self.assertEqual(rows[0]["last_error"]["code"], "empty_file")
                self.assertEqual(rows[0]["indexed_chunks"], 0)
                self.assertEqual(
                    store.file_batch_files(vs_id, batch["id"], filter="completed")["data"],
                    [],
                )

    def test_mixed_returned_and_raised_failures_count_separately(self) -> None:
        store = _store({"file-good": b"alpha beta", "file-empty": b"", "file-space": b" \n"})
        vs_id = store.create()["id"]
        file_ids = ["file-good", "file-empty", "file-missing", "file-space"]
        batch = store.file_batch_create(vs_id, file_ids)

        self.assertEqual(batch["file_counts"], _counts(1, 3))
        self.assertEqual(batch["status"], "completed")
        rows = store.file_batch_files(vs_id, batch["id"])["data"]
        self.assertEqual([row["id"] for row in rows], file_ids)
        failed = store.file_batch_files(vs_id, batch["id"], filter="failed")["data"]
        self.assertEqual([row["id"] for row in failed], file_ids[1:])
        self.assertEqual(
            [row["last_error"]["code"] for row in failed],
            ["empty_file", "file_not_found", "empty_file"],
        )
        completed = store.file_batch_files(vs_id, batch["id"], filter="completed")["data"]
        self.assertEqual([row["id"] for row in completed], ["file-good"])
        self.assertEqual(store.get(vs_id)["file_counts"], _counts(1, 2))

    def test_all_returned_and_raised_failures_produce_failed_batch(self) -> None:
        store = _store({"file-empty": b"", "file-space": b" \n"})
        vs_id = store.create()["id"]
        batch = store.file_batch_create(vs_id, ["file-empty", "file-missing", "file-space"])

        self.assertEqual(batch["file_counts"], _counts(0, 3))
        self.assertEqual(batch["status"], "failed")
        with self.assertRaises(VectorStoreError) as raised:
            store.file_batch_cancel(vs_id, batch["id"])
        self.assertEqual(raised.exception.code, "file_batch_terminal")
        self.assertIn("already failed", str(raised.exception))

    def test_duplicate_failed_attachment_counts_each_request(self) -> None:
        store = _store({"file-empty": b""})
        vs_id = store.create()["id"]
        batch = store.file_batch_create(vs_id, ["file-empty", "file-empty"])

        self.assertEqual(batch["file_counts"], _counts(0, 2))
        self.assertEqual(batch["status"], "failed")
        rows = store.file_batch_files(vs_id, batch["id"])["data"]
        self.assertEqual(
            [row["last_error"]["code"] for row in rows],
            ["empty_file", "file_already_attached"],
        )

    def test_successful_batches_remain_completed(self) -> None:
        store = _store({"file-a": b"alpha", "file-b": b"beta"})
        vs_id = store.create()["id"]
        batch = store.file_batch_create(vs_id, ["file-a", "file-b"])

        self.assertEqual(batch["file_counts"], _counts(2, 0))
        self.assertEqual(batch["status"], "completed")

    def test_detaching_members_preserves_batch_verdict(self) -> None:
        store = _store({"file-good": b"alpha", "file-empty": b""})
        vs_id = store.create()["id"]
        batch = store.file_batch_create(vs_id, ["file-good", "file-empty"])
        rows = store.file_batch_files(vs_id, batch["id"])

        self.assertEqual(batch["file_counts"], _counts(1, 1))
        store.detach(vs_id, "file-good")
        store.detach(vs_id, "file-empty")
        self.assertEqual(store.file_batch_get(vs_id, batch["id"]), batch)
        self.assertEqual(store.file_batch_files(vs_id, batch["id"]), rows)
        self.assertEqual(store.get(vs_id)["file_counts"], _counts(0, 0))

    def test_corrected_verdicts_survive_journal_replay(self) -> None:
        for docs, expected in (
            ({"file-empty": b""}, _counts(0, 2)),
            ({"file-empty": b" \n", "file-good": b"alpha"}, _counts(1, 2)),
        ):
            with self.subTest(docs=docs), TemporaryDirectory() as directory:
                path = Path(directory)
                store = _store(docs, path)
                vs_id = store.create()["id"]
                batch = store.file_batch_create(vs_id, [*docs, "file-missing"])
                rows = store.file_batch_files(vs_id, batch["id"])

                restored = _store(docs, path)
                replayed = restored.file_batch_get(vs_id, batch["id"])
                self.assertEqual(replayed["file_counts"], expected)
                self.assertEqual(
                    replayed["status"], "completed" if expected["completed"] else "failed"
                )
                self.assertEqual(replayed, batch)
                self.assertEqual(restored.file_batch_files(vs_id, batch["id"]), rows)
                self.assertEqual(restored.recover_warnings, [])
