"""Regression tests for file-store write-ahead and recovery invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from fx1.serve.api import _FileStore
from fx1.serve.journal import JobJournal


def _ids(store: _FileStore) -> list[str]:
    return [rec.file_id for rec in store.list()]


def _fail_append(payload: dict[str, Any]) -> None:
    raise OSError("synthetic journal failure")


def test_failed_put_publishes_neither_record_eviction_nor_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _FileStore(2, 1024, state_dir=tmp_path)
    first = store.put(filename="a.jsonl", purpose="batch", content=b"a")
    second = store.put(filename="b.jsonl", purpose="batch", content=b"b")
    before_blobs = {p.name for p in (tmp_path / "files").glob("*.bin")}
    journal = cast("Any", store._journal)
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.put(filename="c.jsonl", purpose="batch", content=b"c")

    assert _ids(store) == [second.file_id, first.file_id]
    assert {p.name for p in (tmp_path / "files").glob("*.bin")} == before_blobs


def test_failed_delete_keeps_live_record_and_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _FileStore(2, 1024, state_dir=tmp_path)
    rec = store.put(filename="a.jsonl", purpose="batch", content=b"a")
    journal = cast("Any", store._journal)
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.delete(rec.file_id)

    assert _ids(store) == [rec.file_id]
    assert (tmp_path / "files" / f"{rec.file_id}.bin").read_bytes() == b"a"


def test_failed_touch_does_not_change_lru_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _FileStore(2, 1024, state_dir=tmp_path)
    first = store.put(filename="a.jsonl", purpose="batch", content=b"a")
    second = store.put(filename="b.jsonl", purpose="batch", content=b"b")
    journal = cast("Any", store._journal)
    monkeypatch.setattr(journal, "append", _fail_append)

    with pytest.raises(OSError, match="synthetic journal failure"):
        store.get(first.file_id)

    assert _ids(store) == [second.file_id, first.file_id]


def test_successful_touch_survives_restart_and_controls_next_eviction(tmp_path: Path) -> None:
    store = _FileStore(2, 1024, state_dir=tmp_path)
    first = store.put(filename="a.jsonl", purpose="batch", content=b"a")
    second = store.put(filename="b.jsonl", purpose="batch", content=b"b")
    assert store.get(first.file_id) is not None

    recovered = _FileStore(2, 1024, state_dir=tmp_path)
    assert _ids(recovered) == [first.file_id, second.file_id]
    third = recovered.put(filename="c.jsonl", purpose="batch", content=b"c")
    assert _ids(recovered) == [third.file_id, first.file_id]
    assert not (tmp_path / "files" / f"{second.file_id}.bin").exists()


def test_broken_chain_refuses_startup_without_rewriting_evidence(tmp_path: Path) -> None:
    store = _FileStore(2, 1024, state_dir=tmp_path)
    store.put(filename="a.jsonl", purpose="batch", content=b"a")
    path = tmp_path / "files.jsonl"
    with path.open("ab") as fh:
        fh.write(b'{"torn":')
    damaged = path.read_bytes()

    with pytest.raises(RuntimeError, match="file journal is damaged"):
        _FileStore(2, 1024, state_dir=tmp_path)

    assert path.read_bytes() == damaged


def test_valid_hash_unknown_operation_refuses_without_compaction(tmp_path: Path) -> None:
    journal = JobJournal(tmp_path / "files.jsonl")
    journal.append({"unexpected": "operation"})
    original = journal.path.read_bytes()

    with pytest.raises(RuntimeError, match="invalid operation"):
        _FileStore(2, 1024, state_dir=tmp_path)

    assert journal.path.read_bytes() == original


def test_missing_or_size_mismatched_blob_refuses_without_compaction(tmp_path: Path) -> None:
    journal = JobJournal(tmp_path / "files.jsonl")
    file_id = "file-00000000000000000000000000000001"
    payload = {
        "file": {
            "file_id": file_id,
            "filename": "a.jsonl",
            "purpose": "batch",
            "size": 2,
            "created_at": 1,
        }
    }
    journal.append(payload)
    original = journal.path.read_bytes()

    with pytest.raises(RuntimeError, match="invalid operation"):
        _FileStore(2, 1024, state_dir=tmp_path)
    assert journal.path.read_bytes() == original

    blob_dir = tmp_path / "files"
    blob_dir.mkdir(exist_ok=True)
    (blob_dir / f"{file_id}.bin").write_bytes(b"a")
    with pytest.raises(RuntimeError, match="invalid operation"):
        _FileStore(2, 1024, state_dir=tmp_path)
    assert journal.path.read_bytes() == original


def test_same_size_blob_corruption_fails_closed_without_rewriting(tmp_path: Path) -> None:
    store = _FileStore(2, 1024, state_dir=tmp_path)
    rec = store.put(filename="a.jsonl", purpose="batch", content=b"abcd")
    blob = tmp_path / "files" / f"{rec.file_id}.bin"
    blob.write_bytes(b"wxyz")
    original = (tmp_path / "files.jsonl").read_bytes()

    with pytest.raises(RuntimeError, match="invalid operation"):
        _FileStore(2, 1024, state_dir=tmp_path)

    assert (tmp_path / "files.jsonl").read_bytes() == original
    assert blob.read_bytes() == b"wxyz"


def test_legacy_record_is_accepted_and_upgraded_with_digest(tmp_path: Path) -> None:
    file_id = "file-00000000000000000000000000000002"
    blob_dir = tmp_path / "files"
    blob_dir.mkdir(parents=True)
    (blob_dir / f"{file_id}.bin").write_bytes(b"old")
    journal = JobJournal(tmp_path / "files.jsonl")
    journal.append(
        {
            "file": {
                "file_id": file_id,
                "filename": "legacy.jsonl",
                "purpose": "batch",
                "size": 3,
                "created_at": 1,
            }
        }
    )

    restored = _FileStore(2, 1024, state_dir=tmp_path)

    assert restored.get(file_id).content == b"old"  # type: ignore[union-attr]
    line = json.loads((tmp_path / "files.jsonl").read_text().splitlines()[0])
    assert len(line["payload"]["file"]["content_sha256"]) == 64


def test_store_enforces_byte_cap_before_writing(tmp_path: Path) -> None:
    store = _FileStore(2, 3, state_dir=tmp_path)

    with pytest.raises(ValueError, match="3-byte cap"):
        store.put(filename="large.jsonl", purpose="batch", content=b"four")

    assert store.list() == []
    assert not (tmp_path / "files").exists()


def test_path_traversing_journal_id_fails_closed(tmp_path: Path) -> None:
    files = tmp_path / "files"
    files.mkdir(parents=True)
    outside = tmp_path.parent / "escape.bin"
    outside.write_bytes(b"outside")
    journal = JobJournal(tmp_path / "files.jsonl")
    journal.append(
        {
            "file": {
                "file_id": "../../escape",
                "filename": "escape.jsonl",
                "purpose": "batch",
                "size": len(outside.read_bytes()),
                "created_at": 1,
            }
        }
    )
    original = journal.path.read_bytes()

    with pytest.raises(RuntimeError, match="invalid operation"):
        _FileStore(2, 1024, state_dir=tmp_path)

    assert outside.read_bytes() == b"outside"
    assert journal.path.read_bytes() == original


def test_clean_restart_removes_interrupted_temp_blob(tmp_path: Path) -> None:
    files = tmp_path / "files"
    files.mkdir(parents=True)
    temporary = files / ".file-00000000000000000000000000000000.tmp"
    temporary.write_bytes(b"partial")

    _FileStore(2, 1024, state_dir=tmp_path)

    assert not temporary.exists()
