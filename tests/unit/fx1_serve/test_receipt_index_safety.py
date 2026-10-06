"""SYNTHETIC receipt metadata, cache-invalidation and resource regressions."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from fx1.serve.receipt_store import ReceiptIndex


def _receipt(root: Path, name: str, digest: str) -> Path:
    path = root / name
    path.write_text(json.dumps({"receipt_sha256": digest}), encoding="utf-8")
    return path


def test_missing_empty_and_removed_store(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    index = ReceiptIndex(root)
    assert index.root == root
    assert index.available() is False
    assert index.items() == []
    root.mkdir()
    assert index.available() is True
    path = _receipt(root, "a.json", "a" * 64)
    assert index.lookup("a" * 64) == path
    path.unlink()
    root.rmdir()
    assert index.available() is False
    assert index.lookup("a" * 64) is None
    root.mkdir()
    _receipt(root, "b.json", "b" * 64)
    assert index.lookup("b" * 64) == root / "b.json"


def test_rename_with_unchanged_count_and_max_mtime_invalidates(tmp_path: Path) -> None:
    path = _receipt(tmp_path, "old.json", "a" * 64)
    index = ReceiptIndex(tmp_path)
    assert index.lookup("a" * 64) == path
    renamed = path.rename(tmp_path / "new.json")
    assert index.lookup("a" * 64) == renamed


def test_replacing_an_old_receipt_is_not_hidden_by_newer_mtime(tmp_path: Path) -> None:
    old = _receipt(tmp_path, "old.json", "a" * 64)
    newest = _receipt(tmp_path, "newest.json", "b" * 64)
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newest, ns=(9_000_000_000, 9_000_000_000))
    index = ReceiptIndex(tmp_path)
    assert index.lookup("a" * 64) == old
    replacement = _receipt(tmp_path, "replacement.tmp", "c" * 64)
    os.utime(replacement, ns=(1_000_000_000, 1_000_000_000))
    os.replace(replacement, old)
    assert index.lookup("a" * 64) is None
    assert index.lookup("c" * 64) == old


def test_directory_replacement_with_identical_count_and_mtime_invalidates(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    root.mkdir()
    path = _receipt(root, "a.json", "a" * 64)
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    index = ReceiptIndex(root)
    assert index.lookup("a" * 64) == path
    root.rename(tmp_path / "old")
    root.mkdir()
    replacement = _receipt(root, "b.json", "b" * 64)
    os.utime(replacement, ns=(1_000_000_000, 1_000_000_000))
    assert index.lookup("a" * 64) is None
    assert index.lookup("b" * 64) == replacement


def test_appending_parses_only_the_new_receipt(tmp_path: Path) -> None:
    for i in range(12):
        _receipt(tmp_path, f"{i:02}.json", f"{i:064x}")
    index = ReceiptIndex(tmp_path)
    assert len(index.items()) == 12
    original = Path.read_bytes
    reads: list[Path] = []

    def tracked(path: Path) -> bytes:
        reads.append(path)
        return original(path)

    new = _receipt(tmp_path, "new.json", "f" * 64)
    with patch.object(Path, "read_bytes", tracked):
        assert index.lookup("f" * 64) == new
        assert len(index.items()) == 13
    assert reads == [new]


def test_unchanged_store_reuses_decoded_receipts(tmp_path: Path) -> None:
    path = _receipt(tmp_path, "a.json", "a" * 64)
    index = ReceiptIndex(tmp_path)
    assert index.lookup("a" * 64) == path
    with patch.object(Path, "read_bytes", side_effect=AssertionError("unexpected reread")):
        assert index.lookup("a" * 64) == path
        assert index.items() == [("a" * 64, path)]


@pytest.mark.parametrize("raw", [b"null", b"[]", b"{", b"\xff", b'{"receipt_sha256":"bad"}'])
def test_malformed_documents_do_not_hide_valid_receipts(tmp_path: Path, raw: bytes) -> None:
    valid = _receipt(tmp_path, "valid.json", "a" * 64)
    (tmp_path / "invalid.json").write_bytes(raw)
    index = ReceiptIndex(tmp_path)
    assert index.available() is True
    assert index.items() == [("a" * 64, valid)]


def test_deeply_nested_invalid_json_does_not_crash_the_index(tmp_path: Path) -> None:
    valid = _receipt(tmp_path, "valid.json", "a" * 64)
    (tmp_path / "nested.json").write_bytes(b"[" * 10000 + b"0" + b"]" * 10000)
    assert ReceiptIndex(tmp_path).items() == [("a" * 64, valid)]


def test_symlinked_json_is_not_read_or_indexed(tmp_path: Path) -> None:
    root = tmp_path / "receipts"
    root.mkdir()
    outside = _receipt(tmp_path, "outside.json", "a" * 64)
    try:
        (root / "linked.json").symlink_to(outside)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with patch.object(Path, "read_bytes", side_effect=AssertionError("symlink followed")):
        assert ReceiptIndex(root).items() == []


def test_directory_named_json_is_not_opened(tmp_path: Path) -> None:
    (tmp_path / "directory.json").mkdir()
    with patch.object(Path, "read_bytes", side_effect=AssertionError("directory opened")):
        assert ReceiptIndex(tmp_path).items() == []


def test_duplicate_digest_has_a_deterministic_filename_winner(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a.json", "a" * 64)
    _receipt(tmp_path, "b.json", "a" * 64)
    assert ReceiptIndex(tmp_path).lookup("a" * 64) == first


def test_returned_items_do_not_alias_cached_state(tmp_path: Path) -> None:
    first = _receipt(tmp_path, "a.json", "a" * 64)
    index = ReceiptIndex(tmp_path)
    items = index.items()
    items.clear()
    assert index.items() == [("a" * 64, first)]


def test_concurrent_readers_observe_the_complete_new_index(tmp_path: Path) -> None:
    for i in range(20):
        _receipt(tmp_path, f"{i:02}.json", f"{i:064x}")
    index = ReceiptIndex(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(pool.map(lambda _: index.items(), range(32)))
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    assert len(snapshots[0]) == 20


def test_unexpected_decoder_error_is_not_hidden(tmp_path: Path) -> None:
    _receipt(tmp_path, "a.json", "a" * 64)
    with (
        patch("fx1.serve.receipt_store.json.loads", side_effect=RuntimeError("SYNTHETIC bug")),
        pytest.raises(RuntimeError, match="SYNTHETIC bug"),
    ):
        ReceiptIndex(tmp_path).items()
