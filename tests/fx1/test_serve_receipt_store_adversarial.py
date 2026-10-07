"""SYNTHETIC adversarial probes for ``fx1.serve.receipt_store``.

Index integrity edges beyond the happy path: sha format strictness,
non-receipt files never indexed, subdirectory isolation, fail-closed
behavior when the directory's state races the scan, and deterministic
items ordering. All fixtures are synthetic documents under tmp_path.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fx1.serve.receipt_store import ReceiptIndex


def _receipt(root: Path, name: str, digest: str | int | list[int], pad: int = 0) -> Path:
    path = root / name
    path.write_text(json.dumps({"receipt_sha256": digest, "pad": pad}), encoding="utf-8")
    return path


def test_lookup_is_exact_lowercase_sha256_only(tmp_path: Path) -> None:
    """Lookup never case-folds or tolerates malformed digests."""
    _receipt(tmp_path, "a.json", "a" * 64)
    index = ReceiptIndex(tmp_path)
    assert index.lookup("a" * 64) == tmp_path / "a.json"
    for bad in ("A" * 64, "a" * 63, "a" * 65, "g" * 64, "", "../a.json"):
        assert index.lookup(bad) is None


def test_non_string_or_malformed_sha_fields_are_not_indexed(tmp_path: Path) -> None:
    """Only a real 64-hex lowercase string declares an indexable receipt."""
    _receipt(tmp_path, "int.json", 5)
    _receipt(tmp_path, "list.json", [1])
    _receipt(tmp_path, "upper.json", "A" * 64)
    _receipt(tmp_path, "short.json", "a" * 32)
    good = _receipt(tmp_path, "good.json", "b" * 64)
    index = ReceiptIndex(tmp_path)
    assert index.available() is True
    assert index.items() == [("b" * 64, good)]


def test_only_top_level_json_files_are_candidates(tmp_path: Path) -> None:
    """Subdirectories and near-miss extensions are never descended or read."""
    (tmp_path / "deep").mkdir()
    _receipt(tmp_path / "deep", "nested.json", "a" * 64)
    (tmp_path / "backup.json.bak").write_text(json.dumps({"receipt_sha256": "c" * 64}))
    (tmp_path / "notes.jsonl").write_text(json.dumps({"receipt_sha256": "d" * 64}))
    good = _receipt(tmp_path, "good.json", "b" * 64)
    index = ReceiptIndex(tmp_path)
    assert index.items() == [("b" * 64, good)]
    assert index.lookup("a" * 64) is None
    assert index.lookup("c" * 64) is None
    assert index.lookup("d" * 64) is None


def test_lookup_paths_never_escape_the_root(tmp_path: Path) -> None:
    """Indexed paths are always direct children of the store root."""
    _receipt(tmp_path, "a.json", "a" * 64)
    index = ReceiptIndex(tmp_path)
    for _sha, path in index.items():
        assert path.parent == tmp_path
        assert path.resolve().is_relative_to(tmp_path.resolve())


def test_items_sorted_by_filename_deterministically(tmp_path: Path) -> None:
    for name in ("z.json", "a.json", "m.json"):
        _receipt(tmp_path, name, hashlib.sha256(name.encode()).hexdigest())
    index = ReceiptIndex(tmp_path)
    names = [path.name for _sha, path in index.items()]
    assert names == ["a.json", "m.json", "z.json"]
    assert index.items() == index.items()


class _VanishingEntry:
    """Scandir entry whose stat fails — simulates delete-during-scan."""

    def __init__(self, name: str) -> None:
        self.name = name

    def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        raise FileNotFoundError("synthetic vanish during scan")


class _VanishingScan:
    def __enter__(self) -> Iterator[_VanishingEntry]:
        return iter([_VanishingEntry("gone.json")])

    def __exit__(self, *_args: Any) -> bool:
        return False


def test_entry_vanishing_mid_scan_fails_closed_then_recovers(tmp_path: Path) -> None:
    """A directory listing that races a delete marks the whole index
    unavailable rather than serving a half-built map; the next scan heals."""
    _receipt(tmp_path, "a.json", "a" * 64)
    index = ReceiptIndex(tmp_path)
    assert index.available() is True
    from unittest.mock import patch

    with patch("fx1.serve.receipt_store.os.scandir", return_value=_VanishingScan()):
        assert index.available() is False
        assert index.items() == []
        assert index.lookup("a" * 64) is None
    assert index.available() is True
    assert index.lookup("a" * 64) == tmp_path / "a.json"


def test_unparseable_receipt_drops_out_not_the_index(tmp_path: Path) -> None:
    """One file that fails to decode as a receipt dict is excluded; the rest
    of the store stays indexed — a bad blob never takes the index down."""
    good = _receipt(tmp_path, "good.json", "a" * 64)
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "list.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    index = ReceiptIndex(tmp_path)
    assert index.available() is True
    assert index.items() == [("a" * 64, good)]


def test_content_replacement_is_never_served_stale(tmp_path: Path) -> None:
    """Same-name content swap (size bump forces a stamp change on every
    filesystem) yields the new sha on next read — a sealed file's identity
    follows its content, not its name."""
    path = _receipt(tmp_path, "a.json", "a" * 64, pad=0)
    index = ReceiptIndex(tmp_path)
    assert index.lookup("a" * 64) == path
    _receipt(tmp_path, "a.json", "b" * 64, pad=123456789)
    assert index.lookup("a" * 64) is None
    assert index.lookup("b" * 64) == path


def test_receipt_claiming_another_files_digest_indexes_by_claim(tmp_path: Path) -> None:
    """The index is a metadata map, not a verifier: a file whose declared
    receipt_sha256 duplicates another's resolves to the lexicographically
    first filename — verification happens downstream, at receipt verify."""
    _receipt(tmp_path, "a.json", "a" * 64)
    _receipt(tmp_path, "z.json", "a" * 64)  # imposter declares same sha
    index = ReceiptIndex(tmp_path)
    assert index.lookup("a" * 64) == tmp_path / "a.json"
    assert index.items() == [("a" * 64, tmp_path / "a.json")]


def test_rename_onto_another_name_invalidates_the_old_claim(tmp_path: Path) -> None:
    """``os.replace`` swaps the blob under a name: the (dev,ino) stamp changes
    even when size and mtime collide — the moved-out claim is never served."""
    _receipt(tmp_path, "fresh.json", "a" * 64, pad=1)
    _receipt(tmp_path, "stale.json", "b" * 64, pad=1)
    index = ReceiptIndex(tmp_path)
    assert index.lookup("a" * 64) == tmp_path / "fresh.json"
    os.replace(tmp_path / "fresh.json", tmp_path / "stale.json")
    assert index.lookup("a" * 64) == tmp_path / "stale.json"
    assert index.lookup("b" * 64) is None
