"""Corruption and crash recovery for fx1 receipt storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fx1.data.receipts import load_receipts
from fx1.serve.receipt_store import ReceiptIndex
from fx1.train.receipts import issue_receipt


def _eligible_receipt(path: Path) -> bytes:
    raw = json.dumps({"schema": "test/v1", "research_only": True, "live_pnl_claim": False}).encode(
        "utf-8"
    )
    path.write_bytes(raw)
    return raw


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff",
        b"[" * 10000 + b"0" + b"]" * 10000,
        b'{"value":' + b"9" * 5000 + b"}",
    ],
)
def test_corrupt_receipt_does_not_hide_valid_sibling(tmp_path: Path, raw: bytes) -> None:
    valid = tmp_path / "valid.json"
    _eligible_receipt(valid)
    (tmp_path / "invalid.json").write_bytes(raw)
    records = load_receipts(tmp_path)
    assert [Path(record.path) for record in records] == [valid]


def test_receipt_digest_binds_the_same_bytes_that_were_parsed(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    raw = _eligible_receipt(path)
    original = Path.read_bytes
    reads: list[Path] = []

    def tracked(self: Path) -> bytes:
        if self == path:
            reads.append(self)
        return original(self)

    with patch.object(Path, "read_bytes", tracked):
        records = load_receipts(tmp_path)
    assert len(records) == 1
    assert records[0].sha256 == hashlib.sha256(raw).hexdigest()
    assert reads == [path]


def test_training_receipt_cannot_replace_existing_evidence(tmp_path: Path) -> None:
    inputs = {name: tmp_path / f"{name}.json" for name in ("config", "corpus", "split", "eval")}
    for path in inputs.values():
        path.write_text("{}", encoding="utf-8")
    out = tmp_path / "training.json"

    def issue(run_name: str) -> None:
        issue_receipt(
            run_name=run_name,
            repo_root=tmp_path,
            config_path=inputs["config"],
            corpus_path=inputs["corpus"],
            split_manifest_path=inputs["split"],
            eval_base_path=inputs["eval"],
            seed=7,
            out_path=out,
        )

    issue("first")
    before = out.read_bytes()
    with pytest.raises(FileExistsError, match="different content"):
        issue("second")
    assert out.read_bytes() == before


def test_transient_receipt_read_failure_is_retried(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"receipt_sha256": "a" * 64}), encoding="utf-8")
    original = Path.read_bytes
    failed = False

    def fail_once(self: Path) -> bytes:
        nonlocal failed
        if self == path and not failed:
            failed = True
            raise OSError("transient read failure")
        return original(self)

    index = ReceiptIndex(tmp_path)
    with patch.object(Path, "read_bytes", fail_once):
        assert index.lookup("a" * 64) is None
        assert index.lookup("a" * 64) == path
    assert failed is True
