"""Corpus epochs: hash-chained membership root makes the store tamper-evident."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.corpus_epoch import (
    GENESIS_PREV,
    check_epoch_chain,
    corpus_epoch,
    epoch_contract_errors,
    epoch_root,
    member_digests,
    write_epoch_receipt,
)
from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_file


def _receipt(dirpath: Path, name: str, marker: str) -> Path:
    body = {
        "kind": "synthetic_fixture.v1",
        "schema": "synthetic_fixture.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "SYNTHETIC",
        "inputs_sha256": "ab" * 32,
        "results": [{"marker": marker}],
    }
    path = dirpath / name
    path.write_text(json.dumps(seal_receipt(body), indent=2, sort_keys=True))
    return path


def _stamp(corpus: Path) -> Path:
    return write_epoch_receipt(corpus_epoch(corpus), corpus)


def test_genesis_epoch(corpus_dir: Path) -> None:
    receipt = corpus_epoch(corpus_dir)
    assert receipt["verdict"] == "genesis"
    assert receipt["prev_epoch_sha256"] == GENESIS_PREV
    assert receipt["prev_epoch_receipt"] is None
    assert receipt["n_members"] == 2
    assert epoch_contract_errors(receipt) == []


def test_epoch_root_deterministic_and_order_free() -> None:
    m1 = {"a": "ab" * 32, "b": "cd" * 32}
    m2 = {"b": "cd" * 32, "a": "ab" * 32}
    assert epoch_root(m1) == epoch_root(m2)
    assert epoch_root(m1) != epoch_root({"a": "ab" * 32})


def test_chain_advances_on_growth(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    _receipt(corpus_dir, "c.json", "3")
    receipt = corpus_epoch(corpus_dir)
    assert receipt["verdict"] == "advancing"
    # c.json joins, and the first epoch receipt is itself a corpus member.
    assert "c.json" in receipt["members_added"]
    assert any(n.startswith("corpus_epoch_") for n in receipt["members_added"])
    assert receipt["members_removed"] == []
    _stamp(corpus_dir)
    assert check_epoch_chain(corpus_dir) == {"errors": [], "unstamped": []}


def test_deleted_member_detected(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    (corpus_dir / "a.json").unlink()
    receipt = corpus_epoch(corpus_dir)
    assert receipt["verdict"] == "shrinking"
    assert receipt["members_removed"] == ["a.json"]
    _stamp(corpus_dir)
    errors = check_epoch_chain(corpus_dir)["errors"]
    assert any("member_removed:a.json" in e for e in errors)


def test_allowed_removal_pin(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    digest = member_digests(corpus_dir)["a.json"]
    (corpus_dir / "a.json").unlink()
    _stamp(corpus_dir)
    errors = check_epoch_chain(corpus_dir, allowed_removals={"a.json": digest})["errors"]
    assert not any("member_removed:a.json" in e for e in errors)


def test_mutated_member_detected(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    (corpus_dir / "a.json").write_text('{"mutated": true}')
    _stamp(corpus_dir)  # same name, new digest — mutation, not removal
    errors = check_epoch_chain(corpus_dir)["errors"]
    assert any("member_mutated:a.json" in e for e in errors)


def test_post_stamp_arrival_is_unstamped_not_error(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    _receipt(corpus_dir, "late.json", "sneaked in post-stamp")
    result = check_epoch_chain(corpus_dir)
    assert result["errors"] == []
    assert "late.json" in result["unstamped"]


def test_dishonest_delta_flagged(corpus_dir: Path) -> None:
    _stamp(corpus_dir)
    _receipt(corpus_dir, "c.json", "3")
    epoch = corpus_epoch(corpus_dir)
    epoch["members_added"] = []  # lie about the delta
    write_epoch_receipt(epoch, corpus_dir)
    errors = check_epoch_chain(corpus_dir)["errors"]
    assert any("members_added_dishonest" in e for e in errors)


def test_contract_rejects_tampered_members(corpus_dir: Path) -> None:
    epoch = corpus_epoch(corpus_dir)
    bad = dict(epoch)
    bad["members"] = bad["members"][:-1]
    errors = epoch_contract_errors(bad)
    assert "n_members_mismatch" in errors
    assert "epoch_root_mismatch" in errors
    with pytest.raises(ValueError, match="contract"):
        write_epoch_receipt(bad, corpus_dir)


def test_written_epoch_verifies(corpus_dir: Path) -> None:
    path = _stamp(corpus_dir)
    ver = verify_receipt_file(path)
    assert ver["valid"], ver["errors"]


def test_no_epochs_reports_missing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assert check_epoch_chain(corpus)["errors"] == ["no_epoch_receipts"]


def test_fails_closed_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        corpus_epoch(tmp_path / "nope")


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    _receipt(root, "a.json", "1")
    _receipt(root, "b.json", "2")
    return root
