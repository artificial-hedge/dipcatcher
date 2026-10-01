"""epoch_position.v1 — O(log n) proof that an epoch receipt occupied a
specific position in the committed chain (the heads pin's chain_root)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from quant_fund.research.corpus_epoch import (
    corpus_epoch,
    load_heads_pin,
    update_heads_pin,
    write_epoch_receipt,
)
from quant_fund.research.epoch_merkle import (
    chain_position_proof,
    chain_tree_root,
    epoch_position_errors,
    epoch_position_receipt,
    verify_chain_position,
    verify_epoch_position,
    verify_epoch_position_pin,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _stamp(corpus: Path) -> str:
    return write_epoch_receipt(corpus_epoch(corpus), corpus).name


@pytest.fixture
def three_epoch(tmp_path: Path) -> tuple[Path, list[str]]:
    corpus = tmp_path / "receipts"
    corpus.mkdir()
    names: list[str] = []
    for i in range(3):
        (corpus / f"m{i}.json").write_text(json.dumps({"v": i}))
        names.append(_stamp(corpus))
    return corpus, names


def test_position_round_trip(three_epoch: tuple[Path, list[str]]) -> None:
    corpus, names = three_epoch
    for i, name in enumerate(names):
        body = epoch_position_receipt(corpus, name)
        assert body["position"] == i
        assert body["n_epochs"] == 3
        assert epoch_position_errors(body) == []
        assert verify_epoch_position(body, corpus) == []


def test_position_proof_binds_index(three_epoch: tuple[Path, list[str]]) -> None:
    corpus, names = three_epoch
    body = epoch_position_receipt(corpus, names[0])
    # Re-present the genesis proof at position 1 — the leaf carries the
    # index internally, so the claim can't be shifted.
    moved = dict(body, position=1)
    assert "position_bounds" not in epoch_position_errors(moved)  # 1 < 3 is in range
    assert "path_invalid" in epoch_position_errors(moved)
    assert "path_invalid" in verify_epoch_position(moved, corpus)


def test_position_forged_digest(three_epoch: tuple[Path, list[str]]) -> None:
    corpus, names = three_epoch
    body = epoch_position_receipt(corpus, names[1])
    forged = dict(body, sha256="0" * 64)
    errors = epoch_position_errors(forged)
    assert "path_invalid" in errors
    assert "path_invalid" in verify_epoch_position(forged, corpus)


def test_position_unknown_receipt_fails(three_epoch: tuple[Path, list[str]]) -> None:
    corpus, _ = three_epoch
    with pytest.raises(ValueError, match="not a committed epoch"):
        epoch_position_receipt(corpus, "corpus_epoch_0000000000000000.json")


def test_position_pin_mode(three_epoch: tuple[Path, list[str]], tmp_path: Path) -> None:
    corpus, names = three_epoch
    pin_path = tmp_path / "epoch_heads.json"
    update_heads_pin(pin_path, corpus, "*.json", corpus / names[-1])
    heads = load_heads_pin(pin_path)
    key = "receipts/*.json"
    # corpus dir is tmp_path/'receipts' — its key is the as_posix path.
    key = f"{corpus.as_posix()}/*.json" if f"{corpus.as_posix()}/*.json" in heads else key
    entry = heads[next(k for k in heads if k.endswith("/*.json"))]
    assert entry["n_epochs"] == 3
    assert "chain_root" in entry

    body = epoch_position_receipt(corpus, names[-1])
    body = dict(body, corpus_key=key if key in heads else next(iter(heads)))
    assert verify_epoch_position_pin(body, entry) == []

    # A head-position proof against a stale pin head must flag.
    stale = dict(entry, receipt=names[0])
    assert "head_receipt_not_pinned" in verify_epoch_position_pin(body, stale)
    # Tampered chain_root can't anchor the proof.
    bad = dict(entry, chain_root="0" * 64)
    assert "chain_root_not_pinned" in verify_epoch_position_pin(body, bad)


def test_position_mid_chain_entry(three_epoch: tuple[Path, list[str]]) -> None:
    """A non-head position verifies — mid-chain history is provable."""
    corpus, names = three_epoch
    body = epoch_position_receipt(corpus, names[0])
    assert body["position"] == 0
    assert verify_epoch_position(body, corpus) == []


def test_chain_tree_root_order_matters(three_epoch: tuple[Path, list[str]]) -> None:
    corpus, names = three_epoch
    from quant_fund.research.epoch_merkle import ordered_epoch_chain

    ordered, errs = ordered_epoch_chain(corpus, pattern="*.json")
    assert errs == []
    pairs = [(n, s) for n, s, _, _, _ in ordered]
    root = chain_tree_root(pairs)
    # Chain order is load-bearing: name-sorted order is a different tree.
    shuffled = sorted(pairs, key=lambda p: p[0])
    if shuffled != pairs:
        assert chain_tree_root(shuffled) != root


def test_chain_position_proof_shape() -> None:
    pairs = [(f"e{i}.json", f"{i:064d}".replace("1", "a")) for i in range(5)]
    # Fix hex legality: sha fields must be hex chars.
    pairs = [(f"e{i}.json", f"{i:0>64x}") for i in range(5)]
    proof = chain_position_proof(pairs, 3)
    assert proof["leaf_index"] == 3 and proof["n_members"] == 5
    assert verify_chain_position(3, "e3.json", pairs[3][1], proof, chain_tree_root(pairs))
    assert not verify_chain_position(2, "e3.json", pairs[3][1], proof, chain_tree_root(pairs))


def test_position_cli_emits_and_checks(three_epoch: tuple[Path, list[str]]) -> None:
    corpus, names = three_epoch
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_fund.cli.main",
            "epoch-position",
            "--corpus-dir",
            str(corpus),
            "--receipt",
            names[1],
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
