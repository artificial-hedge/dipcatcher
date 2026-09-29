"""Tests for epoch_merkle — CT-style inclusion proofs over corpus epochs."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.research.corpus_epoch import corpus_epoch, write_epoch_receipt
from quant_fund.research.epoch_merkle import (
    CORPUS_PROOF_SCHEMA,
    corpus_proof_errors,
    inclusion_proof,
    member_proof,
    merkle_root,
    verify_epoch_proof,
    verify_inclusion,
)


def _corpus(tmp_path: Path, n: int = 7) -> Path:
    corpus = tmp_path / "receipts"
    corpus.mkdir(parents=True)
    for i in range(n):
        (corpus / f"r{i}.json").write_text(f'{{"i": {i}}}')
    return corpus


def test_merkle_root_deterministic_and_order_free() -> None:
    m = {"a": "aa" * 32, "b": "bb" * 32, "c": "cc" * 32}
    assert merkle_root(m) == merkle_root(dict(reversed(list(m.items()))))
    assert len(merkle_root(m)) == 64


def test_merkle_root_single_member() -> None:
    root = merkle_root({"only": "00" * 32})
    assert len(root) == 64


def test_merkle_root_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        merkle_root({})


def test_inclusion_proof_round_trip_all_sizes() -> None:
    for n in (1, 2, 3, 4, 5, 8, 9, 17):
        members = {f"m{i:02d}": f"{i:064x}" for i in range(n)}
        root = merkle_root(members)
        for name in members:
            proof = inclusion_proof(members, name)
            assert proof["n_members"] == n
            assert verify_inclusion(name, members[name], proof, root)


def test_inclusion_proof_rejects_tampered_member() -> None:
    members = {f"m{i}": f"{i:064x}" for i in range(5)}
    root = merkle_root(members)
    proof = inclusion_proof(members, "m2")
    # Same path, wrong claimed digest — leaf hash differs, root recomputes wrong.
    assert not verify_inclusion("m2", "ff" * 32, proof, root)
    # Right digest but wrong member name — leaf binds name.
    assert not verify_inclusion("m0", members["m2"], proof, root)


def test_inclusion_proof_rejects_wrong_root() -> None:
    members = {"a": "11" * 32, "b": "22" * 32}
    proof = inclusion_proof(members, "a")
    assert not verify_inclusion("a", members["a"], proof, "ff" * 32)


def test_inclusion_proof_unknown_member_raises() -> None:
    with pytest.raises(ValueError, match="not a corpus member"):
        inclusion_proof({"a": "00" * 32}, "ghost")


def test_member_proof_against_live_epoch(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 6)
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    # Second epoch: head is digest-named, not lex-last — proves head picking.
    (corpus / "extra.json").write_text("{}")
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    body = member_proof(corpus, "extra.json")
    assert body["schema"] == CORPUS_PROOF_SCHEMA
    assert body["member"] == "extra.json"
    assert corpus_proof_errors(body) == []
    assert verify_epoch_proof(body, corpus) == []


def test_verify_epoch_proof_fails_on_drift(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 4)
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    body = member_proof(corpus, "r1.json")
    # Tampering member_sha256 invalidates the leaf, so the contract layer
    # flags proof_path_invalid before the epoch cross-check runs.
    body2 = dict(body, member_sha256="ff" * 32)
    assert verify_epoch_proof(body2, corpus) == ["proof_path_invalid"]
    # And a digest that the epoch never recorded is rejected too.
    body3 = dict(body, member_sha256=None)
    assert "member_sha256_not_hex" in verify_epoch_proof(body3, corpus)


def test_verify_epoch_proof_fails_on_deleted_epoch(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 3)
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    body = member_proof(corpus, "r0.json")
    (corpus / body["epoch_receipt"]).unlink()
    assert verify_epoch_proof(body, corpus) == ["epoch_receipt_missing"]


def test_verify_epoch_proof_fails_on_fabricated_path(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 5)
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    body = member_proof(corpus, "r2.json")
    forged = dict(body, path=list(body["path"]))
    forged["path"][0] = {**forged["path"][0], "sha256": "00" * 32}
    assert "proof_path_invalid" in corpus_proof_errors(forged)
    assert verify_epoch_proof(forged, corpus)


def test_member_proof_no_epochs_raises(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 2)
    with pytest.raises(ValueError, match="no corpus epochs"):
        member_proof(corpus, "r0.json")
