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


def test_absence_proof_brackets_gap() -> None:
    from quant_fund.research.epoch_merkle import absence_proof, merkle_root, verify_absence
    from quant_fund.utils.hashing import hash_bytes

    mem = {f"{c}.json": hash_bytes(c.encode()) for c in "abcde"}
    root = merkle_root(mem)
    for absent in ("bb.json", "0.json", "zz.json"):
        proof = absence_proof(mem, absent)
        assert verify_absence(proof, root) == [], absent


def test_absence_proof_rejects_forged_index() -> None:
    """A bound's claimed leaf_index cannot be fudged — path shape binds it."""
    from quant_fund.research.epoch_merkle import (
        absence_proof,
        inclusion_proof,
        merkle_root,
        verify_absence,
    )
    from quant_fund.utils.hashing import hash_bytes

    mem = {f"{c}.json": hash_bytes(c.encode()) for c in "abcde"}
    root = merkle_root(mem)
    # try to "prove" real member c.json absent using a+e with fudged indices
    b_a = {"member": "a.json", "member_sha256": mem["a.json"], **inclusion_proof(mem, "a.json")}
    b_e = {
        "member": "e.json",
        "member_sha256": mem["e.json"],
        **inclusion_proof(mem, "e.json"),
        "leaf_index": 1,
    }
    forged = {"name": "c.json", "merkle_root": root, "n_members": 5, "bounds": [b_a, b_e]}
    assert verify_absence(forged, root) != []
    # honest bounds for a real gap still pass after tightening
    assert verify_absence(absence_proof(mem, "bb.json"), root) == []


def test_inclusion_index_is_shape_bound() -> None:
    for n in range(1, 9):
        from quant_fund.research.epoch_merkle import inclusion_proof, merkle_root, verify_inclusion
        from quant_fund.utils.hashing import hash_bytes

        mem = {f"m{i}.json": hash_bytes(f"m{i}".encode()) for i in range(n)}
        root = merkle_root(mem)
        for name in mem:
            proof = inclusion_proof(mem, name)
            assert verify_inclusion(name, mem[name], proof, root)
            if n > 1:
                wrong = dict(proof, leaf_index=(proof["leaf_index"] + 1) % n)
                assert not verify_inclusion(name, mem[name], wrong, root), (n, name)


def test_epoch_payload_carries_member_tree_root(tmp_path: Path) -> None:
    from quant_fund.research.corpus_epoch import epoch_contract_errors, member_tree_root
    from quant_fund.research.epoch_merkle import merkle_root

    corpus = _corpus(tmp_path)
    receipt = corpus_epoch(corpus)
    members = {m["name"]: m["sha256"] for m in receipt["members"]}
    assert receipt["member_tree_root"] == merkle_root(members) == member_tree_root(members)
    assert epoch_contract_errors(receipt) == []
    drifted = dict(receipt, member_tree_root="0" * 64)
    assert "member_tree_root_mismatch" in epoch_contract_errors(drifted)


def test_heads_pin_carries_tree_root(tmp_path: Path) -> None:
    from quant_fund.research.corpus_epoch import (
        corpus_epoch,
        load_heads_pin,
        update_heads_pin,
        write_epoch_receipt,
    )

    corpus = _corpus(tmp_path)
    pin = tmp_path / "quality" / "epoch_heads.json"
    receipt_path = write_epoch_receipt(corpus_epoch(corpus), corpus)
    update_heads_pin(pin, corpus, "*.json", receipt_path)
    from quant_fund.research.corpus_epoch import epoch_heads_key

    heads = load_heads_pin(pin)
    entry = heads[epoch_heads_key(corpus, "*.json")]
    assert entry["tree_root"] == member_tree_root_from_receipt(receipt_path)


def member_tree_root_from_receipt(path: Path) -> str:
    import json as _json

    doc = _json.loads(path.read_text())
    body = doc.get("payload", doc)
    return body["member_tree_root"]


def test_verify_proof_pin_offline(tmp_path: Path) -> None:
    """Third-party path: proof + signed heads pin only — no corpus access."""
    from quant_fund.research.corpus_epoch import (
        corpus_epoch,
        load_heads_pin,
        update_heads_pin,
        write_epoch_receipt,
    )
    from quant_fund.research.epoch_merkle import member_proof, verify_proof_pin

    corpus = _corpus(tmp_path)
    pin = tmp_path / "quality" / "epoch_heads.json"
    receipt_path = write_epoch_receipt(corpus_epoch(corpus), corpus)
    update_heads_pin(pin, corpus, "*.json", receipt_path)
    from quant_fund.research.corpus_epoch import epoch_heads_key

    pin_entry = load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]
    proof = member_proof(corpus, "r3.json")
    assert verify_proof_pin(proof, pin_entry) == []
    # tampered pin
    bad = dict(pin_entry, tree_root="f" * 64)
    assert "merkle_root_not_pinned" in verify_proof_pin(proof, bad)
    # a proof bound to a different epoch never satisfies this pin
    new_path = write_epoch_receipt(corpus_epoch(corpus), corpus)  # advance the head
    proof2 = member_proof(corpus, "r3.json", epoch_receipt=new_path.name)
    assert "epoch_receipt_not_pinned" in verify_proof_pin(proof2, pin_entry)
