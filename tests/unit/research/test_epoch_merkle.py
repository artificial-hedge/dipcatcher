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


def test_absence_receipt_round_trip_and_pin(tmp_path: Path) -> None:
    """corpus_absence.v1: emits, passes contract + epoch + pin verification."""
    from quant_fund.research.corpus_epoch import (
        epoch_heads_key,
        load_heads_pin,
        update_heads_pin,
    )
    from quant_fund.research.epoch_merkle import (
        CORPUS_ABSENCE_SCHEMA,
        absence_receipt,
        corpus_absence_errors,
        verify_absence_pin,
        verify_epoch_absence,
    )

    corpus = _corpus(tmp_path, 5)
    receipt_path = write_epoch_receipt(corpus_epoch(corpus), corpus)
    body = absence_receipt(corpus, "rx.json")
    assert body["schema"] == CORPUS_ABSENCE_SCHEMA
    assert body["epoch_receipt"] == receipt_path.name
    assert corpus_absence_errors(body) == []
    assert verify_epoch_absence(body, corpus) == []

    pin = tmp_path / "quality" / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", receipt_path)
    pin_entry = load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]
    assert verify_absence_pin(body, pin_entry) == []


def test_absence_receipt_rejects_member_and_drift(tmp_path: Path) -> None:
    """A name that IS a member can't be receipted absent; epoch drift fails."""
    from quant_fund.research.epoch_merkle import (
        absence_receipt,
        verify_epoch_absence,
    )

    corpus = _corpus(tmp_path, 4)
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    with pytest.raises(ValueError, match="is a member"):
        absence_receipt(corpus, "r0.json")
    body = absence_receipt(corpus, "rx.json")
    # Absence is epoch-bound: a member added AFTER the stamp can't unwind the
    # proof — the append-only chain keeps epoch N's map frozen.
    (corpus / "zz.json").write_text("{}")
    assert verify_epoch_absence(body, corpus) == []
    # An epoch receipt tampered without resealing fails the seal recompute —
    # the authenticated loader refuses it before the root is even compared.
    import json as _json

    epoch_file = corpus / body["epoch_receipt"]
    doc = _json.loads(epoch_file.read_text())
    doc["epoch_root_sha256"] = "0" * 64
    epoch_file.write_text(_json.dumps(doc))
    assert verify_epoch_absence(body, corpus) == ["epoch_receipt_tampered"]


def test_absence_pin_forged_and_stale(tmp_path: Path) -> None:
    """Pin mode: forged bounds + a stale epoch both fail."""
    from quant_fund.research.corpus_epoch import (
        epoch_heads_key,
        load_heads_pin,
        update_heads_pin,
    )
    from quant_fund.research.epoch_merkle import (
        absence_receipt,
        verify_absence_pin,
    )

    corpus = _corpus(tmp_path, 5)
    receipt_path = write_epoch_receipt(corpus_epoch(corpus), corpus)
    pin = tmp_path / "quality" / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", receipt_path)
    pin_entry = load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]

    body = absence_receipt(corpus, "rx.json")
    # Forge: claim a wrong root — pinned tree_root mismatch.
    forged = dict(body, merkle_root="f" * 64)
    errs = verify_absence_pin(forged, pin_entry)
    assert errs
    # Stale epoch binding.
    new_epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    stale = absence_receipt(corpus, "rx.json", epoch_receipt=new_epoch.name)
    assert "epoch_receipt_not_pinned" in verify_absence_pin(stale, pin_entry)


def test_absence_edge_bound_via_receipt(tmp_path: Path) -> None:
    """A name beyond the last member proves absence with a single edge bound."""
    from quant_fund.research.epoch_merkle import absence_receipt, corpus_absence_errors

    corpus = _corpus(tmp_path, 3)
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    body = absence_receipt(corpus, "zzz.json")
    assert len(body["bounds"]) == 1
    assert corpus_absence_errors(body) == []


def test_proof_layer_semantic_mutation_fuzz(tmp_path: Path) -> None:
    """Crafted forgeries on corpus_proof.v1 / corpus_absence.v1 must be
    rejected — byte mutations are covered by the committed-corpus fuzz; these
    target the proof semantics (shape-bound replay, adjacency, member map)."""
    import copy

    from quant_fund.research.epoch_merkle import (
        absence_receipt,
        corpus_absence_errors,
        corpus_proof_errors,
        verify_epoch_absence,
        verify_epoch_proof,
    )
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    corpus = _corpus(tmp_path, 7)
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    proof = member_proof(corpus, "r3.json")
    absent = absence_receipt(corpus, "r2x.json")  # interior gap: two bounds
    assert len(absent["bounds"]) == 2

    def corrupt(body: dict[str, object], **patch: object) -> dict[str, object]:
        out = copy.deepcopy(body)
        for key, value in patch.items():
            if key == "path0_side":
                path = out["path"]
                assert isinstance(path, list)
                hop = path[0]
                assert isinstance(hop, dict)
                hop["side"] = "right" if hop["side"] == "left" else "left"
            elif key == "path0_hash":
                path = out["path"]
                assert isinstance(path, list)
                hop = path[0]
                assert isinstance(hop, dict)
                hop["sha256"] = "0" * 64
            elif key == "reverse_path":
                out["path"] = list(reversed(out["path"]))  # type: ignore[arg-type]
            elif key == "bound0_index":
                bounds = out["bounds"]
                assert isinstance(bounds, list)
                bound = bounds[0]
                assert isinstance(bound, dict)
                bound["leaf_index"] = int(bound["leaf_index"]) + 1  # type: ignore[call-overload]
            elif key == "swap_bounds":
                out["bounds"] = list(reversed(out["bounds"]))  # type: ignore[arg-type]
            else:
                out[key] = value
        return out

    forgeries = [
        # proof-level: shape-bound index replay catches index/hash/side edits
        (corrupt(proof, leaf_index=proof["leaf_index"] + 1), corpus_proof_errors),
        (corrupt(proof, member="r4.json"), corpus_proof_errors),
        (corrupt(proof, path0_side=True), corpus_proof_errors),
        (corrupt(proof, path0_hash=True), corpus_proof_errors),
        (corrupt(proof, reverse_path=True), corpus_proof_errors),
        (corrupt(proof, merkle_root="f" * 64), corpus_proof_errors),
        # absence-level: adjacency + bound replay
        (corrupt(absent, bound0_index=True), corpus_absence_errors),
        (corrupt(absent, swap_bounds=True), corpus_absence_errors),
        (corrupt(absent, merkle_root="f" * 64), corpus_absence_errors),
    ]
    for i, (forged, contract) in enumerate(forgeries):
        assert contract(forged), f"forgery #{i} passed the internal contract"

    # Epoch-boundary attacks the internal contract can't see — the member map
    # is the ground truth, not the claimed name.
    renamed = corrupt(absent, name="r3.json")
    # Renaming onto a member name makes a bound equal the claim, and the
    # bracket check independently rejects it — coherent renames are dead.
    assert corpus_absence_errors(renamed)
    assert verify_epoch_absence(renamed, corpus) != []
    # …but a member can't be absent. Same for a proof pointing at a name that
    # isn't in the epoch's member map.
    ghost = corrupt(proof, member="ghost.json")
    assert verify_epoch_proof(ghost, corpus) != []
    # Pin-mode forgery: adjacent bounds elsewhere in the tree claimed as an
    # absence proof for a member — there is no member map offline, so only
    # the bracket check can reject it.
    unbracketed = corrupt(absent, name="r5.json")
    assert "bounds_not_bracketing" in corpus_absence_errors(unbracketed)
    # n_members lies that are shape-equivalent at this leaf_index are
    # invisible to the path replay — the epoch's member count pins them.
    inflated = corrupt(proof, n_members=proof["n_members"] + 1)
    assert corpus_proof_errors(inflated) == []
    assert verify_epoch_proof(inflated, corpus) == ["n_members_mismatch"]
    inflated_abs = corrupt(absent, n_members=absent["n_members"] + 1)
    assert corpus_absence_errors(inflated_abs) == []
    assert verify_epoch_absence(inflated_abs, corpus) == ["n_members_mismatch"]
    # Cross-epoch confusion: a proof minted under a different epoch receipt.
    write_epoch_receipt(corpus_epoch(corpus), corpus)  # advance head
    other = absence_receipt(corpus, "zz9.json")
    moved = corrupt(absent, epoch_receipt=other["epoch_receipt"])
    assert verify_epoch_absence(moved, corpus) != []

    # Forged epoch receipt: a co-edited members+root tamper under the pinned
    # filename fails the seal recompute; resealing changes the seal so the
    # pinned filename (sha16 of the seal) can no longer hold the forgery.
    import json as _json

    epoch_file = corpus / str(proof["epoch_receipt"])
    doc = _json.loads(epoch_file.read_text())
    doc["members"].append({"name": "evil.json", "sha256": "0" * 64})
    doc["epoch_root_sha256"] = doc["epoch_root_sha256"]  # fields re-agree or not —
    epoch_file.write_text(_json.dumps(doc))  # the seal is stale either way
    assert verify_epoch_proof(proof, corpus) == ["epoch_receipt_tampered"]

    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    doc["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    epoch_file.write_text(_json.dumps(doc))
    assert verify_epoch_proof(proof, corpus) == ["epoch_receipt_name_mismatch"]


def test_unicode_nfc_fail_closed(tmp_path: Path) -> None:
    """NFD/non-canonical name spellings alias real files on APFS/Windows —
    proofs must name NFC-canonical members, and stamps refuse them."""
    import unicodedata

    import pytest

    from quant_fund.research.corpus_epoch import _nfc_errors, corpus_epoch
    from quant_fund.research.epoch_merkle import (
        absence_proof,
        absence_receipt,
        corpus_absence_errors,
        inclusion_proof,
        member_proof,
        verify_epoch_absence,
        verify_epoch_proof,
    )

    nfd = unicodedata.normalize("NFD", "café.json")
    assert nfd != unicodedata.normalize("NFC", "café.json")  # distinct strings

    # Stamp side: a members map carrying an NFD key is refused outright, and
    # distinct names colliding under NFC+casefold (impossible on APFS/NTFS
    # checkouts) are rejected as name aliases.
    assert _nfc_errors({"ok.json": "0" * 64}) == []
    assert _nfc_errors({nfd: "0" * 64}) == [f"member_name_not_nfc:{nfd!a}"]
    assert _nfc_errors({"A.json": "0" * 64, "a.json": "1" * 64}) == [
        "member_name_alias:'A.json'|'a.json'"
    ]
    assert _nfc_errors({nfd: "0" * 64, unicodedata.normalize("NFC", "café.json"): "1" * 64}) != []

    corpus = _corpus(tmp_path, 5)
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    proof = member_proof(corpus, "r2.json")
    absent = absence_receipt(corpus, "zz9.json")

    # Builders refuse non-canonical names.
    with pytest.raises(ValueError, match="NFC"):
        inclusion_proof({"a.json": "0" * 64}, nfd)
    with pytest.raises(ValueError, match="NFC"):
        absence_proof({"a.json": "0" * 64}, nfd)
    with pytest.raises(ValueError, match="NFC"):
        member_proof(corpus, nfd)
    with pytest.raises(ValueError, match="NFC"):
        absence_receipt(corpus, nfd)

    # Verify side: an NFD-spelled member/name/bound is rejected even when the
    # NFC twin is a real member — the proof must carry canonical spelling.
    forged_member = {**proof, "member": nfd}
    assert verify_epoch_proof(forged_member, corpus) == ["member_name_not_nfc"]
    forged_absent = {**absent, "name": nfd}
    assert corpus_absence_errors(forged_absent) == ["name_not_nfc"]
    assert verify_epoch_absence(forged_absent, corpus) == ["name_not_nfc"]
    forged_bound = {
        **absent,
        "bounds": [{**absent["bounds"][0], "member": nfd}],
    }
    assert corpus_absence_errors(forged_bound) == ["bound_member_not_nfc"]


def _stamped(corpus: Path) -> Path:
    return write_epoch_receipt(corpus_epoch(corpus), corpus)


def test_history_absence_round_trip_and_pin(tmp_path: Path) -> None:
    """corpus_history_absence.v1: emits for a never-member, verifies live
    and offline against the heads pin."""
    from quant_fund.research.corpus_epoch import (
        epoch_heads_key,
        load_heads_pin,
        update_heads_pin,
    )
    from quant_fund.research.epoch_merkle import (
        HISTORY_ABSENCE_SCHEMA,
        history_absence_errors,
        history_absence_receipt,
        verify_history_absence,
        verify_history_absence_pin,
    )

    corpus = _corpus(tmp_path, 4)
    _stamped(corpus)
    (corpus / "late.json").write_text("{}")
    head = _stamped(corpus)

    body = history_absence_receipt(corpus, "secrets.env")
    assert body["schema"] == HISTORY_ABSENCE_SCHEMA
    assert body["n_epochs"] == 2
    assert body["head_receipt"] == head.name
    assert history_absence_errors(body) == []
    assert verify_history_absence(body, corpus) == []

    pin = tmp_path / "quality" / "epoch_heads.json"
    update_heads_pin(pin, corpus, "*.json", head)
    pin_entry = load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]
    assert verify_history_absence_pin(body, pin_entry) == []


def test_history_absence_refuses_mid_chain_member(tmp_path: Path) -> None:
    """The discriminator: present at epoch 2, absent at head — a tip-bound
    absence receipt is valid but a history claim must refuse to build."""
    from quant_fund.research.epoch_merkle import (
        absence_receipt,
        history_absence_receipt,
        verify_epoch_absence,
    )

    corpus = _corpus(tmp_path, 3)
    _stamped(corpus)
    (corpus / "ghost.json").write_text("{}")
    _stamped(corpus)  # ghost stamped at epoch 2
    (corpus / "ghost.json").unlink()
    _stamped(corpus)  # removed again by epoch 3 (head)

    # Tip-bound absence: legitimately clean.
    tip = absence_receipt(corpus, "ghost.json")
    assert verify_epoch_absence(tip, corpus) == []
    # History absence: refuses — ghost WAS evidence at epoch 2.
    with pytest.raises(ValueError, match="is a member at"):
        history_absence_receipt(corpus, "ghost.json")


def test_verify_history_absence_mutations(tmp_path: Path) -> None:
    """Dropped epochs, swapped heads, forged digests all fail closed."""
    from quant_fund.research.corpus_epoch import (
        epoch_heads_key,
        load_heads_pin,
        update_heads_pin,
    )
    from quant_fund.research.epoch_merkle import (
        history_absence_receipt,
        verify_history_absence,
        verify_history_absence_pin,
    )

    corpus = _corpus(tmp_path, 3)
    _stamped(corpus)
    (corpus / "zz.json").write_text("{}")
    head = _stamped(corpus)
    body = history_absence_receipt(corpus, "never.json")

    # Drop the genesis epoch from the claimed list — the next entry then
    # claims to be genesis while carrying a real prev_root: fails shape.
    forged = dict(body, epochs=body["epochs"][1:], n_epochs=1)
    assert verify_history_absence(forged, corpus) == ["genesis_prev_root_nonzero"]
    forged2 = dict(body, epochs=[*body["epochs"], dict(body["epochs"][-1])])
    forged2["epochs"][-1]["receipt"] = "corpus_epoch_deadbeef.json"
    forged2["n_epochs"] = 3
    forged2["head_receipt"] = "corpus_epoch_deadbeef.json"
    assert verify_history_absence(forged2, corpus) == ["history_chain_mismatch"]

    # A name that IS stamped cannot verify absent.
    present_claim = dict(body, name="r0.json")
    assert verify_history_absence(present_claim, corpus) == [
        f"history_member_present:{body['epochs'][0]['receipt']!a}"
    ]

    # Pin mode: head must be the pinned head — a stale-head claim fails.
    pin = tmp_path / "quality" / "epoch_heads.json"
    first = corpus / body["epochs"][0]["receipt"]  # epoch 1, not the head
    update_heads_pin(pin, corpus, "*.json", first)
    pin_entry = load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]
    errs = verify_history_absence_pin(body, pin_entry)
    assert "history_head_not_pinned" in errs
    update_heads_pin(pin, corpus, "*.json", head)
    pin_entry = load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]
    assert verify_history_absence_pin(body, pin_entry) == []


def test_history_absence_broken_chain_fails_closed(tmp_path: Path) -> None:
    """A forked/truncated chain can't emit or verify a history claim."""

    from quant_fund.research.epoch_merkle import (
        history_absence_receipt,
        verify_history_absence,
    )

    corpus = _corpus(tmp_path, 3)
    _stamped(corpus)
    _stamped(corpus)
    # Forge a disconnected third epoch claiming a bogus prev.
    third = corpus_epoch(corpus)
    third["prev_epoch_receipt"] = "corpus_epoch_forged0000.json"
    third["prev_epoch_sha256"] = "0" * 64
    from quant_fund.research.corpus_epoch import write_epoch_receipt as _w

    _w(third, corpus)
    with pytest.raises(ValueError, match="chain not verifiable"):
        history_absence_receipt(corpus, "x.json")
    # And an emitted receipt can't verify against the broken corpus either:
    # the independent re-walk fails before the claimed sequence is compared.
    body_corpus = _corpus(tmp_path / "b", 2)
    _stamped(body_corpus)
    good = history_absence_receipt(body_corpus, "x.json")
    errs = verify_history_absence(good, corpus)
    assert errs and errs[0].startswith("history_chain_unverifiable")
