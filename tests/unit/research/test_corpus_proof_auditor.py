"""Differential pin: scripts/verify_corpus_proof.py ↔ epoch_merkle library.

The standalone script is the third-party oracle — a divergence between its
verdict and the library's is itself a finding. Covers a real emitted proof,
forged-index absence frames, and a stale-pin rejection.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "verify_corpus_proof.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_script_verifies_library_emitted_proof(tmp_path: Path) -> None:
    """Library-emitted sealed proof verifies clean under the script."""
    from quant_fund.research.corpus_epoch import corpus_epoch, epoch_heads_key, write_epoch_receipt
    from quant_fund.research.epoch_merkle import member_proof
    from quant_fund.research.receipt_v2 import seal_receipt

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    members = {m["name"]: m["sha256"] for m in json.loads(epoch.read_text())["members"]}
    from quant_fund.research.epoch_merkle import merkle_root

    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps(
            {
                "heads": {
                    epoch_heads_key(corpus, "*.json"): {
                        "receipt": epoch.name,
                        "sha256": "0" * 64,
                        "tree_root": merkle_root(members),
                    }
                }
            }
        )
    )
    body = member_proof(corpus, "c.json")
    proof_path = tmp_path / "proof.json"
    proof_path.write_text(json.dumps(seal_receipt(body)))

    proc = _run("--proof", str(proof_path), "--pin", str(pin))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


def test_script_rejects_forged_index(tmp_path: Path) -> None:
    """A proof with a fudged leaf_index must not recompute the pin."""
    from quant_fund.research.corpus_epoch import corpus_epoch, epoch_heads_key, write_epoch_receipt
    from quant_fund.research.epoch_merkle import member_proof, merkle_root

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    members = {m["name"]: m["sha256"] for m in json.loads(epoch.read_text())["members"]}
    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps(
            {
                "heads": {
                    epoch_heads_key(corpus, "*.json"): {
                        "receipt": epoch.name,
                        "sha256": "0" * 64,
                        "tree_root": merkle_root(members),
                    }
                }
            }
        )
    )
    body = member_proof(corpus, "c.json")
    forged = dict(body, leaf_index=(body["leaf_index"] + 1) % body["n_members"])
    proof_path = tmp_path / "forged.json"
    proof_path.write_text(json.dumps(forged))
    proc = _run("--proof", str(proof_path), "--pin", str(pin))
    assert proc.returncode == 1
    assert "path_side_mismatch" in proc.stdout or "path_depth_mismatch" in proc.stdout


def test_script_absence_bracket(tmp_path: Path) -> None:
    """A real absence proof verifies; a wrong pin root is rejected."""
    from quant_fund.research.corpus_epoch import corpus_epoch, epoch_heads_key, write_epoch_receipt
    from quant_fund.research.epoch_merkle import absence_proof, merkle_root

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    members = {m["name"]: m["sha256"] for m in json.loads(epoch.read_text())["members"]}
    root = merkle_root(members)
    pin = tmp_path / "pin.json"
    key = epoch_heads_key(corpus, "*.json")
    pin.write_text(
        json.dumps({"heads": {key: {"receipt": epoch.name, "sha256": "0" * 64, "tree_root": root}}})
    )
    proof = absence_proof(members, "b.json")
    ap = tmp_path / "abs.json"
    ap.write_text(json.dumps(proof))
    proc = _run("--absence", str(ap), "--pin", str(pin), "--key", key)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Edge bound on a promoted leaf (idx == n-1, odd n): the promotion level
    # contributes no path entry — a divergence once shipped here.
    edge = absence_proof(members, "z.json")
    ep = tmp_path / "abs_edge.json"
    ep.write_text(json.dumps(edge))
    proc_edge = _run("--absence", str(ep), "--pin", str(pin), "--key", key)
    assert proc_edge.returncode == 0, proc_edge.stdout + proc_edge.stderr

    bad_pin = tmp_path / "bad.json"
    bad_pin.write_text(
        json.dumps(
            {"heads": {key: {"receipt": epoch.name, "sha256": "0" * 64, "tree_root": "f" * 64}}}
        )
    )
    proc2 = _run("--absence", str(ap), "--pin", str(bad_pin), "--key", key)
    assert proc2.returncode == 1
    assert "merkle_root_not_pinned" in proc2.stdout


def _signed_checkpoint(pin_heads: dict, pins: dict, tmp_path: Path) -> tuple[Path, Path]:
    """A checkpoint envelope signed by a throwaway key — exercises the
    script's Ed25519 auth path end-to-end."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    key = Ed25519PrivateKey.generate()
    pub_hex = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    payload = {
        "schema": "integrity_checkpoint.v1",
        "at": "2026-01-01T00:00:00+00:00",
        "pins": pins,
        "heads": pin_heads,
        "missing_pins": [],
        "prev_sha256": None,
        "spine": {"n_archives": 0, "tip": None},
        "witness": {"n_proofs": 0, "proofs": {}},
        "code": {"revision": "test", "worktree_sha256": "0" * 64},
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    sig = key.sign(canon).hex()
    cp = tmp_path / "checkpoint.json"
    cp.write_text(
        json.dumps(
            {
                "schema": "integrity_checkpoint_sig.v1",
                "algorithm": "ed25519",
                "key_id": pub_hex[:16],
                "payload": payload,
                "signature": sig,
            }
        )
    )
    pub = tmp_path / "gate_signing.pub"
    pub.write_text(pub_hex)
    return cp, pub


def test_script_checkpoint_mode(tmp_path: Path) -> None:
    """Proof + signed checkpoint alone (no pin file) verifies; a checkpoint
    signed by a foreign key or cross-pinning other bytes fails."""
    from quant_fund.research.corpus_epoch import corpus_epoch, epoch_heads_key, write_epoch_receipt
    from quant_fund.research.epoch_merkle import member_proof, merkle_root
    from quant_fund.research.receipt_v2 import seal_receipt

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    members = {m["name"]: m["sha256"] for m in json.loads(epoch.read_text())["members"]}
    root = merkle_root(members)
    key = epoch_heads_key(corpus, "*.json")
    pin_heads = {key: {"receipt": epoch.name, "sha256": "0" * 64, "tree_root": root}}

    pin_file = tmp_path / "quality" / "epoch_heads.json"
    pin_file.parent.mkdir(parents=True, exist_ok=True)
    pin_file.write_text(json.dumps({"heads": pin_heads}))
    pins = {"quality/epoch_heads.json": "0" * 64}
    import hashlib

    pins["quality/epoch_heads.json"] = hashlib.sha256(pin_file.read_bytes()).hexdigest()

    cp, pub = _signed_checkpoint(pin_heads, pins, tmp_path)
    body = member_proof(corpus, "c.json")
    proof_path = tmp_path / "proof.json"
    proof_path.write_text(json.dumps(seal_receipt(body)))

    # checkpoint-only mode
    proc = _run(
        "--proof",
        str(proof_path),
        "--checkpoint",
        str(cp),
        "--pubkey",
        str(pub),
        "--key",
        key,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # checkpoint + pin cross-anchor
    proc2 = _run(
        "--proof",
        str(proof_path),
        "--pin",
        str(pin_file),
        "--checkpoint",
        str(cp),
        "--pubkey",
        str(pub),
        "--key",
        key,
    )
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr

    # swapped pin bytes must trip the cross-anchor
    pin_file.write_text(json.dumps({"heads": pin_heads, "pad": 1}))
    proc3 = _run(
        "--proof",
        str(proof_path),
        "--pin",
        str(pin_file),
        "--checkpoint",
        str(cp),
        "--pubkey",
        str(pub),
        "--key",
        key,
    )
    assert proc3.returncode == 1
    assert "checkpoint_pin_mismatch" in proc3.stdout

    # foreign-signed checkpoint must fail the signature layer
    (tmp_path / "f").mkdir()
    cp2, pub2 = _signed_checkpoint(pin_heads, pins, tmp_path / "f")
    forged = json.loads(cp2.read_text())
    forged["payload"]["heads"] = pin_heads  # same content, foreign key
    cp3 = tmp_path / "checkpoint_forged.json"
    cp3.write_text(json.dumps(forged))
    proc4 = _run(
        "--proof",
        str(proof_path),
        "--checkpoint",
        str(cp3),
        "--pubkey",
        str(pub),
        "--key",
        key,
    )
    assert proc4.returncode == 1
    assert "checkpoint_signature_invalid" in proc4.stdout


def _history_fixture(tmp_path: Path) -> tuple[Path, Path, dict, str]:
    """Two-epoch corpus + lib-emitted history-absence body + pin entry."""
    from quant_fund.research.corpus_epoch import (
        corpus_epoch,
        epoch_heads_key,
        write_epoch_receipt,
    )
    from quant_fund.research.epoch_merkle import (
        history_absence_receipt,
        merkle_root,
    )

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    (corpus / "e.json").write_text(json.dumps({"v": "e"}))
    epoch = write_epoch_receipt(corpus_epoch(corpus), corpus)
    members = {m["name"]: m["sha256"] for m in json.loads(epoch.read_text())["members"]}
    import hashlib

    epoch_sha = hashlib.sha256(epoch.read_bytes()).hexdigest()
    key = epoch_heads_key(corpus, "*.json")
    entry = {"receipt": epoch.name, "sha256": epoch_sha, "tree_root": merkle_root(members)}
    body = history_absence_receipt(corpus, "never.env")
    return corpus, epoch, {"body": body, "entry": entry}, key


def test_script_history_absence_pin(tmp_path: Path) -> None:
    """Library-emitted history receipt verifies under --history-absence."""
    _, epoch, fx, key = _history_fixture(tmp_path)
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({"heads": {key: fx["entry"]}}))
    hp = tmp_path / "ha.json"
    hp.write_text(json.dumps(fx["body"]))
    proc = _run("--history-absence", str(hp), "--pin", str(pin), "--key", key)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "epochs=2" in proc.stdout

    # A stale head bound to epoch 1 fails the pin binding.
    pin_stale = tmp_path / "pin_stale.json"
    first = fx["body"]["epochs"][0]
    pin_stale.write_text(
        json.dumps(
            {
                "heads": {
                    key: {
                        "receipt": first["receipt"],
                        "sha256": first["sha256"],
                        "tree_root": "0" * 64,
                    }
                }
            }
        )
    )
    proc2 = _run("--history-absence", str(hp), "--pin", str(pin_stale), "--key", key)
    assert proc2.returncode == 1
    assert "history_head_not_pinned" in proc2.stdout

    # Forged head digest fails.
    forged = dict(fx["body"], head_sha256="f" * 64)
    fp = tmp_path / "ha_forged.json"
    fp.write_text(json.dumps(forged))
    proc3 = _run("--history-absence", str(fp), "--pin", str(pin), "--key", key)
    assert proc3.returncode == 1
    assert "history_head_digest_drift" in proc3.stdout

    # Corrupted interior root breaks the prev_root link — the pinned head
    # transitively authenticates the whole claimed chain.
    bad = dict(fx["body"])
    bad["epochs"] = [dict(e) for e in fx["body"]["epochs"]]
    bad["epochs"][0]["epoch_root_sha256"] = "0" * 64
    bp = tmp_path / "ha_bad_epoch.json"
    bp.write_text(json.dumps(bad))
    proc4 = _run("--history-absence", str(bp), "--pin", str(pin), "--key", key)
    assert proc4.returncode == 1
    assert "history_link_broken:1" in proc4.stdout

    # A claimed genesis carrying a non-zero prev_root fails the shape check
    # (genesis's prev_epoch_sha256 is the "0"*64 sentinel by convention).
    gen = dict(fx["body"])
    gen["epochs"] = [dict(e) for e in fx["body"]["epochs"]]
    gen["epochs"][0]["prev_root"] = "f" * 64
    gp = tmp_path / "ha_gen.json"
    gp.write_text(json.dumps(gen))
    proc5 = _run("--history-absence", str(gp), "--pin", str(pin), "--key", key)
    assert proc5.returncode == 1
    assert "genesis_prev_root_nonzero" in proc5.stdout


def _signed_checkpoint_v2(heads: dict, pins: dict, tmp_path: Path) -> tuple[Path, Path, Path]:
    """v2 quorum envelope: one real Ed25519 signer registered in a
    gate_quorum.v1 registry; payload binds the registry's canonical digest."""
    import hashlib

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    key = Ed25519PrivateKey.generate()
    pub_hex = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    kid = hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()[:16]
    reg = {"schema": "gate_quorum.v1", "threshold": 1, "keys": [{"key_id": kid, "pubkey": pub_hex}]}
    reg_path = tmp_path / "gate_quorum.json"
    reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    payload = {
        "schema": "integrity_checkpoint.v1",
        "at": "2026-01-01T00:00:00+00:00",
        "pins": pins,
        "heads": heads,
        "missing_pins": [],
        "prev_sha256": None,
        "spine": {"n_archives": 0, "tip": None},
        "witness": {"n_proofs": 0, "proofs": {}},
        "code": {"revision": "test", "worktree_sha256": "0" * 64},
        "quorum": {"registry_sha256": hashlib.sha256(reg_path.read_bytes()).hexdigest()},
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    cp = tmp_path / "checkpoint.json"
    cp.write_text(
        json.dumps(
            {
                "schema": "integrity_checkpoint_sig.v2",
                "algorithm": "ed25519",
                "payload": payload,
                "signatures": [{"key_id": kid, "signature": key.sign(canon).hex()}],
            }
        )
    )
    pub = tmp_path / "gate_signing.pub"
    pub.write_text(pub_hex)
    return cp, pub, reg_path


def test_script_history_absence_checkpoint_quorum(tmp_path: Path) -> None:
    """v2 checkpoint + quorum registry authenticate the head pin; foreign
    pubkeys and swapped registries fail."""
    _, epoch, fx, key = _history_fixture(tmp_path)
    heads = {key: fx["entry"]}
    pin_file = tmp_path / "pin.json"
    pin_file.write_text(json.dumps({"heads": heads}))
    import hashlib

    pins = {"quality/epoch_heads.json": hashlib.sha256(pin_file.read_bytes()).hexdigest()}
    cp, pub, reg = _signed_checkpoint_v2(heads, pins, tmp_path)
    hp = tmp_path / "ha.json"
    hp.write_text(json.dumps(fx["body"]))

    proc = _run(
        "--history-absence",
        str(hp),
        "--checkpoint",
        str(cp),
        "--pubkey",
        str(pub),
        "--quorum",
        str(reg),
        "--key",
        key,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Foreign pubkey: not a registry member.
    (tmp_path / "f").mkdir()
    _, pub2, _ = _signed_checkpoint_v2(heads, pins, tmp_path / "f")
    proc2 = _run(
        "--history-absence",
        str(hp),
        "--checkpoint",
        str(cp),
        "--pubkey",
        str(pub2),
        "--quorum",
        str(reg),
        "--key",
        key,
    )
    assert proc2.returncode == 1
    assert "pubkey_not_in_quorum" in proc2.stdout


def test_script_history_absence_live(tmp_path: Path) -> None:
    """--corpus-dir replays the full chain without any pin; a member that
    was present mid-chain (then removed) must surface via lib emission
    refusal, and a forged interior link must break the walk's compare."""
    corpus, epoch, fx, key = _history_fixture(tmp_path)
    hp = tmp_path / "ha.json"
    hp.write_text(json.dumps(fx["body"]))

    proc = _run("--history-absence", str(hp), "--corpus-dir", str(corpus))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Layered: live corpus + pin agree.
    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({"heads": {key: fx["entry"]}}))
    proc = _run(
        "--history-absence",
        str(hp),
        "--corpus-dir",
        str(corpus),
        "--pin",
        str(pin),
        "--key",
        key,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Interior tamper: break the prev_root link — the shape layer's link
    # check catches it before the corpus walk even runs.
    bad = dict(fx["body"])
    bad["epochs"] = [dict(e) for e in fx["body"]["epochs"]]
    bad["epochs"][1]["prev_root"] = "0" * 64
    bp = tmp_path / "ha_link.json"
    bp.write_text(json.dumps(bad))
    proc = _run("--history-absence", str(bp), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "history_link_broken:1" in proc.stdout

    # Name that IS a member at epoch 1 claims absence — live member scan fires.
    memb = dict(fx["body"], name="a.json")
    mp = tmp_path / "ha_member.json"
    mp.write_text(json.dumps(memb))
    proc = _run("--history-absence", str(mp), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "history_member_present" in proc.stdout

    # Squatter epoch file kills the chain — unverifiable, never partial-ok.
    (corpus / "corpus_epoch_deadbeef00.json").write_text("{not json")
    proc = _run("--history-absence", str(hp), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "history_chain_unverifiable" in proc.stdout


def test_script_live_modes_for_proof_and_absence(tmp_path: Path) -> None:
    """--corpus-dir also replays corpus_proof.v1 and corpus_absence.v1:
    member map + merkle root recomputed from the named epoch receipt."""
    from quant_fund.research.corpus_epoch import corpus_epoch, write_epoch_receipt
    from quant_fund.research.epoch_merkle import absence_receipt, member_proof
    from quant_fund.research.receipt_v2 import seal_receipt

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    write_epoch_receipt(corpus_epoch(corpus), corpus)

    # inclusion: real proof verifies live; forged digest fails member map.
    body = member_proof(corpus, "c.json")
    pp = tmp_path / "p.json"
    pp.write_text(json.dumps(seal_receipt(body)))
    proc = _run("--proof", str(pp), "--corpus-dir", str(corpus))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Forged digest fails at the shape layer — the leaf can't recompute the
    # root (same order as the lib: inclusion replay inside the shape check).
    forged = dict(body, member_sha256="0" * 64)
    fp = tmp_path / "pf.json"
    fp.write_text(json.dumps(forged))
    proc = _run("--proof", str(fp), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "merkle_root_mismatch" in proc.stdout

    # A member-map-level forgery the path replay can't see: same leaf sha,
    # wrong claimed n_members — the epoch's real map is authoritative.
    forged_n = dict(body, n_members=body["n_members"] + 1)
    np_ = tmp_path / "pn.json"
    np_.write_text(json.dumps(forged_n))
    proc = _run("--proof", str(np_), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "n_members_mismatch" in proc.stdout

    # absence: a true absent name verifies; claiming a member fails.
    ab = absence_receipt(corpus, "zz.json")
    ap = tmp_path / "a.json"
    ap.write_text(json.dumps(ab))
    proc = _run("--absence", str(ap), "--corpus-dir", str(corpus))
    assert proc.returncode == 0, proc.stdout + proc.stderr

    lying = dict(ab, name="a.json")
    lp = tmp_path / "af.json"
    lp.write_text(json.dumps(lying))
    proc = _run("--absence", str(lp), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "name_is_member" in proc.stdout


def test_script_consistency_live_and_pin(tmp_path: Path) -> None:
    """epoch_consistency proofs: live hop replay + pin-mode to-head binding."""
    from quant_fund.research.corpus_epoch import corpus_epoch, epoch_heads_key, write_epoch_receipt
    from quant_fund.research.epoch_consistency import consistency_proof
    from quant_fund.research.epoch_merkle import merkle_root

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    (corpus / "a.json").write_text(json.dumps({"v": "a"}))
    e1 = write_epoch_receipt(corpus_epoch(corpus), corpus)
    (corpus / "b.json").write_text(json.dumps({"v": "b"}))
    write_epoch_receipt(corpus_epoch(corpus), corpus)
    (corpus / "c.json").write_text(json.dumps({"v": "c"}))
    e3 = write_epoch_receipt(corpus_epoch(corpus), corpus)
    import hashlib

    members = {m["name"]: m["sha256"] for m in json.loads(e3.read_text())["members"]}
    key = epoch_heads_key(corpus, "*.json")
    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps(
            {
                "heads": {
                    key: {
                        "receipt": e3.name,
                        "sha256": hashlib.sha256(e3.read_bytes()).hexdigest(),
                        "tree_root": merkle_root(members),
                    }
                }
            }
        )
    )
    proof = consistency_proof(corpus, e1.name)
    cp = tmp_path / "cons.json"
    cp.write_text(json.dumps(proof))

    # Live: 3-hop extension verifies.
    proc = _run("--consistency", str(cp), "--corpus-dir", str(corpus))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "hops=3" in proc.stdout

    # Pin mode: the to-head must equal the pinned head.
    proc = _run(
        "--consistency", str(cp), "--pin", str(pin), "--key", key, "--corpus-dir", str(corpus)
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    # Held binding: the exact from-head bytes are trusted.
    held = hashlib.sha256(e1.read_bytes()).hexdigest()
    proc = _run("--consistency", str(cp), "--corpus-dir", str(corpus), "--held", held)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc = _run("--consistency", str(cp), "--corpus-dir", str(corpus), "--held", "b" * 64)
    assert proc.returncode == 1
    assert "held_head_digest_mismatch" in proc.stdout

    # Forged interior hop — proof_sha256 honestly recomputed — still fails:
    # the hop's file digest and the successor's member pin both bind.
    forged = dict(proof)
    forged["hops"] = [dict(h) for h in proof["hops"]]
    forged["hops"][1]["sha256"] = "a" * 64
    forged["proof_sha256"] = hashlib.sha256(
        json.dumps(
            {"hops": forged["hops"], "pattern": forged["pattern"]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    fp = tmp_path / "cons_forge.json"
    fp.write_text(json.dumps(forged))
    proc = _run("--consistency", str(fp), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "hop_digest_mismatch" in proc.stdout
    assert "hop_member_digest_mismatch" in proc.stdout

    # Extension to a mid-chain node is vacuous — a fork could rewrite the
    # suffix: truncate the hop list so `to` is epoch 2.
    trunc = dict(proof)
    trunc["hops"] = proof["hops"][:2]
    trunc["n_hops"] = 2
    trunc["to_receipt"] = dict(proof["hops"][1])
    trunc["proof_sha256"] = hashlib.sha256(
        json.dumps(
            {"hops": trunc["hops"], "pattern": trunc["pattern"]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    tp = tmp_path / "cons_trunc.json"
    tp.write_text(json.dumps(trunc))
    proc = _run("--consistency", str(tp), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "to_not_chain_head" in proc.stdout
    # Pin mode says the same thing differently — `to` isn't the pinned head.
    proc = _run("--consistency", str(tp), "--pin", str(pin), "--key", key)
    assert proc.returncode == 1
    assert "consistency_to_not_pinned" in proc.stdout


def _two_epoch_corpus(tmp_path: Path) -> tuple[Path, str, str]:
    """Stamp genesis + one successor epoch; return (dir, prev, next)."""
    from quant_fund.research.corpus_epoch import corpus_epoch, write_epoch_receipt

    corpus = tmp_path / "receipts"
    corpus.mkdir()
    for name in ("a.json", "c.json", "e.json"):
        (corpus / name).write_text(json.dumps({"v": name}))
    prev = write_epoch_receipt(corpus_epoch(corpus), corpus).name
    (corpus / "g.json").write_text(json.dumps({"v": "g"}))  # added
    (corpus / "a.json").write_text(json.dumps({"v": "a2"}))  # changed
    (corpus / "e.json").unlink()  # removed
    nxt = write_epoch_receipt(corpus_epoch(corpus), corpus).name
    return corpus, prev, nxt


def test_script_delta_live_round_trip(tmp_path: Path) -> None:
    """A library-emitted epoch_delta verifies clean under the script."""
    from quant_fund.research.epoch_delta import epoch_delta_receipt

    corpus, prev, nxt = _two_epoch_corpus(tmp_path)
    body = epoch_delta_receipt(corpus, prev, nxt)
    dp = tmp_path / "delta.json"
    dp.write_text(json.dumps(body))
    proc = _run("--delta", str(dp), "--corpus-dir", str(corpus))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


def test_script_delta_tamper_parity(tmp_path: Path) -> None:
    """Script and library flag the same defect classes on a mutated delta."""
    from quant_fund.research.epoch_delta import epoch_delta_receipt, verify_epoch_delta

    corpus, prev, nxt = _two_epoch_corpus(tmp_path)
    body = epoch_delta_receipt(corpus, prev, nxt)

    # Dropped transition: remove the 'removed' row.
    tampered = json.loads(json.dumps(body))
    tampered["transitions"]["removed"] = []
    lib = verify_epoch_delta(tampered, corpus)
    assert "removed_set_incomplete" in lib and "prev_accounting" in lib
    dp = tmp_path / "delta_tampered.json"
    dp.write_text(json.dumps(tampered))
    proc = _run("--delta", str(dp), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "removed_set_incomplete" in proc.stdout
    assert "prev_accounting" in proc.stdout

    # Forged path: an added row with an empty path can't anchor.
    forged = json.loads(json.dumps(body))
    forged["transitions"]["added"][0]["path"] = []
    lib = verify_epoch_delta(forged, corpus)
    assert any(e.startswith("added_path_invalid") for e in lib)
    fp = tmp_path / "delta_forged.json"
    fp.write_text(json.dumps(forged))
    proc = _run("--delta", str(fp), "--corpus-dir", str(corpus))
    assert proc.returncode == 1
    assert "path" in proc.stdout  # path_depth_mismatch or merkle_root_mismatch
