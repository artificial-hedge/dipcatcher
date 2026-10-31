"""Mutation audit of the evidence-integrity substrate.

Each test builds a small synthetic tree, runs the verifier, then mutates an
input or committed state and asserts the failure is CLOSED — a named error
code or a raised exception, never a silent pass. The cross-layer probes also
pin the documented trust boundaries: an unsigned repo can be laundered by
re-pinning and re-stamping (the Ed25519 pin signature is the sole residual),
and a proof bound to a self-sealed fabricated epoch verifies standalone —
the chain check plus heads pin own that surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research import timestamp_anchor as ta
from quant_fund.research.corpus_epoch import (
    GENESIS_PREV,
    check_epoch_chain,
    corpus_epoch,
    epoch_heads_key,
    load_heads_pin,
    member_digests,
    update_heads_pin,
    write_epoch_receipt,
)
from quant_fund.research.crown_jewels import (
    DEFAULT_JEWELS,
    crown_jewels_errors,
    write_crown_jewels_pin,
)
from quant_fund.research.epoch_merkle import (
    corpus_proof_errors as proof_contract_errors,
)
from quant_fund.research.epoch_merkle import (
    inclusion_proof,
    member_proof,
    merkle_root,
    verify_epoch_proof,
)
from quant_fund.research.gate_signatures import (
    SIGNED_FILES,
    generate_keypair,
    sign_pins,
    verify_pin_signatures,
)
from quant_fund.research.receipt_v2 import seal_receipt, verify_receipt_file
from quant_fund.research.repo_integrity import (
    CORPORA,
    repo_integrity_contract_errors,
    verify_repo,
)
from quant_fund.research.timestamp_anchor import (
    ANCHORS_MANIFEST,
    TIMESTAMPS_DIR,
    build_tsq,
    verify_timestamps,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _member(dirpath: Path, name: str, marker: str) -> Path:
    """A sealed synthetic corpus member (receipt-shaped like the real ones)."""
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


def _stamp(corpus: Path, pattern: str = "*.json") -> Path:
    return write_epoch_receipt(corpus_epoch(corpus, pattern=pattern), corpus)


def _corpus(tmp_path: Path, n: int = 3, name: str = "corpus") -> Path:
    corpus = tmp_path / name
    corpus.mkdir(parents=True)
    for i in range(n):
        _member(corpus, f"m{i}.json", str(i))
    return corpus


def _pinned_corpus(tmp_path: Path, n_epochs: int = 2) -> tuple[Path, Path, dict[str, str]]:
    """Corpus with n_epochs stamped and a committed heads pin at the last."""
    corpus = _corpus(tmp_path)
    pin = tmp_path / "epoch_heads.json"
    last = _stamp(corpus)
    for i in range(n_epochs - 1):
        _member(corpus, f"extra{i}.json", str(i))
        last = _stamp(corpus)
    update_heads_pin(pin, corpus, "*.json", last)
    expected = load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]
    return corpus, pin, expected


def _fake_tsr(imprint: bytes) -> bytes:
    """Minimal DER TimeStampResp carrying ``imprint`` — parser-shaped, unsigned."""
    alg_id = ta._seq(ta.SHA256_ALG_OID + ta._tlv(0x05, b""))
    msg_imprint = ta._seq(alg_id + ta._tlv(0x04, imprint))
    tst_info = ta._seq(ta._tlv(0x02, b"\x01") + ta._tlv(0x06, b"\x2b\x06") + msg_imprint)
    encap = ta._seq(ta._tlv(0x06, b"\x2b\x06") + ta._tlv(0xA0, ta._tlv(0x04, tst_info)))
    signed_data = ta._seq(ta._tlv(0x02, b"\x01") + ta._tlv(0x31, b"") + encap)
    content_info = ta._seq(ta._tlv(0x06, b"\x2b\x06") + ta._tlv(0xA0, signed_data))
    status = ta._seq(ta._tlv(0x02, b"\x00"))
    return ta._seq(status + content_info)


def _anchored_repo(tmp_path: Path, target_bytes: bytes = b"pinned state bytes") -> tuple[Path, str]:
    """Repo root with one registered anchor (fake token) over a target file."""
    target = tmp_path / "quality" / "epoch_heads.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(target_bytes)
    digest = hash_bytes(target_bytes)
    ts_dir = tmp_path / TIMESTAMPS_DIR
    ts_dir.mkdir(parents=True)
    (ts_dir / "x.tsr").write_bytes(_fake_tsr(bytes.fromhex(digest)))
    (ts_dir / ANCHORS_MANIFEST).write_text(
        json.dumps(
            {
                "schema": "timestamp_anchors.v1",
                "anchors": {"x.tsr": {"target": "quality/epoch_heads.json", "sha256": digest}},
            }
        )
    )
    return tmp_path, digest


def _signed_repo(tmp_path: Path) -> Path:
    """Repo root carrying the two pin files and a real Ed25519 signature."""
    q = tmp_path / "quality"
    q.mkdir(parents=True)
    for rel in SIGNED_FILES:
        (tmp_path / rel).write_text(f'{{"pin": "{rel}"}}\n')
    priv, pub = generate_keypair()
    sign_pins(tmp_path, priv, pub)
    return tmp_path


def _mini_repo(tmp_path: Path, *, sign: bool = True) -> tuple[Path, str | None]:
    """Minimal tree satisfying every verify_repo gate, optionally signed."""
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    for jewel in DEFAULT_JEWELS:
        p = root / jewel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {jewel}\n")
    pin = root / "quality" / "epoch_heads.json"
    pin.write_text(json.dumps({"schema": "epoch_heads.v1", "heads": {}}))
    for corpus_dir, pattern, _req, _upd, _exempt in CORPORA:
        d = root / corpus_dir
        d.mkdir(parents=True, exist_ok=True)
        seed = f"seed.{pattern[2:]}" if len(pattern) > 1 else "seed"
        # JSON corpora feed the lattice gate — seeds must parse.
        (d / seed).write_text('{"seed": true}' if seed.endswith(".json") else "seed")
        ep = write_epoch_receipt(corpus_epoch(d, pattern=pattern), d)
        update_heads_pin(pin, corpus_dir, pattern, ep)
    write_crown_jewels_pin(root)
    # Re-stamp quality so the pin + jewels file land as members.
    ep = write_epoch_receipt(corpus_epoch(root / "quality"), root / "quality")
    update_heads_pin(pin, "quality", "*.json", ep)
    priv = None
    if sign:
        priv, pub = generate_keypair()
        sign_pins(root, priv, pub)
        write_crown_jewels_pin(root)  # pubkey is a jewel — repin after signing
        ep = write_epoch_receipt(corpus_epoch(root / "quality"), root / "quality")
        update_heads_pin(pin, "quality", "*.json", ep)
        sign_pins(root, priv, pub)  # re-sign over final pin bytes
    return root, priv


# --------------------------------------------------------------------------
# corpus_epoch / check_epoch_chain
# --------------------------------------------------------------------------


def test_member_bytes_mutated_post_stamp_fails(tmp_path: Path) -> None:
    corpus, _pin, expected = _pinned_corpus(tmp_path)
    (corpus / "m0.json").write_text('{"tampered": true}')
    errors = check_epoch_chain(corpus, expected_head=expected)["errors"]
    assert "head_member_digest_drift:'m0.json'" in errors


def test_member_deleted_post_stamp_fails(tmp_path: Path) -> None:
    corpus, _pin, expected = _pinned_corpus(tmp_path)
    (corpus / "m1.json").unlink()
    errors = check_epoch_chain(corpus, expected_head=expected)["errors"]
    assert "head_member_missing_live:'m1.json'" in errors


def test_epoch_receipt_member_map_tampered_fails(tmp_path: Path) -> None:
    """Rewriting a stamped epoch's member map breaks its seal."""
    corpus, _pin, expected = _pinned_corpus(tmp_path)
    head = corpus / expected["receipt"]
    doc = json.loads(head.read_text())
    doc["members"][0]["sha256"] = "ff" * 32
    head.write_text(json.dumps(doc, indent=2, sort_keys=True))
    errors = check_epoch_chain(corpus, expected_head=expected)["errors"]
    assert any(e.startswith("epoch_receipt_invalid:") for e in errors)
    assert any(e.startswith("epoch_head_mutated:") for e in errors)


def test_forged_unsigned_epoch_receipt_flagged(tmp_path: Path) -> None:
    """A corpus_epoch_*.json file claiming the schema without a valid seal is
    an invalid epoch, never a silent unstamped member."""
    corpus, _pin, expected = _pinned_corpus(tmp_path)
    fake = {
        "kind": "corpus_epoch.v1",
        "schema": "corpus_epoch.v1",
        "epoch_root_sha256": "aa" * 32,
        "members": [{"name": "ghost.json", "sha256": "ff" * 32}],
        "n_members": 1,
        "prev_epoch_receipt": None,
        "prev_epoch_sha256": GENESIS_PREV,
    }
    (corpus / "corpus_epoch_deadbeefdeadbeef.json").write_text(json.dumps(fake))
    errors = check_epoch_chain(corpus, expected_head=expected)["errors"]
    assert any(
        e.startswith("epoch_receipt_invalid:") and "corpus_epoch_deadbeef" in e for e in errors
    )


def test_rolled_back_heads_pin_fails(tmp_path: Path) -> None:
    """Pinned head deleted: rewind is epoch_head_missing + epoch_head_rollback,
    and unpinned the same rewind is silent — the pin is what closes it."""
    corpus, _pin, expected = _pinned_corpus(tmp_path, n_epochs=2)
    head_name = expected["receipt"]
    assert check_epoch_chain(corpus, expected_head=expected)["errors"] == []
    (corpus / head_name).unlink()
    errors = check_epoch_chain(corpus, expected_head=expected)["errors"]
    assert f"epoch_head_missing:{head_name!a}" in errors
    assert f"epoch_head_rollback:{head_name!a}" in errors
    unpinned = check_epoch_chain(corpus)
    assert unpinned["errors"] == []  # documented: pin is the rollback detector


def test_stale_pin_accepts_descendant_heads(tmp_path: Path) -> None:
    """A pin naming an older head is legal — the observed head is a
    descendant; pin staleness is a pending-stamp state, not tamper."""
    corpus = _corpus(tmp_path)
    first = _stamp(corpus)
    exp = {"receipt": first.name, "sha256": hash_bytes(first.read_bytes())}
    _member(corpus, "late.json", "x")
    second = _stamp(corpus)
    res = check_epoch_chain(corpus, expected_head=exp)
    assert res["errors"] == [] and res["head"] == second.name


def test_heads_pin_mutated_fails(tmp_path: Path) -> None:
    corpus, _pin, _expected = _pinned_corpus(tmp_path)
    exp = {"receipt": _expected["receipt"], "sha256": "0" * 64}
    errors = check_epoch_chain(corpus, expected_head=exp)["errors"]
    assert any(e.startswith("epoch_head_mutated:") for e in errors)


def test_unstamped_arrival_closes_only_under_require_stamped(tmp_path: Path) -> None:
    corpus, _pin, expected = _pinned_corpus(tmp_path)
    _member(corpus, "rogue.json", "dropped post-stamp")
    res = check_epoch_chain(corpus, expected_head=expected)
    assert res["errors"] == [] and "rogue.json" in res["unstamped"]
    strict = check_epoch_chain(corpus, expected_head=expected, require_stamped=True)
    assert "unstamped_member:'rogue.json'" in strict["errors"]


def test_params_malformed_epoch_flagged_not_silent(tmp_path: Path) -> None:
    """An epoch-claiming file with non-mapping params is flagged, and the
    stamping path must not crash on it."""
    corpus, _pin, expected = _pinned_corpus(tmp_path)
    (corpus / "corpus_epoch_evil00evil00.json").write_text(
        json.dumps({"kind": "corpus_epoch.v1", "schema": "corpus_epoch.v1", "params": "x"})
    )
    corpus_epoch(corpus)  # must not raise
    errors = check_epoch_chain(corpus, expected_head=expected)["errors"]
    assert "epoch_params_malformed:corpus_epoch_evil00evil00.json" in errors


def test_basename_cannot_hide_a_real_member(tmp_path: Path) -> None:
    """The anchors.json exemption is rel-path scoped — a corpus member named
    ``anchors.json`` stays stamped, and deleting it fails closed. A genuine
    quality receipt renamed ``anchors.json`` cannot hide behind the exempt
    name: only ``timestamps/anchors.json`` drops out."""
    corpus = _corpus(tmp_path)
    (corpus / "timestamps").mkdir()
    (corpus / "anchors.json").write_text('{"receipt": "renamed to hide"}')
    (corpus / "timestamps" / "anchors.json").write_text('{"schema": "timestamp_anchors.v1"}')
    (corpus / "epoch_heads.json").write_text('{"schema": "epoch_heads.v1", "heads": {}}')
    members = member_digests(corpus)
    assert "anchors.json" in members  # basename alone does NOT exempt
    assert "timestamps/anchors.json" not in members  # exact relpath exempt
    assert "epoch_heads.json" not in members  # reserved bookkeeping basename
    _stamp(corpus)
    (corpus / "anchors.json").unlink()
    errors = check_epoch_chain(corpus)["errors"]
    assert "head_member_missing_live:'anchors.json'" in errors
    # Restore the member; the exempt manifest can then change freely without
    # staling the chain.
    (corpus / "anchors.json").write_text('{"receipt": "renamed to hide"}')
    (corpus / "timestamps" / "anchors.json").write_text('{"rewritten": true}')
    assert check_epoch_chain(corpus)["errors"] == []


# --------------------------------------------------------------------------
# crown_jewels
# --------------------------------------------------------------------------


def test_jewel_mutated_deleted_symlinked_all_fail(tmp_path: Path) -> None:
    root, _priv = _mini_repo(tmp_path, sign=False)
    (root / "Makefile").write_text("# weakened gates\n")
    errors = crown_jewels_errors(root)
    assert "jewel_mutated:Makefile" in errors
    (root / "Makefile").unlink()
    errors = crown_jewels_errors(root)
    assert "jewel_missing:Makefile" in errors
    (root / "evil.txt").write_text("attacker content")
    (root / "Makefile").symlink_to("evil.txt")
    errors = crown_jewels_errors(root)
    assert "jewel_symlink:Makefile" in errors


def test_jewel_pin_set_shrink_and_grow_fail(tmp_path: Path) -> None:
    root, _priv = _mini_repo(tmp_path, sign=False)
    pin_path = root / "quality" / "crown_jewels.json"
    pin = json.loads(pin_path.read_text())
    dropped = pin["files"].pop("Makefile")
    pin_path.write_text(json.dumps(pin, indent=2, sort_keys=True))
    assert "jewel_unpinned:Makefile" in crown_jewels_errors(root)
    pin["files"][".hidden_config"] = dropped
    pin_path.write_text(json.dumps(pin, indent=2, sort_keys=True))
    assert "jewel_unexpected:.hidden_config" in crown_jewels_errors(root)


def test_jewel_pin_malformed_and_missing_fail(tmp_path: Path) -> None:
    root, _priv = _mini_repo(tmp_path, sign=False)
    pin_path = root / "quality" / "crown_jewels.json"
    pin_path.write_text("{corrupt json")
    assert crown_jewels_errors(root)[0].startswith("pin_malformed:")
    pin_path.unlink()
    assert crown_jewels_errors(root) == ["pin_missing"]


# --------------------------------------------------------------------------
# gate_signatures — the layer that catches re-pinning
# --------------------------------------------------------------------------


def test_repinned_unsigned_drift_is_what_signature_exists_for(tmp_path: Path) -> None:
    """Mutate a jewel AND re-pin: the jewel gate goes clean; only the
    signature sees the lie (pin_drift). In an unsigned tree the same move is
    silent — the documented trust boundary."""
    root = _signed_repo(tmp_path)
    (root / "quality" / "crown_jewels.json").write_text('{"forged": true}\n')
    res = verify_pin_signatures(root)
    assert res["ok"] is False
    assert "pin_drift:quality/crown_jewels.json" in res["errors"]

    unsigned = tmp_path / "unsigned"
    q = unsigned / "quality"
    q.mkdir(parents=True)
    for rel in SIGNED_FILES:
        (unsigned / rel).write_text(f'{{"pin": "{rel}"}}\n')
    res2 = verify_pin_signatures(unsigned)
    assert res2 == {"ok": True, "signed": False, "errors": []}
    (unsigned / "quality" / "crown_jewels.json").write_text('{"forged": true}\n')
    assert verify_pin_signatures(unsigned)["signed"] is False  # nothing to check against


def test_forged_signature_bytes_fail(tmp_path: Path) -> None:
    root = _signed_repo(tmp_path)
    sig = json.loads((root / "gate_pins.sig").read_text())
    sig["signature"] = "00" + sig["signature"][2:]
    (root / "gate_pins.sig").write_text(json.dumps(sig, indent=2, sort_keys=True))
    res = verify_pin_signatures(root)
    assert res["ok"] is False and "signature_invalid" in res["errors"]


def test_sig_payload_mutations_fail(tmp_path: Path) -> None:
    root = _signed_repo(tmp_path)
    sig = json.loads((root / "gate_pins.sig").read_text())
    sig["payload"]["files"].pop("quality/epoch_heads.json")
    (root / "gate_pins.sig").write_text(json.dumps(sig, indent=2, sort_keys=True))
    res = verify_pin_signatures(root)
    assert res["ok"] is False and "signed_file_set_drift" in res["errors"]


def test_attacker_keypair_resign_boundary(tmp_path: Path) -> None:
    """Re-pin AND re-sign with the attacker's own key: the signature gate
    passes standalone — the pubkey *is* the trust root, so swapping it swaps
    what "valid" means. What actually catches it: gate_signing.pub is a
    crown jewel, so the swap trips jewel_mutated, and the new pubkey would
    have to ride in the same commit under git review."""
    root, _priv = _mini_repo(tmp_path, sign=True)
    evil_priv, evil_pub = generate_keypair()
    sign_pins(root, evil_priv, evil_pub)  # attacker resigns + swaps pubkey
    res = verify_pin_signatures(root)
    assert res["ok"] is True  # sig gate alone cannot see a full key swap
    assert "jewel_mutated:quality/gate_signing.pub" in crown_jewels_errors(root)


def test_sig_file_deleted_or_malformed_fails(tmp_path: Path) -> None:
    root = _signed_repo(tmp_path)
    (root / "gate_pins.sig").unlink()
    assert verify_pin_signatures(root)["errors"] == ["signature_file_missing"]
    root2 = _signed_repo(tmp_path / "r2")
    (root2 / "gate_pins.sig").write_text("{corrupt")
    assert verify_pin_signatures(root2)["errors"] == ["signature_file_malformed"]


def test_pubkey_malformed_fails(tmp_path: Path) -> None:
    root = _signed_repo(tmp_path)
    (root / "quality" / "gate_signing.pub").write_text("not-hex-pubkey\n")
    assert verify_pin_signatures(root)["errors"] == ["pubkey_malformed"]


def test_signed_file_deleted_fails(tmp_path: Path) -> None:
    root = _signed_repo(tmp_path)
    (root / "quality" / "epoch_heads.json").unlink()
    res = verify_pin_signatures(root)
    assert "pinned_file_missing:quality/epoch_heads.json" in res["errors"]


# --------------------------------------------------------------------------
# timestamp_anchor
# --------------------------------------------------------------------------


def test_anchor_imprint_mismatch_fails(tmp_path: Path) -> None:
    root, _digest = _anchored_repo(tmp_path)
    manifest = json.loads((root / TIMESTAMPS_DIR / ANCHORS_MANIFEST).read_text())
    manifest["anchors"]["x.tsr"]["sha256"] = "ff" * 32
    (root / TIMESTAMPS_DIR / ANCHORS_MANIFEST).write_text(json.dumps(manifest))
    res = verify_timestamps(root)
    assert res["ok"] is False
    assert res["errors"] == ["imprint_mismatch:quality/epoch_heads.json"]


def test_anchor_token_deleted_fails(tmp_path: Path) -> None:
    root, _digest = _anchored_repo(tmp_path)
    (root / TIMESTAMPS_DIR / "x.tsr").unlink()
    assert verify_timestamps(root)["errors"] == ["tsr_missing:x.tsr"]


def test_anchor_token_corrupted_fails(tmp_path: Path) -> None:
    root, _digest = _anchored_repo(tmp_path)
    (root / TIMESTAMPS_DIR / "x.tsr").write_bytes(b"not a der token")
    assert verify_timestamps(root)["errors"] == ["tsr_malformed:x.tsr"]


def test_anchor_manifest_target_mutations_fail(tmp_path: Path) -> None:
    root, digest = _anchored_repo(tmp_path)
    manifest_path = root / TIMESTAMPS_DIR / ANCHORS_MANIFEST
    manifest = json.loads(manifest_path.read_text())
    # Target repointed at a deleted file — the anchor loses its referent.
    manifest["anchors"]["x.tsr"]["target"] = "quality/gone.json"
    manifest_path.write_text(json.dumps(manifest))
    assert verify_timestamps(root)["errors"] == ["anchor_target_missing:quality/gone.json"]
    # Target repointed at a different live file: imprint binds, freshness
    # drops to False — a valid proof of an earlier state, not tamper.
    other = root / "quality" / "other.json"
    other.write_bytes(b"other bytes")
    manifest["anchors"]["x.tsr"]["target"] = "quality/other.json"
    manifest_path.write_text(json.dumps(manifest))
    res = verify_timestamps(root)
    assert res["ok"] is True and res["fresh"] == {"quality/other.json": False}


def test_anchor_target_bytes_drift_is_stale_not_forged(tmp_path: Path) -> None:
    """Post-stamp pin movement: fresh=False, ok=True — the anchor honestly
    proves the earlier committed bytes."""
    root, _digest = _anchored_repo(tmp_path)
    (root / "quality" / "epoch_heads.json").write_bytes(b"moved on")
    res = verify_timestamps(root)
    assert res["ok"] is True
    assert res["fresh"]["quality/epoch_heads.json"] is False


def test_anchor_manifest_malformed_variants_fail(tmp_path: Path) -> None:
    bodies = (
        "{corrupt",  # unparseable
        '["not", "a", "dict"]',  # non-object manifest — must not crash
        '{"schema": "wrong.v1", "anchors": {}}',  # wrong schema
        '{"schema": "timestamp_anchors.v1", "anchors": []}',  # anchors not a map
    )
    for i, body in enumerate(bodies):
        root, _ = _anchored_repo(tmp_path / f"case{i}")
        (root / TIMESTAMPS_DIR / ANCHORS_MANIFEST).write_text(body)
        res = verify_timestamps(root)
        assert res["ok"] is False
        assert res["errors"] == ["anchors_manifest_malformed"], body


def test_anchor_entry_malformed_fails_closed(tmp_path: Path) -> None:
    """A manifest whose anchor entry isn't an object must degrade to an
    error code, not crash the gate."""
    root, _ = _anchored_repo(tmp_path)
    (root / TIMESTAMPS_DIR / ANCHORS_MANIFEST).write_text(
        json.dumps({"schema": "timestamp_anchors.v1", "anchors": {"x.tsr": "garbage"}})
    )
    res = verify_timestamps(root)
    assert res["ok"] is False
    assert res["errors"] == ["anchor_malformed:x.tsr"]


def _mint_real_anchor(tmp_path: Path) -> Path:
    """Local CA + TSA + real openssl-minted .tsr over a pinned target."""
    import subprocess

    root = tmp_path / "tsarepo"
    target = root / "quality" / "epoch_heads.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"pinned state bytes")
    ts_dir = root / TIMESTAMPS_DIR
    ts_dir.mkdir(parents=True)
    ca_key, ca_crt = tmp_path / "ca.key", tmp_path / "ca.crt"
    tsa_key, tsa_csr, tsa_crt = tmp_path / "tsa.key", tmp_path / "tsa.csr", tmp_path / "tsa.crt"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(ca_key),
            "-out",
            str(ca_crt),
            "-subj",
            "/CN=audit ca",
            "-days",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(tsa_key),
            "-out",
            str(tsa_csr),
            "-subj",
            "/CN=audit tsa",
        ],
        check=True,
        capture_output=True,
    )
    ext = tmp_path / "tsa.ext"
    ext.write_text("extendedKeyUsage=critical,timeStamping\nkeyUsage=critical,digitalSignature\n")
    subprocess.run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(tsa_csr),
            "-CA",
            str(ca_crt),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-out",
            str(tsa_crt),
            "-days",
            "1",
            "-extfile",
            str(ext),
        ],
        check=True,
        capture_output=True,
    )
    digest = hash_bytes(target.read_bytes())
    query = tmp_path / "q.tsq"
    query.write_bytes(build_tsq(digest))
    token = tmp_path / "token.tsr"
    subprocess.run(
        [
            "openssl",
            "ts",
            "-reply",
            "-queryfile",
            str(query),
            "-signer",
            str(tsa_crt),
            "-inkey",
            str(tsa_key),
            "-chain",
            str(ca_crt),
            "-tspolicy",
            "1.2.3.4",
            "-out",
            str(token),
        ],
        check=True,
        capture_output=True,
    )
    (ts_dir / "x.tsr").write_bytes(token.read_bytes())
    (ts_dir / ta.CACERT_NAME).write_bytes(ca_crt.read_bytes())
    (ts_dir / ta.TSA_CERT_NAME).write_bytes(tsa_crt.read_bytes())
    (ts_dir / ANCHORS_MANIFEST).write_text(
        json.dumps(
            {
                "schema": "timestamp_anchors.v1",
                "anchors": {"x.tsr": {"target": "quality/epoch_heads.json", "sha256": digest}},
            }
        )
    )
    return root


def test_swapped_tsa_cacert_fails_chain(tmp_path: Path) -> None:
    """With openssl + committed certs, a CA swap turns a valid token into
    chain_verify_failed — the token can't launder under a foreign root."""
    if not ta._openssl_available():
        pytest.skip("openssl not on PATH")
    root = _mint_real_anchor(tmp_path)
    res = verify_timestamps(root)
    assert res["ok"] is True, res
    assert res["chain"]["quality/epoch_heads.json"] == "chain_verified"
    import subprocess

    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(tmp_path / "evil.key"),
            "-out",
            str(tmp_path / "evil.crt"),
            "-subj",
            "/CN=evil ca",
            "-days",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    (root / TIMESTAMPS_DIR / ta.CACERT_NAME).write_bytes((tmp_path / "evil.crt").read_bytes())
    res2 = verify_timestamps(root)
    assert res2["ok"] is False
    assert res2["errors"] == ["chain_verify_failed:quality/epoch_heads.json"]


def test_deleted_tsa_cacert_degrades_but_jewel_gate_catches(tmp_path: Path) -> None:
    """Deleting the committed CA makes chain_unchecked (honest degrade, not
    silent verify) — the cert files are crown jewels, so the swap/delete is
    caught one layer up."""
    root = _mint_real_anchor(tmp_path)
    (root / TIMESTAMPS_DIR / ta.CACERT_NAME).unlink()
    res = verify_timestamps(root)
    assert res["ok"] is True  # chain_unchecked is reported, not an error
    assert res["chain"]["quality/epoch_heads.json"] == "chain_unchecked"


def test_anchors_manifest_never_stamps_into_quality_chain(tmp_path: Path) -> None:
    """timestamps/anchors.json is chain-exempt; a member sharing only the
    basename is not."""
    corpus = tmp_path / "quality"
    (corpus / "timestamps").mkdir(parents=True)
    _member(corpus, "gate.json", "g")
    (corpus / "timestamps" / "anchors.json").write_text("{}")
    _stamp(corpus)
    (corpus / "timestamps" / "anchors.json").write_text('{"anchors": {"new": {}}}')
    assert check_epoch_chain(corpus)["errors"] == []


# --------------------------------------------------------------------------
# epoch_merkle — corpus_proof.v1
# --------------------------------------------------------------------------


def test_merkle_path_side_flip_fails(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 5, name="receipts")
    _stamp(corpus)
    body = member_proof(corpus, "m2.json")
    forged = dict(body, path=[dict(e) for e in body["path"]])
    forged["path"][0]["side"] = "left" if forged["path"][0]["side"] == "right" else "right"
    assert "proof_path_invalid" in proof_contract_errors(forged)
    assert verify_epoch_proof(forged, corpus) == ["proof_path_invalid"]


def test_merkle_path_sibling_digest_flip_fails(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 5, name="receipts")
    _stamp(corpus)
    body = member_proof(corpus, "m2.json")
    forged = dict(body, path=[dict(e) for e in body["path"]])
    forged["path"][0]["sha256"] = "00" * 32
    assert "proof_path_invalid" in proof_contract_errors(forged)


def test_merkle_path_truncated_and_extended_fail(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 6, name="receipts")
    _stamp(corpus)
    body = member_proof(corpus, "m4.json")
    short = dict(body, path=body["path"][:-1])
    assert "proof_path_invalid" in proof_contract_errors(short)
    long = dict(body, path=[*body["path"], {"sha256": "11" * 32, "side": "left"}])
    assert "proof_path_invalid" in proof_contract_errors(long)


def test_proof_bound_to_deleted_epoch_fails(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 3, name="receipts")
    _stamp(corpus)
    body = member_proof(corpus, "m0.json")
    (corpus / body["epoch_receipt"]).unlink()
    assert verify_epoch_proof(body, corpus) == ["epoch_receipt_missing"]


def test_proof_bound_to_tampered_epoch_fails(tmp_path: Path) -> None:
    """Editing the referenced epoch's member map breaks its seal — the proof
    verifier must check the receipt verifies, not just that fields agree."""
    corpus = _corpus(tmp_path, 3, name="receipts")
    _stamp(corpus)
    body = member_proof(corpus, "m0.json")
    ep = corpus / body["epoch_receipt"]
    doc = json.loads(ep.read_text())
    doc["members"][0]["sha256"] = "ee" * 32
    ep.write_text(json.dumps(doc, indent=2, sort_keys=True))
    # Seal no longer recomputes over the tampered body — the authenticated
    # loader refuses before the root/member comparison is reached.
    assert verify_epoch_proof(body, corpus) == ["epoch_receipt_tampered"]


def test_proof_bound_to_fabricated_unsigned_epoch_fails(tmp_path: Path) -> None:
    """An attacker-written epoch receipt (no valid seal) must not anchor a
    proof — even when the member map agrees with the forged path."""
    corpus = _corpus(tmp_path, 3, name="receipts")
    _stamp(corpus)
    members = {"ghost.json": "ff" * 32, "m0.json": "00" * 32}
    fake_epoch = {
        "kind": "corpus_epoch.v1",
        "schema": "corpus_epoch.v1",
        "epoch_root_sha256": "ab" * 32,
        "members": [{"name": n, "sha256": s} for n, s in sorted(members.items())],
        "prev_epoch_receipt": None,
    }
    (corpus / "corpus_epoch_fake0fake0fak.json").write_text(json.dumps(fake_epoch))
    proof = inclusion_proof(members, "ghost.json")
    forged = {
        "kind": "corpus_proof.v1",
        "schema": "corpus_proof.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "member": "ghost.json",
        "member_sha256": "ff" * 32,
        "epoch_receipt": "corpus_epoch_fake0fake0fak.json",
        "epoch_root_sha256": "ab" * 32,
        "merkle_root": merkle_root(members),
        "leaf_index": proof["leaf_index"],
        "n_members": proof["n_members"],
        "path": proof["path"],
    }
    # No receipt_sha256 at all — the authenticated loader fails closed.
    assert verify_epoch_proof(forged, corpus) == ["epoch_receipt_unsealed"]


def test_proof_bound_to_selfsealed_fabrication_is_chain_owned(tmp_path: Path) -> None:
    """A *fully self-consistent* fabricated epoch (valid self-seal, clean
    contract) still verifies a forged proof standalone — proofs check
    artifact consistency, not provenance. The chain check is the arbiter:
    the fabricated epoch shows up as a second genesis / pin violation."""
    corpus = _corpus(tmp_path, 3, name="receipts")
    real = _stamp(corpus)
    pin = tmp_path / "heads.json"
    update_heads_pin(pin, corpus, "*.json", real)
    expected = load_heads_pin(pin)[epoch_heads_key(corpus, "*.json")]
    members = {"ghost.json": "ff" * 32, "m0.json": "00" * 32}
    epoch_body = {
        "kind": "corpus_epoch.v1",
        "schema": "corpus_epoch.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "inputs_sha256": hash_bytes(canonical_json_bytes({"corpus_dir": "fake"})),
        "params": {"pattern": "*.json"},
        "epoch_root_sha256": hash_bytes(canonical_json_bytes(dict(sorted(members.items())))),
        "members": [{"name": n, "sha256": s} for n, s in sorted(members.items())],
        "n_members": 2,
        "prev_epoch_sha256": GENESIS_PREV,
        "prev_epoch_receipt": None,
        "members_added": sorted(members),
        "members_removed": [],
        "verdict": "genesis",
    }
    canonical = json.loads(canonical_json_bytes(epoch_body))
    canonical["receipt_sha256"] = hash_bytes(canonical_json_bytes(canonical))
    # The name<->seal binding holds: the forgery must occupy the filename its
    # own seal derives (corpus_epoch_<sha16>.json) — any other name fails
    # epoch_receipt_name_mismatch at proof-verify.
    forged_name = f"corpus_epoch_{canonical['receipt_sha256'][:16]}.json"
    (corpus / forged_name).write_text(json.dumps(canonical, indent=2, sort_keys=True))
    # The fabricated epoch is contract-clean and self-sealed: verify holds.
    assert verify_receipt_file(corpus / forged_name)["valid"]
    proof = inclusion_proof(members, "ghost.json")
    forged = {
        "kind": "corpus_proof.v1",
        "schema": "corpus_proof.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "member": "ghost.json",
        "member_sha256": "ff" * 32,
        "epoch_receipt": forged_name,
        "epoch_root_sha256": epoch_body["epoch_root_sha256"],
        "merkle_root": merkle_root(members),
        "leaf_index": proof["leaf_index"],
        "n_members": proof["n_members"],
        "path": proof["path"],
    }
    assert verify_epoch_proof(forged, corpus) == []  # boundary: standalone consistency only
    errors = check_epoch_chain(corpus, expected_head=expected)["errors"]
    assert any(e.startswith("epoch_multiple_genesis:") for e in errors)


def test_proof_bound_to_v2_wrapped_epoch_verifies(tmp_path: Path) -> None:
    """Epochs stamped as receipt.v2 envelopes still anchor proofs — the
    verifier unwraps the envelope before field comparison."""
    corpus = _corpus(tmp_path, 4, name="receipts")
    write_epoch_receipt(corpus_epoch(corpus), corpus, receipt_version=2)
    body = member_proof(corpus, "m1.json")
    assert verify_epoch_proof(body, corpus) == []


def test_proof_member_digest_swap_fails(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, 4, name="receipts")
    _stamp(corpus)
    body = member_proof(corpus, "m1.json")
    claimed = dict(body, member="m0.json")  # path/digest still for m1
    assert verify_epoch_proof(claimed, corpus) == ["proof_path_invalid"]


# --------------------------------------------------------------------------
# repo_integrity — the capstone must never crash on corrupted inputs
# --------------------------------------------------------------------------


def test_verify_repo_all_green_then_each_layer_fails(tmp_path: Path) -> None:
    root, _priv = _mini_repo(tmp_path, sign=True)
    res = verify_repo(root)
    assert res["ok"], res
    # Jewel mutation: crown gate fails, other gates stay green.
    (root / "pyproject.toml").write_text("tampered")
    res = verify_repo(root)
    assert not res["ok"]
    assert res["gates"]["crown_jewels"]["errors"] == ["jewel_mutated:pyproject.toml"]
    assert res["gates"]["epoch:receipts"]["ok"]


def test_verify_repo_malformed_heads_pin_fails_not_crashes(tmp_path: Path) -> None:
    """A corrupt epoch_heads.json must degrade to per-corpus gate errors —
    the capstone returning a verdict is the whole point of the design."""
    root, _priv = _mini_repo(tmp_path, sign=False)
    (root / "quality" / "epoch_heads.json").write_text("{not json")
    res = verify_repo(root)  # must not raise
    assert not res["ok"]
    for name, gate in res["gates"].items():
        if name.startswith("epoch:"):
            assert any(e.startswith("heads_pin_malformed:") for e in gate["errors"]), name
    assert res["gates"]["crown_jewels"]["ok"]


def test_verify_repo_rolled_back_chain_fails(tmp_path: Path) -> None:
    root, priv = _mini_repo(tmp_path, sign=True)
    assert priv is not None
    pin = root / "quality" / "epoch_heads.json"
    # Second stamp on receipts so a head exists to delete. The pin file is a
    # signed file, so re-sign after it legitimately advances.
    _member(root / "receipts", "late.json", "x")
    ep = write_epoch_receipt(corpus_epoch(root / "receipts"), root / "receipts")
    update_heads_pin(pin, "receipts", "*.json", ep)
    sign_pins(root, priv, (root / "quality" / "gate_signing.pub").read_text().strip())
    assert verify_repo(root)["ok"]
    ep.unlink()
    res = verify_repo(root)
    errs = res["gates"]["epoch:receipts"]["errors"]
    assert any(e.startswith("epoch_head_missing:") for e in errs)
    assert any(e.startswith("epoch_head_rollback:") for e in errs)


def test_full_unsigned_launder_is_the_documented_boundary(tmp_path: Path) -> None:
    """Mutate a jewel → re-pin → re-stamp quality → update heads pin: in an
    UNSIGNED tree every content gate goes green again — but the pubkey is a
    crown jewel, so it is always present, and its presence makes the sig gate
    insist on gate_pins.sig. The attacker cannot mint one, so the residual
    error is exactly `signature_file_missing`; git review of the pin/stamp
    commit plus the unreachable sig is the line left. Signed, the same
    sequence dies at pin_drift regardless of restamping."""
    # unsigned: content gates launder; sig gate cannot be cleared
    root, _ = _mini_repo(tmp_path, sign=False)
    base = verify_repo(root)
    assert not base["ok"]
    assert base["gates"]["pin_signatures"]["errors"] == ["signature_file_missing"]
    (root / "Makefile").write_text("# weakened\n")
    write_crown_jewels_pin(root)  # re-pin over the mutated jewel
    pin = root / "quality" / "epoch_heads.json"
    ep = write_epoch_receipt(corpus_epoch(root / "quality"), root / "quality")
    update_heads_pin(pin, "quality", "*.json", ep)
    res = verify_repo(root)
    assert res["gates"]["pin_signatures"]["errors"] == ["signature_file_missing"]
    for name, gate in res["gates"].items():
        if name != "pin_signatures":
            assert gate["ok"], (name, gate)

    # signed: pin_drift survives any restamp
    root2, _ = _mini_repo(tmp_path / "signed", sign=True)
    (root2 / "Makefile").write_text("# weakened\n")
    write_crown_jewels_pin(root2)
    pin2 = root2 / "quality" / "epoch_heads.json"
    ep2 = write_epoch_receipt(corpus_epoch(root2 / "quality"), root2 / "quality")
    update_heads_pin(pin2, "quality", "*.json", ep2)
    res2 = verify_repo(root2)
    assert not res2["ok"]
    sig_errs = res2["gates"]["pin_signatures"]["errors"]
    assert "pin_drift:quality/crown_jewels.json" in sig_errs
    assert "pin_drift:quality/epoch_heads.json" in sig_errs


def test_forged_attestation_fails_both_layers(tmp_path: Path) -> None:
    """A repo_integrity receipt mutated post-seal fails the receipt seal;
    a forged in-memory attestation fails its own contract."""
    from quant_fund.research.repo_integrity import write_repo_integrity_receipt

    root, _priv = _mini_repo(tmp_path, sign=False)
    out = root / "quality" / "repo_integrity.json"
    write_repo_integrity_receipt(out, root)
    doc = json.loads(out.read_text())
    # The lie: claim ok while a gate lists its failures.
    doc["gates"]["crown_jewels"]["ok"] = False
    doc["gates"]["crown_jewels"]["errors"] = ["jewel_mutated:x"]
    doc["ok"] = True
    out.write_text(json.dumps(doc, indent=2, sort_keys=True))
    assert not verify_receipt_file(out)["valid"]
    assert "ok_incoherent" in repo_integrity_contract_errors(doc)
