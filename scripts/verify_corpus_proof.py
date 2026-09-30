#!/usr/bin/env python3
"""Standalone corpus-proof auditor — stdlib only.

Independent reimplementation of ``quant_fund.research.epoch_merkle`` for
third-party auditors who hold ONLY the committed heads pin
(``quality/epoch_heads.json`` — quorum-signed via ``gate_pins.sig`` and
OTS-anchored to Bitcoin). No repo access, no library import.

    python3 scripts/verify_corpus_proof.py \
        --proof corpus_proof_XXXX.json \
        --pin quality/epoch_heads.json [--key receipts/*.json]

    python3 scripts/verify_corpus_proof.py \
        --absence absence_proof.json \
        --pin quality/epoch_heads.json [--key receipts/*.json]

    python3 scripts/verify_corpus_proof.py \
        --proof corpus_proof_XXXX.json \
        --checkpoint quality/checkpoint.json \
        --pubkey quality/gate_signing.pub

Checkpoint mode: the heads pin comes from the checkpoint payload itself —
``payload.heads`` carries the same per-corpus receipt/tree_root map, and the
whole payload is Ed25519-signed under the committed gate pubkey (which is
itself Rekor-witnessed + OTS-anchored via the checkpoint's TSA token). A
swapped pin file can't launder anything: with ``--pin`` also given the
script requires sha256(pin bytes) == ``payload.pins[<pin path>]``.

Verifies, entirely offline:

1. The proof names the pinned head epoch receipt — anything else could be a
   fabricated or superseded state.
2. The proof's ``merkle_root`` equals the pin's ``tree_root``.
3. The leaf is sha256(0x00 || canonical{name, sha256}); each node is
   sha256(0x01 || left || right); odd tails PROMOTE (never duplicate).
4. ``leaf_index`` is cryptographically bound: the path's side sequence is
   replayed against the pure shape function of (index, n_members) — a forged
   index produces a differently-shaped path that cannot recompute the pin.
5. Absence proofs: the two sorted-name neighbors bracketing the gap each
   carry an inclusion path; their leaf_indexes must be adjacent
   (hi == lo + 1) AND the bound names must bracket the claimed name
   (lo < name < hi). A single bound must sit on an edge (index 0 or n-1).

Exit 0 only when every check passes. Prints one verdict line per layer.
This file must never import the library — it is the differential oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


_PORTABLE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


def _portable(name: str) -> bool:
    return all(unicodedata.category(c) not in _PORTABLE_CATEGORIES for c in name)


def _sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _canon(value: Any) -> bytes:
    """canonical_json_bytes equivalent for plain JSON types."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _leaf_hash(name: str, sha256: str) -> bytes:
    return _sha(LEAF_PREFIX + _canon({"name": name, "sha256": sha256}))


def _node_hash(left: bytes, right: bytes) -> bytes:
    return _sha(NODE_PREFIX + left + right)


def _expected_sides(leaf_index: int, n_members: int) -> list[str | None]:
    """Per-level sibling side — the tree shape is a pure function of
    (index, n_members), so this replays what the path must look like."""
    sides: list[str | None] = []
    pos, ln = leaf_index, n_members
    while ln > 1:
        if pos == ln - 1 and ln % 2 == 1:
            sides.append(None)  # odd tail promotes — no sibling this level
        else:
            sides.append("right" if pos % 2 == 0 else "left")
        pos //= 2
        ln = (ln + 1) // 2
    return sides


def _root_from_proof(leaf: bytes, path: list[dict[str, str]]) -> bytes:
    node = leaf
    for entry in path:
        sib = bytes.fromhex(entry["sha256"])
        if len(sib) != 32:
            raise ValueError("path sibling not sha256")
        node = _node_hash(node, sib) if entry["side"] == "right" else _node_hash(sib, node)
    return node


def _verify_inclusion(name: str, sha256: str, proof: dict[str, Any], root_hex: str) -> list[str]:
    errors: list[str] = []
    idx = proof.get("leaf_index")
    n = proof.get("n_members")
    path = proof.get("path")
    if not isinstance(idx, int) or not isinstance(n, int) or not isinstance(path, list):
        return ["proof_shape_malformed"]
    if not (0 <= idx < n):
        errors.append("leaf_index_out_of_range")
    expected = _expected_sides(idx, n)
    # Promotion levels (odd tails) contribute no sibling: the path covers
    # only the non-None sides, in order.
    n_siblings = sum(1 for s in expected if s is not None)
    if len(path) != n_siblings:
        errors.append("path_depth_mismatch")
    else:
        it = iter(path)
        for want_side in expected:
            if want_side is None:
                continue
            entry = next(it)
            if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
                errors.append("path_entry_malformed")
                return sorted(set(errors))
            if entry.get("side") != want_side:
                errors.append("path_side_mismatch")
                break
    leaf = _leaf_hash(name, sha256)
    try:
        recomputed = _root_from_proof(leaf, path).hex()
    except (ValueError, KeyError):
        return sorted(set(errors + ["path_malformed"]))
    if recomputed != root_hex:
        errors.append("merkle_root_mismatch")
    return sorted(set(errors))


def _ed25519_verify(pub_hex: str, sig_hex: str, msg: bytes) -> bool | None:
    """Repo convention: Ed25519 keys/sigs are raw 32/64-byte hex, not PEM.

    Returns None when the cryptography backend is unavailable — a distinct
    state from a signature that verified False.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return None
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex.strip()))
        pub.verify(bytes.fromhex(sig_hex), msg)
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def _ed25519_key_id(pubkey_hex: str) -> str:
    """kid = sha256(pubkey bytes)[:16] — mirrors gate_signatures.key_id."""
    return hashlib.sha256(bytes.fromhex(pubkey_hex.strip())).hexdigest()[:16]


def _checkpoint_heads(
    checkpoint: dict[str, Any], pubkey: Path, quorum: Path | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Authenticate the checkpoint envelope, return its heads map.

    v1: ``signature`` verified under ``pubkey``. v2: a ``signatures`` list
    of ``{key_id, signature}`` — with ``--quorum <registry.json>`` every
    signer must be registered, at least ``threshold`` signatures verify,
    the presented ``--pubkey`` must itself be a registry member, and the
    payload's ``quorum.registry_sha256`` must equal the canonical digest of
    the presented registry (a swapped same-keys file fails). Without
    ``--quorum`` the presented pubkey authenticates signatures directly.
    """
    errors: list[str] = []
    schema = checkpoint.get("schema")
    payload = checkpoint.get("payload")
    if not isinstance(payload, dict):
        return {}, ["checkpoint_payload_missing"]
    if checkpoint.get("algorithm") != "ed25519":
        errors.append("checkpoint_algorithm")
    msg = _canon(payload)
    if schema == "integrity_checkpoint_sig.v1":
        sig = checkpoint.get("signature")
        if not isinstance(sig, str):
            errors.append("checkpoint_signature_malformed")
        else:
            verdict = _ed25519_verify(pubkey.read_text().strip(), sig, msg)
            if verdict is None:
                errors.append("crypto_backend_unavailable")
            elif verdict is False:
                errors.append("checkpoint_signature_invalid")
    elif schema == "integrity_checkpoint_sig.v2":
        sigs = checkpoint.get("signatures")
        if not isinstance(sigs, list) or not sigs:
            errors.append("checkpoint_signature_malformed")
            sigs = []
        registered: dict[str, str] = {}
        threshold = 1
        if quorum is not None:
            try:
                reg = json.loads(quorum.read_bytes())
                if (
                    reg.get("schema") != "gate_quorum.v1"
                    or not isinstance(reg.get("keys"), list)
                    or not isinstance(reg.get("threshold"), int)
                ):
                    errors.append("quorum_registry_malformed")
                else:
                    registered = {
                        str(k["key_id"]): str(k["pubkey"])
                        for k in reg["keys"]
                        if isinstance(k, dict)
                    }
                    threshold = int(reg["threshold"])
                    # The presented --pubkey must itself be a registry
                    # member — else an auditor's trust anchor silently
                    # passes under keys it never chose.
                    presented = _ed25519_key_id(pubkey.read_text())
                    if presented not in registered:
                        errors.append(f"pubkey_not_in_quorum:{presented}")
                    # The signed payload names the authorizing registry's
                    # canonical digest — mirrors registry_sha256. A swapped
                    # --quorum file must fail even when it lists the keys.
                    claimed_q = payload.get("quorum")
                    claimed_digest = (
                        claimed_q.get("registry_sha256") if isinstance(claimed_q, dict) else None
                    )
                    if claimed_digest is None:
                        errors.append("quorum_unbound")
                    else:
                        canon = (json.dumps(reg, indent=2, sort_keys=True) + "\n").encode()
                        if claimed_digest != hashlib.sha256(canon).hexdigest():
                            errors.append("quorum_registry_drift")
            except (OSError, ValueError):
                errors.append("quorum_registry_unreadable")
        else:
            registered = {_ed25519_key_id(pubkey.read_text()): pubkey.read_text().strip()}
        seen: set[str] = set()
        valid = 0
        for entry in sigs:
            if not isinstance(entry, dict):
                errors.append("checkpoint_signature_malformed")
                continue
            kid = str(entry.get("key_id", ""))
            pub = registered.get(kid)
            if pub is None:
                errors.append(f"unknown_signer:{kid}")
                continue
            if kid in seen:
                errors.append(f"duplicate_signer:{kid}")
                continue
            seen.add(kid)
            verdict = _ed25519_verify(pub.strip(), str(entry.get("signature", "")), msg)
            if verdict is None:
                errors.append("crypto_backend_unavailable")
            elif verdict is True:
                valid += 1
            else:
                errors.append(f"checkpoint_signature_invalid:{kid}")
        if valid < threshold:
            errors.append(f"checkpoint_quorum_not_met:{valid}/{threshold}")
    else:
        errors.append("checkpoint_schema")
    heads = payload.get("heads")
    if not isinstance(heads, dict):
        errors.append("checkpoint_heads_missing")
        heads = {}
    return heads, errors


def _pin_entry(
    pin: dict[str, Any], key: str | None, receipt: str
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    heads = pin.get("heads", {}) if isinstance(pin, dict) else {}
    if not isinstance(heads, dict):
        return None, None, "pin_malformed"
    if key is not None:
        entry = heads.get(key)
        return (key, entry if isinstance(entry, dict) else None, None)
    matches = [
        (k, v) for k, v in heads.items() if isinstance(v, dict) and v.get("receipt") == receipt
    ]
    if len(matches) == 1:
        return matches[0][0], matches[0][1], None
    if not matches:
        return None, None, f"epoch_receipt_not_pinned:{receipt}"
    return None, None, "receipt_ambiguous:pass --key"


def audit_proof(proof: dict[str, Any], pin: dict[str, Any], key: str | None) -> list[str]:
    """Inclusion proof against the pin. Returns error labels; [] = verified."""
    errors: list[str] = []
    member = proof.get("member")
    sha = proof.get("member_sha256")
    receipt = proof.get("epoch_receipt")
    root = proof.get("merkle_root")
    if not isinstance(member, str) or not isinstance(sha, str):
        return ["member_fields_malformed"]
    if unicodedata.normalize("NFC", member) != member:
        return ["member_name_not_nfc"]
    if not _portable(member):
        return ["member_name_not_portable"]
    if not isinstance(receipt, str) or not isinstance(root, str):
        return ["epoch_binding_malformed"]
    k, entry, err = _pin_entry(pin, key, receipt)
    if err:
        return [err]
    if not isinstance(entry, dict):
        return [f"pin_key_absent:{key}"]
    if proof.get("corpus_key") not in (None, k):
        errors.append("corpus_key_mismatch")
    pinned_root = entry.get("tree_root")
    if not isinstance(pinned_root, str):
        errors.append("pin_tree_root_absent")
    elif root != pinned_root:
        errors.append("merkle_root_not_pinned")
    # corpus_proof.v1 bodies carry leaf_index/n_members/path flat.
    errors += _verify_inclusion(member, sha, proof, root)
    return sorted(set(errors))


def audit_absence(proof: dict[str, Any], pin: dict[str, Any], key: str | None) -> list[str]:
    """Bounding-neighbor absence proof against the pin."""
    errors: list[str] = []
    name = proof.get("name")
    root = proof.get("merkle_root")
    bounds = proof.get("bounds")
    if not isinstance(name, str) or not name:
        return ["name_missing"]
    if unicodedata.normalize("NFC", name) != name:
        return ["name_not_nfc"]
    if not _portable(name):
        return ["name_not_portable"]
    if not isinstance(root, str) or not isinstance(bounds, list):
        return ["absence_shape_malformed"]
    receipt = proof.get("epoch_receipt")
    resolved_key: str | None = None
    if isinstance(receipt, str):
        resolved_key, entry, err = _pin_entry(pin, key, receipt)
        if err:
            return [err]
    elif key is not None:
        # library absence proofs carry no epoch_receipt — bind by tree_root
        entry = (pin.get("heads") or {}).get(key)
        resolved_key = key
        if not isinstance(entry, dict):
            return [f"pin_key_absent:{key}"]
    else:
        heads = pin.get("heads", {})
        matches = [
            (k, v) for k, v in heads.items() if isinstance(v, dict) and v.get("tree_root") == root
        ]
        if len(matches) != 1:
            return [f"tree_root_not_pinned:{len(matches)}_matches"]
        resolved_key, entry = matches[0]
    if not isinstance(entry, dict):
        return [f"pin_key_absent:{key}"]
    if proof.get("corpus_key") not in (None, resolved_key):
        errors.append("corpus_key_mismatch")
    pinned_root = entry.get("tree_root")
    if not isinstance(pinned_root, str):
        errors.append("pin_tree_root_absent")
    elif root != pinned_root:
        errors.append("merkle_root_not_pinned")
        return sorted(set(errors))
    if proof.get("n_members") == 0 and bounds:
        errors.append("empty_tree_with_bounds")
    idxs: list[int] = []
    for b in bounds:
        if not isinstance(b, dict):
            errors.append("bound_malformed")
            continue
        bname = b.get("member")
        if not isinstance(bname, str) or bname == name:
            errors.append("bound_not_neighbor")
            continue
        if unicodedata.normalize("NFC", bname) != bname:
            errors.append("bound_member_not_nfc")
            continue
        if not _portable(bname):
            errors.append("bound_member_not_portable")
            continue
        sub = _verify_inclusion(bname, str(b.get("member_sha256", "")), b, root)
        if sub:
            errors += [f"bound_invalid:{ascii(bname)}"]
            continue
        idxs.append(int(b["leaf_index"]))
        if not (bname < name or bname > name):
            errors.append("bound_self")
    if errors:
        return sorted(set(errors))
    if len(bounds) > 2:
        errors.append("bounds_len")
    if len(idxs) == 2 and idxs[1] - idxs[0] != 1:
        errors.append("bounds_not_adjacent")
    # Adjacency alone doesn't bracket the name — pin mode has no member map,
    # so bounds elsewhere in the tree would otherwise pass as an "absence"
    # proof for a member that sits outside them.
    if len(idxs) == 2 and not (str(bounds[0]["member"]) < name < str(bounds[1]["member"])):
        errors.append("bounds_not_bracketing")
    n = proof.get("n_members")
    if len(idxs) == 1 and isinstance(n, int) and n > 1:
        b = bounds[0]
        bname = str(b["member"])
        edge_ok = (bname < name and int(b["leaf_index"]) == n - 1) or (
            bname > name and int(b["leaf_index"]) == 0
        )
        if not edge_ok:
            errors.append("single_bound_not_edge")
    return sorted(set(errors))


def audit_history_absence(proof: dict[str, Any], pin: dict[str, Any], key: str | None) -> list[str]:
    """corpus_history_absence.v1 — pin-mode structural + head binding.

    Offline scope (pin only, no corpus): the claimed chain head must equal
    the pinned head (a proof stopping early would hide later membership),
    the epoch list must be internally coherent (n_epochs == len, head ==
    last entry, every entry well-formed), and the claimed name NFC-portable.
    Mid-chain member-map absence itself requires corpus access — this mode
    authenticates the *envelope*; the lib verifier does the full replay.
    """
    errors: list[str] = []
    name = proof.get("name")
    if not isinstance(name, str) or not name:
        errors.append("name_missing")
    elif unicodedata.normalize("NFC", name) != name:
        errors.append("name_not_nfc")
    elif not _portable(name):
        errors.append("name_not_portable")
    epochs = proof.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        errors.append("epochs_missing")
        epochs = []
    else:
        for i, e in enumerate(epochs):
            if not isinstance(e, dict):
                errors.append(f"epoch_entry_malformed:{i}")
                continue
            if not isinstance(e.get("receipt"), str):
                errors.append(f"epoch_entry_malformed:{i}")
            for field in ("sha256", "epoch_root_sha256"):
                v = e.get(field)
                if not (isinstance(v, str) and len(v) == 64):
                    errors.append(f"epoch_entry_malformed:{i}:{field}")
            prev_root = e.get("prev_root")
            if not (isinstance(prev_root, str) and len(prev_root) == 64):
                errors.append(f"epoch_entry_malformed:{i}:prev_root")
            elif i == 0 and prev_root != "0" * 64:
                errors.append("genesis_prev_root_nonzero")
    # Interior links are verifiable offline: each epoch's prev_root must name
    # the previous entry's epoch_root_sha256 — a tampered interior entry
    # breaks the chain that terminates at the pinned head. Well-formedness of
    # both fields was already checked above; only compare when both decode.
    for i in range(1, len(epochs)):
        prev_e, e = epochs[i - 1], epochs[i]
        if (
            isinstance(prev_e, dict)
            and isinstance(e, dict)
            and isinstance(e.get("prev_root"), str)
            and isinstance(prev_e.get("epoch_root_sha256"), str)
            and e["prev_root"] != prev_e["epoch_root_sha256"]
        ):
            errors.append(f"history_link_broken:{i}")
    if proof.get("n_epochs") != len(epochs):
        errors.append("n_epochs_mismatch")
    if epochs and isinstance(epochs[-1], dict):
        if proof.get("head_receipt") != epochs[-1].get("receipt"):
            errors.append("head_receipt_mismatch")
        if proof.get("head_sha256") != epochs[-1].get("sha256"):
            errors.append("head_sha256_mismatch")
    head = proof.get("head_receipt")
    if isinstance(head, str):
        k, entry, err = _pin_entry(pin, key, head)
        if err:
            errors.append(err)
        elif not isinstance(entry, dict):
            errors.append(f"pin_key_absent:{key}")
        else:
            if proof.get("corpus_key") not in (None, k):
                errors.append("corpus_key_mismatch")
            if entry.get("receipt") != head:
                errors.append("history_head_not_pinned")
            elif entry.get("sha256") != proof.get("head_sha256"):
                errors.append("history_head_digest_drift")
    return sorted(set(errors))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proof", type=Path, default=None, help="corpus_proof.v1 receipt")
    ap.add_argument("--absence", type=Path, default=None, help="absence proof JSON")
    ap.add_argument(
        "--history-absence",
        type=Path,
        default=None,
        help="corpus_history_absence.v1 receipt (pin-mode head binding)",
    )
    ap.add_argument("--pin", type=Path, default=None, help="epoch_heads.json")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Signed checkpoint carrying the heads pin map (authenticates --pin)",
    )
    ap.add_argument(
        "--pubkey",
        type=Path,
        default=None,
        help="Gate Ed25519 pubkey (raw hex) — required with --checkpoint",
    )
    ap.add_argument("--key", type=str, default=None, help="pin key e.g. 'receipts/*.json'")
    ap.add_argument(
        "--quorum",
        type=Path,
        default=None,
        help="gate_quorum.v1 registry — v2 checkpoints resolve signers against it",
    )
    args = ap.parse_args()
    n_targets = sum(x is not None for x in (args.proof, args.absence, args.history_absence))
    if n_targets != 1:
        print("pass exactly one of --proof/--absence/--history-absence", file=sys.stderr)
        return 2
    if args.pin is None and args.checkpoint is None:
        print("pass --pin and/or --checkpoint", file=sys.stderr)
        return 2
    if args.checkpoint is not None and args.pubkey is None:
        print("--checkpoint requires --pubkey", file=sys.stderr)
        return 2

    pre_errors: list[str] = []
    if args.checkpoint is not None:
        checkpoint = json.loads(args.checkpoint.read_text())
        assert args.pubkey is not None
        heads, pre_errors = _checkpoint_heads(checkpoint, args.pubkey, args.quorum)
        for e in pre_errors:
            print(f"  {e}")
        pin_heads: dict[str, Any] = heads
        if args.pin is not None:
            # Cross-anchor: the checkpoint must pin this pin file's bytes.
            pins = checkpoint.get("payload", {}).get("pins", {})
            rel = args.pin.as_posix().removeprefix("./")
            want = pins.get(rel)
            if want is None:
                want = next((v for k, v in pins.items() if rel.endswith(k)), None)
            got = hashlib.sha256(args.pin.read_bytes()).hexdigest()
            if want != got:
                pre_errors.append(f"checkpoint_pin_mismatch:{args.pin.name}")
    elif args.pin is not None:
        pin_heads = json.loads(args.pin.read_text()).get("heads", {})
    else:
        pin_heads = {}

    pin = {"heads": pin_heads}
    if args.proof is not None:
        payload = json.loads(args.proof.read_text())
        # sealed receipts may wrap the body under 'payload'
        body = payload.get("payload", payload)
        errors = pre_errors + audit_proof(body, pin, args.key)
        print(f"inclusion member={body.get('member')} -> {len(errors)} error(s)")
    elif args.history_absence is not None:
        payload = json.loads(args.history_absence.read_text())
        body = payload.get("payload", payload)
        errors = pre_errors + audit_history_absence(body, pin, args.key)
        print(
            f"history-absence name={body.get('name')} "
            f"epochs={body.get('n_epochs')} -> {len(errors)} error(s)"
        )
    else:
        payload = json.loads(args.absence.read_text())
        # sealed corpus_absence.v1 receipts wrap the body under 'payload';
        # raw library absence_proof() dicts pass through unchanged.
        body = payload.get("payload", payload)
        errors = pre_errors + audit_absence(body, pin, args.key)
        print(f"absence name={body.get('name')} -> {len(errors)} error(s)")
    for e in errors:
        print(f"  {e}")
    if errors:
        print("FAIL")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
