#!/usr/bin/env python3
"""Standalone corpus-proof auditor — stdlib only.

Independent reimplementation of ``quant_fund.research.epoch_merkle`` for
third-party auditors who hold ONLY the committed heads pin
(``quality/epoch_heads.json`` — quorum-signed via ``gate_pins.sig`` and
OTS-anchored to Bitcoin). No repo access, no library import required —
``--corpus-dir`` additionally replays proofs against a live corpus
(checkpoint-authenticated or not), making this script a full differential
oracle for the library's own verifier.

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

    python3 scripts/verify_corpus_proof.py \
        --consistency epoch_consistency_XXXX.json \
        --corpus-dir receipts [--held <from-head-sha256>]

All four proof kinds verify: corpus_proof.v1 (inclusion),
corpus_absence.v1 (non-membership at a bound epoch),
corpus_history_absence.v1 (absence at every committed epoch), and
epoch_consistency.v1 (the live chain extends a held head — RFC 6962's
consistency half). Pin mode binds a consistency proof's `to` endpoint to
the pinned head; live mode re-walks every hop's digest, prev link, and
successor member pin.

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


def _digest_hex(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _portable(name: str) -> bool:
    return all(unicodedata.category(c) not in _PORTABLE_CATEGORIES for c in name)


def _sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _canon(value: Any) -> bytes:
    """utils.hashing.canonical_json_bytes equivalent for plain JSON inputs:
    sorted keys, tight separators, UTF-8 (ensure_ascii=False), no NaN.
    ``_canonicalize`` is the identity on already-parsed JSON values, which
    is all this script ever serializes."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


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


def _history_shape_errors(proof: dict[str, Any]) -> list[str]:
    """Envelope-only checks for corpus_history_absence.v1 — no pin, no
    corpus. Name portability, epoch-entry shapes, prev_root genesis rule,
    interior link coherence, n_epochs/head self-consistency."""
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
    return sorted(set(errors))


def audit_history_absence(proof: dict[str, Any], pin: dict[str, Any], key: str | None) -> list[str]:
    """corpus_history_absence.v1 — pin-mode structural + head binding.

    Offline scope (pin only, no corpus): the claimed chain head must equal
    the pinned head (a proof stopping early would hide later membership),
    the epoch list must be internally coherent (n_epochs == len, head ==
    last entry, every entry well-formed, interior prev_root links intact),
    and the claimed name NFC-portable. Mid-chain member-map absence itself
    requires corpus access — this mode authenticates the *envelope*; the
    lib verifier (or ``--corpus-dir``) does the full replay.
    """
    errors = _history_shape_errors(proof)
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


def _epoch_members(doc: dict[str, Any]) -> dict[str, str]:
    """`members` list → {name: sha256}; malformed entries drop out —
    mirrors corpus_epoch._member_maps."""
    out: dict[str, str] = {}
    for entry in doc.get("members") or []:
        if isinstance(entry, dict):
            name, sha = entry.get("name"), entry.get("sha256")
            if isinstance(name, str) and isinstance(sha, str):
                out[name] = sha
    return out


def _load_epoch_doc(corpus_dir: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Seal-authenticate one corpus_epoch_<sha16>.json — mirrors
    epoch_merkle._load_epoch_receipt."""
    path = corpus_dir / name
    if not path.is_file():
        return None, "epoch_receipt_missing"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "epoch_receipt_unparseable"
    if not isinstance(doc, dict):
        return None, "epoch_receipt_unparseable"
    seal = doc.get("receipt_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        return None, "epoch_receipt_unsealed"
    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    if hashlib.sha256(_canon(body)).hexdigest() != seal:
        return None, "epoch_receipt_tampered"
    if path.name != f"corpus_epoch_{seal[:16]}.json":
        return None, "epoch_receipt_name_mismatch"
    return doc, None


def _chain_index(
    corpus_dir: Path, pattern: str
) -> tuple[dict[str, tuple[str, dict[str, Any]]], list[str]]:
    """name → (file_sha256, seal-authenticated doc) for one (dir, pattern)
    chain — mirrors epoch_consistency.chain_index: squatters and unsealed/
    tampered receipts report errors and never enter the index."""
    errors: list[str] = []
    by_name: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted(corpus_dir.glob("*.json")):
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            if path.name.startswith("corpus_epoch_"):
                errors.append(f"epoch_prefix_squat:{path.name}")
            continue
        inner = doc.get("payload") if isinstance(doc, dict) else None
        cand = inner if isinstance(inner, dict) else doc
        if not (
            isinstance(cand, dict)
            and (cand.get("schema") == "corpus_epoch.v1" or cand.get("kind") == "corpus_epoch.v1")
        ):
            if path.name.startswith("corpus_epoch_"):
                errors.append(f"epoch_prefix_squat:{path.name}")
            continue
        params = cand.get("params")
        pat = params.get("pattern", "*.json") if isinstance(params, dict) else "*.json"
        if pat != pattern:
            continue
        loaded, err = _load_epoch_doc(corpus_dir, path.name)
        if err is not None:
            errors.append(f"{err}:{path.name}")
            continue
        assert loaded is not None
        by_name[path.name] = (hashlib.sha256(path.read_bytes()).hexdigest(), loaded)
    return by_name, errors


def _ordered_chain(
    corpus_dir: Path, pattern: str
) -> tuple[list[tuple[str, str, str, str | None, dict[str, str]]], list[str]]:
    """Genesis→head chain for one corpus dir — independent reimplementation
    of epoch_merkle.ordered_epoch_chain (squatters, unauth receipts,
    orphans, forks, multi-genesis/head all fail closed)."""
    by_name, errors = _chain_index(corpus_dir, pattern)

    child_of: dict[str, str] = {}
    genesis: list[str] = []
    for name, (_, rec) in by_name.items():
        prev = rec.get("prev_epoch_receipt")
        if prev is None:
            genesis.append(name)
            continue
        if prev not in by_name:
            errors.append(f"epoch_orphan:{name}")
            continue
        if rec.get("prev_epoch_sha256") != by_name[prev][1].get("epoch_root_sha256"):
            errors.append(f"epoch_prev_root_mismatch:{name}")
        if prev in child_of:
            errors.append(f"epoch_fork:{prev}")
        else:
            child_of[prev] = name
    if len(genesis) != 1:
        errors.append(f"epoch_genesis_count:{len(genesis)}")

    ordered: list[tuple[str, str, str, str | None, dict[str, str]]] = []
    seen: set[str] = set()
    cur = genesis[0] if len(genesis) == 1 else None
    while cur is not None and cur in by_name and cur not in seen:
        seen.add(cur)
        file_sha, rec = by_name[cur]
        root_hex = rec.get("epoch_root_sha256")
        prev_hex = rec.get("prev_epoch_sha256")
        ordered.append(
            (
                cur,
                file_sha,
                root_hex if isinstance(root_hex, str) else "",
                prev_hex if isinstance(prev_hex, str) else None,
                _epoch_members(rec),
            )
        )
        cur = child_of.get(cur)
    if len(ordered) != len(by_name):
        errors.append(f"epoch_chain_disconnected:{len(ordered)}/{len(by_name)}")
    return ordered, sorted(set(errors))


def _merkle_root(members: dict[str, str]) -> str:
    """Recompute the epoch member root — sorted-name leaf tree with
    odd-node promotion, mirrors epoch_merkle.merkle_root."""
    leaves = [_leaf_hash(n, s) for n, s in sorted(members.items())]
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = leaves
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_node_hash(level[i], level[i + 1]))
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def audit_proof_shape(proof: dict[str, Any]) -> list[str]:
    """corpus_proof.v1 shape — mirrors corpus_proof_errors (kind/schema,
    honesty stamps, member/sha/root/merkle fields, index bounds, path
    replay). Independent of any pin."""
    errors: list[str] = []
    if proof.get("kind") != "corpus_proof.v1":
        errors.append("kind_not_corpus_proof")
    if proof.get("schema") != "corpus_proof.v1":
        errors.append("schema_not_corpus_proof")
    if proof.get("research_only") is not True:
        errors.append("research_only_not_true")
    if proof.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    member = proof.get("member")
    sha = proof.get("member_sha256")
    if not isinstance(member, str) or not member:
        errors.append("member_not_str")
    elif unicodedata.normalize("NFC", member) != member:
        errors.append("member_name_not_nfc")
    elif not _portable(member):
        errors.append("member_name_not_portable")
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
        errors.append("member_sha256_not_hex")
    if not (
        isinstance(proof.get("epoch_root_sha256"), str) and len(proof["epoch_root_sha256"]) == 64
    ):
        errors.append("epoch_root_sha256")
    if not (isinstance(proof.get("merkle_root"), str) and len(proof["merkle_root"]) == 64):
        errors.append("merkle_root")
    idx, n, path = proof.get("leaf_index"), proof.get("n_members"), proof.get("path")
    if not isinstance(idx, int) or not isinstance(n, int) or idx < 0 or idx >= n:
        errors.append("leaf_index_bounds")
    if not isinstance(path, list) or not all(
        isinstance(e, dict) and e.get("side") in ("left", "right") for e in path
    ):
        errors.append("path_malformed")
    elif (
        not errors
        and isinstance(member, str)
        and isinstance(sha, str)
        and isinstance(proof.get("merkle_root"), str)
    ):
        errors += _verify_inclusion(member, sha, proof, proof["merkle_root"])
    return sorted(set(errors))


def audit_proof_live(proof: dict[str, Any], corpus_dir: Path) -> list[str]:
    """Live replay of corpus_proof.v1 — mirrors verify_epoch_proof."""
    errors = audit_proof_shape(proof)
    if errors:
        return errors
    receipt, err = _load_epoch_doc(corpus_dir, str(proof["epoch_receipt"]))
    if err is not None:
        return [err]
    assert receipt is not None
    if receipt.get("epoch_root_sha256") != proof["epoch_root_sha256"]:
        return ["epoch_root_mismatch"]
    members = _epoch_members(receipt)
    if members.get(str(proof["member"])) != proof["member_sha256"]:
        return ["member_digest_mismatch"]
    if proof["n_members"] != len(members):
        return ["n_members_mismatch"]
    if _merkle_root(members) != proof["merkle_root"]:
        return ["merkle_root_mismatch"]
    return []


def audit_absence_live(proof: dict[str, Any], corpus_dir: Path) -> list[str]:
    """Live replay of corpus_absence.v1 — mirrors verify_epoch_absence."""
    name = proof.get("name")
    if not isinstance(name, str) or not name:
        return ["name_missing"]
    if not isinstance(proof.get("epoch_receipt"), str):
        return ["epoch_binding_malformed"]
    receipt, err = _load_epoch_doc(corpus_dir, str(proof["epoch_receipt"]))
    if err is not None:
        return [err]
    assert receipt is not None
    if receipt.get("epoch_root_sha256") != proof.get("epoch_root_sha256"):
        return ["epoch_root_mismatch"]
    members = _epoch_members(receipt)
    if name in members:
        return ["name_is_member"]
    if proof.get("n_members") != len(members):
        return ["n_members_mismatch"]
    if _merkle_root(members) != proof.get("merkle_root"):
        return ["merkle_root_mismatch"]
    return []


def audit_history_absence_live(proof: dict[str, Any], corpus_dir: Path) -> list[str]:
    """Live replay of corpus_history_absence.v1 — mirrors
    verify_history_absence: full re-walk + exact sequence + member scan."""
    errors = _history_shape_errors(proof)
    if errors:
        return errors
    key = proof.get("corpus_key")
    pattern = key.rsplit("/", 1)[-1] if isinstance(key, str) and "/" in key else "*.json"
    ordered, chain_errors = _ordered_chain(corpus_dir, pattern)
    if chain_errors:
        return [f"history_chain_unverifiable:{chain_errors[0]}"]
    claimed = [
        (e.get("receipt"), e.get("sha256"), e.get("epoch_root_sha256"), e.get("prev_root"))
        for e in proof["epochs"]
        if isinstance(e, dict)
    ]
    actual = [(n, sha, r, p) for n, sha, r, p, _ in ordered]
    if claimed != actual:
        return ["history_chain_mismatch"]
    for n, _, _, _, members in ordered:
        if proof["name"] in members:
            return [f"history_member_present:{n}"]
    return []


def audit_consistency_shape(proof: dict[str, Any]) -> list[str]:
    """epoch_consistency.v1 shape — mirrors consistency_contract_errors
    (kind/schema, honesty stamps, hops list, endpoint coherence, n_hops,
    proof digest) plus a re-derivation of proof_sha256."""
    errors: list[str] = []
    if proof.get("kind") != "epoch_consistency.v1":
        errors.append("kind_not_consistency")
    if proof.get("schema") != "epoch_consistency.v1":
        errors.append("schema_not_consistency")
    if proof.get("research_only") is not True:
        errors.append("research_only_not_true")
    if proof.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    hops = proof.get("hops")
    if not isinstance(hops, list) or not hops:
        errors.append("hops_missing")
        hops = []
    for hop in hops:
        if not isinstance(hop, dict) or not _digest_hex(hop.get("sha256")):
            errors.append("hop_malformed")
            break
        if not isinstance(hop.get("name"), str):
            errors.append("hop_name_not_string")
            break
    for endpoint in ("from_receipt", "to_receipt"):
        e = proof.get(endpoint)
        if not isinstance(e, dict):
            errors.append(f"{endpoint}_missing")
        elif hops:
            expected = hops[0] if endpoint == "from_receipt" else hops[-1]
            if e.get("name") != expected["name"] or e.get("sha256") != expected["sha256"]:
                errors.append(f"{endpoint}_incoherent")
    n_hops = proof.get("n_hops")
    if not isinstance(n_hops, int) or n_hops != len(hops):
        errors.append("n_hops_mismatch")
    proof_sha = proof.get("proof_sha256")
    if not _digest_hex(proof_sha):
        errors.append("proof_sha256")
    elif hops and all(
        isinstance(h, dict) and isinstance(h.get("name"), str) and _digest_hex(h.get("sha256"))
        for h in hops
    ):
        want = hashlib.sha256(_canon({"hops": hops, "pattern": proof.get("pattern")})).hexdigest()
        if proof_sha != want:
            errors.append("proof_sha256_mismatch")
    return sorted(set(errors))


def audit_consistency(proof: dict[str, Any], pin: dict[str, Any], key: str | None) -> list[str]:
    """Pin-mode epoch_consistency.v1 check: the proof's `to` head must be
    the pinned head for its corpus key — extension to a stale head is a
    valid proof of nothing current."""
    errors = audit_consistency_shape(proof)
    if errors:
        return errors
    heads = pin.get("heads", {})
    corpus_key = str(proof.get("corpus_dir", "")) + "/" + str(proof.get("pattern", "*.json"))
    entry = heads.get(key or corpus_key)
    if not isinstance(entry, dict):
        entry = next(
            (
                e
                for k, e in heads.items()
                if isinstance(e, dict) and e.get("receipt") == proof["to_receipt"]["name"]
            ),
            None,
        )
    if not isinstance(entry, dict):
        return ["consistency_no_pin_entry"]
    if entry.get("receipt") != proof["to_receipt"]["name"]:
        errors.append("consistency_to_not_pinned")
    elif entry.get("sha256") != proof["to_receipt"]["sha256"]:
        errors.append("consistency_to_digest_drift")
    return errors


def audit_consistency_live(
    proof: dict[str, Any], corpus_dir: Path, held_sha256: str | None = None
) -> list[str]:
    """Live replay of epoch_consistency.v1 — mirrors verify_consistency:
    every hop's receipt must exist unaltered in the live chain, each
    successor's prev link and member map must pin its predecessor's bytes,
    and `to` must be a live chain head."""
    errors = audit_consistency_shape(proof)
    if errors:
        return errors
    pattern = proof.get("pattern")
    index, index_errors = _chain_index(
        corpus_dir, str(pattern) if isinstance(pattern, str) else "*.json"
    )
    errors += index_errors
    hops = proof["hops"]
    if held_sha256 is not None and hops[0]["sha256"] != held_sha256:
        errors.append("held_head_digest_mismatch")
    prev: dict[str, Any] | None = None
    for hop in hops:
        entry = index.get(hop["name"])
        if entry is None:
            errors.append(f"hop_missing:{hop['name']}")
            continue
        digest, doc = entry
        if digest != hop["sha256"]:
            errors.append(f"hop_digest_mismatch:{hop['name']}")
        if prev is not None:
            if doc.get("prev_epoch_receipt") != prev["name"]:
                errors.append(f"hop_link_broken:{hop['name']}")
            member_sha = _epoch_members(doc).get(prev["name"])
            if member_sha != prev["sha256"]:
                errors.append(f"hop_member_digest_mismatch:{hop['name']}")
        prev = hop
    if hops:
        prevs = {d.get("prev_epoch_receipt") for _, d in index.values()}
        if hops[-1]["name"] in prevs:
            errors.append("to_not_chain_head")
    return sorted(set(errors))


def _member_map_sha(members: dict[str, str]) -> str:
    """sha256 over the canonical sorted {name: sha256} map — mirrors
    epoch_delta.member_map_sha256."""
    return hashlib.sha256(_canon({k: members[k] for k in sorted(members)})).hexdigest()


def audit_delta_shape(delta: dict[str, Any]) -> list[str]:
    """epoch_delta.v1 shape — mirrors epoch_delta_errors: kind/schema,
    honesty stamps, both epoch blocks (receipt name, merkle root, map
    digest, member counts), every transition row's inclusion path replayed
    against its declared root, set disjointness, and the unchanged-count
    accounting pins."""
    errors: list[str] = []
    if delta.get("kind") != "epoch_delta.v1":
        errors.append("kind_not_epoch_delta")
    if delta.get("schema") != "epoch_delta.v1":
        errors.append("schema_not_epoch_delta")
    if delta.get("research_only") is not True:
        errors.append("research_only_not_true")
    if delta.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    prev = delta.get("prev_epoch")
    nxt = delta.get("next_epoch")
    if not isinstance(prev, dict) or not isinstance(nxt, dict):
        errors.append("epoch_blocks_missing")
        return sorted(set(errors))
    for label, block in (("prev", prev), ("next", nxt)):
        if not isinstance(block.get("receipt"), str):
            errors.append(f"{label}_receipt_missing")
        if not _digest_hex(block.get("merkle_root")):
            errors.append(f"{label}_merkle_root_bad")
        if not _digest_hex(block.get("member_map_sha256")):
            errors.append(f"{label}_member_map_sha256_bad")
        if not isinstance(block.get("n_members"), int):
            errors.append(f"{label}_n_members_bad")
    if errors:
        return sorted(set(errors))
    tr = delta.get("transitions")
    if not isinstance(tr, dict):
        return sorted({*errors, "transitions_missing"})
    added = tr.get("added")
    removed = tr.get("removed")
    changed = tr.get("changed")
    unchanged = tr.get("unchanged_count")
    if (
        not isinstance(added, list)
        or not isinstance(removed, list)
        or not isinstance(changed, list)
    ):
        return sorted({*errors, "transitions_malformed"})
    prev_root, next_root = str(prev["merkle_root"]), str(nxt["merkle_root"])

    def _row(row: Any, root: str, which: str) -> list[str]:
        if not isinstance(row, dict):
            return [f"{which}_row_malformed"]
        name, sha = row.get("name"), row.get("sha256")
        if not isinstance(name, str) or not name:
            return [f"{which}_name_bad"]
        if unicodedata.normalize("NFC", name) != name or not _portable(name):
            return [f"{which}_name_bad"]
        if not _digest_hex(sha):
            return [f"{which}_sha256_bad"]
        proof = {k: row.get(k) for k in ("leaf_index", "n_members", "path")}
        return _verify_inclusion(name, str(sha), proof, root)

    for row in added:
        errors += _row(row, next_root, "added")
    for row in removed:
        errors += _row(row, prev_root, "removed")
    for row in changed:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            errors.append("changed_row_malformed")
            continue
        f_sha, t_sha = row.get("from_sha256"), row.get("to_sha256")
        if not _digest_hex(f_sha) or not _digest_hex(t_sha):
            errors.append("changed_sha256_bad")
            continue
        if f_sha == t_sha:
            errors.append("changed_identical_digests")
        if _verify_inclusion(
            row["name"],
            str(f_sha),
            {
                "leaf_index": row.get("prev_leaf_index"),
                "n_members": row.get("n_prev_members"),
                "path": row.get("prev_path"),
            },
            prev_root,
        ):
            errors.append(f"changed_prev_path_invalid:{row['name']}")
        if _verify_inclusion(
            row["name"],
            str(t_sha),
            {
                "leaf_index": row.get("next_leaf_index"),
                "n_members": row.get("n_next_members"),
                "path": row.get("next_path"),
            },
            next_root,
        ):
            errors.append(f"changed_next_path_invalid:{row['name']}")

    def _rnames(rows: list[Any]) -> set[str]:
        return {str(r.get("name")) for r in rows if isinstance(r, dict)}

    a_names, r_names, c_names = _rnames(added), _rnames(removed), _rnames(changed)
    if a_names & r_names or a_names & c_names or r_names & c_names:
        errors.append("transition_sets_overlap")
    if not isinstance(unchanged, int) or unchanged < 0:
        errors.append("unchanged_count_bad")
    else:
        if len(r_names) + len(c_names) + unchanged != prev["n_members"]:
            errors.append("prev_accounting")
        if len(a_names) + len(c_names) + unchanged != nxt["n_members"]:
            errors.append("next_accounting")
    return sorted(set(errors))


def audit_delta_live(delta: dict[str, Any], corpus_dir: Path) -> list[str]:
    """Live replay of epoch_delta.v1 — mirrors verify_epoch_delta: both
    epoch receipts are seal-authenticated and their member maps re-derived;
    the declared transition table must equal the computed set difference
    exactly (completeness — a dropped or invented row fails closed)."""
    errors = audit_delta_shape(delta)
    structural = {"epoch_blocks_missing", "transitions_missing", "transitions_malformed"}
    if any(e in structural for e in errors):
        return sorted(set(errors))
    members: dict[str, dict[str, str]] = {}
    for label in ("prev_epoch", "next_epoch"):
        block = delta[label]
        doc, err = _load_epoch_doc(corpus_dir, str(block["receipt"]))
        if err is not None:
            errors.append(f"{label}_{err}")
            continue
        assert doc is not None
        if doc.get("epoch_root_sha256") != block.get("epoch_root_sha256"):
            errors.append(f"{label}_root_mismatch")
            continue
        m = _epoch_members(doc)
        if _member_map_sha(m) != block.get("member_map_sha256"):
            errors.append(f"{label}_map_digest_mismatch")
        if _merkle_root(m) != block.get("merkle_root"):
            errors.append(f"{label}_merkle_mismatch")
        members[label] = m
    if "prev_epoch" not in members or "next_epoch" not in members:
        return sorted(set(errors))
    prev_m, next_m = members["prev_epoch"], members["next_epoch"]
    names = set(prev_m) | set(next_m)
    a_set = {n for n in names if n in next_m and n not in prev_m}
    r_set = {n for n in names if n in prev_m and n not in next_m}
    c_set = {
        n: (prev_m[n], next_m[n])
        for n in names
        if n in prev_m and n in next_m and prev_m[n] != next_m[n]
    }
    unchanged = sum(1 for n in names if n in prev_m and n in next_m and prev_m[n] == next_m[n])
    tr = delta["transitions"]

    def _rnames(rows: list[Any]) -> set[str]:
        return {str(r.get("name")) for r in rows if isinstance(r, dict)}

    if _rnames(tr["added"]) != a_set:
        errors.append("added_set_incomplete")
    if _rnames(tr["removed"]) != r_set:
        errors.append("removed_set_incomplete")
    decl_c = {
        str(r.get("name")): (str(r.get("from_sha256")), str(r.get("to_sha256")))
        for r in tr["changed"]
        if isinstance(r, dict)
    }
    if decl_c != c_set:
        errors.append("changed_set_incomplete")
    if tr.get("unchanged_count") != unchanged:
        errors.append("unchanged_count_mismatch")
    return sorted(set(errors))


def _chain_leaf(position: int, name: str, sha256: str) -> bytes:
    """Position-bound chain leaf — mirrors epoch_merkle._chain_leaf."""
    return hashlib.sha256(
        b"\x00" + _canon({"position": position, "receipt": name, "sha256": sha256})
    ).digest()


def _chain_root(chain: list[tuple[str, str]]) -> str:
    """Root over ordered chain leaves — mirrors chain_tree_root."""
    level = [_chain_leaf(i, n, s) for i, (n, s) in enumerate(chain)]
    while len(level) > 1:
        nxt = [_node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def _expected_sides(leaf_index: int, n_members: int) -> list[str | None]:
    """Per-level sibling side for a leaf index under odd-promotion —
    mirrors epoch_merkle._expected_sides (pure function of tree shape)."""
    sides: list[str | None] = []
    pos, ln = leaf_index, n_members
    while ln > 1:
        if pos == ln - 1 and ln % 2 == 1:
            sides.append(None)
        else:
            sides.append("right" if pos % 2 == 0 else "left")
        pos //= 2
        ln = (ln + 1) // 2
    return sides


def _root_from_leaf_path(leaf: bytes, leaf_index: int, path: list[Any]) -> bytes:
    node, pos = leaf, leaf_index
    for entry in path:
        sib = bytes.fromhex(entry["sha256"])
        node = _node_hash(node, sib) if entry["side"] == "right" else _node_hash(sib, node)
        pos //= 2
    return node


def audit_position_shape(proof: dict[str, Any]) -> list[str]:
    """epoch_position.v1 shape — mirrors epoch_position_errors: stamps,
    position bounds, digest shapes, and the position-bound path replay."""
    errors: list[str] = []
    if proof.get("kind") != "epoch_position.v1":
        errors.append("kind_not_epoch_position")
    if proof.get("schema") != "epoch_position.v1":
        errors.append("schema_not_epoch_position")
    if proof.get("research_only") is not True:
        errors.append("research_only_not_true")
    if proof.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    receipt = proof.get("receipt")
    if not isinstance(receipt, str) or not receipt.startswith("corpus_epoch_"):
        errors.append("receipt_name_bad")
    if not _digest_hex(proof.get("sha256")):
        errors.append("sha256_bad")
    pos, n = proof.get("position"), proof.get("n_epochs")
    if not isinstance(pos, int) or not isinstance(n, int) or not (0 <= pos < n):
        errors.append("position_bounds")
    if not _digest_hex(proof.get("chain_root")):
        errors.append("chain_root_bad")
    path = proof.get("path")
    if not isinstance(path, list) or not all(
        isinstance(e, dict) and e.get("side") in ("left", "right") for e in path
    ):
        errors.append("path_malformed")
    if errors:
        return sorted(set(errors))
    # Path replay: leaf binds (position, name, sha); shape pinned by (pos, n).
    sides = _expected_sides(pos, n)
    if len(path) != sum(1 for s in sides if s is not None):
        return sorted({*errors, "path_invalid"})
    it = iter(path)
    for want in sides:
        if want is None:
            continue
        step = next(it)
        if step.get("side") != want:
            return sorted({*errors, "path_invalid"})
    leaf = _chain_leaf(pos, str(receipt), str(proof["sha256"]))
    if _root_from_leaf_path(leaf, pos, path).hex() != proof["chain_root"]:
        errors.append("path_invalid")
    return sorted(set(errors))


def audit_position_pin(proof: dict[str, Any], pin: dict[str, Any], key: str | None) -> list[str]:
    """Pin-mode epoch_position.v1 — the entry's chain_root + n_epochs
    anchor the proof; a head-position proof must match the pinned head."""
    errors = audit_position_shape(proof)
    if errors:
        return errors
    heads = pin.get("heads", {})
    corpus_key = key or str(proof.get("corpus_key") or "")
    entry = heads.get(corpus_key)
    if not isinstance(entry, dict):
        entry = next(
            (
                e
                for e in heads.values()
                if isinstance(e, dict) and e.get("chain_root") == proof["chain_root"]
            ),
            None,
        )
    if not isinstance(entry, dict):
        return [*errors, "pin_entry_missing"]
    if entry.get("chain_root") != proof["chain_root"]:
        errors.append("chain_root_not_pinned")
    if isinstance(entry.get("n_epochs"), int) and entry["n_epochs"] != proof["n_epochs"]:
        errors.append("pin_n_epochs_mismatch")
    if proof["position"] == proof["n_epochs"] - 1:
        if entry.get("receipt") != proof["receipt"]:
            errors.append("head_receipt_not_pinned")
        if entry.get("sha256") != proof["sha256"]:
            errors.append("head_sha256_not_pinned")
    return sorted(set(errors))


def audit_position_live(proof: dict[str, Any], corpus_dir: Path) -> list[str]:
    """Live replay — rebuild the chain and require declared position,
    digest, and root to all recompute (mirrors verify_epoch_position)."""
    errors = audit_position_shape(proof)
    if errors:
        return errors
    key = proof.get("corpus_key")
    pattern = key.rsplit("/", 1)[-1] if isinstance(key, str) and "/" in key else "*.json"
    ordered, chain_errors = _ordered_chain(corpus_dir, pattern)
    if chain_errors:
        return [*errors, *[f"chain_unverifiable:{e}" for e in chain_errors]]
    names_sha = [(n, sha) for n, sha, _, _, _ in ordered]
    if len(names_sha) != proof["n_epochs"]:
        errors.append("n_epochs_mismatch")
    if _chain_root(names_sha) != proof["chain_root"]:
        errors.append("chain_root_mismatch")
    pos = int(proof["position"])
    if pos >= len(names_sha):
        return sorted(set(errors + ["position_out_of_range"]))
    name, sha = names_sha[pos]
    if name != proof["receipt"]:
        errors.append("position_name_mismatch")
    if sha != proof["sha256"]:
        errors.append("position_sha256_mismatch")
    if ordered[pos][2] != proof.get("epoch_root_sha256"):
        errors.append("epoch_root_mismatch")
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
    ap.add_argument(
        "--consistency",
        type=Path,
        default=None,
        help="epoch_consistency.v1 proof (extension of a held head to the current head)",
    )
    ap.add_argument(
        "--delta",
        type=Path,
        default=None,
        help="epoch_delta.v1 receipt (completeness-verified epoch change-set)",
    )
    ap.add_argument(
        "--position",
        type=Path,
        default=None,
        help="epoch_position.v1 receipt (chain-position proof under the pin's chain_root)",
    )
    ap.add_argument(
        "--held",
        type=str,
        default=None,
        help="64-hex digest of the from-head bytes you already trust "
        "(consistency proofs only — binds the proof to that exact state)",
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
    ap.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="live corpus dir — replays the proof against real epoch receipts "
        "(combinable with --pin/--checkpoint for layered assurance)",
    )
    args = ap.parse_args()
    n_targets = sum(
        x is not None
        for x in (
            args.proof,
            args.absence,
            args.history_absence,
            args.consistency,
            args.delta,
            args.position,
        )
    )
    if n_targets != 1:
        print(
            "pass exactly one of --proof/--absence/--history-absence/"
            "--consistency/--delta/--position",
            file=sys.stderr,
        )
        return 2
    if args.held is not None and not _digest_hex(args.held):
        print("--held must be 64 lowercase hex", file=sys.stderr)
        return 2
    if args.held is not None and args.corpus_dir is None:
        print("--held binds against a live corpus — pass --corpus-dir", file=sys.stderr)
        return 2
    if args.pin is None and args.checkpoint is None and args.corpus_dir is None:
        print("pass --pin, --checkpoint, and/or --corpus-dir", file=sys.stderr)
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
    live: dict[str, Any] = {}
    if args.corpus_dir is not None:
        live["dir"] = args.corpus_dir
    pinned = args.pin is not None or args.checkpoint is not None
    if args.proof is not None:
        payload = json.loads(args.proof.read_text())
        # sealed receipts may wrap the body under 'payload'
        body = payload.get("payload", payload)
        errors = pre_errors
        if pinned:
            errors += audit_proof(body, pin, args.key)
        if live:
            errors += audit_proof_live(body, live["dir"])
        print(f"inclusion member={body.get('member')} -> {len(errors)} error(s)")
    elif args.consistency is not None:
        payload = json.loads(args.consistency.read_text())
        body = payload.get("payload", payload)
        errors = pre_errors
        if pinned:
            errors += audit_consistency(body, pin, args.key)
        if live:
            errors += audit_consistency_live(body, live["dir"], held_sha256=args.held)
        elif not pinned:
            # --corpus-dir is the only source that binds hops to real bytes;
            # without it, shape alone still runs for completeness.
            errors += audit_consistency_shape(body)
        print(
            f"consistency hops={body.get('n_hops')} "
            f"to={body.get('to_receipt', {}).get('name')} -> {len(errors)} error(s)"
        )
    elif args.history_absence is not None:
        payload = json.loads(args.history_absence.read_text())
        body = payload.get("payload", payload)
        errors = pre_errors
        if pinned:
            errors += audit_history_absence(body, pin, args.key)
        if live:
            errors += audit_history_absence_live(body, live["dir"])
        print(
            f"history-absence name={body.get('name')} "
            f"epochs={body.get('n_epochs')} -> {len(errors)} error(s)"
        )
    elif args.position is not None:
        payload = json.loads(args.position.read_text())
        body = payload.get("payload", payload)
        errors = pre_errors
        if pinned:
            errors += audit_position_pin(body, pin, args.key)
        if live:
            errors += audit_position_live(body, live["dir"])
        elif not pinned:
            errors += audit_position_shape(body)
        print(
            f"position receipt={body.get('receipt')} "
            f"pos={body.get('position')}/{body.get('n_epochs')} -> {len(errors)} error(s)"
        )
    elif args.delta is not None:
        payload = json.loads(args.delta.read_text())
        body = payload.get("payload", payload)
        errors = pre_errors
        if live:
            errors += audit_delta_live(body, live["dir"])
        elif not pinned:
            errors += audit_delta_shape(body)
        else:
            # Pin mode binds only the head — a delta's prev epoch may not
            # be the current head; still run the self-contained shape pass.
            errors += audit_delta_shape(body)
        print(
            f"delta prev={body.get('prev_epoch', {}).get('receipt')} "
            f"next={body.get('next_epoch', {}).get('receipt')} -> {len(errors)} error(s)"
        )
    else:
        payload = json.loads(args.absence.read_text())
        # sealed corpus_absence.v1 receipts wrap the body under 'payload';
        # raw library absence_proof() dicts pass through unchanged.
        body = payload.get("payload", payload)
        errors = pre_errors
        if pinned:
            errors += audit_absence(body, pin, args.key)
        if live:
            errors += audit_absence_live(body, live["dir"])
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
