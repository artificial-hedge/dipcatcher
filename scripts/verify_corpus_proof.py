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
   carry an inclusion path, and their leaf_indexes must be adjacent
   (hi == lo + 1). A single bound must sit on an edge (index 0 or n-1).

Exit 0 only when every check passes. Prints one verdict line per layer.
This file must never import the library — it is the differential oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


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
    if len(path) != len(expected):
        errors.append("path_depth_mismatch")
    else:
        for entry, want_side in zip(path, expected, strict=True):
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
    if not isinstance(root, str) or not isinstance(bounds, list):
        return ["absence_shape_malformed"]
    receipt = proof.get("epoch_receipt")
    if isinstance(receipt, str):
        _k, entry, err = _pin_entry(pin, key, receipt)
        if err:
            return [err]
    elif key is not None:
        # library absence proofs carry no epoch_receipt — bind by tree_root
        entry = (pin.get("heads") or {}).get(key)
        if not isinstance(entry, dict):
            return [f"pin_key_absent:{key}"]
    else:
        heads = pin.get("heads", {})
        matches = [v for v in heads.values() if isinstance(v, dict) and v.get("tree_root") == root]
        if len(matches) != 1:
            return [f"tree_root_not_pinned:{len(matches)}_matches"]
        entry = matches[0]
    if not isinstance(entry, dict):
        return [f"pin_key_absent:{key}"]
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
        sub = _verify_inclusion(bname, str(b.get("member_sha256", "")), b, root)
        if sub:
            errors += [f"bound_invalid:{bname}"]
            continue
        idxs.append(int(b["leaf_index"]))
        if not (bname < name or bname > name):
            errors.append("bound_self")
    if errors:
        return sorted(set(errors))
    if len(idxs) == 2 and idxs[1] - idxs[0] != 1:
        errors.append("bounds_not_adjacent")
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proof", type=Path, default=None, help="corpus_proof.v1 receipt")
    ap.add_argument("--absence", type=Path, default=None, help="absence proof JSON")
    ap.add_argument("--pin", type=Path, required=True, help="epoch_heads.json")
    ap.add_argument("--key", type=str, default=None, help="pin key e.g. 'receipts/*.json'")
    args = ap.parse_args()
    if (args.proof is None) == (args.absence is None):
        print("pass exactly one of --proof/--absence", file=sys.stderr)
        return 2
    pin = json.loads(args.pin.read_text())
    if args.proof is not None:
        payload = json.loads(args.proof.read_text())
        # sealed receipts may wrap the body under 'payload'
        body = payload.get("payload", payload)
        errors = audit_proof(body, pin, args.key)
        print(f"inclusion member={body.get('member')} -> {len(errors)} error(s)")
    else:
        body = json.loads(args.absence.read_text())
        errors = audit_absence(body, pin, args.key)
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
