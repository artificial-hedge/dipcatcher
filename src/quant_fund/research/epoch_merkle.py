"""Merkle inclusion proofs over a corpus epoch — Certificate-Transparency
style evidence for ``corpus_epoch``.

An epoch receipt carries the full member map, which makes membership
verification trivial *if you trust the receipt*. The problem the proof
solves: answering "was this exact file in the corpus at epoch N?" should
not require shipping around the whole epoch receipt's member list — a
proof is O(log n) and self-contained. It also gives a *succinct* object
the verifier can pin to an epoch root that is itself chain-verified by
``corpus-epoch --check`` and head-pinned by ``quality/epoch_heads.json``.

Construction (RFC 6962 discipline):

- leaves are ``sha256(0x00 || canonical_json({"name", "sha256"}))`` —
  domain-separated from interior nodes so a leaf can never pose as a
  subtree root (second-preimage safety);
- interior nodes are ``sha256(0x01 || left || right)`` over the sorted
  name order — the same deterministic member order ``epoch_root_sha256``
  commits to;
- an odd remainder node is **promoted** (moved up unchanged), never
  duplicated — duplicating would admit a second-preimage against the
  same root.

A proof is ``{member, sha256, leaf_index, n_members, path: [{sha256,
side}], epoch_receipt, epoch_root_sha256}`` sealed as a
``corpus_proof.v1`` receipt. ``verify_epoch_proof`` recomputes the root
from the path *and* re-derives it from the referenced epoch receipt —
both must agree with the declared root, so the proof binds to the real
chained state and a fabricated path cannot anchor to a different epoch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes

CORPUS_PROOF_SCHEMA = "corpus_proof.v1"

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def _leaf_hash(name: str, sha256: str) -> bytes:
    body = canonical_json_bytes({"name": name, "sha256": sha256})
    return hashlib.sha256(_LEAF_PREFIX + body).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


def _leaves(members: Mapping[str, str]) -> list[tuple[str, bytes]]:
    return [(name, _leaf_hash(name, sha)) for name, sha in sorted(members.items())]


def merkle_root(members: Mapping[str, str]) -> str:
    """Domain-separated Merkle root over ``{name: sha256}``."""
    if not members:
        raise ValueError("empty member map has no merkle root")
    level = [leaf for _, leaf in _leaves(members)]
    while len(level) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_node_hash(level[i], level[i + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])  # promote, don't duplicate
        level = nxt
    return level[0].hex()


def inclusion_proof(members: Mapping[str, str], name: str) -> dict[str, Any]:
    """Sibling path proving ``name -> sha256`` sits in the tree."""
    if name not in members:
        raise ValueError(f"not a corpus member: {name}")
    items = _leaves(members)
    idx = next(i for i, (n, _) in enumerate(items) if n == name)
    level = [leaf for _, leaf in items]
    path: list[dict[str, str]] = []
    pos = idx
    while len(level) > 1:
        if pos % 2 == 0:
            sibling = pos + 1
            side = "right"
        else:
            sibling = pos - 1
            side = "left"
        if sibling < len(level):
            path.append({"sha256": level[sibling].hex(), "side": side})
        nxt = [_node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
        pos //= 2
    return {
        "leaf_index": idx,
        "n_members": len(items),
        "path": path,
    }


def _root_from_proof(leaf: bytes, leaf_index: int, path: list[Mapping[str, str]]) -> bytes:
    node = leaf
    pos = leaf_index
    for entry in path:
        sib = bytes.fromhex(entry["sha256"])
        if len(sib) != 32:
            raise ValueError("path sibling not sha256")
        node = _node_hash(node, sib) if entry["side"] == "right" else _node_hash(sib, node)
        pos //= 2
    return node


def verify_inclusion(
    name: str,
    sha256: str,
    proof: Mapping[str, Any],
    expected_root: str,
) -> bool:
    """True iff the path recomputes ``expected_root`` from the leaf."""
    try:
        leaf = _leaf_hash(name, sha256)
        idx = proof["leaf_index"]
        path = proof["path"]
        if not isinstance(idx, int) or not isinstance(path, list):
            return False
        return _root_from_proof(leaf, idx, path).hex() == expected_root
    except (KeyError, TypeError, ValueError):
        return False


def _load_epoch_receipt(corpus_dir: Path, name: str) -> dict[str, Any] | None:
    path = corpus_dir / name
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def member_proof(
    corpus_dir: Path | str,
    member: str,
    *,
    epoch_receipt: str | None = None,
) -> dict[str, Any]:
    """Build a ``corpus_proof.v1`` body binding ``member`` to the chain head
    (or ``epoch_receipt`` explicitly)."""
    from quant_fund.research.corpus_epoch import _epoch_receipts

    root = Path(corpus_dir)
    epochs = [(p, e) for p, e in _epoch_receipts(root)]
    if not epochs:
        raise ValueError(f"no corpus epochs under {root} — run corpus-epoch first")
    if epoch_receipt is None:
        # Chain head = the epoch no other epoch names as prev. Filenames are
        # digest-derived — lexical order is not chain order.
        prevs = {e.get("prev_epoch_receipt") for _, e in epochs}
        heads = [(p, e) for p, e in epochs if p.name not in prevs]
        epoch_path, epoch_payload = sorted(heads, key=lambda t: t[0].name)[-1]
    else:
        hit = [(p, e) for p, e in epochs if p.name == epoch_receipt]
        if not hit:
            raise ValueError(f"epoch receipt {epoch_receipt} not found under {root}")
        epoch_path, epoch_payload = hit[0]
    members_list = epoch_payload.get("members")
    if not isinstance(members_list, list):
        raise ValueError(f"{epoch_path.name} has no member map")
    members = {
        m["name"]: m["sha256"]
        for m in members_list
        if isinstance(m, Mapping) and "name" in m and "sha256" in m
    }
    proof = inclusion_proof(members, member)
    return {
        "kind": CORPUS_PROOF_SCHEMA,
        "schema": CORPUS_PROOF_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "member": member,
        "member_sha256": members[member],
        "epoch_receipt": epoch_path.name,
        "epoch_root_sha256": epoch_payload.get("epoch_root_sha256"),
        "merkle_root": merkle_root(members),
        "leaf_index": proof["leaf_index"],
        "n_members": proof["n_members"],
        "path": proof["path"],
    }


def corpus_proof_errors(payload: Mapping[str, Any]) -> list[str]:
    """``corpus_proof.v1`` internal consistency; ``[]`` when clean."""
    errors: list[str] = []
    if payload.get("kind") != CORPUS_PROOF_SCHEMA:
        errors.append("kind_not_corpus_proof")
    if payload.get("schema") != CORPUS_PROOF_SCHEMA:
        errors.append("schema_not_corpus_proof")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    member = payload.get("member")
    sha = payload.get("member_sha256")
    if not isinstance(member, str) or not member:
        errors.append("member_not_str")
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
        errors.append("member_sha256_not_hex")
    root = payload.get("epoch_root_sha256")
    if not (isinstance(root, str) and len(root) == 64):
        errors.append("epoch_root_sha256")
    merkle = payload.get("merkle_root")
    if not (isinstance(merkle, str) and len(merkle) == 64):
        errors.append("merkle_root")
    idx = payload.get("leaf_index")
    n = payload.get("n_members")
    path = payload.get("path")
    if not isinstance(idx, int) or not isinstance(n, int) or idx < 0 or idx >= n:
        errors.append("leaf_index_bounds")
    if not isinstance(path, list) or not all(
        isinstance(e, Mapping) and e.get("side") in ("left", "right") for e in path
    ):
        errors.append("path_malformed")
    elif (
        isinstance(member, str)
        and isinstance(sha, str)
        and isinstance(merkle, str)
        and not verify_inclusion(member, sha, payload, merkle)
    ):
        # The path must recompute the declared merkle_root from the leaf.
        errors.append("proof_path_invalid")
    return errors


def verify_epoch_proof(
    payload: Mapping[str, Any],
    corpus_dir: Path | str,
) -> list[str]:
    """Deep check: the proof must anchor to a real chained epoch receipt.

    Recomputes the root two ways — from the proof path and from the named
    epoch receipt's member map — and requires both to equal the declared
    ``epoch_root_sha256``/``merkle_root``.
    """
    errors = corpus_proof_errors(payload)
    if errors:
        return errors
    root = Path(corpus_dir)
    receipt = _load_epoch_receipt(root, str(payload["epoch_receipt"]))
    if receipt is None:
        return ["epoch_receipt_missing"]
    if receipt.get("epoch_root_sha256") != payload["epoch_root_sha256"]:
        return ["epoch_root_mismatch"]
    members_list = receipt.get("members")
    members = {
        m["name"]: m["sha256"]
        for m in members_list or []
        if isinstance(m, Mapping) and "name" in m and "sha256" in m
    }
    member = str(payload["member"])
    if members.get(member) != payload["member_sha256"]:
        return ["member_digest_mismatch"]
    if merkle_root(members) != payload["merkle_root"]:
        return ["merkle_root_mismatch"]
    return []
