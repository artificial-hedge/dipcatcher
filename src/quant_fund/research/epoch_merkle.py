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
CORPUS_ABSENCE_SCHEMA = "corpus_absence.v1"

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


def _expected_sides(leaf_index: int, n_members: int) -> list[str | None]:
    """Per-level sibling side for a leaf — ``None`` on promotion levels.

    The tree shape is a pure function of ``n_members`` (odd tails promote),
    so ``leaf_index`` fully determines the path's side sequence. Enforcing
    it binds ``leaf_index`` to the path: a forged index produces a
    differently-shaped path that cannot recompute the same root — which is
    what makes adjacency claims in absence proofs unfakeable.
    """
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
    """True iff the path recomputes ``expected_root`` from the leaf.

    When ``proof`` carries ``n_members``, the path shape must equal the
    shape ``(leaf_index, n_members)`` implies — a path for index i can no
    longer be re-presented as belonging to index j.
    """
    try:
        leaf = _leaf_hash(name, sha256)
        idx = proof["leaf_index"]
        path = proof["path"]
        if not isinstance(idx, int) or not isinstance(path, list):
            return False
        n_members = proof.get("n_members")
        if n_members is not None:
            if not isinstance(n_members, int) or not (0 <= idx < n_members):
                return False
            sides = _expected_sides(idx, n_members)
            if len(path) != sum(1 for s in sides if s is not None):
                return False
            it = iter(path)
            for want in sides:
                if want is None:
                    continue
                step = next(it)
                if step.get("side") != want:
                    return False
        return _root_from_proof(leaf, idx, path).hex() == expected_root
    except (KeyError, TypeError, ValueError, StopIteration):
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
    from quant_fund.research.corpus_epoch import epoch_heads_key

    epoch_path, epoch_payload, members = _epoch_members(corpus_dir, epoch_receipt)
    proof = inclusion_proof(members, member)
    return {
        "kind": CORPUS_PROOF_SCHEMA,
        "schema": CORPUS_PROOF_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "corpus_key": epoch_heads_key(
            Path(corpus_dir),
            str((epoch_payload.get("params") or {}).get("pattern") or "*.json"),
        ),
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
    corpus_key = payload.get("corpus_key")
    if corpus_key is not None and not isinstance(corpus_key, str):
        errors.append("corpus_key_not_str")
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
    # n_members only shapes the path check — shape-equivalent forgeries
    # (e.g. n+1 on a leaf whose side sequence is unchanged) are invisible
    # to it, so pin the count to the epoch's authoritative member map.
    if payload["n_members"] != len(members):
        return ["n_members_mismatch"]
    if merkle_root(members) != payload["merkle_root"]:
        return ["merkle_root_mismatch"]
    return []


def verify_proof_pin(
    payload: Mapping[str, Any],
    pin_entry: Mapping[str, Any],
) -> list[str]:
    """Offline verification against a heads-pin entry — no corpus access.

    ``pin_entry`` is ``epoch_heads.json["heads"][key]``: the quorum-signed,
    OTS-anchored root of trust. The proof must name the pinned epoch
    receipt, carry the pinned ``tree_root``, and recompute it from the
    leaf+path. A fabricated path cannot anchor to a pin it doesn't hold.
    """
    errors = corpus_proof_errors(payload)
    if errors:
        return errors
    if payload.get("epoch_receipt") != pin_entry.get("receipt"):
        errors.append("epoch_receipt_not_pinned")
    pinned_root = pin_entry.get("tree_root")
    if not isinstance(pinned_root, str):
        errors.append("pin_tree_root_absent")
    elif payload.get("merkle_root") != pinned_root:
        errors.append("merkle_root_not_pinned")
    return errors


def absence_proof(members: Mapping[str, str], name: str) -> dict[str, Any]:
    """Proof that ``name`` is NOT committed in the sorted member tree.

    Emits the two name-sorted neighbors bracketing the gap, each with its
    own inclusion path. ``leaf_index`` is cryptographically bound — the
    path recomputes the root only at its true position — so adjacent
    ``leaf_index`` values (``hi == lo + 1``) prove no member can sit
    between them. Edge names get a single bound; an empty corpus proves
    absence with no bounds.
    """
    if name in members:
        raise ValueError(f"{name!r} is a member — use inclusion_proof")
    names = sorted(members)
    lo = next((n for n in reversed(names) if n < name), None)
    hi = next((n for n in names if n > name), None)
    bounds: list[dict[str, Any]] = []
    for n in (b for b in (lo, hi) if b is not None):
        p = inclusion_proof(members, n)
        bounds.append({"member": n, "member_sha256": members[n], **p})
    return {
        "name": name,
        "merkle_root": merkle_root(members),
        "n_members": len(names),
        "bounds": bounds,
    }


def verify_absence(proof: Mapping[str, Any], expected_root: str) -> list[str]:
    """Verify a bounding-pair absence proof against a trusted root."""
    errors: list[str] = []
    name = proof.get("name")
    bounds = proof.get("bounds")
    if not isinstance(name, str) or not name:
        return ["name_missing"]
    if not isinstance(bounds, list):
        return ["bounds_missing"]
    if len(bounds) > 2:
        # Absence is proven by the tightest bracketing pair — a wider bound
        # set is malformed, not stronger.
        return ["bounds_len"]
    if proof.get("merkle_root") != expected_root:
        errors.append("merkle_root_mismatch")
    if proof.get("n_members") == 0 and bounds:
        errors.append("empty_tree_with_bounds")
    idxs: list[int] = []
    for b in bounds:
        if not isinstance(b, Mapping):
            return ["bound_malformed"]
        bname = b.get("member")
        if not isinstance(bname, str) or bname == name:
            return ["bound_not_neighbor"]
        if not verify_inclusion(bname, str(b.get("member_sha256", "")), b, expected_root):
            errors.append(f"bound_invalid:{bname}")
            continue
        idxs.append(int(b["leaf_index"]))
        if not (bname < name or bname > name):
            errors.append("bound_self")
    if errors:
        return errors
    if len(idxs) == 2 and idxs[1] - idxs[0] != 1:
        errors.append("bounds_not_adjacent")
    # Adjacent indexes alone don't bracket the name — without this check, a
    # pin-mode verifier (no member map) would accept bounds anywhere in the
    # tree as "absence" for a member that sits outside them.
    if len(idxs) == 2 and not (str(bounds[0]["member"]) < name < str(bounds[1]["member"])):
        errors.append("bounds_not_bracketing")
    if len(idxs) == 1 and isinstance(proof.get("n_members"), int) and int(proof["n_members"]) > 1:
        b = bounds[0]
        bname = str(b["member"])
        edge_ok = (bname < name and int(b["leaf_index"]) == int(proof["n_members"]) - 1) or (
            bname > name and int(b["leaf_index"]) == 0
        )
        if not edge_ok:
            errors.append("single_bound_not_edge")
    return sorted(set(errors))


def _epoch_members(
    corpus_dir: Path | str, epoch_receipt: str | None
) -> tuple[Path, Mapping[str, Any], dict[str, str]]:
    """Resolve an epoch receipt (chain head by default) → (path, payload,
    name→sha256 member map)."""
    from quant_fund.research.corpus_epoch import _epoch_receipts

    root = Path(corpus_dir)
    epochs = [(p, e) for p, e in _epoch_receipts(root)]
    if not epochs:
        raise ValueError(f"no corpus epochs under {root} — run corpus-epoch first")
    if epoch_receipt is None:
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
    members: dict[str, str] = {
        str(m["name"]): str(m["sha256"])
        for m in members_list
        if isinstance(m, Mapping) and "name" in m and "sha256" in m
    }
    return epoch_path, epoch_payload, members


def absence_receipt(
    corpus_dir: Path | str,
    name: str,
    *,
    epoch_receipt: str | None = None,
) -> dict[str, Any]:
    """Build a ``corpus_absence.v1`` body: cryptographic proof that ``name``
    was NOT a member at the bound epoch — the two sorted-name neighbors
    bracketing the gap, each with a shape-bound inclusion path."""
    from quant_fund.research.corpus_epoch import epoch_heads_key

    epoch_path, epoch_payload, members = _epoch_members(corpus_dir, epoch_receipt)
    proof = absence_proof(members, name)
    return {
        "kind": CORPUS_ABSENCE_SCHEMA,
        "schema": CORPUS_ABSENCE_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "corpus_key": epoch_heads_key(
            Path(corpus_dir),
            str((epoch_payload.get("params") or {}).get("pattern") or "*.json"),
        ),
        "name": name,
        "epoch_receipt": epoch_path.name,
        "epoch_root_sha256": epoch_payload.get("epoch_root_sha256"),
        "merkle_root": proof["merkle_root"],
        "n_members": proof["n_members"],
        "bounds": proof["bounds"],
    }


def corpus_absence_errors(payload: Mapping[str, Any]) -> list[str]:
    """``corpus_absence.v1`` internal consistency; ``[]`` when clean.

    Bounds must recompute the declared ``merkle_root`` and bracket ``name``
    adjacently — the soundness anchor (that root is the real corpus's) is
    applied by ``verify_epoch_absence``/``verify_absence_pin``.
    """
    errors: list[str] = []
    if payload.get("kind") != CORPUS_ABSENCE_SCHEMA:
        errors.append("kind_not_corpus_absence")
    if payload.get("schema") != CORPUS_ABSENCE_SCHEMA:
        errors.append("schema_not_corpus_absence")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    corpus_key = payload.get("corpus_key")
    if corpus_key is not None and not isinstance(corpus_key, str):
        errors.append("corpus_key_not_str")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        errors.append("name_not_str")
    root = payload.get("epoch_root_sha256")
    if not (isinstance(root, str) and len(root) == 64):
        errors.append("epoch_root_sha256")
    merkle = payload.get("merkle_root")
    if not (isinstance(merkle, str) and len(merkle) == 64):
        errors.append("merkle_root")
    elif isinstance(merkle, str):
        errors += verify_absence(payload, merkle)
    return sorted(set(errors))


def verify_epoch_absence(
    payload: Mapping[str, Any],
    corpus_dir: Path | str,
) -> list[str]:
    """Deep check: an absence receipt must anchor to a real chained epoch —
    the epoch's member map must recompute ``merkle_root`` and NOT contain
    ``name``."""
    errors = corpus_absence_errors(payload)
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
    if payload["name"] in members:
        return ["name_is_member"]
    if payload["n_members"] != len(members):
        return ["n_members_mismatch"]
    if merkle_root(members) != payload["merkle_root"]:
        return ["merkle_root_mismatch"]
    return []


def verify_absence_pin(
    payload: Mapping[str, Any],
    pin_entry: Mapping[str, Any],
) -> list[str]:
    """Offline absence verification against a heads-pin entry — the pin's
    ``tree_root`` is the trusted root the bounds must recompute."""
    errors = corpus_absence_errors(payload)
    if errors:
        return errors
    if payload.get("epoch_receipt") != pin_entry.get("receipt"):
        errors.append("epoch_receipt_not_pinned")
    pinned_root = pin_entry.get("tree_root")
    if not isinstance(pinned_root, str):
        errors.append("pin_tree_root_absent")
    elif payload.get("merkle_root") != pinned_root:
        errors.append("merkle_root_not_pinned")
    return sorted(set(errors))
