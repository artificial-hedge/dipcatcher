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
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

CORPUS_PROOF_SCHEMA = "corpus_proof.v1"
CORPUS_ABSENCE_SCHEMA = "corpus_absence.v1"

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"

_PORTABLE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


def _portable(name: str) -> bool:
    """No control/format/line-separator chars — they inject fake lines into
    verifier output and bidi overrides visually rename members."""
    return all(unicodedata.category(c) not in _PORTABLE_CATEGORIES for c in name)


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
    if unicodedata.normalize("NFC", name) != name:
        raise ValueError(f"member name must be NFC-canonical: {name!r}")
    if not _portable(name):
        raise ValueError(f"member name must be portable: {name!a}")
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
        if unicodedata.normalize("NFC", name) != name or not _portable(name):
            return False
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


def _chain_leaf(position: int, name: str, sha256: str) -> bytes:
    """Leaf for the epoch-chain tree: binds (position, receipt name, file
    digest) — the index lives *inside* the leaf, so a path can never be
    re-presented at a different position."""
    body = canonical_json_bytes({"position": position, "receipt": name, "sha256": sha256})
    return hashlib.sha256(_LEAF_PREFIX + body).digest()


def chain_tree_root(chain: list[tuple[str, str]]) -> str:
    """RFC 6962 root over the ordered epoch chain ``[(name, file_sha), ...]``.

    Unlike ``merkle_root`` the leaves are kept in chain position order —
    an append-only transcript, exactly like a CT log, so the root commits
    to *sequence*, not just membership.
    """
    if not chain:
        raise ValueError("empty epoch chain has no root")
    level = [_chain_leaf(i, n, s) for i, (n, s) in enumerate(chain)]
    while len(level) > 1:
        nxt = [_node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def chain_position_proof(chain: list[tuple[str, str]], index: int) -> dict[str, Any]:
    """Sibling path proving ``chain[index]`` sits at position ``index``."""
    if not 0 <= index < len(chain):
        raise ValueError(f"position out of range: {index}")
    level = [_chain_leaf(i, n, s) for i, (n, s) in enumerate(chain)]
    path: list[dict[str, str]] = []
    pos = index
    while len(level) > 1:
        sibling = pos + 1 if pos % 2 == 0 else pos - 1
        side = "right" if pos % 2 == 0 else "left"
        if sibling < len(level):
            path.append({"sha256": level[sibling].hex(), "side": side})
        nxt = [_node_hash(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
        pos //= 2
    return {"leaf_index": index, "n_members": len(chain), "path": path}


def verify_chain_position(
    position: int,
    name: str,
    sha256: str,
    proof: Mapping[str, Any],
    expected_root: str,
) -> bool:
    """True iff the path recomputes ``expected_root`` from the position-bound
    leaf — with ``n_members`` carried, the path shape is pinned by
    ``(position, n)`` so indices can't be swapped."""
    try:
        if unicodedata.normalize("NFC", name) != name or not _portable(name):
            return False
        leaf = _chain_leaf(position, name, sha256)
        idx = proof["leaf_index"]
        path = proof["path"]
        if not isinstance(idx, int) or not isinstance(path, list):
            return False
        if idx != position:
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


EPOCH_POSITION_SCHEMA = "epoch_position.v1"


def epoch_position_receipt(
    corpus_dir: Path | str,
    receipt_name: str,
    *,
    pattern: str = "*.json",
) -> dict[str, Any]:
    """Build an ``epoch_position.v1`` body: proof that ``receipt_name``
    occupied a specific position in the corpus's committed epoch chain.

    An auditor holding the quorum-signed heads pin (``chain_root`` +
    ``n_epochs`` per corpus key) verifies this proof in O(log n) — the
    "was this exact corpus state ever committed, and where" question
    without re-walking the chain.
    """
    from quant_fund.research.corpus_epoch import epoch_heads_key

    root = Path(corpus_dir)
    ordered, chain_errors = ordered_epoch_chain(root, pattern=pattern)
    if chain_errors:
        raise ValueError(f"chain unverifiable: {chain_errors}")
    names_sha = [(name, file_sha) for name, file_sha, _, _, _ in ordered]
    positions = {name: i for i, (name, _) in enumerate(names_sha)}
    if receipt_name not in positions:
        raise ValueError(f"not a committed epoch receipt: {receipt_name}")
    idx = positions[receipt_name]
    proof = chain_position_proof(names_sha, idx)
    return {
        "kind": EPOCH_POSITION_SCHEMA,
        "schema": EPOCH_POSITION_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "corpus_key": epoch_heads_key(root, pattern),
        "receipt": receipt_name,
        "sha256": names_sha[idx][1],
        "position": idx,
        "n_epochs": len(names_sha),
        "chain_root": chain_tree_root(names_sha),
        "epoch_root_sha256": ordered[idx][2],
        "path": proof["path"],
    }


def epoch_position_errors(payload: Mapping[str, Any]) -> list[str]:
    """``epoch_position.v1`` self-contained consistency; ``[]`` when clean."""
    errors: list[str] = []
    if payload.get("kind") != EPOCH_POSITION_SCHEMA:
        errors.append("kind_not_epoch_position")
    if payload.get("schema") != EPOCH_POSITION_SCHEMA:
        errors.append("schema_not_epoch_position")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    receipt = payload.get("receipt")
    if not isinstance(receipt, str) or not receipt.startswith("corpus_epoch_"):
        errors.append("receipt_name_bad")
    if not (
        isinstance(payload.get("sha256"), str)
        and len(str(payload.get("sha256"))) == 64
        and all(c in "0123456789abcdef" for c in str(payload.get("sha256")))
    ):
        errors.append("sha256_bad")
    pos = payload.get("position")
    n = payload.get("n_epochs")
    if not isinstance(pos, int) or not isinstance(n, int) or not (0 <= pos < n):
        errors.append("position_bounds")
    if not (
        isinstance(payload.get("chain_root"), str) and len(str(payload.get("chain_root"))) == 64
    ):
        errors.append("chain_root_bad")
    path = payload.get("path")
    if not isinstance(path, list) or not all(
        isinstance(e, Mapping) and e.get("side") in ("left", "right") for e in path
    ):
        errors.append("path_malformed")
    if errors:
        return sorted(set(errors))
    ok = verify_chain_position(
        cast(int, pos),
        str(receipt),
        str(payload["sha256"]),
        {"leaf_index": cast(int, pos), "n_members": cast(int, n), "path": path},
        str(payload["chain_root"]),
    )
    if not ok:
        errors.append("path_invalid")
    return sorted(set(errors))


def verify_epoch_position(payload: Mapping[str, Any], corpus_dir: Path | str) -> list[str]:
    """Live check: rebuild the chain and require the declared position,
    digest, and root to all recompute."""
    errors = epoch_position_errors(payload)
    if errors:
        return errors
    root = Path(corpus_dir)
    key = str(payload.get("corpus_key") or "")
    pattern = key.rsplit("/", 1)[-1] if "/" in key else "*.json"
    ordered, chain_errors = ordered_epoch_chain(root, pattern=pattern)
    if chain_errors:
        return [*errors, *[f"chain_unverifiable:{e}" for e in chain_errors]]
    names_sha = [(name, file_sha) for name, file_sha, _, _, _ in ordered]
    if len(names_sha) != payload["n_epochs"]:
        errors.append("n_epochs_mismatch")
    if chain_tree_root(names_sha) != payload["chain_root"]:
        errors.append("chain_root_mismatch")
    pos = int(payload["position"])
    if pos >= len(names_sha):
        return sorted(set(errors + ["position_out_of_range"]))
    name, sha = names_sha[pos]
    if name != payload["receipt"]:
        errors.append("position_name_mismatch")
    if sha != payload["sha256"]:
        errors.append("position_sha256_mismatch")
    epoch_root = ordered[pos][2]
    if epoch_root != payload.get("epoch_root_sha256"):
        errors.append("epoch_root_mismatch")
    return sorted(set(errors))


def verify_epoch_position_pin(
    payload: Mapping[str, Any], pin_entry: Mapping[str, Any]
) -> list[str]:
    """Offline check against the heads pin: the pin entry's ``chain_root``
    and ``n_epochs`` anchor the proof — ``{receipt, sha256, tree_root,
    chain_root, n_epochs}``."""
    errors = epoch_position_errors(payload)
    if errors:
        return errors
    if not isinstance(pin_entry, Mapping) or not pin_entry:
        return [*errors, "pin_entry_missing"]
    pin_root = pin_entry.get("chain_root")
    if not isinstance(pin_root, str) or len(pin_root) != 64:
        return [*errors, "pin_has_no_chain_root"]
    if pin_root != payload["chain_root"]:
        errors.append("chain_root_not_pinned")
    pin_n = pin_entry.get("n_epochs")
    if isinstance(pin_n, int) and pin_n != payload["n_epochs"]:
        errors.append("pin_n_epochs_mismatch")
    # If the proven epoch claims to be the head, the pin's head fields must
    # agree — a head-position proof for a stale head is worthless.
    if payload["position"] == payload["n_epochs"] - 1:
        if pin_entry.get("receipt") != payload["receipt"]:
            errors.append("head_receipt_not_pinned")
        if pin_entry.get("sha256") != payload["sha256"]:
            errors.append("head_sha256_not_pinned")
    return sorted(set(errors))


def _load_epoch_receipt(corpus_dir: Path, name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Load a sealed epoch receipt — authenticate before trusting it.

    Returns ``(payload, None)`` only when the file parses AND its
    ``receipt_sha256`` re-computes AND the filename is the seal's sha16
    (``corpus_epoch_<sha16>.json`` — the name the pin and chain both
    reference). A co-forged members+root tamper cannot hold the pinned
    name without recomputing the seal, which renames the file.
    """
    path = corpus_dir / name
    if not path.is_file():
        return None, "epoch_receipt_missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "epoch_receipt_unparseable"
    if not isinstance(payload, dict):
        return None, "epoch_receipt_unparseable"
    seal = payload.get("receipt_sha256")
    if not isinstance(seal, str) or len(seal) != 64:
        return None, "epoch_receipt_unsealed"
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    if hash_bytes(canonical_json_bytes(body)) != seal:
        return None, "epoch_receipt_tampered"
    if path.name != f"corpus_epoch_{seal[:16]}.json":
        return None, "epoch_receipt_name_mismatch"
    # Unwrap a receipt.v2 envelope — the epoch body lives under "payload".
    inner = payload.get("payload")
    return (inner if isinstance(inner, dict) else payload), None


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
    elif unicodedata.normalize("NFC", member) != member:
        errors.append("member_name_not_nfc")
    elif not _portable(member):
        errors.append("member_name_not_portable")
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
        and unicodedata.normalize("NFC", member) == member
        and _portable(member)
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
    receipt, load_err = _load_epoch_receipt(root, str(payload["epoch_receipt"]))
    if load_err is not None:
        return [load_err]
    assert receipt is not None
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
    if unicodedata.normalize("NFC", name) != name:
        raise ValueError(f"absence name must be NFC-canonical: {name!r}")
    if not _portable(name):
        raise ValueError(f"absence name must be portable: {name!a}")
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
    if unicodedata.normalize("NFC", name) != name:
        return ["name_not_nfc"]
    if not _portable(name):
        return ["name_not_portable"]
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
        if unicodedata.normalize("NFC", bname) != bname:
            return ["bound_member_not_nfc"]
        if not _portable(bname):
            return ["bound_member_not_portable"]
        if not verify_inclusion(bname, str(b.get("member_sha256", "")), b, expected_root):
            errors.append(f"bound_invalid:{bname!a}")
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
    epochs = [(p, e) for p, e in _epoch_receipts(root)[0]]
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
    receipt, load_err = _load_epoch_receipt(root, str(payload["epoch_receipt"]))
    if load_err is not None:
        return [load_err]
    assert receipt is not None
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


HISTORY_ABSENCE_SCHEMA = "corpus_history_absence.v1"


def ordered_epoch_chain(
    corpus_dir: Path | str, *, pattern: str = "*.json"
) -> tuple[list[tuple[str, str, str, str | None, frozenset[str]]], list[str]]:
    """Genesis→head ordered chain for one (dir, pattern) lane.

    Each entry is ``(receipt_name, file_sha256, epoch_root_sha256,
    prev_root, member_names)`` — ``prev_root`` is the receipt's claimed
    ``prev_epoch_sha256`` (the previous epoch's root; None at genesis).
    Fails closed — squatter files, unauthentic receipts,
    orphans, forks, multiple genesis/heads all report errors instead of
    yielding a partial chain: a *history* claim over a broken chain is
    meaningless, so no chain is returned unless the whole sequence links.
    """
    from quant_fund.research.corpus_epoch import _epoch_receipts, _member_maps

    root = Path(corpus_dir)
    pairs, squatters = _epoch_receipts(root)
    errors = [f"epoch_prefix_squat:{n!a}" for n in squatters]
    by_name: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for path, body in pairs:
        params = body.get("params")
        pat = params.get("pattern", "*.json") if isinstance(params, Mapping) else "*.json"
        if pat != pattern:
            continue
        payload, err = _load_epoch_receipt(root, path.name)
        if err is not None:
            errors.append(f"{err}:{path.name!a}")
            continue
        assert payload is not None
        by_name[path.name] = (hash_bytes(path.read_bytes()), payload)

    child_of: dict[str, str] = {}
    genesis: list[str] = []
    for name, (_, rec) in by_name.items():
        prev = rec.get("prev_epoch_receipt")
        if prev is None:
            genesis.append(name)
            continue
        if prev not in by_name:
            errors.append(f"epoch_orphan:{name!a}")
            continue
        if rec.get("prev_epoch_sha256") != by_name[prev][1].get("epoch_root_sha256"):
            errors.append(f"epoch_prev_root_mismatch:{name!a}")
        if prev in child_of:
            errors.append(f"epoch_fork:{prev!a}")
        else:
            child_of[prev] = name
    if len(genesis) != 1:
        errors.append(f"epoch_genesis_count:{len(genesis)}")

    ordered: list[tuple[str, str, str, str | None, frozenset[str]]] = []
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
                frozenset(_member_maps(rec)),
            )
        )
        cur = child_of.get(cur)
    if len(ordered) != len(by_name):
        errors.append(f"epoch_chain_disconnected:{len(ordered)}/{len(by_name)}")
    return ordered, sorted(set(errors))


def history_absence_receipt(
    corpus_dir: Path | str,
    name: str,
    *,
    pattern: str = "*.json",
) -> dict[str, Any]:
    """``corpus_history_absence.v1`` — ``name`` was absent at EVERY epoch.

    Complements ``corpus_absence.v1`` (absence at one bound epoch): the
    proof is the full ordered chain — receipt names + file digests — so a
    verifier replays genesis→head and confirms the name never enters any
    member map. Refuses to emit on a broken chain or when the name is a
    member at any epoch (fail closed — absence is a claim, not a filter).
    """
    from quant_fund.research.corpus_epoch import epoch_heads_key

    ordered, errors = ordered_epoch_chain(corpus_dir, pattern=pattern)
    if errors:
        raise ValueError(f"chain not verifiable: {errors[0]}")
    if not ordered:
        raise ValueError(f"no epoch chain under {corpus_dir}")
    for receipt_name, _, _, _, members in ordered:
        if name in members:
            raise ValueError(f"{name!a} is a member at {receipt_name}")
    return {
        "kind": HISTORY_ABSENCE_SCHEMA,
        "schema": HISTORY_ABSENCE_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "corpus_key": epoch_heads_key(Path(corpus_dir), pattern),
        "name": name,
        "epochs": [
            {"receipt": n, "sha256": sha, "epoch_root_sha256": r, "prev_root": p}
            for n, sha, r, p, _ in ordered
        ],
        "n_epochs": len(ordered),
        "head_receipt": ordered[-1][0],
        "head_sha256": ordered[-1][1],
    }


def history_absence_errors(payload: Mapping[str, Any]) -> list[str]:
    """``corpus_history_absence.v1`` shape checks; ``[]`` when clean."""
    errors: list[str] = []
    if payload.get("kind") != HISTORY_ABSENCE_SCHEMA:
        errors.append("kind_not_history_absence")
    if payload.get("schema") != HISTORY_ABSENCE_SCHEMA:
        errors.append("schema_not_history_absence")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        errors.append("name_not_str")
    epochs = payload.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        errors.append("epochs_missing")
        epochs = []
    else:
        for i, e in enumerate(epochs):
            if not isinstance(e, Mapping):
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
    if payload.get("n_epochs") != len(epochs):
        errors.append("n_epochs_mismatch")
    if epochs and isinstance(epochs[-1], Mapping):
        if payload.get("head_receipt") != epochs[-1].get("receipt"):
            errors.append("head_receipt_mismatch")
        if payload.get("head_sha256") != epochs[-1].get("sha256"):
            errors.append("head_sha256_mismatch")
    return sorted(set(errors))


def _chain_pattern_from_key(corpus_key: object) -> str:
    """``receipts/*.json`` → ``*.json``; falls back to the default glob."""
    if isinstance(corpus_key, str) and "/" in corpus_key:
        return corpus_key.rsplit("/", 1)[-1]
    return "*.json"


def verify_history_absence(
    payload: Mapping[str, Any],
    corpus_dir: Path | str,
) -> list[str]:
    """Deep check: re-derive the chain independently, require the claimed
    epoch sequence byte-identical, then confirm ``name`` never a member."""
    errors = history_absence_errors(payload)
    if errors:
        return errors
    ordered, chain_errors = ordered_epoch_chain(
        Path(corpus_dir), pattern=_chain_pattern_from_key(payload.get("corpus_key"))
    )
    if chain_errors:
        return [f"history_chain_unverifiable:{chain_errors[0]}"]
    claimed = [
        (e.get("receipt"), e.get("sha256"), e.get("epoch_root_sha256"), e.get("prev_root"))
        for e in payload["epochs"]  # type: ignore[index]
        if isinstance(e, Mapping)
    ]
    actual = [(n, sha, r, p) for n, sha, r, p, _ in ordered]
    if claimed != actual:
        return ["history_chain_mismatch"]
    for n, _, _, _, members in ordered:
        if payload["name"] in members:
            return [f"history_member_present:{n!a}"]
    return []


def verify_history_absence_pin(
    payload: Mapping[str, Any],
    pin_entry: Mapping[str, Any],
) -> list[str]:
    """Offline check: the claimed chain head must be the pinned head —
    a proof that stops early would hide later membership."""
    errors = history_absence_errors(payload)
    if errors:
        return errors
    # Interior links authenticate the claimed chain offline: each entry's
    # prev_root must equal the previous entry's epoch_root_sha256, so a
    # forged interior entry breaks the chain terminating at the pinned head.
    epochs = payload["epochs"]  # type: ignore[index]
    for i in range(1, len(epochs)):
        prev_e, e = epochs[i - 1], epochs[i]
        if e["prev_root"] != prev_e["epoch_root_sha256"]:
            errors.append(f"history_link_broken:{i}")
    if payload.get("head_receipt") != pin_entry.get("receipt"):
        errors.append("history_head_not_pinned")
    elif payload.get("head_sha256") != pin_entry.get("sha256"):
        errors.append("history_head_digest_drift")
    return sorted(set(errors))
