"""Epoch deltas: a sealed, completeness-verified attestation of the exact
change-set between two corpus epochs.

``corpus_epoch`` receipts chain endpoint roots; ``corpus_proof`` /
``corpus_absence`` answer point membership questions. Neither answers the
auditor's real question — "what *changed* between epoch A and epoch B?" —
as a sealed artifact. An ``epoch_delta.v1`` receipt binds the two epoch
receipts (names + file digests), both member-map digests, both Merkle
roots, and the complete transition table:

- ``added``   — members present in next, absent in prev (inclusion path
  against ``next_merkle_root``);
- ``removed`` — members present in prev, absent in next (inclusion path
  against ``prev_merkle_root``);
- ``changed`` — same name, different sha256 (paths against both roots);
- ``unchanged_count`` — the size of the identical intersection, so the
  table's accounting closes (``n_prev = removed + changed + unchanged``,
  ``n_next = added + changed + unchanged``).

Two verification strengths:

- ``epoch_delta_errors`` — self-contained contract: every transition row
  recomputes the declared Merkle root from its embedded path (rows can't
  anchor to roots they don't belong to), the table partitions the name
  space (disjoint added/removed/changed), and the accounting balances.
- ``verify_epoch_delta`` — deep check given corpus access: both epoch
  receipts are loaded through the authenticating loader (seal + name
  re-derivation), the member maps are re-derived, and the declared
  transition table must equal the computed set difference **exactly** —
  completeness, not just consistency. A dropped or invented transition
  fails closed.

Fail closed everywhere: a missing/tampered epoch receipt, a path that
recomputes a different root, a malformed row, or an incomplete table all
yield errors — the receipt attests what was observed, never what was hoped.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.research.epoch_merkle import (
    _epoch_members,
    _load_epoch_receipt,
    _portable,
    inclusion_proof,
    merkle_root,
    verify_inclusion,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

EPOCH_DELTA_SCHEMA = "epoch_delta.v1"


def member_map_sha256(members: Mapping[str, str]) -> str:
    """Canonical digest of a ``{name: sha256}`` member map."""
    return hash_bytes(canonical_json_bytes({k: members[k] for k in sorted(members)}))


def _transitions(
    prev: Mapping[str, str], next_: Mapping[str, str]
) -> tuple[set[str], set[str], dict[str, tuple[str, str]], int]:
    names = set(prev) | set(next_)
    added = {n for n in names if n in next_ and n not in prev}
    removed = {n for n in names if n in prev and n not in next_}
    changed = {
        n: (prev[n], next_[n]) for n in names if n in prev and n in next_ and prev[n] != next_[n]
    }
    unchanged = sum(1 for n in names if n in prev and n in next_ and prev[n] == next_[n])
    return added, removed, changed, unchanged


def _rows_with_paths(
    names: set[str] | dict[str, tuple[str, str]],
    members: Mapping[str, str],
    *,
    side: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(names):
        p = inclusion_proof(members, name)
        row: dict[str, Any] = {
            "name": name,
            "sha256": members[name],
            "leaf_index": p["leaf_index"],
            "n_members": p["n_members"],
            "path": p["path"],
        }
        if side == "changed":
            assert isinstance(names, dict)
            row["from_sha256"], row["to_sha256"] = names[name]
        rows.append(row)
    return rows


def epoch_delta_receipt(
    corpus_dir: Path | str,
    prev_epoch: str,
    next_epoch: str,
) -> dict[str, Any]:
    """Build an ``epoch_delta.v1`` body for ``prev_epoch`` → ``next_epoch``.

    Both arguments are epoch receipt filenames (``corpus_epoch_*.json``)
    inside ``corpus_dir`` — the receipts carry the authoritative member
    maps, so the delta is computed from sealed state, not the live tree.
    """
    from quant_fund.research.corpus_epoch import epoch_heads_key

    root = Path(corpus_dir)
    prev_path, prev_payload, prev_members = _epoch_members(root, prev_epoch)
    next_path, next_payload, next_members = _epoch_members(root, next_epoch)
    if prev_path.name == next_path.name:
        raise ValueError("prev and next epoch receipts must differ")

    added, removed, changed, unchanged = _transitions(prev_members, next_members)
    prev_root = merkle_root(prev_members) if prev_members else None
    next_root = merkle_root(next_members)
    pattern = str((next_payload.get("params") or {}).get("pattern") or "*.json")

    transitions: dict[str, Any] = {
        "added": _rows_with_paths(added, next_members, side="added"),
        "removed": _rows_with_paths(removed, prev_members, side="removed"),
        "changed": [
            {
                "name": name,
                "from_sha256": prev_sha,
                "to_sha256": next_sha,
                "prev_leaf_index": inclusion_proof(prev_members, name)["leaf_index"],
                "next_leaf_index": inclusion_proof(next_members, name)["leaf_index"],
                "prev_path": inclusion_proof(prev_members, name)["path"],
                "next_path": inclusion_proof(next_members, name)["path"],
                "n_prev_members": len(prev_members),
                "n_next_members": len(next_members),
            }
            for name, (prev_sha, next_sha) in sorted(changed.items())
        ],
        "unchanged_count": unchanged,
    }
    return {
        "kind": EPOCH_DELTA_SCHEMA,
        "schema": EPOCH_DELTA_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "corpus_key": epoch_heads_key(root, pattern),
        "prev_epoch": {
            "receipt": prev_path.name,
            "epoch_root_sha256": prev_payload.get("epoch_root_sha256"),
            "merkle_root": prev_root,
            "member_map_sha256": member_map_sha256(prev_members),
            "n_members": len(prev_members),
        },
        "next_epoch": {
            "receipt": next_path.name,
            "epoch_root_sha256": next_payload.get("epoch_root_sha256"),
            "merkle_root": next_root,
            "member_map_sha256": member_map_sha256(next_members),
            "n_members": len(next_members),
        },
        "transitions": transitions,
    }


def _sha(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _name_ok(n: Any) -> bool:
    return isinstance(n, str) and bool(n) and unicodedata.normalize("NFC", n) == n and _portable(n)


def _row_path_errors(row: Any, expected_root: str, which: str) -> list[str]:
    """One added/removed row: verify its embedded path recomputes the root."""
    if not isinstance(row, Mapping):
        return [f"{which}_row_malformed"]
    name = row.get("name")
    sha = row.get("sha256")
    if not _name_ok(name):
        return [f"{which}_name_bad"]
    if not _sha(sha):
        return [f"{which}_sha256_bad"]
    proof = {k: row.get(k) for k in ("leaf_index", "n_members", "path")}
    if not verify_inclusion(str(name), str(sha), proof, expected_root):
        return [f"{which}_path_invalid:{name}"]
    return []


def epoch_delta_errors(payload: Mapping[str, Any]) -> list[str]:
    """``epoch_delta.v1`` self-contained consistency; ``[]`` when clean.

    Verifies every transition row's Merkle path recomputes the declared
    root, the three sets are pairwise disjoint, and the accounting closes
    against the declared member counts.
    """
    errors: list[str] = []
    if payload.get("kind") != EPOCH_DELTA_SCHEMA:
        errors.append("kind_not_epoch_delta")
    if payload.get("schema") != EPOCH_DELTA_SCHEMA:
        errors.append("schema_not_epoch_delta")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    prev = payload.get("prev_epoch")
    nxt = payload.get("next_epoch")
    if not isinstance(prev, Mapping) or not isinstance(nxt, Mapping):
        errors.append("epoch_blocks_missing")
        return errors
    for label, block in (("prev", prev), ("next", nxt)):
        if not isinstance(block.get("receipt"), str):
            errors.append(f"{label}_receipt_missing")
        if not _sha(block.get("merkle_root")):
            errors.append(f"{label}_merkle_root_bad")
        if not _sha(block.get("member_map_sha256")):
            errors.append(f"{label}_member_map_sha256_bad")
        if not isinstance(block.get("n_members"), int):
            errors.append(f"{label}_n_members_bad")
    if errors:
        return errors
    tr = payload.get("transitions")
    if not isinstance(tr, Mapping):
        return [*errors, "transitions_missing"]
    added = tr.get("added")
    removed = tr.get("removed")
    changed = tr.get("changed")
    unchanged = tr.get("unchanged_count")
    if (
        not isinstance(added, list)
        or not isinstance(removed, list)
        or not isinstance(changed, list)
    ):
        return [*errors, "transitions_malformed"]
    prev_root = str(prev["merkle_root"])
    next_root = str(nxt["merkle_root"])
    for row in added:
        errors.extend(_row_path_errors(row, next_root, "added"))
    for row in removed:
        errors.extend(_row_path_errors(row, prev_root, "removed"))
    for row in changed:
        if not isinstance(row, Mapping) or not _name_ok(row.get("name")):
            errors.append("changed_row_malformed")
            continue
        if not _sha(row.get("from_sha256")) or not _sha(row.get("to_sha256")):
            errors.append("changed_sha256_bad")
            continue
        if row["from_sha256"] == row["to_sha256"]:
            errors.append("changed_identical_digests")
        prev_proof = {
            "leaf_index": row.get("prev_leaf_index"),
            "n_members": row.get("n_prev_members"),
            "path": row.get("prev_path"),
        }
        next_proof = {
            "leaf_index": row.get("next_leaf_index"),
            "n_members": row.get("n_next_members"),
            "path": row.get("next_path"),
        }
        if not verify_inclusion(str(row["name"]), str(row["from_sha256"]), prev_proof, prev_root):
            errors.append(f"changed_prev_path_invalid:{row['name']}")
        if not verify_inclusion(str(row["name"]), str(row["to_sha256"]), next_proof, next_root):
            errors.append(f"changed_next_path_invalid:{row['name']}")
    if not isinstance(unchanged, int) or unchanged < 0:
        errors.append("unchanged_count_bad")
        unchanged = None
    a_names = {str(r.get("name")) for r in added if isinstance(r, Mapping)}
    r_names = {str(r.get("name")) for r in removed if isinstance(r, Mapping)}
    c_names = {str(r.get("name")) for r in changed if isinstance(r, Mapping)}
    if a_names & r_names or a_names & c_names or r_names & c_names:
        errors.append("transition_sets_overlap")
    # Accounting runs even when rows failed — a malformed table must also
    # not balance; masking the count check hides dropped-row forgeries.
    if isinstance(unchanged, int):
        if len(r_names) + len(c_names) + unchanged != prev["n_members"]:
            errors.append("prev_accounting")
        if len(a_names) + len(c_names) + unchanged != nxt["n_members"]:
            errors.append("next_accounting")
    return sorted(set(errors))


def verify_epoch_delta(payload: Mapping[str, Any], corpus_dir: Path | str) -> list[str]:
    """Deep check: re-derive the transitions from the two epoch receipts.

    Both receipts load through the authenticating loader (seal + filename
    re-derivation), their member maps are re-derived, and the declared
    transition table must equal the computed difference — a dropped or
    invented row fails closed.
    """
    errors = epoch_delta_errors(payload)
    root = Path(corpus_dir)
    # Structural failures (missing/malformed table) make the deep pass
    # impossible; row-level and accounting failures still proceed so every
    # defect class is reported together, not masked by the first failure.
    structural = {"epoch_blocks_missing", "transitions_missing", "transitions_malformed"}
    if any(e in structural for e in errors):
        return sorted(set(errors))
    members: dict[str, dict[str, str]] = {}
    for label in ("prev_epoch", "next_epoch"):
        block = payload[label]
        receipt, load_err = _load_epoch_receipt(root, str(block["receipt"]))
        if load_err is not None:
            errors.append(f"{label}_{load_err}")
            continue
        assert receipt is not None
        if receipt.get("epoch_root_sha256") != block.get("epoch_root_sha256"):
            errors.append(f"{label}_root_mismatch")
            continue
        m = {
            str(e["name"]): str(e["sha256"])
            for e in receipt.get("members") or []
            if isinstance(e, Mapping) and "name" in e and "sha256" in e
        }
        if member_map_sha256(m) != block.get("member_map_sha256"):
            errors.append(f"{label}_map_digest_mismatch")
        if merkle_root(m) != block.get("merkle_root"):
            errors.append(f"{label}_merkle_mismatch")
        members[label] = m
    if "prev_epoch" not in members or "next_epoch" not in members:
        return sorted(set(errors))
    prev_m, next_m = members["prev_epoch"], members["next_epoch"]
    a_set, r_set, c_set, unchanged = _transitions(prev_m, next_m)
    tr = payload["transitions"]
    decl_a = {str(r["name"]) for r in tr["added"]}
    decl_r = {str(r["name"]) for r in tr["removed"]}
    decl_c = {str(r["name"]): (str(r["from_sha256"]), str(r["to_sha256"])) for r in tr["changed"]}
    if decl_a != a_set:
        errors.append("added_set_incomplete")
    if decl_r != r_set:
        errors.append("removed_set_incomplete")
    if decl_c != {k: (f, t) for k, (f, t) in c_set.items()}:
        errors.append("changed_set_incomplete")
    if tr.get("unchanged_count") != unchanged:
        errors.append("unchanged_count_mismatch")
    return sorted(set(errors))


def write_epoch_delta(
    corpus_dir: Path | str,
    prev_epoch: str,
    next_epoch: str,
    out_dir: Path | str,
) -> Path:
    """Emit the sealed ``epoch_delta_<sha16>.json`` receipt."""
    body = epoch_delta_receipt(corpus_dir, prev_epoch, next_epoch)
    body = {**body, "receipt_sha256": hash_bytes(canonical_json_bytes(body))}
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"epoch_delta_{body['receipt_sha256'][:16]}.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
