"""Corpus epochs: a hash-chained integrity root over the evidence store.

``verify-receipt`` seals individual receipts; ``receipt_lattice`` cross-checks
claims inside them. Neither watches the *container*: today a receipt can be
deleted or rewritten under ``receipts/`` and nothing in the receipt format
notices — only git history does, and git is outside the sealed-evidence model.

A ``corpus_epoch.v1`` receipt stamps the corpus's full membership:

- ``members`` — every ``*.json`` in the corpus dir as ``{name, sha256}``
  (digest over raw file bytes, sealing-agnostic — pre-seal artifacts count),
  sorted by name;
- ``epoch_root_sha256`` — sha256 over the canonical member map, a Merkle-style
  root for the corpus at this instant;
- ``prev_epoch_sha256`` / ``prev_epoch_receipt`` — link to the previous epoch
  receipt, forming a chain over corpus *states*;
- ``members_added`` / ``members_removed`` — the delta vs that previous epoch.

``check_epoch_chain`` walks the committed chain and enforces the corpus
discipline: epochs form one fork-free chain, membership is non-decreasing
(a deleted receipt flips ``members_removed``), and the head epoch's root
matches the live corpus — so a receipt modified or dropped after the last
epoch stamp surfaces as ``corpus_drift_since_head_epoch``.

Verdicts: ``genesis`` (no previous epoch), ``advancing`` (monotone growth),
``shrinking`` (members removed — recorded fact, honest when intentional).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

EPOCH_SCHEMA = "corpus_epoch.v1"
GENESIS_PREV = "0" * 64


def member_digests(corpus_dir: Path | str) -> dict[str, str]:
    """``{filename: sha256-of-bytes}`` for every ``*.json`` in the corpus."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise ValueError(f"corpus dir {root} does not exist")
    return {
        path.name: hash_bytes(path.read_bytes())
        for path in sorted(root.glob("*.json"))
        if path.is_file()
    }


def epoch_root(members: Mapping[str, str]) -> str:
    """Merkle-style root over the member map — order-free via canonical JSON."""
    for name, digest in members.items():
        if not isinstance(name, str) or not name:
            raise ValueError("member name must be a non-empty string")
        if not (isinstance(digest, str) and len(digest) == 64):
            raise ValueError(f"member {name} digest must be 64-hex")
        int(digest, 16)  # raises on non-hex
    return hash_bytes(canonical_json_bytes(dict(sorted(members.items()))))


def _member_maps(payload: Mapping[str, Any]) -> dict[str, str]:
    """`members` list → `{name: sha256}`; malformed entries map to empty."""
    out: dict[str, str] = {}
    for entry in payload.get("members") or []:
        if isinstance(entry, Mapping):
            name, sha = entry.get("name"), entry.get("sha256")
            if isinstance(name, str) and isinstance(sha, str):
                out[name] = sha
    return out


def _epoch_receipts(corpus_dir: Path) -> list[tuple[Path, Mapping[str, Any]]]:
    """Committed epoch receipts in the corpus, in filename order."""
    out: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(corpus_dir.glob("*.json")):
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, UnicodeError, ValueError):
            continue
        body: object = doc.get("payload") if isinstance(doc, Mapping) else None
        candidate = body if isinstance(body, Mapping) else doc
        if isinstance(candidate, Mapping) and (
            candidate.get("schema") == EPOCH_SCHEMA or candidate.get("kind") == EPOCH_SCHEMA
        ):
            out.append((path, candidate))
    return out


def corpus_epoch(
    corpus_dir: Path | str,
    *,
    head_sha: str | None = None,
) -> dict[str, Any]:
    """Build the epoch receipt over the corpus's current membership.

    Links to the newest committed epoch receipt (highest chain position) as
    ``prev``; membership delta is computed against it.
    """
    root = Path(corpus_dir)
    members = member_digests(root)
    epochs = _epoch_receipts(root)
    # The chain head is the epoch no other epoch names as prev.
    prevs = {e.get("prev_epoch_receipt") for _, e in epochs}
    heads = [(p, e) for p, e in epochs if p.name not in prevs]
    prev_name: str | None = None
    prev_root = GENESIS_PREV
    prev_members: dict[str, str] = {}
    if heads:
        # Deterministic pick if a fork already exists (chain check flags it).
        prev_path, prev_payload = sorted(heads, key=lambda t: t[0].name)[-1]
        prev_name = prev_path.name
        prev_members = _member_maps(prev_payload)
        prev_root = str(prev_payload.get("epoch_root_sha256") or GENESIS_PREV)
    added = sorted(set(members) - set(prev_members))
    removed = sorted(set(prev_members) - set(members))
    if prev_name is None:
        verdict = "genesis"
    elif removed:
        verdict = "shrinking"
    else:
        verdict = "advancing"
    return {
        "kind": EPOCH_SCHEMA,
        "schema": EPOCH_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "inputs_sha256": hash_bytes(canonical_json_bytes({"corpus_dir": str(root)})),
        "params": {"head_sha": head_sha} if head_sha else {},
        "epoch_root_sha256": epoch_root(members),
        "members": [{"name": n, "sha256": s} for n, s in members.items()],
        "n_members": len(members),
        "prev_epoch_sha256": prev_root,
        "prev_epoch_receipt": prev_name,
        "members_added": added,
        "members_removed": removed,
        "verdict": verdict,
    }


def epoch_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``corpus_epoch.v1`` internal consistency; ``[]`` when clean."""
    errors: list[str] = []
    if payload.get("kind") != EPOCH_SCHEMA:
        errors.append("kind_not_corpus_epoch")
    if payload.get("schema") != EPOCH_SCHEMA:
        errors.append("schema_not_corpus_epoch")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    members = payload.get("members")
    if not isinstance(members, list):
        errors.append("members_not_list")
        members = []
    member_map = _member_maps(payload)
    if len(member_map) != len(members):
        errors.append("members_malformed_or_dup_names")
    names = [
        m["name"] for m in members if isinstance(m, Mapping) and isinstance(m.get("name"), str)
    ]
    if len(names) != len(members):
        errors.append("member_names_not_strings")
    elif names != sorted(names):
        errors.append("members_not_sorted")
    n_members = payload.get("n_members")
    if not isinstance(n_members, int) or n_members != len(members):
        errors.append("n_members_mismatch")
    root = payload.get("epoch_root_sha256")
    if not (isinstance(root, str) and len(root) == 64):
        errors.append("epoch_root_sha256")
    elif member_map:
        try:
            if epoch_root(member_map) != root:
                errors.append("epoch_root_mismatch")
        except ValueError:
            errors.append("member_digest_not_hex")
    prev = payload.get("prev_epoch_sha256")
    if not (isinstance(prev, str) and len(prev) == 64):
        errors.append("prev_epoch_sha256")
    for field in ("members_added", "members_removed"):
        lst = payload.get(field)
        if not isinstance(lst, list) or not all(isinstance(x, str) for x in lst):
            errors.append(f"{field}_not_name_list")
            continue
        if lst != sorted(lst):
            errors.append(f"{field}_not_sorted")
        if field == "members_removed" and set(lst) & set(names):
            errors.append("members_removed_still_present")
    verdict = payload.get("verdict")
    if verdict not in ("genesis", "advancing", "shrinking"):
        errors.append("verdict_unknown")
    else:
        removed = payload.get("members_removed") or []
        prev_name = payload.get("prev_epoch_receipt")
        if verdict == "genesis" and prev_name is not None:
            errors.append("verdict_genesis_with_prev")
        if verdict == "shrinking" and not removed:
            errors.append("verdict_shrinking_without_removals")
        if verdict == "advancing" and (removed or prev_name is None):
            errors.append("verdict_advancing_invalid")
    inputs = payload.get("inputs_sha256")
    if not (isinstance(inputs, str) and len(inputs) == 64):
        errors.append("inputs_sha256")
    return errors


def check_epoch_chain(
    corpus_dir: Path | str,
    *,
    allowed_removals: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Walk the committed epoch chain against the live corpus.

    Returns ``{"errors": [...], "unstamped": [...], "head": <name|None>,
    "head_epoch_root": <hex|None>}``. Errors are integrity violations the
    chain is authoritative over: forked/disconnected/invalid epoch receipts,
    stamped members removed (outside ``allowed_removals`` name→sha256 pins),
    stamped members mutated, and dishonest delta fields. ``unstamped`` lists
    corpus members not covered by the head epoch — arrivals between stamps
    are the normal state, recorded not flagged.
    """
    root = Path(corpus_dir)
    errors: list[str] = []
    unstamped: list[str] = []
    epochs = _epoch_receipts(root)
    if not epochs:
        return {
            "errors": ["no_epoch_receipts"],
            "unstamped": [],
            "head": None,
            "head_epoch_root": None,
        }

    # Seal-verify every epoch receipt before trusting its claims.
    from quant_fund.research.receipt_v2 import verify_receipt_file

    sealed: dict[str, Mapping[str, Any]] = {}
    for path, payload in epochs:
        ver = verify_receipt_file(path)
        if not ver["valid"]:
            errors.append(f"epoch_receipt_invalid:{path.name}")
            continue
        sealed[path.name] = payload
    if not sealed:
        errors.append("no_valid_epoch_receipts")
        return {
            "errors": errors,
            "unstamped": unstamped,
            "head": None,
            "head_epoch_root": None,
        }

    # Link the chain: each non-genesis epoch names prev_epoch_receipt.
    by_name = sealed
    child_of: dict[str, str] = {}  # prev_name -> epoch name (for fork check)
    genesis: list[str] = []
    for name, payload in by_name.items():
        prev = payload.get("prev_epoch_receipt")
        if prev is None:
            genesis.append(name)
            continue
        if prev not in by_name:
            errors.append(f"epoch_orphan:{name}")
            continue
        if payload.get("prev_epoch_sha256") != by_name[prev].get("epoch_root_sha256"):
            errors.append(f"epoch_prev_root_mismatch:{name}")
        if prev in child_of:
            errors.append(f"epoch_fork:{prev}->{child_of[prev]},{name}")
        child_of[prev] = name
    if len(genesis) > 1:
        errors.append(f"epoch_multiple_genesis:{','.join(sorted(genesis))}")

    # Membership must be non-decreasing along the chain.
    allowed = dict(allowed_removals or {})
    for prev_name, cur_name in child_of.items():
        prev_members = _member_maps(by_name[prev_name])
        cur_members = _member_maps(by_name[cur_name])
        for gone in sorted(set(prev_members) - set(cur_members)):
            if allowed.get(gone) != prev_members[gone]:
                errors.append(f"member_removed:{gone}@{cur_name}")
        declared_removed = set(by_name[cur_name].get("members_removed") or [])
        actual_removed = set(prev_members) - set(cur_members)
        if declared_removed != actual_removed:
            errors.append(f"members_removed_dishonest:{cur_name}")
        declared_added = set(by_name[cur_name].get("members_added") or [])
        actual_added = set(cur_members) - set(prev_members)
        if declared_added != actual_added:
            errors.append(f"members_added_dishonest:{cur_name}")
        # Mutated members: same name, different digest.
        for kept in set(prev_members) & set(cur_members):
            if prev_members[kept] != cur_members[kept]:
                errors.append(f"member_mutated:{kept}@{cur_name}")

    # Head vs live corpus: stamped membership must hold exactly; files added
    # after the head stamp are unstamped (normal), not violations.
    heads = set(by_name) - set(child_of)
    head_name: str | None = None
    head_root: str | None = None
    if len(heads) == 1:
        head_name = next(iter(heads))
        head = by_name[head_name]
        root_val = head.get("epoch_root_sha256")
        head_root = root_val if isinstance(root_val, str) else None
        live = member_digests(root)
        head_members = _member_maps(head)
        stamped = set(head_members)
        for name in stamped - set(live):
            if allowed.get(name) != head_members[name]:
                errors.append(f"head_member_missing_live:{name}")
        for name, sha in head_members.items():
            if name in live and live[name] != sha:
                errors.append(f"head_member_digest_drift:{name}")
        unstamped = sorted(set(live) - stamped - set(by_name))
    elif len(heads) > 1:
        errors.append(f"epoch_multiple_heads:{','.join(sorted(heads))}")
    return {
        "errors": errors,
        "unstamped": unstamped,
        "head": head_name,
        "head_epoch_root": head_root,
    }


def write_epoch_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal + atomically write a corpus-epoch receipt; fails closed on contract violation."""
    errors = epoch_contract_errors(receipt)
    if errors:
        raise ValueError(f"corpus_epoch receipt violates contract: {errors}")
    dirpath = Path(receipts_dir)
    dirpath.mkdir(parents=True, exist_ok=True)

    from quant_fund.research.fleet_eval import _atomic_write_text
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass" if receipt.get("verdict") in ("genesis", "advancing") else "fail",
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = dirpath / f"corpus_epoch_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
