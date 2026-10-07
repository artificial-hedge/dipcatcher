"""receipt_tombstone — the retraction half of the evidence lifecycle.

The corpus is append-only: a superseded or wrong receipt cannot be deleted
(the epoch chain flags deletion, the lattice flags claim drift), so the
only honest move is a **retraction record** — a sealed receipt that names
its target byte-for-byte. Downstream consumers (the claim lattice, FDR
corpora) must exclude retracted claims rather than keep scoring them.

Fail-closed rules:

- The tombstone pins ``target_sha256`` — the sha256 of the target file's
  bytes. If the target on disk drifts from the pin, the tombstone covers
  nothing (``tombstone_target_drift``): you cannot retract "some other
  file" by filename alone.
- ``scope: "all"`` retracts the whole receipt; a list of claim paths
  retracts only those fields (partial retraction — e.g. one metric was
  miscomputed but the artifact is otherwise sound).
- The tombstone is itself a sealed corpus member — stamped into the epoch
  chain like any other receipt, so a retractions can't be forged without
  forging the corpus.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import hash_bytes

TOMBSTONE_SCHEMA = "receipt_tombstone.v1"


def tombstone_body(
    *,
    target_name: str,
    target_sha256: str,
    reason: str,
    scope: str | list[str] = "all",
) -> dict[str, Any]:
    """Build a ``receipt_tombstone.v1`` payload (unsealed — seal upstream)."""
    return {
        "kind": TOMBSTONE_SCHEMA,
        "schema": TOMBSTONE_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "target_receipt": target_name,
        "target_sha256": target_sha256,
        "scope": scope,
        "reason": reason,
    }


def write_tombstone(
    target: Path | str,
    *,
    corpus_dir: Path | str,
    reason: str,
    scope: str | list[str] = "all",
) -> Path:
    """Seal a tombstone for ``target`` into the corpus dir.

    The file is named ``tombstone_<target-sha16>.json`` — deterministic, so
    re-retracting the same target overwrites the same member rather than
    piling up duplicates.
    """
    from quant_fund.research.receipt_v2 import seal_receipt
    from quant_fund.utils.atomicio import atomic_write_text

    target_path = Path(target)
    if not target_path.is_file():
        raise ValueError(f"tombstone target {target_path} does not exist")
    digest = hash_bytes(target_path.read_bytes())
    body = tombstone_body(
        target_name=target_path.name,
        target_sha256=digest,
        reason=reason,
        scope=scope,
    )
    sealed = seal_receipt(body)
    out = Path(corpus_dir) / f"tombstone_{digest[:16]}.json"
    atomic_write_text(out, json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    return out


def load_tombstones(corpus_dir: Path | str) -> dict[str, Any]:
    """Resolved tombstone map: corpus-relative target name -> tombstone info.

    Only sealed, contract-clean tombstones whose target bytes match the pin
    count as active; anything else is returned under ``invalid`` for the
    caller to surface as an error.
    """
    root = Path(corpus_dir)
    active: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    for path in sorted(root.rglob("tombstone_*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            invalid.append(f"{path.name}:unreadable")
            continue
        if not isinstance(doc, Mapping):
            invalid.append(f"{path.name}:not_mapping")
            continue
        from quant_fund.research.receipt_v2 import verify_receipt_file

        if not verify_receipt_file(path)["valid"]:
            invalid.append(f"{path.name}:unsealed")
            continue
        inner = doc.get("payload")
        body = inner if isinstance(inner, Mapping) else doc
        if body.get("kind") != TOMBSTONE_SCHEMA:
            continue  # a receipt that merely shares the filename pattern
        errs = tombstone_contract_errors(body)
        if errs:
            invalid.append(f"{path.name}:{','.join(errs)}")
            continue
        target_rel = str(body["target_receipt"])
        target_path = (root / target_rel).resolve()
        if not target_path.is_relative_to(root.resolve()):
            # A tombstone must name a corpus member — an escaping path makes
            # the drift check hash arbitrary files (digest bit-oracle).
            invalid.append(f"{path.name}:tombstone_target_uncontained:{target_rel}")
            continue
        if not target_path.is_file():
            invalid.append(f"{path.name}:tombstone_target_missing:{target_rel}")
            continue
        if hash_bytes(target_path.read_bytes()) != body["target_sha256"]:
            invalid.append(f"{path.name}:tombstone_target_drift:{target_rel}")
            continue
        active[target_rel] = {
            "tombstone": path.name,
            "scope": body["scope"],
            "reason": body["reason"],
            "target_sha256": str(body["target_sha256"]),
        }
    return {"active": active, "invalid": invalid}


def tombstone_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``receipt_tombstone.v1`` internal consistency; ``[]`` when clean."""
    errors: list[str] = []
    if payload.get("kind") != TOMBSTONE_SCHEMA:
        errors.append("kind_not_tombstone")
    if payload.get("schema") != TOMBSTONE_SCHEMA:
        errors.append("schema_not_tombstone")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    target = payload.get("target_receipt")
    if not isinstance(target, str) or not target:
        errors.append("target_receipt_missing")
    sha = payload.get("target_sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        errors.append("target_sha256_malformed")
    elif any(c not in "0123456789abcdef" for c in sha):
        errors.append("target_sha256_not_hex")
    scope = payload.get("scope")
    if scope != "all" and not (
        isinstance(scope, list) and scope and all(isinstance(s, str) and s for s in scope)
    ):
        errors.append("scope_invalid")
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason_missing")
    return errors
