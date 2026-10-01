"""promotion_gate — admissibility verdict for a named head/model.

Deciding whether a head may be *referenced* by fleet evals is itself an
evidence question: which committed receipts mention it, do they verify,
and what labels do they carry? This lane audits the committed receipt
corpus for a name and emits a fail-closed verdict.

A receipt "mentions" the name when it appears as a dict key or a whole
string value anywhere in the payload (no substring matching — a head
named ``ar`` must not inherit mentions of ``var``).

Verdict:

- ``rejected`` — at least one mentioning receipt fails ``verify-receipt``
  (the evidence about it is corrupt or unsealed);
- ``unverified`` — no committed receipt mentions it;
- ``synthetic_only`` — every verified mention is SYNTHETIC;
- ``admitted`` — every mention verifies and at least one carries REAL or
  MIXED evidence.

The gate is monotone in corpus quality: corrupt evidence can only move a
name *down*, never up. Emits a sealed ``promotion_gate.v1`` receipt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

PROMOTION_GATE_SCHEMA = "promotion_gate.v1"


def _mentions(node: Any, name: str, path: str = "$") -> list[str]:
    """JSON paths where `name` appears as a key or a whole string value."""
    hits: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}"
            if k == name:
                hits.append(p + " (key)")
            hits.extend(_mentions(v, name, p))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits.extend(_mentions(v, name, f"{path}[{i}]"))
    elif node == name and isinstance(node, str):
        hits.append(path)
    return hits


def _receipt_verified(path: Path) -> tuple[bool, list[str], dict[str, Any]]:
    from quant_fund.research.receipt_v2 import verify_receipt_file

    result = verify_receipt_file(path)
    if isinstance(result, dict):
        errors = [str(e) for e in result.get("errors") or []]
    else:  # ReceiptVerification-style objects
        errors = [str(e) for e in getattr(result, "errors", None) or []]
        if not errors and getattr(result, "valid", True) is False:
            errors = ["verification_failed_unspecified"]
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        payload = {}
    return not errors, errors, payload if isinstance(payload, dict) else {}


def promotion_audit(name: str, receipts_dir: Path) -> dict[str, Any]:
    """Scan the committed corpus for evidence mentioning `name`."""
    if not name or not name.strip():
        raise ValueError("promotion gate requires a non-empty head name")
    if not receipts_dir.is_dir():
        raise FileNotFoundError(f"receipts dir not found: {receipts_dir}")

    mentions: list[dict[str, Any]] = []
    for path in sorted(receipts_dir.glob("*.json")):
        verified, errors, rec = _receipt_verified(path)
        paths = _mentions(rec, name)
        if not paths:
            continue
        mentions.append(
            {
                "receipt": path.name,
                "kind": rec.get("kind"),
                "data_label": rec.get("data_label"),
                "verified": verified,
                "errors": errors,
                "mention_paths": paths[:5],
            }
        )

    if any(not m["verified"] for m in mentions):
        verdict = "rejected"
    elif not mentions:
        verdict = "unverified"
    elif all(m["data_label"] == "SYNTHETIC" for m in mentions):
        verdict = "synthetic_only"
    else:
        verdict = "admitted"

    out: dict[str, Any] = {
        "kind": "promotion_gate",
        "schema": PROMOTION_GATE_SCHEMA,
        "subject": name,
        "verdict": verdict,
        "n_mentions": len(mentions),
        "n_verified": sum(1 for m in mentions if m["verified"]),
        "labels": sorted({str(m["data_label"]) for m in mentions}),
        "mentions": mentions,
        "claim": f"promotion_gate:{verdict}",
        "interpretation": (
            "A head is admissible only when every committed receipt that "
            "mentions it verifies and at least one carries non-synthetic "
            "evidence. Corrupt or unsealed evidence demotes the verdict — "
            "the gate cannot be gamed by writing more receipts."
        ),
    }
    out["git_revision"] = git_revision()
    out["data_label"] = "MIXED"
    out["research_only"] = True
    out["receipt_sha256"] = hash_bytes(canonical_json_bytes(out))
    return out
