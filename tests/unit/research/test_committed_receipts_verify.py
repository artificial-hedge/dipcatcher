"""Every committed receipt in ``receipts/`` must verify or be quarantined.

Append-only evidence is only evidence when it verifies. Retained receipts
that cannot verify (pre-seal era, detached preregistration seals, contracts
that postdate the bytes, unpinned tapes, absent honesty declarations) are
listed in ``quality/legacy_quarantine.json`` with byte pins, exact error sets
and reasons. This test runs the *production* path — ``suite_health`` with
the committed registry, heads pin and epoch acknowledgment ledger — so unit
acceptance cannot drift from ``suite-health --strict`` again (the tape-legacy
allowlist used to exist only here while the CI gate still failed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.research.corpus_epoch import (
    acks_for_corpus,
    epoch_heads_key,
    load_epoch_acks,
    load_heads_pin,
)
from quant_fund.research.legacy_unsealed import load_quarantine
from quant_fund.research.receipt_v2 import verify_receipt_payload
from quant_fund.research.suite_health import suite_health

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPTS = REPO_ROOT / "receipts"
QUARANTINE = REPO_ROOT / "quality" / "legacy_quarantine.json"
HEADS_PIN = REPO_ROOT / "quality" / "epoch_heads.json"
EPOCH_ACKS = REPO_ROOT / "quality" / "epoch_allowed_removals.json"


def test_committed_receipts_verify_or_are_quarantined() -> None:
    key = epoch_heads_key("receipts", "*.json")
    frame, receipt = suite_health(
        RECEIPTS,
        quarantine=load_quarantine(QUARANTINE),
        expected_head=load_heads_pin(HEADS_PIN)[key],
        epoch_acks=acks_for_corpus(load_epoch_acks(EPOCH_ACKS), key),
    )
    assert receipt["invalid"] == {}, (
        "committed receipts fail verification and are not quarantined:\n"
        + "\n".join(f"{k}: {v}" for k, v in receipt["invalid"].items())
    )
    assert receipt["quarantine"]["stale_entries"] == []
    # Every registry entry is in use — the registry only ever describes
    # files that exist and still fail exactly as recorded.
    assert receipt["n_quarantined"] == receipt["quarantine"]["n_registered"]
    assert receipt["n_ok"] + receipt["n_quarantined"] == receipt["n_receipts"] == frame.height
    epoch_state = receipt["corpus_epoch"]
    assert epoch_state["pinned"] is True
    assert epoch_state["chain_ok"], epoch_state["errors"]


def _sealed_committed() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(RECEIPTS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("receipt_sha256"), str):
            out.append((path.name, payload))
    return out


def _first_leaf(payload: dict[str, Any], *, want: type | tuple[type, ...]) -> list[Any] | None:
    stack: list[tuple[dict[str, Any], str]] = [(payload, k) for k in payload]
    while stack:
        parent, key = stack.pop()
        value = parent[key]
        if isinstance(value, want) and not isinstance(value, bool):
            return [parent, key]
        if isinstance(value, dict):
            stack.extend((value, k) for k in value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            stack.extend((value[0], k) for k in value[0])
    return None


def test_committed_sealed_receipts_reject_tampering() -> None:
    """The seal must catch post-hoc edits on every committed receipt.

    Mutations are in-memory: the receipt is re-verified after each edit and
    every variant must come back invalid. A committed receipt that survives
    a claim edit would be evidence the seal does not cover that surface.
    """
    sealed = _sealed_committed()
    assert sealed, "no sealed committed receipts found — fixture drift"
    misses: list[str] = []
    for name, payload in sealed:
        variants: dict[str, dict[str, Any]] = {}
        # 1. seal flip
        v = dict(payload)
        sha = str(v["receipt_sha256"])
        v["receipt_sha256"] = ("0" if sha[0] != "0" else "1") + sha[1:]
        variants["seal_flip"] = v
        # 2. seal removed (a live receipt must not pass unsealed)
        v = dict(payload)
        v.pop("receipt_sha256")
        variants["seal_removed"] = v
        # 3. rename the claimed kind/schema (dispatch evasion)
        for k in ("kind", "schema"):
            if isinstance(payload.get(k), str):
                v = dict(payload)
                v[k] = "forged_kind"
                variants[f"rename_{k}"] = v
        # 4. perturb the first numeric leaf (a claim digit) — relative, so
        #    huge floats where +1.0 is below epsilon still change value
        loc = _first_leaf(payload, want=(int, float))
        if loc is not None:
            v = json.loads(json.dumps(payload))
            dst = _first_leaf(v, want=(int, float))
            assert dst is not None
            parent, key = dst
            parent[key] = float(parent[key]) * 1.5 + 1.0
            variants["numeric_claim"] = v
        # 5. drop each top-level key that exists
        for k in payload:
            if k == "receipt_sha256":
                continue
            v = dict(payload)
            v.pop(k)
            variants[f"drop_{k}"] = v
        for label, variant in variants.items():
            result = verify_receipt_payload(variant)
            if result["valid"]:
                misses.append(f"{name}:{label}")
    assert misses == [], "tampered receipts still verify — seal/contract hole:\n" + "\n".join(
        misses
    )
