"""Every committed receipt in ``receipts/`` must pass ``verify-receipt``.

Append-only evidence is only evidence when it verifies. ``receipts/legacy-
unsealed/`` holds the pre-seal-era artifacts quarantined as *unverifiable*;
anything in ``receipts/`` root is claimed live and must seal-verify.
``KNOWN_UNSEALED`` is a shrink-only ratchet for the legacy receipts that have
not migrated yet — a new unsealed receipt in ``receipts/`` fails this test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from quant_fund.research.receipt_v2 import verify_receipt_file, verify_receipt_payload

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPTS = REPO_ROOT / "receipts"

# Pre-seal-era receipts pending the legacy-unsealed/ migration (#231).
# Shrink-only: removing an entry is always safe; adding one is not.
# Values pin the file bytes — an unsealed receipt can't verify, so without
# the digest the exemption would also mask edits to its claims.
KNOWN_UNSEALED = {
    "adaptive_mix_20asset_1d_20260922.json": "b59422415e556f674ea03fb8e9a3b867408bf6d163cf7c11c64feeb7e33b4313",
    "adaptive_mix_band_search_20asset_1d_20260922.json": "5e02918850944a68d1d45815b5928f1eed462d456eac4723ac9290514e7b97f4",
    "basis_pair_candidate_20asset_1d_20260922.json": "d021801c94e33d8720f9084762138893978a61e36ef670b4c0140edeaf37c1ec",
    "basis_reversion_screen_20asset_1d_20260922.json": "61c9c8f5b3e502f0c5218dd743ad030ecbb962d6822bb7b81a2b962445aa378f",
    "dip_bench_crypto_1d_20260925.json": "584eb681dcd18fc65835c1573360b4b5fc9ebb72aa662bcbb9e7c101b06e2f03",
    "fast_replay_p42_conformance_20260927.json": "8236a26489e9253dfcc1f3d879a2fd276c0023fb4d167c764c5330406ce950d8",
    "incumbent_bench_qlib.json": "f455123351b44151d68876d1a93fa3a2dc449c51d280ee83926f22cd9861984f",
}


def test_committed_receipts_verify_or_are_known_legacy() -> None:
    unsealed: list[str] = []
    failures: list[str] = []
    for path in sorted(RECEIPTS.glob("*.json")):
        result = verify_receipt_file(path)
        if result["valid"]:
            continue
        if (
            result["errors"] == ["receipt_sha256_missing_or_invalid"]
            and path.name in KNOWN_UNSEALED
            and hashlib.sha256(path.read_bytes()).hexdigest() == KNOWN_UNSEALED[path.name]
        ):
            unsealed.append(path.name)
            continue
        failures.append(f"{path.name}: {result['errors']}")
    assert failures == [], "committed receipts fail verification:\n" + "\n".join(failures)
    assert not set(KNOWN_UNSEALED) - set(unsealed), (
        "KNOWN_UNSEALED has entries no longer unsealed in receipts/ — shrink the set"
    )


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
        # 4. perturb the first numeric leaf (a claim digit)
        loc = _first_leaf(payload, want=(int, float))
        if loc is not None:
            v = json.loads(json.dumps(payload))
            dst = _first_leaf(v, want=(int, float))
            assert dst is not None
            parent, key = dst
            parent[key] = float(parent[key]) + 1.0
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
