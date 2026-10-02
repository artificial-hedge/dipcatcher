"""wave24_map — sealed synthesis of the standing-spread campaign.

Reads the wave-24 receipts out of ``receipts/`` and composes their claim
fields into one verdict map over the emptied-touch arc's last channel:
the tape's standing 9-63-tick spread. Which mechanism each lane
attacked, whether the band opened, whether it composed with the
emptied-touch stack, and what residual remains.

This is a synthesis bench, not a measurement: it re-verifies each member
receipt's seal before reading its claims, so the map is only as honest
as the receipts underneath — and the seal check is in the claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.research.receipt_v2 import verify_receipt_file
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

WAVE24_MAP_SCHEMA = "wave24_map.v1"

# The wave-24 lane receipts and the mechanism each one attacked.
_LANES: tuple[tuple[str, str], ...] = (
    ("spread_reopen.json", "anchor depth (lo_offset × flow) on all-pins cell"),
    ("band_shape.json", "deposit-density family (density_exponent β)"),
    ("quote_floor.json", "maker min-depth floor (min_quote_dist)"),
    ("floor_compose.json", "floor × emptied-touch stack composition"),
    ("floor_reseed.json", "vacancy-repost channel under the floor"),
    ("floor_stability.json", "per-pin hold rates under the floor"),
    ("floor_rate.json", "band-open rate over floor × flow × seed"),
    ("floor_pins.json", "full 7-pin contract at the structural corner"),
    ("gap_close.json", "repost-geometry repost_band/pp_band scan"),
    ("iid_floor.json", "floor under pure iid flow"),
)


def wave24_map(receipts_dir: Path | str = "receipts") -> dict[str, Any]:
    """Compose the wave-24 receipts into the standing-spread verdict map."""
    receipts_dir = Path(receipts_dir)
    lanes: list[dict[str, Any]] = []
    all_sealed = True
    for fname, mechanism in _LANES:
        path = receipts_dir / fname
        entry: dict[str, Any] = {"receipt": fname, "mechanism": mechanism}
        if not path.exists():
            entry["sealed"] = False
            entry["claims"] = None
            all_sealed = False
        else:
            entry["sealed"] = bool(verify_receipt_file(path)["valid"])
            all_sealed = all_sealed and entry["sealed"]
            entry["claims"] = json.loads(path.read_text()).get("claims", {})
        lanes.append(entry)

    found = [e for e in lanes if e["claims"] is not None]

    claims = {
        "all_member_receipts_sealed": bool(all_sealed),
        "all_lanes_present": len(lanes) == len(_LANES)
        and all(e["claims"] is not None for e in lanes),
        # The spread band opened — at least one lane landed a cell
        # inside the tape's [9, 63] occupancy band.
        "band_opened": any(
            e["claims"].get("floor_opens_spread")
            or e["claims"].get("spread_survives_scan")
            or e["claims"].get("opens_under_iid")
            for e in found
        ),
        # The floor composes with emptied-touch — some cell reached
        # at least 6/7 pins.
        "floor_composes": any(e["claims"].get("floor_composes") for e in found),
        # A 7/7 draw exists — full closure is reachable in principle.
        "seven_pin_reachable": any(
            e["claims"].get("full_closure")
            or e["claims"].get("all_pins_at_corner")
            or e["claims"].get("iid_seven_pin_draw")
            for e in found
        ),
        # But no cell is structural — closure stays a rate, not a
        # certainty (stable_closure FALSE everywhere).
        "stable_closure": any(e["claims"].get("stable_closure") for e in found),
    }
    mechanism_map = {
        "spread_band": "min_quote_dist — maker min-depth floor",
        "reseed": "fill-triggered reposts + vacancy reposts",
        "empty_persistence": "split-flow metaorder sweeps",
        "joint": "6/7 at g10+d280+rp60 under split; iid stays a rate",
    }
    falsified = [
        "anchor depth (spread_reopen — non-monotone)",
        "deposit density family (band_shape — equilibrium, not grammar)",
        "repost geometry (gap_close — reveal bound to floor height)",
        "pure iid flow (iid_floor — band rate capped ~1/3)",
    ]
    payload: dict[str, Any] = {
        "schema": WAVE24_MAP_SCHEMA,
        "kind": "wave24_map",
        "asset": "AMZN",
        "n_lanes": len(lanes),
        "lanes": lanes,
        "mechanism_map": mechanism_map,
        "falsified_channels": falsified,
        "claims": claims,
        "interpretation": (
            "Wave-24 verdict: the tape's standing spread is reachable — the "
            "maker min-depth floor (min_quote_dist) is the first placement "
            "class that lands sim draws inside the tape's 9-63-tick band, "
            "and its dose-response is monotone (g12 opens every split draw). "
            "It composes with the emptied-touch stack at 6/7 pins (the "
            "reveal gap misses by 0.29 ticks, bound geometrically to the "
            "floor height), and a 7/7 draw exists under both split and iid "
            "flow — but no cell is stable: closure is a rate phenomenon. "
            "The residual is pinned to two honest facts: under iid, lone "
            "fills reseed the vacancy before it persists; and the reveal "
            "gap equals the floor height by construction."
        ),
        "git_revision": git_revision(),
        "data_label": "MIXED",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["WAVE24_MAP_SCHEMA", "wave24_map"]
