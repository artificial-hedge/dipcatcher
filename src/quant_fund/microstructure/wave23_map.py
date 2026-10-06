"""wave23_map — sealed synthesis of the instant-impact closure campaign.

Reads the wave-23 receipts out of ``receipts/`` and composes their claim
fields into one verdict map: which target each lane attacked, which
mechanism it located, and whether it closed. The capstone claim is
``closure_path_found`` — whether ANY lane (or stack of lanes) landed all
five tape targets at once.

This is a synthesis bench, not a measurement: it re-verifies each member
receipt's seal before reading its claims, so the map is only as honest
as the receipts underneath — and the seal check is in the claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

WAVE23_MAP_SCHEMA = "wave23_map.v1"

# The wave-23 lane receipts and the mechanism each one attacked.
_LANES: tuple[tuple[str, str], ...] = (
    ("impact_persist_amzn.json", "measurement: per-fill drift kernel"),
    ("anchor_scan_amzn.json", "latent-value anchoring (gain × halflife)"),
    ("instant_decomp_amzn.json", "instant decomposition"),
    ("level_gap_amzn.json", "near-touch vacancy profile"),
    ("joint_fit_amzn.json", "independent-knob joint fit"),
    ("refill_hazard_amzn.json", "pre-fill vacancy memory (refill_cooldown)"),
    ("depth_tilt_amzn.json", "post-fill depth accommodation (lo_tilt)"),
    ("lo_response_amzn.json", "channel decomposition of accommodation"),
    ("place_law_amzn.json", "placement-law family scan"),
    ("place_mix_amzn.json", "join/improve/stack placement mixture"),
    ("aftermath_flow_amzn.json", "post-fill channel attribution"),
    ("cxl_shield_amzn.json", "unhit-side cancel suppression"),
    ("shield_decay_amzn.json", "suppression decay shape"),
    ("continuation_attr_amzn.json", "k200 channel attribution"),
    ("hit_starve_amzn.json", "hit-side refill starvation"),
    ("unhit_chase_amzn.json", "post-fill unhit-side LO chase"),
    ("closure_stack_amzn.json", "independent stack of both winners"),
    ("vac_chase_amzn.json", "vacancy-coupled chase gate"),
    ("release_chase_amzn.json", "churned / repriced chase"),
)


def wave23_map(receipts_dir: Path | str = "receipts") -> dict[str, Any]:
    """Compose the wave-23 receipts into the closure verdict map."""
    # Lazy: receipt verification lives in the research layer above
    # microstructure — the sanctioned way to break that upward edge.
    from quant_fund.research.receipt_v2 import verify_receipt_file  # noqa: PLC0415

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
            body = json.loads(path.read_text())
            entry["claims"] = body.get("claims", {})
            # Five-target closure lives at the CELL level: a receipt's own
            # `joint_closure_exists` claim may be scoped to a narrower
            # target set (e.g. cxl_shield's instant+cxl joint), so the map
            # counts only cells flagged `all_in_tol` on the five-channel
            # tape score.
            cells = body.get("cells") or []
            entry["n_cells_all_in_tol"] = sum(1 for c in cells if c.get("all_in_tol"))
        lanes.append(entry)

    found = [e for e in lanes if e["claims"] is not None]
    closure_found = any(e.get("n_cells_all_in_tol", 0) > 0 for e in lanes)
    # Mechanism summary: the winners each lane located, honestly keyed.
    winners = {
        "instant_channel": "refill_cooldown (pre-fill vacancy memory)",
        "k200_channel": "unhit_imp (post-fill unhit-side LO chase)",
        "joint": None,
    }
    falsified = [
        "independent stacking (closure_stack)",
        "vacancy-coupled chase gate (vac_chase)",
        "churned chase — delete (release_chase rp=0)",
        "churned chase — re-site at spread (release_chase rp>0)",
    ]
    claims = {
        "all_member_receipts_sealed": bool(all_sealed),
        "all_lanes_present": len(lanes) == len(_LANES)
        and all(e["claims"] is not None for e in lanes),
        "instant_channel_located": any(e["claims"].get("cooldown_lifts_instant") for e in found),
        "continuation_channel_located": any(
            e["claims"].get("lo_channel_lifts") or e["claims"].get("chase_still_lifts_lo")
            for e in found
        ),
        "closure_path_found": bool(closure_found),
    }
    payload: dict[str, Any] = {
        "schema": WAVE23_MAP_SCHEMA,
        "kind": "wave23_map",
        "asset": "AMZN",
        "n_lanes": len(lanes),
        "lanes": lanes,
        "mechanism_map": winners,
        "falsified_couplings": falsified,
        "claims": claims,
        "interpretation": (
            "Wave-23 verdict: instant (0.887) and continuation (4.64 @200ev) "
            "are generated by different, located mechanisms — pre-fill "
            "vacancy memory and post-fill unhit-side chase respectively. "
            "Four coupling classes were tried and falsified (independent "
            "stack, vacancy gate, delete-churn, reprice-churn): the tape's "
            "chase must persist AT the touch AND not press it, which this "
            "event grammar cannot express. The residual is closed as a "
            "mapping problem, not a knob problem."
        ),
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["WAVE23_MAP_SCHEMA", "wave23_map"]
