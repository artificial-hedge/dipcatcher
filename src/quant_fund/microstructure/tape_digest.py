"""tape_digest — consolidated divergence table across the wave-21b lanes.

Each real-tape lane emits per-arm fact coverage. This digest reads the
committed receipts and produces the arm × fact coverage matrix: which
flow mechanism (iid / regime / split) reproduces which stylized fact,
and which facts no arm captures. The sealed output is the campaign's
single-page answer to "how realistic is the sim".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

LANE_RECEIPTS = {
    "propagator": "propagator_real_amzn.json",
    "hawkes": "hawkes_real_amzn.json",
    "order_lifetime": "order_lifetime_amzn.json",
    "vpin": "vpin_amzn.json",
    "intraday_shape": "intraday_shape_amzn.json",
    "sign_autocorr": "sign_autocorr_real_amzn.json",
    "exec_cost": "exec_cost_real_amzn.json",
    "quote_place": "quote_place_amzn.json",
    "cancel_cluster": "cancel_cluster_amzn.json",
    "round_lot": "round_lot_amzn.json",
    "hidden_depth": "hidden_depth_amzn.json",
    "spread_dynamics": "spread_dynamics_amzn.json",
    "event_matrix": "event_matrix_amzn.json",
    "streak_stats": "streak_stats_amzn.json",
}


ARMS = ("iid", "regime", "split")


def _arm_status(receipt: dict[str, Any]) -> dict[str, str]:
    """Per arm: 'covered' | 'diverged' | 'unscored'.

    A lane with no `divergences` list measured real-vs-sim without
    per-arm flagging — mark its arms unscored rather than crediting
    coverage. `sim_*`-prefixed divergence keys are mechanism gaps:
    every arm fails equally.
    """
    if "divergences" not in receipt or receipt.get("divergences") is None:
        return {a: "unscored" for a in ARMS}
    divs = [str(d) for d in receipt["divergences"]]
    if any(d.startswith("sim_") or d.startswith("mean_spread") for d in divs):
        return {a: "diverged" for a in ARMS}
    return {
        a: ("diverged" if any(d.startswith(f"{a}_") for d in divs) else "covered") for a in ARMS
    }


def tape_digest(receipts_dir: Path) -> dict[str, Any]:
    """Build the arm × fact coverage matrix from committed receipts."""
    rows: dict[str, dict[str, Any]] = {}
    arm_scores = {a: 0 for a in ARMS}
    n_present = n_scored = 0
    for lane, fname in LANE_RECEIPTS.items():
        path = receipts_dir / fname
        if not path.exists():
            rows[lane] = {"present": False}
            continue
        n_present += 1
        receipt = json.loads(path.read_text())
        status = _arm_status(receipt)
        n_scored += int(all(s != "unscored" for s in status.values()))
        rows[lane] = {
            "present": True,
            "receipt_sha256": receipt.get("receipt_sha256"),
            "n_divergences": len(receipt.get("divergences") or []),
            "arm_status": status,
        }
        for arm in ARMS:
            arm_scores[arm] += int(status[arm] == "covered")
    best_arm = max(arm_scores, key=lambda a: arm_scores[a]) if n_scored else None
    uncovered_lanes = [
        lane
        for lane, r in rows.items()
        if r.get("present") and all(s == "diverged" for s in r["arm_status"].values())
    ]
    payload: dict[str, Any] = {
        "kind": "tape_digest",
        "schema": "tape_digest.v1",
        "n_lanes_present": n_present,
        "n_lanes_scored": n_scored,
        "lanes": rows,
        "arm_coverage_score": arm_scores,
        "best_arm": best_arm,
        "uncovered_lanes": uncovered_lanes,
        "claim": "sim_realism_coverage_matrix",
        "interpretation": (
            "arm_status: covered = lane flagged no divergence for that "
            "arm; diverged = flagged; unscored = lane measured but has "
            "no per-arm divergence list. uncovered_lanes flag facts no "
            "arm reproduces — the sim's priority list."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
