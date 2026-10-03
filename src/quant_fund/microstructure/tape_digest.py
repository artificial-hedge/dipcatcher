"""tape_digest — consolidated divergence table across the real-tape lanes.

Each real-tape lane emits per-arm fact coverage. This digest reads the
committed receipts and produces the arm x fact coverage matrix: which
flow mechanism reproduces which stylized fact, and which facts no arm
captures. The sealed output is the campaign's single-page answer to
"how realistic is the sim".

v2: the arm axis is discovered per lane from the receipt's own
``sim_arms`` table (or ``arms``), canonicalized (``markov_regime`` is
the regime flow), so mechanism lanes beyond the original
iid/regime/split triple — deep_split, empirical sizes, powerlaw clock,
improve/reload/depth variants — are scored in place rather than
invisible. Lanes without an arms table fall back to the legacy triple.
Divergences may be strings prefixed ``{arm}[_:@]`` or dicts
``{"arm": name, "diverges": bool}``; ``sim_*`` / ``mean_spread`` keys
mark a mechanism-level gap every arm fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

LANE_RECEIPTS = {
    "abc_calibrate": "abc_calibrate_amzn.json",
    "cancel_cluster": "cancel_cluster_amzn.json",
    "deep_microprice": "deep_microprice_amzn.json",
    "depth_consumption": "depth_consumption_amzn.json",
    "event_burst": "event_burst_amzn.json",
    "event_granger": "event_granger_amzn.json",
    "event_matrix": "event_matrix_amzn.json",
    "exec_cost": "exec_cost_real_amzn.json",
    "hawkes": "hawkes_real_amzn.json",
    "hawkes_mv": "hawkes_mv.json",
    "hidden_depth": "hidden_depth_amzn.json",
    "impact_instant": "impact_instant_amzn.json",
    "intraday_shape": "intraday_shape_amzn.json",
    "lob_exec": "lob_exec_amzn.json",
    "lob_resilience": "lob_resilience_amzn.json",
    "marketable_limit": "marketable_limit_amzn.json",
    "markout": "markout_amzn.json",
    "metaorder_detect": "metaorder_detect_amzn.json",
    "mid_jump": "mid_jump_amzn.json",
    "order_lifetime": "order_lifetime_amzn.json",
    "order_revision": "order_revision_amzn.json",
    "post_trade_drift": "post_trade_drift_amzn.json",
    "price_improvement": "price_improvement_amzn.json",
    "propagator": "propagator_real_amzn.json",
    "queue_jump": "queue_jump_amzn.json",
    "queue_occupancy": "queue_occupancy_amzn.json",
    "quote_place": "quote_place_amzn.json",
    "round_lot": "round_lot_amzn.json",
    "side_imbalance": "side_imbalance_amzn.json",
    "sign_autocorr": "sign_autocorr_real_amzn.json",
    "sign_predict": "sign_predict_amzn.json",
    "spread_dynamics": "spread_dynamics_amzn.json",
    "spread_response": "spread_response_amzn.json",
    "stale_quote": "stale_quote_amzn.json",
    "streak_stats": "streak_stats_amzn.json",
    "tick_rule": "tick_rule_amzn.json",
    "vol_signature": "vol_signature_amzn.json",
    "vpin": "vpin_amzn.json",
}


ARMS = ("iid", "regime", "split")

# Same flow class under a different arm label.
_ARM_ALIASES = {"markov_regime": "regime"}

_SEPARATORS = ("_", "@", ":")


def _canonical_arm(name: Any) -> str:
    a = _ARM_ALIASES.get(str(name), str(name))
    return a


def _lane_arms(receipt: dict[str, Any]) -> tuple[str, ...]:
    """Discovered arm axis for this lane; falls back to the legacy triple."""
    arms = None
    for key in ("sim_arms", "arms"):
        if arms is None and isinstance(receipt.get(key), dict) and receipt[key]:
            arms = receipt[key]
    table = receipt.get("table")
    if arms is None and isinstance(table, dict):
        t_arms = table.get("sim_arms")
        if isinstance(t_arms, dict) and t_arms:
            arms = t_arms
    if not arms:
        return ARMS
    canon = tuple(dict.fromkeys(_canonical_arm(a) for a in arms))
    return canon or ARMS


def _raw_arms(receipt: dict[str, Any]) -> tuple[str, ...]:
    """Arm names as written in the receipt (pre-alias), for prefix matching."""
    for key in ("sim_arms", "arms"):
        if isinstance(receipt.get(key), dict) and receipt[key]:
            return tuple(str(a) for a in receipt[key])
    table = receipt.get("table")
    if isinstance(table, dict) and isinstance(table.get("sim_arms"), dict):
        return tuple(str(a) for a in table["sim_arms"])
    return ARMS


def _divergence_flags(receipt: dict[str, Any], arms: tuple[str, ...]) -> tuple[set[str], bool]:
    """(diverged arm names, mechanism-wide gap) from the divergences list.

    String entries attribute to an arm by longest-prefix match against the
    lane's discovered arms followed by ``_``, ``@``, or ``:`` — so
    ``deep_split_lag1_response_off`` lands on ``deep_split``, not ``deep``.
    Unknown prefixes don't attribute (kept in n_divergences but not blamed
    on a named arm). Dict entries carry ``{"arm": ..., "diverges": bool}``.
    """
    diverged: set[str] = set()
    mechanism_gap = False
    matchable = tuple(dict.fromkeys((*_raw_arms(receipt), *arms)))
    ordered = sorted(matchable, key=len, reverse=True)
    for d in receipt.get("divergences") or []:
        if isinstance(d, dict):
            arm = _canonical_arm(d.get("arm"))
            if d.get("diverges"):
                diverged.add(arm)
            continue
        s = str(d)
        if s.startswith("sim_") or s.startswith("mean_spread"):
            mechanism_gap = True
            continue
        for arm in ordered:
            if any(s.startswith(f"{arm}{sep}") for sep in _SEPARATORS):
                diverged.add(_canonical_arm(arm))
                break
    return diverged, mechanism_gap


def _arm_status(receipt: dict[str, Any]) -> dict[str, str]:
    """Per arm: 'covered' | 'diverged' | 'unscored'.

    A lane with no `divergences` list measured real-vs-sim without
    per-arm flagging — mark its arms unscored rather than crediting
    coverage.
    """
    arms = _lane_arms(receipt)
    if "divergences" not in receipt or receipt.get("divergences") is None:
        return {a: "unscored" for a in arms}
    diverged, mechanism_gap = _divergence_flags(receipt, arms)
    if mechanism_gap:
        return {a: "diverged" for a in arms}
    return {a: ("diverged" if a in diverged else "covered") for a in arms}


def tape_digest(receipts_dir: Path) -> dict[str, Any]:
    """Build the arm x fact coverage matrix from committed receipts."""
    rows: dict[str, dict[str, Any]] = {}
    arm_scores: dict[str, int] = {}
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
        for arm, s in status.items():
            if s == "unscored":
                continue
            arm_scores.setdefault(arm, 0)
            arm_scores[arm] += int(s == "covered")
    best_arm = max(arm_scores, key=lambda a: arm_scores[a]) if arm_scores else None
    uncovered_lanes = [
        lane
        for lane, r in rows.items()
        if r.get("present")
        and r["arm_status"]
        and all(s == "diverged" for s in r["arm_status"].values())
    ]
    payload: dict[str, Any] = {
        "kind": "tape_digest",
        "schema": "tape_digest.v1",
        "n_lanes_present": n_present,
        "n_lanes_scored": n_scored,
        "lanes": rows,
        "arm_universe": sorted(arm_scores),
        "arm_coverage_score": dict(sorted(arm_scores.items())),
        "best_arm": best_arm,
        "uncovered_lanes": uncovered_lanes,
        "claim": "sim_realism_coverage_matrix",
        "interpretation": (
            "arm_status: covered = lane flagged no divergence for that "
            "arm; diverged = flagged; unscored = lane measured but has "
            "no per-arm divergence list. The arm axis is discovered per "
            "lane from its sim_arms table (markov_regime aliases to "
            "regime), so mechanism arms beyond iid/regime/split score "
            "where they run. uncovered_lanes flag facts no arm "
            "reproduces — the sim's priority list."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
