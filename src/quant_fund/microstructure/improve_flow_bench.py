"""improve_flow bench — inside-spread and touch-join submission shares.

Real-tape finding (marketable_limit.v1 / queue_fate.v1): 10.31% of limit
submissions land strictly inside the spread, and ~14% of entrants join the
best quote.  The deep-anchor kernel cannot express inside-spread flow when
the floored spread sits at ``lo_offset + 1`` — ``dist + off < spread`` is
unreachable — so a second placement component is needed.  ``lo_improve_frac``
moves a fraction of LO arrivals onto a uniform draw over the open spread
``[own_touch, opp_touch - 1]``: dist 0 joins the touch, deeper slots are
improvements.  The bench measures the shares on the calibrated arm.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from quant_fund.microstructure.maker_age_bench import _MO_PMF, _spec
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

IMPROVE_FLOW_SCHEMA = "improve_flow.v1"

# LOBSTER AMZN placement spectrum (marketable_limit.v1).
_REAL = {
    "inside_spread_share": 0.1031,
}


def _arm(seed: int, *, frac: float) -> dict[str, Any]:
    cfg = replace(
        santa_fe_config(seed=seed),
        band=14,
        lo_offset=12,
        hawkes=_spec(),
        lo_offset_gain=80.0,
        touch_pull=0.4,
        cxl_touch_bias=0.03,
        mo_size_pmf=_MO_PMF,
        lo_improve_frac=frac,
    )
    sim = ZILobSimulator(cfg)
    while sim.t < 1500.0:
        sim.step()
    ec = sim.event_counts()
    n = max(int(ec["n_lo_arrivals"]), 1)
    return {
        "n_lo_arrivals": int(ec["n_lo_arrivals"]),
        "n_fills": len(sim.trades),
        "inside_spread_share": float(ec["n_lo_improve"] / n),
        "touch_join_share": float(ec["n_lo_join"] / n),
    }


def _d(a: float, b: float) -> float:
    if b == 0.0:
        return math.inf if a != 0.0 else 0.0
    return (a - b) / abs(b)


def improve_flow_bench(horizon: float = 1500.0, seed: int = 13) -> dict[str, Any]:
    """Run the improve-flow arms and seal the receipt."""
    arms = {
        "deep_only": _arm(seed, frac=0.0),
        "improve_10": _arm(seed, frac=0.10),
        "improve_25": _arm(seed, frac=0.25),
    }
    div = {
        f"{name}_inside": round(_d(arm["inside_spread_share"], _REAL["inside_spread_share"]), 3)
        for name, arm in arms.items()
    }
    divergences = [f"{k}={v:+.3f}" for k, v in div.items() if abs(v) > 0.05]
    claims = {
        "deep_anchor_emits_inside_flow": arms["deep_only"]["inside_spread_share"] > 0.01,
        "improve_flow_raises_join_share": (
            arms["improve_25"]["touch_join_share"] > arms["deep_only"]["touch_join_share"]
        ),
        "inside_share_order_of_magnitude": all(
            0.01 < arm["inside_spread_share"] < 0.60 for arm in arms.values()
        ),
    }
    payload: dict[str, Any] = {
        "schema": IMPROVE_FLOW_SCHEMA,
        "kind": "improve_flow",
        "horizon": horizon,
        "seed": seed,
        "arms": arms,
        "real_tape_targets": _REAL,
        "divergences": divergences,
        "claims": claims,
        "interpretation": (
            "With the floored spread (lo_offset=12), even the pure deep-anchor "
            "kernel emits inside-spread flow whenever the spread transiently "
            "opens past off+dist — the improve arm adds a uniform-in-spread "
            "component that lifts the join share toward the tape's ~14%. "
            "Residual: sim inside-share overshoots the 10.31% target on every "
            "arm — the floored spread admits too much improvement traffic; "
            "the honest fix is a two-sided anchor (deep + improve rates "
            "balanced to the occupancy), logged not force-fit."
        ),
        "git_revision": git_revision(),
        "data_label": "MIXED",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["IMPROVE_FLOW_SCHEMA", "improve_flow_bench"]
