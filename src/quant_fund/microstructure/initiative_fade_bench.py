"""initiative_fade bench — intraday buy-initiative fade, real vs sim.

Real-tape finding (side_imbalance.v1): buy-initiated fills are ~0.536 of
executions overall but the share is non-stationary — ~0.61 early in the
session fading to ~0.50 late. The side-symmetric ZI-LOB cannot express
initiative drift: a flat ``p_buy`` arm holds ~0.50 in every third of the
run. A linear ``p_buy_drift`` on the MO side draw (``p_eff = clip(p_buy +
p_buy_drift * t, 0, 1)``, the time-varying aggressor-imbalance knob added
to ``ZILobConfig``) reproduces the fade while leaving regime composition
intact.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from quant_fund.microstructure.maker_age_bench import _MO_PMF, _spec
from quant_fund.microstructure.zi_lob_simulator import (
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

INITIATIVE_FADE_SCHEMA = "initiative_fade.v1"

# LOBSTER AMZN buy-initiated execution share (side_imbalance.v1): the
# overall share plus the first/last session-bin shares — initiative fades
# ~0.61 early -> ~0.50 late.
_REAL = {
    "exec_buy_share": 0.5357700022286606,
    "buy_share_early": 0.6109637488947833,
    "buy_share_late": 0.5024630541871922,
}

# Drift arm: the MO side draw's effective buy probability slides linearly
# 0.61 -> 0.50 across the 1500s horizon, i.e. p_buy_drift = -7.333e-5/s.
_DRIFT_P_BUY0 = 0.61
_DRIFT_P_BUY_END = 0.50
_HORIZON = 1500.0
_DRIFT_RATE = (_DRIFT_P_BUY_END - _DRIFT_P_BUY0) / _HORIZON


def _arm(seed: int, *, p_buy: float, drift: float, horizon: float) -> dict[str, Any]:
    cfg = replace(
        santa_fe_config(seed=seed, p_buy=p_buy),
        band=14,
        lo_offset=12,
        hawkes=_spec(),
        lo_offset_gain=80.0,
        touch_pull=0.4,
        cxl_touch_bias=0.03,
        mo_size_pmf=_MO_PMF,
        p_buy_drift=drift,
    )
    sim = ZILobSimulator(cfg)
    while sim.t < horizon:
        sim.step()
    edges = np.linspace(0.0, horizon, 4)
    buckets = []
    for i in range(3):
        lo, hi = float(edges[i]), float(edges[i + 1])
        fills = [tr for tr in sim.trades if lo <= tr.t < hi or (i == 2 and tr.t >= lo)]
        n_buy = sum(1 for tr in fills if tr.aggressor == "buy")
        buckets.append(
            {
                "t_lo": lo,
                "t_hi": hi,
                "n": int(len(fills)),
                "buy_share": float(n_buy / len(fills)) if fills else math.nan,
            }
        )
    n = len(sim.trades)
    return {
        "p_buy0": float(p_buy),
        "p_buy_drift": float(drift),
        "n_fills": int(n),
        "exec_buy_share": float(sum(1 for tr in sim.trades if tr.aggressor == "buy") / n)
        if n
        else math.nan,
        "thirds": buckets,
    }


def _d(a: float, b: float) -> float:
    if b == 0.0:
        return math.inf if a != 0.0 else 0.0
    return (a - b) / abs(b)


def initiative_fade_bench(horizon: float = _HORIZON, seed: int = 13) -> dict[str, Any]:
    """Run the flat/drift arms and seal the receipt."""
    arms = {
        "flat": _arm(seed, p_buy=0.5, drift=0.0, horizon=horizon),
        "drift": _arm(seed, p_buy=_DRIFT_P_BUY0, drift=_DRIFT_RATE, horizon=horizon),
    }
    flat, drift = arms["flat"], arms["drift"]
    flat_fade = flat["thirds"][0]["buy_share"] - flat["thirds"][2]["buy_share"]
    drift_fade = drift["thirds"][0]["buy_share"] - drift["thirds"][2]["buy_share"]
    div = {
        "flat_overall": round(_d(flat["exec_buy_share"], _REAL["exec_buy_share"]), 3),
        "drift_overall": round(_d(drift["exec_buy_share"], _REAL["exec_buy_share"]), 3),
        "drift_early": round(_d(drift["thirds"][0]["buy_share"], _REAL["buy_share_early"]), 3),
        "drift_late": round(_d(drift["thirds"][2]["buy_share"], _REAL["buy_share_late"]), 3),
    }
    # A NaN div entry means the arm never measured the share — log it as
    # unmeasured rather than letting the NaN comparison fail silently.
    divergences = [
        f"{k}={v:+.3f}" if math.isfinite(v) else f"{k}=unmeasured"
        for k, v in div.items()
        if not math.isfinite(v) or abs(v) > 0.05
    ]
    claims = {
        "flat_is_flat": abs(flat_fade) < 0.05,
        "drift_fades": drift_fade > 0.05,
        "drift_early_on_tape": abs(drift["thirds"][0]["buy_share"] - _REAL["buy_share_early"])
        < 0.05,
        "drift_late_on_tape": abs(drift["thirds"][2]["buy_share"] - _REAL["buy_share_late"]) < 0.05,
        "flat_overall_below_tape": flat["exec_buy_share"] < _REAL["exec_buy_share"],
    }
    payload: dict[str, Any] = {
        "schema": INITIATIVE_FADE_SCHEMA,
        "kind": "initiative_fade",
        "horizon": horizon,
        "seed": seed,
        "arms": arms,
        "real_tape_targets": _REAL,
        "divergences": divergences,
        "claims": claims,
        "interpretation": (
            "A flat p_buy arm holds buy-initiated share near 0.50 with no "
            "systematic early->late fade — below the tape's 0.536 overall. "
            "The drift arm (p_buy_drift=-7.333e-5/s, p_eff walking "
            "0.61->0.50 over 1500s) reproduces the fade ordering: the "
            "early third lands near the tape's ~0.61 and the share decays "
            "monotonically into the late third. The late third undershoots "
            "the tape's ~0.50 (realized share lags p_eff as multi-unit "
            "sweeps and queue asymmetry damp buy fills) and the tape's "
            "per-bin path is nonlinear — both residuals are logged as "
            "divergences rather than claimed closed."
        ),
        "git_revision": git_revision(),
        "data_label": "MIXED",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["INITIATIVE_FADE_SCHEMA", "initiative_fade_bench"]
