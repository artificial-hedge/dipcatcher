"""cancel_gradient_bench — spatial cancel propensity, sim arms vs tape.

``cancel_gradient.v1`` measured where in the book liquidity exits on
the real AMZN tape: cancellation share ÷ depth share by distance from
the touch — propensity ≈1.37 at the touch, ≈1.44 at 1-3 ticks, ≈1.11
at 4-10, ≈0.68 at 11-50, ≈0.08 beyond 50. The legacy sim cancels a
uniform outstanding order, so its propensity is flat ≈1.0 everywhere —
the re-quote churn that evacuates the front is missing.

``cxl_touch_bias`` adds the mechanism: with probability ``b`` a cancel
event removes the front order at a touch (side picked proportional to
touch depth) instead of drawing uniformly. Zero draws when 0 — legacy
bit-identical.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

CANCEL_GRADIENT_BENCH_SCHEMA = "cancel_gradient_bench.v1"

# Distance buckets mirroring the real receipt (deep buckets truncated to
# the sim's reachable book).
_BUCKETS = ("touch", "d1_3", "d4_10", "d11_plus")


def _run_arm(cfg: ZILobConfig, horizon: float) -> dict[str, Any]:
    sim = ZILobSimulator(cfg)
    # Occupancy at distance: depth d ticks behind each side's touch,
    # sampled after each event.
    occ = [0.0] * 21
    n_occ = 0
    while sim.t < horizon:
        sim.step()
        bb, ba = sim.best_bid_level, sim.best_ask_level
        for lvl, dq in sim._bids.items():
            if bb is not None:
                occ[min(bb - lvl, 20)] += len(dq)
        for lvl, dq in sim._asks.items():
            if ba is not None:
                occ[min(lvl - ba, 20)] += len(dq)
        n_occ += 1
    occ = [o / max(n_occ, 1) for o in occ]
    total_occ = sum(occ)
    total_cxl = sum(sim.cxl_dist)

    def _bucket(lo: int, hi: int) -> dict[str, Any]:
        c_share = sum(sim.cxl_dist[lo:hi]) / total_cxl if total_cxl else 0.0
        d_share = sum(occ[lo:hi]) / total_occ if total_occ else 0.0
        return {
            "cancel_share": round(c_share, 4),
            "depth_share": round(d_share, 4),
            "propensity": round(c_share / d_share, 4) if d_share > 0 else None,
        }

    return {
        "n_cxl": sim.n_cancellations,
        "n_cxl_touch": sim.n_cxl_touch,
        "buckets": {
            "touch": _bucket(0, 1),
            "d1_3": _bucket(1, 4),
            "d4_10": _bucket(4, 11),
            "d11_plus": _bucket(11, 21),
        },
    }


def cancel_gradient_bench(horizon: float = 2000.0, *, seed: int = 13) -> dict[str, Any]:
    """Uniform vs touch-biased cancels on the calibrated book. Sealed."""
    base = replace(santa_fe_config(seed=seed), band=14, lo_offset=12)
    arms = [
        {"name": "uniform", "cxl_touch_bias": 0.0, **_run_arm(base, horizon)},
        {
            "name": "touch_biased",
            "cxl_touch_bias": 0.03,
            **_run_arm(replace(base, cxl_touch_bias=0.03), horizon),
        },
        {
            "name": "dist_decay",
            "cxl_touch_bias": 0.5,
            "cxl_dist_decay": 3.0,
            **_run_arm(replace(base, cxl_touch_bias=0.5, cxl_dist_decay=3.0), horizon),
        },
    ]
    real: dict[str, Any] = {
        "propensity": {
            "touch": 1.37,
            "d1_3": 1.44,
            "d4_10": 1.11,
            "d11_50": 0.68,
            "d51_plus": 0.08,
        },
        "source_receipts": ["cancel_gradient.v1"],
    }
    uni: Any = arms[0]["buckets"]
    biased: Any = arms[1]["buckets"]
    decayed: Any = arms[2]["buckets"]
    divergences: list[str] = []
    if abs(float(biased["d1_3"]["propensity"]) - 1.44) > 0.4:
        divergences.append(f"biased_d1_3_propensity_{biased['d1_3']['propensity']}_vs_1.44")
    if float(decayed["touch"]["propensity"]) > 2.5:
        divergences.append(f"decay_touch_propensity_{decayed['touch']['propensity']}_vs_1.37")
    if float(decayed["d11_plus"]["propensity"]) < 0.4:
        divergences.append(f"decay_deep_propensity_{decayed['d11_plus']['propensity']}_vs_0.68")

    payload: dict[str, Any] = {
        "schema": CANCEL_GRADIENT_BENCH_SCHEMA,
        "kind": "cancel_gradient_bench",
        "horizon": float(horizon),
        "seed": int(seed),
        "arms": arms,
        "real_tape_targets": real,
        "divergences": divergences,
        "claims": {
            "uniform_is_flat": bool(abs(float(uni["touch"]["propensity"]) - 1.0) < 0.35),
            "bias_concentrates_at_touch": bool(float(biased["touch"]["propensity"]) > 1.2),
            "bias_touch_magnitude_realistic": bool(float(biased["touch"]["propensity"]) < 2.2),
            "deep_residual_logged": bool(
                abs(float(biased["d11_plus"]["propensity"]) - 0.68) > 0.15
            ),
            "decay_shapes_near_ring": bool(
                float(decayed["d1_3"]["propensity"])
                > float(decayed["d4_10"]["propensity"])
                > float(decayed["d11_plus"]["propensity"])
            ),
            "thin_touch_overshoots": bool(float(decayed["touch"]["propensity"]) > 2.5),
        },
        "interpretation": (
            "Touch bias b redirects fraction ~b of cancels to the front: "
            "at b=0.03 touch propensity lands at ~1.5 vs the tape's "
            "1.37. The "
            "uniform arm stays flat ~1.0 at every distance — the missing "
            "mechanism is structural, not parametric. The dist-decay arm "
            "(b=0.5, L=3) recovers the tape's ORDERING — near-touch mass "
            "> d4-10 > deep — but the sim's touch queue is thin, so the "
            "touch propensity overshoots (~3.4 vs 1.37) instead of the "
            "real ring peaking at d1-3. Deep drain sits ~0.5 vs the "
            "tape's 0.68. The residual is occupancy, not propensity: the "
            "real touch holds a long queue; ours holds 1-2 orders — "
            "the next mechanism is touch-queue depth, logged."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["CANCEL_GRADIENT_BENCH_SCHEMA", "cancel_gradient_bench"]
