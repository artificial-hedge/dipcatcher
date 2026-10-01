"""Sealed bench: deep-anchored LO placement (``lo_offset``) vs the tape.

The real AMZN book lives at 9-21 tick spreads ~75% of the time
(spread_dynamics.v1) while the Santa Fe sim hugs 1-2 — deposits anchor
AT the opposite touch, so the near-touch band refills instantly and the
spread can never breathe. ``ZILobConfig.lo_offset`` shifts the anchor
back: deposits land strictly deeper than ``opposite_touch - offset``,
flooring the spread near ``offset + 1`` and letting inside-spread quote
improvements emerge when the book is wide (real: 10.3% of LO submits
land inside the spread — the sim's legacy placement produces zero).

Arms sweep lo_offset over {0, 8, 12} at band 14 and measure the full
occupancy histogram, moments, tight share, and the improvement share.

Evidence: MIXED — synthetic arms vs committed real-tape targets. A
``spread_floor.v1`` sealed receipt records the sweep.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from quant_fund.microstructure.zi_lob_simulator import (
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SPREAD_FLOOR_SCHEMA = "spread_floor.v1"

# Same bucket edges as spread_dynamics.v1.
_BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 13), (14, 21), (22, 34), (35, -1))


def _occupancy(spreads: list[int]) -> dict[str, Any]:
    s = np.asarray(spreads, dtype=np.float64)
    n = float(s.size)
    occ: dict[str, float] = {}
    for lo, hi in _BUCKETS:
        label = f"{lo}-{hi}" if hi > 0 else "35-inf"
        m = (s >= lo) & (s <= hi) if hi > 0 else (s >= lo)
        occ[label] = float(m.sum() / n) if n > 0 else 0.0
    return {
        "n_obs": int(n),
        "mean_spread_ticks": float(s.mean()) if n else 0.0,
        "median_spread_ticks": float(np.median(s)) if n else 0.0,
        "p90_spread_ticks": float(np.quantile(s, 0.9)) if n else 0.0,
        "tight_share_le2": float((s <= 2).mean()) if n else 0.0,
        "spread_occupancy": occ,
    }


def _run_arm(cfg: ZILobConfig, horizon: float) -> dict[str, Any]:
    sim = ZILobSimulator(cfg)
    spreads: list[int] = []
    while sim.t < horizon:
        sim.step()
        sp = sim.spread_ticks
        if sp is not None:
            spreads.append(int(sp))
    counts = sim.event_counts()
    n_lo = max(1, counts["n_lo_arrivals"])
    out = _occupancy(spreads)
    out["lo_improve_share"] = float(counts["n_lo_improve"]) / n_lo
    out["n_events"] = sim.n_events
    out["n_fills"] = counts["n_fills"]
    return out


def spread_floor_bench(horizon: float = 4000.0, seed: int = 7) -> dict[str, Any]:
    """Run the lo_offset sweep; returns the sealed receipt payload."""
    if not isinstance(horizon, (int, float)) or not float(horizon) > 0:
        raise ValueError(f"horizon must be positive, got {horizon!r}")
    base = replace(santa_fe_config(seed=seed), band=14)
    arms = [
        {
            "name": f"offset_{off}",
            "lo_offset": off,
            **_run_arm(replace(base, lo_offset=off), horizon),
        }
        for off in (0, 8, 12)
    ]
    # Committed real-tape targets (spread_dynamics.v1, marketable_limit.v1).
    real: dict[str, Any] = {
        "mean_spread_ticks": 13.086,
        "median_spread_ticks": 13.0,
        "p90_spread_ticks": 19.0,
        "tight_share_le2": 0.0138,
        "lo_improve_share": 0.1031,
        "source_receipts": ["spread_dynamics.v1", "marketable_limit.v1"],
    }
    divergences: list[str] = []
    for a in arms:
        name = a["name"]
        if abs(float(a["mean_spread_ticks"]) - float(real["mean_spread_ticks"])) > 5.0:
            divergences.append(
                f"{name}_mean_{a['mean_spread_ticks']:.3f}_vs_{real['mean_spread_ticks']}"
            )
        if abs(float(a["lo_improve_share"]) - float(real["lo_improve_share"])) > 0.08:
            divergences.append(
                f"{name}_improve_{a['lo_improve_share']:.3f}_vs_{real['lo_improve_share']}"
            )

    payload: dict[str, Any] = {
        "schema": SPREAD_FLOOR_SCHEMA,
        "kind": "spread_floor_bench",
        "horizon": float(horizon),
        "seed": int(seed),
        "band": 14,
        "arms": arms,
        "real_tape_targets": real,
        "divergences": divergences,
        "claims": {
            "legacy_hugs_touch": bool(arms[0]["tight_share_le2"] > 0.5),
            "offset_widens_spread": bool(
                arms[1]["median_spread_ticks"] > arms[0]["median_spread_ticks"] + 5
            ),
            "offset_produces_improvements": bool(arms[1]["lo_improve_share"] > 0.02),
            "wide_occupancy_emerges": bool(
                arms[1]["spread_occupancy"]["9-13"] > 0.05
                or arms[1]["spread_occupancy"]["14-21"] > 0.05
            ),
        },
        "interpretation": (
            "Deep anchoring is the zero-intelligence counterpart of "
            "adverse-selection-aware quoting: makers refuse the touch "
            "and post deeper, flooring the spread near the offset while "
            "MOs still push it wider. The offset arm moves occupancy "
            "out of the 1-2 tick trap toward the tape's 9-21 plateau "
            "and produces inside-spread quote improvements (real: "
            "10.3% of submits) that the touch-anchored placement "
            "structurally cannot. Residual shape gaps (the real "
            "distribution is a plateau, not a point mass) are logged as "
            "divergences — spread *stochasticity* needs a state-"
            "dependent floor, e.g. excitation-driven offsets."
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "MIXED"
    payload["research_only"] = True
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["SPREAD_FLOOR_SCHEMA", "spread_floor_bench"]
