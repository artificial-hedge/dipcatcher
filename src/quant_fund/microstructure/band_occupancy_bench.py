"""band_occupancy — does the floored sim LIVE in the band?

``spread_dynamics.v1`` measured the tape's spread occupancy: AMZN
spends ~75% of the session at 9-21 ticks and only 1.4% below 2 —
the *time-share* fact, distinct from the per-draw mean the pin
suite uses. ``floor_rate.v1`` showed the floor opens the band on
most draws — but opening is not occupying: a cell whose mean sits
at 14 while the trajectory bounces 2 <-> 30 does not reproduce the
tape's standing spread.

This bench measures the occupancy profile itself on the floored
cells — per-step spread series binned on the tape's own histogram
grid — against the tape's committed profile (spread_dynamics_amzn):
share inside [9,63], tight-share <=2, median. The question is not
"does the floor widen the mean" but "does the floored book spend
the tape's share of time at tape width".

Evidence class: research / MIXED (sim cells vs committed tape pins).
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.full_impact_bench import _FULL
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.spread_dynamics import _spread_profile
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

BAND_OCCUPANCY_SCHEMA = "band_occupancy.v1"

# Tape occupancy reference from spread_dynamics_amzn.json (AMZN
# 2012-06-21): mean 13.086, median 13, tight_share_le2 0.0138,
# share in [9,63] ~ 0.80 (9-13: .358 + 14-21: .395 + 22-34: .044
# + 35+: .007), share in [9,21] ~ 0.753.
_TAPE = {
    "mean_spread_ticks": 13.086,
    "median_spread_ticks": 13.0,
    "tight_share_le2": 0.0138,
    "share_9_63": 0.8039,
    "share_9_21": 0.7532,
}

# (label, floor, repost_frac, fill_repost_delay, flow_intensity)
_CELLS: tuple[tuple[str, int, float, int, float | None], ...] = (
    ("g0_split", 0, 0.0, 160, 2.0),
    ("g10_d280_split", 10, 0.6, 280, 2.0),
    ("g12_d280_split", 12, 0.6, 280, 2.0),
    ("g12_rp90_d400_iid", 12, 0.9, 400, None),
    ("g12_d280_iid", 12, 0.6, 280, None),
)


def _sim_spread_profile(
    extra: dict[str, Any], *, horizon: int, seed: int, inten: float | None
) -> dict[str, Any]:
    """Per-step spread series under a floored config -> occupancy."""
    cfg = _calibrated(seed, extra)
    flow = _split(inten, seed + 1) if inten is not None else None
    sim = ZILobSimulator(cfg, flow)
    spreads: list[float] = []
    for _ in range(horizon):
        sim.step()
        if sim.spread_ticks is not None:
            spreads.append(float(sim.spread_ticks))
    prof = _spread_profile(spreads)
    if prof.get("ok") is False:
        return prof
    arr_share_9_63 = sum(1 for s in spreads if 9.0 <= s <= 63.0) / len(spreads)
    prof["share_9_63"] = round(arr_share_9_63, 4)
    prof["share_9_21"] = round(sum(1 for s in spreads if 9.0 <= s <= 21.0) / len(spreads), 4)
    return prof


def band_occupancy_bench(*, horizon: int = 12000, seed: int = 7) -> dict[str, Any]:
    """occupancy profile of the floored cells vs the tape profile."""
    cells: list[dict[str, Any]] = []
    for i, (label, floor, rp, delay, inten) in enumerate(_CELLS):
        extra = dict(
            _FULL,
            min_quote_dist=floor,
            fill_repost_delay=delay,
            repost_frac=rp,
            repost_band=4,
            repost_window=500,
        )
        prof = _sim_spread_profile(extra, horizon=horizon, seed=seed + i, inten=inten)
        prof["regime"] = label
        prof["min_quote_dist"] = floor
        cells.append(prof)

    divergences: list[str] = []
    for c in cells:
        if c.get("share_9_63") is not None and abs(c["share_9_63"] - _TAPE["share_9_63"]) > 0.2:
            divergences.append(f"{c['regime']}:band_share_off_tape")
        if c.get("tight_share_le2") is not None and c["tight_share_le2"] > 0.15:
            divergences.append(f"{c['regime']}:tight_share_high")

    claims = {
        "cells_measured": all(c.get("n_obs", 0) > 0 for c in cells),
        # Some floored cell spends at least the tape's share (minus
        # 0.15 slack) of steps inside [9,63].
        "floor_reaches_occupancy": any(
            c.get("share_9_63") is not None
            and c["share_9_63"] >= _TAPE["share_9_63"] - 0.15
            and c["min_quote_dist"] > 0
            for c in cells
        ),
        # Some floored cell's tight share <= 10x the tape's — the
        # floor clears out the 1-2-tick regime the unmodified sim
        # lives in.
        "tight_regime_cleared": any(
            c.get("tight_share_le2") is not None
            and c["tight_share_le2"] <= 10 * _TAPE["tight_share_le2"]
            and c["min_quote_dist"] > 0
            for c in cells
        ),
        # The median spread lands within +-6 ticks of the tape's 13
        # on some floored cell.
        "median_on_tape": any(
            c.get("median_spread_ticks") is not None
            and abs(c["median_spread_ticks"] - 13.0) <= 6.0
            and c["min_quote_dist"] > 0
            for c in cells
        ),
    }
    body: dict[str, Any] = {
        "schema": BAND_OCCUPANCY_SCHEMA,
        "kind": "sim_vs_real",
        "git_revision": git_revision(),
        "research_only": True,
        "data_label": "MIXED",
        "horizon": horizon,
        "seed": seed,
        "tape_reference": _TAPE,
        "cells": cells,
        "divergences": divergences,
        "claims": claims,
        "notes": (
            "The tape's spread fact is an OCCUPANCY fact — AMZN "
            "spends ~75% of the session at 9-21 ticks (1.4% below 2). "
            "floor_rate showed the floor opens the band on most "
            "draws; this measures whether the floored book LIVES "
            "there — per-step occupancy on the tape's own bin grid "
            "for the corner (g10) and structural (g12) cells under "
            "both flows. Measured: ALL THREE occupancy claims FALSE "
            "— the floored book spends 0-20% of steps in-band (tape "
            "80%) and still sits at <=2 ticks 26-75% of the time. "
            "The floor opens the MEAN but not the OCCUPANCY: band "
            "episodes are excursions, not a standing regime. This is "
            "the sharpest wave-24 falsification — the tape's spread "
            "is a persistent state the floor can't hold."
        ),
    }
    body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return body
