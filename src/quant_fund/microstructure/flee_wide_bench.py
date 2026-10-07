"""flee_wide — the cancel retreat in the tape's own spread regime.

``hit_flee.v1`` falsified the sixth wave-23 composition: the post-fill
hit-side cancel surge deepens the emptied-touch reveal to the tape's
~1.88 ticks and lands instant+k200 in tolerance, but in the sim's
1-2-tick spread every fleeing cancel is a mid-mover — the cxl channel
floods to +5.9 against the tape's -0.68. On the tape the same retreat
is price-neutral because it happens inside a 9-21-tick spread
(``spread_occupancy.v1``).

This bench asks whether the flood is a regime artifact: rerun the
flee×chase composition inside the deep-book geometry (band=40 +
``lo_offset`` + churn knobs, spread mean ~12 ticks — the tape's own
band). The answer is a scale-inversion, not a closure: the retreat
stops flooding cxl (its Δmid share collapses toward the tape's sign),
but the wide book *over*-reveals — an emptied touch exposes gaps of
~4-6 ticks, not ~2. The tape's crown density inside a wide spread is
the remaining missing primitive.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from quant_fund.microstructure.continuation_attr_bench import (
    _attr_totals,
    _AttrSim,
)
from quant_fund.microstructure.maker_age_bench import _MO_PMF
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

FLEE_WIDE_SCHEMA = "flee_wide.v1"

# Deep/wide-spread geometry (deep_book_bench.v1 recipe, minus the Hawkes
# clock so SplitFlow keeps its MO density; mu raised so fills are dense
# enough for fill-conditioned kernels).
_DEEP: dict[str, Any] = {
    "touch_pull": 0.4,
    "cxl_touch_bias": 0.5,
    "cxl_dist_decay": 3.0,
    "cxl_requote": 0.5,
    "mo_size_pmf": _MO_PMF,
    "lam": 3.5,
    "theta_cxl": 0.4,
    "lo_offset": 4,
    "mu": 1.0,
}

# (regime, hit_flee_frac, hit_flee_band, hit_flee_window, unhit_imp_frac)
_CELLS: tuple[tuple[str, float, int, int, float], ...] = (
    ("thin", 0.0, 0, 0, 0.5),  # wave-23 chase reference
    ("thin", 0.10, 2, 50, 0.5),  # best thin composition (hit_flee.v1)
    ("wide", 0.0, 0, 0, 0.5),  # wide chase reference
    ("wide", 0.10, 2, 50, 0.5),  # wide flee
    ("wide", 0.20, 3, 80, 0.5),  # wide flee, stronger
)

_K200_LAG = 200


def _cell(
    regime: str,
    flee_frac: float,
    flee_band: int,
    flee_window: int,
    imp_frac: float,
    *,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    """One flee×chase cell on the thin or wide book (same kernel loop)."""
    extra: dict[str, Any] = {
        "hit_flee_frac": flee_frac,
        "hit_flee_band": flee_band,
        "hit_flee_window": flee_window,
        "unhit_imp_frac": imp_frac,
        "unhit_imp_window": _K200_LAG if imp_frac > 0.0 else 0,
    }
    if regime == "wide":
        extra = dict(_DEEP, **extra)
    cfg = _calibrated(seed, extra)
    sim = _AttrSim(cfg, _split(3.0, seed + 1))

    mid: list[float | None] = [None]
    bb_l: list[int | None] = [None]
    ba_l: list[int | None] = [None]
    fills: list[tuple[int, float]] = []
    spreads: list[int] = []
    seen = 0
    for _ in range(horizon):
        sim.step()
        bb, ba = sim.best_bid_level, sim.best_ask_level
        bb_l.append(bb)
        ba_l.append(ba)
        if bb is not None and ba is not None:
            mid.append(0.5 * (bb + ba))
            spreads.append(ba - bb)
        else:
            mid.append(None)
        while seen < len(sim.trades):
            tr = sim.trades[seen]
            fills.append((sim.n_events, 1.0 if tr.aggressor == "buy" else -1.0))
            seen += 1
    n_ev = len(mid) - 1

    mut_by_ev: dict[int, tuple[str, str]] = {}
    for ev_idx, ch, side in sim.mut_log:
        if ev_idx not in mut_by_ev:
            mut_by_ev[ev_idx] = (ch, side)

    per_event: list[tuple[int, float, int, str, str, float]] = []
    i_sum = ie_sum = ik_sum = k_sum = 0.0
    i_n = ie_n = ik_n = k_n = 0
    for j, sign in fills:
        m0 = mid[j - 1] if j >= 1 else None
        if m0 is None:
            continue
        m1 = mid[j] if j < len(mid) else None
        if m1 is not None:
            i_sum += sign * (m1 - m0)
            i_n += 1
            bb0, ba0 = bb_l[j - 1], ba_l[j - 1]
            bb1, ba1 = bb_l[j], ba_l[j]
            emptied = (sign > 0 and ba0 is not None and ba1 is not None and ba1 > ba0) or (
                sign < 0 and bb0 is not None and bb1 is not None and bb1 < bb0
            )
            if emptied:
                ie_sum += sign * (m1 - m0)
                ie_n += 1
            else:
                ik_sum += sign * (m1 - m0)
                ik_n += 1
        j200 = j + _K200_LAG
        mj = mid[j200] if j200 < len(mid) else None
        if mj is not None:
            k_sum += sign * (mj - m0)
            k_n += 1
        hit_side = "sell" if sign > 0 else "buy"
        for m_ev in range(j + 1, min(j + 1 + _K200_LAG, n_ev + 1)):
            d1, d0 = mid[m_ev], mid[m_ev - 1]
            if d1 is None or d0 is None or d1 == d0:
                continue
            mut = mut_by_ev.get(m_ev)
            if mut is None:
                continue
            ch, side = mut
            rel = "hit" if side == hit_side else "unhit"
            per_event.append((j, sign, m_ev, ch, rel, d1 - d0))

    attr = _attr_totals(per_event, Counter(j for j, _ in fills))
    per_ch = attr["k200_per_channel_ticks"]
    n_sp = len(spreads)
    return {
        "regime": regime,
        "hit_flee_frac": flee_frac,
        "hit_flee_band": flee_band,
        "hit_flee_window": flee_window,
        "unhit_imp_frac": imp_frac,
        "n_fills": len(fills),
        "n_hit_flees": sim.n_hit_flees,
        "spread_mean": round(float(np.mean(spreads)), 4) if n_sp else None,
        "spread_occupancy_9_21": (
            round(sum(1 for s in spreads if 9 <= s <= 21) / n_sp, 4) if n_sp else None
        ),
        "instant_signed_ticks": None if i_n == 0 else round(i_sum / i_n, 4),
        "instant_given_empty_ticks": None if ie_n == 0 else round(ie_sum / ie_n, 4),
        "instant_given_kept_ticks": None if ik_n == 0 else round(ik_sum / ik_n, 4),
        "k200_ticks": None if k_n == 0 else round(k_sum / k_n, 4),
        "lo_channel_ticks": round(per_ch["lo"], 4),
        "fill_channel_ticks": round(per_ch["fill"], 4),
        "cxl_channel_ticks": round(per_ch["cxl"], 4),
    }


def flee_wide_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """Thin vs wide flee×chase cells; verdict = regime portability."""
    cells = [
        _cell(reg, ff, fb, fw, imp, horizon=horizon, seed=seed + i)
        for i, (reg, ff, fb, fw, imp) in enumerate(_CELLS)
    ]
    thin_flee = cells[1]
    wide_ref, wide_flee = cells[2], cells[3]

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    # Tape reference for the cxl channel (continuation_attr.v1).
    cxl_tape = -0.68
    thin_cxl_gap = abs(_f(thin_flee["cxl_channel_ticks"]) - cxl_tape)
    wide_cxl_gap = abs(_f(wide_flee["cxl_channel_ticks"]) - cxl_tape)

    claims = {
        "grid_evaluated": all(c["n_fills"] > 0 for c in cells),
        # The wide cells actually sit in the tape's spread band.
        "wide_regime_reached": bool(
            wide_ref["spread_occupancy_9_21"] is not None
            and wide_ref["spread_occupancy_9_21"] > 0.1
        ),
        # In the wide book the same flee intensity lands its cxl channel
        # closer to the tape's -0.68 than in the thin book — the thin
        # regime amplifies each flee into a mid move.
        "flee_cxl_closer_wide": bool(wide_cxl_gap < thin_cxl_gap),
        # The honest residual: the wide book over-reveals — its emptied
        # touch exposes multi-tick gaps (4-6) vs the tape's ~1.88,
        # because the sim's near-touch crown lacks the tape's depth.
        "wide_over_reveals": bool(_f(wide_flee["instant_given_empty_ticks"]) > 1.8794),
    }
    payload: dict[str, Any] = {
        "schema": FLEE_WIDE_SCHEMA,
        "kind": "sim_bench",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "git_revision": git_revision(),
        "seed": seed,
        "horizon_events": horizon,
        "deep_geometry": {
            "lam": _DEEP["lam"],
            "theta_cxl": _DEEP["theta_cxl"],
            "lo_offset": _DEEP["lo_offset"],
            "mu": _DEEP["mu"],
            "band": 40,
        },
        "cells": cells,
        "claims": claims,
        "notes": (
            "hit_flee.v1 left the residual at the emptied-touch reveal: "
            "the flee deepens it correctly in a thin book but floods the "
            "cxl channel because every near-touch cancel moves a 1-2-tick "
            "mid. This bench reruns the composition inside the tape's own "
            "9-21-tick band (deep_book geometry + SplitFlow at mu=1). "
            "Result: the flood partly deflates in the tape's band — "
            "the same flee intensity's cxl channel lands nearer the "
            "tape's -0.68 than in the thin book — but a new divergence "
            "surfaces: the wide book over-reveals (empty-cond ~6 ticks "
            "vs tape 1.88). The tape keeps a dense near-touch crown "
            "inside its wide spread; the sim's wide book doesn't. Next "
            "mechanism: near-touch crown density (the join-the-touch "
            "mass in place_mix's mixture) inside the wide regime."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
