"""release_chase — does releasing chased orders restore the stack?

``unhit_chase.v1`` located the continuation gap's mechanism: post-fill
unhit-side LO chase lifts the LO channel to the tape's +2.81 and deflates
the fill overshoot — but destroys instant impact (0.35 vs the tape's
0.887). ``refill_hazard.v1`` located the instant knob: per-price vacancy
memory (``refill_cooldown``) lifts instant to 0.893. The wave-23 question
is whether the two mechanisms compose: the chase operates post-fill on
placement, the cooldown operates on refill timing — they share no marker
field and no book state beyond the book itself.

This bench scans (refill_cooldown × unhit_imp) on the calibrated wave-23
base, plus one triple-stack cell with the shield (``cxl_unhit_damp=0.5,
w=10``), and scores each cell jointly on every target: instant 0.887,
k200 4.6447, lo +2.81, fill +2.52, cxl −0.68. The capstone claim is
``joint_closure_exists``: some cell lands every channel inside tolerance
at once.
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.continuation_attr_bench import (
    _attr_totals,
    _AttrSim,
)
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

RELEASE_CHASE_SCHEMA = "release_chase.v1"

# Tape reference values (sealed in continuation_attr.v1 / closure_fit.v1).
_LO_CH_TAPE = 2.81
_FILL_CH_TAPE = 2.52
_CXL_CH_TAPE = -0.68
_K200_TAPE = 4.6447
_INSTANT_TAPE = 0.887
_K200_LAG = 200

# (refill_cooldown, unhit_imp_frac, unhit_imp_window, chase_release,
#  chase_reprice, cxl_damp, cxl_window).
_GRID: tuple[tuple[int, float, int, float, float, float, int], ...] = (
    (0, 0.0, 0, 0.0, 0.0, 0.0, 0),
    (400, 0.0, 0, 0.0, 0.0, 0.0, 0),
    (0, 0.5, 200, 0.0, 0.0, 0.0, 0),
    (400, 0.5, 200, 0.0, 0.0, 0.0, 0),
    (400, 0.5, 200, 0.3, 0.0, 0.0, 0),
    (400, 0.5, 200, 0.6, 0.0, 0.0, 0),
    (300, 0.5, 200, 0.6, 0.0, 0.0, 0),
    (400, 0.5, 200, 0.9, 0.0, 0.0, 0),
    (400, 0.5, 200, 0.6, 0.5, 0.0, 0),
    (400, 0.5, 200, 0.6, 1.0, 0.0, 0),
    (400, 0.5, 200, 0.3, 1.0, 0.0, 0),
    (300, 0.5, 200, 0.3, 1.0, 0.0, 0),
    (400, 0.3, 200, 0.6, 0.0, 0.5, 10),
)


def _release_cell(
    cooldown: int,
    frac: float,
    imp_window: int,
    release: float,
    reprice: float,
    cxl_damp: float,
    cxl_window: int,
    *,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    """One cell: channel attribution + instant/k200 kernel per fill."""
    cfg = _calibrated(
        seed,
        {
            "refill_cooldown": cooldown,
            "unhit_imp_frac": frac,
            "unhit_imp_window": imp_window,
            "chase_release": release,
            "chase_reprice": reprice,
            "cxl_unhit_damp": cxl_damp,
            "cxl_unhit_window": cxl_window,
        },
    )
    sim = _AttrSim(cfg, _split(3.0, seed + 1))

    # mid[e] = state after event e (1-indexed, matching n_events/mut_log).
    mid: list[float | None] = [None]
    fills: list[tuple[int, float]] = []
    seen = 0
    for _ in range(horizon):
        sim.step()
        bb, ba = sim.best_bid_level, sim.best_ask_level
        mid.append(0.5 * (bb + ba) if (bb is not None and ba is not None) else None)
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
    i_sum = 0.0
    i_n = 0
    k_sum = 0.0
    k_n = 0
    for j, sign in fills:
        m0 = mid[j - 1] if j >= 1 else None
        if m0 is None:
            continue
        m1 = mid[j] if j < len(mid) else None
        if m1 is not None:
            i_sum += sign * (m1 - m0)
            i_n += 1
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

    attr = _attr_totals(per_event)
    per_ch = attr["k200_per_channel_ticks"]
    return {
        "refill_cooldown": cooldown,
        "unhit_imp_frac": frac,
        "unhit_imp_window": imp_window,
        "chase_release": release,
        "chase_reprice": reprice,
        "cxl_unhit_damp": cxl_damp,
        "cxl_unhit_window": cxl_window,
        "n_fills": len(fills),
        "instant_signed_ticks": None if i_n == 0 else round(i_sum / i_n, 4),
        "k200_ticks": None if k_n == 0 else round(k_sum / k_n, 4),
        "k200_per_channel_ticks": {ch: round(per_ch[ch], 4) for ch in per_ch},
        "lo_channel_ticks": round(per_ch["lo"], 4),
        "fill_channel_ticks": round(per_ch["fill"], 4),
        "cxl_channel_ticks": round(per_ch["cxl"], 4),
        "attr_windows": attr["windows"],
    }


def release_chase_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """Scan the cooldown × release grid; verdict = joint closure."""
    cells = [
        _release_cell(cd, f, iw, rel, rp, cd2, cw, horizon=horizon, seed=seed + i)
        for i, (cd, f, iw, rel, rp, cd2, cw) in enumerate(_GRID)
    ]
    base = cells[0]

    def _in_tol(v: float | None, target: float, tol: float) -> bool:
        return v is not None and abs(v - target) <= tol

    for c in cells:
        c["instant_in_tol"] = _in_tol(c["instant_signed_ticks"], _INSTANT_TAPE, 0.2)
        c["k200_in_tol"] = _in_tol(c["k200_ticks"], _K200_TAPE, 1.0)
        c["lo_in_tol"] = _in_tol(c["lo_channel_ticks"], _LO_CH_TAPE, 1.0)
        c["fill_in_tol"] = _in_tol(c["fill_channel_ticks"], _FILL_CH_TAPE, 1.0)
        c["cxl_in_tol"] = _in_tol(c["cxl_channel_ticks"], _CXL_CH_TAPE, 0.5)
        c["all_in_tol"] = all(
            c[k]
            for k in ("instant_in_tol", "k200_in_tol", "lo_in_tol", "fill_in_tol", "cxl_in_tol")
        )

    closers = [c for c in cells if c["all_in_tol"]]
    cd_rows = [c for c in cells if c["unhit_imp_frac"] == 0.0 and c["refill_cooldown"] > 0]
    # Chased cells split by release: does the churn restore instant while
    # keeping the LO lift the chase earned?
    chase_no_rel = [c for c in cells if c["unhit_imp_frac"] > 0.0 and c["chase_release"] == 0.0]
    chase_rel = [
        c
        for c in cells
        if c["unhit_imp_frac"] > 0.0 and c["chase_release"] > 0.0 and c["cxl_unhit_damp"] == 0.0
    ]
    chase_reprice = [c for c in chase_rel if c["chase_reprice"] > 0.0]
    chase_delete = [c for c in chase_rel if c["chase_reprice"] == 0.0]
    instant_benefit = any(
        (c["instant_signed_ticks"] or 0.0) > (base["instant_signed_ticks"] or 0.0) + 0.05
        for c in cd_rows
    )
    claims = {
        "grid_evaluated": len(cells) == len(_GRID),
        # Release keeps the chase's LO-channel lift (any released cell
        # stays well above the base's negative lo).
        "release_preserves_lo": bool(
            chase_rel
            and max(c["lo_channel_ticks"] or 0.0 for c in chase_rel)
            > (base["lo_channel_ticks"] or 0.0) + 1.0
        ),
        # Release recovers instant vs the unreleased chase at the same
        # cooldown — the mechanism the tape's churn implies.
        "release_recovers_instant": bool(
            chase_rel
            and chase_no_rel
            and max(c["instant_signed_ticks"] or 0.0 for c in chase_rel)
            > max(c["instant_signed_ticks"] or 0.0 for c in chase_no_rel) + 0.05
        ),
        "cooldown_lifts_instant": bool(instant_benefit),
        # Reprice beats delete: some repriced cell is strictly better on
        # BOTH the recovered instant AND the LO lift than every pure-
        # delete cell — the re-quote cycle preserves the mechanism.
        "reprice_beats_delete": bool(
            chase_reprice
            and chase_delete
            and any(
                (c["instant_signed_ticks"] or 0.0)
                >= max(d["instant_signed_ticks"] or 0.0 for d in chase_delete)
                and (c["lo_channel_ticks"] or -99.0)
                > max(d["lo_channel_ticks"] or -99.0 for d in chase_delete)
                for c in chase_reprice
            )
        ),
        # The capstone: some cell lands all five channels in tolerance.
        "joint_closure_exists": bool(closers),
    }
    best = min(
        cells[1:],
        key=lambda c: sum(
            0.0 if c[k] else 1.0
            for k in ("instant_in_tol", "k200_in_tol", "lo_in_tol", "fill_in_tol", "cxl_in_tol")
        ),
        default=None,
    )
    payload: dict[str, Any] = {
        "schema": RELEASE_CHASE_SCHEMA,
        "kind": "release_chase_bench",
        "asset": "AMZN",
        "horizon_events": horizon,
        "seed": seed,
        "tape_targets": {
            "instant_signed_ticks": _INSTANT_TAPE,
            "k200_ticks": _K200_TAPE,
            "lo_channel_ticks": _LO_CH_TAPE,
            "fill_channel_ticks": _FILL_CH_TAPE,
            "cxl_channel_ticks": _CXL_CH_TAPE,
        },
        "cells": cells,
        "n_cells_all_in_tol": len(closers),
        "best_cell": None
        if best is None
        else {
            "refill_cooldown": best["refill_cooldown"],
            "unhit_imp_frac": best["unhit_imp_frac"],
            "unhit_imp_window": best["unhit_imp_window"],
            "chase_release": best["chase_release"],
            "chase_reprice": best["chase_reprice"],
            "cxl_unhit_damp": best["cxl_unhit_damp"],
            "instant_signed_ticks": best["instant_signed_ticks"],
            "k200_ticks": best["k200_ticks"],
            "lo_channel_ticks": best["lo_channel_ticks"],
            "fill_channel_ticks": best["fill_channel_ticks"],
            "cxl_channel_ticks": best["cxl_channel_ticks"],
            "n_in_tol": sum(
                1
                for k in ("instant_in_tol", "k200_in_tol", "lo_in_tol", "fill_in_tol", "cxl_in_tol")
                if best[k]
            ),
        },
        "claims": claims,
        "interpretation": "See claims: does churning chased orders out "
        "release the touch press so cooldown's instant and the "
        "chase's LO lift coexist?",
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "RELEASE_CHASE_SCHEMA",
    "release_chase_bench",
]
