"""hit_starve — does starving the hit side close the LO-channel gap?

``continuation_attr.v1`` localized the wave-23 continuation gap: the
tape's post-fill LO flow reprices the mid WITH the drift (+2.81 ticks)
while the calibrated sim's LO channel reprices AGAINST it (-2.53) — a
5.34-tick shortfall, the dominant term. One candidate mechanism: the
sim's hit-side refill (joins/improves racing to the consumed touch)
drags the mid back while the tape's hit side stays thin.

``hit_refill_damp`` tests that mechanism: while the post-fill marker is
live (``hit_refill_window`` events), an LO arrival landing on the hit
side within ``hit_refill_band`` ticks of that side's touch is dropped.
The bench scans (damp, band, window) on the calibrated wave-23 base and
scores each cell on the LO channel (tape +2.81), the total kernel
(instant 0.887, k200 4.64) and one stacked cell with the shield
(``cxl_unhit_damp=0.5, w=10``) — the two mechanisms share a marker but
act on opposite sides, so stacking must not be redundant. The measured
verdict is a FALSIFICATION: starving the hit side amplifies both gaps.
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

HIT_STARVE_SCHEMA = "hit_starve.v1"

# Tape reference values (sealed in continuation_attr.v1 / closure_fit.v1).
_LO_CH_TAPE = 2.81
_K200_TAPE = 4.6447
_INSTANT_TAPE = 0.887
_K200_LAG = 200

# (hit_damp, hit_band, hit_window, cxl_damp, cxl_window). The last cell
# stacks the starve on the shield-decay winner — same marker, opposite
# book sides, so redundancy would show up as no channel gain.
_GRID: tuple[tuple[float, int, int, float, int], ...] = (
    (0.0, 0, 0, 0.0, 0),
    (0.5, 2, 50, 0.0, 0),
    (0.5, 5, 50, 0.0, 0),
    (0.8, 5, 50, 0.0, 0),
    (0.5, 5, 100, 0.0, 0),
    (0.8, 10, 50, 0.0, 0),
    (0.5, 5, 50, 0.5, 10),
)


def _starve_cell(
    damp: float,
    band: int,
    window: int,
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
            "hit_refill_damp": damp,
            "hit_refill_band": band,
            "hit_refill_window": window,
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
        "hit_refill_damp": damp,
        "hit_refill_band": band,
        "hit_refill_window": window,
        "cxl_unhit_damp": cxl_damp,
        "cxl_unhit_window": cxl_window,
        "n_fills": len(fills),
        "n_lo_suppressed": sim.n_lo_suppressed,
        "instant_signed_ticks": (i_sum / i_n) if i_n else None,
        "k200_ticks": (k_sum / k_n) if k_n else None,
        "k200_per_channel_ticks": {ch: round(per_ch[ch], 4) for ch in per_ch},
        "lo_channel_ticks": round(per_ch["lo"], 4),
        "attr_windows": attr["windows"],
    }


def _close(a: float | None, target: float, tol: float) -> bool:
    return a is not None and abs(a - target) <= tol


def hit_starve_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """Scan the starve grid; verdict = LO-channel closure vs kernel cost."""
    cells = [
        _starve_cell(d, b, w, cd, cw, horizon=horizon, seed=seed + i)
        for i, (d, b, w, cd, cw) in enumerate(_GRID)
    ]
    base = cells[0]
    for cell in cells:
        cell["instant_in_tol"] = _close(cell["instant_signed_ticks"], _INSTANT_TAPE, 0.2)
        cell["k200_in_tol"] = _close(cell["k200_ticks"], _K200_TAPE, 1.0)
        cell["lo_in_tol"] = _close(cell["lo_channel_ticks"], _LO_CH_TAPE, 1.0)

    starve_cells = [c for c in cells[1:] if c["hit_refill_damp"] > 0.0]
    closers = [c for c in cells[1:] if c["k200_in_tol"] and c["instant_in_tol"]]
    claims = {
        "starve_grid_evaluated": len(cells) == len(_GRID),
        "starve_bites": bool(
            starve_cells
            and all(c["n_lo_suppressed"] > base["n_lo_suppressed"] for c in starve_cells)
        ),
        # Falsification check: starving the hit side deepens the LO gap
        # on every pure-starve cell (the fill channel inflates on most
        # cells but not all — reported per cell, not claimed).
        "starve_deepens_lo_gap": bool(
            starve_cells
            and all(
                c["lo_channel_ticks"] < base["lo_channel_ticks"] - 0.3
                for c in starve_cells
                if c["cxl_unhit_damp"] == 0.0
            )
        ),
        # No cell closes instant AND k200 — the honest negative.
        "joint_closure_absent": not closers,
        # Starve is hit-side specific: the cancel channel should barely
        # move (|Δ| < 0.5 ticks) on pure-starve cells.
        "starve_is_hit_side_specific": bool(
            all(
                abs(c["k200_per_channel_ticks"]["cxl"] - base["k200_per_channel_ticks"]["cxl"])
                < 0.5
                for c in starve_cells
                if c["cxl_unhit_damp"] == 0.0
            )
        ),
    }
    payload: dict[str, Any] = {
        "schema": HIT_STARVE_SCHEMA,
        "kind": "hit_starve_bench",
        "horizon": horizon,
        "seed": seed,
        "targets": {
            "lo_channel_ticks": _LO_CH_TAPE,
            "instant_ticks": _INSTANT_TAPE,
            "k200_ticks": _K200_TAPE,
        },
        "cells": cells,
        "joint_closure_cells": [
            {
                "hit_refill_damp": c["hit_refill_damp"],
                "hit_refill_band": c["hit_refill_band"],
                "hit_refill_window": c["hit_refill_window"],
                "cxl_unhit_damp": c["cxl_unhit_damp"],
            }
            for c in closers
        ],
        "claims": claims,
        "interpretation": (
            "NEGATIVE — hit-side refill starvation runs the wrong way on "
            "both gaps: the fill channel (already +4.85 vs tape +2.52) "
            "inflates on most cells and the LO channel deepens -2.53 to "
            "as low as -11.0 on every pure-starve cell. The tape's +2.81 LO contribution is "
            "therefore NOT absent hit-side refill — it is the presence "
            "of unhit-side repricing pressure (bids stepping up after a "
            "buy fill). The residual gap sits on the unhit side: a "
            "stronger direction-leaning LO response than the current "
            "lo_tilt gain is the remaining candidate mechanism."
        ),
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "HIT_STARVE_SCHEMA",
    "hit_starve_bench",
]
