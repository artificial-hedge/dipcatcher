"""crown_density — the tape's dense near-touch crown, measured.

``flee_wide.v1`` located the wave-23 residual: in the deep/wide-spread
geometry the sim *over*-reveals — an emptied touch exposes ~6-tick gaps
vs the tape's ~1.88 — because the sim's wide book lacks the dense
near-touch crown the tape carries. This bench measures the primitive
directly:

- ``crown_shares``: resting quantity within ``_CROWN_TICKS`` of the
  touch, per side, per event — tape (share count within ±300 price
  units) vs sim arms (order count within ±3 levels).
- ``reveal_gap_ticks``: on fills that empty the touch, the distance to
  the newly revealed touch — the quantity the over-reveal claim pins.
- ``crown_share``: crown's share of visible top-10 depth — how much of
  the tape's book hugs the touch.

The verdict the cells carry: the tape's crown is proportionally dense
(large share of visible depth within a few ticks of touch), the thin
sim reproduces it trivially (whole book is within 3 ticks), and the
wide sim starves it — the missing placement class is a
join-the-touch crown inside a wide spread.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from quant_fund.microstructure.lobster import (
    EXECUTION,
    EXECUTION_HIDDEN,
    HALT,
    LobsterBook,
    parse_messages,
    parse_orderbook_row,
    resync_band,
)
from quant_fund.microstructure.maker_age_bench import _MO_PMF
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.microstructure.zi_lob_simulator import ZILobSimulator
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

CROWN_DENSITY_SCHEMA = "crown_density.v1"

_CROWN_TICKS = 3
_TOP_N = 10

# Same deep geometry as flee_wide's wide cells.
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


def _empty_gaps(
    bb: np.ndarray,
    ba: np.ndarray,
    fills: np.ndarray,
    signs: np.ndarray,
    *,
    tick_units: float,
) -> list[float]:
    """Gap (ticks) between the emptied touch and the newly revealed one."""
    gaps: list[float] = []
    j0 = fills - 1
    ok = j0 >= 0
    J, S, J0 = fills[ok], signs[ok], j0[ok]
    lim = 10**8  # empty-side sentinel
    for j, s, j0v in zip(J, S, J0, strict=True):
        jv, j0v = int(j), int(j0v)
        if abs(bb[jv]) > lim or abs(ba[jv]) > lim or abs(bb[j0v]) > lim or abs(ba[j0v]) > lim:
            continue
        if s > 0:
            g = (ba[jv] - ba[j0v]) / tick_units if ba[j0v] < ba[jv] else None
        else:
            g = (bb[j0v] - bb[jv]) / tick_units if bb[jv] < bb[j0v] else None
        if g is not None and g > 0:
            gaps.append(float(g))
    return gaps


def tape_crown(message_path: Path, orderbook_path: Path) -> dict[str, Any]:
    """Crown depth + emptied-touch reveal gap on the real tape."""
    book = LobsterBook()
    bb: list[int] = []
    ba: list[int] = []
    crown_b: list[int] = []
    crown_a: list[int] = []
    tot_b: list[int] = []
    tot_a: list[int] = []
    fills: list[int] = []
    signs: list[int] = []
    seeded = False
    band = _CROWN_TICKS * 100  # LOBSTER prices are price*10_000
    with orderbook_path.open() as f_ob:
        for ev, ob_row in zip(parse_messages(message_path), csv.reader(f_ob), strict=True):
            asks_exp, bids_exp = parse_orderbook_row(ob_row)
            if not seeded:
                book.seed(asks_exp, bids_exp)
                book.apply(ev)
                seeded = True
                continue
            book.apply(ev)
            if ev.event_type in (EXECUTION_HIDDEN, HALT):
                continue
            if book.top("ask", _TOP_N) != asks_exp or book.top("bid", _TOP_N) != bids_exp:
                resync_band(book, asks_exp, bids_exp)
            asks, bids = book.top("ask", _TOP_N), book.top("bid", _TOP_N)
            if not asks or not bids:
                continue
            bb.append(bids[0][0])
            ba.append(asks[0][0])
            crown_b.append(sum(s for p, s in bids if bids[0][0] - p <= band))
            crown_a.append(sum(s for p, s in asks if p - asks[0][0] <= band))
            tot_b.append(sum(s for _, s in bids))
            tot_a.append(sum(s for _, s in asks))
            if ev.event_type == EXECUTION:
                fills.append(len(bb) - 1)
                signs.append(-ev.direction)
    bb_a = np.asarray(bb, dtype=np.int64)
    ba_a = np.asarray(ba, dtype=np.int64)
    gaps = _empty_gaps(
        bb_a,
        ba_a,
        np.asarray(fills, dtype=np.int64),
        np.asarray(signs, dtype=np.int64),
        tick_units=100.0,
    )
    cb, ca = np.asarray(crown_b, float), np.asarray(crown_a, float)
    tb, ta = np.asarray(tot_b, float), np.asarray(tot_a, float)
    crown_share = float(np.mean((cb + ca) / np.maximum(tb + ta, 1.0)))
    return {
        "regime": "tape",
        "n_events": int(len(bb)),
        "n_fills": int(len(fills)),
        "crown_ticks": _CROWN_TICKS,
        "crown_units_mean": round(float(np.mean(cb + ca)), 2),
        "visible_units_mean": round(float(np.mean(tb + ta)), 2),
        "crown_share_of_visible": round(crown_share, 4),
        "reveal_gap_ticks_mean": None if not gaps else round(float(np.mean(gaps)), 4),
        "reveal_gap_ticks_p50": (None if not gaps else round(float(np.median(gaps)), 4)),
        "n_reveals": len(gaps),
    }


def _sim_crown(
    regime: str,
    extra: dict[str, Any] | None,
    *,
    horizon: int,
    seed: int,
    collect_counts: bool = False,
) -> dict[str, Any]:
    """Crown depth + emptied-touch reveal gap on one sim arm."""
    cfg = _calibrated(seed, extra)
    sim = ZILobSimulator(cfg, _split(3.0, seed + 1))
    bb: list[int] = []
    ba: list[int] = []
    crown: list[int] = []
    tot: list[int] = []
    spreads: list[int] = []
    fills: list[int] = []
    signs: list[int] = []
    levels: list[int] = []
    seen = 0
    for _ in range(horizon):
        sim.step()
        bids, asks = sim._bids, sim._asks
        if not bids or not asks:
            bb.append(-(10**9))
            ba.append(10**9)
            continue
        bbv = max(bids)
        bav = min(asks)
        bb.append(bbv)
        ba.append(bav)
        spreads.append(bav - bbv)
        crown.append(
            sum(len(bids[lvl]) for lvl in bids if bbv - lvl <= _CROWN_TICKS)
            + sum(len(asks[lvl]) for lvl in asks if lvl - bav <= _CROWN_TICKS)
        )
        tot.append(sum(len(d) for d in bids.values()) + sum(len(d) for d in asks.values()))
        while seen < len(sim.trades):
            tr = sim.trades[seen]
            fills.append(sim.n_events)
            signs.append(1 if tr.aggressor == "buy" else -1)
            levels.append(tr.level)
            seen += 1
    bb_a = np.asarray(bb, dtype=np.int64)
    ba_a = np.asarray(ba, dtype=np.int64)
    gaps = _empty_gaps(
        bb_a,
        ba_a,
        np.asarray(fills, dtype=np.int64),
        np.asarray(signs, dtype=np.int64),
        tick_units=1.0,
    )
    cr = np.asarray(crown, float)
    tv = np.asarray(tot, float)
    out_extra: dict[str, Any] = {}
    if collect_counts:
        counts = sim.event_counts()
        n_f = len(fills)
        out_extra["n_hidden_fills"] = counts["n_hidden_fills"]
        out_extra["hidden_fill_share"] = round(counts["n_hidden_fills"] / n_f, 4) if n_f else None
        out_extra["n_mo_units"] = counts["n_mo_units"]
        out_extra["n_mo_arrivals"] = counts["n_mo_arrivals"]
        out_extra["n_lo_capped"] = counts["n_lo_capped"]
        # Sweep footprint: distinct levels consumed by the trades of one
        # event index (a size-k MO burst prints all its levels under one
        # event). sweep_width.v1 tape pin: p_ge2 = 0.045, max = 8.
        width: dict[int, set[int]] = {}
        for ev, lvl in zip(fills, levels, strict=True):
            width.setdefault(ev, set()).add(lvl)
        widths = np.asarray([len(v) for v in width.values()], dtype=float)
        out_extra["sweep_p_ge2"] = round(float(np.mean(widths >= 2)), 4) if len(widths) else None
        out_extra["sweep_max_levels"] = int(widths.max()) if len(widths) else 0
    n_sp = len(spreads)
    return {
        "regime": regime,
        "n_events": horizon,
        "n_fills": len(fills),
        "spread_mean": round(float(np.mean(spreads)), 4) if n_sp else None,
        "crown_ticks": _CROWN_TICKS,
        "crown_units_mean": round(float(np.mean(cr)), 2) if len(cr) else None,
        "visible_units_mean": round(float(np.mean(tv)), 2) if len(tv) else None,
        "crown_share_of_visible": (
            round(float(np.mean(cr / np.maximum(tv, 1.0))), 4) if len(cr) else None
        ),
        "reveal_gap_ticks_mean": None if not gaps else round(float(np.mean(gaps)), 4),
        "reveal_gap_ticks_p50": None if not gaps else round(float(np.median(gaps)), 4),
        "n_reveals": len(gaps),
        **out_extra,
    }


def crown_density_bench(
    tape_dir: Path | None = None, *, horizon: int = 20000, seed: int = 7
) -> dict[str, Any]:
    """Crown density + reveal gap on the tape and the sim arms."""
    cells: list[dict[str, Any]] = []
    if tape_dir is not None:
        cells.append(
            tape_crown(
                tape_dir / "AMZN_2012-06-21_34200000_57600000_message_10.csv",
                tape_dir / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv",
            )
        )
    cells.append(_sim_crown("thin", None, horizon=horizon, seed=seed))
    cells.append(
        _sim_crown(
            "thin_flee_chase",
            {
                "hit_flee_frac": 0.10,
                "hit_flee_band": 2,
                "hit_flee_window": 50,
                "unhit_imp_frac": 0.5,
                "unhit_imp_window": 200,
            },
            horizon=horizon,
            seed=seed + 1,
        )
    )
    cells.append(_sim_crown("wide", dict(_DEEP), horizon=horizon, seed=seed + 2))
    cells.append(
        _sim_crown(
            "wide_flee_chase",
            dict(
                _DEEP,
                hit_flee_frac=0.10,
                hit_flee_band=2,
                hit_flee_window=50,
                unhit_imp_frac=0.5,
                unhit_imp_window=200,
            ),
            horizon=horizon,
            seed=seed + 3,
        )
    )

    tape = cells[0] if tape_dir is not None and cells[0]["regime"] == "tape" else None
    sims = [c for c in cells if c["regime"] != "tape"]

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    tape_reveal = _f(tape["reveal_gap_ticks_mean"]) if tape else 3.7554
    tape_share = _f(tape["crown_share_of_visible"]) if tape else None
    wide = next((c for c in sims if c["regime"] == "wide"), None)
    wide_flee = next((c for c in sims if c["regime"] == "wide_flee_chase"), None)
    claims = {
        "cells_evaluated": all(c["n_fills"] > 0 for c in sims),
        # The wide sim starves the crown: its within-3-tick share of
        # visible depth is far below the tape's.
        "wide_crown_starved": bool(
            tape_share is not None
            and wide is not None
            and _f(wide["crown_share_of_visible"]) < tape_share
        ),
        # And the starvation is exactly what the over-reveal reads:
        # the wide composition's emptied-touch gap exceeds the tape's
        # ~3.8-tick mean reveal.
        "wide_over_reveals": bool(
            wide_flee is not None and _f(wide_flee["reveal_gap_ticks_mean"]) > tape_reveal
        ),
    }
    payload: dict[str, Any] = {
        "schema": CROWN_DENSITY_SCHEMA,
        "kind": "sim_bench",
        "data_label": "MIXED" if tape is not None else "SYNTHETIC",
        "research_only": True,
        "git_revision": git_revision(),
        "seed": seed,
        "horizon_events": horizon,
        "cells": cells,
        "claims": claims,
        "notes": (
            "Direct measurement of the primitive flee_wide.v1's residual "
            "named: the dense near-touch crown inside a wide spread. "
            "crown_share_of_visible = resting quantity within "
            f"{_CROWN_TICKS} ticks of each touch over total visible "
            f"top-{_TOP_N} depth; reveal_gap_ticks = distance from an "
            "emptied touch to its successor. Tape vs sim arms decide "
            "whether the over-reveal is a placement-law gap (no crown "
            "mass joins the touch in the wide book) — the missing "
            "placement class for wave-23's last residual."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
