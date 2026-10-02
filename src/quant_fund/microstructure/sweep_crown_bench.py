"""sweep_crown — the last composition: ice crown under multi-level MOs.

``ice_crown.v1`` localized the remaining crown residual precisely: the
hidden channel reaches the tape's band (0.13-0.30 vs 0.214) and the
visible crown stays thin, but the emptied-touch channel sticks at ~1%
because reloaded levels resist single-unit sweeps. The tape's 47%
empty rate needs aggressors that eat *through* persistence —
``sweep_width.v1``: 4.5% of impulses print >=2 levels, max 8.

``mo_size_pmf`` already gives MO events a multi-unit burst (each unit
consumes the current best, so a burst that exhausts the touch walks
deeper). This grid composes the ice crown with an MO tail calibrated
toward the tape's footprint and measures whether the emptied-touch
channel reopens at tape scale.
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SWEEP_CROWN_SCHEMA = "sweep_crown.v1"

_FLEE = {
    "hit_flee_frac": 0.10,
    "hit_flee_band": 2,
    "hit_flee_window": 50,
    "unhit_imp_frac": 0.5,
    "unhit_imp_window": 200,
}
# MO size tails. ``_DEEP`` already runs the heavy ``_MO_PMF``
# ((1,.35),(2,.15),(5,.20),(10,.15),(25,.10),(50,.05)) — pass None to keep
# it; the calibrated-tape tail is much lighter.
_UNIT = ((1, 1.0),)
_TAIL_TAPE = ((1, 0.93), (2, 0.04), (4, 0.02), (8, 0.01))
_TAIL_MID = ((1, 0.70), (2, 0.15), (5, 0.10), (12, 0.05))
# (label, crown_frac, iceberg_reload, mo_size_pmf; None = _DEEP default)
_CELLS: tuple[tuple[str, float, float, tuple[tuple[int, float], ...] | None], ...] = (
    ("wide_flee", 0.0, 0.0, None),
    ("ice30", 0.30, 0.30, None),
    ("ice30_unitmo", 0.30, 0.30, _UNIT),
    ("ice30_tape_tail", 0.30, 0.30, _TAIL_TAPE),
    ("ice30_mid_tail", 0.30, 0.30, _TAIL_MID),
    ("tail_noice", 0.30, 0.0, _TAIL_TAPE),
    ("ice50_tape_tail", 0.30, 0.50, _TAIL_TAPE),
)

# Tape pins: crown_density.v1 / depth_consumption.v1 / hidden_depth.v1 /
# sweep_width.v1.
_TAPE_CROWN_SHARE = 0.2082
_TAPE_OCC_LO, _TAPE_OCC_HI = 9, 21
_TAPE_EMPTY_RATE = 0.4737
_TAPE_HIDDEN_SHARE = 0.214
_TAPE_SWEEP_P_GE2 = 0.045


def sweep_crown_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """(crown × iceberg × MO-tail) grid; verdict vs tape pins."""
    cells = []
    for i, (label, cf, ice, pmf) in enumerate(_CELLS):
        extra = dict(
            _DEEP,
            crown_stack_frac=cf,
            crown_stack_span=2,
            crown_offset=1,
            iceberg_reload=ice,
            **_FLEE,
        )
        if pmf is not None:
            extra["mo_size_pmf"] = pmf
        cell = _sim_crown(
            label,
            extra,
            horizon=horizon,
            seed=seed + i,
            collect_counts=True,
        )
        cell["crown_stack_frac"] = cf
        cell["iceberg_reload"] = ice
        cell["mo_tail"] = pmf is not None
        cell["empty_share"] = (
            round(cell["n_reveals"] / cell["n_fills"], 4) if cell["n_fills"] else None
        )
        cells.append(cell)

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    tail_cells = [c for c in cells if c["mo_tail"]]
    heavy = cells[1]  # ice30 under _DEEP's heavy _MO_PMF
    # The full joint: crown share live, empties reopen toward the tape,
    # spread in band, hidden channel on, sweep footprint near the tape's.
    joint = [
        c
        for c in cells
        if _f(c["crown_share_of_visible"]) >= 0.4 * _TAPE_CROWN_SHARE
        and _f(c["empty_share"]) >= 0.1 * _TAPE_EMPTY_RATE
        and _TAPE_OCC_LO <= _f(c["spread_mean"]) <= 3 * _TAPE_OCC_HI
        and _f(c["hidden_fill_share"]) >= 0.05
        and 0.5 * _TAPE_SWEEP_P_GE2 <= _f(c["sweep_p_ge2"]) <= 4.0 * _TAPE_SWEEP_P_GE2
    ]
    claims = {
        "grid_evaluated": all(c["n_fills"] > 0 for c in cells),
        # The heavy _MO_PMF arm already overshoots the tape's sweep
        # footprint by an order of magnitude — sweep width is not the
        # limiting channel (the falsification evidence).
        "mopmf_overshoots_tape": _f(heavy["sweep_p_ge2"]) > 10.0 * _TAPE_SWEEP_P_GE2,
        # The calibrated-tape tail produces a footprint in band.
        "sweep_footprint_in_band": any(
            0.5 * _TAPE_SWEEP_P_GE2 <= _f(c["sweep_p_ge2"]) <= 4.0 * _TAPE_SWEEP_P_GE2
            for c in tail_cells
        ),
        # Sweep intensity does not control the empty channel: no ice
        # cell reaches even 10% of the tape's 0.474 at any footprint —
        # per-unit iceberg reload keeps the level alive regardless of
        # burst size.
        "empty_not_sweep_limited": all(
            _f(c["empty_share"]) < 0.1 * _TAPE_EMPTY_RATE for c in cells if c["iceberg_reload"] > 0
        ),
        # A joint cell exists across all five pins.
        "joint_sweep_cell": len(joint) > 0,
    }
    payload: dict[str, Any] = {
        "schema": SWEEP_CROWN_SCHEMA,
        "kind": "sim_bench",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "git_revision": git_revision(),
        "seed": seed,
        "horizon_events": horizon,
        "tape_pins": {
            "crown_share": _TAPE_CROWN_SHARE,
            "spread_band": [_TAPE_OCC_LO, _TAPE_OCC_HI],
            "empty_share": _TAPE_EMPTY_RATE,
            "hidden_fill_share": _TAPE_HIDDEN_SHARE,
            "sweep_p_ge2": _TAPE_SWEEP_P_GE2,
        },
        "cells": cells,
        "claims": claims,
        "notes": (
            "The last composition class: ice crown x MO size tail. "
            "Measured: the _DEEP arm's _MO_PMF already prints a sweep "
            "footprint ~11x the tape's (p_ge2 ~0.5 vs 0.045) and yet the "
            "emptied-touch channel stays ~1% — sweep width is not the "
            "binding constraint. Iceberg reload fires per consumed unit, "
            "so ANY mid-burst reload keeps the level alive no matter how "
            "deep the burst cuts. The residual sharpens again: hidden "
            "persistence must be per-level (reload gated on the burst "
            "ending / residual visible depth), not per-unit."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
