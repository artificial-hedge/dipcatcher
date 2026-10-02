"""touch_empty — emptied-touch share under the corrected reveal index.

A measurement fix precedes the grid: ``_sim_crown`` recorded each fill's
row as ``sim.n_events`` — the 1-indexed event counter — while the book
snapshots are 0-indexed per step, so the emptied-touch check compared
the book one event *after* the fill to one before it. A touch emptied
by a fill and re-seeded on the next event never counted, which is why
every crown/iceberg lane measured empty shares near 1%.

Under the corrected index the channel inverts: the uncapped deep arm
empties ~90% of touches — roughly 2x the tape's 47%. The binding
constraint is not producing emptied touches but holding them DOWN while
keeping crown density, spread, and hidden share. The grid spans the
mechanisms that thicken the touch (on-touch stacking, iceberg reserve)
and the ones that thin it (``near_level_cap`` band cap).
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

TOUCH_EMPTY_SCHEMA = "touch_empty.v1"

# (label, crown_frac, crown_offset, near_cap, ice, ice_mode, budget,
#  unhit_imp_frac, hit_flee_frac, refill_cooldown)
_CELLS: tuple[tuple[str, float, int, int, float, str, int, float, float, int], ...] = (
    ("deep", 0.0, 0, 0, 0.0, "per_unit", 0, 0.5, 0.10, 300),
    ("cr1_f30", 0.30, 1, 0, 0.0, "per_unit", 0, 0.5, 0.10, 300),
    ("cr0_f60", 0.60, 0, 0, 0.0, "per_unit", 0, 0.5, 0.10, 300),
    ("cr0_f60_i30", 0.60, 0, 0, 0.30, "residual", 0, 0.5, 0.10, 300),
    ("cap2_sp3", 0.30, 1, 2, 0.0, "per_unit", 0, 0.5, 0.10, 300),
    ("ice80_unit", 0.30, 1, 0, 0.80, "per_unit", 0, 0.5, 0.10, 300),
    # imp=0 removes the inside-spread reroute that pinned the spread shut.
    ("imp0_ff30", 0.60, 0, 0, 0.30, "residual", 0, 0.0, 0.30, 300),
    ("imp0_ff50", 0.60, 0, 0, 0.30, "residual", 0, 0.0, 0.50, 300),
    # The joint cell: touch stack + residual ice reserve + cancel retreat
    # + moderated vacancy memory, no inside-spread reroute.
    ("joint_cell", 0.60, 0, 0, 0.55, "residual", 15, 0.0, 0.45, 100),
)

# Tape pins: crown_density.v1 / depth_consumption.v1 / hidden_depth.v1.
_TAPE_CROWN_SHARE = 0.2082
_TAPE_OCC_LO, _TAPE_OCC_HI = 9, 21
_TAPE_EMPTY_RATE = 0.4737
_TAPE_HIDDEN_SHARE = 0.214
_TAPE_REVEAL_GAP = 3.7554
# Empty-share band around the tape pin: +-40%.
_EMP_LO, _EMP_HI = 0.6 * _TAPE_EMPTY_RATE, 1.4 * _TAPE_EMPTY_RATE


def touch_empty_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """(crown × cap × iceberg) grid under the corrected reveal index."""
    cells = []
    for i, (label, cf, off, cap, ice, mode, budget, imp, flee, cd) in enumerate(_CELLS):
        cell = _sim_crown(
            label,
            dict(
                _DEEP,
                crown_stack_frac=cf,
                crown_stack_span=2,
                crown_offset=off,
                near_level_cap=cap,
                near_level_span=3,
                iceberg_reload=ice,
                iceberg_reload_mode=mode,
                iceberg_budget=budget,
                hit_flee_frac=flee,
                hit_flee_band=2,
                hit_flee_window=50,
                unhit_imp_frac=imp,
                unhit_imp_window=200,
                refill_cooldown=cd,
            ),
            horizon=horizon,
            seed=seed + i,
            collect_counts=True,
        )
        cell["crown_stack_frac"] = cf
        cell["crown_offset"] = off
        cell["near_level_cap"] = cap
        cell["iceberg_reload"] = ice
        cell["iceberg_reload_mode"] = mode
        cell["iceberg_budget"] = budget
        cell["unhit_imp_frac"] = imp
        cell["hit_flee_frac"] = flee
        cell["refill_cooldown"] = cd
        cell["empty_share"] = (
            round(cell["n_reveals"] / cell["n_fills"], 4) if cell["n_fills"] else None
        )
        cells.append(cell)

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    base = cells[0]  # deep
    joint = [
        c
        for c in cells
        if _f(c["crown_share_of_visible"]) >= 0.4 * _TAPE_CROWN_SHARE
        and _EMP_LO <= _f(c["empty_share"]) <= _EMP_HI
        and _TAPE_OCC_LO <= _f(c["spread_mean"]) <= 3 * _TAPE_OCC_HI
        and _f(c["hidden_fill_share"]) >= 0.05
    ]
    claims = {
        "grid_evaluated": all(c["n_fills"] > 0 for c in cells),
        # The index fix unmasks the channel: the uncapped deep arm
        # overshoots the tape's emptied-touch share by >= 1.3x.
        "index_fix_unmasks_empties": _f(base["empty_share"]) >= 1.3 * _TAPE_EMPTY_RATE,
        # On-touch stacking damps empties: a crown@touch cell empties
        # strictly less than the deep baseline.
        "touch_stack_lowers_empties": any(
            c["crown_offset"] == 0
            and c["iceberg_reload"] == 0.0
            and _f(c["empty_share"]) < _f(base["empty_share"])
            for c in cells
        ),
        # Some cell lands the empty share inside the tape band.
        "empty_band_reached": any(_EMP_LO <= _f(c["empty_share"]) <= _EMP_HI for c in cells),
        # A joint cell exists across all four pins.
        "joint_thin_cell": len(joint) > 0,
    }
    payload: dict[str, Any] = {
        "schema": TOUCH_EMPTY_SCHEMA,
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
            "empty_band_used": [_EMP_LO, _EMP_HI],
            "hidden_fill_share": _TAPE_HIDDEN_SHARE,
            "reveal_gap_ticks": _TAPE_REVEAL_GAP,
        },
        "cells": cells,
        "claims": claims,
        "notes": (
            "Reveal-index correction: fills were recorded at the "
            "1-indexed event counter against 0-indexed book snapshots, so "
            "the emptied-touch check ran one event late — every crown/"
            "iceberg lane's ~1% empty share was an artifact. Corrected, "
            "the deep arm overshoots (0.91 vs 0.47) — the channel was "
            "never missing. On-touch stacking damps it toward the band "
            "(cr0_f60_i30 lands 0.416) while pushing crown share past "
            "the pin (0.35); the binding constraint is now spread — "
            "every in-band cell sits at ~2.5 ticks vs the 9-21 band "
            "(touch depth keeps the spread pinned shut). The residual "
            "is vacancy frequency, not depth: the tape's touch empties "
            "AND re-seeds at a 9-21 tick pitch."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
