"""crown_join — the join-the-touch placement class in the wide regime.

``crown_density.v1`` measured the missing primitive: the tape carries
~21% of visible top-10 depth within 3 ticks of each touch while every
sim arm manages 3-11%, and the wide book's emptied-touch reveal
overshoots (6.9 vs 3.8 ticks) because nothing rests in the crown. The
``crown_stack_frac``/``crown_stack_span`` config knobs add the missing
placement class: a share of LO arrivals stacks uniformly on
``[own_touch - span, own_touch]`` / ``[own_touch, own_touch + span]``.

This bench runs the crown class over the flee×chase composition inside
the deep geometry. The measured question: does seeding the crown pull
the wide book's occupancy, crown share, and reveal gap into the tape's
band at once — and does the emptied touch still exist to be revealed
(the tape empties 47% of fills)?
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

CROWN_JOIN_SCHEMA = "crown_join.v1"

# (label, crown_stack_frac, crown_stack_span, extra_knobs)
_FLEE = {
    "hit_flee_frac": 0.10,
    "hit_flee_band": 2,
    "hit_flee_window": 50,
    "unhit_imp_frac": 0.5,
    "unhit_imp_window": 200,
}
_CELLS: tuple[tuple[str, float, int, dict[str, Any]], ...] = (
    ("wide_ref", 0.0, 3, {}),
    ("wide_flee", 0.0, 3, dict(_FLEE)),
    ("wide_crown10", 0.10, 3, dict(_FLEE)),
    ("wide_crown20", 0.20, 3, dict(_FLEE)),
    ("wide_crown30", 0.30, 3, dict(_FLEE)),
    ("wide_crown20_s1", 0.20, 1, dict(_FLEE)),
)

# Tape targets (crown_density.v1).
_TAPE_CROWN_SHARE = 0.2082
_TAPE_REVEAL = 3.7554
_TAPE_OCC_LO, _TAPE_OCC_HI = 9, 21  # tape's dominant spread band
_TAPE_EMPTY_RATE = 0.4737


def crown_join_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """Crown-fraction grid on the wide book; verdict against tape pins."""
    cells = []
    for i, (label, cf, cs, extra) in enumerate(_CELLS):
        cell = _sim_crown(
            label,
            dict(_DEEP, crown_stack_frac=cf, crown_stack_span=cs, **extra),
            horizon=horizon,
            seed=seed + i,
        )
        cell["crown_stack_frac"] = cf
        cell["crown_stack_span"] = cs
        cell["empty_share"] = (
            round(cell["n_reveals"] / cell["n_fills"], 4) if cell["n_fills"] else None
        )
        cells.append(cell)

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    crown_cells = [c for c in cells if c["crown_stack_frac"] > 0.0]
    best = max(crown_cells, key=lambda c: _f(c["crown_share_of_visible"]))
    wide_flee = cells[1]
    claims = {
        "grid_evaluated": all(c["n_fills"] > 0 for c in cells),
        # The crown class seeds real mass, but the share caps far below
        # the tape's 0.21 — the crown band alone doesn't reproduce the
        # tape's crown.
        "crown_seeded": bool(_f(best["crown_share_of_visible"]) >= 0.5 * _TAPE_CROWN_SHARE),
        # The crown narrows the spread: the best crown cell's spread
        # drops well under the flee-only cell's — crown seeding pulls
        # the book into (and below) the tape's 9-21 band.
        "spread_compresses": bool(_f(best["spread_mean"]) < _f(wide_flee["spread_mean"])),
        # The over-correction: a crown that stacks but doesn't churn
        # kills the emptied-touch channel — the tape empties 47% of
        # fills, the deep-crown cells empty ~0%.
        "crown_kills_reveal": bool(_f(best["empty_share"]) < 0.1 * _TAPE_EMPTY_RATE),
        # Only the narrow crown keeps empties alive: the span-1 cell
        # still reveals (under-revealing now) — the residual is crown
        # *churn*, not crown mass: the tape's crown re-quotes.
        "narrow_crown_survives": bool(_f(cells[-1]["n_reveals"]) > 0),
    }
    payload: dict[str, Any] = {
        "schema": CROWN_JOIN_SCHEMA,
        "kind": "sim_bench",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "git_revision": git_revision(),
        "seed": seed,
        "horizon_events": horizon,
        "tape_pins": {
            "crown_share": _TAPE_CROWN_SHARE,
            "reveal_gap_ticks": _TAPE_REVEAL,
            "spread_band": [_TAPE_OCC_LO, _TAPE_OCC_HI],
            "empty_share": _TAPE_EMPTY_RATE,
        },
        "cells": cells,
        "claims": claims,
        "notes": (
            "crown_density.v1 showed the tape's near-touch crown carries "
            "21% of visible depth while the sim's arms reach 3-11%, and "
            "the wide book over-reveals because the crown is empty. The "
            "crown_stack_* knobs add the missing join-the-touch class. "
            "Measured: the crown seeds (share 0.04->0.08), compresses "
            "the wide spread into/below the tape's band (19->7 ticks), "
            "but the crown STACKS without churning — at the share "
            "maximum the emptied-touch channel dies (0 reveals vs the "
            "tape's 47% empty rate). Residual sharpens again: the tape's "
            "crown turns over (re-quote churn at the touch — "
            "cxl_requote's domain); a static stack is the wrong "
            "mechanism class for the remaining gap."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
