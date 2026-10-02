"""crown_size — bounded and sized crown variants; the share↔sweep tension.

``crown_behind.v1`` closed the geometry channel (crown mass behind a
thin touch restores the emptied-touch channel) but left magnitudes
short: crown share 0.105 vs tape 0.208, empty rate 0.09 vs 0.47. Two
bounded variants close the mass gap on paper:

- ``crown_cap``: per-level depth cap — a crown draw landing on a
  full level falls through to the default anchor placement.
- ``crown_size_pmf``: crown placements draw their own size from a
  heavy table — a few LARGE orders, matching the tape's 60-200-share
  crown orders, instead of many unit stacks.

This grid measures whether either bound reaches the tape's joint
(dense crown + live empty channel + 9-21 spread band) — or whether
visible depth and sweepability are the same variable in this grammar.
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

CROWN_SIZE_SCHEMA = "crown_size.v1"

# Heavy crown-order tables: the tape's crown is 60-200-share orders.
_SZ_A = ((5, 0.4), (15, 0.3), (40, 0.2), (100, 0.1))
_SZ_B = ((2, 0.5), (8, 0.3), (25, 0.15), (60, 0.05))

_FLEE = {
    "hit_flee_frac": 0.10,
    "hit_flee_band": 2,
    "hit_flee_window": 50,
    "unhit_imp_frac": 0.5,
    "unhit_imp_window": 200,
}
# (label, crown_stack_frac, span, offset, cap, size_pmf)
_CELLS: tuple[tuple[str, float, int, int, int, tuple[tuple[int, float], ...] | None], ...] = (
    ("wide_flee", 0.0, 2, 1, 0, None),
    ("behind30", 0.30, 2, 1, 0, None),  # crown_behind's best cell
    ("behind30_cap8", 0.30, 2, 1, 8, None),
    ("behind30_cap3", 0.30, 2, 1, 3, None),
    ("sized05_heavy", 0.05, 2, 1, 0, _SZ_A),
    ("sized10_mid_cap5", 0.10, 2, 1, 5, _SZ_B),
    ("behind40_wide", 0.40, 3, 1, 0, None),
)

# Tape targets (crown_density.v1 / depth_consumption.v1).
_TAPE_CROWN_SHARE = 0.2082
_TAPE_OCC_LO, _TAPE_OCC_HI = 9, 21
_TAPE_EMPTY_RATE = 0.4737


def crown_size_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """Cap/size grid on the behind-touch crown; verdict vs tape pins."""
    cells = []
    for i, (label, cf, cs, co, cap, sz) in enumerate(_CELLS):
        cell = _sim_crown(
            label,
            dict(
                _DEEP,
                crown_stack_frac=cf,
                crown_stack_span=cs,
                crown_offset=co,
                crown_cap=cap,
                crown_size_pmf=sz,
                **_FLEE,
            ),
            horizon=horizon,
            seed=seed + i,
        )
        cell["crown_stack_frac"] = cf
        cell["crown_stack_span"] = cs
        cell["crown_offset"] = co
        cell["crown_cap"] = cap
        cell["sized"] = sz is not None
        cell["empty_share"] = (
            round(cell["n_reveals"] / cell["n_fills"], 4) if cell["n_fills"] else None
        )
        cells.append(cell)

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    behind = cells[1]
    capped = [c for c in cells if c["crown_cap"] > 0]
    sized = [c for c in cells if c["sized"]]
    # Joint target: crown share within a factor of the tape's, the
    # emptied-touch channel alive, and the spread inside the tape's
    # 9-21 occupancy band.
    joint = [
        c
        for c in cells
        if _f(c["crown_share_of_visible"]) >= 0.5 * _TAPE_CROWN_SHARE
        and _f(c["empty_share"]) >= 0.05 * _TAPE_EMPTY_RATE
        and _TAPE_OCC_LO <= _f(c["spread_mean"]) <= _TAPE_OCC_HI
    ]
    claims = {
        "grid_evaluated": all(c["n_fills"] > 0 for c in cells),
        # The cap binds: capped cells carry strictly less crown mass
        # than the uncapped behind-touch cell.
        "cap_binds_share": bool(
            max(_f(c["crown_share_of_visible"]) for c in capped)
            < _f(behind["crown_share_of_visible"])
        ),
        # Uncapped sized crowns flood share past the tape's level —
        # heavy orders put more volume in the crown than the tape has.
        "sized_floods_share": bool(
            max(_f(c["crown_share_of_visible"]) for c in sized) > _TAPE_CROWN_SHARE
        ),
        # Yet sized crowns suppress the emptied-touch channel: every
        # sized cell empties at <10% of the tape's rate — volume and
        # sweepability are the same variable in this grammar.
        "sized_empty_suppressed": all(_f(c["empty_share"]) < 0.1 * _TAPE_EMPTY_RATE for c in sized),
        # The honest verdict: no bounded crown cell holds share, empty
        # channel, and the spread band at once — the residual is hidden
        # depth (iceberg crowns), not visible mass.
        "joint_unreachable": len(joint) == 0,
    }
    payload: dict[str, Any] = {
        "schema": CROWN_SIZE_SCHEMA,
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
        },
        "cells": cells,
        "claims": claims,
        "notes": (
            "Two bounded variants of the behind-touch crown, both "
            "falsified as closers. capping per-level depth binds crown "
            "share without buying back empties (cap8: share 0.047, "
            "empty ~0.01): the emptied-touch channel depends on touch "
            "depth, which crown bounds don't set. Sizing crown orders "
            "heavier floods share past the tape's 0.21 uncapped (0.38) "
            "and suppresses the empty channel under every variant "
            "— visible depth and sweepability are the same variable in "
            "this grammar (order count = volume). The tape decouples "
            "them via hidden liquidity: hidden_depth.v1 measured 21.4% "
            "of fills printing inside the visible book. The residual is "
            "an iceberg crown — shadow depth behind the thin visible "
            "touch that only surfaces when the touch empties."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
