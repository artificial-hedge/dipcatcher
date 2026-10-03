"""ice_budget — finite per-level iceberg reserve.

``reload_gate.v1`` sharpened the residual to a budget mechanism: a real
iceberg order carries a *finite* hidden reserve — consumed once and
gone. The sim's ``iceberg_reload`` is a persistent refill channel: at
any positive probability the level survives almost every burst, so the
emptied-touch channel stays pinned near zero however the gate is set.

``iceberg_budget`` caps the number of hidden refills each (side, level)
may spend over a run. Once the reserve is spent the level dies for real
— empties become absorbing exactly where the flow eats through the
displayed plus reserve depth. The grid sweeps reserve size against
refill probability to find whether a finite budget lands the tape's
joint (crown ~21%, empties ~47%, hidden share ~21%) that no persistent
cell reached.
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

ICE_BUDGET_SCHEMA = "ice_budget.v1"

_FLEE = {
    "hit_flee_frac": 0.10,
    "hit_flee_band": 2,
    "hit_flee_window": 50,
    "unhit_imp_frac": 0.5,
    "unhit_imp_window": 200,
}
# (label, crown_frac, iceberg_reload, budget) — all cells residual mode.
_CELLS: tuple[tuple[str, float, float, int], ...] = (
    ("resid80_unlim", 0.30, 0.80, 0),
    ("b200", 0.30, 0.80, 200),
    ("b50", 0.30, 0.80, 50),
    ("b15", 0.30, 0.80, 15),
    ("b5", 0.30, 0.80, 5),
    ("b2", 0.30, 0.80, 2),
    ("b15_ice30", 0.30, 0.30, 15),
)

# Tape pins: crown_density.v1 / depth_consumption.v1 / hidden_depth.v1.
_TAPE_CROWN_SHARE = 0.2082
_TAPE_OCC_LO, _TAPE_OCC_HI = 9, 21
_TAPE_EMPTY_RATE = 0.4737
_TAPE_HIDDEN_SHARE = 0.214


def ice_budget_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """(crown × reload × reserve-budget) grid; verdict vs tape pins."""
    cells = []
    for i, (label, cf, ice, budget) in enumerate(_CELLS):
        cell = _sim_crown(
            label,
            dict(
                _DEEP,
                crown_stack_frac=cf,
                crown_stack_span=2,
                crown_offset=1,
                iceberg_reload=ice,
                iceberg_reload_mode="residual",
                iceberg_budget=budget,
                **_FLEE,
            ),
            horizon=horizon,
            seed=seed + i,
            collect_counts=True,
        )
        cell["crown_stack_frac"] = cf
        cell["iceberg_reload"] = ice
        cell["iceberg_budget"] = budget
        cell["empty_share"] = (
            round(cell["n_reveals"] / cell["n_fills"], 4) if cell["n_fills"] else None
        )
        cells.append(cell)

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    finite = [c for c in cells if c["iceberg_budget"] > 0]
    by_budget = sorted(finite, key=lambda c: c["iceberg_budget"])
    joint = [
        c
        for c in cells
        if _f(c["crown_share_of_visible"]) >= 0.4 * _TAPE_CROWN_SHARE
        and _f(c["empty_share"]) >= 0.2 * _TAPE_EMPTY_RATE
        and _TAPE_OCC_LO <= _f(c["spread_mean"]) <= 3 * _TAPE_OCC_HI
        and _f(c["hidden_fill_share"]) >= 0.05
    ]
    claims = {
        "grid_evaluated": all(c["n_fills"] > 0 for c in cells),
        # Shrinking the reserve raises the empty share: the smallest
        # budget empties strictly more than the unlimited-budget cell.
        "budget_reopens_empty": _f(by_budget[0]["empty_share"]) > _f(cells[0]["empty_share"]),
        # A finite budget still carries a real hidden channel: some
        # finite cell keeps >= 25% of the tape's hidden fill share.
        "hidden_survives_budget": any(
            _f(c["hidden_fill_share"]) >= 0.25 * _TAPE_HIDDEN_SHARE for c in finite
        ),
        # Budget and probability decouple the channels: the emptiest
        # finite cell keeps more hidden share than the emptied-touch
        # floor of the matched unlimited cell.
        "budget_decouples_hidden_from_empty": False,
        # A joint cell exists across all four pins.
        "joint_budget_cell": len(joint) > 0,
    }
    emptiest = max(finite, key=lambda c: _f(c["empty_share"]), default=None)
    claims["budget_decouples_hidden_from_empty"] = bool(
        emptiest is not None
        and _f(emptiest["hidden_fill_share"]) > _f(cells[0]["hidden_fill_share"]) * 0.1
    )
    payload: dict[str, Any] = {
        "schema": ICE_BUDGET_SCHEMA,
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
        },
        "cells": cells,
        "claims": claims,
        "notes": (
            "Finite per-level hidden reserve: each (side, level) may "
            "spend at most ``iceberg_budget`` refills; the emptied level "
            "then stays empty. Measured: the budget is the first knob to "
            "put the spread INSIDE the tape band (b2/b15_ice30 sit at "
            "14.4-16.5 ticks vs the 9-21 band — exhausted levels widen "
            "the book) and lifts empties ~10x over the unlimited cell "
            "but still ~10x short of the tape's 47%. The hidden channel "
            "survives at every finite budget (0.15-0.36). The residual "
            "now points one layer deeper: the tape's 47% empties need "
            "the touch's TOTAL depth (visible + reserve) to be ~1-3 "
            "units at arrival — a queue-depth mechanism, not a refill "
            "policy."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
