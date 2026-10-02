"""joint_stability — seed robustness of the emptied-touch joint cell.

`touch_empty` found the first cell holding all four tape pins at once —
emptied-touch share, spread occupancy, hidden fill share, and crown
density — at a single seed. A single-draw joint claim says nothing about
whether the cell sits on a *ridge* (small perturbations keep the pins)
or a *knife-edge* (it passed at its seed by luck). This bench re-runs the
joint cell and its four axis neighbors across a seed list and reports
per-pin pass rates plus the joint pass rate per cell.

Evidence class: research / SYNTHETIC (ZI-LOB sim vs LOBSTER tape pins).

Claims:
  ``joint_cell_stable``   joint cell passes all four pins at >= 3/4 seeds.
  ``joint_ridge``         >= 3 of the 5 cells pass all four at >= 1 seed.
  ``emp_is_binding_pin``  the emptied-share pin is the joint cell's
                          weakest pin across seeds.
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.crown_density_bench import _DEEP, _sim_crown
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

JOINT_STABILITY_SCHEMA = "joint_stability.v1"

# Tape pins: same as touch_empty.v1.
_TAPE_CROWN_SHARE = 0.2082
_TAPE_OCC_LO, _TAPE_OCC_HI = 9, 21
_TAPE_EMPTY_RATE = 0.4737
_TAPE_HIDDEN_SHARE = 0.214
_EMP_LO, _EMP_HI = 0.6 * _TAPE_EMPTY_RATE, 1.4 * _TAPE_EMPTY_RATE

_SEEDS: tuple[int, ...] = (7, 11, 19, 23)

_BASE: dict[str, Any] = dict(
    _DEEP,
    crown_stack_frac=0.60,
    crown_stack_span=2,
    crown_offset=0,
    iceberg_reload=0.55,
    iceberg_reload_mode="residual",
    iceberg_budget=15,
    unhit_imp_frac=0.0,
    unhit_imp_window=200,
    hit_flee_frac=0.45,
    hit_flee_band=2,
    hit_flee_window=50,
    refill_cooldown=100,
)

# Axis neighbors: one-knob perturbations around the joint cell.
_CELLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("joint", {}),
    ("cf50", {"crown_stack_frac": 0.50}),
    ("cf70", {"crown_stack_frac": 0.70}),
    ("ff35", {"hit_flee_frac": 0.35}),
    ("ff55", {"hit_flee_frac": 0.55}),
)


def _pins(*, emp: float, sp: float, crown: float, hid: float) -> dict[str, bool]:
    return {
        "empty_share": _EMP_LO <= emp <= _EMP_HI,
        "spread_mean": _TAPE_OCC_LO <= sp <= 3 * _TAPE_OCC_HI,
        "crown_share": crown >= 0.4 * _TAPE_CROWN_SHARE,
        "hidden_share": hid >= 0.05,
    }


def joint_stability_bench(*, horizon: int = 15000) -> dict[str, Any]:
    """(joint cell + axis neighbors) x seeds robustness surface."""
    rows: list[dict[str, Any]] = []
    for label, over in _CELLS:
        extra = dict(_BASE)
        extra.update(over)
        per_seed: list[dict[str, Any]] = []
        for seed in _SEEDS:
            c = _sim_crown(
                f"{label}_s{seed}",
                extra,
                horizon=horizon,
                seed=seed,
                collect_counts=True,
            )
            emp = c["n_reveals"] / c["n_fills"] if c["n_fills"] else 0.0
            pins = _pins(
                emp=emp,
                sp=c["spread_mean"],
                crown=c["crown_share_of_visible"],
                hid=c["hidden_fill_share"],
            )
            per_seed.append(
                {
                    "seed": seed,
                    "empty_share": round(emp, 4),
                    "spread_mean": c["spread_mean"],
                    "crown_share_of_visible": c["crown_share_of_visible"],
                    "hidden_fill_share": c["hidden_fill_share"],
                    "n_fills": c["n_fills"],
                    "pins": pins,
                    "joint": all(pins.values()),
                }
            )
        rows.append(
            {
                "cell": label,
                "overrides": over,
                "n_seeds": len(_SEEDS),
                "n_joint_seeds": sum(r["joint"] for r in per_seed),
                "per_pin_pass": {
                    pin: sum(r["pins"][pin] for r in per_seed)
                    for pin in ("empty_share", "spread_mean", "crown_share", "hidden_share")
                },
                "seeds": per_seed,
            }
        )
    joint_row = rows[0]
    claims = {
        "joint_cell_stable": joint_row["n_joint_seeds"] >= (3 * len(_SEEDS)) // 4,
        "joint_ridge": sum(r["n_joint_seeds"] > 0 for r in rows) >= 3,
        # The emptied-share pin is the joint cell's weakest across seeds.
        "emp_is_binding_pin": joint_row["per_pin_pass"]["empty_share"]
        == min(joint_row["per_pin_pass"].values()),
    }
    payload: dict[str, Any] = {
        "schema": JOINT_STABILITY_SCHEMA,
        "kind": "sim_bench",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "git_revision": git_revision(),
        "seeds": list(_SEEDS),
        "horizon_events": horizon,
        "tape_pins": {
            "crown_share": _TAPE_CROWN_SHARE,
            "spread_band": [_TAPE_OCC_LO, _TAPE_OCC_HI],
            "empty_share": _TAPE_EMPTY_RATE,
            "empty_band_used": [_EMP_LO, _EMP_HI],
            "hidden_fill_share": _TAPE_HIDDEN_SHARE,
        },
        "cells": rows,
        "claims": claims,
        "notes": (
            "The touch_empty joint cell held all four pins at one seed; "
            "this bench prices its robustness. A joint claim that flips "
            "on re-seeding is a knife-edge, not a mechanism — per-pin "
            "pass rates identify which channel is binding."
        ),
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = ["JOINT_STABILITY_SCHEMA", "joint_stability_bench"]
