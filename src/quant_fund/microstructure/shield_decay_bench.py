"""shield_decay — which kernel *shape* is the post-fill unhit-side shield?

``aftermath_flow`` measured the tape's unhit-side cancel protection by
window: unhit cxl/add is 0.61 in [1,10) events after a fill but reverts
to ~0.90/0.93 in [10,50)/[50,200) — the shield is heavy immediately and
gone within ~50 events. ``cxl_shield`` showed suppression (``cxl_unhit_
damp``) is the right mechanism class but used a flat window; a flat
hundred-event shield over-protects the late windows.

This bench discriminates two kernel shapes that both fit the [1,10)
channel: a SHORT flat shield (damp≈0.5, window≈10-25 events) versus a
DECAYED shield (``cxl_unhit_damp_decay`` — damp * exp(-elapsed/decay)
with a long window cap). Each cell is scored on all three window ratios
*and* the instant drift target — the shape that fits the whole profile
wins, the other is logged as a divergence.
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.cxl_shield_bench import (
    _INSTANT_TARGET,
    _close,
    _shield_cell,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

SHIELD_DECAY_SCHEMA = "shield_decay.v1"

# Tape's unhit cxl/add ratio per window (aftermath_flow.v1 real pane).
_U_RATIO_TAPE = {"1_10": 0.610, "10_50": 0.903, "50_200": 0.933}
_U_RATIO_TOL = 0.12

# (damp, decay, window): flat short shields, a flat long reference, and
# decayed shields capped at window=100.
_GRID: tuple[tuple[float, float, int], ...] = (
    (0.0, 0.0, 0),  # no shield — baseline leakage reference
    (0.5, 0.0, 10),
    (0.5, 0.0, 15),
    (0.5, 0.0, 25),
    (0.6, 0.0, 15),
    (0.5, 0.0, 100),  # flat long shield — the cxl_shield cell
    (0.6, 10.0, 100),
    (0.8, 20.0, 100),
    (0.8, 40.0, 100),
)


def _cell_score(cell: dict[str, Any]) -> dict[str, Any]:
    ratios: dict[str, float | None] = {}
    in_tol = True
    for wname, target in _U_RATIO_TAPE.items():
        u = cell["windows"][wname]["unhit_channel_rates"]
        r = u["cxl"] / u["add"] if u["add"] else None
        ratios[wname] = r
        if not _close(r, target, _U_RATIO_TOL):
            in_tol = False
    instant_ok = _close(cell["instant_signed_ticks"], _INSTANT_TARGET, 0.2)
    return {
        "unhit_cxl_per_add_by_window": ratios,
        "profile_in_tol": in_tol,
        "instant_in_tol": instant_ok,
        "joint_in_tol": in_tol and instant_ok,
    }


def shield_decay_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for damp, decay, window in _GRID:
        cell = _shield_cell(0.0, damp, window, horizon=horizon, seed=seed, damp_decay=decay)
        cell.update(_cell_score(cell))
        cells.append(cell)

    def _flat(cell: dict[str, Any]) -> bool:
        return bool(cell["damp_decay"] == 0.0 and cell["damp"] > 0.0)

    def _decayed(cell: dict[str, Any]) -> bool:
        return bool(cell["damp_decay"] > 0.0)

    short_flat = [c for c in cells if _flat(c) and 0 < c["window"] <= 25 and c["profile_in_tol"]]
    decay_ok = [c for c in cells if _decayed(c) and c["profile_in_tol"]]
    long_flat_late_fail = all(
        not c["profile_in_tol"]
        for c in cells
        if _flat(c) and c["window"] >= 100 and c["damp"] >= 0.4
    )
    joint = [c for c in cells if c["joint_in_tol"]]

    claims = {
        "kernel_grid_evaluated": len(cells) == len(_GRID)
        and all(c["unhit_cxl_per_add_by_window"]["10_50"] is not None for c in cells),
        "short_shield_matches": bool(short_flat),
        "decay_kernel_matches": bool(decay_ok),
        "long_flat_overshields_late": bool(long_flat_late_fail),
        "joint_profile_plus_instant": bool(joint),
    }
    payload: dict[str, Any] = {
        "schema": SHIELD_DECAY_SCHEMA,
        "kind": "bench",
        "ticker": "AMZN_2012-06-21",
        "horizon": horizon,
        "seed": seed,
        "u_ratio_tape": _U_RATIO_TAPE,
        "u_ratio_tol": _U_RATIO_TOL,
        "grid": [{"damp": d, "decay": de, "window": w} for d, de, w in _GRID],
        "cells": cells,
        "matching_flat_cells": [{"damp": c["damp"], "window": c["window"]} for c in short_flat],
        "matching_decay_cells": [
            {"damp": c["damp"], "decay": c["damp_decay"], "window": c["window"]} for c in decay_ok
        ],
        "joint_cells": [
            {"damp": c["damp"], "decay": c["damp_decay"], "window": c["window"]} for c in joint
        ],
        "claims": claims,
        "interpretation": (
            "The tape's unhit-side protection is a SHORT, strong shield: "
            "unhit cxl/add is 0.61 within 10 events of a fill and back to "
            "~0.9+ beyond 50. A flat 100-event shield over-protects the "
            "late windows; the question is whether a short flat window or "
            "a decayed kernel matches the whole profile."
        ),
        "git_revision": git_revision(),
        "data_label": "SYNTHETIC",
        "research_only": True,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
