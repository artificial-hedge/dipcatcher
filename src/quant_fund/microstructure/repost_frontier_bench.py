"""repost_frontier — which repost grammar reproduces the tape's reseeding?

The tape's emptied-touch levels re-seed at 0.538 within 500 events, ~75%
back at the touch (reseed_hazard.v1). Under the per-vacation measure —
an emptied level counted once per episode, as on the tape — the joint
cell under-reseeds (0.16) and the arrival-driven repost arm
(joint_rp60_b3) composes both targets.

This bench compares the two re-post grammars the sim now carries:

- arrival-driven (``repost_frac``): each LO arrival re-sites at a
  still-vacant level from the per-side vacancy ledger, optionally
  band-restricted near the touch and cause-filtered to fill-emptied
  vacancies.
- fill-triggered (``fill_repost_frac``): each fill that empties a level
  schedules a re-post of ``repost_depth`` units at that level, due after
  an exponential delay of mean ``fill_repost_delay`` events — the tape's
  maker re-quote after being lifted (p50 ~110 events on the tape).

The question: which grammar lands inside both tape bands (rate within
±30%, at-touch share ≥ 0.6) on the joint cell, and whether the delay
knob tracks the tape's latency profile.

Evidence class: research / SYNTHETIC (ZI-LOB joint-regime arms vs
LOBSTER tape pins from reseed_hazard.v1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_fund.microstructure.reseed_hazard_bench import (
    _JOINT,
    _WINDOW,
    lobster_reseed,
    sim_reseed,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

REPOST_FRONTIER_SCHEMA = "repost_frontier.v1"

_TAPE_RATE = 0.5376
_TAPE_TOUCH = 0.7512
_TAPE_P50 = 110.0
# Tolerance bands: rate within ±30% of tape, touch share >= 0.6.
_RATE_LO, _RATE_HI = 0.7 * _TAPE_RATE, 1.3 * _TAPE_RATE
_TOUCH_MIN = 0.6

# (label, sim config deltas on the joint cell).
_CELLS: tuple[tuple[str, dict[str, Any]], ...] = (
    # reference: no repost memory
    ("joint", {}),
    # arrival-driven repost channel (the #752 arm that composes)
    ("arr_f60_b3", dict(repost_frac=0.6, repost_window=_WINDOW, repost_band=3)),
    ("arr_f60_b0", dict(repost_frac=0.6, repost_window=_WINDOW)),
    (
        "arr_f60_b3_fill",
        dict(repost_frac=0.6, repost_window=_WINDOW, repost_band=3, repost_cause="fill"),
    ),
    # fill-triggered delayed repost — the tape mechanism candidate
    ("evt_f30_m160", dict(fill_repost_frac=0.3, fill_repost_delay=160)),
    ("evt_f55_m160", dict(fill_repost_frac=0.55, fill_repost_delay=160)),
    ("evt_f80_m160", dict(fill_repost_frac=0.8, fill_repost_delay=160)),
    ("evt_f55_m40", dict(fill_repost_frac=0.55, fill_repost_delay=40)),
    ("evt_f55_m320", dict(fill_repost_frac=0.55, fill_repost_delay=320)),
    ("evt_f55_m160_d8", dict(fill_repost_frac=0.55, fill_repost_delay=160, repost_depth=8)),
)


def repost_frontier_bench(
    tape_dir: Path | None = None, *, horizon: int = 20000, seed: int = 7
) -> dict[str, Any]:
    """Repost-grammar frontier on the joint cell."""
    tape: dict[str, Any] | None = None
    if tape_dir is not None:
        msg = tape_dir / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
        ob = tape_dir / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
        if msg.exists() and ob.exists():
            tape = lobster_reseed(msg, ob)
    cells: list[dict[str, Any]] = []
    for i, (label, extra) in enumerate(_CELLS):
        st = sim_reseed(label, dict(_JOINT, **extra), horizon=horizon, seed=seed + i)
        st["deltas"] = {k: v for k, v in extra.items()}
        cells.append(st)

    def _f(v: float | None) -> float:
        return v if v is not None else 0.0

    def _in_tol(c: dict[str, Any]) -> bool:
        return (
            _RATE_LO <= _f(c["reseed_rate_500"]) <= _RATE_HI
            and _f(c["reseed_as_touch_share"]) >= _TOUCH_MIN
        )

    composable = [c["regime"] for c in cells if _in_tol(c)]
    evt = [c for c in cells if c["regime"].startswith("evt_")]
    evt_in_tol = [c["regime"] for c in evt if _in_tol(c)]
    # The delay knob should order the reseed latency: m40 < m160 < m320.
    p40 = _f(cells[7]["reseed_latency_p50"])
    p160 = _f(cells[5]["reseed_latency_p50"])
    p320 = _f(cells[8]["reseed_latency_p50"])
    claims = {
        "frontier_cells_measured": all(c["n_emptied"] > 0 for c in cells),
        "composable_cell_exists": bool(composable),
        "fill_trigger_composes_too": bool(evt_in_tol),
        "delay_sets_latency": p40 < p160 < p320,
        "joint_undeerseeds": bool(tape is not None and _f(cells[0]["reseed_rate_500"]) < _RATE_LO),
        "tape_reseeds_majority": bool(tape is None or _f(tape["reseed_rate_500"]) > 0.5),
        "tape_reseed_returns_to_touch": bool(
            tape is None or _f(tape["reseed_as_touch_share"]) >= 0.6
        ),
    }
    body: dict[str, Any] = {
        "schema": REPOST_FRONTIER_SCHEMA,
        "kind": "sim_vs_real",
        "git_revision": git_revision(),
        "research_only": True,
        "data_label": "MIXED",
        "horizon": horizon,
        "seed": seed,
        "repost_window": _WINDOW,
        "targets": {
            "tape_reseed_rate": _TAPE_RATE,
            "tape_reseed_as_touch_share": _TAPE_TOUCH,
            "tape_reseed_latency_p50": _TAPE_P50,
            "rate_band": [_RATE_LO, _RATE_HI],
            "touch_min": _TOUCH_MIN,
        },
        "tape": tape,
        "cells": cells,
        "composable_cells": composable,
        "claims": claims,
        "notes": (
            "Two re-post grammars on the joint cell: arrival-driven "
            "(repost_frac over the vacancy ledger) vs fill-triggered "
            "(fill_repost_frac schedules a delayed re-post on each "
            "fill-emptied level). Delay is an exponential mean in "
            "events; a due repost fires only if the level is still "
            "absent and still legal. Question: which grammar lands "
            "inside both tape bands, and does fill_repost_delay track "
            "the tape's ~110-event latency? Result is reported, not "
            "asserted."
        ),
    }
    body["receipt_sha256"] = hash_bytes(canonical_json_bytes(body))
    return body
